import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from ops import Voxelization, nms_cuda
from utils import limit_period


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 

            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0)
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0)

        return pillars, coors_batch, npoints_per_pillar


class SPP(nn.Module):
    def __init__(self, in_channels=9):
        super(SPP, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 10, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, 18, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels, 27, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        out = torch.cat((x, x1, x2, x3), dim=1)
        return out


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = SPP(in_channel)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp

        voxel_ids = torch.arange(0, pillars.size(1)).to(device)
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]
        mask = mask.permute(1, 0).contiguous()
        features *= mask[:, :, None]

        features = features.permute(0, 2, 1).contiguous()
        features = F.relu(self.bn(self.conv(features)))
        pooling_features = torch.max(features, dim=-1)[0]

        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)
        return batched_canvas


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x



class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()

        self.block = nn.Sequential(
            BasicConv(channels, channels, 1),
            BasicConv(channels, channels, 3),
        )

    def forward(self, x):
        return x + self.block(x)


class Residual_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(Residual_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
        self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
        self.blocks_conv = nn.Sequential(
            *[Residual(channels=out_channels // 2) for _ in range(num_blocks)],
            BasicConv(out_channels // 2, out_channels // 2, 1)
        )
        self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat((x0, x1), dim=1)
        x = self.concat_conv(x)
        return x

class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=None):
        super().__init__()
        if layer_strides is None:
            layer_strides = [2, 2, 2]
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)

        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            self.multi_blocks.append(Residual_body(in_channel, out_channels[i], layer_nums[i]))
            in_channel = out_channels[i]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        """
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.up_conv3 = nn.ConvTranspose2d(in_channels[0], out_channels[0],
                                           upsample_strides[0], stride=upsample_strides[0], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[0], eps=1e-3, momentum=0.01)
        self.up_conv2 = nn.ConvTranspose2d(in_channels[1], out_channels[1],
                                           upsample_strides[1], stride=upsample_strides[1], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1], eps=1e-3, momentum=0.01)
        self.up_conv1 = nn.ConvTranspose2d(in_channels[2], out_channels[2],
                                           upsample_strides[2], stride=upsample_strides[2], bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[2], eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        x3 = self.relu(self.bn3(self.up_conv3(x[2])))

        x32 = torch.cat((x3, x[1]), dim=1)
        x2 = self.relu(self.bn2(self.up_conv2(x32)))

        x21 = torch.cat((x2, x[0]), dim=1)
        out = self.relu(self.bn1(self.up_conv1(x21)))

        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class SEblock(nn.Module):  # Squeeze and Excitation block
    def __init__(self, channels, ratio=16):
        super(SEblock, self).__init__()
        channels = channels
        hidden_channels = channels // ratio
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attn(x)
        return weights * x


class PointPillars(nn.Module):
    def __init__(self,
                 nclasses=3,
                 voxel_size=None,
                 point_cloud_range=None,
                 max_num_points=16,
                 max_voxels=(16000, 40000)):
        super().__init__()
        if point_cloud_range is None:
            point_cloud_range = [[0, -39.68, -3, 69.12, 39.68, -1], [0, -39.68, -1, 69.12, 39.68, 1]]
        if voxel_size is None:
            voxel_size = [0.16, 0.16, 2]
        self.nclasses = nclasses
        self.pillar_layer1 = PillarLayer(voxel_size=voxel_size,
                                         point_cloud_range=point_cloud_range[0],
                                         max_num_points=max_num_points,
                                         max_voxels=max_voxels)
        self.pillar_encoder1 = PillarEncoder(voxel_size=voxel_size,
                                             point_cloud_range=point_cloud_range[0],
                                             in_channel=9,
                                             out_channel=64)
        self.pillar_layer2 = PillarLayer(voxel_size=voxel_size,
                                         point_cloud_range=point_cloud_range[1],
                                         max_num_points=max_num_points,
                                         max_voxels=max_voxels)
        self.pillar_encoder2 = PillarEncoder(voxel_size=voxel_size,
                                             point_cloud_range=point_cloud_range[1],
                                             in_channel=9,
                                             out_channel=64)
        self.se = SEblock(channels=64)
        self.conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.backbone = Backbone(in_channel=64,
                                 out_channels=[64, 128, 256],
                                 layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[256, 256, 320],
                         upsample_strides=[2, 2, 1],
                         out_channels=[128, 256, 384])
        self.head = Head(in_channel=384, n_anchors=2*nclasses, n_classes=nclasses)
        
        # anchors
        ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations)
        
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d,
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)

        pillars1, coors_batch1, npoints_per_pillar1 = self.pillar_layer1(batched_pts)

        pillar_features1 = self.pillar_encoder1(pillars1, coors_batch1, npoints_per_pillar1)

        pillars2, coors_batch2, npoints_per_pillar2 = self.pillar_layer2(batched_pts)
        pillar_features2 = self.pillar_encoder2(pillars2, coors_batch2, npoints_per_pillar2)

        pillar_features1 = self.se(pillar_features1)
        pillar_features2 = self.se(pillar_features2)

        pillar_features = torch.cat((pillar_features1, pillar_features2), dim=1)
        pillar_features = self.conv(pillar_features)

        xs = self.backbone(pillar_features)

        x = self.neck(xs)

        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   


if __name__ == '__main__':
    net = PointPillars()
    print(net)
