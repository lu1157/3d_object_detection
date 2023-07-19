# Pillar-based multilayer pseudo-image 3D object detection

## mAP on KITTI validation set (Easy, Moderate, Hard)

| Metric |             Car         | Pedestrian | Cyclist |         overall         |
| :---: |:-----------------------:| :---: | :---: |:-----------------------:|
| 3D-BBox | 85.9249 76.5969 73.8000 | 53.0198 46.4163 42.4853 | 83.1709 68.1972 62.9714 | 74.0385 63.7368 59.7522 | 
| BEV | 89.9877 87.4712 85.3771 | 60.7563 54.4434 49.2384 | 84.9880 70.9317 66.4696 | 78.5773 70.9488 67.0284 | 
| 2D-BBox | 90.5792 89.2327 86.4601 | 65.3155 60.7809 57.5007 | 88.8844 80.1228 76.0186 | 81.5930 76.7121 73.3265 |
| AOS | 90.4825 88.7307 85.6343 | 45.2281 42.3803 40.1351 | 88.4558 78.0712 73.8720 | 74.7221 69.7274 66.5471 |


## Compile

```
cd ops
python setup.py develop
```

## Datasets

1. Download

    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7418 .bin)
    ```

2. Pre-process KITTI datasets First

    ```
    cd PointPillars/
    python pre_process_kitti.py --data_root kitti_path
    ```

## Training

```
cd PointPillars/
python train.py --data_root kitti_path
```

## Evaluation

```
cd PointPillars/
python evaluate.py --ckpt pretrained/epoch_x.pth --data_root kitti_path 
```

## Visualization

```
cd PointPillars/

# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_x.pth --pc_path pc_path 

# 2. infer and visualize point cloud detection and gound truth.
python test.py --ckpt pretrained/epoch_x.pth --pc_path pc_path --calib_path calib_path  --gt_path gt_path

# 3. infer and visualize point cloud & image detection
python test.py --ckpt pretrained/epoch_x.pth --pc_path pc_path --calib_path calib_path --img_path img_path

```
