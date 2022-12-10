# 基于张量图的弱监督WSI分割 WSNTG


## Data Preparation

The mask-level fully-annotated dataset `data_glas` looks like this:
```
data_glas
├── train
│   ├── images
│   │   ├── train-1.png
│   │   └── train-2.png
│   └── masks
│       ├── train-1.png
│       └── train-2.png
└── val
    ├── images
    │   ├── val-1.png
    │   └── val-2.png
    └── masks
        ├── val-1.png
        └── val-2.png
```


### Generating point labels

```bash
generate_points.py
```
“labels”目录下的点标签csv文件，一共三列属性；每一行代表一个点，对应一个训练图像，
```csv
p1_top,p1_left,p1_class
p2_top,p2_left,p2_class
```

### Visualizing point labels

```bash
visualize_points.py 
```


## Training

### Training from scratch

```bash
python train.py
```



### Recording

记录目录的结构如下：

```
records/20190423-1122-AM
├── checkpoints
│   ├── ckpt.0001.pth
│   ├── ckpt.0002.pth
│   └── ckpt.0003.pth
├── curves
│   ├── loss.png
│   ├── pixel_acc.png
│   └── sp_acc.png
├── history.csv
├── params
│   ├── 0.json
│   └── 1.json
└── source
```

- `checkpoints`  目录存储所有培训检查点
- `curves` 存储学习损失曲线和所有指标
- `params` 存储参数
- `source` 存储所有源代码文件的快照
- `history.csv` 记录训练历史log

## Inference

```bash
infer_test_tile_DP2019.py
```
