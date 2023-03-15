# IA Detection
官方链接：https://github.com/Project-MONAI/tutorials/tree/main/detection
## 项目准备

### 文件夹
- environment：存放config.yaml
- train_event：存放训练结果
- result：存放测试结果
- objs：存放推理生成框文件

## 数据处理
1. 对数据和标签进行resample
```bash
python resample.py
```

2. 根据分割标签生成检测标签
```bash
python data_preparation/get_bbox.py
```

## 训练

```bash
python training.py
```
查看tensorboard：```tensorboard --logdir path```

## 测试
```bash
python testing.py
```

## 根据bbox.json生成obj
- 配合原始数据在3D Slicer上可视化
- 也可以转换标签的box.json用于数据验证
```bash
python save_obj.py
```