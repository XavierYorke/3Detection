# IA Detection

## 数据处理

1. 对数据和标签进行resample
```bash
python resample.py
```

2. 根据分割标签生成检测标签
```bash
python data_preparation/nii2csv.py
python data_preparation/data_analysis.py
```

## 训练

```bash
python training.py
```