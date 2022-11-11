# 3D Detection

## 相关链接
https://github.com/Project-MONAI/tutorials/tree/main/detection

## 数据处理
``` bash
python luna16_prepare_env_files.py
python luna16_prepare_images.py -c ./config/config_train_luna16_16g.json
```

## 数据集转换
```bash
python dataset_conversion/conversion.py
python dataset_conversion/json_generation.py
```

## 训练
```bash
python luna16_training.py \
    -e ./config/environment_luna16_fold0.json \
    -c ./config/config_train_luna16_16g.json
```

## 查看日志
``` bash
tensorboard --logdir=tfevent_train/event002
```