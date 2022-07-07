# 3D Detection

## 相关链接
https://github.com/Project-MONAI/tutorials/tree/main/detection

## 数据处理
``` bash
python3 luna16_prepare_env_files.py
python3 luna16_prepare_images.py -c ./config/config_train_luna16_16g.json
```

## 训练
```bash
python3 luna16_training.py \
    -e ./config/environment_luna16_fold0.json \
    -c ./config/config_train_luna16_16g.json
```