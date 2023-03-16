import torch.cuda
from monai.transforms import ScaleIntensityRanged
from generate_transforms import *
from monai.data import DataLoader, Dataset, box_utils, load_decathlon_datalist, CacheDataset
from monai.data.utils import no_collation


def datasets(gt_box_mode, patch_size, batch_size,
             data_list_file_path, data_base_dir, train_len, a_min, a_max, train_batch, cache=False):
    # define transform
    intensity_transform = ScaleIntensityRanged(
        keys=['image'],
        a_min=a_min,
        a_max=a_max,
        b_min=0.0,
        b_max=1.0,
        clip=True
    )

    train_transforms = generate_detection_train_transform(
        'image',
        'box',
        'label',
        gt_box_mode,
        intensity_transform,
        patch_size,
        batch_size,
        affine_lps_to_ras=True,     # True
        amp=True
    )

    val_transforms = generate_detection_val_transform(
        'image',
        'box',
        'label',
        gt_box_mode,
        intensity_transform,
        affine_lps_to_ras=True,     # True
        amp=True
    )

    train_data = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key='training',
        base_dir=data_base_dir
    )

    val_data = load_decathlon_datalist(
        data_list_file_path,
        is_segmentation=True,
        data_list_key='validation',
        base_dir=data_base_dir
    )

    if cache:
        train_ds = CacheDataset(data=train_data[: int(train_len * len(train_data))],
                                transform=train_transforms, cache_rate=1.0, num_workers=4)
        val_ds = CacheDataset(data=val_data[: int(train_len * len(val_data))],
                              transform=val_transforms, cache_rate=1.0, num_workers=4)
    else:
        train_ds = Dataset(
            data=train_data[: int(train_len * len(train_data))],
            transform=train_transforms
        )
        val_ds = Dataset(
            data=val_data[: int(train_len * len(val_data))],
            transform=val_transforms
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),   # 将数据从主机内存加载到CUDA可访问内存
        collate_fn=no_collation,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=no_collation,
        persistent_workers=True
    )

    return train_ds, train_loader, val_loader
