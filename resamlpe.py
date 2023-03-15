import argparse
import json
import yaml
import logging
import sys
from tqdm import tqdm

import monai
import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SaveImaged,
    Spacingd,
)


def main():
    parser = argparse.ArgumentParser(description="Detection Image Resampling")
    parser.add_argument(
        "-c",
        "--config",
        default="./environment/ias/config.yaml",
        help="config yaml file that stores hyper-parameters",
    )
    args = parser.parse_args()
    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    resample_config = config['resample_config']
    data_config = config['data_config']

    monai.config.print_config()

    # 1. define transform
    # resample images to args.spacing defined in args.config_file.
    process_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                meta_key_postfix="meta_dict",
                reader="itkreader",
                affine_lps_to_ras=True,
            ),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float16),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=resample_config['spacing'], padding_mode="border"),
        ]
    )
    # saved images to Nifti
    post_transforms = Compose(
        [
            SaveImaged(
                keys="image",
                meta_keys="image_meta_dict",
                output_dir=data_config['data_base_dir'],
                output_postfix="",
                resample=False,
            ),
        ]
    )

    # 2. prepare data
    for data_list_key in ["training", "validation"]:
        # create a data loader
        process_data = load_decathlon_datalist(
            data_config['data_list_file_path'],
            is_segmentation=True,
            data_list_key=data_list_key,
            base_dir=resample_config['orig_data_base_dir'],
        )
        process_ds = Dataset(
            data=process_data,
            transform=process_transforms,
        )
        process_loader = DataLoader(
            process_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=no_collation,
        )

        print("-" * 10)
        for batch_data in tqdm(process_loader):
            for batch_data_i in batch_data:
                batch_data_i = post_transforms(batch_data_i)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
