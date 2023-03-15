import torch
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector


def network(returned_layers, base_anchor_shapes,
        conv1_t_stride, n_input_channels, spatial_dims, fg_labels,
        balanced_sampler_pos_fraction, score_thresh, nms_thresh, val_patch_size,
        device, net_path):

    # 1. anchor generator
    # returned_layers: 目标boxes越小，设置越小
    # base_anchor_shapes: 最高分辨率的输出，目标boxes越小，设置越小
    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[2**l for l in range(len(returned_layers) + 1)],
        base_anchor_shapes=base_anchor_shapes
    )

    # 2. network
    conv1_t_size = [max(7, 2 * s + 1) for s in conv1_t_stride]
    backbone = resnet.ResNet(
        block=resnet.ResNetBottleneck,          # 深层网络选择 Bottleneck 结构，增加了 1x1卷积 减少参数量
        layers=[3, 4, 6, 3],                    # ResNet 各层设计   [3, 4, 6, 3]
        block_inplanes=resnet.get_inplanes(),   # [64, 128, 256, 512] 输出通道
        n_input_channels=n_input_channels,      # 第一个卷积层的输入 channel
        conv1_t_stride=conv1_t_stride,          # 第一个卷积核的 stride
        conv1_t_size=conv1_t_size               # 第一个卷积层的大小，决定 kernel 和 padding。
    )

    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=False,              # If pretrained_backbone is False, valid_trainable_backbone_layers = 5.
        trainable_backbone_layers=None,         # trainable_backbone_layers or 3 if None
        returned_layers=returned_layers         # 提取特征图的返回层
    )

    num_anchors = anchor_generator.num_anchors_per_location()[0]    # 3
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]  # [16, 16, 8]
    net = torch.jit.script(
        RetinaNet(
            spatial_dims=spatial_dims,
            num_classes=len(fg_labels),
            num_anchors=num_anchors,            # Return number of anchor shapes for each feature map.
            feature_extractor=feature_extractor,
            size_divisible=size_divisible       # 网络输入的空间大小应可由feature_extractor决定的size_divisible整除。
        )
    )

    if net_path != '':
        net = torch.jit.load(net_path).to(device)
        print(f"Load model from {net_path}")

    # 3 detector
    detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False).to(device)

    # set training components
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(
        batch_size_per_image=64,
        positive_fraction=balanced_sampler_pos_fraction,
        pool_size=20,
        min_neg=16,
    )
    detector.set_target_keys(box_key="box", label_key="label")

    # set validation components
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,          # 选择检测框阈值
        topk_candidates_per_level=1000,     # 每个feature map上保留的候选框数量    1000
        nms_thresh=nms_thresh,              # 去除重叠的检测框
        detections_per_img=100,             # 每张图最多保留的检测框数量          100
    )
    detector.set_sliding_window_inferer(
        roi_size=val_patch_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cpu",
    )

    return detector

