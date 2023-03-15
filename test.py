import torch
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
import yaml
from torchsummary import summary


# config = open('environment/ias/test.yaml')
# config = yaml.load(config, Loader=yaml.FullLoader)
# train_config = config['train_config']
# data_config = config['data_config']
# net_config = config['net_config']
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
# conv1_t_size = [max(7, 2 * s + 1) for s in net_config['conv1_t_stride']]
# backbone = resnet.ResNet(
#     block=resnet.ResNetBottleneck,
#     layers=[3, 4, 6, 3],
#     block_inplanes=resnet.get_inplanes(),
#     n_input_channels=net_config['n_input_channels'],
#     conv1_t_stride=net_config['conv1_t_stride'],
#     conv1_t_size=conv1_t_size
# )
#
# feature_extractor = resnet_fpn_feature_extractor(
#     backbone=backbone,
#     spatial_dims=net_config['spatial_dims'],
#     pretrained_backbone=False,
#     trainable_backbone_layers=None,
#     returned_layers=net_config['returned_layers']
# )
#
# anchor_generator = AnchorGeneratorWithAnchorShape(
#     feature_map_scales=[2 ** l for l in range(len(net_config['returned_layers']) + 1)],
#     base_anchor_shapes=net_config['base_anchor_shapes']
# )
#
# num_anchors = anchor_generator.num_anchors_per_location()[0]  # Return number of anchor shapes for each feature map.
# size_divisible = [s * 2 * 2 ** max(net_config['returned_layers']) for s in feature_extractor.body.conv1.stride]
#
# net = torch.jit.script(
#     RetinaNet(
#         spatial_dims=net_config['spatial_dims'],
#         num_classes=len(net_config['fg_labels']),
#         num_anchors=num_anchors,
#         feature_extractor=feature_extractor,
#         size_divisible=size_divisible  # 网络输入的空间大小应可由feature_extractor决定的size_divisible整除。
#     )
# ).to(device)
#
#
# patch = torch.randn([1, 1, 192, 192, 80]).to(device)
# print(patch.shape)
# out = net(patch)
# print(out)

from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

# 3D example outputs of one image from detector
val_outputs_all = [
    {"boxes": torch.tensor([[1, 1, 1, 3, 4, 5]], dtype=torch.float16),
     "labels": torch.randint(1, (1,)),
     "scores": torch.randn((1,)).absolute()},
]
val_targets_all = [
    {"boxes": torch.tensor([[1, 1, 1, 2, 6, 4]], dtype=torch.float16),
     "labels": torch.randint(1, (1,))},
]

coco_metric = COCOMetric(
    classes=['c0'], iou_list=[0.1], max_detection=[10]
)
results_metric = matching_batch(
    iou_fn=box_iou,
    iou_thresholds=coco_metric.iou_thresholds,
    pred_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_outputs_all],
    pred_classes=[val_data_i["labels"].numpy() for val_data_i in val_outputs_all],
    pred_scores=[val_data_i["scores"].numpy() for val_data_i in val_outputs_all],
    gt_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_targets_all],
    gt_classes=[val_data_i["labels"].numpy() for val_data_i in val_targets_all],
)
val_metric_dict = coco_metric(results_metric)[0]
for k in val_metric_dict.keys():
    print('{:40}  {:10}'.format(k, val_metric_dict[k]))
