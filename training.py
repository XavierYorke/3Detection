import argparse
import torch
import torch.backends.cudnn
import yaml
from monai.utils import set_determinism
import monai
from data_loader import datasets
from network import network
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import time
from monai.apps.detection.metrics.coco import COCOMetric
import gc
from visualize_image import visualize_one_xy_slice_in_3d_image
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import box_utils
import os
import numpy as np
import matplotlib.pyplot as plt


def main(seed, lr, event_path,
        val_interval, max_epochs,
        resume, val_patch_size):
    set_determinism(seed)

    amp = True
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    monai.config.print_config()

    # cudnn衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
    torch.backends.cudnn.benchmark = True
    # 设置运行时线程数，过多会导致CPU负载过高，降低性能
    torch.set_num_threads(4)

    # initial optimizer
    optimizer = torch.optim.SGD(
        detector.network.parameters(),
        float(lr),
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True,
    )
    after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    optimizer.zero_grad()
    optimizer.step()

    # initialize tensorboard writer

    if resume > 0:
        print(f'resume: {resume}')
    else:
        event_path = os.path.join(event_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.exists(event_path):
            os.makedirs(event_path)
    # batch_img = os.path.join(event_path, 'batch_img')
    # if not os.path.exists(batch_img):
    #     os.makedirs(batch_img)

    with open(event_path + '/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f, allow_unicode=True)
    tensorboard_writer = SummaryWriter(event_path)
    print('tensorboard_writer prepared, training soon...')

    # 5. train
    val_interval = val_interval  # do validation every val_interval epochs
    coco_metric = COCOMetric(classes=["nodule"], iou_list=[0.1], max_detection=[100])
    best_val_epoch_metric = 0.0
    best_val_epoch = -1  # the epoch that gives best validation metrics

    max_epochs = max_epochs
    epoch_len = len(train_ds) // train_loader.batch_size
    if len(train_ds) % train_loader.batch_size != 0:
        epoch_len += 1
    w_cls = 1.0  # weight between classification loss and box regression loss, default 1.0
    for epoch in range(resume, max_epochs):
        # ------------- Training -------------
        print(f"epoch {epoch + 1}/{max_epochs}")
        detector.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_reg_loss = 0
        step = 0
        start_time = time.time()
        scheduler_warmup.step()
        # Training
        for batch_data in train_loader:     # batch_size(16) * list - batch_size(4) * list - box image label
            step += 1
            inputs = [      # batch_size(16) * batch_size(4)
                batch_data_ii["image"].to(device).contiguous() for batch_data_i in batch_data for batch_data_ii in batch_data_i
            ]
            targets = [
                dict(
                    label=batch_data_ii["label"].to(device).contiguous(),
                    box=batch_data_ii["box"].to(device).contiguous(),
                )
                for batch_data_i in batch_data
                for batch_data_ii in batch_data_i
            ]

            for idx in range(len(targets)):
                if targets[idx]["box"].numel() != 0:
                    draw_img = visualize_one_xy_slice_in_3d_image(
                        gt_boxes=targets[idx]["box"].cpu().detach().numpy(),
                        image=inputs[idx][0, ...].cpu().detach().numpy(),
                        pred_boxes=[],
                    )
                    # plt.imshow(draw_img, cmap='gray')

                    # plt.imsave(os.path.join(batch_img, 'epoch_' + str(epoch) + '_step_' + str(step) + '.png'),
                    #            draw_img, cmap='gray')
                    # plt.show()
                    tensorboard_writer.add_image("train_img_xy",
                                                 draw_img.transpose([2, 1, 0]), epoch_len * epoch + step)
                    break
                if idx == 1:
                    break

            for param in detector.network.parameters():
                param.grad = None

            # inputs = [i.contiguous() for i in inputs]

            if amp and (scaler is not None):
                with torch.cuda.amp.autocast():
                    outputs = detector(inputs, targets)
                    loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = detector(inputs, targets)
                loss = w_cls * outputs[detector.cls_key] + outputs[detector.box_reg_key]
                loss.backward()
                optimizer.step()

            # save to tensorboard
            epoch_loss += loss.detach().item()
            epoch_cls_loss += outputs[detector.cls_key].detach().item()
            epoch_box_reg_loss += outputs[detector.box_reg_key].detach().item()
            print(f"{epoch + 1}/{max_epochs} - {step}/{epoch_len}, train_loss: {loss.item():.4f}")
            tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)

        end_time = time.time()
        print(f"Training time: {end_time - start_time}s")
        del inputs, batch_data
        torch.cuda.empty_cache()
        gc.collect()

        # save to tensorboard
        epoch_loss /= step
        epoch_cls_loss /= step
        epoch_box_reg_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_cls_loss", epoch_cls_loss, epoch + 1)
        tensorboard_writer.add_scalar("avg_train_box_reg_loss", epoch_box_reg_loss, epoch + 1)
        tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # save last trained model
        torch.jit.save(detector.network, os.path.join(event_path, 'last-' + str(epoch + 1) + '.pt'))
        if os.path.exists(os.path.join(event_path, 'last-' + str(epoch) + '.pt')):
            os.remove(os.path.join(event_path, 'last-' + str(epoch) + '.pt'))
        print("saved last model")

        # ------------- Validation for model selection -------------
        if (epoch + 1) % val_interval == 0:
            detector.eval()
            val_outputs_all = []
            val_targets_all = []
            start_time = time.time()
            with torch.no_grad():
                for val_data in val_loader:
                    # if all val_data_i["image"] smaller than args.val_patch_size, no need to use inferer
                    # otherwise, need inferer to handle large input images.
                    use_inferer = not all(
                        [val_data_i["image"][0, ...].numel() < np.prod(val_patch_size) for val_data_i in val_data]
                    )
                    val_inputs = [val_data_i.pop("image").to(device) for val_data_i in val_data]

                    if amp:
                        with torch.cuda.amp.autocast():
                            val_outputs = detector(val_inputs, use_inferer=use_inferer)
                    else:
                        val_outputs = detector(val_inputs, use_inferer=use_inferer)

                    # save outputs for evaluation
                    val_outputs_all += val_outputs
                    val_targets_all += val_data

            end_time = time.time()
            print(f"Validation time: {end_time - start_time}s")

            # visualize an inference image and boxes to tensorboard
            draw_img = visualize_one_xy_slice_in_3d_image(
                gt_boxes=val_data[0]["box"].cpu().detach().numpy(),
                image=val_inputs[0][0, ...].cpu().detach().numpy(),
                pred_boxes=val_outputs[0][detector.target_box_key].cpu().detach().numpy(),
            )
            tensorboard_writer.add_image("val_img_xy", draw_img.transpose([2, 1, 0]), epoch + 1)

            # compute metrics
            del val_inputs
            torch.cuda.empty_cache()
            results_metric = matching_batch(
                iou_fn=box_utils.box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[
                    val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                pred_scores=[
                    val_data_i[detector.pred_score_key].cpu().detach().numpy() for val_data_i in val_outputs_all
                ],
                gt_boxes=[val_data_i[detector.target_box_key].cpu().detach().numpy() for val_data_i in val_targets_all],
                gt_classes=[
                    val_data_i[detector.target_label_key].cpu().detach().numpy() for val_data_i in val_targets_all
                ],
            )
            val_epoch_metric_dict = coco_metric(results_metric)[0]
            # print(val_epoch_metric_dict)

            # write to tensorboard event
            for k in val_epoch_metric_dict.keys():
                tensorboard_writer.add_scalar("val_" + k, val_epoch_metric_dict[k], epoch + 1)
                print('{:40}  {:10}'.format(k, val_epoch_metric_dict[k]))
            val_epoch_metric = val_epoch_metric_dict.values()
            val_epoch_metric = sum(val_epoch_metric) / len(val_epoch_metric)
            tensorboard_writer.add_scalar("val_metric", val_epoch_metric, epoch + 1)

            # save best trained model
            if val_epoch_metric >= best_val_epoch_metric:
                best_val_epoch_metric = val_epoch_metric
                best_val_epoch = epoch + 1
                torch.jit.save(detector.network, os.path.join(event_path, 'best.pt'))
                print("saved new best metric model")
            print(
                "current epoch: {} current metric: {:.4f} "
                "best metric: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_metric, best_val_epoch_metric, best_val_epoch
                )
            )

    print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
    tensorboard_writer.close()


def parser_args():
    parser = argparse.ArgumentParser(description="3D Detection Training")
    parser.add_argument('-c', '--config', default='./environment/ias/config.yaml')
    # parser.add_argument('-c', '--config', default='./environment/Lung/config.yaml')
    # parser.add_argument('-c', '--config', default='./environment/ias/config-A.yaml')
    # parser.add_argument('-c', '--config', default='./environment/luna16/luna16_config.yaml')
    # parser.add_argument('-c', '--config', default='./environment/CADA/config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    train_config = config['train_config']
    data_config = config['data_config']
    net_config = config['net_config']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, train_loader, val_loader = datasets(**data_config)
    detector = network(**net_config, device=device)

    main(**train_config, val_patch_size=net_config['val_patch_size'])
