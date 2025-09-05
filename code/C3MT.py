import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils.losses import loss_sup, loss_diff, ConLoss, contrastive_loss_sup
from config import get_config
import h5py

from dataloaders import utils
from dataloaders.dataset import (
    BaseDataSets,
    CTATransform,
    TwoStreamBatchSampler,
)

from networks.vision_transformer import SwinUnet as ViT_seg
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume

import re
from xml.etree.ElementInclude import default_loader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributions import Categorical
import augmentations
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/thyroid', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='thyroid/Cross_teaching_min_max', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vit', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
# costs
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency1', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency2', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency3', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument("--load", default=True, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per epoch')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')


parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "130": 1132, "126": 1058, "140": 1312}
    elif "Synapse2D" in dataset:
        ref_dict = {"2": 238, "4": 478}  # classnum = 14
    elif "LA" in dataset:
        ref_dict = {"8": 462,"16":942,"4":224}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(consistency, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_teacher_variables(model1, model2, ema_model, coefficient=0.99, alpha=0.99, global_step=0):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for t_param, t1_param, t2_param in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        t_param.data.mul_(alpha).add_(coefficient * (1 - alpha), t1_param.data).add_(
            (1 - coefficient) * (1 - alpha), t2_param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    #     loss_type = 'MT_loss'

    def create_model(net_type, ema=False):
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    student1 = ViT_seg(config, img_size=args.patch_size,
                       num_classes=args.num_classes).cuda()
    student1.load_from(config)
    student2 = ViT_seg(config, img_size=args.patch_size,
                       num_classes=args.num_classes).cuda()
    student2.load_from(config)
    teacher = ViT_seg(config, img_size=args.patch_size,
                      num_classes=args.num_classes).cuda()
    teacher.load_from(config)
    for param in teacher.parameters():
        param.detach_()
    projector_1 = create_model('projector', ema=True)
    projector_2 = create_model('projector', ema=True)
    projector_3 = create_model('projector')
    projector_4 = create_model('projector')

    mix_size = 32
    mixlistlong = mix_size * mix_size
    topk = 25
    sort_size = 2 * topk
    h, w = 224 // mix_size, 224 // mix_size
    s = h
    unfolds = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)
    folds = torch.nn.Fold(output_size=(224, 224), kernel_size=(h, w), stride=s).to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result

    def refresh_policies(db_train, cta, random_depth_weak, random_depth_strong):
        db_train.ops_weak = cta.policy(probe=False, weak=True)
        db_train.ops_strong = cta.policy(probe=False, weak=False)
        cta.random_depth_weak = random_depth_weak
        cta.random_depth_strong = random_depth_strong
        if max(Counter([a.f for a in db_train.ops_weak]).values()) >= 3 or max(
                Counter([a.f for a in db_train.ops_strong]).values()) >= 3:
            print('too deep with one transform, refresh again')
            refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)
        logging.info(f"CTA depth weak: {cta.random_depth_weak}")
        logging.info(f"CTA depth strong: {cta.random_depth_strong}")
        logging.info(f"\nWeak Policy: {db_train.ops_weak}")
        #         logging.info(f"\nWeak Policy: {max(Counter([a.f for a in db_train.ops_weak]).values())}")
        logging.info(f"Strong Policy: {db_train.ops_strong}")

    cta = augmentations.CTAugment()  # CTA增强
    # del cta.OPS['color']  # MRI是灰度图像
    transform = CTATransform(args.patch_size, cta)

    # sample initial weak and strong augmentation policies (CTAugment)样本初始弱增强和强增强策略
    ops_weak = cta.policy(probe=False, weak=True)
    ops_strong = cta.policy(probe=False, weak=False)
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transform,
        ops_weak=ops_weak,
        ops_strong=ops_strong,
    )

    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(student1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(student2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    alpha = 0.1
    iter_num = 0
    start_epoch = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    contrastive_loss_sup_criter = contrastive_loss_sup()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    # max_epoch = max_iterations
    best_performance1 = 0.0
    best_performance2 = 0.0
    lr_ = base_lr
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:  # 训练
        epoch_errors = []
        if iter_num <= 10000:
            random_depth_weak = np.random.randint(3, high=5)
            random_depth_strong = np.random.randint(2, high=5)
        elif iter_num >= 20000:
            random_depth_weak = 2
            random_depth_strong = 2
        else:
            random_depth_weak = np.random.randint(2, high=5)
            random_depth_strong = np.random.randint(2, high=5)
        refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)  # 刷新策略
        running_loss = 0.0
        running_sup_loss = 0.0
        running_unsup_loss = 0.0
        running_con_l_l = 0
        running_con_l_u = 0
        running_con_loss = 0
        ############################
        # Train
        ############################
        student1.train()
        student2.train()
        teacher.train()
        for i_batch, sampled_batch in enumerate(zip(trainloader)):
            raw_batch, weak_batch, strong_batch, label_batch_aug, label_batch = (
                sampled_batch[0]["image"],
                sampled_batch[0]["image_weak"],
                sampled_batch[0]["image_strong"],
                sampled_batch[0]["label_aug"],
                sampled_batch[0]["label"],
            )
            label_batch_aug[label_batch_aug >= num_classes] = 0
            label_batch_aug[label_batch_aug < 0] = 0
            image, imageA1, label = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch_aug.cuda(),
            )  # 将数据载入gpu
            # handle unfavorable cropping处理不利的裁剪
            # 非零值
            non_zero_ratio = torch.count_nonzero(label_batch) / (24 * 224 * 224)
            non_zero_ratio_aug = torch.count_nonzero(label_batch_aug) / (24 * 224 * 224)
            if non_zero_ratio > 0 and non_zero_ratio_aug < 0.005:  # try 0.01
                logging.info("Refreshing policy...")
                refresh_policies(db_train, cta, random_depth_weak, random_depth_strong)
            '''
            Step 1
            '''
            #################################################################################################################################
            # outputs for model输入强弱增强数据获取模型输出
            Train1_outputs_weak1 = student1(image)
            Train1_outputs_weak2 = student2(image)
            Train1_outputs_weak = teacher(image)
            Train1_outputs_strong1 = student1(imageA1)
            Train1_outputs_strong2 = student2(imageA1)

            Train1_outputs_weak_soft1 = torch.softmax(Train1_outputs_weak1, dim=1)
            Train1_outputs_weak_soft2 = torch.softmax(Train1_outputs_weak2, dim=1)
            Train1_outputs_weak_soft = torch.softmax(Train1_outputs_weak, dim=1)
            Train1_outputs_strong_soft1 = torch.softmax(Train1_outputs_strong1, dim=1)
            Train1_outputs_strong_soft2 = torch.softmax(Train1_outputs_strong2, dim=1)
            #################################################################################################################################
            '''
            Step 2
            '''
            Train1_pseudo_mask = (normalize(Train1_outputs_weak_soft) > args.conf_thresh).float()  # 阈值选择
            Train1_outputs_weak_soft_masked = (normalize(Train1_outputs_weak_soft)) * Train1_pseudo_mask
            Train1_pseudo = torch.argmax(Train1_outputs_weak_soft_masked.detach(), dim=1, keepdim=False)


            # Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map11 = torch.mean(torch.stack([Train1_outputs_weak_soft1, Train1_outputs_weak_soft]), dim=0)
                uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11 * torch.log(uncertainty_map11 + 1e-6), dim=1,
                                                     keepdim=True)
                uncertainty_map22 = torch.mean(torch.stack([Train1_outputs_weak_soft2, Train1_outputs_weak_soft]), dim=0)
                uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22 * torch.log(uncertainty_map22 + 1e-6), dim=1,
                                                     keepdim=True)
                B, C = image.shape[0], image.shape[1]
                # for student 1
                x11 = unfolds(uncertainty_map11)  # B x C*kernel_size[0]*kernel_size[1] x L
                x11 = x11.view(B, 1, h, w, -1)  # B x C x h x w x L
                x11_mean = torch.mean(x11, dim=(1, 2, 3))  # B x L
                _, x11_max_index = torch.sort(x11_mean, dim=1, descending=True)  # B x L B x L
                # for student 2
                x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L
                x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L
                x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L
                _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L
                ima_unfold = unfolds(image).view(B, C, h, w, -1)  # B x C x h x w x L
                img_unfold = unfolds(imageA1).view(B, C, h, w, -1)  # B x C x h x w x L
                lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L
                pse_unfold = unfolds(Train1_pseudo.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x

                # get label
                lab_unfold_reshaped = lab_unfold.squeeze(1).permute(0, 3, 1, 2)  # [B, L, h, w]
                lab_unfold_flat = lab_unfold_reshaped.reshape(batch_size, mixlistlong, -1)
                lab_unfold_majority = torch.mode(lab_unfold_flat, dim=2)[0]
                for i in range(B):
                    current_labels = lab_unfold_majority[i]  # L
                    idx1_high_uncertainty = x11_max_index[i, :topk]
                    # idx2_low_uncertainty = x22_max_index[i, -topk:]
                    expanded_low_range = min(sort_size, len(x22_max_index[i]))
                    idx2_low_uncertainty = x22_max_index[i, -expanded_low_range:]
                    available_low_blocks = idx2_low_uncertainty.tolist()
                    for idx_high in idx1_high_uncertainty:
                        high_label = current_labels[idx_high].item()
                        matching_low = [idx for idx in available_low_blocks if current_labels[idx] == high_label]
                        if matching_low:
                            idx_low = matching_low[0]
                            ima_unfold[i, :, :, :, idx_high] = ima_unfold[i, :, :, :, idx_low]
                            img_unfold[i, :, :, :, idx_high] = img_unfold[i, :, :, :, idx_low]
                            lab_unfold[i, :, :, :, idx_high] = lab_unfold[i, :, :, :, idx_low]
                            pse_unfold[i, :, :, :, idx_high] = pse_unfold[i, :, :, :, idx_low]
                            available_low_blocks.remove(idx_low)
                        else:
                            zero_label_low = [idx for idx in available_low_blocks if current_labels[idx] == 0]
                            if zero_label_low:
                                idx_low = zero_label_low[0]
                                ima_unfold[i, :, :, :, idx_high] = ima_unfold[i, :, :, :, idx_low]
                                img_unfold[i, :, :, :, idx_high] = img_unfold[i, :, :, :, idx_low]
                                lab_unfold[i, :, :, :, idx_high] = lab_unfold[i, :, :, :, idx_low]
                                pse_unfold[i, :, :, :, idx_high] = pse_unfold[i, :, :, :, idx_low]
                                available_low_blocks.remove(idx_low)
                    idx2_high_uncertainty = x22_max_index[i, :topk]
                    # idx1_low_uncertainty = x11_max_index[i, -topk:]
                    expanded_low_range = min(sort_size, len(x11_max_index[i]))
                    idx1_low_uncertainty = x11_max_index[i, -expanded_low_range:]
                    available_low_blocks2 = idx1_low_uncertainty.tolist()
                    for idx_high in idx2_high_uncertainty:
                        high_label = current_labels[idx_high].item()
                        matching_low = [idx for idx in available_low_blocks2 if current_labels[idx] == high_label]
                        if matching_low:
                            idx_low = matching_low[0]
                            ima_unfold[i, :, :, :, idx_high] = ima_unfold[i, :, :, :, idx_low]
                            img_unfold[i, :, :, :, idx_high] = img_unfold[i, :, :, :, idx_low]
                            lab_unfold[i, :, :, :, idx_high] = lab_unfold[i, :, :, :, idx_low]
                            pse_unfold[i, :, :, :, idx_high] = pse_unfold[i, :, :, :, idx_low]
                            available_low_blocks2.remove(idx_low)
                        else:
                            zero_label_low = [idx for idx in available_low_blocks2 if current_labels[idx] == 0]
                            if zero_label_low:
                                idx_low = zero_label_low[0]
                                ima_unfold[i, :, :, :, idx_high] = ima_unfold[i, :, :, :, idx_low]
                                img_unfold[i, :, :, :, idx_high] = img_unfold[i, :, :, :, idx_low]
                                lab_unfold[i, :, :, :, idx_high] = lab_unfold[i, :, :, :, idx_low]
                                pse_unfold[i, :, :, :, idx_high] = pse_unfold[i, :, :, :, idx_low]
                                available_low_blocks2.remove(idx_low)
                image2 = folds(ima_unfold.view(B, C * h * w, -1))
                imageA2 = folds(img_unfold.view(B, C * h * w, -1))
                label2 = folds(lab_unfold.view(B, 1 * h * w, -1)).squeeze(1)
                Train2_pseudo = folds(pse_unfold.view(B, 1 * h * w, -1)).squeeze(1).long()
            # #################################################################################################################################
            # outputs for model输入强弱增强数据获取模型输出
            Train2_outputs_weak1 = student1(image2)
            Train2_outputs_weak2 = student2(image2)
            Train2_outputs_strong1 = student1(imageA2)
            Train2_outputs_strong2 = student2(imageA2)

            Train2_outputs_weak_soft1 = torch.softmax(Train2_outputs_weak1, dim=1)
            Train2_outputs_weak_soft2 = torch.softmax(Train2_outputs_weak2, dim=1)
            Train2_outputs_strong_soft1 = torch.softmax(Train2_outputs_strong1, dim=1)
            Train2_outputs_strong_soft2 = torch.softmax(Train2_outputs_strong2, dim=1)
            # #################################################################################################################################
            consistency_weight1 = get_current_consistency_weight(args.consistency1,
                                                                 iter_num // 150)  # 一致性权重
            consistency_weight2 = get_current_consistency_weight(args.consistency2,
                                                                 iter_num // 150)  # 一致性权重
            consistency_weight3 = get_current_consistency_weight(args.consistency3,
                                                                 iter_num // 150)

            # supervised loss监督损失
            Train1_sup_loss1 = ce_loss(Train1_outputs_weak1[: args.labeled_bs],
                                       label[:][: args.labeled_bs].long(), ) + dice_loss(
                Train1_outputs_weak_soft1[: args.labeled_bs],
                label[: args.labeled_bs].unsqueeze(1),
            )

            Train1_sup_loss2 = ce_loss(Train1_outputs_weak2[: args.labeled_bs],
                                       label[:][: args.labeled_bs].long(), ) + dice_loss(
                Train1_outputs_weak_soft2[: args.labeled_bs],
                label[: args.labeled_bs].unsqueeze(1),
            )
            Train1_sup_loss = Train1_sup_loss1 + Train1_sup_loss2
            #             unsupervised loss standard无监督损失标准（强增强数据）
            Train1_unsup_loss1 = (
                    ce_loss(Train1_outputs_strong1[args.labeled_bs:], Train1_pseudo[args.labeled_bs:])
                    + dice_loss(Train1_outputs_strong_soft1[args.labeled_bs:],
                                Train1_pseudo[args.labeled_bs:].unsqueeze(1))
            )
            Train1_unsup_loss2 = (
                    ce_loss(Train1_outputs_strong2[args.labeled_bs:], Train1_pseudo[args.labeled_bs:])
                    + dice_loss(Train1_outputs_strong_soft2[args.labeled_bs:],
                                Train1_pseudo[args.labeled_bs:].unsqueeze(1))
            )
            Train1_unsup_loss = Train1_unsup_loss1 + Train1_unsup_loss2
            # # contrastive loss对比损失
            Train1_feat_l_q = projector_3(Train1_outputs_weak1[:args.labeled_bs])  # torch.Size([12, 16, 56, 56])
            Train1_feat_l_k = projector_4(Train1_outputs_weak2[:args.labeled_bs])
            Train1_Loss_contrast_l = contrastive_loss_sup_criter(Train1_feat_l_q, Train1_feat_l_k)  # 对比损失

            Train1_feat_q = projector_1(Train1_outputs_weak1[args.labeled_bs:])
            Train1_feat_q.detach()
            Train1_feat_k = projector_4(Train1_outputs_strong2[args.labeled_bs:])
            Train1_Loss_contrast_u_1 = contrastive_loss_sup_criter(Train1_feat_q, Train1_feat_k)  # 交叉对比损失

            Train1_feat_q = projector_2(Train1_outputs_weak2[args.labeled_bs:])
            Train1_feat_q.detach()
            Train1_feat_k = projector_3(Train1_outputs_strong1[args.labeled_bs:])
            Train1_Loss_contrast_u_2 = contrastive_loss_sup_criter(Train1_feat_q, Train1_feat_k)

            Train1_Loss_contrast_u = Train1_Loss_contrast_u_1 + Train1_Loss_contrast_u_2
            Train1_contrastive_loss = (Train1_Loss_contrast_l + Train1_Loss_contrast_u)

            Train1_loss = (Train1_sup_loss +
                           consistency_weight2 * Train1_Loss_contrast_l +
                           consistency_weight1 * Train1_unsup_loss +
                           consistency_weight3 * Train1_Loss_contrast_u)

            # supervised loss监督损失
            Train2_sup_loss1 = ce_loss(Train2_outputs_weak1[: args.labeled_bs],
                                       label2[:][: args.labeled_bs].long(), ) + dice_loss(
                Train2_outputs_weak_soft1[: args.labeled_bs],
                label2[: args.labeled_bs].unsqueeze(1),
            )

            Train2_sup_loss2 = ce_loss(Train2_outputs_weak2[: args.labeled_bs],
                                       label2[:][: args.labeled_bs].long(), ) + dice_loss(
                Train2_outputs_weak_soft2[: args.labeled_bs],
                label2[: args.labeled_bs].unsqueeze(1),
            )
            Train2_sup_loss = Train2_sup_loss1 + Train2_sup_loss2
            #             unsupervised loss standard无监督损失标准（强增强数据）
            Train2_unsup_loss1 = (
                    ce_loss(Train2_outputs_strong1[args.labeled_bs:], Train2_pseudo[args.labeled_bs:])
                    + dice_loss(Train2_outputs_strong_soft1[args.labeled_bs:],
                                Train2_pseudo[args.labeled_bs:].unsqueeze(1))
            )
            Train2_unsup_loss2 = (
                    ce_loss(Train2_outputs_strong2[args.labeled_bs:], Train2_pseudo[args.labeled_bs:])
                    + dice_loss(Train2_outputs_strong_soft2[args.labeled_bs:],
                                Train2_pseudo[args.labeled_bs:].unsqueeze(1))
            )
            Train2_unsup_loss = Train2_unsup_loss1 + Train2_unsup_loss2
            # # contrastive loss对比损失
            Train2_feat_l_q = projector_3(Train2_outputs_weak1[:args.labeled_bs])  # torch.Size([12, 16, 56, 56])
            Train2_feat_l_k = projector_4(Train2_outputs_weak2[:args.labeled_bs])
            Train2_Loss_contrast_l = contrastive_loss_sup_criter(Train2_feat_l_q, Train2_feat_l_k)  # 对比损失

            Train2_feat_q = projector_1(Train2_outputs_weak1[args.labeled_bs:])
            Train2_feat_q.detach()
            Train2_feat_k = projector_4(Train2_outputs_strong2[args.labeled_bs:])
            Train2_Loss_contrast_u_1 = contrastive_loss_sup_criter(Train2_feat_q, Train2_feat_k)  # 交叉对比损失

            Train2_feat_q = projector_2(Train2_outputs_weak2[args.labeled_bs:])
            Train2_feat_q.detach()
            Train2_feat_k = projector_3(Train2_outputs_strong1[args.labeled_bs:])
            Train2_Loss_contrast_u_2 = contrastive_loss_sup_criter(Train2_feat_q, Train2_feat_k)

            Train2_Loss_contrast_u = Train2_Loss_contrast_u_1 + Train2_Loss_contrast_u_2
            Train2_contrastive_loss = (Train2_Loss_contrast_l + Train2_Loss_contrast_u)
            contrastive_loss = (consistency_weight2 * (Train1_Loss_contrast_l + Train2_Loss_contrast_l) +
                                consistency_weight3 * (Train1_Loss_contrast_u + Train2_Loss_contrast_u)) * 0.5

            Train2_loss = (Train2_sup_loss +
                           consistency_weight2 * Train2_Loss_contrast_l +
                           consistency_weight1 * Train2_unsup_loss +
                           consistency_weight3 * Train2_Loss_contrast_u)

            loss = Train1_loss + Train2_loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            running_loss += loss
            running_sup_loss += (Train1_sup_loss1 + Train1_sup_loss2 + Train2_sup_loss1 + Train2_sup_loss2)
            running_unsup_loss += (Train1_unsup_loss + Train2_unsup_loss)
            running_con_loss += contrastive_loss
            running_con_l_l += (Train1_Loss_contrast_l+Train2_Loss_contrast_l)
            running_con_l_u += (Train1_Loss_contrast_u+Train2_Loss_contrast_u)

            update_ema_variables(projector_3, projector_1, args.ema_decay, iter_num)  # ema更新参数
            update_ema_variables(projector_4, projector_2, args.ema_decay, iter_num)
            update_teacher_variables(student1, student2, teacher, 0.99, args.ema_decay, iter_num)
            # track batch-level error, used to update augmentation policy
            epoch_errors.append(0.5 * loss.item())  # 损失转换为标量并记录

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  # 学习率
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("consistency_weight/consistency_weight1", consistency_weight1, iter_num)
            writer.add_scalar("consistency_weight/consistency_weight2", consistency_weight2, iter_num)
            writer.add_scalar("loss/model_loss", loss, iter_num)
            logging.info("Train2: iteration %d : model loss : %f" % (iter_num, loss.item()))

            if iter_num % 50 == 0:
                idx = args.labeled_bs
                # show weakly augmented image
                image = image[idx, 0:1, :, :]
                writer.add_image("train/WeakImage", image, iter_num)
                # show strongly augmented image
                image_strong = imageA1[idx, 0:1, :, :]
                writer.add_image("train/StrongImage", image_strong, iter_num)
                # show ground truth label
                labs = label_batch[idx, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)


            if iter_num > 0 and iter_num % 200 == 0:
                student1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], student1, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/student1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/student1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                for class_i in range(num_classes - 1):
                    logging.info('iteration %d : mean_dice : %f' % (iter_num, metric_list[class_i, 0]))
                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('eval/student1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('eval/student1_val_mean_hd95',
                                  mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    if performance1 > 0:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'student1_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance1, 4)))
                        save_best1 = os.path.join(snapshot_path,
                                                  '{}_best_student1.pth'.format(args.model))
                        util.save_checkpoint(epoch_num, student1, optimizer1, projector_1, projector_3,
                                             best_performance1,
                                             save_mode_path)
                        util.save_checkpoint(epoch_num, student1, optimizer1, projector_1, projector_3,
                                             best_performance1,
                                             save_best1)

                logging.info(
                    'iteration %d : student1_mean_dice : %f student1_mean_hd95 : %f' % (
                    iter_num, performance1, mean_hd951))
                student1.train()

                student2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], student2, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/student2_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/student2_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                for class_i in range(num_classes - 1):
                    logging.info('iteration %d : mean_dice : %f' % (iter_num, metric_list[class_i, 0]))
                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('eval/student2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('eval/student2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    if performance2 > 0:
                        save_mode_path = os.path.join(snapshot_path,
                                                      'student2_iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(best_performance2, 4)))
                        save_best2 = os.path.join(snapshot_path,
                                                  '{}_best_student2.pth'.format(args.model))

                        util.save_checkpoint(epoch_num, student2, optimizer2, projector_2, projector_4,
                                             best_performance2,
                                             save_mode_path)
                        util.save_checkpoint(epoch_num, student2, optimizer2, projector_2, projector_4,
                                             best_performance2,
                                             save_best2)

                logging.info(
                    'iteration %d : student2_mean_dice : %f student2_mean_hd95 : %f' % (
                    iter_num, performance2, mean_hd952))
                student2.train()
                logging.info(
                    'current best dice coef model 1 {}, model 2 {}'.format(best_performance1, best_performance2))

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'student1_iter_' + str(iter_num) + '.pth')
                util.save_checkpoint(epoch_num, student1, optimizer1, projector_1, projector_3, best_performance1,
                                     save_mode_path)
                logging.info("save student1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'student2_iter_' + str(iter_num) + '.pth')

                util.save_checkpoint(epoch_num, student2, optimizer2, projector_2, projector_4, best_performance2,
                                     save_mode_path)
                logging.info("save student2 to {}".format(save_mode_path))

        iterator.close()
        epoch_loss = running_loss / len(trainloader)
        epoch_sup_loss = running_sup_loss / len(trainloader)
        epoch_unsup_loss = running_unsup_loss / len(trainloader)
        epoch_con_loss = running_con_loss / len(trainloader)
        epoch_con_loss_u = running_con_l_u / len(trainloader)
        epoch_con_loss_l = running_con_l_l / len(trainloader)

        logging.info('{} Epoch [{:03d}/{:03d}]'.
                     format(datetime.now(), epoch_num, max_epoch))
        logging.info('Train loss: {}'.format(epoch_loss))
        writer.add_scalar('Train/Loss', epoch_loss, epoch_num)

        logging.info('Train sup loss: {}'.format(epoch_sup_loss))
        writer.add_scalar('Train/sup_loss', epoch_sup_loss, epoch_num)

        logging.info('Train unsup loss: {}'.format(epoch_unsup_loss))
        writer.add_scalar('Train/unsup_loss', epoch_unsup_loss, epoch_num)

        logging.info('Train contrastive loss: {}'.format(epoch_con_loss))
        writer.add_scalar('Train/contrastive_loss', epoch_con_loss, epoch_num)

        logging.info('Train contrastive loss l: {}'.format(epoch_con_loss_l))
        writer.add_scalar('Train/contrastive_loss_l', epoch_con_loss_l, epoch_num)

        logging.info('Train contrastive loss u: {}'.format(epoch_con_loss_u))
        writer.add_scalar('Train/contrastive_loss_u', epoch_con_loss_u, epoch_num)
        # update policy parameter bins for sampling
        mean_epoch_error = np.mean(epoch_errors)
        cta.update_rates(db_train.ops_weak, 1.0 - 0.5 * mean_epoch_error)
        cta.update_rates(db_train.ops_strong, 1.0 - 0.5 * mean_epoch_error)

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
