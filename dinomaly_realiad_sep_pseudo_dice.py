# Real-IAD sep (per-class) training with pseudo anomaly + Dice — full copy of dinomaly_realiad_sep.py.
# Dataset: dataset_pseudo_dice.RealIADDatasetTrainWithPseudoMask; loss aligned with dinomaly_realiad_uni_pseudo_dice.

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset_pseudo_dice import get_data_transforms, RealIADDataset, RealIADDatasetTrainWithPseudoMask
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, WarmCosineScheduler, \
    cal_anomaly_maps
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import json
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REALIAD_UNI_CLASSES = [
    'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
    'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
    'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
    'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
    'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper',
]


def parse_class_list(arg_classes):
    if not arg_classes:
        return list(REALIAD_UNI_CLASSES)
    unknown = [c for c in arg_classes if c not in REALIAD_UNI_CLASSES]
    if unknown:
        raise SystemExit('Unknown class names: {}'.format(unknown))
    return list(arg_classes)


def calculate_dice_loss(anomaly_map, gt_mask, temperature=0.1, smooth=1.0):
    """Same as dinomaly_realiad_uni_pseudo_dice: only gt_mask.sum()>0; norm + sigmoid/temp before Dice."""
    B = anomaly_map.size(0)
    total = None
    valid = 0
    for i in range(B):
        if gt_mask[i].sum() <= 0:
            continue
        p = anomaly_map[i : i + 1]
        t = gt_mask[i : i + 1].float()
        if t.shape[1] > 1:
            t = torch.max(t, dim=1, keepdim=True)[0]
        p_norm = (p - p.mean()) / (p.std() + 1e-8)
        p_prob = torch.sigmoid(p_norm / temperature)
        p_flat = p_prob.reshape(-1)
        t_flat = t.reshape(-1)
        inter = (p_flat * t_flat).sum()
        dice = (2.0 * inter + smooth) / (p_flat.sum() + t_flat.sum() + smooth)
        li = 1.0 - dice
        total = li if total is None else total + li
        valid += 1
    if valid == 0:
        return (anomaly_map * 0.0).sum()
    return total / valid


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


def get_logger(name, save_path=None, level='INFO', log_filename='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, log_filename))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item):
    setup_seed(1)
    print_fn(item)
    total_iters = 5000
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # root_path = '/data/disk8T2/guoj/Real-IAD'
    # root_path = '/home/ubuntu/nfs/8T2/guoj/Real-IAD'
    root_path = args.data_path
    train_data = RealIADDatasetTrainWithPseudoMask(
        root=root_path, category=item,
        image_size=image_size, crop_size=crop_size,
        pseudo_prob=args.pseudo_prob,
        min_patch_ratio=args.pseudo_min_ratio,
        max_patch_ratio=args.pseudo_max_ratio,
    )
    test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                               phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # encoder_name = 'dinov2reg_vit_small_14'
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            tmp_weight = m.weight.detach().cpu()
            trunc_normal_(tmp_weight, std=0.01, a=-0.03, b=0.03)
            m.weight.data.copy_(tmp_weight.to(m.weight.device))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))
    print_fn('pseudo_prob={}, dice_alpha={}, dice_temp={}, pseudo_min/max={}/{}'.format(
        args.pseudo_prob, args.dice_alpha, args.dice_temp, args.pseudo_min_ratio, args.pseudo_max_ratio))

    it = 0
    iters_per_epoch = len(train_dataloader)
    total_epochs = int(np.ceil(total_iters / iters_per_epoch))
    print_fn('total_iters:{}, iters_per_epoch:{}, approx_epochs:{}'.format(total_iters, iters_per_epoch, total_epochs))
    pbar = tqdm(total=total_iters, desc='train-{}'.format(item), dynamic_ncols=True)
    for epoch in range(total_epochs):
        model.train()

        loss_list = []
        loss_recon_list = []
        loss_dice_list = []
        for img, label, gt_mask in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            gt_mask = gt_mask.to(device)

            en, de = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss_recon = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            loss_dice = calculate_dice_loss(anomaly_map, gt_mask, temperature=args.dice_temp, smooth=1.0)
            loss = loss_recon + args.dice_alpha * loss_dice

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            loss_recon_list.append(loss_recon.item())
            loss_dice_list.append(loss_dice.item())
            lr_scheduler.step()

            if (it + 1) % 5000 == 0:
                print_fn('[eval] {} start: {} test images'.format(item, len(test_data)))
                results = evaluation_batch(
                    model, test_dataloader, device, _class_=item, max_ratio=0.01, resize_mask=256,
                    show_progress=True, progress_desc='eval-{}'.format(item), progress_leave=False
                )
                auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                print_fn(
                    '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                print_fn('[eval] {} done'.format(item))
                model.train()

            it += 1
            pbar.update(1)

            if it % 20 == 0 and len(loss_list) > 0:
                pbar.set_postfix(
                    epoch='{}/{}'.format(epoch + 1, total_epochs),
                    loss='{:.4f}'.format(np.mean(loss_list)),
                    recon='{:.4f}'.format(np.mean(loss_recon_list)),
                    dice='{:.4f}'.format(np.mean(loss_dice_list)),
                )

            if it % 100 == 0 or it == total_iters:
                print_fn(
                    'iter [{}/{}], loss_recon:{:.4f}, loss_dice:{:.4f}, loss:{:.4f} (dice_w*{:.2f}: {:.4f})'.format(
                        it, total_iters,
                        np.mean(loss_recon_list), np.mean(loss_dice_list), np.mean(loss_list),
                        args.dice_alpha, args.dice_alpha * np.mean(loss_dice_list)))
                if it % 100 == 0:
                    loss_list = []
                    loss_recon_list = []
                    loss_dice_list = []

            if it == total_iters:
                break

        if it == total_iters:
            break

    pbar.close()

    if not args.no_save_model:
        class_root = os.path.join(args.save_dir, args.save_name, 'classes', item)
        ckpt_dir = os.path.join(class_root, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'model.pth')
        torch.save(model.state_dict(), ckpt_path)
        print_fn('saved checkpoint: {}'.format(ckpt_path))

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/data/disk8T2/guoj/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_realiad_sep_pseudo_dice_dinov2br_it5k_b16')
    parser.add_argument('--pseudo_prob', type=float, default=0.5)
    parser.add_argument('--pseudo_min_ratio', type=float, default=0.05)
    parser.add_argument('--pseudo_max_ratio', type=float, default=0.15)
    parser.add_argument('--dice_alpha', type=float, default=0.5,
                        help='loss = loss_recon + alpha * loss_dice (try 0.1 if Dice dominates)')
    parser.add_argument('--dice_temp', type=float, default=0.1)
    parser.add_argument('--classes', type=str, nargs='*', default=None,
                        help='Categories to train/eval (one model per class). Omit for all 30 Real-IAD uni classes.')
    parser.add_argument('--no_save_model', action='store_true',
                        help='Do not save checkpoints under <save_dir>/<save_name>/classes/<category>/checkpoints/')
    args = parser.parse_args()

    item_list = parse_class_list(args.classes)
    run_root = os.path.join(args.save_dir, args.save_name)
    log_dir = os.path.join(run_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(args.save_name, log_dir, log_filename='run.log')
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    print_fn('run_root: {}'.format(run_root))
    print_fn('classes ({}): {}'.format(len(item_list), item_list))

    result_list = []
    for i, item in enumerate(item_list):
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = train(item)
        result_list.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))

    metrics_dir = os.path.join(run_root, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, 'summary.json')
    summary = {
        'save_name': args.save_name,
        'data_path': args.data_path,
        'pseudo_prob': args.pseudo_prob,
        'pseudo_min_ratio': args.pseudo_min_ratio,
        'pseudo_max_ratio': args.pseudo_max_ratio,
        'dice_alpha': args.dice_alpha,
        'dice_temp': args.dice_temp,
        'classes': item_list,
        'per_class': [
            {
                'category': r[0],
                'auroc_sp': float(r[1]),
                'ap_sp': float(r[2]),
                'f1_sp': float(r[3]),
                'auroc_px': float(r[4]),
                'ap_px': float(r[5]),
                'f1_px': float(r[6]),
                'aupro_px': float(r[7]),
            }
            for r in result_list
        ],
        'mean': {
            'auroc_sp': float(mean_auroc_sp),
            'ap_sp': float(mean_ap_sp),
            'f1_sp': float(mean_f1_sp),
            'auroc_px': float(mean_auroc_px),
            'ap_px': float(mean_ap_px),
            'f1_px': float(mean_f1_px),
            'aupro_px': float(mean_aupro_px),
        },
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_fn('metrics written: {}'.format(summary_path))
