# Real-IAD uni training with pseudo anomaly + Dice loss (full copy of dinomaly_realiad_uni.py).
# Dice: only samples with gt_mask.sum()>0; per-map norm + sigmoid(x/dice_temp). See calculate_dice_loss.

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset_pseudo_dice import get_data_transforms, RealIADDataset, RealIADDatasetTrainWithPseudoMask
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine_hm_percent, regional_cosine_focal, \
    regional_cosine_hm, WarmCosineScheduler, visualize, cal_anomaly_maps
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
import sys
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def calculate_dice_loss(anomaly_map, gt_mask, temperature=0.1, smooth=1.0):
    """
    Dice only on samples with pseudo anomaly (gt_mask.sum() > 0).
    Per-image standardization + temperature-scaled sigmoid before Dice (avoids empty-mask
    degeneracy and cosine-scale / saturation issues).
    anomaly_map: [B, 1, H, W]; gt_mask: [B, 1, H, W]
    """
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


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def flush_logger(name):
    """Ensure progress lines appear in screen/log immediately (avoids looking 'stuck')."""
    for h in logging.getLogger(name).handlers:
        if hasattr(h, 'flush'):
            h.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_loss_curve(loss_history, save_path):
    if len(loss_history) == 0:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(item_list):
    setup_seed(1)

    total_iters = 50000
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        # root_path = '/data/disk8T2/guoj/Real-IAD'
        root_path = args.data_path

        train_data = RealIADDatasetTrainWithPseudoMask(
            root=root_path, category=item,
            image_size=image_size, crop_size=crop_size,
            pseudo_prob=args.pseudo_prob,
            min_patch_ratio=args.pseudo_min_ratio,
            max_patch_ratio=args.pseudo_max_ratio,
        )
        train_data.classes = item
        train_data.class_to_idx = {item: i}

        test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    # test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #                         for test_data in test_data_list]

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

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4))
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
    print_fn('pseudo_prob={}, dice_alpha={}, dice_temp={}, pseudo_min/max={}/{}'.format(
        args.pseudo_prob, args.dice_alpha, args.dice_temp, args.pseudo_min_ratio, args.pseudo_max_ratio))
    loss_history = []

    it = 0
    iters_per_epoch = len(train_dataloader)
    total_epochs = int(np.ceil(total_iters / iters_per_epoch))
    print_fn('total_iters:{}, iters_per_epoch:{}, approx_epochs:{}'.format(total_iters, iters_per_epoch, total_epochs))
    pbar = tqdm(total=total_iters, desc='train', dynamic_ncols=True)

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
            loss_history.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 50000 == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                print_fn('[eval] full test on {} classes (num_workers=0 to avoid dataloader hangs); first log may take long on large sets.'.format(len(item_list)))
                flush_logger(args.save_name)
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                for item, test_data in zip(item_list, test_data_list):
                    print_fn('[eval] {} — start ({} test images, metrics print right after inference)'.format(item, len(test_data)))
                    flush_logger(args.save_name)
                    # num_workers=0: long runs + CUDA often deadlock with forked workers; also no tqdm here so it looks "stuck".
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                   num_workers=0, pin_memory=False)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                    flush_logger(args.save_name)

                    if args.vis_samples_per_class > 0 and len(test_data) > 0:
                        print_fn('[eval] {} — saving heatmaps ({} samples)...'.format(item, min(args.vis_samples_per_class, len(test_data))))
                        flush_logger(args.save_name)
                        vis_count = min(args.vis_samples_per_class, len(test_data))
                        vis_indices = np.linspace(0, len(test_data) - 1, num=vis_count, dtype=int).tolist()
                        vis_subset = Subset(test_data, vis_indices)
                        vis_dataloader = torch.utils.data.DataLoader(vis_subset, batch_size=batch_size, shuffle=False,
                                                                      num_workers=0, pin_memory=False)
                        visualize(model, vis_dataloader, device, _class_=item,
                                  save_name=os.path.join(args.save_name, 'heatmap'))
                        flush_logger(args.save_name)
                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                model.train()

            it += 1
            pbar.update(1)

            if it % 20 == 0 and len(loss_list) > 0:
                pbar.set_postfix(epoch='{}/{}'.format(epoch + 1, total_epochs), loss='{:.4f}'.format(np.mean(loss_list)))

            if it % 100 == 0 or it == total_iters:
                if len(loss_list) > 0:
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
    loss_plot_path = os.path.join(args.save_dir, args.save_name, 'loss_curve.png')
    save_loss_curve(loss_history, loss_plot_path)
    print_fn('loss curve saved to {}'.format(loss_plot_path))
    print_fn('heatmaps saved under {}'.format(os.path.join('./visualize', args.save_name, 'heatmap')))

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/data/disk8T2/guoj/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_realiad_uni_pseudo_dice_dinov2br_c392r_en29_bn4dp4_de8_it50k_b16')
    parser.add_argument('--vis_samples_per_class', type=int, default=64)
    parser.add_argument('--pseudo_prob', type=float, default=0.5,
                        help='per-sample prob of CutPaste or Gaussian pseudo anomaly (same as MVTEC pseudo_dice)')
    parser.add_argument('--pseudo_min_ratio', type=float, default=0.05,
                        help='min pseudo patch size ratio (larger = stabler Dice early)')
    parser.add_argument('--pseudo_max_ratio', type=float, default=0.15)
    parser.add_argument('--dice_alpha', type=float, default=0.1,
                        help='loss = loss_recon + alpha * loss_dice')
    parser.add_argument('--dice_temp', type=float, default=0.1,
                        help='temperature in sigmoid after per-map norm: sharper maps when smaller')
    args = parser.parse_args()
    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # Use the first visible CUDA device by default to avoid invalid ordinal errors.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
