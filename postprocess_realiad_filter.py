import os
import json
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import get_data_transforms, RealIADDataset
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, compute_pro


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def f1_score_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return float(np.nanmax(f1s)) if f1s.size > 0 else 0.0


def build_vitill_base(device: str) -> nn.Module:
    encoder_name = "dinov2reg_vit_base_14"
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

    encoder = vit_encoder.load(encoder_name)
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.0, attn=LinearAttention2
        )
        for _ in range(8)
    ])
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    return model.to(device)


class SpatialFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, 1, 1] = 1.0
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-8)


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = prob.reshape(prob.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    dice = (2.0 * inter + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)
    return (1.0 - dice).mean()


def split_val_test(labels: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels.astype(bool))[0]
    neg_idx = np.where(~labels.astype(bool))[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_val = max(1, int(len(pos_idx) * val_ratio))
    n_neg_val = max(1, int(len(neg_idx) * val_ratio))
    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    test_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
    if len(test_idx) == 0:
        raise ValueError("No hold-out test samples left. Please reduce --val_ratio.")
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return val_idx.tolist(), test_idx.tolist()


@torch.no_grad()
def get_raw_map(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    en, de = model(img)
    raw_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
    return raw_map


def evaluate(model, filt, loader, device, max_ratio=0.01):
    model.eval()
    filt.eval()
    gt_px, before_px, after_px = [], [], []
    gt_sp, before_sp, after_sp = [], [], []
    print("[eval] start: collecting predictions for holdout set")

    with torch.no_grad():
        for img, gt, label, _ in tqdm(loader, desc="eval", dynamic_ncols=True, leave=False):
            img = img.to(device)
            gt = gt.to(device)
            raw = get_raw_map(model, img)
            raw_n = normalize_map(raw)
            after = torch.sigmoid(filt(raw_n))

            gt_bin = gt.bool()
            if gt_bin.shape[1] > 1:
                gt_bin = torch.max(gt_bin, dim=1, keepdim=True)[0]

            gt_np = gt_bin[:, 0].cpu().numpy().astype(np.uint8)
            before_np = raw[:, 0].cpu().numpy()
            after_np = after[:, 0].cpu().numpy()

            gt_px.append(gt_np.reshape(-1))
            before_px.append(before_np.reshape(-1))
            after_px.append(after_np.reshape(-1))
            gt_sp.append(label.numpy().astype(np.uint8).reshape(-1))

            if max_ratio <= 0:
                before_s = before_np.reshape(before_np.shape[0], -1).max(axis=1)
                after_s = after_np.reshape(after_np.shape[0], -1).max(axis=1)
            else:
                k = max(1, int(before_np.shape[1] * before_np.shape[2] * max_ratio))
                b_flat = before_np.reshape(before_np.shape[0], -1)
                a_flat = after_np.reshape(after_np.shape[0], -1)
                before_s = np.sort(b_flat, axis=1)[:, -k:].mean(axis=1)
                after_s = np.sort(a_flat, axis=1)[:, -k:].mean(axis=1)
            before_sp.append(before_s)
            after_sp.append(after_s)

    gt_px = np.concatenate(gt_px)
    before_px = np.concatenate(before_px)
    after_px = np.concatenate(after_px)
    gt_sp = np.concatenate(gt_sp)
    before_sp = np.concatenate(before_sp)
    after_sp = np.concatenate(after_sp)

    gt_px_3d = np.concatenate([x.reshape(1, *loader.dataset[i][1][0].shape) for i, x in enumerate([])]) if False else None
    # compute_pro expects [N, H, W], reconstruct from flattened arrays is inconvenient here;
    # use dataset-shaped accumulation for PRO below.

    # Re-run compact pass for PRO only to keep logic clear and avoid storing all maps in RAM.
    masks_3d, before_3d, after_3d = [], [], []
    print("[eval] computing 3D maps for AUPRO")
    with torch.no_grad():
        for img, gt, _, _ in tqdm(loader, desc="eval-pro-prep", dynamic_ncols=True, leave=False):
            img = img.to(device)
            gt = gt.to(device)
            raw = get_raw_map(model, img)
            after = torch.sigmoid(filt(normalize_map(raw)))
            gt_bin = gt.bool()
            if gt_bin.shape[1] > 1:
                gt_bin = torch.max(gt_bin, dim=1, keepdim=True)[0]
            masks_3d.append(gt_bin[:, 0].cpu().numpy().astype(np.uint8))
            before_3d.append(raw[:, 0].cpu().numpy())
            after_3d.append(after[:, 0].cpu().numpy())
    masks_3d = np.concatenate(masks_3d, axis=0)
    before_3d = np.concatenate(before_3d, axis=0)
    after_3d = np.concatenate(after_3d, axis=0)
    print("[eval] computing scalar metrics + AUPRO (this can take time)")

    print("[eval] AUPRO before...")
    aupro_before = float(compute_pro(masks_3d, before_3d))
    print("[eval] AUPRO after...")
    aupro_after = float(compute_pro(masks_3d, after_3d))

    result = {
        "before": {
            "auroc_sp": float(roc_auc_score(gt_sp, before_sp)),
            "ap_sp": float(average_precision_score(gt_sp, before_sp)),
            "f1_sp": float(f1_score_max(gt_sp, before_sp)),
            "auroc_px": float(roc_auc_score(gt_px, before_px)),
            "ap_px": float(average_precision_score(gt_px, before_px)),
            "f1_px": float(f1_score_max(gt_px, before_px)),
            "aupro_px": aupro_before,
        },
        "after": {
            "auroc_sp": float(roc_auc_score(gt_sp, after_sp)),
            "ap_sp": float(average_precision_score(gt_sp, after_sp)),
            "f1_sp": float(f1_score_max(gt_sp, after_sp)),
            "auroc_px": float(roc_auc_score(gt_px, after_px)),
            "ap_px": float(average_precision_score(gt_px, after_px)),
            "f1_px": float(f1_score_max(gt_px, after_px)),
            "aupro_px": aupro_after,
        },
    }
    print("[eval] done")
    return result


def main():
    parser = argparse.ArgumentParser("Train lightweight spatial post-filter on Real-IAD validation split")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--category", type=str, default="transistor1")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./saved_results")
    parser.add_argument("--save_name", type=str, default="postfilter_tran1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bce_weight", type=float, default=0.0)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_root = os.path.join(args.save_dir, args.save_name)
    os.makedirs(run_root, exist_ok=True)
    post_dir = os.path.join(run_root, "post_filter")
    os.makedirs(post_dir, exist_ok=True)

    print(f"device: {device}")
    print(f"run_root: {run_root}")
    print(f"backbone_ckpt: {args.backbone_ckpt}")

    data_transform, gt_transform = get_data_transforms(448, 392)
    full_test = RealIADDataset(
        root=args.data_path,
        category=args.category,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test",
    )
    labels = full_test.labels.astype(bool)
    val_idx, test_idx = split_val_test(labels, args.val_ratio, args.seed)
    val_set = Subset(full_test, val_idx)
    test_set = Subset(full_test, test_idx)
    print(f"full_test={len(full_test)}, val={len(val_set)}, holdout_test={len(test_set)}")
    print(f"val positives={int(labels[val_idx].sum())}, holdout positives={int(labels[test_idx].sum())}")

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    split_info = {
        "category": args.category,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "full_test_size": int(len(full_test)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "val_indices": [int(i) for i in val_idx],
        "test_indices": [int(i) for i in test_idx],
        "val_image_paths": [str(full_test.img_paths[i]) for i in val_idx],
        "test_image_paths": [str(full_test.img_paths[i]) for i in test_idx],
    }
    split_path = os.path.join(post_dir, "split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"saved split: {split_path}")

    model = build_vitill_base(device)
    state = torch.load(args.backbone_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    filt = SpatialFilter().to(device)
    opt = torch.optim.AdamW(filt.parameters(), lr=args.lr, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        filt.train()
        loss_hist = []
        pbar = tqdm(val_loader, desc=f"filter-train epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for img, gt, _, _ in pbar:
            img = img.to(device)
            gt = gt.to(device).float()
            with torch.no_grad():
                raw = get_raw_map(model, img)
                raw_n = normalize_map(raw)

            logits = filt(raw_n)
            prob = torch.sigmoid(logits)
            loss_dice = soft_dice_loss(prob, gt)
            loss = loss_dice
            if args.bce_weight > 0:
                loss_bce = bce_loss(logits, gt)
                loss = loss + args.bce_weight * loss_bce

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(loss_hist):.4f}", dice=f"{loss_dice.item():.4f}")

        print(f"epoch {epoch + 1}/{args.epochs}, mean_loss={np.mean(loss_hist):.6f}")

    filt_path = os.path.join(post_dir, "filter.pth")
    torch.save(filt.state_dict(), filt_path)
    print(f"saved filter: {filt_path}")

    metrics = evaluate(model, filt, test_loader, device)
    metrics_path = os.path.join(post_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved metrics: {metrics_path}")

    cfg = vars(args)
    cfg["device"] = device
    cfg_path = os.path.join(post_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"saved config: {cfg_path}")


if __name__ == "__main__":
    main()
