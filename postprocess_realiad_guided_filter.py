"""
Zero-parameter Guided Filter post-processing for Real-IAD anomaly detection.

算法核心：
  S_sharpened = GuidedFilter(Guide=I_RGB_gray, Input=S_raw, radius=r, eps=eps)

数学原理：
  局部线性模型  q_i = a_k * I_i + b_k  (i ∈ window k)
  a_k = (Cov(I,p) + eps) / (Var(I) + eps)
  b_k = mean(p) - a_k * mean(I)
  输出 q = mean(a_k) * I + mean(b_k)

  边缘区域 Var(I) >> eps  →  a_k ≈ Cov(I,p)/Var(I)  (边缘保留)
  平坦区域 Var(I) << eps  →  a_k ≈ 0, q ≈ mean(p)   (弥散压平)

使用方法：
  # 单组参数
  python postprocess_realiad_guided_filter.py \
    --data_path /root/autodl-tmp/Real-IAD \
    --category transistor1 \
    --backbone_ckpt ./saved_results/stand_tran1/classes/transistor1/checkpoints/model.pth \
    --save_dir ./saved_results --save_name gf_tran1 \
    --radius 8 --eps 0.01

  # 参数 sweep
  python postprocess_realiad_guided_filter.py ... --sweep \
    --radii 4,8,16,32 --epsilons 0.001,0.01,0.05,0.1
"""

import os
import json
import time
import random
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from dataset import get_data_transforms, RealIADDataset
from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, compute_pro


# ─────────────────────────────────────────────
#  基础工具
# ─────────────────────────────────────────────

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


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ─────────────────────────────────────────────
#  构建 Backbone
# ─────────────────────────────────────────────

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
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=target_layers, mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder,
    )
    return model.to(device)


# ─────────────────────────────────────────────
#  Guided Filter（纯 numpy / scipy 实现）
# ─────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def tensor_to_gray(img_tensor: torch.Tensor) -> np.ndarray:
    """
    将 DataLoader 输出的归一化图像 Tensor (C,H,W) 还原为 [0,1] 灰度图 (H,W)。
    """
    rgb = img_tensor.cpu().numpy() * _IMAGENET_STD + _IMAGENET_MEAN  # (3, H, W)
    rgb = np.clip(rgb, 0.0, 1.0)
    # ITU-R BT.601 亮度权重
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  # (H, W)
    return gray.astype(np.float32)


def guided_filter_single(guide: np.ndarray, src: np.ndarray,
                          radius: int, eps: float) -> np.ndarray:
    """
    单张图的 Guided Filter。
    guide : (H, W) float32，灰度引导图，范围 [0,1]
    src   : (H, W) float32，待滤波输入（原始异常热力图）
    返回  : (H, W) float32，滤波后热力图
    """
    d = 2 * radius + 1

    def box(x):
        return uniform_filter(x.astype(np.float64), size=d, mode='reflect').astype(np.float32)

    mean_I  = box(guide)
    mean_p  = box(src)
    mean_Ip = box(guide * src)
    mean_II = box(guide * guide)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I  = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a)
    mean_b = box(b)

    return mean_a * guide + mean_b


# ─────────────────────────────────────────────
#  推理 + 收集所有预测（一次性缓存，避免重复推理）
# ─────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: str):
    """
    遍历 loader 一次，返回：
      imgs_gray : list[np.ndarray (H,W)]   — 灰度引导图
      raw_maps  : list[np.ndarray (H,W)]   — 原始异常热力图
      masks     : list[np.ndarray (H,W)]   — 二值 GT mask
      labels    : list[int]                — 图像级 GT 标签
    """
    model.eval()
    imgs_gray, raw_maps, masks, labels = [], [], [], []

    print("[collect] start: running backbone inference on test set")
    for img_batch, gt_batch, label_batch, _ in tqdm(
        loader, desc="backbone-infer", dynamic_ncols=True
    ):
        img_batch = img_batch.to(device)

        en, de = model(img_batch)
        raw_batch, _ = cal_anomaly_maps(en, de, img_batch.shape[-1])  # (B,1,H,W)
        raw_batch = raw_batch[:, 0].cpu().numpy()  # (B,H,W)

        gt_bin = gt_batch.bool()
        if gt_bin.shape[1] > 1:
            gt_bin = torch.max(gt_bin, dim=1, keepdim=True)[0]
        gt_bin = gt_bin[:, 0].numpy().astype(np.uint8)  # (B,H,W)

        for b in range(img_batch.shape[0]):
            imgs_gray.append(tensor_to_gray(img_batch[b].cpu()))
            raw_maps.append(raw_batch[b])
            masks.append(gt_bin[b])
            labels.append(int(label_batch[b].item()))

    print(f"[collect] done: {len(raw_maps)} images")
    return imgs_gray, raw_maps, masks, labels


# ─────────────────────────────────────────────
#  应用 Guided Filter 并计算所有指标
# ─────────────────────────────────────────────

def compute_baseline(raw_maps, masks, labels, max_ratio=0.01):
    """
    计算 before 基准指标（只需运行一次，与 guided filter 参数无关）。
    返回 (before_dict, cached_arrays) 供后续 compute_after 复用。
    """
    H, W = raw_maps[0].shape
    raw_px_all = np.concatenate([m.reshape(-1) for m in raw_maps])
    mask_all   = np.concatenate([m.reshape(-1) for m in masks])
    labels_arr = np.array(labels, dtype=np.uint8)
    k          = max(1, int(H * W * max_ratio))
    raw_sp     = np.array([np.sort(m.reshape(-1))[-k:].mean() for m in raw_maps])
    mask_3d    = np.stack(masks,    axis=0).astype(np.uint8)
    raw_3d     = np.stack(raw_maps, axis=0)

    print("[baseline] computing AUPRO for raw maps (runs once) ...", flush=True)
    t0 = time.time()
    aupro_raw = float(compute_pro(
        mask_3d, raw_3d, show_progress=True, progress_desc="[AUPRO] baseline/raw"
    ))
    print(f"[baseline] AUPRO raw done ({time.time()-t0:.1f}s)  aupro={aupro_raw:.4f}", flush=True)

    before = {
        "auroc_sp": float(roc_auc_score(labels_arr, raw_sp)),
        "ap_sp"   : float(average_precision_score(labels_arr, raw_sp)),
        "f1_sp"   : float(f1_score_max(labels_arr, raw_sp)),
        "auroc_px": float(roc_auc_score(mask_all, raw_px_all)),
        "ap_px"   : float(average_precision_score(mask_all, raw_px_all)),
        "f1_px"   : float(f1_score_max(mask_all, raw_px_all)),
        "aupro_px": aupro_raw,
    }
    cache = {
        "mask_all": mask_all, "raw_px_all": raw_px_all,
        "labels_arr": labels_arr, "raw_sp": raw_sp,
        "mask_3d": mask_3d, "k": k,
    }
    return before, cache


def compute_after(filtered_maps, before, cache, tag=""):
    """
    计算单组 guided filter 参数的 after 指标，复用 baseline 缓存。
    """
    filt_px_all = np.concatenate([m.reshape(-1) for m in filtered_maps])
    filt_sp     = np.array([np.sort(m.reshape(-1))[-cache["k"]:].mean()
                            for m in filtered_maps])
    filt_3d     = np.stack(filtered_maps, axis=0)

    print(f"[metrics{tag}] AUPRO filt ...", flush=True)
    t0 = time.time()
    aupro_filt = float(compute_pro(
        cache["mask_3d"], filt_3d, show_progress=True, progress_desc=f"[AUPRO] filtered{tag}"
    ))
    print(f"[metrics{tag}] AUPRO filt done ({time.time()-t0:.1f}s)  aupro={aupro_filt:.4f}",
          flush=True)

    after = {
        "auroc_sp": float(roc_auc_score(cache["labels_arr"], filt_sp)),
        "ap_sp"   : float(average_precision_score(cache["labels_arr"], filt_sp)),
        "f1_sp"   : float(f1_score_max(cache["labels_arr"], filt_sp)),
        "auroc_px": float(roc_auc_score(cache["mask_all"], filt_px_all)),
        "ap_px"   : float(average_precision_score(cache["mask_all"], filt_px_all)),
        "f1_px"   : float(f1_score_max(cache["mask_all"], filt_px_all)),
        "aupro_px": aupro_filt,
    }
    return {"before": before, "after": after}


def apply_guided_filter(imgs_gray, raw_maps, radius, eps):
    """对整个列表批量应用 guided filter，返回滤波结果列表。"""
    filtered = []
    iterator = zip(imgs_gray, raw_maps)
    iterator = tqdm(
        iterator,
        total=len(raw_maps),
        desc=f"guided-filter r={radius} eps={eps}",
        dynamic_ncols=True,
        leave=False,
    )
    for guide, src in iterator:
        filtered.append(guided_filter_single(guide, src, radius, eps))
    return filtered


# ─────────────────────────────────────────────
#  主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Zero-parameter Guided Filter post-processing for Real-IAD"
    )
    # 数据 & 模型
    parser.add_argument("--data_path",     type=str, required=True)
    parser.add_argument("--category",      type=str, default="transistor1")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--save_dir",      type=str, default="./saved_results")
    parser.add_argument("--save_name",     type=str, default="gf_tran1")
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--seed",          type=int, default=1)
    parser.add_argument("--max_ratio",     type=float, default=0.01,
                        help="Top-K ratio for image-level score aggregation")

    # Guided Filter 单组参数
    parser.add_argument("--radius", type=int,   default=8,
                        help="Local window radius for guided filter")
    parser.add_argument("--eps",    type=float, default=0.01,
                        help="Regularization epsilon for guided filter")

    # 参数 sweep
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep over multiple radius/eps combinations")
    parser.add_argument("--radii",    type=str, default="4,8,16,32",
                        help="Comma-separated list of radii for sweep")
    parser.add_argument("--epsilons", type=str, default="0.001,0.01,0.05,0.1",
                        help="Comma-separated list of epsilon values for sweep")
    args = parser.parse_args()

    setup_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    run_root = os.path.join(args.save_dir, args.save_name)
    post_dir  = os.path.join(run_root, "guided_filter")
    os.makedirs(post_dir, exist_ok=True)

    print(f"device       : {device}")
    print(f"run_root     : {run_root}")
    print(f"backbone_ckpt: {args.backbone_ckpt}")
    print(f"category     : {args.category}")

    # ── 数据集（使用全部测试集，无需验证集划分）─────────────
    data_transform, gt_transform = get_data_transforms(448, 392)
    test_set = RealIADDataset(
        root=args.data_path, category=args.category,
        transform=data_transform, gt_transform=gt_transform, phase="test",
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4, drop_last=False,
    )
    n_pos = int(test_set.labels.astype(bool).sum())
    print(f"test set: {len(test_set)} images ({n_pos} anomalous / {len(test_set)-n_pos} normal)")

    # ── 加载 Backbone（冻结）──────────────────────────────
    model = build_vitill_base(device)
    state = torch.load(args.backbone_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("backbone loaded and frozen")

    # ── 一次性推理，缓存所有预测──────────────────────────
    imgs_gray, raw_maps, masks, labels = collect_predictions(model, test_loader, device)

    # ── 确定要评估的 (radius, eps) 组合 ──────────────────
    if args.sweep:
        radii_list    = parse_int_list(args.radii)
        epsilons_list = parse_float_list(args.epsilons)
        combos = [(r, e) for r in radii_list for e in epsilons_list]
        print(f"\nsweep: {len(combos)} combinations  "
              f"radii={radii_list}  epsilons={epsilons_list}\n")
    else:
        combos = [(args.radius, args.eps)]

    # ── baseline（只算一次，sweep 全程共享）────────────────
    print("\n" + "="*60)
    print("[baseline] computing raw-map metrics (shared across all sweep combos)")
    before, cache = compute_baseline(raw_maps, masks, labels, max_ratio=args.max_ratio)
    print("[baseline] scalar metrics (before):")
    for k_m, v in before.items():
        print(f"  {k_m:<12} {v:.4f}")

    sweep_results = []
    best_score    = -1.0
    best_entry    = None

    overall_bar = tqdm(combos, desc="guided-filter-sweep", dynamic_ncols=True)
    for radius, eps in overall_bar:
        overall_bar.set_postfix(radius=radius, eps=eps)
        tag = f" [r={radius}, eps={eps}]"

        print(f"\n{'='*60}")
        print(f"[guided filter]{tag}  applying ...")
        t_start = time.time()
        filtered_maps = apply_guided_filter(imgs_gray, raw_maps, radius, eps)
        print(f"[guided filter]{tag}  filter done ({time.time()-t_start:.1f}s), computing metrics ...")

        result = compute_after(filtered_maps, before, cache, tag=tag)

        before = result["before"]
        after  = result["after"]

        # 综合得分：ap_px + aupro_px（同之前 sweep 脚本保持一致）
        score = after["ap_px"] + after["aupro_px"]

        delta = {k: round(after[k] - before[k], 6) for k in before}
        entry = {
            "radius": radius, "eps": eps,
            "score_ap_aupro_sum": round(score, 6),
            "before": before, "after": after, "delta": delta,
        }
        sweep_results.append(entry)

        print(f"[result]{tag}")
        print(f"  {'metric':<12} {'before':>8}  {'after':>8}  {'delta':>8}")
        print(f"  {'-'*44}")
        for k in before:
            print(f"  {k:<12} {before[k]:>8.4f}  {after[k]:>8.4f}  {after[k]-before[k]:>+8.4f}")
        print(f"  score(ap_px+aupro_px) = {score:.4f}")

        if score > best_score:
            best_score = score
            best_entry = entry

    # ── 保存结果 ─────────────────────────────────────────
    sweep_path = os.path.join(post_dir, "sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved sweep results : {sweep_path}")

    best_path = os.path.join(post_dir, "best_by_ap_aupro_sum.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_entry, f, indent=2, ensure_ascii=False)
    print(f"saved best config   : {best_path}")

    # 单组参数时额外保存 metrics.json（方便快速查看）
    if not args.sweep:
        metrics_path = os.path.join(post_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results[0], f, indent=2, ensure_ascii=False)
        print(f"saved metrics       : {metrics_path}")

    cfg = {k: v for k, v in vars(args).items()}
    cfg["device"]     = device
    cfg["test_size"]  = len(test_set)
    cfg["n_anomalous"] = n_pos
    cfg_path = os.path.join(post_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"saved config        : {cfg_path}")

    print("\n" + "="*60)
    print("BEST CONFIGURATION:")
    print(json.dumps(best_entry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
