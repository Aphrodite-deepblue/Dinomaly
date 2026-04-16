import time, numpy as np, sys
sys.path.insert(0, '/root/autodl-tmp/Dinomaly')
from utils import compute_pro

rng = np.random.RandomState(0)
N, H, W = 200, 392, 392
masks = np.zeros((N, H, W), dtype=np.uint8)
for i in range(120):
    r0, c0 = rng.randint(50, 300, 2)
    r1 = r0 + rng.randint(20, 60)
    c1 = c0 + rng.randint(20, 60)
    masks[i, r0:r1, c0:c1] = 1
amaps = rng.rand(N, H, W).astype(np.float32)
print(f"N={N}, H={H}, W={W}", flush=True)
t0 = time.time()
pro = compute_pro(masks, amaps, num_th=200)
elapsed = time.time() - t0
print(f"elapsed={elapsed:.1f}s  pro={pro:.4f}", flush=True)
print(f"Extrapolated N=3755: {elapsed*3755/N:.0f}s  ({elapsed*3755/N/60:.1f} min)", flush=True)
