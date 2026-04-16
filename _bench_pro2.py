import time, numpy as np, sys
sys.path.insert(0, '/root/autodl-tmp/Dinomaly')
from utils import compute_pro

rng = np.random.RandomState(0)
# 用小N只测阈值循环开销
for N in [50, 100]:
    masks = np.zeros((N, 392, 392), dtype=np.uint8)
    for i in range(int(N*0.6)):
        r0,c0=rng.randint(50,300,2); masks[i,r0:r0+40,c0:c0+40]=1
    amaps = rng.rand(N, 392, 392).astype(np.float32)
    t0=time.time()
    pro=compute_pro(masks, amaps, num_th=200)
    e=time.time()-t0
    print(f"N={N:4d}  elapsed={e:6.1f}s  => N=3755 est={(e*3755/N/60):.1f}min", flush=True)
