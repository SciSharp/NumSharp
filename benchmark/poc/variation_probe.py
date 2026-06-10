# NumPy side of variation_probe.cs — identical shapes/ops, end-to-end
# (both sides allocate results; no out=).
import numpy as np, time

ROUNDS = 7
def t_ms(f, iters, warm):
    for _ in range(warm): f()
    t0 = time.perf_counter()
    for _ in range(iters): f()
    return (time.perf_counter() - t0) * 1000.0 / iters
def med(f):
    r = sorted(f() for _ in range(ROUNDS))
    return r[ROUNDS // 2]
def P(name, ms): print(f"{name:<46} {ms:9.3f} ms")

N = 4_000_000
side = 2000

aC = np.arange(N).astype(np.float32) + 1
bC = np.arange(N).astype(np.float32) + 2
a2 = (np.arange(side*side).astype(np.float32) + 1).reshape(side, side)
b2 = (np.arange(side*side).astype(np.float32) + 2).reshape(side, side)
row = (np.arange(side).astype(np.float32) + 3).reshape(1, side)
col = (np.arange(side).astype(np.float32) + 3).reshape(side, 1)
aF = np.asfortranarray(a2); bF = np.asfortranarray(b2)
i32 = np.arange(N).astype(np.int32)
f64 = np.arange(N).astype(np.float64) + 1.0
wide = np.arange(2*N).astype(np.float32) + 1
sa = wide[::2]
cond = (np.arange(N) % 3).astype(bool)
small1 = np.arange(1000).astype(np.float32) + 1
small2 = np.arange(1000).astype(np.float32) + 2
a5d = (np.arange(N).astype(np.float32) + 1).reshape(10,10,10,10,400)

# overlap reference
ov = np.arange(8).astype(np.float64) + 1
np.add(ov[:-1], ov[:-1], out=ov[1:])
print("numpy overlap reference:", ov)

print()
print(f"{'probe':<46} NumPy")
print('-'*60)
P("P1  contig binary  a+b f32 4M",         med(lambda: t_ms(lambda: aC + bC, 30, 10)))
P("P2  row broadcast  (2k,2k)+(1,2k)",     med(lambda: t_ms(lambda: a2 + row, 30, 10)))
P("P3  col broadcast  (2k,2k)+(2k,1)",     med(lambda: t_ms(lambda: a2 + col, 30, 10)))
P("P4  scalar broadcast a+5",              med(lambda: t_ms(lambda: aC + np.float32(5), 30, 10)))
P("P5  neg-stride unary sqrt(a[::-1])",    med(lambda: t_ms(lambda: np.sqrt(aC[::-1]), 30, 10)))
P("P6  neg-stride binary a[::-1]+b[::-1]", med(lambda: t_ms(lambda: aC[::-1] + bC[::-1], 30, 10)))
P("P7  F-order binary aF+bF",              med(lambda: t_ms(lambda: aF + bF, 30, 10)))
P("P8  transposed binary a.T+b.T",         med(lambda: t_ms(lambda: a2.T + b2.T, 30, 10)))
P("P9  mixed dtype i32+f64 4M",            med(lambda: t_ms(lambda: i32 + f64, 30, 10)))
P("P10 astype strided a[::2]->f64",        med(lambda: t_ms(lambda: sa.astype(np.float64), 30, 10)))
P("P11 where(cond,x,y) contig 4M",         med(lambda: t_ms(lambda: np.where(cond, aC, bC), 30, 10)))
P("P12 sum axis=0 f32 (2k,2k)",            med(lambda: t_ms(lambda: a2.sum(axis=0), 30, 10)))
P("P13 sum axis=1 f32 (2k,2k)",            med(lambda: t_ms(lambda: a2.sum(axis=1), 30, 10)))
P("P14 5-D contig unary sqrt",             med(lambda: t_ms(lambda: np.sqrt(a5d), 30, 10)))
us = med(lambda: t_ms(lambda: small1 + small2, 20000, 2000)) * 1000.0
print(f"{'P15 small-N binary 1K (us/call)':<46} {us:9.3f} us")
P("P16 mean f32 contig 4M",                med(lambda: t_ms(lambda: np.mean(aC), 30, 10)))

n1 = 1_000_000
w1 = np.arange(2*n1).astype(np.float32) + 1
w2 = np.arange(2*n1).astype(np.float32) + 2
ss1 = w1[::2]; ss2 = w2[::2]
big2 = (np.arange(4*n1).astype(np.float32) + 1).reshape(2000, 2000)
sv2d = big2[::2, ::2]
print(f"{'S1  strided binary a[::2]+b[::2] f32 1M':<46} {med(lambda: t_ms(lambda: ss1 + ss2, 100, 30))*1000:9.0f} us")
print(f"{'S2  strided 2-D sqrt(a[::2,::2]) f32 1M':<46} {med(lambda: t_ms(lambda: np.sqrt(sv2d), 100, 30))*1000:9.0f} us")
print(f"{'S3  strided sum(a[::2]) f32 1M':<46} {med(lambda: t_ms(lambda: np.sum(ss1), 100, 30))*1000:9.0f} us")
print("[done]")
