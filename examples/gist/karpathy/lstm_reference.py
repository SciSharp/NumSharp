"""Independent NumPy oracle for Karpathy's IFOG batched LSTM equations.

No framework, downloads, interactive input or top-level training. Source provenance
is in sources.json beside this file. The array operation order follows the source.
"""
import numpy as np


def forward(x, w, c0=None, h0=None):
    n, b, inputs = x.shape
    d = w.shape[1] // 4
    if c0 is None: c0 = np.zeros((b, d))
    if h0 is None: h0 = np.zeros((b, d))
    hin = np.zeros((n, b, w.shape[0]))
    h = np.zeros((n, b, d))
    gates = np.zeros((n, b, 4*d))
    activated = np.zeros_like(gates)
    c = np.zeros_like(h)
    ct = np.zeros_like(h)
    for t in range(n):
        hin[t, :, 0] = 1
        hin[t, :, 1:inputs+1] = x[t]
        hin[t, :, inputs+1:] = h[t-1] if t else h0
        gates[t] = hin[t].dot(w)
        activated[t, :, :3*d] = 1.0 / (1.0 + np.exp(-gates[t, :, :3*d]))
        activated[t, :, 3*d:] = np.tanh(gates[t, :, 3*d:])
        prev = c[t-1] if t else c0
        c[t] = activated[t, :, :d] * activated[t, :, 3*d:] + activated[t, :, d:2*d] * prev
        ct[t] = np.tanh(c[t])
        h[t] = activated[t, :, 2*d:3*d] * ct[t]
    return dict(w=w, hin=hin, h=h, gates=gates, activated=activated, c=c, ct=ct, c0=c0, h0=h0)


def backward(dh_input, cache, dcn=None, dhn=None):
    w, hin, h, a, c, ct, c0 = (cache[k] for k in ('w','hin','h','activated','c','ct','c0'))
    n, b, d = h.shape
    inputs = w.shape[0] - d - 1
    da = np.zeros_like(a)
    dg = np.zeros_like(a)
    dw = np.zeros_like(w)
    di = np.zeros_like(hin)
    dc = np.zeros_like(c)
    dx = np.zeros((n, b, inputs))
    dc0 = np.zeros((b, d))
    dh0 = np.zeros((b, d))
    dh = dh_input.copy()
    if dcn is not None: dc[-1] += dcn.copy()
    if dhn is not None: dh[-1] += dhn.copy()
    for t in reversed(range(n)):
        da[t, :, 2*d:3*d] = ct[t] * dh[t]
        dc[t] += (1 - ct[t]**2) * (a[t, :, 2*d:3*d] * dh[t])
        da[t, :, d:2*d] = (c[t-1] if t else c0) * dc[t]
        if t: dc[t-1] += a[t, :, d:2*d] * dc[t]
        else: dc0 = a[t, :, d:2*d] * dc[t]
        da[t, :, :d] = a[t, :, 3*d:] * dc[t]
        da[t, :, 3*d:] = a[t, :, :d] * dc[t]
        dg[t, :, 3*d:] = (1 - a[t, :, 3*d:]**2) * da[t, :, 3*d:]
        y = a[t, :, :3*d]
        dg[t, :, :3*d] = (y * (1.0 - y)) * da[t, :, :3*d]
        dw += np.dot(hin[t].T, dg[t])
        di[t] = dg[t].dot(w.T)
        dx[t] = di[t, :, 1:inputs+1]
        if t: dh[t-1] += di[t, :, inputs+1:]
        else: dh0 += di[t, :, inputs+1:]
    return dict(dx=dx, dw=dw, dc0=dc0, dh0=dh0)


if __name__ == '__main__':
    rng = np.random.RandomState(7)
    w = rng.randn(15, 16) / np.sqrt(14)
    w[0] = 0
    w[0, 4:8] = 3
    x = rng.randn(5, 3, 10)
    h0 = rng.randn(3, 4)
    c0 = rng.randn(3, 4)
    cache = forward(x, w, c0, h0)
    dh = rng.randn(5, 3, 4)
    grad = backward(dh, cache)
    maximum = 0.0
    count = 0
    for p, g in [(x,grad['dx']), (w,grad['dw']), (c0,grad['dc0']), (h0,grad['dh0'])]:
        for i in range(p.size):
            saved = p.flat[i]
            p.flat[i] = saved + 1e-5
            plus = np.sum(forward(x,w,c0,h0)['h'] * dh)
            p.flat[i] = saved - 1e-5
            minus = np.sum(forward(x,w,c0,h0)['h'] * dh)
            p.flat[i] = saved
            numeric = (plus-minus)/2e-5
            if max(abs(numeric),abs(g.flat[i])) >= 1e-7:
                maximum = max(maximum, abs(numeric-g.flat[i])/abs(numeric+g.flat[i]))
            count += 1
    print(dict(checked=count, maximum_relative_error=maximum,
               loss=float(np.sum(cache['h']*dh)), first_hidden=float(cache['h'].flat[0]),
               first_dx=float(grad['dx'].flat[0])))
