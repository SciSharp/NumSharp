# Operation Matrix — NumSharp vs NumPy (full report)

> _Auto-generated after each release by the [Benchmark workflow](https://github.com/SciSharp/NumSharp/blob/master/.github/workflows/benchmark.yml) — do not edit by hand. Discussion + cards: [Benchmarks vs NumPy](benchmarks.md)._


**Baseline:** NumPy · measured across all array sizes (per-(op, dtype, N))

**Ratio** = NumSharp ÷ NumPy → Lower is better for NumSharp

| | Status | Ratio | Meaning |
|:-:|--------|:-----:|---------|
|✅| Faster | <1.0 | NumSharp beats NumPy |
|🟡| Close | 1-2x | Acceptable parity |
|🟠| Slower | 2-5x | Optimization target |
|🔴| Slow | >5x | Priority fix |
|▫| Negligible | <1µs / >20x | Too fast to compare — excluded from rankings |
|⚪| Pending | - | C# benchmark not run |

---

**Summary:** 1233 ops | ✅ 305 | 🟡 255 | 🟠 169 | 🔴 103 | ▫ 275 | ⚪ 126

## Summary by size

| N | ops | ✅ faster | 🟡 close | 🟠 slower | 🔴 much | ▫ negl | ⚪ n/a | geomean |
|---:|----:|--------:|--------:|---------:|------:|-----:|-----:|--------:|
| 500 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | - |
| 900 | 3 | 0 | 0 | 0 | 0 | 0 | 3 | - |
| 1,000 | 409 | 41 | 19 | 28 | 21 | 258 | 42 | 1.37x |
| 50,000 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | - |
| 100,000 | 409 | 104 | 65 | 121 | 69 | 10 | 40 | 1.82x |
| 5,000,000 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | - |
| 10,000,000 | 409 | 160 | 171 | 20 | 13 | 7 | 38 | 1.02x |

---

### 🏆 Top 15 Best (NumSharp closest to / beating NumPy)

_Ranked over 832 credible comparisons (both sides ≥1µs, speedup ≤20×); 275 negligible rows excluded as non-comparable (▫). Ratio = NumSharp ÷ NumPy — below 1.0 = NumSharp faster._

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|✅| np.nansum(a) (float64) | float64 | 100,000 | 0.242 | 0.019 | 0.08x |
|✅| np.percentile(a, 50) (float64) | float64 | 1,000 | 0.025 | 0.002 | 0.10x |
|✅| np.percentile(a, 50) (float32) | float32 | 1,000 | 0.025 | 0.002 | 0.10x |
|✅| np.average(a) (float32) | float32 | 10,000,000 | 9.598 | 0.937 | 0.10x |
|✅| np.quantile(a, 0.5) (float32) | float32 | 1,000 | 0.024 | 0.002 | 0.10x |
|✅| np.quantile(a, 0.5) (float64) | float64 | 1,000 | 0.023 | 0.002 | 0.10x |
|✅| np.nanprod(a) (float32) | float32 | 10,000,000 | 18.515 | 1.904 | 0.10x |
|✅| np.nansum(a) (float32) | float32 | 10,000,000 | 14.349 | 1.488 | 0.10x |
|✅| np.nanprod(a) (float64) | float64 | 100,000 | 0.287 | 0.032 | 0.11x |
|✅| np.average(a) (float32) | float32 | 100,000 | 0.018 | 0.002 | 0.12x |
|✅| np.count_nonzero(a) (float32) | float32 | 100,000 | 0.038 | 0.005 | 0.12x |
|✅| np.nanstd(a) (float32) | float32 | 1,000 | 0.020 | 0.003 | 0.12x |
|✅| np.std (float32) | float32 | 1,000 | 0.008 | 0.001 | 0.13x |
|✅| np.nanvar(a) (float32) | float32 | 1,000 | 0.019 | 0.003 | 0.13x |
|✅| np.nanquantile(a, 0.5) (float64) | float64 | 1,000 | 0.028 | 0.004 | 0.13x |

### 🔻 Top 15 Worst (Optimization priorities)

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🔴| np.zeros (int64) | int64 | 10,000,000 | 0.012 | 10.747 | 879.57x |
|🔴| np.zeros (int32) | int32 | 10,000,000 | 0.011 | 5.622 | 518.20x |
|🔴| np.zeros (float64) | float64 | 10,000,000 | 0.021 | 10.755 | 507.65x |
|🔴| np.zeros (float32) | float32 | 10,000,000 | 0.017 | 5.673 | 334.03x |
|🔴| np.argsort(a) (int64) | int64 | 100,000 | 0.472 | 12.893 | 27.34x |
|🔴| np.argsort(a) (int32) | int32 | 100,000 | 0.442 | 10.404 | 23.54x |
|🔴| a * 2 (literal) (float32) | float32 | 100,000 | 0.007 | 0.129 | 19.37x |
|🔴| np.right_shift(a, 2) (int64) | int64 | 1,000 | 0.001 | 0.019 | 19.20x |
|🔴| np.left_shift(a, 2) (int64) | int64 | 1,000 | 0.001 | 0.020 | 19.11x |
|🔴| np.sum axis=1 (uint8) | uint8 | 10,000,000 | 3.115 | 49.741 | 15.97x |
|🔴| np.right_shift(a, 2) (int32) | int32 | 1,000 | 0.001 | 0.017 | 15.61x |
|🔴| np.sum axis=0 (uint16) | uint16 | 10,000,000 | 4.620 | 71.694 | 15.52x |
|🔴| np.sum axis=1 (uint16) | uint16 | 10,000,000 | 3.365 | 49.896 | 14.83x |
|🔴| a + 5 (literal) (float32) | float32 | 100,000 | 0.007 | 0.097 | 14.66x |
|🔴| np.sum axis=1 (uint8) | uint8 | 100,000 | 0.037 | 0.500 | 13.45x |

---

### Arithmetic

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟡| a % 7 (literal) (float32) | float32 | 1,000 | 0.0141 | 0.0190 | 1.34x |
|🟡| a % 7 (literal) (float32) | float32 | 100,000 | 1.6551 | 1.9677 | 1.19x |
|🟡| a % 7 (literal) (float32) | float32 | 10,000,000 | 167.3112 | 194.6954 | 1.16x |
|🟠| a % 7 (literal) (float64) | float64 | 1,000 | 0.0114 | 0.0238 | 2.09x |
|🟡| a % 7 (literal) (float64) | float64 | 100,000 | 1.4792 | 1.7960 | 1.21x |
|🟡| a % 7 (literal) (float64) | float64 | 10,000,000 | 155.5835 | 178.7748 | 1.15x |
|🟡| a % 7 (literal) (int32) | int32 | 1,000 | 0.0020 | 0.0039 | 1.92x |
|🟡| a % 7 (literal) (int32) | int32 | 100,000 | 0.4121 | 0.6918 | 1.68x |
|🟡| a % 7 (literal) (int32) | int32 | 10,000,000 | 45.8307 | 70.8023 | 1.54x |
|🟡| a % 7 (literal) (int64) | int64 | 1,000 | 0.0042 | 0.0057 | 1.35x |
|🟠| a % 7 (literal) (int64) | int64 | 100,000 | 0.4216 | 0.9121 | 2.16x |
|🟡| a % 7 (literal) (int64) | int64 | 10,000,000 | 51.6809 | 93.8372 | 1.82x |
|✅| a % b (element-wise) (float32) | float32 | 1,000 | 0.0126 | 0.0123 | 0.98x |
|🟡| a % b (element-wise) (float32) | float32 | 100,000 | 1.5069 | 1.6644 | 1.10x |
|🟡| a % b (element-wise) (float32) | float32 | 10,000,000 | 156.3306 | 166.9308 | 1.07x |
|✅| a % b (element-wise) (float64) | float64 | 1,000 | 0.0099 | 0.0098 | 0.99x |
|🟡| a % b (element-wise) (float64) | float64 | 100,000 | 1.3230 | 1.4718 | 1.11x |
|🟡| a % b (element-wise) (float64) | float64 | 10,000,000 | 143.2559 | 151.6138 | 1.06x |
|🟡| a % b (element-wise) (int32) | int32 | 1,000 | 0.0020 | 0.0038 | 1.89x |
|🟡| a % b (element-wise) (int32) | int32 | 100,000 | 0.3760 | 0.6165 | 1.64x |
|🟡| a % b (element-wise) (int32) | int32 | 10,000,000 | 43.2277 | 64.9449 | 1.50x |
|🟡| a % b (element-wise) (int64) | int64 | 1,000 | 0.0037 | 0.0039 | 1.05x |
|🟡| a % b (element-wise) (int64) | int64 | 100,000 | 0.4161 | 0.6301 | 1.51x |
|🟡| a % b (element-wise) (int64) | int64 | 10,000,000 | 48.6074 | 67.4008 | 1.39x |
|▫| a * 2 (literal) (float32) | float32 | 1,000 | 0.0008 | 0.0063 | 8.21x |
|🔴| a * 2 (literal) (float32) | float32 | 100,000 | 0.0067 | 0.1288 | 19.37x |
|🟡| a * 2 (literal) (float32) | float32 | 10,000,000 | 8.3163 | 12.2127 | 1.47x |
|▫| a * 2 (literal) (float64) | float64 | 1,000 | 0.0008 | 0.0071 | 9.02x |
|🔴| a * 2 (literal) (float64) | float64 | 100,000 | 0.0133 | 0.1505 | 11.34x |
|🟡| a * 2 (literal) (float64) | float64 | 10,000,000 | 17.3763 | 21.6842 | 1.25x |
|🔴| a * 2 (literal) (int16) | int16 | 1,000 | 0.0010 | 0.0055 | 5.38x |
|🟠| a * 2 (literal) (int16) | int16 | 100,000 | 0.0230 | 0.0935 | 4.07x |
|🟠| a * 2 (literal) (int16) | int16 | 10,000,000 | 4.6629 | 9.7334 | 2.09x |
|🟠| a * 2 (literal) (int32) | int32 | 1,000 | 0.0010 | 0.0038 | 3.81x |
|🟠| a * 2 (literal) (int32) | int32 | 100,000 | 0.0228 | 0.0550 | 2.42x |
|🟡| a * 2 (literal) (int32) | int32 | 10,000,000 | 8.7622 | 10.1737 | 1.16x |
|▫| a * 2 (literal) (int64) | int64 | 1,000 | 0.0009 | 0.0069 | 7.66x |
|🔴| a * 2 (literal) (int64) | int64 | 100,000 | 0.0223 | 0.1206 | 5.41x |
|🟡| a * 2 (literal) (int64) | int64 | 10,000,000 | 18.5884 | 22.7628 | 1.23x |
|🔴| a * 2 (literal) (uint16) | uint16 | 1,000 | 0.0010 | 0.0065 | 6.43x |
|🔴| a * 2 (literal) (uint16) | uint16 | 100,000 | 0.0225 | 0.1190 | 5.29x |
|🟠| a * 2 (literal) (uint16) | uint16 | 10,000,000 | 4.4333 | 9.0584 | 2.04x |
|🔴| a * 2 (literal) (uint32) | uint32 | 1,000 | 0.0010 | 0.0060 | 5.97x |
|🟠| a * 2 (literal) (uint32) | uint32 | 100,000 | 0.0246 | 0.1099 | 4.47x |
|🟡| a * 2 (literal) (uint32) | uint32 | 10,000,000 | 8.5106 | 12.0673 | 1.42x |
|▫| a * 2 (literal) (uint64) | uint64 | 1,000 | 0.0009 | 0.0074 | 8.17x |
|🔴| a * 2 (literal) (uint64) | uint64 | 100,000 | 0.0234 | 0.1581 | 6.77x |
|🟡| a * 2 (literal) (uint64) | uint64 | 10,000,000 | 15.8401 | 22.1489 | 1.40x |
|▫| a * 2 (literal) (uint8) | uint8 | 1,000 | 0.0009 | 0.0043 | 4.99x |
|🟠| a * 2 (literal) (uint8) | uint8 | 100,000 | 0.0236 | 0.1037 | 4.40x |
|🟠| a * 2 (literal) (uint8) | uint8 | 10,000,000 | 3.6469 | 8.4932 | 2.33x |
|▫| a * a (square) (float32) | float32 | 1,000 | 0.0005 | 0.0018 | 3.41x |
|🔴| a * a (square) (float32) | float32 | 100,000 | 0.0079 | 0.0543 | 6.90x |
|🟡| a * a (square) (float32) | float32 | 10,000,000 | 8.0929 | 11.1120 | 1.37x |
|▫| a * a (square) (float64) | float64 | 1,000 | 0.0005 | 0.0026 | 5.33x |
|🔴| a * a (square) (float64) | float64 | 100,000 | 0.0161 | 0.1043 | 6.48x |
|🟡| a * a (square) (float64) | float64 | 10,000,000 | 16.9687 | 20.3622 | 1.20x |
|▫| a * a (square) (int16) | int16 | 1,000 | 0.0008 | 0.0016 | 2.11x |
|✅| a * a (square) (int16) | int16 | 100,000 | 0.0301 | 0.0280 | 0.93x |
|🟡| a * a (square) (int16) | int16 | 10,000,000 | 5.0053 | 5.5910 | 1.12x |
|▫| a * a (square) (int32) | int32 | 1,000 | 0.0008 | 0.0017 | 2.18x |
|🟠| a * a (square) (int32) | int32 | 100,000 | 0.0286 | 0.0579 | 2.02x |
|🟡| a * a (square) (int32) | int32 | 10,000,000 | 8.7437 | 10.0335 | 1.15x |
|▫| a * a (square) (int64) | int64 | 1,000 | 0.0007 | 0.0029 | 3.93x |
|🟠| a * a (square) (int64) | int64 | 100,000 | 0.0292 | 0.1098 | 3.77x |
|🟡| a * a (square) (int64) | int64 | 10,000,000 | 17.1429 | 21.0324 | 1.23x |
|▫| a * a (square) (uint16) | uint16 | 1,000 | 0.0008 | 0.0017 | 2.10x |
|✅| a * a (square) (uint16) | uint16 | 100,000 | 0.0282 | 0.0274 | 0.97x |
|🟡| a * a (square) (uint16) | uint16 | 10,000,000 | 4.9792 | 5.4124 | 1.09x |
|▫| a * a (square) (uint32) | uint32 | 1,000 | 0.0008 | 0.0018 | 2.28x |
|🟡| a * a (square) (uint32) | uint32 | 100,000 | 0.0305 | 0.0550 | 1.80x |
|🟡| a * a (square) (uint32) | uint32 | 10,000,000 | 8.4000 | 10.8109 | 1.29x |
|▫| a * a (square) (uint64) | uint64 | 1,000 | 0.0007 | 0.0028 | 3.94x |
|🟠| a * a (square) (uint64) | uint64 | 100,000 | 0.0295 | 0.1148 | 3.89x |
|🟡| a * a (square) (uint64) | uint64 | 10,000,000 | 16.3411 | 21.5055 | 1.32x |
|▫| a * a (square) (uint8) | uint8 | 1,000 | 0.0006 | 0.0014 | 2.24x |
|✅| a * a (square) (uint8) | uint8 | 100,000 | 0.0278 | 0.0159 | 0.57x |
|✅| a * a (square) (uint8) | uint8 | 10,000,000 | 3.8696 | 2.2714 | 0.59x |
|▫| a * b (element-wise) (float32) | float32 | 1,000 | 0.0005 | 0.0019 | 3.52x |
|🔴| a * b (element-wise) (float32) | float32 | 100,000 | 0.0071 | 0.0544 | 7.67x |
|🟡| a * b (element-wise) (float32) | float32 | 10,000,000 | 8.9544 | 14.0772 | 1.57x |
|▫| a * b (element-wise) (float64) | float64 | 1,000 | 0.0005 | 0.0030 | 5.81x |
|🟠| a * b (element-wise) (float64) | float64 | 100,000 | 0.0311 | 0.1140 | 3.66x |
|🟡| a * b (element-wise) (float64) | float64 | 10,000,000 | 17.6849 | 26.5088 | 1.50x |
|▫| a * b (element-wise) (int16) | int16 | 1,000 | 0.0008 | 0.0019 | 2.43x |
|✅| a * b (element-wise) (int16) | int16 | 100,000 | 0.0284 | 0.0281 | 0.99x |
|🟡| a * b (element-wise) (int16) | int16 | 10,000,000 | 5.1147 | 7.0273 | 1.37x |
|▫| a * b (element-wise) (int32) | int32 | 1,000 | 0.0008 | 0.0018 | 2.18x |
|🟡| a * b (element-wise) (int32) | int32 | 100,000 | 0.0303 | 0.0577 | 1.91x |
|🟡| a * b (element-wise) (int32) | int32 | 10,000,000 | 10.0582 | 14.3013 | 1.42x |
|▫| a * b (element-wise) (int64) | int64 | 1,000 | 0.0008 | 0.0029 | 3.60x |
|🟠| a * b (element-wise) (int64) | int64 | 100,000 | 0.0326 | 0.1179 | 3.62x |
|🟡| a * b (element-wise) (int64) | int64 | 10,000,000 | 18.7233 | 28.7622 | 1.54x |
|✅| a * b (element-wise) (uint16) | uint16 | 1,000 | 0.0019 | 0.0017 | 0.90x |
|🟡| a * b (element-wise) (uint16) | uint16 | 100,000 | 0.0279 | 0.0289 | 1.04x |
|🟡| a * b (element-wise) (uint16) | uint16 | 10,000,000 | 5.3957 | 6.9261 | 1.28x |
|▫| a * b (element-wise) (uint32) | uint32 | 1,000 | 0.0008 | 0.0022 | 2.75x |
|🟡| a * b (element-wise) (uint32) | uint32 | 100,000 | 0.0297 | 0.0550 | 1.85x |
|🟡| a * b (element-wise) (uint32) | uint32 | 10,000,000 | 8.8650 | 13.5566 | 1.53x |
|▫| a * b (element-wise) (uint64) | uint64 | 1,000 | 0.0007 | 0.0028 | 3.98x |
|🟠| a * b (element-wise) (uint64) | uint64 | 100,000 | 0.0312 | 0.1133 | 3.63x |
|🟡| a * b (element-wise) (uint64) | uint64 | 10,000,000 | 19.0202 | 31.9026 | 1.68x |
|▫| a * b (element-wise) (uint8) | uint8 | 1,000 | 0.0007 | 0.0017 | 2.63x |
|✅| a * b (element-wise) (uint8) | uint8 | 100,000 | 0.0277 | 0.0153 | 0.55x |
|✅| a * b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0144 | 3.3705 | 0.84x |
|▫| a * scalar (float32) | float32 | 1,000 | 0.0007 | 0.0019 | 2.60x |
|🔴| a * scalar (float32) | float32 | 100,000 | 0.0062 | 0.0574 | 9.26x |
|🟡| a * scalar (float32) | float32 | 10,000,000 | 8.0655 | 10.2659 | 1.27x |
|▫| a * scalar (float64) | float64 | 1,000 | 0.0007 | 0.0017 | 2.51x |
|🔴| a * scalar (float64) | float64 | 100,000 | 0.0150 | 0.1151 | 7.69x |
|🟡| a * scalar (float64) | float64 | 10,000,000 | 18.5637 | 20.1307 | 1.08x |
|▫| a * scalar (int16) | int16 | 1,000 | 0.0009 | 0.0020 | 2.23x |
|🟡| a * scalar (int16) | int16 | 100,000 | 0.0238 | 0.0273 | 1.15x |
|🟡| a * scalar (int16) | int16 | 10,000,000 | 4.6196 | 5.4073 | 1.17x |
|▫| a * scalar (int32) | int32 | 1,000 | 0.0009 | 0.0019 | 2.14x |
|🟠| a * scalar (int32) | int32 | 100,000 | 0.0232 | 0.0546 | 2.35x |
|🟡| a * scalar (int32) | int32 | 10,000,000 | 8.3511 | 10.1460 | 1.22x |
|▫| a * scalar (int64) | int64 | 1,000 | 0.0008 | 0.0028 | 3.37x |
|🟠| a * scalar (int64) | int64 | 100,000 | 0.0252 | 0.1085 | 4.30x |
|🟡| a * scalar (int64) | int64 | 10,000,000 | 19.4204 | 21.9689 | 1.13x |
|▫| a * scalar (uint16) | uint16 | 1,000 | 0.0009 | 0.0020 | 2.17x |
|🟡| a * scalar (uint16) | uint16 | 100,000 | 0.0227 | 0.0285 | 1.25x |
|🟡| a * scalar (uint16) | uint16 | 10,000,000 | 4.4608 | 5.3498 | 1.20x |
|▫| a * scalar (uint32) | uint32 | 1,000 | 0.0009 | 0.0021 | 2.46x |
|🟠| a * scalar (uint32) | uint32 | 100,000 | 0.0234 | 0.0526 | 2.25x |
|🟡| a * scalar (uint32) | uint32 | 10,000,000 | 8.1414 | 10.3010 | 1.26x |
|▫| a * scalar (uint64) | uint64 | 1,000 | 0.0008 | 0.0027 | 3.49x |
|🟠| a * scalar (uint64) | uint64 | 100,000 | 0.0227 | 0.1114 | 4.92x |
|🟡| a * scalar (uint64) | uint64 | 10,000,000 | 16.9767 | 21.1373 | 1.25x |
|▫| a * scalar (uint8) | uint8 | 1,000 | 0.0008 | 0.0018 | 2.44x |
|✅| a * scalar (uint8) | uint8 | 100,000 | 0.0242 | 0.0157 | 0.65x |
|✅| a * scalar (uint8) | uint8 | 10,000,000 | 3.7185 | 2.4709 | 0.66x |
|▫| a + 5 (literal) (float32) | float32 | 1,000 | 0.0008 | 0.0059 | 7.01x |
|🔴| a + 5 (literal) (float32) | float32 | 100,000 | 0.0066 | 0.0968 | 14.66x |
|🟡| a + 5 (literal) (float32) | float32 | 10,000,000 | 8.6016 | 12.9626 | 1.51x |
|▫| a + 5 (literal) (float64) | float64 | 1,000 | 0.0008 | 0.0071 | 9.16x |
|🔴| a + 5 (literal) (float64) | float64 | 100,000 | 0.0137 | 0.1214 | 8.87x |
|🟡| a + 5 (literal) (float64) | float64 | 10,000,000 | 16.3098 | 21.8587 | 1.34x |
|🔴| a + 5 (literal) (int16) | int16 | 1,000 | 0.0011 | 0.0065 | 6.10x |
|🟠| a + 5 (literal) (int16) | int16 | 100,000 | 0.0247 | 0.0883 | 3.58x |
|🟡| a + 5 (literal) (int16) | int16 | 10,000,000 | 7.7897 | 9.6845 | 1.24x |
|🟠| a + 5 (literal) (int32) | int32 | 1,000 | 0.0010 | 0.0032 | 3.19x |
|🟠| a + 5 (literal) (int32) | int32 | 100,000 | 0.0243 | 0.0530 | 2.18x |
|🟡| a + 5 (literal) (int32) | int32 | 10,000,000 | 9.4343 | 10.1109 | 1.07x |
|▫| a + 5 (literal) (int64) | int64 | 1,000 | 0.0010 | 0.0041 | 4.08x |
|🔴| a + 5 (literal) (int64) | int64 | 100,000 | 0.0238 | 0.1342 | 5.63x |
|🟡| a + 5 (literal) (int64) | int64 | 10,000,000 | 16.0346 | 22.1195 | 1.38x |
|🔴| a + 5 (literal) (uint16) | uint16 | 1,000 | 0.0010 | 0.0066 | 6.42x |
|🟠| a + 5 (literal) (uint16) | uint16 | 100,000 | 0.0260 | 0.0895 | 3.44x |
|🟡| a + 5 (literal) (uint16) | uint16 | 10,000,000 | 4.8116 | 9.2841 | 1.93x |
|🔴| a + 5 (literal) (uint32) | uint32 | 1,000 | 0.0010 | 0.0064 | 6.27x |
|🟠| a + 5 (literal) (uint32) | uint32 | 100,000 | 0.0249 | 0.0954 | 3.83x |
|🟡| a + 5 (literal) (uint32) | uint32 | 10,000,000 | 8.2202 | 12.3833 | 1.51x |
|🔴| a + 5 (literal) (uint64) | uint64 | 1,000 | 0.0010 | 0.0075 | 7.45x |
|🟠| a + 5 (literal) (uint64) | uint64 | 100,000 | 0.0262 | 0.1244 | 4.75x |
|🟡| a + 5 (literal) (uint64) | uint64 | 10,000,000 | 15.7466 | 22.7518 | 1.45x |
|▫| a + 5 (literal) (uint8) | uint8 | 1,000 | 0.0009 | 0.0049 | 5.58x |
|🟠| a + 5 (literal) (uint8) | uint8 | 100,000 | 0.0256 | 0.0886 | 3.46x |
|🟠| a + 5 (literal) (uint8) | uint8 | 10,000,000 | 3.4863 | 8.8818 | 2.55x |
|▫| a + b (element-wise) (float32) | float32 | 1,000 | 0.0006 | 0.0021 | 3.69x |
|🔴| a + b (element-wise) (float32) | float32 | 100,000 | 0.0070 | 0.0504 | 7.25x |
|🟡| a + b (element-wise) (float32) | float32 | 10,000,000 | 9.5103 | 13.8917 | 1.46x |
|▫| a + b (element-wise) (float64) | float64 | 1,000 | 0.0007 | 0.0032 | 4.68x |
|🟠| a + b (element-wise) (float64) | float64 | 100,000 | 0.0301 | 0.1169 | 3.88x |
|🟡| a + b (element-wise) (float64) | float64 | 10,000,000 | 18.9762 | 26.6012 | 1.40x |
|▫| a + b (element-wise) (int16) | int16 | 1,000 | 0.0008 | 0.0022 | 2.81x |
|✅| a + b (element-wise) (int16) | int16 | 100,000 | 0.0303 | 0.0294 | 0.97x |
|🟡| a + b (element-wise) (int16) | int16 | 10,000,000 | 6.6919 | 7.2019 | 1.08x |
|🟡| a + b (element-wise) (int32) | int32 | 1,000 | 0.0010 | 0.0010 | 1.02x |
|🟡| a + b (element-wise) (int32) | int32 | 100,000 | 0.0295 | 0.0575 | 1.95x |
|🟡| a + b (element-wise) (int32) | int32 | 10,000,000 | 9.0923 | 13.8151 | 1.52x |
|▫| a + b (element-wise) (int64) | int64 | 1,000 | 0.0007 | 0.0030 | 4.08x |
|🟠| a + b (element-wise) (int64) | int64 | 100,000 | 0.0337 | 0.1194 | 3.55x |
|🟡| a + b (element-wise) (int64) | int64 | 10,000,000 | 19.8215 | 26.2861 | 1.33x |
|▫| a + b (element-wise) (uint16) | uint16 | 1,000 | 0.0008 | 0.0021 | 2.62x |
|✅| a + b (element-wise) (uint16) | uint16 | 100,000 | 0.0299 | 0.0287 | 0.96x |
|🟡| a + b (element-wise) (uint16) | uint16 | 10,000,000 | 5.3257 | 7.1654 | 1.34x |
|▫| a + b (element-wise) (uint32) | uint32 | 1,000 | 0.0008 | 0.0020 | 2.53x |
|🟡| a + b (element-wise) (uint32) | uint32 | 100,000 | 0.0323 | 0.0523 | 1.62x |
|🟡| a + b (element-wise) (uint32) | uint32 | 10,000,000 | 9.0103 | 14.3468 | 1.59x |
|▫| a + b (element-wise) (uint64) | uint64 | 1,000 | 0.0008 | 0.0034 | 4.49x |
|🟠| a + b (element-wise) (uint64) | uint64 | 100,000 | 0.0350 | 0.1053 | 3.01x |
|🟡| a + b (element-wise) (uint64) | uint64 | 10,000,000 | 18.6852 | 26.5867 | 1.42x |
|▫| a + b (element-wise) (uint8) | uint8 | 1,000 | 0.0007 | 0.0015 | 2.11x |
|✅| a + b (element-wise) (uint8) | uint8 | 100,000 | 0.0288 | 0.0181 | 0.63x |
|✅| a + b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0514 | 3.6027 | 0.89x |
|▫| a + scalar (float32) | float32 | 1,000 | 0.0008 | 0.0024 | 3.09x |
|🔴| a + scalar (float32) | float32 | 100,000 | 0.0064 | 0.0543 | 8.51x |
|🟡| a + scalar (float32) | float32 | 10,000,000 | 8.1738 | 10.5122 | 1.29x |
|▫| a + scalar (float64) | float64 | 1,000 | 0.0006 | 0.0031 | 4.89x |
|🔴| a + scalar (float64) | float64 | 100,000 | 0.0132 | 0.1095 | 8.30x |
|🟡| a + scalar (float64) | float64 | 10,000,000 | 16.1087 | 19.6990 | 1.22x |
|▫| a + scalar (int16) | int16 | 1,000 | 0.0009 | 0.0020 | 2.20x |
|🟡| a + scalar (int16) | int16 | 100,000 | 0.0248 | 0.0286 | 1.15x |
|✅| a + scalar (int16) | int16 | 10,000,000 | 6.9438 | 5.0678 | 0.73x |
|🟡| a + scalar (int32) | int32 | 1,000 | 0.0011 | 0.0017 | 1.54x |
|🟠| a + scalar (int32) | int32 | 100,000 | 0.0245 | 0.0528 | 2.16x |
|🟡| a + scalar (int32) | int32 | 10,000,000 | 9.2976 | 10.1713 | 1.09x |
|▫| a + scalar (int64) | int64 | 1,000 | 0.0008 | 0.0032 | 3.80x |
|🟠| a + scalar (int64) | int64 | 100,000 | 0.0238 | 0.1109 | 4.67x |
|🟡| a + scalar (int64) | int64 | 10,000,000 | 15.7965 | 19.6952 | 1.25x |
|▫| a + scalar (uint16) | uint16 | 1,000 | 0.0009 | 0.0017 | 1.86x |
|🟡| a + scalar (uint16) | uint16 | 100,000 | 0.0247 | 0.0262 | 1.06x |
|🟡| a + scalar (uint16) | uint16 | 10,000,000 | 5.1277 | 5.4373 | 1.06x |
|▫| a + scalar (uint32) | uint32 | 1,000 | 0.0009 | 0.0018 | 1.98x |
|🟠| a + scalar (uint32) | uint32 | 100,000 | 0.0245 | 0.0582 | 2.38x |
|🟡| a + scalar (uint32) | uint32 | 10,000,000 | 8.1236 | 10.3644 | 1.28x |
|▫| a + scalar (uint64) | uint64 | 1,000 | 0.0009 | 0.0030 | 3.43x |
|🟠| a + scalar (uint64) | uint64 | 100,000 | 0.0249 | 0.1055 | 4.23x |
|🟡| a + scalar (uint64) | uint64 | 10,000,000 | 16.2198 | 20.3895 | 1.26x |
|▫| a + scalar (uint8) | uint8 | 1,000 | 0.0008 | 0.0015 | 2.00x |
|✅| a + scalar (uint8) | uint8 | 100,000 | 0.0249 | 0.0140 | 0.56x |
|✅| a + scalar (uint8) | uint8 | 10,000,000 | 3.6095 | 2.4040 | 0.67x |
|▫| a - b (element-wise) (float32) | float32 | 1,000 | 0.0006 | 0.0020 | 3.55x |
|🔴| a - b (element-wise) (float32) | float32 | 100,000 | 0.0073 | 0.0576 | 7.91x |
|🟡| a - b (element-wise) (float32) | float32 | 10,000,000 | 9.0617 | 14.1836 | 1.56x |
|▫| a - b (element-wise) (float64) | float64 | 1,000 | 0.0005 | 0.0031 | 6.16x |
|🟠| a - b (element-wise) (float64) | float64 | 100,000 | 0.0297 | 0.1124 | 3.79x |
|🟡| a - b (element-wise) (float64) | float64 | 10,000,000 | 17.6095 | 26.5901 | 1.51x |
|▫| a - b (element-wise) (int16) | int16 | 1,000 | 0.0008 | 0.0020 | 2.54x |
|🟡| a - b (element-wise) (int16) | int16 | 100,000 | 0.0294 | 0.0298 | 1.01x |
|🟡| a - b (element-wise) (int16) | int16 | 10,000,000 | 7.1685 | 7.3131 | 1.02x |
|▫| a - b (element-wise) (int32) | int32 | 1,000 | 0.0008 | 0.0024 | 2.99x |
|🟠| a - b (element-wise) (int32) | int32 | 100,000 | 0.0297 | 0.0616 | 2.07x |
|🟡| a - b (element-wise) (int32) | int32 | 10,000,000 | 10.2594 | 14.1255 | 1.38x |
|▫| a - b (element-wise) (int64) | int64 | 1,000 | 0.0007 | 0.0031 | 4.20x |
|🟠| a - b (element-wise) (int64) | int64 | 100,000 | 0.0344 | 0.1160 | 3.37x |
|🟡| a - b (element-wise) (int64) | int64 | 10,000,000 | 18.0470 | 27.6480 | 1.53x |
|▫| a - b (element-wise) (uint16) | uint16 | 1,000 | 0.0008 | 0.0019 | 2.33x |
|✅| a - b (element-wise) (uint16) | uint16 | 100,000 | 0.0321 | 0.0299 | 0.93x |
|🟡| a - b (element-wise) (uint16) | uint16 | 10,000,000 | 5.2552 | 6.9091 | 1.31x |
|▫| a - b (element-wise) (uint32) | uint32 | 1,000 | 0.0008 | 0.0019 | 2.23x |
|🟡| a - b (element-wise) (uint32) | uint32 | 100,000 | 0.0319 | 0.0563 | 1.77x |
|🟡| a - b (element-wise) (uint32) | uint32 | 10,000,000 | 8.8780 | 14.3279 | 1.61x |
|▫| a - b (element-wise) (uint64) | uint64 | 1,000 | 0.0007 | 0.0029 | 3.87x |
|🟠| a - b (element-wise) (uint64) | uint64 | 100,000 | 0.0334 | 0.1087 | 3.26x |
|🟡| a - b (element-wise) (uint64) | uint64 | 10,000,000 | 18.6896 | 27.2793 | 1.46x |
|▫| a - b (element-wise) (uint8) | uint8 | 1,000 | 0.0006 | 0.0017 | 2.63x |
|✅| a - b (element-wise) (uint8) | uint8 | 100,000 | 0.0290 | 0.0156 | 0.54x |
|✅| a - b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0510 | 3.3608 | 0.83x |
|▫| a - scalar (float32) | float32 | 1,000 | 0.0007 | 0.0018 | 2.53x |
|🔴| a - scalar (float32) | float32 | 100,000 | 0.0063 | 0.0556 | 8.84x |
|🟡| a - scalar (float32) | float32 | 10,000,000 | 8.2991 | 10.2151 | 1.23x |
|▫| a - scalar (float64) | float64 | 1,000 | 0.0006 | 0.0028 | 4.45x |
|🔴| a - scalar (float64) | float64 | 100,000 | 0.0164 | 0.1059 | 6.47x |
|🟡| a - scalar (float64) | float64 | 10,000,000 | 16.1449 | 20.1159 | 1.25x |
|▫| a - scalar (int16) | int16 | 1,000 | 0.0009 | 0.0020 | 2.24x |
|🟡| a - scalar (int16) | int16 | 100,000 | 0.0253 | 0.0280 | 1.11x |
|✅| a - scalar (int16) | int16 | 10,000,000 | 5.4927 | 5.4867 | 1.00x |
|▫| a - scalar (int32) | int32 | 1,000 | 0.0009 | 0.0022 | 2.41x |
|🟠| a - scalar (int32) | int32 | 100,000 | 0.0253 | 0.0562 | 2.22x |
|🟡| a - scalar (int32) | int32 | 10,000,000 | 9.2399 | 10.0023 | 1.08x |
|▫| a - scalar (int64) | int64 | 1,000 | 0.0009 | 0.0029 | 3.38x |
|🟠| a - scalar (int64) | int64 | 100,000 | 0.0256 | 0.1178 | 4.61x |
|🟡| a - scalar (int64) | int64 | 10,000,000 | 15.5433 | 20.3870 | 1.31x |
|▫| a - scalar (uint16) | uint16 | 1,000 | 0.0009 | 0.0021 | 2.38x |
|🟡| a - scalar (uint16) | uint16 | 100,000 | 0.0246 | 0.0296 | 1.20x |
|🟡| a - scalar (uint16) | uint16 | 10,000,000 | 5.1743 | 5.3098 | 1.03x |
|▫| a - scalar (uint32) | uint32 | 1,000 | 0.0009 | 0.0021 | 2.30x |
|🟠| a - scalar (uint32) | uint32 | 100,000 | 0.0256 | 0.0567 | 2.22x |
|🟡| a - scalar (uint32) | uint32 | 10,000,000 | 7.9586 | 10.5442 | 1.32x |
|▫| a - scalar (uint64) | uint64 | 1,000 | 0.0009 | 0.0028 | 3.22x |
|🟠| a - scalar (uint64) | uint64 | 100,000 | 0.0252 | 0.1117 | 4.44x |
|🟡| a - scalar (uint64) | uint64 | 10,000,000 | 16.4043 | 20.4110 | 1.24x |
|▫| a - scalar (uint8) | uint8 | 1,000 | 0.0008 | 0.0016 | 2.12x |
|✅| a - scalar (uint8) | uint8 | 100,000 | 0.0250 | 0.0152 | 0.61x |
|✅| a - scalar (uint8) | uint8 | 10,000,000 | 3.5888 | 2.2351 | 0.62x |
|▫| a / b (element-wise) (float32) | float32 | 1,000 | 0.0005 | 0.0023 | 4.52x |
|🔴| a / b (element-wise) (float32) | float32 | 100,000 | 0.0123 | 0.0662 | 5.38x |
|🟡| a / b (element-wise) (float32) | float32 | 10,000,000 | 9.1487 | 13.7920 | 1.51x |
|▫| a / b (element-wise) (float64) | float64 | 1,000 | 0.0008 | 0.0041 | 5.16x |
|🔴| a / b (element-wise) (float64) | float64 | 100,000 | 0.0380 | 0.2011 | 5.29x |
|🟡| a / b (element-wise) (float64) | float64 | 10,000,000 | 19.1611 | 27.3718 | 1.43x |
|🟠| a / b (element-wise) (int32) | int32 | 1,000 | 0.0021 | 0.0065 | 3.05x |
|🟠| a / b (element-wise) (int32) | int32 | 100,000 | 0.0884 | 0.2040 | 2.31x |
|🟡| a / b (element-wise) (int32) | int32 | 10,000,000 | 20.6747 | 25.0264 | 1.21x |
|🟠| a / b (element-wise) (int64) | int64 | 1,000 | 0.0019 | 0.0064 | 3.43x |
|🟠| a / b (element-wise) (int64) | int64 | 100,000 | 0.0839 | 0.2048 | 2.44x |
|🟡| a / b (element-wise) (int64) | int64 | 10,000,000 | 26.5556 | 31.1644 | 1.17x |
|▫| a / scalar (float32) | float32 | 1,000 | 0.0008 | 0.0022 | 2.69x |
|🔴| a / scalar (float32) | float32 | 100,000 | 0.0127 | 0.0657 | 5.16x |
|🟡| a / scalar (float32) | float32 | 10,000,000 | 8.4858 | 10.5237 | 1.24x |
|▫| a / scalar (float64) | float64 | 1,000 | 0.0009 | 0.0045 | 4.99x |
|🟠| a / scalar (float64) | float64 | 100,000 | 0.0381 | 0.1875 | 4.92x |
|🟡| a / scalar (float64) | float64 | 10,000,000 | 16.4902 | 23.7052 | 1.44x |
|🟠| a / scalar (int32) | int32 | 1,000 | 0.0017 | 0.0079 | 4.63x |
|🟠| a / scalar (int32) | int32 | 100,000 | 0.0711 | 0.2066 | 2.90x |
|🟡| a / scalar (int32) | int32 | 10,000,000 | 17.1919 | 24.3257 | 1.42x |
|🟠| a / scalar (int64) | int64 | 1,000 | 0.0017 | 0.0070 | 4.15x |
|🟠| a / scalar (int64) | int64 | 100,000 | 0.0602 | 0.2092 | 3.47x |
|🟡| a / scalar (int64) | int64 | 10,000,000 | 19.6405 | 24.9502 | 1.27x |
|▫| np.add(a, b) (float32) | float32 | 1,000 | 0.0006 | 0.0018 | 3.10x |
|🔴| np.add(a, b) (float32) | float32 | 100,000 | 0.0071 | 0.0515 | 7.28x |
|🟡| np.add(a, b) (float32) | float32 | 10,000,000 | 8.9121 | 13.2948 | 1.49x |
|▫| np.add(a, b) (float64) | float64 | 1,000 | 0.0006 | 0.0030 | 5.17x |
|🟠| np.add(a, b) (float64) | float64 | 100,000 | 0.0301 | 0.1134 | 3.76x |
|🟡| np.add(a, b) (float64) | float64 | 10,000,000 | 17.9589 | 27.0545 | 1.51x |
|▫| np.add(a, b) (int16) | int16 | 1,000 | 0.0008 | 0.0020 | 2.47x |
|✅| np.add(a, b) (int16) | int16 | 100,000 | 0.0309 | 0.0285 | 0.92x |
|✅| np.add(a, b) (int16) | int16 | 10,000,000 | 7.3309 | 7.1899 | 0.98x |
|▫| np.add(a, b) (int32) | int32 | 1,000 | 0.0008 | 0.0018 | 2.28x |
|🟡| np.add(a, b) (int32) | int32 | 100,000 | 0.0316 | 0.0611 | 1.93x |
|🟡| np.add(a, b) (int32) | int32 | 10,000,000 | 9.9565 | 14.0945 | 1.42x |
|▫| np.add(a, b) (int64) | int64 | 1,000 | 0.0008 | 0.0031 | 4.05x |
|🟠| np.add(a, b) (int64) | int64 | 100,000 | 0.0332 | 0.1083 | 3.26x |
|🟡| np.add(a, b) (int64) | int64 | 10,000,000 | 20.6288 | 27.6310 | 1.34x |
|▫| np.add(a, b) (uint16) | uint16 | 1,000 | 0.0008 | 0.0016 | 2.07x |
|✅| np.add(a, b) (uint16) | uint16 | 100,000 | 0.0298 | 0.0257 | 0.86x |
|🟡| np.add(a, b) (uint16) | uint16 | 10,000,000 | 5.3638 | 7.1577 | 1.33x |
|▫| np.add(a, b) (uint32) | uint32 | 1,000 | 0.0008 | 0.0018 | 2.31x |
|🟡| np.add(a, b) (uint32) | uint32 | 100,000 | 0.0373 | 0.0543 | 1.46x |
|🟡| np.add(a, b) (uint32) | uint32 | 10,000,000 | 9.5713 | 14.2468 | 1.49x |
|▫| np.add(a, b) (uint64) | uint64 | 1,000 | 0.0008 | 0.0032 | 4.16x |
|🟠| np.add(a, b) (uint64) | uint64 | 100,000 | 0.0332 | 0.1151 | 3.47x |
|🟡| np.add(a, b) (uint64) | uint64 | 10,000,000 | 18.7016 | 26.5089 | 1.42x |
|▫| np.add(a, b) (uint8) | uint8 | 1,000 | 0.0007 | 0.0020 | 2.81x |
|✅| np.add(a, b) (uint8) | uint8 | 100,000 | 0.0289 | 0.0158 | 0.55x |
|✅| np.add(a, b) (uint8) | uint8 | 10,000,000 | 4.0238 | 3.0195 | 0.75x |
|▫| scalar - a (float32) | float32 | 1,000 | 0.0007 | 0.0020 | 2.72x |
|🔴| scalar - a (float32) | float32 | 100,000 | 0.0067 | 0.0536 | 7.94x |
|🟡| scalar - a (float32) | float32 | 10,000,000 | 8.4095 | 10.3296 | 1.23x |
|▫| scalar - a (float64) | float64 | 1,000 | 0.0007 | 0.0028 | 4.26x |
|🔴| scalar - a (float64) | float64 | 100,000 | 0.0137 | 0.1064 | 7.76x |
|🟡| scalar - a (float64) | float64 | 10,000,000 | 16.6031 | 19.7539 | 1.19x |
|▫| scalar - a (int16) | int16 | 1,000 | 0.0009 | 0.0016 | 1.78x |
|🟡| scalar - a (int16) | int16 | 100,000 | 0.0245 | 0.0289 | 1.18x |
|🟡| scalar - a (int16) | int16 | 10,000,000 | 4.6706 | 5.1130 | 1.09x |
|▫| scalar - a (int32) | int32 | 1,000 | 0.0009 | 0.0020 | 2.16x |
|🟠| scalar - a (int32) | int32 | 100,000 | 0.0260 | 0.0565 | 2.17x |
|🟡| scalar - a (int32) | int32 | 10,000,000 | 9.3299 | 10.1595 | 1.09x |
|▫| scalar - a (int64) | int64 | 1,000 | 0.0009 | 0.0031 | 3.63x |
|🟠| scalar - a (int64) | int64 | 100,000 | 0.0270 | 0.1098 | 4.07x |
|🟡| scalar - a (int64) | int64 | 10,000,000 | 16.0797 | 20.5011 | 1.27x |
|▫| scalar - a (uint16) | uint16 | 1,000 | 0.0009 | 0.0020 | 2.12x |
|🟡| scalar - a (uint16) | uint16 | 100,000 | 0.0260 | 0.0278 | 1.07x |
|🟡| scalar - a (uint16) | uint16 | 10,000,000 | 4.9032 | 5.4813 | 1.12x |
|▫| scalar - a (uint32) | uint32 | 1,000 | 0.0009 | 0.0019 | 2.07x |
|🟠| scalar - a (uint32) | uint32 | 100,000 | 0.0257 | 0.0537 | 2.09x |
|🟡| scalar - a (uint32) | uint32 | 10,000,000 | 8.2139 | 10.4418 | 1.27x |
|▫| scalar - a (uint64) | uint64 | 1,000 | 0.0010 | 0.0031 | 3.23x |
|🟠| scalar - a (uint64) | uint64 | 100,000 | 0.0248 | 0.1110 | 4.47x |
|🟡| scalar - a (uint64) | uint64 | 10,000,000 | 16.0049 | 19.7881 | 1.24x |
|▫| scalar - a (uint8) | uint8 | 1,000 | 0.0008 | 0.0018 | 2.25x |
|✅| scalar - a (uint8) | uint8 | 100,000 | 0.0244 | 0.0154 | 0.63x |
|✅| scalar - a (uint8) | uint8 | 10,000,000 | 3.7318 | 2.3712 | 0.64x |
|▫| scalar / a (float32) | float32 | 1,000 | 0.0009 | 0.0023 | 2.72x |
|🔴| scalar / a (float32) | float32 | 100,000 | 0.0124 | 0.0664 | 5.33x |
|🟡| scalar / a (float32) | float32 | 10,000,000 | 8.3590 | 10.1975 | 1.22x |
|▫| scalar / a (float64) | float64 | 1,000 | 0.0009 | 0.0041 | 4.47x |
|🔴| scalar / a (float64) | float64 | 100,000 | 0.0376 | 0.2015 | 5.36x |
|🟡| scalar / a (float64) | float64 | 10,000,000 | 16.3767 | 24.1568 | 1.48x |
|🟠| scalar / a (int32) | int32 | 1,000 | 0.0018 | 0.0073 | 4.13x |
|🟠| scalar / a (int32) | int32 | 100,000 | 0.0645 | 0.2063 | 3.20x |
|🟡| scalar / a (int32) | int32 | 10,000,000 | 17.3386 | 23.7354 | 1.37x |
|🟠| scalar / a (int64) | int64 | 1,000 | 0.0017 | 0.0076 | 4.45x |
|🟠| scalar / a (int64) | int64 | 100,000 | 0.0591 | 0.2074 | 3.51x |
|🟡| scalar / a (int64) | int64 | 10,000,000 | 19.6936 | 24.9236 | 1.27x |

### Unary

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.abs (float32) | float32 | 1,000 | 0.0005 | 0.0018 | 3.49x |
|🔴| np.abs (float32) | float32 | 100,000 | 0.0063 | 0.0641 | 10.14x |
|🟡| np.abs (float32) | float32 | 10,000,000 | 7.2200 | 10.9816 | 1.52x |
|▫| np.abs (float64) | float64 | 1,000 | 0.0005 | 0.0040 | 7.45x |
|🔴| np.abs (float64) | float64 | 100,000 | 0.0118 | 0.1196 | 10.16x |
|🟡| np.abs (float64) | float64 | 10,000,000 | 16.1376 | 20.9449 | 1.30x |
|🟠| np.cbrt(a) (float32) | float32 | 1,000 | 0.0062 | 0.0143 | 2.31x |
|🟡| np.cbrt(a) (float32) | float32 | 100,000 | 0.8787 | 1.5399 | 1.75x |
|🟡| np.cbrt(a) (float32) | float32 | 10,000,000 | 94.6669 | 150.8870 | 1.59x |
|🟠| np.cbrt(a) (float64) | float64 | 1,000 | 0.0095 | 0.0201 | 2.11x |
|🟠| np.cbrt(a) (float64) | float64 | 100,000 | 1.0613 | 2.1329 | 2.01x |
|🟡| np.cbrt(a) (float64) | float64 | 10,000,000 | 116.2440 | 210.7332 | 1.81x |
|▫| np.ceil (float32) | float32 | 1,000 | 0.0005 | 0.0020 | 4.04x |
|🔴| np.ceil (float32) | float32 | 100,000 | 0.0063 | 0.0537 | 8.55x |
|🟡| np.ceil (float32) | float32 | 10,000,000 | 7.9457 | 10.8311 | 1.36x |
|▫| np.ceil (float64) | float64 | 1,000 | 0.0006 | 0.0030 | 5.42x |
|🔴| np.ceil (float64) | float64 | 100,000 | 0.0110 | 0.1075 | 9.79x |
|🟡| np.ceil (float64) | float64 | 10,000,000 | 15.8831 | 21.5457 | 1.36x |
|🟡| np.cos (float32) | float32 | 1,000 | 0.0051 | 0.0082 | 1.61x |
|🟡| np.cos (float32) | float32 | 100,000 | 0.7044 | 1.1593 | 1.65x |
|🟡| np.cos (float32) | float32 | 10,000,000 | 80.2264 | 121.5219 | 1.51x |
|🟠| np.cos (float64) | float64 | 1,000 | 0.0049 | 0.0119 | 2.45x |
|🟡| np.cos (float64) | float64 | 100,000 | 0.7019 | 1.2530 | 1.78x |
|🟡| np.cos (float64) | float64 | 10,000,000 | 79.2760 | 128.1130 | 1.62x |
|🟠| np.exp (float32) | float32 | 1,000 | 0.0013 | 0.0060 | 4.56x |
|🔴| np.exp (float32) | float32 | 100,000 | 0.0574 | 0.3996 | 6.96x |
|🟠| np.exp (float32) | float32 | 10,000,000 | 14.0653 | 42.5193 | 3.02x |
|🟠| np.exp (float64) | float64 | 1,000 | 0.0029 | 0.0074 | 2.50x |
|🟠| np.exp (float64) | float64 | 100,000 | 0.2525 | 0.5145 | 2.04x |
|🟡| np.exp (float64) | float64 | 10,000,000 | 33.8602 | 53.0973 | 1.57x |
|▫| np.floor (float32) | float32 | 1,000 | 0.0005 | 0.0022 | 4.13x |
|🔴| np.floor (float32) | float32 | 100,000 | 0.0065 | 0.0578 | 8.91x |
|🟡| np.floor (float32) | float32 | 10,000,000 | 8.0349 | 10.8412 | 1.35x |
|▫| np.floor (float64) | float64 | 1,000 | 0.0006 | 0.0031 | 5.69x |
|🔴| np.floor (float64) | float64 | 100,000 | 0.0112 | 0.1256 | 11.21x |
|🟡| np.floor (float64) | float64 | 10,000,000 | 16.5852 | 21.3637 | 1.29x |
|🟠| np.log (float32) | float32 | 1,000 | 0.0013 | 0.0059 | 4.39x |
|🔴| np.log (float32) | float32 | 100,000 | 0.0876 | 0.4897 | 5.59x |
|🟠| np.log (float32) | float32 | 10,000,000 | 16.0105 | 46.6480 | 2.91x |
|🟠| np.log (float64) | float64 | 1,000 | 0.0028 | 0.0075 | 2.65x |
|🟠| np.log (float64) | float64 | 100,000 | 0.2495 | 0.5997 | 2.40x |
|🟡| np.log (float64) | float64 | 10,000,000 | 31.7591 | 61.5853 | 1.94x |
|🟠| np.log10 (float32) | float32 | 1,000 | 0.0024 | 0.0055 | 2.29x |
|🟠| np.log10 (float32) | float32 | 100,000 | 0.1905 | 0.4873 | 2.56x |
|🟡| np.log10 (float32) | float32 | 10,000,000 | 23.3160 | 46.3308 | 1.99x |
|🟠| np.log10 (float64) | float64 | 1,000 | 0.0029 | 0.0077 | 2.65x |
|🟠| np.log10 (float64) | float64 | 100,000 | 0.2461 | 0.6713 | 2.73x |
|🟡| np.log10 (float64) | float64 | 10,000,000 | 33.4928 | 64.1697 | 1.92x |
|▫| np.negative(a) (float32) | float32 | 1,000 | 0.0005 | 0.0020 | 3.82x |
|🔴| np.negative(a) (float32) | float32 | 100,000 | 0.0063 | 0.0531 | 8.44x |
|🟡| np.negative(a) (float32) | float32 | 10,000,000 | 8.2190 | 10.4472 | 1.27x |
|▫| np.negative(a) (float64) | float64 | 1,000 | 0.0005 | 0.0027 | 4.97x |
|🔴| np.negative(a) (float64) | float64 | 100,000 | 0.0134 | 0.0977 | 7.29x |
|🟡| np.negative(a) (float64) | float64 | 10,000,000 | 16.8373 | 20.2151 | 1.20x |
|▫| np.positive(a) (float32) | float32 | 1,000 | 0.0008 | 0.0021 | 2.73x |
|🟠| np.positive(a) (float32) | float32 | 100,000 | 0.0193 | 0.0510 | 2.64x |
|✅| np.positive(a) (float32) | float32 | 10,000,000 | 8.1732 | 7.4167 | 0.91x |
|▫| np.positive(a) (float64) | float64 | 1,000 | 0.0007 | 0.0030 | 4.12x |
|🔴| np.positive(a) (float64) | float64 | 100,000 | 0.0205 | 0.1042 | 5.09x |
|🟡| np.positive(a) (float64) | float64 | 10,000,000 | 14.9741 | 15.0630 | 1.01x |
|▫| np.reciprocal(a) (float32) | float32 | 1,000 | 0.0006 | 0.0021 | 3.43x |
|🟠| np.reciprocal(a) (float32) | float32 | 100,000 | 0.0144 | 0.0646 | 4.48x |
|🟡| np.reciprocal(a) (float32) | float32 | 10,000,000 | 7.3153 | 10.4864 | 1.43x |
|▫| np.reciprocal(a) (float64) | float64 | 1,000 | 0.0008 | 0.0045 | 5.81x |
|🔴| np.reciprocal(a) (float64) | float64 | 100,000 | 0.0380 | 0.2049 | 5.40x |
|🟡| np.reciprocal(a) (float64) | float64 | 10,000,000 | 15.6905 | 23.3012 | 1.49x |
|⚪| np.round (float32) | float32 | 1,000 | 0.0011 | - | - |
|⚪| np.round (float32) | float32 | 100,000 | 0.0069 | - | - |
|⚪| np.round (float32) | float32 | 10,000,000 | 8.9861 | - | - |
|⚪| np.round (float64) | float64 | 1,000 | 0.0012 | - | - |
|⚪| np.round (float64) | float64 | 100,000 | 0.0118 | - | - |
|⚪| np.round (float64) | float64 | 10,000,000 | 16.6718 | - | - |
|🟠| np.sign (float32) | float32 | 1,000 | 0.0012 | 0.0042 | 3.62x |
|🟡| np.sign (float32) | float32 | 100,000 | 0.2996 | 0.5602 | 1.87x |
|🟡| np.sign (float32) | float32 | 10,000,000 | 36.4791 | 60.9430 | 1.67x |
|🟠| np.sign (float64) | float64 | 1,000 | 0.0011 | 0.0042 | 3.96x |
|🟠| np.sign (float64) | float64 | 100,000 | 0.2916 | 0.5861 | 2.01x |
|🟡| np.sign (float64) | float64 | 10,000,000 | 45.0343 | 63.7798 | 1.42x |
|🟡| np.sin (float32) | float32 | 1,000 | 0.0048 | 0.0090 | 1.89x |
|🟡| np.sin (float32) | float32 | 100,000 | 0.7145 | 1.2215 | 1.71x |
|🟡| np.sin (float32) | float32 | 10,000,000 | 79.8713 | 123.5466 | 1.55x |
|🟠| np.sin (float64) | float64 | 1,000 | 0.0047 | 0.0117 | 2.47x |
|🟡| np.sin (float64) | float64 | 100,000 | 0.7068 | 1.2552 | 1.78x |
|🟡| np.sin (float64) | float64 | 10,000,000 | 79.5760 | 127.2737 | 1.60x |
|🟡| np.sqrt (float32) | float32 | 1,000 | 0.0013 | 0.0023 | 1.85x |
|🔴| np.sqrt (float32) | float32 | 100,000 | 0.0143 | 0.0777 | 5.42x |
|🟡| np.sqrt (float32) | float32 | 10,000,000 | 7.3215 | 11.0763 | 1.51x |
|🔴| np.sqrt (float64) | float64 | 1,000 | 0.0010 | 0.0052 | 5.20x |
|🔴| np.sqrt (float64) | float64 | 100,000 | 0.0580 | 0.3056 | 5.27x |
|🟠| np.sqrt (float64) | float64 | 10,000,000 | 15.8606 | 33.0776 | 2.09x |
|▫| np.square(a) (float32) | float32 | 1,000 | 0.0005 | 0.0017 | 3.45x |
|🔴| np.square(a) (float32) | float32 | 100,000 | 0.0067 | 0.0562 | 8.36x |
|🟡| np.square(a) (float32) | float32 | 10,000,000 | 7.6096 | 10.4224 | 1.37x |
|▫| np.square(a) (float64) | float64 | 1,000 | 0.0005 | 0.0030 | 6.09x |
|🔴| np.square(a) (float64) | float64 | 100,000 | 0.0109 | 0.1017 | 9.33x |
|🟡| np.square(a) (float64) | float64 | 10,000,000 | 15.6136 | 19.9551 | 1.28x |
|▫| np.trunc(a) (float32) | float32 | 1,000 | 0.0005 | 0.0021 | 3.97x |
|🔴| np.trunc(a) (float32) | float32 | 100,000 | 0.0058 | 0.0538 | 9.28x |
|🟡| np.trunc(a) (float32) | float32 | 10,000,000 | 7.6326 | 10.5661 | 1.38x |
|▫| np.trunc(a) (float64) | float64 | 1,000 | 0.0005 | 0.0029 | 5.52x |
|🔴| np.trunc(a) (float64) | float64 | 100,000 | 0.0109 | 0.1040 | 9.50x |
|🟡| np.trunc(a) (float64) | float64 | 10,000,000 | 14.6536 | 20.0106 | 1.37x |

### Reduction

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.amax (float32) | float32 | 1,000 | 0.0017 | 0.0008 | 0.47x |
|🟠| np.amax (float32) | float32 | 100,000 | 0.0060 | 0.0138 | 2.30x |
|🟡| np.amax (float32) | float32 | 10,000,000 | 1.4959 | 2.0328 | 1.36x |
|✅| np.amax (float64) | float64 | 1,000 | 0.0017 | 0.0010 | 0.60x |
|🟠| np.amax (float64) | float64 | 100,000 | 0.0102 | 0.0271 | 2.66x |
|🟡| np.amax (float64) | float64 | 10,000,000 | 3.7656 | 4.2964 | 1.14x |
|▫| np.amax (int16) | int16 | 1,000 | 0.0016 | 0.0008 | 0.49x |
|✅| np.amax (int16) | int16 | 100,000 | 0.0030 | 0.0020 | 0.66x |
|🟡| np.amax (int16) | int16 | 10,000,000 | 0.3013 | 0.3398 | 1.13x |
|▫| np.amax (int32) | int32 | 1,000 | 0.0024 | 0.0007 | 0.27x |
|✅| np.amax (int32) | int32 | 100,000 | 0.0042 | 0.0032 | 0.77x |
|🟡| np.amax (int32) | int32 | 10,000,000 | 1.2020 | 1.2196 | 1.01x |
|▫| np.amax (int64) | int64 | 1,000 | 0.0017 | 0.0008 | 0.47x |
|✅| np.amax (int64) | int64 | 100,000 | 0.0091 | 0.0074 | 0.82x |
|🟡| np.amax (int64) | int64 | 10,000,000 | 3.7203 | 3.8740 | 1.04x |
|▫| np.amax (uint16) | uint16 | 1,000 | 0.0016 | 0.0008 | 0.49x |
|✅| np.amax (uint16) | uint16 | 100,000 | 0.0031 | 0.0020 | 0.64x |
|✅| np.amax (uint16) | uint16 | 10,000,000 | 0.3344 | 0.3302 | 0.99x |
|▫| np.amax (uint32) | uint32 | 1,000 | 0.0016 | 0.0008 | 0.48x |
|✅| np.amax (uint32) | uint32 | 100,000 | 0.0042 | 0.0032 | 0.77x |
|✅| np.amax (uint32) | uint32 | 10,000,000 | 1.2650 | 1.2167 | 0.96x |
|▫| np.amax (uint64) | uint64 | 1,000 | 0.0017 | 0.0008 | 0.45x |
|✅| np.amax (uint64) | uint64 | 100,000 | 0.0120 | 0.0097 | 0.81x |
|🟡| np.amax (uint64) | uint64 | 10,000,000 | 4.0728 | 4.0940 | 1.00x |
|▫| np.amax (uint8) | uint8 | 1,000 | 0.0016 | 0.0006 | 0.40x |
|✅| np.amax (uint8) | uint8 | 100,000 | 0.0024 | 0.0012 | 0.49x |
|✅| np.amax (uint8) | uint8 | 10,000,000 | 0.1476 | 0.1468 | 0.99x |
|✅| np.amin (float32) | float32 | 1,000 | 0.0016 | 0.0013 | 0.81x |
|🔴| np.amin (float32) | float32 | 100,000 | 0.0066 | 0.0512 | 7.78x |
|🟠| np.amin (float32) | float32 | 10,000,000 | 1.4832 | 5.2599 | 3.55x |
|🟡| np.amin (float64) | float64 | 1,000 | 0.0017 | 0.0021 | 1.26x |
|🔴| np.amin (float64) | float64 | 100,000 | 0.0102 | 0.0922 | 9.06x |
|🟠| np.amin (float64) | float64 | 10,000,000 | 3.9366 | 10.3287 | 2.62x |
|▫| np.amin (int16) | int16 | 1,000 | 0.0016 | 0.0006 | 0.40x |
|✅| np.amin (int16) | int16 | 100,000 | 0.0036 | 0.0027 | 0.77x |
|🟠| np.amin (int16) | int16 | 10,000,000 | 0.3248 | 0.7557 | 2.33x |
|▫| np.amin (int32) | int32 | 1,000 | 0.0016 | 0.0008 | 0.48x |
|🟡| np.amin (int32) | int32 | 100,000 | 0.0046 | 0.0048 | 1.04x |
|🟠| np.amin (int32) | int32 | 10,000,000 | 1.2313 | 3.5990 | 2.92x |
|✅| np.amin (int64) | int64 | 1,000 | 0.0017 | 0.0010 | 0.62x |
|🟠| np.amin (int64) | int64 | 100,000 | 0.0121 | 0.0275 | 2.27x |
|🟠| np.amin (int64) | int64 | 10,000,000 | 3.6693 | 8.4604 | 2.31x |
|▫| np.amin (uint16) | uint16 | 1,000 | 0.0016 | 0.0008 | 0.52x |
|✅| np.amin (uint16) | uint16 | 100,000 | 0.0034 | 0.0028 | 0.83x |
|🟠| np.amin (uint16) | uint16 | 10,000,000 | 0.3120 | 0.7134 | 2.29x |
|▫| np.amin (uint32) | uint32 | 1,000 | 0.0016 | 0.0008 | 0.49x |
|🟡| np.amin (uint32) | uint32 | 100,000 | 0.0048 | 0.0056 | 1.17x |
|🟠| np.amin (uint32) | uint32 | 10,000,000 | 1.3068 | 3.7324 | 2.86x |
|✅| np.amin (uint64) | uint64 | 1,000 | 0.0017 | 0.0012 | 0.70x |
|🟠| np.amin (uint64) | uint64 | 100,000 | 0.0122 | 0.0364 | 2.98x |
|🟠| np.amin (uint64) | uint64 | 10,000,000 | 3.7867 | 8.9571 | 2.37x |
|▫| np.amin (uint8) | uint8 | 1,000 | 0.0016 | 0.0007 | 0.44x |
|✅| np.amin (uint8) | uint8 | 100,000 | 0.0026 | 0.0019 | 0.76x |
|🟡| np.amin (uint8) | uint8 | 10,000,000 | 0.1496 | 0.2392 | 1.60x |
|▫| np.argmax (float32) | float32 | 1,000 | 0.0009 | 0.0012 | 1.38x |
|🔴| np.argmax (float32) | float32 | 100,000 | 0.0088 | 0.0562 | 6.42x |
|🟠| np.argmax (float32) | float32 | 10,000,000 | 2.0610 | 5.8117 | 2.82x |
|▫| np.argmax (float64) | float64 | 1,000 | 0.0010 | 0.0012 | 1.25x |
|🟠| np.argmax (float64) | float64 | 100,000 | 0.0166 | 0.0566 | 3.42x |
|🟡| np.argmax (float64) | float64 | 10,000,000 | 4.9803 | 6.9656 | 1.40x |
|▫| np.argmax (int16) | int16 | 1,000 | 0.0009 | 0.0007 | 0.83x |
|✅| np.argmax (int16) | int16 | 100,000 | 0.0034 | 0.0020 | 0.58x |
|✅| np.argmax (int16) | int16 | 10,000,000 | 0.4154 | 0.3655 | 0.88x |
|▫| np.argmax (int32) | int32 | 1,000 | 0.0009 | 0.0007 | 0.84x |
|✅| np.argmax (int32) | int32 | 100,000 | 0.0055 | 0.0037 | 0.67x |
|✅| np.argmax (int32) | int32 | 10,000,000 | 1.9769 | 1.4307 | 0.72x |
|▫| np.argmax (int64) | int64 | 1,000 | 0.0009 | 0.0009 | 0.98x |
|🟡| np.argmax (int64) | int64 | 100,000 | 0.0145 | 0.0282 | 1.95x |
|🟡| np.argmax (int64) | int64 | 10,000,000 | 4.6597 | 4.7984 | 1.03x |
|▫| np.argmax (uint16) | uint16 | 1,000 | 0.0009 | 0.0007 | 0.82x |
|✅| np.argmax (uint16) | uint16 | 100,000 | 0.0050 | 0.0020 | 0.40x |
|✅| np.argmax (uint16) | uint16 | 10,000,000 | 0.6705 | 0.3817 | 0.57x |
|▫| np.argmax (uint32) | uint32 | 1,000 | 0.0009 | 0.0008 | 0.87x |
|✅| np.argmax (uint32) | uint32 | 100,000 | 0.0088 | 0.0037 | 0.42x |
|✅| np.argmax (uint32) | uint32 | 10,000,000 | 2.0347 | 1.3987 | 0.69x |
|▫| np.argmax (uint64) | uint64 | 1,000 | 0.0010 | 0.0009 | 0.94x |
|🟡| np.argmax (uint64) | uint64 | 100,000 | 0.0210 | 0.0330 | 1.57x |
|🟡| np.argmax (uint64) | uint64 | 10,000,000 | 4.5903 | 5.1927 | 1.13x |
|▫| np.argmax (uint8) | uint8 | 1,000 | 0.0010 | 0.0007 | 0.68x |
|✅| np.argmax (uint8) | uint8 | 100,000 | 0.0031 | 0.0013 | 0.40x |
|✅| np.argmax (uint8) | uint8 | 10,000,000 | 0.2231 | 0.1490 | 0.67x |
|▫| np.argmin (float32) | float32 | 1,000 | 0.0009 | 0.0012 | 1.39x |
|🔴| np.argmin (float32) | float32 | 100,000 | 0.0088 | 0.0566 | 6.39x |
|🟠| np.argmin (float32) | float32 | 10,000,000 | 1.9838 | 5.7762 | 2.91x |
|▫| np.argmin (float64) | float64 | 1,000 | 0.0010 | 0.0010 | 1.00x |
|🟠| np.argmin (float64) | float64 | 100,000 | 0.0165 | 0.0571 | 3.45x |
|🟡| np.argmin (float64) | float64 | 10,000,000 | 4.9495 | 6.8670 | 1.39x |
|▫| np.argmin (int16) | int16 | 1,000 | 0.0009 | 0.0006 | 0.72x |
|✅| np.argmin (int16) | int16 | 100,000 | 0.0039 | 0.0022 | 0.58x |
|✅| np.argmin (int16) | int16 | 10,000,000 | 0.5638 | 0.3626 | 0.64x |
|▫| np.argmin (int32) | int32 | 1,000 | 0.0009 | 0.0008 | 0.88x |
|✅| np.argmin (int32) | int32 | 100,000 | 0.0054 | 0.0036 | 0.66x |
|✅| np.argmin (int32) | int32 | 10,000,000 | 2.0515 | 1.3733 | 0.67x |
|▫| np.argmin (int64) | int64 | 1,000 | 0.0009 | 0.0008 | 0.90x |
|🟠| np.argmin (int64) | int64 | 100,000 | 0.0141 | 0.0284 | 2.00x |
|✅| np.argmin (int64) | int64 | 10,000,000 | 4.9670 | 4.6915 | 0.94x |
|▫| np.argmin (uint16) | uint16 | 1,000 | 0.0009 | 0.0007 | 0.85x |
|✅| np.argmin (uint16) | uint16 | 100,000 | 0.0050 | 0.0021 | 0.43x |
|✅| np.argmin (uint16) | uint16 | 10,000,000 | 0.5498 | 0.3746 | 0.68x |
|▫| np.argmin (uint32) | uint32 | 1,000 | 0.0009 | 0.0006 | 0.72x |
|✅| np.argmin (uint32) | uint32 | 100,000 | 0.0087 | 0.0036 | 0.41x |
|✅| np.argmin (uint32) | uint32 | 10,000,000 | 2.0280 | 1.2602 | 0.62x |
|▫| np.argmin (uint64) | uint64 | 1,000 | 0.0010 | 0.0010 | 0.98x |
|🟡| np.argmin (uint64) | uint64 | 100,000 | 0.0170 | 0.0331 | 1.95x |
|🟡| np.argmin (uint64) | uint64 | 10,000,000 | 4.4945 | 5.1462 | 1.15x |
|▫| np.argmin (uint8) | uint8 | 1,000 | 0.0009 | 0.0008 | 0.83x |
|✅| np.argmin (uint8) | uint8 | 100,000 | 0.0030 | 0.0012 | 0.40x |
|✅| np.argmin (uint8) | uint8 | 10,000,000 | 0.2171 | 0.1481 | 0.68x |
|🔴| np.cumprod(a) (float32) | float32 | 1,000 | 0.0037 | 0.0190 | 5.17x |
|🟡| np.cumprod(a) (float32) | float32 | 100,000 | 0.1714 | 0.2774 | 1.62x |
|🟡| np.cumprod(a) (float32) | float32 | 10,000,000 | 22.7044 | 23.9230 | 1.05x |
|🟠| np.cumprod(a) (float64) | float64 | 1,000 | 0.0043 | 0.0169 | 3.94x |
|🟠| np.cumprod(a) (float64) | float64 | 100,000 | 0.1718 | 0.4216 | 2.45x |
|🟡| np.cumprod(a) (float64) | float64 | 10,000,000 | 25.3609 | 39.2889 | 1.55x |
|▫| np.mean (float32) | float32 | 1,000 | 0.0037 | 0.0008 | 0.21x |
|✅| np.mean (float32) | float32 | 100,000 | 0.0190 | 0.0032 | 0.17x |
|✅| np.mean (float32) | float32 | 10,000,000 | 3.0599 | 1.0998 | 0.36x |
|▫| np.mean (float64) | float64 | 1,000 | 0.0024 | 0.0008 | 0.34x |
|✅| np.mean (float64) | float64 | 100,000 | 0.0176 | 0.0040 | 0.23x |
|✅| np.mean (float64) | float64 | 10,000,000 | 5.0231 | 2.9177 | 0.58x |
|⚪| np.mean (int16) | int16 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (int16) | int16 | 100,000 | 0.0520 | - | - |
|⚪| np.mean (int16) | int16 | 10,000,000 | 5.1284 | - | - |
|✅| np.mean (int32) | int32 | 1,000 | 0.0030 | 0.0012 | 0.40x |
|✅| np.mean (int32) | int32 | 100,000 | 0.0465 | 0.0191 | 0.41x |
|✅| np.mean (int32) | int32 | 10,000,000 | 4.5937 | 2.8233 | 0.61x |
|✅| np.mean (int64) | int64 | 1,000 | 0.0029 | 0.0014 | 0.48x |
|✅| np.mean (int64) | int64 | 100,000 | 0.0338 | 0.0045 | 0.13x |
|✅| np.mean (int64) | int64 | 10,000,000 | 6.3172 | 2.9911 | 0.47x |
|⚪| np.mean (uint16) | uint16 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint16) | uint16 | 100,000 | 0.0549 | - | - |
|⚪| np.mean (uint16) | uint16 | 10,000,000 | 5.0823 | - | - |
|⚪| np.mean (uint32) | uint32 | 1,000 | 0.0029 | - | - |
|⚪| np.mean (uint32) | uint32 | 100,000 | 0.0399 | - | - |
|⚪| np.mean (uint32) | uint32 | 10,000,000 | 4.7566 | - | - |
|⚪| np.mean (uint64) | uint64 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint64) | uint64 | 100,000 | 0.0519 | - | - |
|⚪| np.mean (uint64) | uint64 | 10,000,000 | 7.8053 | - | - |
|⚪| np.mean (uint8) | uint8 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint8) | uint8 | 100,000 | 0.0544 | - | - |
|⚪| np.mean (uint8) | uint8 | 10,000,000 | 5.0084 | - | - |
|✅| np.nanmax(a) (float32) | float32 | 1,000 | 0.0029 | 0.0013 | 0.43x |
|🔴| np.nanmax(a) (float32) | float32 | 100,000 | 0.0078 | 0.0518 | 6.60x |
|🟠| np.nanmax(a) (float32) | float32 | 10,000,000 | 1.4629 | 3.3143 | 2.27x |
|✅| np.nanmax(a) (float64) | float64 | 1,000 | 0.0029 | 0.0019 | 0.65x |
|🔴| np.nanmax(a) (float64) | float64 | 100,000 | 0.0114 | 0.1022 | 8.93x |
|🟡| np.nanmax(a) (float64) | float64 | 10,000,000 | 4.0560 | 6.9618 | 1.72x |
|✅| np.nanmean(a) (float32) | float32 | 1,000 | 0.0100 | 0.0018 | 0.18x |
|✅| np.nanmean(a) (float32) | float32 | 100,000 | 0.0732 | 0.0725 | 0.99x |
|✅| np.nanmean(a) (float32) | float32 | 10,000,000 | 19.8275 | 4.1943 | 0.21x |
|✅| np.nanmean(a) (float64) | float64 | 1,000 | 0.0085 | 0.0019 | 0.22x |
|✅| np.nanmean(a) (float64) | float64 | 100,000 | 0.3215 | 0.0745 | 0.23x |
|✅| np.nanmean(a) (float64) | float64 | 10,000,000 | 33.4663 | 5.6865 | 0.17x |
|✅| np.nanmedian(a) (float32) | float32 | 1,000 | 0.0131 | 0.0037 | 0.28x |
|🟡| np.nanmedian(a) (float32) | float32 | 100,000 | 0.4984 | 0.9645 | 1.94x |
|🟡| np.nanmedian(a) (float32) | float32 | 10,000,000 | 77.8307 | 80.5666 | 1.03x |
|✅| np.nanmedian(a) (float64) | float64 | 1,000 | 0.0116 | 0.0039 | 0.33x |
|🟠| np.nanmedian(a) (float64) | float64 | 100,000 | 0.4838 | 0.9947 | 2.06x |
|✅| np.nanmedian(a) (float64) | float64 | 10,000,000 | 93.1146 | 92.3006 | 0.99x |
|✅| np.nanmin(a) (float32) | float32 | 1,000 | 0.0029 | 0.0012 | 0.42x |
|🔴| np.nanmin(a) (float32) | float32 | 100,000 | 0.0071 | 0.0523 | 7.38x |
|🟠| np.nanmin(a) (float32) | float32 | 10,000,000 | 1.6131 | 3.3612 | 2.08x |
|✅| np.nanmin(a) (float64) | float64 | 1,000 | 0.0029 | 0.0019 | 0.66x |
|🔴| np.nanmin(a) (float64) | float64 | 100,000 | 0.0115 | 0.1021 | 8.85x |
|🟡| np.nanmin(a) (float64) | float64 | 10,000,000 | 4.2492 | 6.9814 | 1.64x |
|✅| np.nanpercentile(a, 50) (float32) | float32 | 1,000 | 0.0259 | 0.0037 | 0.14x |
|🟡| np.nanpercentile(a, 50) (float32) | float32 | 100,000 | 0.7090 | 0.9670 | 1.36x |
|🟡| np.nanpercentile(a, 50) (float32) | float32 | 10,000,000 | 52.5157 | 80.7603 | 1.54x |
|✅| np.nanpercentile(a, 50) (float64) | float64 | 1,000 | 0.0276 | 0.0038 | 0.14x |
|🟡| np.nanpercentile(a, 50) (float64) | float64 | 100,000 | 0.7817 | 1.0271 | 1.31x |
|🟡| np.nanpercentile(a, 50) (float64) | float64 | 10,000,000 | 66.2596 | 90.8813 | 1.37x |
|✅| np.nanprod(a) (float32) | float32 | 1,000 | 0.0050 | 0.0014 | 0.28x |
|✅| np.nanprod(a) (float32) | float32 | 100,000 | 0.0959 | 0.0162 | 0.17x |
|✅| np.nanprod(a) (float32) | float32 | 10,000,000 | 18.5148 | 1.9040 | 0.10x |
|✅| np.nanprod(a) (float64) | float64 | 1,000 | 0.0050 | 0.0010 | 0.20x |
|✅| np.nanprod(a) (float64) | float64 | 100,000 | 0.2867 | 0.0319 | 0.11x |
|✅| np.nanprod(a) (float64) | float64 | 10,000,000 | 27.1782 | 4.5263 | 0.17x |
|✅| np.nanquantile(a, 0.5) (float32) | float32 | 1,000 | 0.0251 | 0.0037 | 0.15x |
|🟡| np.nanquantile(a, 0.5) (float32) | float32 | 100,000 | 0.7374 | 0.9637 | 1.31x |
|🟡| np.nanquantile(a, 0.5) (float32) | float32 | 10,000,000 | 66.8036 | 80.4490 | 1.20x |
|✅| np.nanquantile(a, 0.5) (float64) | float64 | 1,000 | 0.0283 | 0.0037 | 0.13x |
|🟡| np.nanquantile(a, 0.5) (float64) | float64 | 100,000 | 0.7495 | 0.9853 | 1.31x |
|🟡| np.nanquantile(a, 0.5) (float64) | float64 | 10,000,000 | 64.7940 | 90.9806 | 1.40x |
|✅| np.nanstd(a) (float32) | float32 | 1,000 | 0.0203 | 0.0025 | 0.12x |
|✅| np.nanstd(a) (float32) | float32 | 100,000 | 0.1635 | 0.1517 | 0.93x |
|✅| np.nanstd(a) (float32) | float32 | 10,000,000 | 32.7545 | 9.2835 | 0.28x |
|✅| np.nanstd(a) (float64) | float64 | 1,000 | 0.0183 | 0.0024 | 0.13x |
|✅| np.nanstd(a) (float64) | float64 | 100,000 | 0.4565 | 0.1477 | 0.32x |
|✅| np.nanstd(a) (float64) | float64 | 10,000,000 | 52.9156 | 11.4366 | 0.22x |
|✅| np.nansum(a) (float32) | float32 | 1,000 | 0.0037 | 0.0013 | 0.34x |
|✅| np.nansum(a) (float32) | float32 | 100,000 | 0.0324 | 0.0096 | 0.30x |
|✅| np.nansum(a) (float32) | float32 | 10,000,000 | 14.3488 | 1.4880 | 0.10x |
|✅| np.nansum(a) (float64) | float64 | 1,000 | 0.0037 | 0.0014 | 0.37x |
|✅| np.nansum(a) (float64) | float64 | 100,000 | 0.2425 | 0.0192 | 0.08x |
|✅| np.nansum(a) (float64) | float64 | 10,000,000 | 25.5404 | 3.6530 | 0.14x |
|✅| np.nanvar(a) (float32) | float32 | 1,000 | 0.0195 | 0.0025 | 0.13x |
|✅| np.nanvar(a) (float32) | float32 | 100,000 | 0.1731 | 0.1550 | 0.90x |
|✅| np.nanvar(a) (float32) | float32 | 10,000,000 | 33.3949 | 9.2884 | 0.28x |
|✅| np.nanvar(a) (float64) | float64 | 1,000 | 0.0175 | 0.0024 | 0.14x |
|✅| np.nanvar(a) (float64) | float64 | 100,000 | 0.4367 | 0.1528 | 0.35x |
|✅| np.nanvar(a) (float64) | float64 | 10,000,000 | 56.9159 | 11.7838 | 0.21x |
|✅| np.std (float32) | float32 | 1,000 | 0.0082 | 0.0011 | 0.13x |
|✅| np.std (float32) | float32 | 100,000 | 0.0480 | 0.0097 | 0.20x |
|✅| np.std (float32) | float32 | 10,000,000 | 16.7538 | 2.5973 | 0.15x |
|▫| np.std (float64) | float64 | 1,000 | 0.0067 | 0.0009 | 0.13x |
|✅| np.std (float64) | float64 | 100,000 | 0.0578 | 0.0190 | 0.33x |
|✅| np.std (float64) | float64 | 10,000,000 | 32.8482 | 6.7593 | 0.21x |
|▫| np.sum (float32) | float32 | 1,000 | 0.0018 | 0.0008 | 0.42x |
|✅| np.sum (float32) | float32 | 100,000 | 0.0160 | 0.0032 | 0.20x |
|✅| np.sum (float32) | float32 | 10,000,000 | 2.9675 | 1.0551 | 0.36x |
|▫| np.sum (float64) | float64 | 1,000 | 0.0017 | 0.0008 | 0.44x |
|🔴| np.sum (float64) | float64 | 100,000 | 0.0176 | 0.2085 | 11.83x |
|✅| np.sum (float64) | float64 | 10,000,000 | 5.0434 | 3.4958 | 0.69x |
|▫| np.sum (int16) | int16 | 1,000 | 0.0022 | 0.0008 | 0.37x |
|✅| np.sum (int16) | int16 | 100,000 | 0.0334 | 0.0189 | 0.56x |
|✅| np.sum (int16) | int16 | 10,000,000 | 3.3717 | 1.9534 | 0.58x |
|▫| np.sum (int32) | int32 | 1,000 | 0.0024 | 0.0008 | 0.33x |
|✅| np.sum (int32) | int32 | 100,000 | 0.0352 | 0.0190 | 0.54x |
|✅| np.sum (int32) | int32 | 10,000,000 | 4.4795 | 2.7217 | 0.61x |
|▫| np.sum (int64) | int64 | 1,000 | 0.0017 | 0.0008 | 0.44x |
|✅| np.sum (int64) | int64 | 100,000 | 0.0199 | 0.0066 | 0.33x |
|✅| np.sum (int64) | int64 | 10,000,000 | 4.5897 | 2.7890 | 0.61x |
|▫| np.sum (uint16) | uint16 | 1,000 | 0.0021 | 0.0008 | 0.39x |
|✅| np.sum (uint16) | uint16 | 100,000 | 0.0343 | 0.0189 | 0.55x |
|✅| np.sum (uint16) | uint16 | 10,000,000 | 3.3164 | 1.9407 | 0.58x |
|▫| np.sum (uint32) | uint32 | 1,000 | 0.0021 | 0.0008 | 0.38x |
|✅| np.sum (uint32) | uint32 | 100,000 | 0.0330 | 0.0190 | 0.57x |
|✅| np.sum (uint32) | uint32 | 10,000,000 | 4.2657 | 2.6585 | 0.62x |
|▫| np.sum (uint64) | uint64 | 1,000 | 0.0017 | 0.0007 | 0.42x |
|✅| np.sum (uint64) | uint64 | 100,000 | 0.0189 | 0.0062 | 0.33x |
|✅| np.sum (uint64) | uint64 | 10,000,000 | 4.9134 | 2.8027 | 0.57x |
|▫| np.sum (uint8) | uint8 | 1,000 | 0.0022 | 0.0008 | 0.36x |
|✅| np.sum (uint8) | uint8 | 100,000 | 0.0357 | 0.0186 | 0.52x |
|✅| np.sum (uint8) | uint8 | 10,000,000 | 3.2143 | 1.8386 | 0.57x |
|▫| np.sum axis=0 (float32) | float32 | 1,000 | 0.0019 | 0.0007 | 0.36x |
|✅| np.sum axis=0 (float32) | float32 | 100,000 | 0.0082 | 0.0049 | 0.59x |
|✅| np.sum axis=0 (float32) | float32 | 10,000,000 | 1.5596 | 1.3524 | 0.87x |
|▫| np.sum axis=0 (float64) | float64 | 1,000 | 0.0019 | 0.0007 | 0.37x |
|✅| np.sum axis=0 (float64) | float64 | 100,000 | 0.0139 | 0.0099 | 0.71x |
|✅| np.sum axis=0 (float64) | float64 | 10,000,000 | 3.8823 | 3.2508 | 0.84x |
|🟡| np.sum axis=0 (int16) | int16 | 1,000 | 0.0024 | 0.0033 | 1.38x |
|🔴| np.sum axis=0 (int16) | int16 | 100,000 | 0.0467 | 0.4029 | 8.62x |
|🔴| np.sum axis=0 (int16) | int16 | 10,000,000 | 4.6351 | 58.1349 | 12.54x |
|▫| np.sum axis=0 (int32) | int32 | 1,000 | 0.0025 | 0.0008 | 0.32x |
|✅| np.sum axis=0 (int32) | int32 | 100,000 | 0.0502 | 0.0122 | 0.24x |
|🟡| np.sum axis=0 (int32) | int32 | 10,000,000 | 5.4902 | 6.2016 | 1.13x |
|▫| np.sum axis=0 (int64) | int64 | 1,000 | 0.0020 | 0.0007 | 0.37x |
|✅| np.sum axis=0 (int64) | int64 | 100,000 | 0.0273 | 0.0103 | 0.38x |
|✅| np.sum axis=0 (int64) | int64 | 10,000,000 | 5.3570 | 3.2692 | 0.61x |
|🟡| np.sum axis=0 (uint16) | uint16 | 1,000 | 0.0024 | 0.0048 | 1.99x |
|🔴| np.sum axis=0 (uint16) | uint16 | 100,000 | 0.0472 | 0.5051 | 10.70x |
|🔴| np.sum axis=0 (uint16) | uint16 | 10,000,000 | 4.6196 | 71.6938 | 15.52x |
|▫| np.sum axis=0 (uint32) | uint32 | 1,000 | 0.0024 | 0.0008 | 0.34x |
|✅| np.sum axis=0 (uint32) | uint32 | 100,000 | 0.0508 | 0.0123 | 0.24x |
|🟡| np.sum axis=0 (uint32) | uint32 | 10,000,000 | 5.5936 | 5.9806 | 1.07x |
|▫| np.sum axis=0 (uint64) | uint64 | 1,000 | 0.0020 | 0.0008 | 0.39x |
|✅| np.sum axis=0 (uint64) | uint64 | 100,000 | 0.0274 | 0.0103 | 0.38x |
|✅| np.sum axis=0 (uint64) | uint64 | 10,000,000 | 5.6359 | 3.2594 | 0.58x |
|🟡| np.sum axis=0 (uint8) | uint8 | 1,000 | 0.0025 | 0.0048 | 1.89x |
|🔴| np.sum axis=0 (uint8) | uint8 | 100,000 | 0.0495 | 0.4962 | 10.03x |
|🔴| np.sum axis=0 (uint8) | uint8 | 10,000,000 | 4.4068 | 55.3512 | 12.56x |
|▫| np.sum axis=1 (float32) | float32 | 1,000 | 0.0019 | 0.0008 | 0.42x |
|✅| np.sum axis=1 (float32) | float32 | 100,000 | 0.0161 | 0.0033 | 0.21x |
|✅| np.sum axis=1 (float32) | float32 | 10,000,000 | 3.1946 | 1.0783 | 0.34x |
|▫| np.sum axis=1 (float64) | float64 | 1,000 | 0.0019 | 0.0008 | 0.41x |
|✅| np.sum axis=1 (float64) | float64 | 100,000 | 0.0184 | 0.0069 | 0.38x |
|✅| np.sum axis=1 (float64) | float64 | 10,000,000 | 5.3940 | 2.9397 | 0.55x |
|🟡| np.sum axis=1 (int16) | int16 | 1,000 | 0.0023 | 0.0034 | 1.44x |
|🔴| np.sum axis=1 (int16) | int16 | 100,000 | 0.0372 | 0.4069 | 10.95x |
|🔴| np.sum axis=1 (int16) | int16 | 10,000,000 | 3.3824 | 40.8465 | 12.08x |
|▫| np.sum axis=1 (int32) | int32 | 1,000 | 0.0025 | 0.0009 | 0.37x |
|✅| np.sum axis=1 (int32) | int32 | 100,000 | 0.0397 | 0.0160 | 0.40x |
|✅| np.sum axis=1 (int32) | int32 | 10,000,000 | 4.2000 | 1.8994 | 0.45x |
|▫| np.sum axis=1 (int64) | int64 | 1,000 | 0.0019 | 0.0008 | 0.43x |
|✅| np.sum axis=1 (int64) | int64 | 100,000 | 0.0172 | 0.0074 | 0.43x |
|✅| np.sum axis=1 (int64) | int64 | 10,000,000 | 4.5772 | 2.9075 | 0.64x |
|🟠| np.sum axis=1 (uint16) | uint16 | 1,000 | 0.0023 | 0.0048 | 2.05x |
|🔴| np.sum axis=1 (uint16) | uint16 | 100,000 | 0.0402 | 0.4973 | 12.38x |
|🔴| np.sum axis=1 (uint16) | uint16 | 10,000,000 | 3.3650 | 49.8962 | 14.83x |
|✅| np.sum axis=1 (uint32) | uint32 | 1,000 | 0.0023 | 0.0011 | 0.46x |
|🟡| np.sum axis=1 (uint32) | uint32 | 100,000 | 0.0382 | 0.0402 | 1.05x |
|✅| np.sum axis=1 (uint32) | uint32 | 10,000,000 | 4.3360 | 4.0859 | 0.94x |
|▫| np.sum axis=1 (uint64) | uint64 | 1,000 | 0.0019 | 0.0007 | 0.37x |
|✅| np.sum axis=1 (uint64) | uint64 | 100,000 | 0.0182 | 0.0074 | 0.41x |
|✅| np.sum axis=1 (uint64) | uint64 | 10,000,000 | 5.0466 | 2.9711 | 0.59x |
|🟡| np.sum axis=1 (uint8) | uint8 | 1,000 | 0.0025 | 0.0048 | 1.92x |
|🔴| np.sum axis=1 (uint8) | uint8 | 100,000 | 0.0372 | 0.5005 | 13.45x |
|🔴| np.sum axis=1 (uint8) | uint8 | 10,000,000 | 3.1152 | 49.7407 | 15.97x |
|▫| np.var (float32) | float32 | 1,000 | 0.0078 | 0.0008 | 0.10x |
|✅| np.var (float32) | float32 | 100,000 | 0.0477 | 0.0096 | 0.20x |
|✅| np.var (float32) | float32 | 10,000,000 | 16.9566 | 2.6033 | 0.15x |
|▫| np.var (float64) | float64 | 1,000 | 0.0063 | 0.0008 | 0.12x |
|✅| np.var (float64) | float64 | 100,000 | 0.0557 | 0.0190 | 0.34x |
|✅| np.var (float64) | float64 | 10,000,000 | 31.7478 | 6.7135 | 0.21x |

### Broadcasting

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| matrix + col_vector (N,M)+(N,1) | float64 | 1,000 | 0.0015 | - | - |
|⚪| matrix + col_vector (N,M)+(N,1) | float64 | 100,000 | 0.0300 | - | - |
|✅| matrix + col_vector (N,M)+(N,1) | float64 | 10,000,000 | 16.7419 | 14.4534 | 0.86x |
|⚪| matrix + row_vector (N,M)+(M,) | float64 | 1,000 | 0.0011 | - | - |
|⚪| matrix + row_vector (N,M)+(M,) | float64 | 100,000 | 0.0286 | - | - |
|✅| matrix + row_vector (N,M)+(M,) | float64 | 10,000,000 | 16.9725 | 13.4835 | 0.79x |
|⚪| matrix + scalar | float64 | 1,000 | 0.0007 | - | - |
|⚪| matrix + scalar | float64 | 100,000 | 0.0132 | - | - |
|✅| matrix + scalar | float64 | 10,000,000 | 17.0427 | 13.6337 | 0.80x |
|⚪| np.broadcast_to(row, (N,M)) | float64 | 1,000 | 0.0018 | - | - |
|⚪| np.broadcast_to(row, (N,M)) | float64 | 100,000 | 0.0018 | - | - |
|▫| np.broadcast_to(row, (N,M)) | float64 | 10,000,000 | 0.0019 | 0.0006 | 0.31x |

### Creation

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.copy (float32) | float32 | 1,000 | 0.0006 | 0.0097 | 16.27x |
|🟠| np.copy (float32) | float32 | 100,000 | 0.0060 | 0.0179 | 2.99x |
|✅| np.copy (float32) | float32 | 10,000,000 | 9.3184 | 5.4432 | 0.58x |
|▫| np.copy (float64) | float64 | 1,000 | 0.0006 | 0.0201 | 33.78x |
|🟠| np.copy (float64) | float64 | 100,000 | 0.0114 | 0.0325 | 2.85x |
|✅| np.copy (float64) | float64 | 10,000,000 | 18.5103 | 11.0427 | 0.60x |
|▫| np.copy (int32) | int32 | 1,000 | 0.0006 | 0.0087 | 14.88x |
|🟠| np.copy (int32) | int32 | 100,000 | 0.0061 | 0.0233 | 3.83x |
|✅| np.copy (int32) | int32 | 10,000,000 | 6.6283 | 5.4291 | 0.82x |
|▫| np.copy (int64) | int64 | 1,000 | 0.0006 | 0.0192 | 31.76x |
|🟠| np.copy (int64) | int64 | 100,000 | 0.0114 | 0.0358 | 3.14x |
|✅| np.copy (int64) | int64 | 10,000,000 | 18.5224 | 11.1762 | 0.60x |
|▫| np.empty (float32) | float32 | 1,000 | 0.0003 | 0.0076 | 23.03x |
|▫| np.empty (float32) | float32 | 100,000 | 0.0003 | 0.0049 | 14.79x |
|✅| np.empty (float32) | float32 | 10,000,000 | 0.0196 | 0.0077 | 0.39x |
|▫| np.empty (float64) | float64 | 1,000 | 0.0003 | 0.0083 | 27.82x |
|▫| np.empty (float64) | float64 | 100,000 | 0.0003 | 0.0061 | 21.20x |
|⚪| np.empty (float64) | float64 | 10,000,000 | 0.0095 | - | - |
|▫| np.empty (int32) | int32 | 1,000 | 0.0003 | 0.0076 | 23.08x |
|▫| np.empty (int32) | int32 | 100,000 | 0.0003 | 0.0126 | 40.73x |
|✅| np.empty (int32) | int32 | 10,000,000 | 0.0106 | 0.0066 | 0.62x |
|▫| np.empty (int64) | int64 | 1,000 | 0.0003 | 0.0101 | 30.74x |
|▫| np.empty (int64) | int64 | 100,000 | 0.0003 | 0.0092 | 28.37x |
|⚪| np.empty (int64) | int64 | 10,000,000 | 0.0209 | - | - |
|🔴| np.full (float32) | float32 | 1,000 | 0.0010 | 0.0070 | 6.91x |
|🟠| np.full (float32) | float32 | 100,000 | 0.0054 | 0.0176 | 3.27x |
|✅| np.full (float32) | float32 | 10,000,000 | 9.6279 | 5.7904 | 0.60x |
|▫| np.full (float64) | float64 | 1,000 | 0.0009 | 0.0122 | 13.84x |
|🟠| np.full (float64) | float64 | 100,000 | 0.0097 | 0.0300 | 3.09x |
|✅| np.full (float64) | float64 | 10,000,000 | 18.7710 | 10.9587 | 0.58x |
|▫| np.full (int32) | int32 | 1,000 | 0.0009 | 0.0083 | 9.27x |
|🟠| np.full (int32) | int32 | 100,000 | 0.0054 | 0.0204 | 3.77x |
|✅| np.full (int32) | int32 | 10,000,000 | 7.5837 | 5.6049 | 0.74x |
|▫| np.full (int64) | int64 | 1,000 | 0.0009 | 0.0128 | 14.83x |
|🟠| np.full (int64) | int64 | 100,000 | 0.0098 | 0.0300 | 3.05x |
|✅| np.full (int64) | int64 | 10,000,000 | 18.6308 | 10.6743 | 0.57x |
|▫| np.ones (float32) | float32 | 1,000 | 0.0010 | 0.0089 | 8.99x |
|🟠| np.ones (float32) | float32 | 100,000 | 0.0052 | 0.0173 | 3.31x |
|✅| np.ones (float32) | float32 | 10,000,000 | 9.3400 | 5.8107 | 0.62x |
|▫| np.ones (float64) | float64 | 1,000 | 0.0009 | 0.0141 | 16.33x |
|🟠| np.ones (float64) | float64 | 100,000 | 0.0097 | 0.0291 | 2.99x |
|✅| np.ones (float64) | float64 | 10,000,000 | 18.3874 | 10.9115 | 0.59x |
|▫| np.ones (int32) | int32 | 1,000 | 0.0009 | 0.0089 | 10.05x |
|🟠| np.ones (int32) | int32 | 100,000 | 0.0054 | 0.0186 | 3.42x |
|✅| np.ones (int32) | int32 | 10,000,000 | 7.4068 | 5.6583 | 0.76x |
|▫| np.ones (int64) | int64 | 1,000 | 0.0008 | 0.0152 | 17.96x |
|🟠| np.ones (int64) | int64 | 100,000 | 0.0097 | 0.0297 | 3.04x |
|✅| np.ones (int64) | int64 | 10,000,000 | 15.1807 | 10.6316 | 0.70x |
|▫| np.zeros (float32) | float32 | 1,000 | 0.0004 | 0.0084 | 19.65x |
|🟠| np.zeros (float32) | float32 | 100,000 | 0.0048 | 0.0176 | 3.63x |
|🔴| np.zeros (float32) | float32 | 10,000,000 | 0.0170 | 5.6725 | 334.03x |
|▫| np.zeros (float64) | float64 | 1,000 | 0.0004 | 0.0097 | 26.85x |
|🟠| np.zeros (float64) | float64 | 100,000 | 0.0094 | 0.0305 | 3.25x |
|🔴| np.zeros (float64) | float64 | 10,000,000 | 0.0212 | 10.7550 | 507.65x |
|▫| np.zeros (int32) | int32 | 1,000 | 0.0004 | 0.0081 | 20.60x |
|🟠| np.zeros (int32) | int32 | 100,000 | 0.0051 | 0.0186 | 3.67x |
|🔴| np.zeros (int32) | int32 | 10,000,000 | 0.0108 | 5.6225 | 518.20x |
|▫| np.zeros (int64) | int64 | 1,000 | 0.0004 | 0.0094 | 24.68x |
|🟠| np.zeros (int64) | int64 | 100,000 | 0.0093 | 0.0303 | 3.25x |
|🔴| np.zeros (int64) | int64 | 10,000,000 | 0.0122 | 10.7466 | 879.57x |
|🔴| np.zeros_like (float32) | float32 | 1,000 | 0.0010 | 0.0093 | 8.88x |
|🟠| np.zeros_like (float32) | float32 | 100,000 | 0.0054 | 0.0173 | 3.20x |
|✅| np.zeros_like (float32) | float32 | 10,000,000 | 9.3810 | 5.6115 | 0.60x |
|▫| np.zeros_like (float64) | float64 | 1,000 | 0.0010 | 0.0087 | 8.66x |
|🟠| np.zeros_like (float64) | float64 | 100,000 | 0.0099 | 0.0311 | 3.15x |
|✅| np.zeros_like (float64) | float64 | 10,000,000 | 18.4492 | 10.7073 | 0.58x |
|🟠| np.zeros_like (int32) | int32 | 1,000 | 0.0012 | 0.0056 | 4.55x |
|🟠| np.zeros_like (int32) | int32 | 100,000 | 0.0054 | 0.0179 | 3.29x |
|✅| np.zeros_like (int32) | int32 | 10,000,000 | 7.6603 | 5.6188 | 0.73x |
|🔴| np.zeros_like (int64) | int64 | 1,000 | 0.0010 | 0.0086 | 8.19x |
|🟠| np.zeros_like (int64) | int64 | 100,000 | 0.0102 | 0.0316 | 3.11x |
|✅| np.zeros_like (int64) | int64 | 10,000,000 | 19.3508 | 10.7454 | 0.56x |

### Manipulation

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| a.T (2D) | float64 | 1,000 | 0.0001 | - | - |
|⚪| a.T (2D) | float64 | 100,000 | 0.0001 | - | - |
|⚪| a.T (2D) | float64 | 10,000,000 | 0.0001 | - | - |
|⚪| a.flatten | float64 | 1,000 | 0.0005 | - | - |
|⚪| a.flatten | float64 | 100,000 | 0.0113 | - | - |
|⚪| a.flatten | float64 | 10,000,000 | 13.5031 | - | - |
|⚪| np.concatenate | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.concatenate | float64 | 100,000 | 0.3073 | - | - |
|⚪| np.concatenate | float64 | 10,000,000 | 39.3042 | - | - |
|⚪| np.ravel | float64 | 1,000 | 0.0004 | - | - |
|▫| np.ravel | float64 | 100,000 | 0.0003 | 0.0006 | 1.63x |
|▫| np.ravel | float64 | 10,000,000 | 0.0003 | 0.0005 | 1.44x |
|⚪| np.stack | float64 | 1,000 | 0.0021 | - | - |
|⚪| np.stack | float64 | 100,000 | 0.3302 | - | - |
|⚪| np.stack | float64 | 10,000,000 | 44.9542 | - | - |
|⚪| np.transpose (2D) | float64 | 1,000 | 0.0004 | - | - |
|⚪| np.transpose (2D) | float64 | 100,000 | 0.0004 | - | - |
|⚪| np.transpose (2D) | float64 | 10,000,000 | 0.0004 | - | - |
|⚪| reshape 1D->2D | float64 | 1,000 | 0.0002 | - | - |
|⚪| reshape 1D->2D | float64 | 100,000 | 0.0002 | - | - |
|⚪| reshape 1D->2D | float64 | 10,000,000 | 0.0002 | - | - |
|⚪| reshape 2D->1D | float64 | 1,000 | 0.0002 | - | - |
|⚪| reshape 2D->1D | float64 | 100,000 | 0.0002 | - | - |
|⚪| reshape 2D->1D | float64 | 10,000,000 | 0.0002 | - | - |

### Slicing

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| a[100:1000] (contiguous) | float64 | 1,000 | 0.0002 | - | - |
|⚪| a[100:1000] (contiguous) | float64 | 100,000 | 0.0002 | - | - |
|⚪| a[100:1000] (contiguous) | float64 | 10,000,000 | 0.0001 | - | - |
|⚪| a[::-1] (reversed) | float64 | 1,000 | 0.0001 | - | - |
|▫| a[::-1] (reversed) | float64 | 100,000 | 0.0001 | 0.0012 | 7.72x |
|▫| a[::-1] (reversed) | float64 | 10,000,000 | 0.0001 | 0.0013 | 9.77x |
|⚪| a[::2] (strided) | float64 | 1,000 | 0.0002 | - | - |
|⚪| a[::2] (strided) | float64 | 100,000 | 0.0001 | - | - |
|⚪| a[::2] (strided) | float64 | 10,000,000 | 0.0002 | - | - |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0016 | - | - |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0016 | - | - |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0017 | - | - |
|⚪| np.sum(strided_slice) | float64 | 500 | 0.0017 | - | - |
|⚪| np.sum(strided_slice) | float64 | 50,000 | 0.0100 | - | - |
|⚪| np.sum(strided_slice) | float64 | 5,000,000 | 4.9331 | - | - |

### Comparison

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| a != b (float32) | float32 | 1,000 | 0.0005 | 0.0007 | 1.38x |
|🟠| a != b (float32) | float32 | 100,000 | 0.0055 | 0.0161 | 2.95x |
|✅| a != b (float32) | float32 | 10,000,000 | 9.7856 | 4.0476 | 0.41x |
|▫| a != b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.51x |
|🟠| a != b (float64) | float64 | 100,000 | 0.0102 | 0.0229 | 2.24x |
|✅| a != b (float64) | float64 | 10,000,000 | 18.4551 | 6.6072 | 0.36x |
|▫| a != b (int32) | int32 | 1,000 | 0.0005 | 0.0007 | 1.30x |
|🟠| a != b (int32) | int32 | 100,000 | 0.0070 | 0.0181 | 2.58x |
|✅| a != b (int32) | int32 | 10,000,000 | 4.6312 | 4.0588 | 0.88x |
|▫| a != b (int64) | int64 | 1,000 | 0.0005 | 0.0007 | 1.56x |
|🟠| a != b (int64) | int64 | 100,000 | 0.0126 | 0.0276 | 2.20x |
|✅| a != b (int64) | int64 | 10,000,000 | 7.2078 | 6.5913 | 0.91x |
|▫| a < b (float32) | float32 | 1,000 | 0.0004 | 0.0007 | 1.69x |
|🟠| a < b (float32) | float32 | 100,000 | 0.0056 | 0.0143 | 2.56x |
|✅| a < b (float32) | float32 | 10,000,000 | 10.3517 | 3.9582 | 0.38x |
|▫| a < b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.65x |
|🟠| a < b (float64) | float64 | 100,000 | 0.0107 | 0.0228 | 2.13x |
|✅| a < b (float64) | float64 | 10,000,000 | 18.6797 | 6.4870 | 0.35x |
|▫| a < b (int32) | int32 | 1,000 | 0.0004 | 0.0007 | 1.62x |
|🟠| a < b (int32) | int32 | 100,000 | 0.0069 | 0.0168 | 2.42x |
|✅| a < b (int32) | int32 | 10,000,000 | 4.5830 | 3.9640 | 0.86x |
|▫| a < b (int64) | int64 | 1,000 | 0.0005 | 0.0007 | 1.28x |
|🟡| a < b (int64) | int64 | 100,000 | 0.0179 | 0.0260 | 1.46x |
|✅| a < b (int64) | int64 | 10,000,000 | 13.4370 | 6.6691 | 0.50x |
|▫| a <= b (float32) | float32 | 1,000 | 0.0004 | 0.0008 | 1.94x |
|🟠| a <= b (float32) | float32 | 100,000 | 0.0057 | 0.0135 | 2.38x |
|✅| a <= b (float32) | float32 | 10,000,000 | 10.5595 | 3.9351 | 0.37x |
|▫| a <= b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.60x |
|🟡| a <= b (float64) | float64 | 100,000 | 0.0105 | 0.0204 | 1.95x |
|✅| a <= b (float64) | float64 | 10,000,000 | 18.1656 | 6.4961 | 0.36x |
|▫| a <= b (int32) | int32 | 1,000 | 0.0004 | 0.0006 | 1.52x |
|🟠| a <= b (int32) | int32 | 100,000 | 0.0071 | 0.0169 | 2.39x |
|✅| a <= b (int32) | int32 | 10,000,000 | 4.4351 | 4.1131 | 0.93x |
|▫| a <= b (int64) | int64 | 1,000 | 0.0006 | 0.0007 | 1.23x |
|🟡| a <= b (int64) | int64 | 100,000 | 0.0180 | 0.0276 | 1.53x |
|✅| a <= b (int64) | int64 | 10,000,000 | 18.9495 | 6.8388 | 0.36x |
|▫| a == b (float32) | float32 | 1,000 | 0.0004 | 0.0007 | 1.57x |
|🟠| a == b (float32) | float32 | 100,000 | 0.0055 | 0.0162 | 2.98x |
|✅| a == b (float32) | float32 | 10,000,000 | 10.4596 | 3.9621 | 0.38x |
|▫| a == b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.60x |
|🟠| a == b (float64) | float64 | 100,000 | 0.0101 | 0.0248 | 2.46x |
|✅| a == b (float64) | float64 | 10,000,000 | 18.0360 | 6.5660 | 0.36x |
|▫| a == b (int32) | int32 | 1,000 | 0.0004 | 0.0007 | 1.72x |
|🟠| a == b (int32) | int32 | 100,000 | 0.0071 | 0.0163 | 2.28x |
|✅| a == b (int32) | int32 | 10,000,000 | 4.5825 | 3.9854 | 0.87x |
|▫| a == b (int64) | int64 | 1,000 | 0.0004 | 0.0007 | 1.47x |
|🟠| a == b (int64) | int64 | 100,000 | 0.0125 | 0.0263 | 2.10x |
|✅| a == b (int64) | int64 | 10,000,000 | 6.9799 | 6.4802 | 0.93x |
|▫| a > b (float32) | float32 | 1,000 | 0.0004 | 0.0007 | 1.65x |
|🟠| a > b (float32) | float32 | 100,000 | 0.0056 | 0.0142 | 2.53x |
|✅| a > b (float32) | float32 | 10,000,000 | 10.1546 | 3.9550 | 0.39x |
|▫| a > b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.53x |
|🟠| a > b (float64) | float64 | 100,000 | 0.0103 | 0.0249 | 2.42x |
|✅| a > b (float64) | float64 | 10,000,000 | 19.3510 | 6.4686 | 0.33x |
|▫| a > b (int32) | int32 | 1,000 | 0.0004 | 0.0007 | 1.57x |
|🟠| a > b (int32) | int32 | 100,000 | 0.0070 | 0.0172 | 2.46x |
|✅| a > b (int32) | int32 | 10,000,000 | 4.2011 | 3.9656 | 0.94x |
|▫| a > b (int64) | int64 | 1,000 | 0.0005 | 0.0007 | 1.29x |
|🟡| a > b (int64) | int64 | 100,000 | 0.0182 | 0.0267 | 1.47x |
|✅| a > b (int64) | int64 | 10,000,000 | 19.0728 | 6.6286 | 0.35x |
|▫| a >= b (float32) | float32 | 1,000 | 0.0005 | 0.0007 | 1.30x |
|🟠| a >= b (float32) | float32 | 100,000 | 0.0057 | 0.0140 | 2.47x |
|✅| a >= b (float32) | float32 | 10,000,000 | 9.9412 | 3.9784 | 0.40x |
|▫| a >= b (float64) | float64 | 1,000 | 0.0004 | 0.0007 | 1.56x |
|🟠| a >= b (float64) | float64 | 100,000 | 0.0103 | 0.0251 | 2.45x |
|✅| a >= b (float64) | float64 | 10,000,000 | 19.0127 | 6.5439 | 0.34x |
|▫| a >= b (int32) | int32 | 1,000 | 0.0004 | 0.0007 | 1.58x |
|🟠| a >= b (int32) | int32 | 100,000 | 0.0070 | 0.0163 | 2.33x |
|✅| a >= b (int32) | int32 | 10,000,000 | 4.4060 | 4.0361 | 0.92x |
|▫| a >= b (int64) | int64 | 1,000 | 0.0005 | 0.0007 | 1.25x |
|🟡| a >= b (int64) | int64 | 100,000 | 0.0185 | 0.0273 | 1.48x |
|✅| a >= b (int64) | int64 | 10,000,000 | 20.9543 | 6.6942 | 0.32x |

### Bitwise

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| a & b (bool) | bool | 1,000 | 0.0004 | 0.0012 | 3.22x |
|🔴| a & b (bool) | bool | 100,000 | 0.0033 | 0.0225 | 6.88x |
|🟡| a & b (bool) | bool | 10,000,000 | 2.0034 | 2.7889 | 1.39x |
|▫| a & b (int16) | int16 | 1,000 | 0.0008 | 0.0046 | 5.96x |
|✅| a & b (int16) | int16 | 100,000 | 0.0295 | 0.0101 | 0.34x |
|✅| a & b (int16) | int16 | 10,000,000 | 9.2627 | 3.7953 | 0.41x |
|▫| a & b (int32) | int32 | 1,000 | 0.0008 | 0.0054 | 6.84x |
|✅| a & b (int32) | int32 | 100,000 | 0.0325 | 0.0211 | 0.65x |
|✅| a & b (int32) | int32 | 10,000,000 | 16.8456 | 7.6036 | 0.45x |
|▫| a & b (int64) | int64 | 1,000 | 0.0008 | 0.0119 | 15.18x |
|🟡| a & b (int64) | int64 | 100,000 | 0.0352 | 0.0449 | 1.27x |
|✅| a & b (int64) | int64 | 10,000,000 | 37.9424 | 14.9278 | 0.39x |
|▫| a & b (uint16) | uint16 | 1,000 | 0.0008 | 0.0040 | 5.08x |
|✅| a & b (uint16) | uint16 | 100,000 | 0.0309 | 0.0105 | 0.34x |
|✅| a & b (uint16) | uint16 | 10,000,000 | 9.5081 | 3.7945 | 0.40x |
|▫| a & b (uint32) | uint32 | 1,000 | 0.0008 | 0.0083 | 10.36x |
|✅| a & b (uint32) | uint32 | 100,000 | 0.0329 | 0.0206 | 0.63x |
|✅| a & b (uint32) | uint32 | 10,000,000 | 18.7328 | 7.6036 | 0.41x |
|▫| a & b (uint64) | uint64 | 1,000 | 0.0008 | 0.0094 | 11.97x |
|🟡| a & b (uint64) | uint64 | 100,000 | 0.0360 | 0.0442 | 1.23x |
|✅| a & b (uint64) | uint64 | 10,000,000 | 39.5694 | 15.0468 | 0.38x |
|▫| a & b (uint8) | uint8 | 1,000 | 0.0008 | 0.0015 | 1.88x |
|✅| a & b (uint8) | uint8 | 100,000 | 0.0285 | 0.0061 | 0.21x |
|✅| a & b (uint8) | uint8 | 10,000,000 | 3.8970 | 1.8615 | 0.48x |
|▫| a ^ b (bool) | bool | 1,000 | 0.0004 | 0.0015 | 3.46x |
|🔴| a ^ b (bool) | bool | 100,000 | 0.0031 | 0.0224 | 7.20x |
|🟡| a ^ b (bool) | bool | 10,000,000 | 1.8487 | 2.8187 | 1.52x |
|▫| a ^ b (int16) | int16 | 1,000 | 0.0008 | 0.0027 | 3.48x |
|✅| a ^ b (int16) | int16 | 100,000 | 0.0285 | 0.0095 | 0.34x |
|✅| a ^ b (int16) | int16 | 10,000,000 | 9.7269 | 3.7538 | 0.39x |
|▫| a ^ b (int32) | int32 | 1,000 | 0.0008 | 0.0081 | 10.44x |
|✅| a ^ b (int32) | int32 | 100,000 | 0.0288 | 0.0218 | 0.76x |
|✅| a ^ b (int32) | int32 | 10,000,000 | 17.4185 | 7.5248 | 0.43x |
|▫| a ^ b (int64) | int64 | 1,000 | 0.0008 | 0.0093 | 12.04x |
|🟡| a ^ b (int64) | int64 | 100,000 | 0.0363 | 0.0453 | 1.25x |
|✅| a ^ b (int64) | int64 | 10,000,000 | 33.1728 | 14.8136 | 0.45x |
|▫| a ^ b (uint16) | uint16 | 1,000 | 0.0008 | 0.0032 | 4.11x |
|✅| a ^ b (uint16) | uint16 | 100,000 | 0.0293 | 0.0107 | 0.37x |
|✅| a ^ b (uint16) | uint16 | 10,000,000 | 9.3024 | 3.7737 | 0.41x |
|▫| a ^ b (uint32) | uint32 | 1,000 | 0.0008 | 0.0060 | 7.68x |
|✅| a ^ b (uint32) | uint32 | 100,000 | 0.0286 | 0.0211 | 0.74x |
|✅| a ^ b (uint32) | uint32 | 10,000,000 | 18.9513 | 7.5649 | 0.40x |
|▫| a ^ b (uint64) | uint64 | 1,000 | 0.0008 | 0.0124 | 15.80x |
|🟡| a ^ b (uint64) | uint64 | 100,000 | 0.0358 | 0.0433 | 1.21x |
|✅| a ^ b (uint64) | uint64 | 10,000,000 | 42.5116 | 15.2940 | 0.36x |
|▫| a ^ b (uint8) | uint8 | 1,000 | 0.0007 | 0.0013 | 1.98x |
|✅| a ^ b (uint8) | uint8 | 100,000 | 0.0285 | 0.0068 | 0.24x |
|✅| a ^ b (uint8) | uint8 | 10,000,000 | 6.5357 | 1.8194 | 0.28x |
|▫| a | b (bool) | bool | 1,000 | 0.0004 | 0.0017 | 4.21x |
|🔴| a | b (bool) | bool | 100,000 | 0.0028 | 0.0238 | 8.41x |
|🟡| a | b (bool) | bool | 10,000,000 | 1.8630 | 3.2311 | 1.73x |
|▫| a | b (int16) | int16 | 1,000 | 0.0008 | 0.0030 | 3.80x |
|✅| a | b (int16) | int16 | 100,000 | 0.0285 | 0.0112 | 0.39x |
|✅| a | b (int16) | int16 | 10,000,000 | 9.4168 | 3.7613 | 0.40x |
|▫| a | b (int32) | int32 | 1,000 | 0.0008 | 0.0083 | 10.50x |
|✅| a | b (int32) | int32 | 100,000 | 0.0303 | 0.0222 | 0.73x |
|✅| a | b (int32) | int32 | 10,000,000 | 16.5894 | 7.5214 | 0.45x |
|▫| a | b (int64) | int64 | 1,000 | 0.0008 | 0.0130 | 16.47x |
|🟡| a | b (int64) | int64 | 100,000 | 0.0366 | 0.0429 | 1.17x |
|✅| a | b (int64) | int64 | 10,000,000 | 36.1764 | 14.8245 | 0.41x |
|▫| a | b (uint16) | uint16 | 1,000 | 0.0008 | 0.0032 | 4.07x |
|✅| a | b (uint16) | uint16 | 100,000 | 0.0286 | 0.0112 | 0.39x |
|✅| a | b (uint16) | uint16 | 10,000,000 | 9.4580 | 3.7895 | 0.40x |
|▫| a | b (uint32) | uint32 | 1,000 | 0.0008 | 0.0065 | 8.20x |
|✅| a | b (uint32) | uint32 | 100,000 | 0.0286 | 0.0194 | 0.68x |
|✅| a | b (uint32) | uint32 | 10,000,000 | 19.7626 | 7.5865 | 0.38x |
|▫| a | b (uint64) | uint64 | 1,000 | 0.0008 | 0.0126 | 15.71x |
|🟡| a | b (uint64) | uint64 | 100,000 | 0.0353 | 0.0452 | 1.28x |
|✅| a | b (uint64) | uint64 | 10,000,000 | 38.8894 | 15.0789 | 0.39x |
|▫| a | b (uint8) | uint8 | 1,000 | 0.0007 | 0.0011 | 1.65x |
|✅| a | b (uint8) | uint8 | 100,000 | 0.0304 | 0.0064 | 0.21x |
|✅| a | b (uint8) | uint8 | 10,000,000 | 4.3234 | 1.8380 | 0.42x |
|▫| np.invert(a) (bool) | bool | 1,000 | 0.0004 | 0.0017 | 4.55x |
|🔴| np.invert(a) (bool) | bool | 100,000 | 0.0026 | 0.0241 | 9.32x |
|🟡| np.invert(a) (bool) | bool | 10,000,000 | 1.6922 | 3.0191 | 1.78x |
|▫| np.invert(a) (int16) | int16 | 1,000 | 0.0007 | 0.0035 | 4.73x |
|✅| np.invert(a) (int16) | int16 | 100,000 | 0.0262 | 0.0102 | 0.39x |
|✅| np.invert(a) (int16) | int16 | 10,000,000 | 7.9035 | 3.3957 | 0.43x |
|▫| np.invert(a) (int32) | int32 | 1,000 | 0.0008 | 0.0075 | 9.64x |
|✅| np.invert(a) (int32) | int32 | 100,000 | 0.0352 | 0.0205 | 0.58x |
|✅| np.invert(a) (int32) | int32 | 10,000,000 | 14.1460 | 7.1424 | 0.51x |
|▫| np.invert(a) (int64) | int64 | 1,000 | 0.0007 | 0.0088 | 12.04x |
|🟡| np.invert(a) (int64) | int64 | 100,000 | 0.0263 | 0.0464 | 1.76x |
|✅| np.invert(a) (int64) | int64 | 10,000,000 | 26.1526 | 13.5612 | 0.52x |
|▫| np.invert(a) (uint16) | uint16 | 1,000 | 0.0007 | 0.0030 | 4.09x |
|✅| np.invert(a) (uint16) | uint16 | 100,000 | 0.0360 | 0.0103 | 0.28x |
|✅| np.invert(a) (uint16) | uint16 | 10,000,000 | 6.7184 | 3.4013 | 0.51x |
|▫| np.invert(a) (uint32) | uint32 | 1,000 | 0.0008 | 0.0100 | 12.92x |
|✅| np.invert(a) (uint32) | uint32 | 100,000 | 0.0338 | 0.0200 | 0.59x |
|✅| np.invert(a) (uint32) | uint32 | 10,000,000 | 13.9398 | 6.9702 | 0.50x |
|▫| np.invert(a) (uint64) | uint64 | 1,000 | 0.0007 | 0.0098 | 13.45x |
|🟡| np.invert(a) (uint64) | uint64 | 100,000 | 0.0262 | 0.0387 | 1.48x |
|✅| np.invert(a) (uint64) | uint64 | 10,000,000 | 33.5033 | 13.6426 | 0.41x |
|▫| np.invert(a) (uint8) | uint8 | 1,000 | 0.0006 | 0.0013 | 2.18x |
|✅| np.invert(a) (uint8) | uint8 | 100,000 | 0.0258 | 0.0069 | 0.27x |
|✅| np.invert(a) (uint8) | uint8 | 10,000,000 | 5.7393 | 1.6899 | 0.29x |
|⚪| np.left_shift(a, 2) (bool) | bool | 1,000 | 0.0015 | - | - |
|⚪| np.left_shift(a, 2) (bool) | bool | 100,000 | 0.1933 | - | - |
|⚪| np.left_shift(a, 2) (bool) | bool | 10,000,000 | 15.1279 | - | - |
|🔴| np.left_shift(a, 2) (int16) | int16 | 1,000 | 0.0011 | 0.0101 | 9.55x |
|🟠| np.left_shift(a, 2) (int16) | int16 | 100,000 | 0.0289 | 0.0641 | 2.21x |
|🟡| np.left_shift(a, 2) (int16) | int16 | 10,000,000 | 7.5456 | 11.1718 | 1.48x |
|▫| np.left_shift(a, 2) (int32) | int32 | 1,000 | 0.0010 | 0.0144 | 14.48x |
|🟠| np.left_shift(a, 2) (int32) | int32 | 100,000 | 0.0192 | 0.0657 | 3.42x |
|✅| np.left_shift(a, 2) (int32) | int32 | 10,000,000 | 14.7614 | 13.8055 | 0.94x |
|🔴| np.left_shift(a, 2) (int64) | int64 | 1,000 | 0.0010 | 0.0199 | 19.11x |
|🟠| np.left_shift(a, 2) (int64) | int64 | 100,000 | 0.0199 | 0.0806 | 4.04x |
|✅| np.left_shift(a, 2) (int64) | int64 | 10,000,000 | 25.6758 | 19.0870 | 0.74x |
|🔴| np.left_shift(a, 2) (uint16) | uint16 | 1,000 | 0.0011 | 0.0103 | 9.67x |
|🟠| np.left_shift(a, 2) (uint16) | uint16 | 100,000 | 0.0295 | 0.0635 | 2.15x |
|🟡| np.left_shift(a, 2) (uint16) | uint16 | 10,000,000 | 7.6829 | 11.1922 | 1.46x |
|▫| np.left_shift(a, 2) (uint32) | uint32 | 1,000 | 0.0010 | 0.0089 | 8.93x |
|🟠| np.left_shift(a, 2) (uint32) | uint32 | 100,000 | 0.0198 | 0.0654 | 3.30x |
|✅| np.left_shift(a, 2) (uint32) | uint32 | 10,000,000 | 15.2253 | 13.5976 | 0.89x |
|▫| np.left_shift(a, 2) (uint64) | uint64 | 1,000 | 0.0010 | 0.0243 | 24.86x |
|🟠| np.left_shift(a, 2) (uint64) | uint64 | 100,000 | 0.0191 | 0.0732 | 3.83x |
|✅| np.left_shift(a, 2) (uint64) | uint64 | 10,000,000 | 34.3965 | 19.0897 | 0.56x |
|▫| np.left_shift(a, 2) (uint8) | uint8 | 1,000 | 0.0009 | 0.0057 | 6.22x |
|🟠| np.left_shift(a, 2) (uint8) | uint8 | 100,000 | 0.0282 | 0.0632 | 2.24x |
|🟡| np.left_shift(a, 2) (uint8) | uint8 | 10,000,000 | 6.2092 | 10.3006 | 1.66x |
|⚪| np.right_shift(a, 2) (bool) | bool | 1,000 | 0.0016 | - | - |
|⚪| np.right_shift(a, 2) (bool) | bool | 100,000 | 0.1884 | - | - |
|⚪| np.right_shift(a, 2) (bool) | bool | 10,000,000 | 15.2908 | - | - |
|🔴| np.right_shift(a, 2) (int16) | int16 | 1,000 | 0.0012 | 0.0088 | 7.54x |
|🟡| np.right_shift(a, 2) (int16) | int16 | 100,000 | 0.0375 | 0.0661 | 1.76x |
|🟡| np.right_shift(a, 2) (int16) | int16 | 10,000,000 | 9.3594 | 11.1885 | 1.20x |
|🔴| np.right_shift(a, 2) (int32) | int32 | 1,000 | 0.0011 | 0.0172 | 15.61x |
|🟠| np.right_shift(a, 2) (int32) | int32 | 100,000 | 0.0284 | 0.0664 | 2.34x |
|✅| np.right_shift(a, 2) (int32) | int32 | 10,000,000 | 15.3178 | 13.4878 | 0.88x |
|🔴| np.right_shift(a, 2) (int64) | int64 | 1,000 | 0.0010 | 0.0195 | 19.20x |
|🟠| np.right_shift(a, 2) (int64) | int64 | 100,000 | 0.0296 | 0.0775 | 2.62x |
|✅| np.right_shift(a, 2) (int64) | int64 | 10,000,000 | 31.6268 | 19.1637 | 0.61x |
|🔴| np.right_shift(a, 2) (uint16) | uint16 | 1,000 | 0.0012 | 0.0095 | 7.74x |
|🟠| np.right_shift(a, 2) (uint16) | uint16 | 100,000 | 0.0294 | 0.0644 | 2.19x |
|🟡| np.right_shift(a, 2) (uint16) | uint16 | 10,000,000 | 7.1666 | 11.3780 | 1.59x |
|▫| np.right_shift(a, 2) (uint32) | uint32 | 1,000 | 0.0010 | 0.0151 | 15.13x |
|🟠| np.right_shift(a, 2) (uint32) | uint32 | 100,000 | 0.0195 | 0.0665 | 3.41x |
|✅| np.right_shift(a, 2) (uint32) | uint32 | 10,000,000 | 14.9491 | 13.5401 | 0.91x |
|▫| np.right_shift(a, 2) (uint64) | uint64 | 1,000 | 0.0010 | 0.0156 | 15.71x |
|🟠| np.right_shift(a, 2) (uint64) | uint64 | 100,000 | 0.0309 | 0.0724 | 2.35x |
|✅| np.right_shift(a, 2) (uint64) | uint64 | 10,000,000 | 32.6266 | 19.1042 | 0.59x |
|▫| np.right_shift(a, 2) (uint8) | uint8 | 1,000 | 0.0009 | 0.0079 | 8.66x |
|🟠| np.right_shift(a, 2) (uint8) | uint8 | 100,000 | 0.0284 | 0.0645 | 2.27x |
|🟡| np.right_shift(a, 2) (uint8) | uint8 | 10,000,000 | 6.3306 | 10.3850 | 1.64x |

### Logic

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| np.all(a) (bool) | bool | 1,000 | 0.0014 | - | - |
|⚪| np.all(a) (bool) | bool | 100,000 | 0.0014 | - | - |
|⚪| np.all(a) (bool) | bool | 10,000,000 | 0.0039 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 1,000 | 0.0134 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 100,000 | 0.0794 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 10,000,000 | 103.4373 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 1,000 | 0.0139 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 100,000 | 0.7088 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 10,000,000 | 186.0116 | - | - |
|⚪| np.any(a) (bool) | bool | 1,000 | 0.0014 | - | - |
|⚪| np.any(a) (bool) | bool | 100,000 | 0.0014 | - | - |
|⚪| np.any(a) (bool) | bool | 10,000,000 | 0.0041 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 1,000 | 0.0023 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 100,000 | 0.0067 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 10,000,000 | 10.8635 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 1,000 | 0.0018 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 100,000 | 0.0133 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 10,000,000 | 19.7996 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 1,000 | 0.0118 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 100,000 | 0.0781 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 10,000,000 | 98.4756 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 1,000 | 0.0120 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 100,000 | 0.7128 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 10,000,000 | 187.4282 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 1,000 | 0.0004 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 100,000 | 0.0052 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 10,000,000 | 3.6660 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 1,000 | 0.0005 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 100,000 | 0.0104 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 10,000,000 | 10.9932 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 1,000 | 0.0004 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 100,000 | 0.0053 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 10,000,000 | 3.7905 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 1,000 | 0.0005 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 100,000 | 0.0101 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 10,000,000 | 12.2420 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 1,000 | 0.0005 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 100,000 | 0.0042 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 10,000,000 | 3.8648 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 1,000 | 0.0005 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 100,000 | 0.0086 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 10,000,000 | 11.1539 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 1,000 | 0.0006 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 100,000 | 0.0071 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 10,000,000 | 8.9753 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 1,000 | 0.0008 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 100,000 | 0.0301 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 10,000,000 | 33.1903 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 1,000 | 0.0006 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 100,000 | 0.0074 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 10,000,000 | 9.0843 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 1,000 | 0.0006 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 100,000 | 0.0293 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 10,000,000 | 32.3178 | - | - |

### Statistics

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.average(a) (float32) | float32 | 1,000 | 0.0043 | 0.0007 | 0.15x |
|✅| np.average(a) (float32) | float32 | 100,000 | 0.0177 | 0.0021 | 0.12x |
|✅| np.average(a) (float32) | float32 | 10,000,000 | 9.5978 | 0.9373 | 0.10x |
|▫| np.average(a) (float64) | float64 | 1,000 | 0.0029 | 0.0006 | 0.23x |
|✅| np.average(a) (float64) | float64 | 100,000 | 0.0181 | 0.0040 | 0.22x |
|✅| np.average(a) (float64) | float64 | 10,000,000 | 17.2902 | 2.5462 | 0.15x |
|▫| np.count_nonzero(a) (float32) | float32 | 1,000 | 0.0008 | 0.0001 | 0.10x |
|✅| np.count_nonzero(a) (float32) | float32 | 100,000 | 0.0379 | 0.0046 | 0.12x |
|✅| np.count_nonzero(a) (float32) | float32 | 10,000,000 | 8.0124 | 1.5426 | 0.19x |
|▫| np.count_nonzero(a) (float64) | float64 | 1,000 | 0.0006 | 0.0001 | 0.19x |
|✅| np.count_nonzero(a) (float64) | float64 | 100,000 | 0.0381 | 0.0088 | 0.23x |
|✅| np.count_nonzero(a) (float64) | float64 | 10,000,000 | 22.6049 | 3.7369 | 0.17x |
|✅| np.median(a) (float32) | float32 | 1,000 | 0.0110 | 0.0024 | 0.22x |
|🟡| np.median(a) (float32) | float32 | 100,000 | 0.4716 | 0.7425 | 1.57x |
|✅| np.median(a) (float32) | float32 | 10,000,000 | 87.7171 | 85.5722 | 0.98x |
|✅| np.median(a) (float64) | float64 | 1,000 | 0.0098 | 0.0023 | 0.24x |
|🟡| np.median(a) (float64) | float64 | 100,000 | 0.4704 | 0.7067 | 1.50x |
|✅| np.median(a) (float64) | float64 | 10,000,000 | 113.1357 | 87.8338 | 0.78x |
|✅| np.percentile(a, 50) (float32) | float32 | 1,000 | 0.0248 | 0.0024 | 0.10x |
|🟡| np.percentile(a, 50) (float32) | float32 | 100,000 | 0.7319 | 0.7428 | 1.01x |
|🟡| np.percentile(a, 50) (float32) | float32 | 10,000,000 | 68.3265 | 85.4781 | 1.25x |
|✅| np.percentile(a, 50) (float64) | float64 | 1,000 | 0.0245 | 0.0023 | 0.10x |
|✅| np.percentile(a, 50) (float64) | float64 | 100,000 | 0.7119 | 0.7082 | 0.99x |
|🟡| np.percentile(a, 50) (float64) | float64 | 10,000,000 | 82.2651 | 87.7597 | 1.07x |
|✅| np.ptp(a) (float32) | float32 | 1,000 | 0.0031 | 0.0020 | 0.63x |
|🟡| np.ptp(a) (float32) | float32 | 100,000 | 0.0140 | 0.0275 | 1.97x |
|✅| np.ptp(a) (float32) | float32 | 10,000,000 | 7.7188 | 3.4002 | 0.44x |
|✅| np.ptp(a) (float64) | float64 | 1,000 | 0.0033 | 0.0026 | 0.77x |
|🟠| np.ptp(a) (float64) | float64 | 100,000 | 0.0198 | 0.0528 | 2.67x |
|✅| np.ptp(a) (float64) | float64 | 10,000,000 | 18.9636 | 10.1399 | 0.54x |
|✅| np.quantile(a, 0.5) (float32) | float32 | 1,000 | 0.0240 | 0.0024 | 0.10x |
|🟡| np.quantile(a, 0.5) (float32) | float32 | 100,000 | 0.6880 | 0.7436 | 1.08x |
|🟡| np.quantile(a, 0.5) (float32) | float32 | 10,000,000 | 64.1916 | 85.6317 | 1.33x |
|✅| np.quantile(a, 0.5) (float64) | float64 | 1,000 | 0.0232 | 0.0023 | 0.10x |
|🟡| np.quantile(a, 0.5) (float64) | float64 | 100,000 | 0.7042 | 0.7066 | 1.00x |
|🟡| np.quantile(a, 0.5) (float64) | float64 | 10,000,000 | 86.1589 | 87.6596 | 1.02x |

### Sorting

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🔴| np.argsort(a) (float32) | float32 | 1,000 | 0.0118 | 0.0691 | 5.88x |
|🔴| np.argsort(a) (float32) | float32 | 100,000 | 1.5577 | 12.9884 | 8.34x |
|🟡| np.argsort(a) (float32) | float32 | 10,000,000 | 1524.6232 | 2861.3197 | 1.88x |
|🔴| np.argsort(a) (float64) | float64 | 1,000 | 0.0105 | 0.0707 | 6.76x |
|🔴| np.argsort(a) (float64) | float64 | 100,000 | 1.4217 | 13.4715 | 9.47x |
|🟡| np.argsort(a) (float64) | float64 | 10,000,000 | 2030.5666 | 3133.5306 | 1.54x |
|🟠| np.argsort(a) (int32) | int32 | 1,000 | 0.0118 | 0.0387 | 3.27x |
|🔴| np.argsort(a) (int32) | int32 | 100,000 | 0.4419 | 10.4036 | 23.54x |
|🔴| np.argsort(a) (int32) | int32 | 10,000,000 | 368.7844 | 2162.0888 | 5.86x |
|🟠| np.argsort(a) (int64) | int64 | 1,000 | 0.0132 | 0.0594 | 4.51x |
|🔴| np.argsort(a) (int64) | int64 | 100,000 | 0.4716 | 12.8928 | 27.34x |
|🟠| np.argsort(a) (int64) | int64 | 10,000,000 | 572.7782 | 2835.7755 | 4.95x |
|✅| np.nonzero(a) (float32) | float32 | 1,000 | 0.0027 | 0.0021 | 0.77x |
|✅| np.nonzero(a) (float32) | float32 | 100,000 | 0.1952 | 0.0853 | 0.44x |
|✅| np.nonzero(a) (float32) | float32 | 10,000,000 | 43.6328 | 18.7015 | 0.43x |
|✅| np.nonzero(a) (float64) | float64 | 1,000 | 0.0028 | 0.0022 | 0.78x |
|✅| np.nonzero(a) (float64) | float64 | 100,000 | 0.1868 | 0.0927 | 0.50x |
|✅| np.nonzero(a) (float64) | float64 | 10,000,000 | 56.0464 | 21.9806 | 0.39x |
|🟡| np.nonzero(a) (int32) | int32 | 1,000 | 0.0017 | 0.0021 | 1.19x |
|✅| np.nonzero(a) (int32) | int32 | 100,000 | 0.1037 | 0.0842 | 0.81x |
|✅| np.nonzero(a) (int32) | int32 | 10,000,000 | 32.4047 | 18.6123 | 0.57x |
|🟡| np.nonzero(a) (int64) | int64 | 1,000 | 0.0018 | 0.0022 | 1.25x |
|✅| np.nonzero(a) (int64) | int64 | 100,000 | 0.1045 | 0.0973 | 0.93x |
|✅| np.nonzero(a) (int64) | int64 | 10,000,000 | 57.7190 | 22.4015 | 0.39x |
|▫| np.searchsorted(a, v) (float32) | float32 | 1,000 | 0.0015 | 0.0000 | 0.01x |
|▫| np.searchsorted(a, v) (float32) | float32 | 100,000 | 0.0239 | 0.0000 | 0.00x |
|▫| np.searchsorted(a, v) (float32) | float32 | 10,000,000 | 22.9514 | 0.0000 | 0.00x |
|▫| np.searchsorted(a, v) (float64) | float64 | 1,000 | 0.0009 | 0.0000 | 0.02x |
|▫| np.searchsorted(a, v) (float64) | float64 | 100,000 | 0.0011 | 0.0000 | 0.02x |
|▫| np.searchsorted(a, v) (float64) | float64 | 10,000,000 | 0.0028 | 0.0000 | 0.01x |
|▫| np.searchsorted(a, v) (int32) | int32 | 1,000 | 0.0015 | 0.0000 | 0.01x |
|▫| np.searchsorted(a, v) (int32) | int32 | 100,000 | 0.0317 | 0.0000 | 0.00x |
|▫| np.searchsorted(a, v) (int32) | int32 | 10,000,000 | 22.8202 | 0.0000 | 0.00x |
|▫| np.searchsorted(a, v) (int64) | int64 | 1,000 | 0.0009 | 0.0000 | 0.02x |
|▫| np.searchsorted(a, v) (int64) | int64 | 100,000 | 0.0009 | 0.0000 | 0.02x |
|▫| np.searchsorted(a, v) (int64) | int64 | 10,000,000 | 0.0027 | 0.0000 | 0.01x |

### LinearAlgebra

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.dot(a, b) (float64) | float64 | 1,000 | 0.0007 | 0.0032 | 4.40x |
|✅| np.dot(a, b) (float64) | float64 | 100,000 | 0.1106 | 0.0714 | 0.65x |
|🔴| np.dot(a, b) (float64) | float64 | 10,000,000 | 1.2316 | 16.4598 | 13.36x |
|🟠| np.matmul(A, B) (float64) | float64 | 1,000 | 0.0026 | 0.0052 | 2.03x |
|🔴| np.matmul(A, B) (float64) | float64 | 100,000 | 0.6011 | 3.2324 | 5.38x |
|🔴| np.matmul(A, B) (float64) | float64 | 10,000,000 | 0.7194 | 4.2604 | 5.92x |
|🟠| np.outer(a, b) (float64) | float64 | 1,000 | 0.0021 | 0.0049 | 2.36x |
|🟡| np.outer(a, b) (float64) | float64 | 100,000 | 0.0380 | 0.0493 | 1.30x |
|✅| np.outer(a, b) (float64) | float64 | 10,000,000 | 14.5048 | 11.8529 | 0.82x |

### Selection

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|▫| np.where(cond) (float64) | float64 | 1,000 | 0.0009 | 0.0014 | 1.61x |
|🟠| np.where(cond) (float64) | float64 | 100,000 | 0.0290 | 0.0596 | 2.06x |
|🟡| np.where(cond) (float64) | float64 | 10,000,000 | 7.4846 | 9.6489 | 1.29x |
|🟡| np.where(cond, a, b) (float64) | float64 | 1,000 | 0.0017 | 0.0020 | 1.19x |
|🟡| np.where(cond, a, b) (float64) | float64 | 100,000 | 0.0408 | 0.0653 | 1.60x |
|✅| np.where(cond, a, b) (float64) | float64 | 10,000,000 | 18.7539 | 14.8534 | 0.79x |
