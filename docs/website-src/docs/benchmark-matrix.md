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
|⚪| Pending | - | C# benchmark not run |

---

**Summary:** 1233 ops | ✅ 377 | 🟡 290 | 🟠 269 | 🔴 175 | ⚪ 122

## Summary by size

| N | ops | ✅ faster | 🟡 close | 🟠 slower | 🔴 much | ⚪ n/a | geomean |
|---:|----:|--------:|--------:|---------:|------:|-----:|--------:|
| 500 | 1 | 0 | 0 | 0 | 0 | 1 | - |
| 900 | 3 | 0 | 0 | 0 | 0 | 3 | - |
| 1,000 | 409 | 102 | 53 | 128 | 84 | 42 | 1.96x |
| 50,000 | 1 | 0 | 0 | 0 | 0 | 1 | - |
| 100,000 | 409 | 109 | 66 | 121 | 75 | 38 | 1.83x |
| 5,000,000 | 1 | 0 | 0 | 0 | 0 | 1 | - |
| 10,000,000 | 409 | 166 | 171 | 20 | 16 | 36 | 1.00x |

---

### 🏆 Top 15 Best (NumSharp closest to NumPy)

| | Operation | Type | N | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|----:|------:|---------:|------:|
|✅| np.copy (float64) | float64 | 10,000,000 | 18.5 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int32) | int32 | 100,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float32) | float32 | 100,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int32) | int32 | 10,000,000 | 22.8 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float32) | float32 | 10,000,000 | 23.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int32) | int32 | 1,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float32) | float32 | 1,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int64) | int64 | 10,000,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float64) | float64 | 10,000,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int64) | int64 | 1,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float64) | float64 | 1,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (int64) | int64 | 100,000 | 0.0 | 0.0 | 0.0x |
|✅| np.searchsorted(a, v) (float64) | float64 | 100,000 | 0.0 | 0.0 | 0.0x |
|✅| np.nansum(a) (float64) | float64 | 100,000 | 0.2 | 0.0 | 0.1x |
|✅| np.var (float32) | float32 | 1,000 | 0.0 | 0.0 | 0.1x |

### 🔻 Top 15 Worst (Optimization priorities)

| | Operation | Type | N | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|----:|------:|---------:|------:|
|🔴| np.zeros (int64) | int64 | 10,000,000 | 0.0 | 10.7 | 879.6x |
|🔴| np.zeros (int32) | int32 | 10,000,000 | 0.0 | 5.6 | 518.2x |
|🔴| np.zeros (float64) | float64 | 10,000,000 | 0.0 | 10.8 | 507.6x |
|🔴| np.zeros (float32) | float32 | 10,000,000 | 0.0 | 5.7 | 334.0x |
|🔴| np.copy (float64) | float64 | 1,000 | 0.0 | 0.0 | 33.8x |
|🔴| np.copy (int64) | int64 | 1,000 | 0.0 | 0.0 | 31.8x |
|🔴| np.argsort(a) (int64) | int64 | 100,000 | 0.5 | 12.9 | 27.3x |
|🔴| np.left_shift(a, 2) (uint64) | uint64 | 1,000 | 0.0 | 0.0 | 24.9x |
|🔴| np.argsort(a) (int32) | int32 | 100,000 | 0.4 | 10.4 | 23.5x |
|🔴| a * 2 (literal) (float32) | float32 | 100,000 | 0.0 | 0.1 | 19.4x |
|🔴| np.right_shift(a, 2) (int64) | int64 | 1,000 | 0.0 | 0.0 | 19.2x |
|🔴| np.left_shift(a, 2) (int64) | int64 | 1,000 | 0.0 | 0.0 | 19.1x |
|🔴| np.ones (int64) | int64 | 1,000 | 0.0 | 0.0 | 18.0x |
|🔴| a | b (int64) | int64 | 1,000 | 0.0 | 0.0 | 16.5x |
|🔴| np.ones (float64) | float64 | 1,000 | 0.0 | 0.0 | 16.3x |

---

### Arithmetic

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟡| a % 7 (literal) (float32) | float32 | 1,000 | 0.0140 | 0.0190 | 1.34x |
|🟡| a % 7 (literal) (float32) | float32 | 100,000 | 1.6550 | 1.9680 | 1.19x |
|🟡| a % 7 (literal) (float32) | float32 | 10,000,000 | 167.3110 | 194.6950 | 1.16x |
|🟠| a % 7 (literal) (float64) | float64 | 1,000 | 0.0110 | 0.0240 | 2.09x |
|🟡| a % 7 (literal) (float64) | float64 | 100,000 | 1.4790 | 1.7960 | 1.21x |
|🟡| a % 7 (literal) (float64) | float64 | 10,000,000 | 155.5830 | 178.7750 | 1.15x |
|🟡| a % 7 (literal) (int32) | int32 | 1,000 | 0.0020 | 0.0040 | 1.92x |
|🟡| a % 7 (literal) (int32) | int32 | 100,000 | 0.4120 | 0.6920 | 1.68x |
|🟡| a % 7 (literal) (int32) | int32 | 10,000,000 | 45.8310 | 70.8020 | 1.54x |
|🟡| a % 7 (literal) (int64) | int64 | 1,000 | 0.0040 | 0.0060 | 1.35x |
|🟠| a % 7 (literal) (int64) | int64 | 100,000 | 0.4220 | 0.9120 | 2.16x |
|🟡| a % 7 (literal) (int64) | int64 | 10,000,000 | 51.6810 | 93.8370 | 1.82x |
|✅| a % b (element-wise) (float32) | float32 | 1,000 | 0.0130 | 0.0120 | 0.98x |
|🟡| a % b (element-wise) (float32) | float32 | 100,000 | 1.5070 | 1.6640 | 1.10x |
|🟡| a % b (element-wise) (float32) | float32 | 10,000,000 | 156.3310 | 166.9310 | 1.07x |
|✅| a % b (element-wise) (float64) | float64 | 1,000 | 0.0100 | 0.0100 | 0.99x |
|🟡| a % b (element-wise) (float64) | float64 | 100,000 | 1.3230 | 1.4720 | 1.11x |
|🟡| a % b (element-wise) (float64) | float64 | 10,000,000 | 143.2560 | 151.6140 | 1.06x |
|🟡| a % b (element-wise) (int32) | int32 | 1,000 | 0.0020 | 0.0040 | 1.89x |
|🟡| a % b (element-wise) (int32) | int32 | 100,000 | 0.3760 | 0.6160 | 1.64x |
|🟡| a % b (element-wise) (int32) | int32 | 10,000,000 | 43.2280 | 64.9450 | 1.50x |
|🟡| a % b (element-wise) (int64) | int64 | 1,000 | 0.0040 | 0.0040 | 1.05x |
|🟡| a % b (element-wise) (int64) | int64 | 100,000 | 0.4160 | 0.6300 | 1.51x |
|🟡| a % b (element-wise) (int64) | int64 | 10,000,000 | 48.6070 | 67.4010 | 1.39x |
|🔴| a * 2 (literal) (float32) | float32 | 1,000 | 0.0010 | 0.0060 | 8.21x |
|🔴| a * 2 (literal) (float32) | float32 | 100,000 | 0.0070 | 0.1290 | 19.37x |
|🟡| a * 2 (literal) (float32) | float32 | 10,000,000 | 8.3160 | 12.2130 | 1.47x |
|🔴| a * 2 (literal) (float64) | float64 | 1,000 | 0.0010 | 0.0070 | 9.02x |
|🟠| a * 2 (literal) (float64) | float64 | 100,000 | 0.0130 | 0.0470 | 3.55x |
|✅| a * 2 (literal) (float64) | float64 | 10,000,000 | 17.3760 | 8.9150 | 0.51x |
|🔴| a * 2 (literal) (int16) | int16 | 1,000 | 0.0010 | 0.0060 | 5.38x |
|🟠| a * 2 (literal) (int16) | int16 | 100,000 | 0.0230 | 0.0940 | 4.07x |
|🟠| a * 2 (literal) (int16) | int16 | 10,000,000 | 4.6630 | 9.7330 | 2.09x |
|🟠| a * 2 (literal) (int32) | int32 | 1,000 | 0.0010 | 0.0040 | 3.81x |
|🟠| a * 2 (literal) (int32) | int32 | 100,000 | 0.0230 | 0.0550 | 2.42x |
|🟡| a * 2 (literal) (int32) | int32 | 10,000,000 | 8.7620 | 10.1740 | 1.16x |
|🔴| a * 2 (literal) (int64) | int64 | 1,000 | 0.0010 | 0.0070 | 7.66x |
|🔴| a * 2 (literal) (int64) | int64 | 100,000 | 0.0220 | 0.1210 | 5.41x |
|🟡| a * 2 (literal) (int64) | int64 | 10,000,000 | 18.5880 | 22.7630 | 1.22x |
|🔴| a * 2 (literal) (uint16) | uint16 | 1,000 | 0.0010 | 0.0060 | 6.43x |
|🔴| a * 2 (literal) (uint16) | uint16 | 100,000 | 0.0220 | 0.1190 | 5.29x |
|🟠| a * 2 (literal) (uint16) | uint16 | 10,000,000 | 4.4330 | 9.0580 | 2.04x |
|🔴| a * 2 (literal) (uint32) | uint32 | 1,000 | 0.0010 | 0.0060 | 5.97x |
|🟠| a * 2 (literal) (uint32) | uint32 | 100,000 | 0.0250 | 0.1100 | 4.47x |
|🟡| a * 2 (literal) (uint32) | uint32 | 10,000,000 | 8.5110 | 12.0670 | 1.42x |
|🔴| a * 2 (literal) (uint64) | uint64 | 1,000 | 0.0010 | 0.0070 | 8.17x |
|🔴| a * 2 (literal) (uint64) | uint64 | 100,000 | 0.0230 | 0.1580 | 6.77x |
|🟡| a * 2 (literal) (uint64) | uint64 | 10,000,000 | 15.8400 | 22.1490 | 1.40x |
|🟠| a * 2 (literal) (uint8) | uint8 | 1,000 | 0.0010 | 0.0040 | 4.99x |
|🟠| a * 2 (literal) (uint8) | uint8 | 100,000 | 0.0240 | 0.1040 | 4.40x |
|🟠| a * 2 (literal) (uint8) | uint8 | 10,000,000 | 3.6470 | 8.4930 | 2.33x |
|🟠| a * a (square) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.41x |
|🔴| a * a (square) (float32) | float32 | 100,000 | 0.0080 | 0.0540 | 6.90x |
|🟡| a * a (square) (float32) | float32 | 10,000,000 | 8.0930 | 11.1120 | 1.37x |
|🔴| a * a (square) (float64) | float64 | 1,000 | 0.0000 | 0.0030 | 5.33x |
|🔴| a * a (square) (float64) | float64 | 100,000 | 0.0160 | 0.1040 | 6.48x |
|🟡| a * a (square) (float64) | float64 | 10,000,000 | 16.9690 | 20.3620 | 1.20x |
|🟠| a * a (square) (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.11x |
|✅| a * a (square) (int16) | int16 | 100,000 | 0.0300 | 0.0280 | 0.93x |
|🟡| a * a (square) (int16) | int16 | 10,000,000 | 5.0050 | 5.5910 | 1.12x |
|🟠| a * a (square) (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.18x |
|🟠| a * a (square) (int32) | int32 | 100,000 | 0.0290 | 0.0580 | 2.02x |
|🟡| a * a (square) (int32) | int32 | 10,000,000 | 8.7440 | 10.0340 | 1.15x |
|🟠| a * a (square) (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.93x |
|🟠| a * a (square) (int64) | int64 | 100,000 | 0.0290 | 0.1100 | 3.77x |
|🟡| a * a (square) (int64) | int64 | 10,000,000 | 17.1430 | 21.0320 | 1.23x |
|🟠| a * a (square) (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.10x |
|✅| a * a (square) (uint16) | uint16 | 100,000 | 0.0280 | 0.0270 | 0.97x |
|🟡| a * a (square) (uint16) | uint16 | 10,000,000 | 4.9790 | 5.4120 | 1.09x |
|🟠| a * a (square) (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.28x |
|🟡| a * a (square) (uint32) | uint32 | 100,000 | 0.0300 | 0.0550 | 1.80x |
|🟡| a * a (square) (uint32) | uint32 | 10,000,000 | 8.4000 | 10.8110 | 1.29x |
|🟠| a * a (square) (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.94x |
|🟠| a * a (square) (uint64) | uint64 | 100,000 | 0.0300 | 0.1150 | 3.89x |
|🟡| a * a (square) (uint64) | uint64 | 10,000,000 | 16.3410 | 21.5050 | 1.32x |
|🟠| a * a (square) (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 2.24x |
|✅| a * a (square) (uint8) | uint8 | 100,000 | 0.0280 | 0.0160 | 0.57x |
|✅| a * a (square) (uint8) | uint8 | 10,000,000 | 3.8700 | 2.2710 | 0.59x |
|🟠| a * b (element-wise) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.52x |
|🔴| a * b (element-wise) (float32) | float32 | 100,000 | 0.0070 | 0.0540 | 7.67x |
|🟡| a * b (element-wise) (float32) | float32 | 10,000,000 | 8.9540 | 14.0770 | 1.57x |
|🔴| a * b (element-wise) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 5.81x |
|🟠| a * b (element-wise) (float64) | float64 | 100,000 | 0.0310 | 0.1140 | 3.66x |
|🟡| a * b (element-wise) (float64) | float64 | 10,000,000 | 17.6850 | 26.5090 | 1.50x |
|🟠| a * b (element-wise) (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.43x |
|✅| a * b (element-wise) (int16) | int16 | 100,000 | 0.0280 | 0.0280 | 0.99x |
|🟡| a * b (element-wise) (int16) | int16 | 10,000,000 | 5.1150 | 7.0270 | 1.37x |
|🟠| a * b (element-wise) (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.18x |
|🟡| a * b (element-wise) (int32) | int32 | 100,000 | 0.0300 | 0.0580 | 1.90x |
|🟡| a * b (element-wise) (int32) | int32 | 10,000,000 | 10.0580 | 14.3010 | 1.42x |
|🟠| a * b (element-wise) (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.60x |
|🟠| a * b (element-wise) (int64) | int64 | 100,000 | 0.0330 | 0.1180 | 3.62x |
|🟡| a * b (element-wise) (int64) | int64 | 10,000,000 | 18.7230 | 28.7620 | 1.54x |
|✅| a * b (element-wise) (uint16) | uint16 | 1,000 | 0.0020 | 0.0020 | 0.90x |
|🟡| a * b (element-wise) (uint16) | uint16 | 100,000 | 0.0280 | 0.0290 | 1.04x |
|🟡| a * b (element-wise) (uint16) | uint16 | 10,000,000 | 5.3960 | 6.9260 | 1.28x |
|🟠| a * b (element-wise) (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.75x |
|🟡| a * b (element-wise) (uint32) | uint32 | 100,000 | 0.0300 | 0.0550 | 1.86x |
|🟡| a * b (element-wise) (uint32) | uint32 | 10,000,000 | 8.8650 | 13.5570 | 1.53x |
|🟠| a * b (element-wise) (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.98x |
|🟠| a * b (element-wise) (uint64) | uint64 | 100,000 | 0.0310 | 0.1130 | 3.63x |
|🟡| a * b (element-wise) (uint64) | uint64 | 10,000,000 | 19.0200 | 31.9030 | 1.68x |
|🟠| a * b (element-wise) (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.63x |
|✅| a * b (element-wise) (uint8) | uint8 | 100,000 | 0.0280 | 0.0150 | 0.55x |
|✅| a * b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0140 | 3.3710 | 0.84x |
|🟠| a * scalar (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.59x |
|🔴| a * scalar (float32) | float32 | 100,000 | 0.0060 | 0.0570 | 9.26x |
|🟡| a * scalar (float32) | float32 | 10,000,000 | 8.0650 | 10.2660 | 1.27x |
|🟠| a * scalar (float64) | float64 | 1,000 | 0.0010 | 0.0020 | 2.51x |
|🔴| a * scalar (float64) | float64 | 100,000 | 0.0150 | 0.1150 | 7.69x |
|🟡| a * scalar (float64) | float64 | 10,000,000 | 18.5640 | 20.1310 | 1.08x |
|🟠| a * scalar (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.23x |
|🟡| a * scalar (int16) | int16 | 100,000 | 0.0240 | 0.0270 | 1.15x |
|🟡| a * scalar (int16) | int16 | 10,000,000 | 4.6200 | 5.4070 | 1.17x |
|🟠| a * scalar (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.14x |
|🟠| a * scalar (int32) | int32 | 100,000 | 0.0230 | 0.0550 | 2.35x |
|🟡| a * scalar (int32) | int32 | 10,000,000 | 8.3510 | 10.1460 | 1.21x |
|🟠| a * scalar (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.37x |
|🟠| a * scalar (int64) | int64 | 100,000 | 0.0250 | 0.1090 | 4.30x |
|🟡| a * scalar (int64) | int64 | 10,000,000 | 19.4200 | 21.9690 | 1.13x |
|🟠| a * scalar (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.17x |
|🟡| a * scalar (uint16) | uint16 | 100,000 | 0.0230 | 0.0280 | 1.25x |
|🟡| a * scalar (uint16) | uint16 | 10,000,000 | 4.4610 | 5.3500 | 1.20x |
|🟠| a * scalar (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.46x |
|🟠| a * scalar (uint32) | uint32 | 100,000 | 0.0230 | 0.0530 | 2.24x |
|🟡| a * scalar (uint32) | uint32 | 10,000,000 | 8.1410 | 10.3010 | 1.27x |
|🟠| a * scalar (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.49x |
|🟠| a * scalar (uint64) | uint64 | 100,000 | 0.0230 | 0.1110 | 4.92x |
|🟡| a * scalar (uint64) | uint64 | 10,000,000 | 16.9770 | 21.1370 | 1.25x |
|🟠| a * scalar (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.44x |
|✅| a * scalar (uint8) | uint8 | 100,000 | 0.0240 | 0.0160 | 0.65x |
|✅| a * scalar (uint8) | uint8 | 10,000,000 | 3.7190 | 2.4710 | 0.66x |
|🔴| a + 5 (literal) (float32) | float32 | 1,000 | 0.0010 | 0.0060 | 7.01x |
|🔴| a + 5 (literal) (float32) | float32 | 100,000 | 0.0070 | 0.0970 | 14.66x |
|🟡| a + 5 (literal) (float32) | float32 | 10,000,000 | 8.6020 | 12.9630 | 1.51x |
|🔴| a + 5 (literal) (float64) | float64 | 1,000 | 0.0010 | 0.0070 | 9.16x |
|🔴| a + 5 (literal) (float64) | float64 | 100,000 | 0.0140 | 0.1210 | 8.87x |
|🟡| a + 5 (literal) (float64) | float64 | 10,000,000 | 16.3100 | 21.8590 | 1.34x |
|🔴| a + 5 (literal) (int16) | int16 | 1,000 | 0.0010 | 0.0070 | 6.10x |
|🟠| a + 5 (literal) (int16) | int16 | 100,000 | 0.0250 | 0.0880 | 3.58x |
|🟡| a + 5 (literal) (int16) | int16 | 10,000,000 | 7.7900 | 9.6850 | 1.24x |
|🟠| a + 5 (literal) (int32) | int32 | 1,000 | 0.0010 | 0.0030 | 3.19x |
|🟠| a + 5 (literal) (int32) | int32 | 100,000 | 0.0240 | 0.0530 | 2.18x |
|🟡| a + 5 (literal) (int32) | int32 | 10,000,000 | 9.4340 | 10.1110 | 1.07x |
|🟠| a + 5 (literal) (int64) | int64 | 1,000 | 0.0010 | 0.0040 | 4.08x |
|🔴| a + 5 (literal) (int64) | int64 | 100,000 | 0.0240 | 0.1340 | 5.63x |
|🟡| a + 5 (literal) (int64) | int64 | 10,000,000 | 16.0350 | 22.1190 | 1.38x |
|🔴| a + 5 (literal) (uint16) | uint16 | 1,000 | 0.0010 | 0.0070 | 6.42x |
|🟠| a + 5 (literal) (uint16) | uint16 | 100,000 | 0.0260 | 0.0900 | 3.45x |
|🟡| a + 5 (literal) (uint16) | uint16 | 10,000,000 | 4.8120 | 9.2840 | 1.93x |
|🔴| a + 5 (literal) (uint32) | uint32 | 1,000 | 0.0010 | 0.0060 | 6.27x |
|🟠| a + 5 (literal) (uint32) | uint32 | 100,000 | 0.0250 | 0.0950 | 3.83x |
|🟡| a + 5 (literal) (uint32) | uint32 | 10,000,000 | 8.2200 | 12.3830 | 1.51x |
|🔴| a + 5 (literal) (uint64) | uint64 | 1,000 | 0.0010 | 0.0080 | 7.45x |
|🟠| a + 5 (literal) (uint64) | uint64 | 100,000 | 0.0260 | 0.1240 | 4.75x |
|🟡| a + 5 (literal) (uint64) | uint64 | 10,000,000 | 15.7470 | 22.7520 | 1.44x |
|🔴| a + 5 (literal) (uint8) | uint8 | 1,000 | 0.0010 | 0.0050 | 5.58x |
|🟠| a + 5 (literal) (uint8) | uint8 | 100,000 | 0.0260 | 0.0890 | 3.46x |
|🟠| a + 5 (literal) (uint8) | uint8 | 10,000,000 | 3.4860 | 8.8820 | 2.55x |
|🟠| a + b (element-wise) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.69x |
|🔴| a + b (element-wise) (float32) | float32 | 100,000 | 0.0070 | 0.0500 | 7.25x |
|🟡| a + b (element-wise) (float32) | float32 | 10,000,000 | 9.5100 | 13.8920 | 1.46x |
|🟠| a + b (element-wise) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.68x |
|🟠| a + b (element-wise) (float64) | float64 | 100,000 | 0.0300 | 0.1170 | 3.88x |
|🟡| a + b (element-wise) (float64) | float64 | 10,000,000 | 18.9760 | 26.6010 | 1.40x |
|🟠| a + b (element-wise) (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.81x |
|✅| a + b (element-wise) (int16) | int16 | 100,000 | 0.0300 | 0.0290 | 0.97x |
|🟡| a + b (element-wise) (int16) | int16 | 10,000,000 | 6.6920 | 7.2020 | 1.08x |
|🟡| a + b (element-wise) (int32) | int32 | 1,000 | 0.0010 | 0.0010 | 1.02x |
|🟡| a + b (element-wise) (int32) | int32 | 100,000 | 0.0300 | 0.0580 | 1.95x |
|🟡| a + b (element-wise) (int32) | int32 | 10,000,000 | 9.0920 | 13.8150 | 1.52x |
|🟠| a + b (element-wise) (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 4.08x |
|🟠| a + b (element-wise) (int64) | int64 | 100,000 | 0.0340 | 0.1190 | 3.55x |
|🟡| a + b (element-wise) (int64) | int64 | 10,000,000 | 19.8210 | 26.2860 | 1.33x |
|🟠| a + b (element-wise) (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.61x |
|✅| a + b (element-wise) (uint16) | uint16 | 100,000 | 0.0300 | 0.0290 | 0.96x |
|🟡| a + b (element-wise) (uint16) | uint16 | 10,000,000 | 5.3260 | 7.1650 | 1.35x |
|🟠| a + b (element-wise) (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.53x |
|🟡| a + b (element-wise) (uint32) | uint32 | 100,000 | 0.0320 | 0.0520 | 1.62x |
|🟡| a + b (element-wise) (uint32) | uint32 | 10,000,000 | 9.0100 | 14.3470 | 1.59x |
|🟠| a + b (element-wise) (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 4.49x |
|🟠| a + b (element-wise) (uint64) | uint64 | 100,000 | 0.0350 | 0.1050 | 3.01x |
|🟡| a + b (element-wise) (uint64) | uint64 | 10,000,000 | 18.6850 | 26.5870 | 1.42x |
|🟠| a + b (element-wise) (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.11x |
|✅| a + b (element-wise) (uint8) | uint8 | 100,000 | 0.0290 | 0.0180 | 0.63x |
|✅| a + b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0510 | 3.6030 | 0.89x |
|🟠| a + scalar (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.09x |
|🔴| a + scalar (float32) | float32 | 100,000 | 0.0060 | 0.0540 | 8.51x |
|🟡| a + scalar (float32) | float32 | 10,000,000 | 8.1740 | 10.5120 | 1.29x |
|🟠| a + scalar (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.89x |
|🔴| a + scalar (float64) | float64 | 100,000 | 0.0130 | 0.1100 | 8.30x |
|🟡| a + scalar (float64) | float64 | 10,000,000 | 16.1090 | 19.6990 | 1.22x |
|🟠| a + scalar (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.20x |
|🟡| a + scalar (int16) | int16 | 100,000 | 0.0250 | 0.0290 | 1.15x |
|✅| a + scalar (int16) | int16 | 10,000,000 | 6.9440 | 5.0680 | 0.73x |
|🟡| a + scalar (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 1.54x |
|🟠| a + scalar (int32) | int32 | 100,000 | 0.0240 | 0.0530 | 2.16x |
|🟡| a + scalar (int32) | int32 | 10,000,000 | 9.2980 | 10.1710 | 1.09x |
|🟠| a + scalar (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.80x |
|🟠| a + scalar (int64) | int64 | 100,000 | 0.0240 | 0.1110 | 4.67x |
|🟡| a + scalar (int64) | int64 | 10,000,000 | 15.7960 | 19.6950 | 1.25x |
|🟡| a + scalar (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 1.86x |
|🟡| a + scalar (uint16) | uint16 | 100,000 | 0.0250 | 0.0260 | 1.06x |
|🟡| a + scalar (uint16) | uint16 | 10,000,000 | 5.1280 | 5.4370 | 1.06x |
|🟡| a + scalar (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 1.98x |
|🟠| a + scalar (uint32) | uint32 | 100,000 | 0.0240 | 0.0580 | 2.38x |
|🟡| a + scalar (uint32) | uint32 | 10,000,000 | 8.1240 | 10.3640 | 1.28x |
|🟠| a + scalar (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.43x |
|🟠| a + scalar (uint64) | uint64 | 100,000 | 0.0250 | 0.1050 | 4.23x |
|🟡| a + scalar (uint64) | uint64 | 10,000,000 | 16.2200 | 20.3900 | 1.26x |
|🟠| a + scalar (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.00x |
|✅| a + scalar (uint8) | uint8 | 100,000 | 0.0250 | 0.0140 | 0.56x |
|✅| a + scalar (uint8) | uint8 | 10,000,000 | 3.6100 | 2.4040 | 0.67x |
|🟠| a - b (element-wise) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.55x |
|🔴| a - b (element-wise) (float32) | float32 | 100,000 | 0.0070 | 0.0580 | 7.91x |
|🟡| a - b (element-wise) (float32) | float32 | 10,000,000 | 9.0620 | 14.1840 | 1.57x |
|🔴| a - b (element-wise) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 6.16x |
|🟠| a - b (element-wise) (float64) | float64 | 100,000 | 0.0300 | 0.1120 | 3.79x |
|🟡| a - b (element-wise) (float64) | float64 | 10,000,000 | 17.6100 | 26.5900 | 1.51x |
|🟠| a - b (element-wise) (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.54x |
|🟡| a - b (element-wise) (int16) | int16 | 100,000 | 0.0290 | 0.0300 | 1.01x |
|🟡| a - b (element-wise) (int16) | int16 | 10,000,000 | 7.1680 | 7.3130 | 1.02x |
|🟠| a - b (element-wise) (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.99x |
|🟠| a - b (element-wise) (int32) | int32 | 100,000 | 0.0300 | 0.0620 | 2.07x |
|🟡| a - b (element-wise) (int32) | int32 | 10,000,000 | 10.2590 | 14.1260 | 1.38x |
|🟠| a - b (element-wise) (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 4.20x |
|🟠| a - b (element-wise) (int64) | int64 | 100,000 | 0.0340 | 0.1160 | 3.37x |
|🟡| a - b (element-wise) (int64) | int64 | 10,000,000 | 18.0470 | 27.6480 | 1.53x |
|🟠| a - b (element-wise) (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.33x |
|✅| a - b (element-wise) (uint16) | uint16 | 100,000 | 0.0320 | 0.0300 | 0.93x |
|🟡| a - b (element-wise) (uint16) | uint16 | 10,000,000 | 5.2550 | 6.9090 | 1.31x |
|🟠| a - b (element-wise) (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.24x |
|🟡| a - b (element-wise) (uint32) | uint32 | 100,000 | 0.0320 | 0.0560 | 1.77x |
|🟡| a - b (element-wise) (uint32) | uint32 | 10,000,000 | 8.8780 | 14.3280 | 1.61x |
|🟠| a - b (element-wise) (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.87x |
|🟠| a - b (element-wise) (uint64) | uint64 | 100,000 | 0.0330 | 0.1090 | 3.26x |
|🟡| a - b (element-wise) (uint64) | uint64 | 10,000,000 | 18.6900 | 27.2790 | 1.46x |
|🟠| a - b (element-wise) (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.64x |
|✅| a - b (element-wise) (uint8) | uint8 | 100,000 | 0.0290 | 0.0160 | 0.54x |
|✅| a - b (element-wise) (uint8) | uint8 | 10,000,000 | 4.0510 | 3.3610 | 0.83x |
|🟠| a - scalar (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.53x |
|🔴| a - scalar (float32) | float32 | 100,000 | 0.0060 | 0.0560 | 8.84x |
|🟡| a - scalar (float32) | float32 | 10,000,000 | 8.2990 | 10.2150 | 1.23x |
|🟠| a - scalar (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.45x |
|🔴| a - scalar (float64) | float64 | 100,000 | 0.0160 | 0.1060 | 6.47x |
|🟡| a - scalar (float64) | float64 | 10,000,000 | 16.1450 | 20.1160 | 1.25x |
|🟠| a - scalar (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.24x |
|🟡| a - scalar (int16) | int16 | 100,000 | 0.0250 | 0.0280 | 1.11x |
|✅| a - scalar (int16) | int16 | 10,000,000 | 5.4930 | 5.4870 | 1.00x |
|🟠| a - scalar (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.41x |
|🟠| a - scalar (int32) | int32 | 100,000 | 0.0250 | 0.0560 | 2.22x |
|🟡| a - scalar (int32) | int32 | 10,000,000 | 9.2400 | 10.0020 | 1.08x |
|🟠| a - scalar (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.38x |
|🟠| a - scalar (int64) | int64 | 100,000 | 0.0260 | 0.1180 | 4.61x |
|🟡| a - scalar (int64) | int64 | 10,000,000 | 15.5430 | 20.3870 | 1.31x |
|🟠| a - scalar (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.38x |
|🟡| a - scalar (uint16) | uint16 | 100,000 | 0.0250 | 0.0300 | 1.20x |
|🟡| a - scalar (uint16) | uint16 | 10,000,000 | 5.1740 | 5.3100 | 1.03x |
|🟠| a - scalar (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.30x |
|🟠| a - scalar (uint32) | uint32 | 100,000 | 0.0260 | 0.0570 | 2.22x |
|🟡| a - scalar (uint32) | uint32 | 10,000,000 | 7.9590 | 10.5440 | 1.32x |
|🟠| a - scalar (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.22x |
|🟠| a - scalar (uint64) | uint64 | 100,000 | 0.0250 | 0.1120 | 4.44x |
|🟡| a - scalar (uint64) | uint64 | 10,000,000 | 16.4040 | 20.4110 | 1.24x |
|🟠| a - scalar (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.12x |
|✅| a - scalar (uint8) | uint8 | 100,000 | 0.0250 | 0.0150 | 0.61x |
|✅| a - scalar (uint8) | uint8 | 10,000,000 | 3.5890 | 2.2350 | 0.62x |
|🟠| a / b (element-wise) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 4.52x |
|🔴| a / b (element-wise) (float32) | float32 | 100,000 | 0.0120 | 0.0660 | 5.38x |
|🟡| a / b (element-wise) (float32) | float32 | 10,000,000 | 9.1490 | 13.7920 | 1.51x |
|🔴| a / b (element-wise) (float64) | float64 | 1,000 | 0.0010 | 0.0040 | 5.16x |
|🔴| a / b (element-wise) (float64) | float64 | 100,000 | 0.0380 | 0.2010 | 5.29x |
|🟡| a / b (element-wise) (float64) | float64 | 10,000,000 | 19.1610 | 27.3720 | 1.43x |
|🟠| a / b (element-wise) (int32) | int32 | 1,000 | 0.0020 | 0.0060 | 3.05x |
|🟠| a / b (element-wise) (int32) | int32 | 100,000 | 0.0880 | 0.2040 | 2.31x |
|🟡| a / b (element-wise) (int32) | int32 | 10,000,000 | 20.6750 | 25.0260 | 1.21x |
|🟠| a / b (element-wise) (int64) | int64 | 1,000 | 0.0020 | 0.0060 | 3.43x |
|🟠| a / b (element-wise) (int64) | int64 | 100,000 | 0.0840 | 0.2050 | 2.44x |
|🟡| a / b (element-wise) (int64) | int64 | 10,000,000 | 26.5560 | 31.1640 | 1.17x |
|🟠| a / scalar (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.69x |
|🔴| a / scalar (float32) | float32 | 100,000 | 0.0130 | 0.0660 | 5.16x |
|🟡| a / scalar (float32) | float32 | 10,000,000 | 8.4860 | 10.5240 | 1.24x |
|🟠| a / scalar (float64) | float64 | 1,000 | 0.0010 | 0.0050 | 4.99x |
|🟠| a / scalar (float64) | float64 | 100,000 | 0.0380 | 0.1870 | 4.92x |
|🟡| a / scalar (float64) | float64 | 10,000,000 | 16.4900 | 23.7050 | 1.44x |
|🟠| a / scalar (int32) | int32 | 1,000 | 0.0020 | 0.0080 | 4.63x |
|🟠| a / scalar (int32) | int32 | 100,000 | 0.0710 | 0.2070 | 2.91x |
|🟡| a / scalar (int32) | int32 | 10,000,000 | 17.1920 | 24.3260 | 1.41x |
|🟠| a / scalar (int64) | int64 | 1,000 | 0.0020 | 0.0070 | 4.15x |
|🟠| a / scalar (int64) | int64 | 100,000 | 0.0600 | 0.2090 | 3.47x |
|🟡| a / scalar (int64) | int64 | 10,000,000 | 19.6400 | 24.9500 | 1.27x |
|🟠| np.add(a, b) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.10x |
|🔴| np.add(a, b) (float32) | float32 | 100,000 | 0.0070 | 0.0510 | 7.28x |
|🟡| np.add(a, b) (float32) | float32 | 10,000,000 | 8.9120 | 13.2950 | 1.49x |
|🔴| np.add(a, b) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 5.18x |
|🟠| np.add(a, b) (float64) | float64 | 100,000 | 0.0300 | 0.1130 | 3.76x |
|🟡| np.add(a, b) (float64) | float64 | 10,000,000 | 17.9590 | 27.0550 | 1.51x |
|🟠| np.add(a, b) (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 2.47x |
|✅| np.add(a, b) (int16) | int16 | 100,000 | 0.0310 | 0.0290 | 0.92x |
|✅| np.add(a, b) (int16) | int16 | 10,000,000 | 7.3310 | 7.1900 | 0.98x |
|🟠| np.add(a, b) (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.28x |
|🟡| np.add(a, b) (int32) | int32 | 100,000 | 0.0320 | 0.0610 | 1.93x |
|🟡| np.add(a, b) (int32) | int32 | 10,000,000 | 9.9570 | 14.0940 | 1.42x |
|🟠| np.add(a, b) (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 4.05x |
|🟠| np.add(a, b) (int64) | int64 | 100,000 | 0.0330 | 0.1080 | 3.26x |
|🟡| np.add(a, b) (int64) | int64 | 10,000,000 | 20.6290 | 27.6310 | 1.34x |
|🟠| np.add(a, b) (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.07x |
|✅| np.add(a, b) (uint16) | uint16 | 100,000 | 0.0300 | 0.0260 | 0.86x |
|🟡| np.add(a, b) (uint16) | uint16 | 10,000,000 | 5.3640 | 7.1580 | 1.33x |
|🟠| np.add(a, b) (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.31x |
|🟡| np.add(a, b) (uint32) | uint32 | 100,000 | 0.0370 | 0.0540 | 1.45x |
|🟡| np.add(a, b) (uint32) | uint32 | 10,000,000 | 9.5710 | 14.2470 | 1.49x |
|🟠| np.add(a, b) (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 4.16x |
|🟠| np.add(a, b) (uint64) | uint64 | 100,000 | 0.0330 | 0.1150 | 3.47x |
|🟡| np.add(a, b) (uint64) | uint64 | 10,000,000 | 18.7020 | 26.5090 | 1.42x |
|🟠| np.add(a, b) (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.81x |
|✅| np.add(a, b) (uint8) | uint8 | 100,000 | 0.0290 | 0.0160 | 0.55x |
|✅| np.add(a, b) (uint8) | uint8 | 10,000,000 | 4.0240 | 3.0200 | 0.75x |
|🟠| scalar - a (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.72x |
|🔴| scalar - a (float32) | float32 | 100,000 | 0.0070 | 0.0540 | 7.94x |
|🟡| scalar - a (float32) | float32 | 10,000,000 | 8.4100 | 10.3300 | 1.23x |
|🟠| scalar - a (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.26x |
|🔴| scalar - a (float64) | float64 | 100,000 | 0.0140 | 0.1060 | 7.76x |
|🟡| scalar - a (float64) | float64 | 10,000,000 | 16.6030 | 19.7540 | 1.19x |
|🟡| scalar - a (int16) | int16 | 1,000 | 0.0010 | 0.0020 | 1.78x |
|🟡| scalar - a (int16) | int16 | 100,000 | 0.0250 | 0.0290 | 1.18x |
|🟡| scalar - a (int16) | int16 | 10,000,000 | 4.6710 | 5.1130 | 1.09x |
|🟠| scalar - a (int32) | int32 | 1,000 | 0.0010 | 0.0020 | 2.16x |
|🟠| scalar - a (int32) | int32 | 100,000 | 0.0260 | 0.0560 | 2.17x |
|🟡| scalar - a (int32) | int32 | 10,000,000 | 9.3300 | 10.1600 | 1.09x |
|🟠| scalar - a (int64) | int64 | 1,000 | 0.0010 | 0.0030 | 3.64x |
|🟠| scalar - a (int64) | int64 | 100,000 | 0.0270 | 0.1100 | 4.07x |
|🟡| scalar - a (int64) | int64 | 10,000,000 | 16.0800 | 20.5010 | 1.27x |
|🟠| scalar - a (uint16) | uint16 | 1,000 | 0.0010 | 0.0020 | 2.12x |
|🟡| scalar - a (uint16) | uint16 | 100,000 | 0.0260 | 0.0280 | 1.07x |
|🟡| scalar - a (uint16) | uint16 | 10,000,000 | 4.9030 | 5.4810 | 1.12x |
|🟠| scalar - a (uint32) | uint32 | 1,000 | 0.0010 | 0.0020 | 2.07x |
|🟠| scalar - a (uint32) | uint32 | 100,000 | 0.0260 | 0.0540 | 2.09x |
|🟡| scalar - a (uint32) | uint32 | 10,000,000 | 8.2140 | 10.4420 | 1.27x |
|🟠| scalar - a (uint64) | uint64 | 1,000 | 0.0010 | 0.0030 | 3.23x |
|🟠| scalar - a (uint64) | uint64 | 100,000 | 0.0250 | 0.1110 | 4.47x |
|🟡| scalar - a (uint64) | uint64 | 10,000,000 | 16.0050 | 19.7880 | 1.24x |
|🟠| scalar - a (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 2.26x |
|✅| scalar - a (uint8) | uint8 | 100,000 | 0.0240 | 0.0150 | 0.63x |
|✅| scalar - a (uint8) | uint8 | 10,000,000 | 3.7320 | 2.3710 | 0.64x |
|🟠| scalar / a (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.72x |
|🔴| scalar / a (float32) | float32 | 100,000 | 0.0120 | 0.0660 | 5.33x |
|🟡| scalar / a (float32) | float32 | 10,000,000 | 8.3590 | 10.1970 | 1.22x |
|🟠| scalar / a (float64) | float64 | 1,000 | 0.0010 | 0.0040 | 4.47x |
|🔴| scalar / a (float64) | float64 | 100,000 | 0.0380 | 0.2020 | 5.36x |
|🟡| scalar / a (float64) | float64 | 10,000,000 | 16.3770 | 24.1570 | 1.48x |
|🟠| scalar / a (int32) | int32 | 1,000 | 0.0020 | 0.0070 | 4.14x |
|🟠| scalar / a (int32) | int32 | 100,000 | 0.0650 | 0.2060 | 3.20x |
|🟡| scalar / a (int32) | int32 | 10,000,000 | 17.3390 | 23.7350 | 1.37x |
|🟠| scalar / a (int64) | int64 | 1,000 | 0.0020 | 0.0080 | 4.45x |
|🟠| scalar / a (int64) | int64 | 100,000 | 0.0590 | 0.2070 | 3.51x |
|🟡| scalar / a (int64) | int64 | 10,000,000 | 19.6940 | 24.9240 | 1.27x |

### Unary

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟠| np.abs (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.49x |
|🔴| np.abs (float32) | float32 | 100,000 | 0.0060 | 0.0640 | 10.14x |
|🟡| np.abs (float32) | float32 | 10,000,000 | 7.2200 | 10.9820 | 1.52x |
|🔴| np.abs (float64) | float64 | 1,000 | 0.0010 | 0.0040 | 7.45x |
|🔴| np.abs (float64) | float64 | 100,000 | 0.0120 | 0.1200 | 10.16x |
|🟡| np.abs (float64) | float64 | 10,000,000 | 16.1380 | 20.9450 | 1.30x |
|🟠| np.cbrt(a) (float32) | float32 | 1,000 | 0.0060 | 0.0140 | 2.31x |
|🟡| np.cbrt(a) (float32) | float32 | 100,000 | 0.8790 | 1.5400 | 1.75x |
|🟡| np.cbrt(a) (float32) | float32 | 10,000,000 | 94.6670 | 150.8870 | 1.59x |
|🟠| np.cbrt(a) (float64) | float64 | 1,000 | 0.0100 | 0.0200 | 2.11x |
|🟠| np.cbrt(a) (float64) | float64 | 100,000 | 1.0610 | 2.1330 | 2.01x |
|🟡| np.cbrt(a) (float64) | float64 | 10,000,000 | 116.2440 | 210.7330 | 1.81x |
|🟠| np.ceil (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 4.04x |
|🔴| np.ceil (float32) | float32 | 100,000 | 0.0060 | 0.0540 | 8.55x |
|🟡| np.ceil (float32) | float32 | 10,000,000 | 7.9460 | 10.8310 | 1.36x |
|🔴| np.ceil (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 5.43x |
|🔴| np.ceil (float64) | float64 | 100,000 | 0.0110 | 0.1070 | 9.79x |
|🟡| np.ceil (float64) | float64 | 10,000,000 | 15.8830 | 21.5460 | 1.36x |
|🟡| np.cos (float32) | float32 | 1,000 | 0.0050 | 0.0080 | 1.61x |
|🟡| np.cos (float32) | float32 | 100,000 | 0.7040 | 1.1590 | 1.65x |
|🟡| np.cos (float32) | float32 | 10,000,000 | 80.2260 | 121.5220 | 1.51x |
|🟠| np.cos (float64) | float64 | 1,000 | 0.0050 | 0.0120 | 2.45x |
|🟡| np.cos (float64) | float64 | 100,000 | 0.7020 | 1.2530 | 1.79x |
|🟡| np.cos (float64) | float64 | 10,000,000 | 79.2760 | 128.1130 | 1.62x |
|🟠| np.exp (float32) | float32 | 1,000 | 0.0010 | 0.0060 | 4.56x |
|🔴| np.exp (float32) | float32 | 100,000 | 0.0570 | 0.4000 | 6.96x |
|🟠| np.exp (float32) | float32 | 10,000,000 | 14.0650 | 42.5190 | 3.02x |
|🟠| np.exp (float64) | float64 | 1,000 | 0.0030 | 0.0070 | 2.50x |
|🟠| np.exp (float64) | float64 | 100,000 | 0.2520 | 0.5140 | 2.04x |
|🟡| np.exp (float64) | float64 | 10,000,000 | 33.8600 | 53.0970 | 1.57x |
|🟠| np.floor (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 4.13x |
|🔴| np.floor (float32) | float32 | 100,000 | 0.0060 | 0.0580 | 8.91x |
|🟡| np.floor (float32) | float32 | 10,000,000 | 8.0350 | 10.8410 | 1.35x |
|🔴| np.floor (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 5.69x |
|🔴| np.floor (float64) | float64 | 100,000 | 0.0110 | 0.1260 | 11.21x |
|🟡| np.floor (float64) | float64 | 10,000,000 | 16.5850 | 21.3640 | 1.29x |
|🟠| np.log (float32) | float32 | 1,000 | 0.0010 | 0.0060 | 4.39x |
|🔴| np.log (float32) | float32 | 100,000 | 0.0880 | 0.4900 | 5.59x |
|🟠| np.log (float32) | float32 | 10,000,000 | 16.0110 | 46.6480 | 2.91x |
|🟠| np.log (float64) | float64 | 1,000 | 0.0030 | 0.0070 | 2.65x |
|🟠| np.log (float64) | float64 | 100,000 | 0.2500 | 0.6000 | 2.40x |
|🟡| np.log (float64) | float64 | 10,000,000 | 31.7590 | 61.5850 | 1.94x |
|🟠| np.log10 (float32) | float32 | 1,000 | 0.0020 | 0.0060 | 2.29x |
|🟠| np.log10 (float32) | float32 | 100,000 | 0.1900 | 0.4870 | 2.56x |
|🟡| np.log10 (float32) | float32 | 10,000,000 | 23.3160 | 46.3310 | 1.99x |
|🟠| np.log10 (float64) | float64 | 1,000 | 0.0030 | 0.0080 | 2.65x |
|🟠| np.log10 (float64) | float64 | 100,000 | 0.2460 | 0.6710 | 2.73x |
|🟡| np.log10 (float64) | float64 | 10,000,000 | 33.4930 | 64.1700 | 1.92x |
|🟠| np.negative(a) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.82x |
|🔴| np.negative(a) (float32) | float32 | 100,000 | 0.0060 | 0.0530 | 8.44x |
|🟡| np.negative(a) (float32) | float32 | 10,000,000 | 8.2190 | 10.4470 | 1.27x |
|🟠| np.negative(a) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.97x |
|🔴| np.negative(a) (float64) | float64 | 100,000 | 0.0130 | 0.0980 | 7.29x |
|🟡| np.negative(a) (float64) | float64 | 10,000,000 | 16.8370 | 20.2150 | 1.20x |
|🟠| np.positive(a) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 2.74x |
|🟠| np.positive(a) (float32) | float32 | 100,000 | 0.0190 | 0.0510 | 2.64x |
|✅| np.positive(a) (float32) | float32 | 10,000,000 | 8.1730 | 7.4170 | 0.91x |
|🟠| np.positive(a) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.12x |
|🔴| np.positive(a) (float64) | float64 | 100,000 | 0.0200 | 0.1040 | 5.09x |
|🟡| np.positive(a) (float64) | float64 | 10,000,000 | 14.9740 | 15.0630 | 1.01x |
|🟠| np.reciprocal(a) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.43x |
|🟠| np.reciprocal(a) (float32) | float32 | 100,000 | 0.0140 | 0.0650 | 4.48x |
|🟡| np.reciprocal(a) (float32) | float32 | 10,000,000 | 7.3150 | 10.4860 | 1.43x |
|🔴| np.reciprocal(a) (float64) | float64 | 1,000 | 0.0010 | 0.0050 | 5.81x |
|🔴| np.reciprocal(a) (float64) | float64 | 100,000 | 0.0380 | 0.2050 | 5.40x |
|🟡| np.reciprocal(a) (float64) | float64 | 10,000,000 | 15.6900 | 23.3010 | 1.49x |
|⚪| np.round (float32) | float32 | 1,000 | 0.0010 | - | - |
|⚪| np.round (float32) | float32 | 100,000 | 0.0070 | - | - |
|⚪| np.round (float32) | float32 | 10,000,000 | 8.9860 | - | - |
|⚪| np.round (float64) | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.round (float64) | float64 | 100,000 | 0.0120 | - | - |
|⚪| np.round (float64) | float64 | 10,000,000 | 16.6720 | - | - |
|🟠| np.sign (float32) | float32 | 1,000 | 0.0010 | 0.0040 | 3.62x |
|🟡| np.sign (float32) | float32 | 100,000 | 0.3000 | 0.5600 | 1.87x |
|🟡| np.sign (float32) | float32 | 10,000,000 | 36.4790 | 60.9430 | 1.67x |
|🟠| np.sign (float64) | float64 | 1,000 | 0.0010 | 0.0040 | 3.96x |
|🟠| np.sign (float64) | float64 | 100,000 | 0.2920 | 0.5860 | 2.01x |
|🟡| np.sign (float64) | float64 | 10,000,000 | 45.0340 | 63.7800 | 1.42x |
|🟡| np.sin (float32) | float32 | 1,000 | 0.0050 | 0.0090 | 1.89x |
|🟡| np.sin (float32) | float32 | 100,000 | 0.7150 | 1.2220 | 1.71x |
|🟡| np.sin (float32) | float32 | 10,000,000 | 79.8710 | 123.5470 | 1.55x |
|🟠| np.sin (float64) | float64 | 1,000 | 0.0050 | 0.0120 | 2.47x |
|🟡| np.sin (float64) | float64 | 100,000 | 0.7070 | 1.2550 | 1.78x |
|🟡| np.sin (float64) | float64 | 10,000,000 | 79.5760 | 127.2740 | 1.60x |
|🟡| np.sqrt (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 1.85x |
|🔴| np.sqrt (float32) | float32 | 100,000 | 0.0140 | 0.0780 | 5.42x |
|🟡| np.sqrt (float32) | float32 | 10,000,000 | 7.3220 | 11.0760 | 1.51x |
|🔴| np.sqrt (float64) | float64 | 1,000 | 0.0010 | 0.0050 | 5.20x |
|🔴| np.sqrt (float64) | float64 | 100,000 | 0.0580 | 0.3060 | 5.27x |
|🟠| np.sqrt (float64) | float64 | 10,000,000 | 15.8610 | 33.0780 | 2.09x |
|🟠| np.square(a) (float32) | float32 | 1,000 | 0.0000 | 0.0020 | 3.45x |
|🔴| np.square(a) (float32) | float32 | 100,000 | 0.0070 | 0.0560 | 8.36x |
|🟡| np.square(a) (float32) | float32 | 10,000,000 | 7.6100 | 10.4220 | 1.37x |
|🔴| np.square(a) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 6.09x |
|🔴| np.square(a) (float64) | float64 | 100,000 | 0.0110 | 0.1020 | 9.33x |
|🟡| np.square(a) (float64) | float64 | 10,000,000 | 15.6140 | 19.9550 | 1.28x |
|🟠| np.trunc(a) (float32) | float32 | 1,000 | 0.0010 | 0.0020 | 3.97x |
|🔴| np.trunc(a) (float32) | float32 | 100,000 | 0.0060 | 0.0540 | 9.28x |
|🟡| np.trunc(a) (float32) | float32 | 10,000,000 | 7.6330 | 10.5660 | 1.38x |
|🔴| np.trunc(a) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 5.52x |
|🔴| np.trunc(a) (float64) | float64 | 100,000 | 0.0110 | 0.1040 | 9.50x |
|🟡| np.trunc(a) (float64) | float64 | 10,000,000 | 14.6540 | 20.0110 | 1.37x |

### Reduction

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|✅| np.amax (float32) | float32 | 1,000 | 0.0020 | 0.0010 | 0.47x |
|🟠| np.amax (float32) | float32 | 100,000 | 0.0060 | 0.0140 | 2.30x |
|🟡| np.amax (float32) | float32 | 10,000,000 | 1.4960 | 2.0330 | 1.36x |
|✅| np.amax (float64) | float64 | 1,000 | 0.0020 | 0.0010 | 0.60x |
|🟠| np.amax (float64) | float64 | 100,000 | 0.0100 | 0.0270 | 2.66x |
|🟡| np.amax (float64) | float64 | 10,000,000 | 3.7660 | 4.2960 | 1.14x |
|✅| np.amax (int16) | int16 | 1,000 | 0.0020 | 0.0010 | 0.49x |
|✅| np.amax (int16) | int16 | 100,000 | 0.0030 | 0.0020 | 0.66x |
|🟡| np.amax (int16) | int16 | 10,000,000 | 0.3010 | 0.3400 | 1.13x |
|✅| np.amax (int32) | int32 | 1,000 | 0.0020 | 0.0010 | 0.27x |
|✅| np.amax (int32) | int32 | 100,000 | 0.0040 | 0.0030 | 0.77x |
|🟡| np.amax (int32) | int32 | 10,000,000 | 1.2020 | 1.2200 | 1.01x |
|✅| np.amax (int64) | int64 | 1,000 | 0.0020 | 0.0010 | 0.47x |
|✅| np.amax (int64) | int64 | 100,000 | 0.0090 | 0.0070 | 0.82x |
|🟡| np.amax (int64) | int64 | 10,000,000 | 3.7200 | 3.8740 | 1.04x |
|✅| np.amax (uint16) | uint16 | 1,000 | 0.0020 | 0.0010 | 0.49x |
|✅| np.amax (uint16) | uint16 | 100,000 | 0.0030 | 0.0020 | 0.64x |
|✅| np.amax (uint16) | uint16 | 10,000,000 | 0.3340 | 0.3300 | 0.99x |
|✅| np.amax (uint32) | uint32 | 1,000 | 0.0020 | 0.0010 | 0.48x |
|✅| np.amax (uint32) | uint32 | 100,000 | 0.0040 | 0.0030 | 0.77x |
|✅| np.amax (uint32) | uint32 | 10,000,000 | 1.2650 | 1.2170 | 0.96x |
|✅| np.amax (uint64) | uint64 | 1,000 | 0.0020 | 0.0010 | 0.45x |
|✅| np.amax (uint64) | uint64 | 100,000 | 0.0120 | 0.0100 | 0.81x |
|🟡| np.amax (uint64) | uint64 | 10,000,000 | 4.0730 | 4.0940 | 1.01x |
|✅| np.amax (uint8) | uint8 | 1,000 | 0.0020 | 0.0010 | 0.40x |
|✅| np.amax (uint8) | uint8 | 100,000 | 0.0020 | 0.0010 | 0.49x |
|✅| np.amax (uint8) | uint8 | 10,000,000 | 0.1480 | 0.1470 | 0.99x |
|✅| np.amin (float32) | float32 | 1,000 | 0.0020 | 0.0010 | 0.81x |
|🔴| np.amin (float32) | float32 | 100,000 | 0.0070 | 0.0510 | 7.78x |
|🟠| np.amin (float32) | float32 | 10,000,000 | 1.4830 | 5.2600 | 3.55x |
|🟡| np.amin (float64) | float64 | 1,000 | 0.0020 | 0.0020 | 1.26x |
|🔴| np.amin (float64) | float64 | 100,000 | 0.0100 | 0.0920 | 9.06x |
|🟠| np.amin (float64) | float64 | 10,000,000 | 3.9370 | 10.3290 | 2.62x |
|✅| np.amin (int16) | int16 | 1,000 | 0.0020 | 0.0010 | 0.40x |
|✅| np.amin (int16) | int16 | 100,000 | 0.0040 | 0.0030 | 0.77x |
|🟠| np.amin (int16) | int16 | 10,000,000 | 0.3250 | 0.7560 | 2.33x |
|✅| np.amin (int32) | int32 | 1,000 | 0.0020 | 0.0010 | 0.49x |
|🟡| np.amin (int32) | int32 | 100,000 | 0.0050 | 0.0050 | 1.04x |
|🟠| np.amin (int32) | int32 | 10,000,000 | 1.2310 | 3.5990 | 2.92x |
|✅| np.amin (int64) | int64 | 1,000 | 0.0020 | 0.0010 | 0.62x |
|🟠| np.amin (int64) | int64 | 100,000 | 0.0120 | 0.0280 | 2.27x |
|🟠| np.amin (int64) | int64 | 10,000,000 | 3.6690 | 8.4600 | 2.31x |
|✅| np.amin (uint16) | uint16 | 1,000 | 0.0020 | 0.0010 | 0.52x |
|✅| np.amin (uint16) | uint16 | 100,000 | 0.0030 | 0.0030 | 0.83x |
|🟠| np.amin (uint16) | uint16 | 10,000,000 | 0.3120 | 0.7130 | 2.29x |
|✅| np.amin (uint32) | uint32 | 1,000 | 0.0020 | 0.0010 | 0.49x |
|🟡| np.amin (uint32) | uint32 | 100,000 | 0.0050 | 0.0060 | 1.17x |
|🟠| np.amin (uint32) | uint32 | 10,000,000 | 1.3070 | 3.7320 | 2.86x |
|✅| np.amin (uint64) | uint64 | 1,000 | 0.0020 | 0.0010 | 0.70x |
|🟠| np.amin (uint64) | uint64 | 100,000 | 0.0120 | 0.0360 | 2.98x |
|🟠| np.amin (uint64) | uint64 | 10,000,000 | 3.7870 | 8.9570 | 2.37x |
|✅| np.amin (uint8) | uint8 | 1,000 | 0.0020 | 0.0010 | 0.44x |
|✅| np.amin (uint8) | uint8 | 100,000 | 0.0030 | 0.0020 | 0.76x |
|🟡| np.amin (uint8) | uint8 | 10,000,000 | 0.1500 | 0.2390 | 1.60x |
|🟡| np.argmax (float32) | float32 | 1,000 | 0.0010 | 0.0010 | 1.38x |
|🔴| np.argmax (float32) | float32 | 100,000 | 0.0090 | 0.0560 | 6.42x |
|🟠| np.argmax (float32) | float32 | 10,000,000 | 2.0610 | 5.8120 | 2.82x |
|🟡| np.argmax (float64) | float64 | 1,000 | 0.0010 | 0.0010 | 1.25x |
|🟠| np.argmax (float64) | float64 | 100,000 | 0.0170 | 0.0570 | 3.42x |
|🟡| np.argmax (float64) | float64 | 10,000,000 | 4.9800 | 6.9660 | 1.40x |
|✅| np.argmax (int16) | int16 | 1,000 | 0.0010 | 0.0010 | 0.83x |
|✅| np.argmax (int16) | int16 | 100,000 | 0.0030 | 0.0020 | 0.58x |
|✅| np.argmax (int16) | int16 | 10,000,000 | 0.4150 | 0.3650 | 0.88x |
|✅| np.argmax (int32) | int32 | 1,000 | 0.0010 | 0.0010 | 0.84x |
|✅| np.argmax (int32) | int32 | 100,000 | 0.0060 | 0.0040 | 0.66x |
|✅| np.argmax (int32) | int32 | 10,000,000 | 1.9770 | 1.4310 | 0.72x |
|✅| np.argmax (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 0.98x |
|🟡| np.argmax (int64) | int64 | 100,000 | 0.0140 | 0.0280 | 1.95x |
|🟡| np.argmax (int64) | int64 | 10,000,000 | 4.6600 | 4.7980 | 1.03x |
|✅| np.argmax (uint16) | uint16 | 1,000 | 0.0010 | 0.0010 | 0.82x |
|✅| np.argmax (uint16) | uint16 | 100,000 | 0.0050 | 0.0020 | 0.40x |
|✅| np.argmax (uint16) | uint16 | 10,000,000 | 0.6700 | 0.3820 | 0.57x |
|✅| np.argmax (uint32) | uint32 | 1,000 | 0.0010 | 0.0010 | 0.87x |
|✅| np.argmax (uint32) | uint32 | 100,000 | 0.0090 | 0.0040 | 0.42x |
|✅| np.argmax (uint32) | uint32 | 10,000,000 | 2.0350 | 1.3990 | 0.69x |
|✅| np.argmax (uint64) | uint64 | 1,000 | 0.0010 | 0.0010 | 0.94x |
|🟡| np.argmax (uint64) | uint64 | 100,000 | 0.0210 | 0.0330 | 1.57x |
|🟡| np.argmax (uint64) | uint64 | 10,000,000 | 4.5900 | 5.1930 | 1.13x |
|✅| np.argmax (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 0.68x |
|✅| np.argmax (uint8) | uint8 | 100,000 | 0.0030 | 0.0010 | 0.40x |
|✅| np.argmax (uint8) | uint8 | 10,000,000 | 0.2230 | 0.1490 | 0.67x |
|🟡| np.argmin (float32) | float32 | 1,000 | 0.0010 | 0.0010 | 1.39x |
|🔴| np.argmin (float32) | float32 | 100,000 | 0.0090 | 0.0570 | 6.40x |
|🟠| np.argmin (float32) | float32 | 10,000,000 | 1.9840 | 5.7760 | 2.91x |
|✅| np.argmin (float64) | float64 | 1,000 | 0.0010 | 0.0010 | 1.00x |
|🟠| np.argmin (float64) | float64 | 100,000 | 0.0170 | 0.0570 | 3.45x |
|🟡| np.argmin (float64) | float64 | 10,000,000 | 4.9500 | 6.8670 | 1.39x |
|✅| np.argmin (int16) | int16 | 1,000 | 0.0010 | 0.0010 | 0.72x |
|✅| np.argmin (int16) | int16 | 100,000 | 0.0040 | 0.0020 | 0.58x |
|✅| np.argmin (int16) | int16 | 10,000,000 | 0.5640 | 0.3630 | 0.64x |
|✅| np.argmin (int32) | int32 | 1,000 | 0.0010 | 0.0010 | 0.88x |
|✅| np.argmin (int32) | int32 | 100,000 | 0.0050 | 0.0040 | 0.66x |
|✅| np.argmin (int32) | int32 | 10,000,000 | 2.0520 | 1.3730 | 0.67x |
|✅| np.argmin (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 0.90x |
|🟠| np.argmin (int64) | int64 | 100,000 | 0.0140 | 0.0280 | 2.01x |
|✅| np.argmin (int64) | int64 | 10,000,000 | 4.9670 | 4.6920 | 0.94x |
|✅| np.argmin (uint16) | uint16 | 1,000 | 0.0010 | 0.0010 | 0.85x |
|✅| np.argmin (uint16) | uint16 | 100,000 | 0.0050 | 0.0020 | 0.43x |
|✅| np.argmin (uint16) | uint16 | 10,000,000 | 0.5500 | 0.3750 | 0.68x |
|✅| np.argmin (uint32) | uint32 | 1,000 | 0.0010 | 0.0010 | 0.72x |
|✅| np.argmin (uint32) | uint32 | 100,000 | 0.0090 | 0.0040 | 0.41x |
|✅| np.argmin (uint32) | uint32 | 10,000,000 | 2.0280 | 1.2600 | 0.62x |
|✅| np.argmin (uint64) | uint64 | 1,000 | 0.0010 | 0.0010 | 0.98x |
|🟡| np.argmin (uint64) | uint64 | 100,000 | 0.0170 | 0.0330 | 1.95x |
|🟡| np.argmin (uint64) | uint64 | 10,000,000 | 4.4940 | 5.1460 | 1.15x |
|✅| np.argmin (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 0.83x |
|✅| np.argmin (uint8) | uint8 | 100,000 | 0.0030 | 0.0010 | 0.40x |
|✅| np.argmin (uint8) | uint8 | 10,000,000 | 0.2170 | 0.1480 | 0.68x |
|🔴| np.cumprod(a) (float32) | float32 | 1,000 | 0.0040 | 0.0190 | 5.17x |
|🟡| np.cumprod(a) (float32) | float32 | 100,000 | 0.1710 | 0.2770 | 1.62x |
|🟡| np.cumprod(a) (float32) | float32 | 10,000,000 | 22.7040 | 23.9230 | 1.05x |
|🟠| np.cumprod(a) (float64) | float64 | 1,000 | 0.0040 | 0.0170 | 3.94x |
|🟠| np.cumprod(a) (float64) | float64 | 100,000 | 0.1720 | 0.4220 | 2.45x |
|🟡| np.cumprod(a) (float64) | float64 | 10,000,000 | 25.3610 | 39.2890 | 1.55x |
|✅| np.mean (float32) | float32 | 1,000 | 0.0040 | 0.0010 | 0.21x |
|✅| np.mean (float32) | float32 | 100,000 | 0.0190 | 0.0030 | 0.17x |
|✅| np.mean (float32) | float32 | 10,000,000 | 3.0600 | 1.1000 | 0.36x |
|✅| np.mean (float64) | float64 | 1,000 | 0.0020 | 0.0010 | 0.34x |
|✅| np.mean (float64) | float64 | 100,000 | 0.0180 | 0.0040 | 0.23x |
|✅| np.mean (float64) | float64 | 10,000,000 | 5.0230 | 2.9180 | 0.58x |
|⚪| np.mean (int16) | int16 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (int16) | int16 | 100,000 | 0.0520 | - | - |
|⚪| np.mean (int16) | int16 | 10,000,000 | 5.1280 | - | - |
|✅| np.mean (int32) | int32 | 1,000 | 0.0030 | 0.0010 | 0.40x |
|✅| np.mean (int32) | int32 | 100,000 | 0.0470 | 0.0190 | 0.41x |
|✅| np.mean (int32) | int32 | 10,000,000 | 4.5940 | 2.8230 | 0.61x |
|✅| np.mean (int64) | int64 | 1,000 | 0.0030 | 0.0010 | 0.48x |
|✅| np.mean (int64) | int64 | 100,000 | 0.0340 | 0.0050 | 0.13x |
|✅| np.mean (int64) | int64 | 10,000,000 | 6.3170 | 2.9910 | 0.47x |
|⚪| np.mean (uint16) | uint16 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint16) | uint16 | 100,000 | 0.0550 | - | - |
|⚪| np.mean (uint16) | uint16 | 10,000,000 | 5.0820 | - | - |
|⚪| np.mean (uint32) | uint32 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint32) | uint32 | 100,000 | 0.0400 | - | - |
|⚪| np.mean (uint32) | uint32 | 10,000,000 | 4.7570 | - | - |
|⚪| np.mean (uint64) | uint64 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint64) | uint64 | 100,000 | 0.0520 | - | - |
|⚪| np.mean (uint64) | uint64 | 10,000,000 | 7.8050 | - | - |
|⚪| np.mean (uint8) | uint8 | 1,000 | 0.0030 | - | - |
|⚪| np.mean (uint8) | uint8 | 100,000 | 0.0540 | - | - |
|⚪| np.mean (uint8) | uint8 | 10,000,000 | 5.0080 | - | - |
|✅| np.nanmax(a) (float32) | float32 | 1,000 | 0.0030 | 0.0010 | 0.43x |
|🔴| np.nanmax(a) (float32) | float32 | 100,000 | 0.0080 | 0.0520 | 6.60x |
|🟠| np.nanmax(a) (float32) | float32 | 10,000,000 | 1.4630 | 3.3140 | 2.27x |
|✅| np.nanmax(a) (float64) | float64 | 1,000 | 0.0030 | 0.0020 | 0.65x |
|🔴| np.nanmax(a) (float64) | float64 | 100,000 | 0.0110 | 0.1020 | 8.93x |
|🟡| np.nanmax(a) (float64) | float64 | 10,000,000 | 4.0560 | 6.9620 | 1.72x |
|✅| np.nanmean(a) (float32) | float32 | 1,000 | 0.0100 | 0.0020 | 0.18x |
|✅| np.nanmean(a) (float32) | float32 | 100,000 | 0.0730 | 0.0720 | 0.99x |
|✅| np.nanmean(a) (float32) | float32 | 10,000,000 | 19.8280 | 4.1940 | 0.21x |
|✅| np.nanmean(a) (float64) | float64 | 1,000 | 0.0080 | 0.0020 | 0.22x |
|✅| np.nanmean(a) (float64) | float64 | 100,000 | 0.3220 | 0.0750 | 0.23x |
|✅| np.nanmean(a) (float64) | float64 | 10,000,000 | 33.4660 | 5.6870 | 0.17x |
|✅| np.nanmedian(a) (float32) | float32 | 1,000 | 0.0130 | 0.0040 | 0.28x |
|🟡| np.nanmedian(a) (float32) | float32 | 100,000 | 0.4980 | 0.9640 | 1.94x |
|🟡| np.nanmedian(a) (float32) | float32 | 10,000,000 | 77.8310 | 80.5670 | 1.04x |
|✅| np.nanmedian(a) (float64) | float64 | 1,000 | 0.0120 | 0.0040 | 0.33x |
|🟠| np.nanmedian(a) (float64) | float64 | 100,000 | 0.4840 | 0.9950 | 2.06x |
|✅| np.nanmedian(a) (float64) | float64 | 10,000,000 | 93.1150 | 92.3010 | 0.99x |
|✅| np.nanmin(a) (float32) | float32 | 1,000 | 0.0030 | 0.0010 | 0.42x |
|🔴| np.nanmin(a) (float32) | float32 | 100,000 | 0.0070 | 0.0520 | 7.38x |
|🟠| np.nanmin(a) (float32) | float32 | 10,000,000 | 1.6130 | 3.3610 | 2.08x |
|✅| np.nanmin(a) (float64) | float64 | 1,000 | 0.0030 | 0.0020 | 0.66x |
|🔴| np.nanmin(a) (float64) | float64 | 100,000 | 0.0120 | 0.1020 | 8.85x |
|🟡| np.nanmin(a) (float64) | float64 | 10,000,000 | 4.2490 | 6.9810 | 1.64x |
|✅| np.nanpercentile(a, 50) (float32) | float32 | 1,000 | 0.0260 | 0.0040 | 0.14x |
|🟡| np.nanpercentile(a, 50) (float32) | float32 | 100,000 | 0.7090 | 0.9670 | 1.36x |
|🟡| np.nanpercentile(a, 50) (float32) | float32 | 10,000,000 | 52.5160 | 80.7600 | 1.54x |
|✅| np.nanpercentile(a, 50) (float64) | float64 | 1,000 | 0.0280 | 0.0040 | 0.14x |
|🟡| np.nanpercentile(a, 50) (float64) | float64 | 100,000 | 0.7820 | 1.0270 | 1.31x |
|🟡| np.nanpercentile(a, 50) (float64) | float64 | 10,000,000 | 66.2600 | 90.8810 | 1.37x |
|✅| np.nanprod(a) (float32) | float32 | 1,000 | 0.0050 | 0.0010 | 0.28x |
|✅| np.nanprod(a) (float32) | float32 | 100,000 | 0.0960 | 0.0160 | 0.17x |
|✅| np.nanprod(a) (float32) | float32 | 10,000,000 | 18.5150 | 1.9040 | 0.10x |
|✅| np.nanprod(a) (float64) | float64 | 1,000 | 0.0050 | 0.0010 | 0.20x |
|✅| np.nanprod(a) (float64) | float64 | 100,000 | 0.2870 | 0.0320 | 0.11x |
|✅| np.nanprod(a) (float64) | float64 | 10,000,000 | 27.1780 | 4.5260 | 0.17x |
|✅| np.nanquantile(a, 0.5) (float32) | float32 | 1,000 | 0.0250 | 0.0040 | 0.15x |
|🟡| np.nanquantile(a, 0.5) (float32) | float32 | 100,000 | 0.7370 | 0.9640 | 1.31x |
|🟡| np.nanquantile(a, 0.5) (float32) | float32 | 10,000,000 | 66.8040 | 80.4490 | 1.20x |
|✅| np.nanquantile(a, 0.5) (float64) | float64 | 1,000 | 0.0280 | 0.0040 | 0.13x |
|🟡| np.nanquantile(a, 0.5) (float64) | float64 | 100,000 | 0.7500 | 0.9850 | 1.31x |
|🟡| np.nanquantile(a, 0.5) (float64) | float64 | 10,000,000 | 64.7940 | 90.9810 | 1.40x |
|✅| np.nanstd(a) (float32) | float32 | 1,000 | 0.0200 | 0.0030 | 0.12x |
|✅| np.nanstd(a) (float32) | float32 | 100,000 | 0.1630 | 0.1520 | 0.93x |
|✅| np.nanstd(a) (float32) | float32 | 10,000,000 | 32.7550 | 9.2840 | 0.28x |
|✅| np.nanstd(a) (float64) | float64 | 1,000 | 0.0180 | 0.0020 | 0.13x |
|✅| np.nanstd(a) (float64) | float64 | 100,000 | 0.4570 | 0.1480 | 0.32x |
|✅| np.nanstd(a) (float64) | float64 | 10,000,000 | 52.9160 | 11.4370 | 0.22x |
|✅| np.nansum(a) (float32) | float32 | 1,000 | 0.0040 | 0.0010 | 0.34x |
|✅| np.nansum(a) (float32) | float32 | 100,000 | 0.0320 | 0.0100 | 0.30x |
|✅| np.nansum(a) (float32) | float32 | 10,000,000 | 14.3490 | 1.4880 | 0.10x |
|✅| np.nansum(a) (float64) | float64 | 1,000 | 0.0040 | 0.0010 | 0.37x |
|✅| np.nansum(a) (float64) | float64 | 100,000 | 0.2430 | 0.0190 | 0.08x |
|✅| np.nansum(a) (float64) | float64 | 10,000,000 | 25.5400 | 3.6530 | 0.14x |
|✅| np.nanvar(a) (float32) | float32 | 1,000 | 0.0200 | 0.0030 | 0.13x |
|✅| np.nanvar(a) (float32) | float32 | 100,000 | 0.1730 | 0.1550 | 0.90x |
|✅| np.nanvar(a) (float32) | float32 | 10,000,000 | 33.3950 | 9.2880 | 0.28x |
|✅| np.nanvar(a) (float64) | float64 | 1,000 | 0.0180 | 0.0020 | 0.14x |
|✅| np.nanvar(a) (float64) | float64 | 100,000 | 0.4370 | 0.1530 | 0.35x |
|✅| np.nanvar(a) (float64) | float64 | 10,000,000 | 56.9160 | 11.7840 | 0.21x |
|✅| np.std (float32) | float32 | 1,000 | 0.0080 | 0.0010 | 0.13x |
|✅| np.std (float32) | float32 | 100,000 | 0.0480 | 0.0100 | 0.20x |
|✅| np.std (float32) | float32 | 10,000,000 | 16.7540 | 2.5970 | 0.16x |
|✅| np.std (float64) | float64 | 1,000 | 0.0070 | 0.0010 | 0.13x |
|✅| np.std (float64) | float64 | 100,000 | 0.0580 | 0.0190 | 0.33x |
|✅| np.std (float64) | float64 | 10,000,000 | 32.8480 | 6.7590 | 0.21x |
|✅| np.sum (float32) | float32 | 1,000 | 0.0020 | 0.0010 | 0.42x |
|✅| np.sum (float32) | float32 | 100,000 | 0.0160 | 0.0030 | 0.20x |
|✅| np.sum (float32) | float32 | 10,000,000 | 2.9670 | 1.0550 | 0.36x |
|✅| np.sum (float64) | float64 | 1,000 | 0.0020 | 0.0010 | 0.44x |
|🔴| np.sum (float64) | float64 | 100,000 | 0.0180 | 0.2090 | 11.83x |
|✅| np.sum (float64) | float64 | 10,000,000 | 5.0430 | 3.4960 | 0.69x |
|✅| np.sum (int16) | int16 | 1,000 | 0.0020 | 0.0010 | 0.37x |
|✅| np.sum (int16) | int16 | 100,000 | 0.0330 | 0.0190 | 0.57x |
|✅| np.sum (int16) | int16 | 10,000,000 | 3.3720 | 1.9530 | 0.58x |
|✅| np.sum (int32) | int32 | 1,000 | 0.0020 | 0.0010 | 0.33x |
|✅| np.sum (int32) | int32 | 100,000 | 0.0350 | 0.0190 | 0.54x |
|✅| np.sum (int32) | int32 | 10,000,000 | 4.4800 | 2.7220 | 0.61x |
|✅| np.sum (int64) | int64 | 1,000 | 0.0020 | 0.0010 | 0.44x |
|✅| np.sum (int64) | int64 | 100,000 | 0.0200 | 0.0070 | 0.33x |
|✅| np.sum (int64) | int64 | 10,000,000 | 4.5900 | 2.7890 | 0.61x |
|✅| np.sum (uint16) | uint16 | 1,000 | 0.0020 | 0.0010 | 0.39x |
|✅| np.sum (uint16) | uint16 | 100,000 | 0.0340 | 0.0190 | 0.55x |
|✅| np.sum (uint16) | uint16 | 10,000,000 | 3.3160 | 1.9410 | 0.59x |
|✅| np.sum (uint32) | uint32 | 1,000 | 0.0020 | 0.0010 | 0.38x |
|✅| np.sum (uint32) | uint32 | 100,000 | 0.0330 | 0.0190 | 0.58x |
|✅| np.sum (uint32) | uint32 | 10,000,000 | 4.2660 | 2.6580 | 0.62x |
|✅| np.sum (uint64) | uint64 | 1,000 | 0.0020 | 0.0010 | 0.43x |
|✅| np.sum (uint64) | uint64 | 100,000 | 0.0190 | 0.0060 | 0.33x |
|✅| np.sum (uint64) | uint64 | 10,000,000 | 4.9130 | 2.8030 | 0.57x |
|✅| np.sum (uint8) | uint8 | 1,000 | 0.0020 | 0.0010 | 0.36x |
|✅| np.sum (uint8) | uint8 | 100,000 | 0.0360 | 0.0190 | 0.52x |
|✅| np.sum (uint8) | uint8 | 10,000,000 | 3.2140 | 1.8390 | 0.57x |
|✅| np.sum axis=0 (float32) | float32 | 1,000 | 0.0020 | 0.0010 | 0.36x |
|✅| np.sum axis=0 (float32) | float32 | 100,000 | 0.0080 | 0.0050 | 0.59x |
|✅| np.sum axis=0 (float32) | float32 | 10,000,000 | 1.5600 | 1.3520 | 0.87x |
|✅| np.sum axis=0 (float64) | float64 | 1,000 | 0.0020 | 0.0010 | 0.37x |
|✅| np.sum axis=0 (float64) | float64 | 100,000 | 0.0140 | 0.0100 | 0.71x |
|✅| np.sum axis=0 (float64) | float64 | 10,000,000 | 3.8820 | 3.2510 | 0.84x |
|🟡| np.sum axis=0 (int16) | int16 | 1,000 | 0.0020 | 0.0030 | 1.38x |
|🔴| np.sum axis=0 (int16) | int16 | 100,000 | 0.0470 | 0.4030 | 8.62x |
|🔴| np.sum axis=0 (int16) | int16 | 10,000,000 | 4.6350 | 58.1350 | 12.54x |
|✅| np.sum axis=0 (int32) | int32 | 1,000 | 0.0030 | 0.0010 | 0.31x |
|✅| np.sum axis=0 (int32) | int32 | 100,000 | 0.0500 | 0.0120 | 0.24x |
|🟡| np.sum axis=0 (int32) | int32 | 10,000,000 | 5.4900 | 6.2020 | 1.13x |
|✅| np.sum axis=0 (int64) | int64 | 1,000 | 0.0020 | 0.0010 | 0.37x |
|✅| np.sum axis=0 (int64) | int64 | 100,000 | 0.0270 | 0.0100 | 0.38x |
|✅| np.sum axis=0 (int64) | int64 | 10,000,000 | 5.3570 | 3.2690 | 0.61x |
|🟡| np.sum axis=0 (uint16) | uint16 | 1,000 | 0.0020 | 0.0050 | 1.98x |
|🔴| np.sum axis=0 (uint16) | uint16 | 100,000 | 0.0470 | 0.5050 | 10.70x |
|🔴| np.sum axis=0 (uint16) | uint16 | 10,000,000 | 4.6200 | 71.6940 | 15.52x |
|✅| np.sum axis=0 (uint32) | uint32 | 1,000 | 0.0020 | 0.0010 | 0.34x |
|✅| np.sum axis=0 (uint32) | uint32 | 100,000 | 0.0510 | 0.0120 | 0.24x |
|🟡| np.sum axis=0 (uint32) | uint32 | 10,000,000 | 5.5940 | 5.9810 | 1.07x |
|✅| np.sum axis=0 (uint64) | uint64 | 1,000 | 0.0020 | 0.0010 | 0.39x |
|✅| np.sum axis=0 (uint64) | uint64 | 100,000 | 0.0270 | 0.0100 | 0.38x |
|✅| np.sum axis=0 (uint64) | uint64 | 10,000,000 | 5.6360 | 3.2590 | 0.58x |
|🟡| np.sum axis=0 (uint8) | uint8 | 1,000 | 0.0030 | 0.0050 | 1.89x |
|🔴| np.sum axis=0 (uint8) | uint8 | 100,000 | 0.0490 | 0.4960 | 10.03x |
|🔴| np.sum axis=0 (uint8) | uint8 | 10,000,000 | 4.4070 | 55.3510 | 12.56x |
|✅| np.sum axis=1 (float32) | float32 | 1,000 | 0.0020 | 0.0010 | 0.42x |
|✅| np.sum axis=1 (float32) | float32 | 100,000 | 0.0160 | 0.0030 | 0.21x |
|✅| np.sum axis=1 (float32) | float32 | 10,000,000 | 3.1950 | 1.0780 | 0.34x |
|✅| np.sum axis=1 (float64) | float64 | 1,000 | 0.0020 | 0.0010 | 0.41x |
|✅| np.sum axis=1 (float64) | float64 | 100,000 | 0.0180 | 0.0070 | 0.38x |
|✅| np.sum axis=1 (float64) | float64 | 10,000,000 | 5.3940 | 2.9400 | 0.54x |
|🟡| np.sum axis=1 (int16) | int16 | 1,000 | 0.0020 | 0.0030 | 1.44x |
|🔴| np.sum axis=1 (int16) | int16 | 100,000 | 0.0370 | 0.4070 | 10.95x |
|🔴| np.sum axis=1 (int16) | int16 | 10,000,000 | 3.3820 | 40.8460 | 12.08x |
|✅| np.sum axis=1 (int32) | int32 | 1,000 | 0.0020 | 0.0010 | 0.37x |
|✅| np.sum axis=1 (int32) | int32 | 100,000 | 0.0400 | 0.0160 | 0.40x |
|✅| np.sum axis=1 (int32) | int32 | 10,000,000 | 4.2000 | 1.8990 | 0.45x |
|✅| np.sum axis=1 (int64) | int64 | 1,000 | 0.0020 | 0.0010 | 0.43x |
|✅| np.sum axis=1 (int64) | int64 | 100,000 | 0.0170 | 0.0070 | 0.43x |
|✅| np.sum axis=1 (int64) | int64 | 10,000,000 | 4.5770 | 2.9080 | 0.64x |
|🟠| np.sum axis=1 (uint16) | uint16 | 1,000 | 0.0020 | 0.0050 | 2.05x |
|🔴| np.sum axis=1 (uint16) | uint16 | 100,000 | 0.0400 | 0.4970 | 12.38x |
|🔴| np.sum axis=1 (uint16) | uint16 | 10,000,000 | 3.3650 | 49.8960 | 14.83x |
|✅| np.sum axis=1 (uint32) | uint32 | 1,000 | 0.0020 | 0.0010 | 0.46x |
|🟡| np.sum axis=1 (uint32) | uint32 | 100,000 | 0.0380 | 0.0400 | 1.05x |
|✅| np.sum axis=1 (uint32) | uint32 | 10,000,000 | 4.3360 | 4.0860 | 0.94x |
|✅| np.sum axis=1 (uint64) | uint64 | 1,000 | 0.0020 | 0.0010 | 0.37x |
|✅| np.sum axis=1 (uint64) | uint64 | 100,000 | 0.0180 | 0.0070 | 0.41x |
|✅| np.sum axis=1 (uint64) | uint64 | 10,000,000 | 5.0470 | 2.9710 | 0.59x |
|🟡| np.sum axis=1 (uint8) | uint8 | 1,000 | 0.0020 | 0.0050 | 1.92x |
|🔴| np.sum axis=1 (uint8) | uint8 | 100,000 | 0.0370 | 0.5010 | 13.45x |
|🔴| np.sum axis=1 (uint8) | uint8 | 10,000,000 | 3.1150 | 49.7410 | 15.97x |
|✅| np.var (float32) | float32 | 1,000 | 0.0080 | 0.0010 | 0.10x |
|✅| np.var (float32) | float32 | 100,000 | 0.0480 | 0.0100 | 0.20x |
|✅| np.var (float32) | float32 | 10,000,000 | 16.9570 | 2.6030 | 0.15x |
|✅| np.var (float64) | float64 | 1,000 | 0.0060 | 0.0010 | 0.12x |
|✅| np.var (float64) | float64 | 100,000 | 0.0560 | 0.0190 | 0.34x |
|✅| np.var (float64) | float64 | 10,000,000 | 31.7480 | 6.7140 | 0.21x |

### Broadcasting

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| matrix + col_vector (N,M)+(N,1) | float64 | 1,000 | 0.0010 | - | - |
|⚪| matrix + col_vector (N,M)+(N,1) | float64 | 100,000 | 0.0300 | - | - |
|✅| matrix + col_vector (N,M)+(N,1) | float64 | 10,000,000 | 16.7420 | 14.4530 | 0.86x |
|⚪| matrix + row_vector (N,M)+(M,) | float64 | 1,000 | 0.0010 | - | - |
|⚪| matrix + row_vector (N,M)+(M,) | float64 | 100,000 | 0.0290 | - | - |
|✅| matrix + row_vector (N,M)+(M,) | float64 | 10,000,000 | 16.9730 | 13.4840 | 0.79x |
|⚪| matrix + scalar | float64 | 1,000 | 0.0010 | - | - |
|⚪| matrix + scalar | float64 | 100,000 | 0.0130 | - | - |
|✅| matrix + scalar | float64 | 10,000,000 | 17.0430 | 13.6340 | 0.80x |
|⚪| np.broadcast_to(row, (N,M)) | float64 | 1,000 | 0.0020 | - | - |
|⚪| np.broadcast_to(row, (N,M)) | float64 | 100,000 | 0.0020 | - | - |
|✅| np.broadcast_to(row, (N,M)) | float64 | 10,000,000 | 0.0020 | 0.0010 | 0.31x |

### Creation

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🔴| np.copy (float32) | float32 | 1,000 | 0.0010 | 0.0100 | 16.27x |
|🟠| np.copy (float32) | float32 | 100,000 | 0.0060 | 0.0180 | 2.99x |
|✅| np.copy (float32) | float32 | 10,000,000 | 9.3180 | 5.4430 | 0.58x |
|🔴| np.copy (float64) | float64 | 1,000 | 0.0010 | 0.0200 | 33.78x |
|✅| np.copy (float64) | float64 | 100,000 | 0.0110 | 0.0040 | 0.33x |
|✅| np.copy (float64) | float64 | 10,000,000 | 18.5100 | 0.0040 | 0.00x |
|🔴| np.copy (int32) | int32 | 1,000 | 0.0010 | 0.0090 | 14.88x |
|🟠| np.copy (int32) | int32 | 100,000 | 0.0060 | 0.0230 | 3.83x |
|✅| np.copy (int32) | int32 | 10,000,000 | 6.6280 | 5.4290 | 0.82x |
|🔴| np.copy (int64) | int64 | 1,000 | 0.0010 | 0.0190 | 31.76x |
|🟠| np.copy (int64) | int64 | 100,000 | 0.0110 | 0.0360 | 3.14x |
|✅| np.copy (int64) | int64 | 10,000,000 | 18.5220 | 11.1760 | 0.60x |
|🔴| np.empty (float32) | float32 | 1,000 | 0.0000 | 0.0080 | 23.03x |
|🔴| np.empty (float32) | float32 | 100,000 | 0.0000 | 0.0050 | 14.79x |
|✅| np.empty (float32) | float32 | 10,000,000 | 0.0200 | 0.0080 | 0.39x |
|🔴| np.empty (float64) | float64 | 1,000 | 0.0000 | 0.0080 | 27.82x |
|🔴| np.empty (float64) | float64 | 100,000 | 0.0000 | 0.0060 | 21.20x |
|⚪| np.empty (float64) | float64 | 10,000,000 | 0.0100 | - | - |
|🔴| np.empty (int32) | int32 | 1,000 | 0.0000 | 0.0080 | 23.08x |
|🔴| np.empty (int32) | int32 | 100,000 | 0.0000 | 0.0130 | 40.73x |
|✅| np.empty (int32) | int32 | 10,000,000 | 0.0110 | 0.0070 | 0.62x |
|🔴| np.empty (int64) | int64 | 1,000 | 0.0000 | 0.0100 | 30.74x |
|🔴| np.empty (int64) | int64 | 100,000 | 0.0000 | 0.0090 | 28.37x |
|⚪| np.empty (int64) | int64 | 10,000,000 | 0.0210 | - | - |
|🔴| np.full (float32) | float32 | 1,000 | 0.0010 | 0.0070 | 6.91x |
|🟠| np.full (float32) | float32 | 100,000 | 0.0050 | 0.0180 | 3.27x |
|✅| np.full (float32) | float32 | 10,000,000 | 9.6280 | 5.7900 | 0.60x |
|🔴| np.full (float64) | float64 | 1,000 | 0.0010 | 0.0120 | 13.84x |
|🟠| np.full (float64) | float64 | 100,000 | 0.0100 | 0.0300 | 3.09x |
|✅| np.full (float64) | float64 | 10,000,000 | 18.7710 | 10.9590 | 0.58x |
|🔴| np.full (int32) | int32 | 1,000 | 0.0010 | 0.0080 | 9.27x |
|🟠| np.full (int32) | int32 | 100,000 | 0.0050 | 0.0200 | 3.77x |
|✅| np.full (int32) | int32 | 10,000,000 | 7.5840 | 5.6050 | 0.74x |
|🔴| np.full (int64) | int64 | 1,000 | 0.0010 | 0.0130 | 14.83x |
|🟠| np.full (int64) | int64 | 100,000 | 0.0100 | 0.0300 | 3.05x |
|✅| np.full (int64) | int64 | 10,000,000 | 18.6310 | 10.6740 | 0.57x |
|🔴| np.ones (float32) | float32 | 1,000 | 0.0010 | 0.0090 | 8.99x |
|🟠| np.ones (float32) | float32 | 100,000 | 0.0050 | 0.0170 | 3.31x |
|✅| np.ones (float32) | float32 | 10,000,000 | 9.3400 | 5.8110 | 0.62x |
|🔴| np.ones (float64) | float64 | 1,000 | 0.0010 | 0.0140 | 16.33x |
|🟠| np.ones (float64) | float64 | 100,000 | 0.0100 | 0.0290 | 2.99x |
|✅| np.ones (float64) | float64 | 10,000,000 | 18.3870 | 10.9120 | 0.59x |
|🔴| np.ones (int32) | int32 | 1,000 | 0.0010 | 0.0090 | 10.05x |
|🟠| np.ones (int32) | int32 | 100,000 | 0.0050 | 0.0190 | 3.42x |
|✅| np.ones (int32) | int32 | 10,000,000 | 7.4070 | 5.6580 | 0.76x |
|🔴| np.ones (int64) | int64 | 1,000 | 0.0010 | 0.0150 | 17.96x |
|🟠| np.ones (int64) | int64 | 100,000 | 0.0100 | 0.0300 | 3.04x |
|✅| np.ones (int64) | int64 | 10,000,000 | 15.1810 | 10.6320 | 0.70x |
|🔴| np.zeros (float32) | float32 | 1,000 | 0.0000 | 0.0080 | 19.65x |
|🟠| np.zeros (float32) | float32 | 100,000 | 0.0050 | 0.0180 | 3.63x |
|🔴| np.zeros (float32) | float32 | 10,000,000 | 0.0170 | 5.6730 | 334.03x |
|🔴| np.zeros (float64) | float64 | 1,000 | 0.0000 | 0.0100 | 26.85x |
|🟠| np.zeros (float64) | float64 | 100,000 | 0.0090 | 0.0300 | 3.25x |
|🔴| np.zeros (float64) | float64 | 10,000,000 | 0.0210 | 10.7550 | 507.65x |
|🔴| np.zeros (int32) | int32 | 1,000 | 0.0000 | 0.0080 | 20.60x |
|🟠| np.zeros (int32) | int32 | 100,000 | 0.0050 | 0.0190 | 3.67x |
|🔴| np.zeros (int32) | int32 | 10,000,000 | 0.0110 | 5.6220 | 518.20x |
|🔴| np.zeros (int64) | int64 | 1,000 | 0.0000 | 0.0090 | 24.68x |
|🟠| np.zeros (int64) | int64 | 100,000 | 0.0090 | 0.0300 | 3.25x |
|🔴| np.zeros (int64) | int64 | 10,000,000 | 0.0120 | 10.7470 | 879.57x |
|🔴| np.zeros_like (float32) | float32 | 1,000 | 0.0010 | 0.0090 | 8.88x |
|🟠| np.zeros_like (float32) | float32 | 100,000 | 0.0050 | 0.0170 | 3.20x |
|✅| np.zeros_like (float32) | float32 | 10,000,000 | 9.3810 | 5.6110 | 0.60x |
|🔴| np.zeros_like (float64) | float64 | 1,000 | 0.0010 | 0.0090 | 8.66x |
|🟠| np.zeros_like (float64) | float64 | 100,000 | 0.0100 | 0.0310 | 3.15x |
|✅| np.zeros_like (float64) | float64 | 10,000,000 | 18.4490 | 10.7070 | 0.58x |
|🟠| np.zeros_like (int32) | int32 | 1,000 | 0.0010 | 0.0060 | 4.55x |
|🟠| np.zeros_like (int32) | int32 | 100,000 | 0.0050 | 0.0180 | 3.29x |
|✅| np.zeros_like (int32) | int32 | 10,000,000 | 7.6600 | 5.6190 | 0.73x |
|🔴| np.zeros_like (int64) | int64 | 1,000 | 0.0010 | 0.0090 | 8.19x |
|🟠| np.zeros_like (int64) | int64 | 100,000 | 0.0100 | 0.0320 | 3.11x |
|✅| np.zeros_like (int64) | int64 | 10,000,000 | 19.3510 | 10.7450 | 0.56x |

### Manipulation

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| a.T (2D) | float64 | 1,000 | 0.0000 | - | - |
|⚪| a.T (2D) | float64 | 100,000 | 0.0000 | - | - |
|⚪| a.T (2D) | float64 | 10,000,000 | 0.0000 | - | - |
|⚪| a.flatten | float64 | 1,000 | 0.0010 | - | - |
|⚪| a.flatten | float64 | 100,000 | 0.0110 | - | - |
|⚪| a.flatten | float64 | 10,000,000 | 13.5030 | - | - |
|⚪| np.concatenate | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.concatenate | float64 | 100,000 | 0.3070 | - | - |
|⚪| np.concatenate | float64 | 10,000,000 | 39.3040 | - | - |
|⚪| np.ravel | float64 | 1,000 | 0.0000 | - | - |
|🟡| np.ravel | float64 | 100,000 | 0.0000 | 0.0010 | 1.63x |
|🟡| np.ravel | float64 | 10,000,000 | 0.0000 | 0.0000 | 1.44x |
|⚪| np.stack | float64 | 1,000 | 0.0020 | - | - |
|⚪| np.stack | float64 | 100,000 | 0.3300 | - | - |
|⚪| np.stack | float64 | 10,000,000 | 44.9540 | - | - |
|⚪| np.transpose (2D) | float64 | 1,000 | 0.0000 | - | - |
|⚪| np.transpose (2D) | float64 | 100,000 | 0.0000 | - | - |
|⚪| np.transpose (2D) | float64 | 10,000,000 | 0.0000 | - | - |
|⚪| reshape 1D->2D | float64 | 1,000 | 0.0000 | - | - |
|⚪| reshape 1D->2D | float64 | 100,000 | 0.0000 | - | - |
|⚪| reshape 1D->2D | float64 | 10,000,000 | 0.0000 | - | - |
|⚪| reshape 2D->1D | float64 | 1,000 | 0.0000 | - | - |
|⚪| reshape 2D->1D | float64 | 100,000 | 0.0000 | - | - |
|⚪| reshape 2D->1D | float64 | 10,000,000 | 0.0000 | - | - |

### Slicing

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| a[100:1000] (contiguous) | float64 | 1,000 | 0.0000 | - | - |
|🔴| a[100:1000] (contiguous) | float64 | 100,000 | 0.0000 | 0.0010 | 7.33x |
|🔴| a[100:1000] (contiguous) | float64 | 10,000,000 | 0.0000 | 0.0010 | 8.96x |
|⚪| a[::-1] (reversed) | float64 | 1,000 | 0.0000 | - | - |
|🔴| a[::-1] (reversed) | float64 | 100,000 | 0.0000 | 0.0010 | 7.72x |
|🔴| a[::-1] (reversed) | float64 | 10,000,000 | 0.0000 | 0.0010 | 9.76x |
|⚪| a[::2] (strided) | float64 | 1,000 | 0.0000 | - | - |
|🔴| a[::2] (strided) | float64 | 100,000 | 0.0000 | 0.0010 | 8.64x |
|🔴| a[::2] (strided) | float64 | 10,000,000 | 0.0000 | 0.0010 | 8.61x |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0020 | - | - |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0020 | - | - |
|⚪| np.sum(contiguous_slice) | float64 | 900 | 0.0020 | - | - |
|⚪| np.sum(strided_slice) | float64 | 500 | 0.0020 | - | - |
|⚪| np.sum(strided_slice) | float64 | 50,000 | 0.0100 | - | - |
|⚪| np.sum(strided_slice) | float64 | 5,000,000 | 4.9330 | - | - |

### Comparison

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟡| a != b (float32) | float32 | 1,000 | 0.0000 | 0.0010 | 1.38x |
|🟠| a != b (float32) | float32 | 100,000 | 0.0050 | 0.0160 | 2.95x |
|✅| a != b (float32) | float32 | 10,000,000 | 9.7860 | 4.0480 | 0.41x |
|🟡| a != b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.51x |
|🟠| a != b (float64) | float64 | 100,000 | 0.0100 | 0.0230 | 2.24x |
|✅| a != b (float64) | float64 | 10,000,000 | 18.4550 | 6.6070 | 0.36x |
|🟡| a != b (int32) | int32 | 1,000 | 0.0010 | 0.0010 | 1.30x |
|🟠| a != b (int32) | int32 | 100,000 | 0.0070 | 0.0180 | 2.58x |
|✅| a != b (int32) | int32 | 10,000,000 | 4.6310 | 4.0590 | 0.88x |
|🟡| a != b (int64) | int64 | 1,000 | 0.0000 | 0.0010 | 1.56x |
|🟠| a != b (int64) | int64 | 100,000 | 0.0130 | 0.0280 | 2.20x |
|✅| a != b (int64) | int64 | 10,000,000 | 7.2080 | 6.5910 | 0.91x |
|🟡| a < b (float32) | float32 | 1,000 | 0.0000 | 0.0010 | 1.69x |
|🟠| a < b (float32) | float32 | 100,000 | 0.0060 | 0.0140 | 2.56x |
|✅| a < b (float32) | float32 | 10,000,000 | 10.3520 | 3.9580 | 0.38x |
|🟡| a < b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.65x |
|🟠| a < b (float64) | float64 | 100,000 | 0.0110 | 0.0230 | 2.13x |
|✅| a < b (float64) | float64 | 10,000,000 | 18.6800 | 6.4870 | 0.35x |
|🟡| a < b (int32) | int32 | 1,000 | 0.0000 | 0.0010 | 1.62x |
|🟠| a < b (int32) | int32 | 100,000 | 0.0070 | 0.0170 | 2.41x |
|✅| a < b (int32) | int32 | 10,000,000 | 4.5830 | 3.9640 | 0.86x |
|🟡| a < b (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 1.28x |
|🟡| a < b (int64) | int64 | 100,000 | 0.0180 | 0.0260 | 1.46x |
|✅| a < b (int64) | int64 | 10,000,000 | 13.4370 | 6.6690 | 0.50x |
|🟡| a <= b (float32) | float32 | 1,000 | 0.0000 | 0.0010 | 1.94x |
|🟠| a <= b (float32) | float32 | 100,000 | 0.0060 | 0.0140 | 2.38x |
|✅| a <= b (float32) | float32 | 10,000,000 | 10.5600 | 3.9350 | 0.37x |
|🟡| a <= b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.60x |
|🟡| a <= b (float64) | float64 | 100,000 | 0.0100 | 0.0200 | 1.95x |
|✅| a <= b (float64) | float64 | 10,000,000 | 18.1660 | 6.4960 | 0.36x |
|🟡| a <= b (int32) | int32 | 1,000 | 0.0000 | 0.0010 | 1.52x |
|🟠| a <= b (int32) | int32 | 100,000 | 0.0070 | 0.0170 | 2.39x |
|✅| a <= b (int32) | int32 | 10,000,000 | 4.4350 | 4.1130 | 0.93x |
|🟡| a <= b (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 1.23x |
|🟡| a <= b (int64) | int64 | 100,000 | 0.0180 | 0.0280 | 1.53x |
|✅| a <= b (int64) | int64 | 10,000,000 | 18.9500 | 6.8390 | 0.36x |
|🟡| a == b (float32) | float32 | 1,000 | 0.0000 | 0.0010 | 1.57x |
|🟠| a == b (float32) | float32 | 100,000 | 0.0050 | 0.0160 | 2.98x |
|✅| a == b (float32) | float32 | 10,000,000 | 10.4600 | 3.9620 | 0.38x |
|🟡| a == b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.60x |
|🟠| a == b (float64) | float64 | 100,000 | 0.0100 | 0.0250 | 2.46x |
|✅| a == b (float64) | float64 | 10,000,000 | 18.0360 | 6.5660 | 0.36x |
|🟡| a == b (int32) | int32 | 1,000 | 0.0000 | 0.0010 | 1.72x |
|🟠| a == b (int32) | int32 | 100,000 | 0.0070 | 0.0160 | 2.28x |
|✅| a == b (int32) | int32 | 10,000,000 | 4.5820 | 3.9850 | 0.87x |
|🟡| a == b (int64) | int64 | 1,000 | 0.0000 | 0.0010 | 1.47x |
|🟠| a == b (int64) | int64 | 100,000 | 0.0130 | 0.0260 | 2.10x |
|✅| a == b (int64) | int64 | 10,000,000 | 6.9800 | 6.4800 | 0.93x |
|🟡| a > b (float32) | float32 | 1,000 | 0.0000 | 0.0010 | 1.65x |
|🟠| a > b (float32) | float32 | 100,000 | 0.0060 | 0.0140 | 2.53x |
|✅| a > b (float32) | float32 | 10,000,000 | 10.1550 | 3.9550 | 0.39x |
|🟡| a > b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.53x |
|🟠| a > b (float64) | float64 | 100,000 | 0.0100 | 0.0250 | 2.42x |
|✅| a > b (float64) | float64 | 10,000,000 | 19.3510 | 6.4690 | 0.33x |
|🟡| a > b (int32) | int32 | 1,000 | 0.0000 | 0.0010 | 1.57x |
|🟠| a > b (int32) | int32 | 100,000 | 0.0070 | 0.0170 | 2.46x |
|✅| a > b (int32) | int32 | 10,000,000 | 4.2010 | 3.9660 | 0.94x |
|🟡| a > b (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 1.29x |
|🟡| a > b (int64) | int64 | 100,000 | 0.0180 | 0.0270 | 1.47x |
|✅| a > b (int64) | int64 | 10,000,000 | 19.0730 | 6.6290 | 0.35x |
|🟡| a >= b (float32) | float32 | 1,000 | 0.0010 | 0.0010 | 1.30x |
|🟠| a >= b (float32) | float32 | 100,000 | 0.0060 | 0.0140 | 2.47x |
|✅| a >= b (float32) | float32 | 10,000,000 | 9.9410 | 3.9780 | 0.40x |
|🟡| a >= b (float64) | float64 | 1,000 | 0.0000 | 0.0010 | 1.56x |
|🟠| a >= b (float64) | float64 | 100,000 | 0.0100 | 0.0250 | 2.45x |
|✅| a >= b (float64) | float64 | 10,000,000 | 19.0130 | 6.5440 | 0.34x |
|🟡| a >= b (int32) | int32 | 1,000 | 0.0000 | 0.0010 | 1.58x |
|🟠| a >= b (int32) | int32 | 100,000 | 0.0070 | 0.0160 | 2.34x |
|✅| a >= b (int32) | int32 | 10,000,000 | 4.4060 | 4.0360 | 0.92x |
|🟡| a >= b (int64) | int64 | 1,000 | 0.0010 | 0.0010 | 1.25x |
|🟡| a >= b (int64) | int64 | 100,000 | 0.0180 | 0.0270 | 1.48x |
|✅| a >= b (int64) | int64 | 10,000,000 | 20.9540 | 6.6940 | 0.32x |

### Bitwise

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟠| a & b (bool) | bool | 1,000 | 0.0000 | 0.0010 | 3.22x |
|🔴| a & b (bool) | bool | 100,000 | 0.0030 | 0.0230 | 6.88x |
|🟡| a & b (bool) | bool | 10,000,000 | 2.0030 | 2.7890 | 1.39x |
|🔴| a & b (int16) | int16 | 1,000 | 0.0010 | 0.0050 | 5.96x |
|✅| a & b (int16) | int16 | 100,000 | 0.0290 | 0.0100 | 0.34x |
|✅| a & b (int16) | int16 | 10,000,000 | 9.2630 | 3.7950 | 0.41x |
|🔴| a & b (int32) | int32 | 1,000 | 0.0010 | 0.0050 | 6.84x |
|✅| a & b (int32) | int32 | 100,000 | 0.0320 | 0.0210 | 0.65x |
|✅| a & b (int32) | int32 | 10,000,000 | 16.8460 | 7.6040 | 0.45x |
|🔴| a & b (int64) | int64 | 1,000 | 0.0010 | 0.0120 | 15.18x |
|🟡| a & b (int64) | int64 | 100,000 | 0.0350 | 0.0450 | 1.28x |
|✅| a & b (int64) | int64 | 10,000,000 | 37.9420 | 14.9280 | 0.39x |
|🔴| a & b (uint16) | uint16 | 1,000 | 0.0010 | 0.0040 | 5.08x |
|✅| a & b (uint16) | uint16 | 100,000 | 0.0310 | 0.0110 | 0.34x |
|✅| a & b (uint16) | uint16 | 10,000,000 | 9.5080 | 3.7950 | 0.40x |
|🔴| a & b (uint32) | uint32 | 1,000 | 0.0010 | 0.0080 | 10.36x |
|✅| a & b (uint32) | uint32 | 100,000 | 0.0330 | 0.0210 | 0.63x |
|✅| a & b (uint32) | uint32 | 10,000,000 | 18.7330 | 7.6040 | 0.41x |
|🔴| a & b (uint64) | uint64 | 1,000 | 0.0010 | 0.0090 | 11.97x |
|🟡| a & b (uint64) | uint64 | 100,000 | 0.0360 | 0.0440 | 1.23x |
|✅| a & b (uint64) | uint64 | 10,000,000 | 39.5690 | 15.0470 | 0.38x |
|🟡| a & b (uint8) | uint8 | 1,000 | 0.0010 | 0.0020 | 1.88x |
|✅| a & b (uint8) | uint8 | 100,000 | 0.0290 | 0.0060 | 0.21x |
|✅| a & b (uint8) | uint8 | 10,000,000 | 3.8970 | 1.8610 | 0.48x |
|🟠| a ^ b (bool) | bool | 1,000 | 0.0000 | 0.0010 | 3.46x |
|🔴| a ^ b (bool) | bool | 100,000 | 0.0030 | 0.0220 | 7.20x |
|🟡| a ^ b (bool) | bool | 10,000,000 | 1.8490 | 2.8190 | 1.52x |
|🟠| a ^ b (int16) | int16 | 1,000 | 0.0010 | 0.0030 | 3.49x |
|✅| a ^ b (int16) | int16 | 100,000 | 0.0280 | 0.0100 | 0.33x |
|✅| a ^ b (int16) | int16 | 10,000,000 | 9.7270 | 3.7540 | 0.39x |
|🔴| a ^ b (int32) | int32 | 1,000 | 0.0010 | 0.0080 | 10.44x |
|✅| a ^ b (int32) | int32 | 100,000 | 0.0290 | 0.0220 | 0.76x |
|✅| a ^ b (int32) | int32 | 10,000,000 | 17.4190 | 7.5250 | 0.43x |
|🔴| a ^ b (int64) | int64 | 1,000 | 0.0010 | 0.0090 | 12.04x |
|🟡| a ^ b (int64) | int64 | 100,000 | 0.0360 | 0.0450 | 1.25x |
|✅| a ^ b (int64) | int64 | 10,000,000 | 33.1730 | 14.8140 | 0.45x |
|🟠| a ^ b (uint16) | uint16 | 1,000 | 0.0010 | 0.0030 | 4.11x |
|✅| a ^ b (uint16) | uint16 | 100,000 | 0.0290 | 0.0110 | 0.37x |
|✅| a ^ b (uint16) | uint16 | 10,000,000 | 9.3020 | 3.7740 | 0.41x |
|🔴| a ^ b (uint32) | uint32 | 1,000 | 0.0010 | 0.0060 | 7.68x |
|✅| a ^ b (uint32) | uint32 | 100,000 | 0.0290 | 0.0210 | 0.74x |
|✅| a ^ b (uint32) | uint32 | 10,000,000 | 18.9510 | 7.5650 | 0.40x |
|🔴| a ^ b (uint64) | uint64 | 1,000 | 0.0010 | 0.0120 | 15.80x |
|🟡| a ^ b (uint64) | uint64 | 100,000 | 0.0360 | 0.0430 | 1.21x |
|✅| a ^ b (uint64) | uint64 | 10,000,000 | 42.5120 | 15.2940 | 0.36x |
|🟡| a ^ b (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 1.98x |
|✅| a ^ b (uint8) | uint8 | 100,000 | 0.0290 | 0.0070 | 0.24x |
|✅| a ^ b (uint8) | uint8 | 10,000,000 | 6.5360 | 1.8190 | 0.28x |
|🟠| a | b (bool) | bool | 1,000 | 0.0000 | 0.0020 | 4.21x |
|🔴| a | b (bool) | bool | 100,000 | 0.0030 | 0.0240 | 8.41x |
|🟡| a | b (bool) | bool | 10,000,000 | 1.8630 | 3.2310 | 1.73x |
|🟠| a | b (int16) | int16 | 1,000 | 0.0010 | 0.0030 | 3.80x |
|✅| a | b (int16) | int16 | 100,000 | 0.0280 | 0.0110 | 0.39x |
|✅| a | b (int16) | int16 | 10,000,000 | 9.4170 | 3.7610 | 0.40x |
|🔴| a | b (int32) | int32 | 1,000 | 0.0010 | 0.0080 | 10.50x |
|✅| a | b (int32) | int32 | 100,000 | 0.0300 | 0.0220 | 0.73x |
|✅| a | b (int32) | int32 | 10,000,000 | 16.5890 | 7.5210 | 0.45x |
|🔴| a | b (int64) | int64 | 1,000 | 0.0010 | 0.0130 | 16.47x |
|🟡| a | b (int64) | int64 | 100,000 | 0.0370 | 0.0430 | 1.17x |
|✅| a | b (int64) | int64 | 10,000,000 | 36.1760 | 14.8250 | 0.41x |
|🟠| a | b (uint16) | uint16 | 1,000 | 0.0010 | 0.0030 | 4.07x |
|✅| a | b (uint16) | uint16 | 100,000 | 0.0290 | 0.0110 | 0.39x |
|✅| a | b (uint16) | uint16 | 10,000,000 | 9.4580 | 3.7900 | 0.40x |
|🔴| a | b (uint32) | uint32 | 1,000 | 0.0010 | 0.0070 | 8.20x |
|✅| a | b (uint32) | uint32 | 100,000 | 0.0290 | 0.0190 | 0.68x |
|✅| a | b (uint32) | uint32 | 10,000,000 | 19.7630 | 7.5860 | 0.38x |
|🔴| a | b (uint64) | uint64 | 1,000 | 0.0010 | 0.0130 | 15.71x |
|🟡| a | b (uint64) | uint64 | 100,000 | 0.0350 | 0.0450 | 1.28x |
|✅| a | b (uint64) | uint64 | 10,000,000 | 38.8890 | 15.0790 | 0.39x |
|🟡| a | b (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 1.65x |
|✅| a | b (uint8) | uint8 | 100,000 | 0.0300 | 0.0060 | 0.21x |
|✅| a | b (uint8) | uint8 | 10,000,000 | 4.3230 | 1.8380 | 0.43x |
|🟠| np.invert(a) (bool) | bool | 1,000 | 0.0000 | 0.0020 | 4.55x |
|🔴| np.invert(a) (bool) | bool | 100,000 | 0.0030 | 0.0240 | 9.32x |
|🟡| np.invert(a) (bool) | bool | 10,000,000 | 1.6920 | 3.0190 | 1.78x |
|🟠| np.invert(a) (int16) | int16 | 1,000 | 0.0010 | 0.0030 | 4.73x |
|✅| np.invert(a) (int16) | int16 | 100,000 | 0.0260 | 0.0100 | 0.39x |
|✅| np.invert(a) (int16) | int16 | 10,000,000 | 7.9030 | 3.3960 | 0.43x |
|🔴| np.invert(a) (int32) | int32 | 1,000 | 0.0010 | 0.0070 | 9.64x |
|✅| np.invert(a) (int32) | int32 | 100,000 | 0.0350 | 0.0200 | 0.58x |
|✅| np.invert(a) (int32) | int32 | 10,000,000 | 14.1460 | 7.1420 | 0.50x |
|🔴| np.invert(a) (int64) | int64 | 1,000 | 0.0010 | 0.0090 | 12.04x |
|🟡| np.invert(a) (int64) | int64 | 100,000 | 0.0260 | 0.0460 | 1.76x |
|✅| np.invert(a) (int64) | int64 | 10,000,000 | 26.1530 | 13.5610 | 0.52x |
|🟠| np.invert(a) (uint16) | uint16 | 1,000 | 0.0010 | 0.0030 | 4.09x |
|✅| np.invert(a) (uint16) | uint16 | 100,000 | 0.0360 | 0.0100 | 0.29x |
|✅| np.invert(a) (uint16) | uint16 | 10,000,000 | 6.7180 | 3.4010 | 0.51x |
|🔴| np.invert(a) (uint32) | uint32 | 1,000 | 0.0010 | 0.0100 | 12.92x |
|✅| np.invert(a) (uint32) | uint32 | 100,000 | 0.0340 | 0.0200 | 0.59x |
|✅| np.invert(a) (uint32) | uint32 | 10,000,000 | 13.9400 | 6.9700 | 0.50x |
|🔴| np.invert(a) (uint64) | uint64 | 1,000 | 0.0010 | 0.0100 | 13.45x |
|🟡| np.invert(a) (uint64) | uint64 | 100,000 | 0.0260 | 0.0390 | 1.48x |
|✅| np.invert(a) (uint64) | uint64 | 10,000,000 | 33.5030 | 13.6430 | 0.41x |
|🟠| np.invert(a) (uint8) | uint8 | 1,000 | 0.0010 | 0.0010 | 2.18x |
|✅| np.invert(a) (uint8) | uint8 | 100,000 | 0.0260 | 0.0070 | 0.27x |
|✅| np.invert(a) (uint8) | uint8 | 10,000,000 | 5.7390 | 1.6900 | 0.29x |
|⚪| np.left_shift(a, 2) (bool) | bool | 1,000 | 0.0010 | - | - |
|⚪| np.left_shift(a, 2) (bool) | bool | 100,000 | 0.1930 | - | - |
|⚪| np.left_shift(a, 2) (bool) | bool | 10,000,000 | 15.1280 | - | - |
|🔴| np.left_shift(a, 2) (int16) | int16 | 1,000 | 0.0010 | 0.0100 | 9.55x |
|🟠| np.left_shift(a, 2) (int16) | int16 | 100,000 | 0.0290 | 0.0640 | 2.21x |
|🟡| np.left_shift(a, 2) (int16) | int16 | 10,000,000 | 7.5460 | 11.1720 | 1.48x |
|🔴| np.left_shift(a, 2) (int32) | int32 | 1,000 | 0.0010 | 0.0140 | 14.48x |
|🟠| np.left_shift(a, 2) (int32) | int32 | 100,000 | 0.0190 | 0.0660 | 3.42x |
|✅| np.left_shift(a, 2) (int32) | int32 | 10,000,000 | 14.7610 | 13.8050 | 0.94x |
|🔴| np.left_shift(a, 2) (int64) | int64 | 1,000 | 0.0010 | 0.0200 | 19.11x |
|🟠| np.left_shift(a, 2) (int64) | int64 | 100,000 | 0.0200 | 0.0810 | 4.04x |
|✅| np.left_shift(a, 2) (int64) | int64 | 10,000,000 | 25.6760 | 19.0870 | 0.74x |
|🔴| np.left_shift(a, 2) (uint16) | uint16 | 1,000 | 0.0010 | 0.0100 | 9.67x |
|🟠| np.left_shift(a, 2) (uint16) | uint16 | 100,000 | 0.0290 | 0.0630 | 2.15x |
|🟡| np.left_shift(a, 2) (uint16) | uint16 | 10,000,000 | 7.6830 | 11.1920 | 1.46x |
|🔴| np.left_shift(a, 2) (uint32) | uint32 | 1,000 | 0.0010 | 0.0090 | 8.93x |
|🟠| np.left_shift(a, 2) (uint32) | uint32 | 100,000 | 0.0200 | 0.0650 | 3.30x |
|✅| np.left_shift(a, 2) (uint32) | uint32 | 10,000,000 | 15.2250 | 13.5980 | 0.89x |
|🔴| np.left_shift(a, 2) (uint64) | uint64 | 1,000 | 0.0010 | 0.0240 | 24.86x |
|🟠| np.left_shift(a, 2) (uint64) | uint64 | 100,000 | 0.0190 | 0.0730 | 3.83x |
|✅| np.left_shift(a, 2) (uint64) | uint64 | 10,000,000 | 34.3970 | 19.0900 | 0.55x |
|🔴| np.left_shift(a, 2) (uint8) | uint8 | 1,000 | 0.0010 | 0.0060 | 6.22x |
|🟠| np.left_shift(a, 2) (uint8) | uint8 | 100,000 | 0.0280 | 0.0630 | 2.24x |
|🟡| np.left_shift(a, 2) (uint8) | uint8 | 10,000,000 | 6.2090 | 10.3010 | 1.66x |
|⚪| np.right_shift(a, 2) (bool) | bool | 1,000 | 0.0020 | - | - |
|⚪| np.right_shift(a, 2) (bool) | bool | 100,000 | 0.1880 | - | - |
|⚪| np.right_shift(a, 2) (bool) | bool | 10,000,000 | 15.2910 | - | - |
|🔴| np.right_shift(a, 2) (int16) | int16 | 1,000 | 0.0010 | 0.0090 | 7.54x |
|🟡| np.right_shift(a, 2) (int16) | int16 | 100,000 | 0.0380 | 0.0660 | 1.76x |
|🟡| np.right_shift(a, 2) (int16) | int16 | 10,000,000 | 9.3590 | 11.1880 | 1.20x |
|🔴| np.right_shift(a, 2) (int32) | int32 | 1,000 | 0.0010 | 0.0170 | 15.61x |
|🟠| np.right_shift(a, 2) (int32) | int32 | 100,000 | 0.0280 | 0.0660 | 2.34x |
|✅| np.right_shift(a, 2) (int32) | int32 | 10,000,000 | 15.3180 | 13.4880 | 0.88x |
|🔴| np.right_shift(a, 2) (int64) | int64 | 1,000 | 0.0010 | 0.0200 | 19.20x |
|🟠| np.right_shift(a, 2) (int64) | int64 | 100,000 | 0.0300 | 0.0770 | 2.62x |
|✅| np.right_shift(a, 2) (int64) | int64 | 10,000,000 | 31.6270 | 19.1640 | 0.61x |
|🔴| np.right_shift(a, 2) (uint16) | uint16 | 1,000 | 0.0010 | 0.0090 | 7.74x |
|🟠| np.right_shift(a, 2) (uint16) | uint16 | 100,000 | 0.0290 | 0.0640 | 2.19x |
|🟡| np.right_shift(a, 2) (uint16) | uint16 | 10,000,000 | 7.1670 | 11.3780 | 1.59x |
|🔴| np.right_shift(a, 2) (uint32) | uint32 | 1,000 | 0.0010 | 0.0150 | 15.13x |
|🟠| np.right_shift(a, 2) (uint32) | uint32 | 100,000 | 0.0190 | 0.0660 | 3.41x |
|✅| np.right_shift(a, 2) (uint32) | uint32 | 10,000,000 | 14.9490 | 13.5400 | 0.91x |
|🔴| np.right_shift(a, 2) (uint64) | uint64 | 1,000 | 0.0010 | 0.0160 | 15.71x |
|🟠| np.right_shift(a, 2) (uint64) | uint64 | 100,000 | 0.0310 | 0.0720 | 2.35x |
|✅| np.right_shift(a, 2) (uint64) | uint64 | 10,000,000 | 32.6270 | 19.1040 | 0.59x |
|🔴| np.right_shift(a, 2) (uint8) | uint8 | 1,000 | 0.0010 | 0.0080 | 8.66x |
|🟠| np.right_shift(a, 2) (uint8) | uint8 | 100,000 | 0.0280 | 0.0640 | 2.27x |
|🟡| np.right_shift(a, 2) (uint8) | uint8 | 10,000,000 | 6.3310 | 10.3850 | 1.64x |

### Logic

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|⚪| np.all(a) (bool) | bool | 1,000 | 0.0010 | - | - |
|⚪| np.all(a) (bool) | bool | 100,000 | 0.0010 | - | - |
|⚪| np.all(a) (bool) | bool | 10,000,000 | 0.0040 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 1,000 | 0.0130 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 100,000 | 0.0790 | - | - |
|⚪| np.allclose(a, b) (float32) | float32 | 10,000,000 | 103.4370 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 1,000 | 0.0140 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 100,000 | 0.7090 | - | - |
|⚪| np.allclose(a, b) (float64) | float64 | 10,000,000 | 186.0120 | - | - |
|⚪| np.any(a) (bool) | bool | 1,000 | 0.0010 | - | - |
|⚪| np.any(a) (bool) | bool | 100,000 | 0.0010 | - | - |
|⚪| np.any(a) (bool) | bool | 10,000,000 | 0.0040 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 1,000 | 0.0020 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 100,000 | 0.0070 | - | - |
|⚪| np.array_equal(a, b) (float32) | float32 | 10,000,000 | 10.8640 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 1,000 | 0.0020 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 100,000 | 0.0130 | - | - |
|⚪| np.array_equal(a, b) (float64) | float64 | 10,000,000 | 19.8000 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 1,000 | 0.0120 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 100,000 | 0.0780 | - | - |
|⚪| np.isclose(a, b) (float32) | float32 | 10,000,000 | 98.4760 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 1,000 | 0.0120 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 100,000 | 0.7130 | - | - |
|⚪| np.isclose(a, b) (float64) | float64 | 10,000,000 | 187.4280 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 1,000 | 0.0000 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 100,000 | 0.0050 | - | - |
|⚪| np.isfinite(a) (float32) | float32 | 10,000,000 | 3.6660 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 1,000 | 0.0000 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 100,000 | 0.0100 | - | - |
|⚪| np.isfinite(a) (float64) | float64 | 10,000,000 | 10.9930 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 1,000 | 0.0000 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 100,000 | 0.0050 | - | - |
|⚪| np.isinf(a) (float32) | float32 | 10,000,000 | 3.7900 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 100,000 | 0.0100 | - | - |
|⚪| np.isinf(a) (float64) | float64 | 10,000,000 | 12.2420 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 1,000 | 0.0010 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 100,000 | 0.0040 | - | - |
|⚪| np.isnan(a) (float32) | float32 | 10,000,000 | 3.8650 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 1,000 | 0.0000 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 100,000 | 0.0090 | - | - |
|⚪| np.isnan(a) (float64) | float64 | 10,000,000 | 11.1540 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 1,000 | 0.0010 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 100,000 | 0.0070 | - | - |
|⚪| np.maximum(a, b) (float32) | float32 | 10,000,000 | 8.9750 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 100,000 | 0.0300 | - | - |
|⚪| np.maximum(a, b) (float64) | float64 | 10,000,000 | 33.1900 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 1,000 | 0.0010 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 100,000 | 0.0070 | - | - |
|⚪| np.minimum(a, b) (float32) | float32 | 10,000,000 | 9.0840 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 1,000 | 0.0010 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 100,000 | 0.0290 | - | - |
|⚪| np.minimum(a, b) (float64) | float64 | 10,000,000 | 32.3180 | - | - |

### Statistics

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|✅| np.average(a) (float32) | float32 | 1,000 | 0.0040 | 0.0010 | 0.15x |
|✅| np.average(a) (float32) | float32 | 100,000 | 0.0180 | 0.0020 | 0.12x |
|✅| np.average(a) (float32) | float32 | 10,000,000 | 9.5980 | 0.9370 | 0.10x |
|✅| np.average(a) (float64) | float64 | 1,000 | 0.0030 | 0.0010 | 0.23x |
|✅| np.average(a) (float64) | float64 | 100,000 | 0.0180 | 0.0040 | 0.22x |
|✅| np.average(a) (float64) | float64 | 10,000,000 | 17.2900 | 2.5460 | 0.15x |
|✅| np.count_nonzero(a) (float32) | float32 | 1,000 | 0.0010 | 0.0000 | 0.10x |
|✅| np.count_nonzero(a) (float32) | float32 | 100,000 | 0.0380 | 0.0050 | 0.12x |
|✅| np.count_nonzero(a) (float32) | float32 | 10,000,000 | 8.0120 | 1.5430 | 0.19x |
|✅| np.count_nonzero(a) (float64) | float64 | 1,000 | 0.0010 | 0.0000 | 0.19x |
|✅| np.count_nonzero(a) (float64) | float64 | 100,000 | 0.0380 | 0.0090 | 0.23x |
|✅| np.count_nonzero(a) (float64) | float64 | 10,000,000 | 22.6050 | 3.7370 | 0.17x |
|✅| np.median(a) (float32) | float32 | 1,000 | 0.0110 | 0.0020 | 0.22x |
|🟡| np.median(a) (float32) | float32 | 100,000 | 0.4720 | 0.7420 | 1.57x |
|✅| np.median(a) (float32) | float32 | 10,000,000 | 87.7170 | 85.5720 | 0.98x |
|✅| np.median(a) (float64) | float64 | 1,000 | 0.0100 | 0.0020 | 0.24x |
|🟡| np.median(a) (float64) | float64 | 100,000 | 0.4700 | 0.7070 | 1.50x |
|✅| np.median(a) (float64) | float64 | 10,000,000 | 113.1360 | 87.8340 | 0.78x |
|✅| np.percentile(a, 50) (float32) | float32 | 1,000 | 0.0250 | 0.0020 | 0.10x |
|🟡| np.percentile(a, 50) (float32) | float32 | 100,000 | 0.7320 | 0.7430 | 1.01x |
|🟡| np.percentile(a, 50) (float32) | float32 | 10,000,000 | 68.3270 | 85.4780 | 1.25x |
|✅| np.percentile(a, 50) (float64) | float64 | 1,000 | 0.0240 | 0.0020 | 0.10x |
|✅| np.percentile(a, 50) (float64) | float64 | 100,000 | 0.7120 | 0.7080 | 0.99x |
|🟡| np.percentile(a, 50) (float64) | float64 | 10,000,000 | 82.2650 | 87.7600 | 1.07x |
|✅| np.ptp(a) (float32) | float32 | 1,000 | 0.0030 | 0.0020 | 0.63x |
|🟡| np.ptp(a) (float32) | float32 | 100,000 | 0.0140 | 0.0280 | 1.97x |
|✅| np.ptp(a) (float32) | float32 | 10,000,000 | 7.7190 | 3.4000 | 0.44x |
|✅| np.ptp(a) (float64) | float64 | 1,000 | 0.0030 | 0.0030 | 0.77x |
|🟠| np.ptp(a) (float64) | float64 | 100,000 | 0.0200 | 0.0530 | 2.67x |
|✅| np.ptp(a) (float64) | float64 | 10,000,000 | 18.9640 | 10.1400 | 0.53x |
|✅| np.quantile(a, 0.5) (float32) | float32 | 1,000 | 0.0240 | 0.0020 | 0.10x |
|🟡| np.quantile(a, 0.5) (float32) | float32 | 100,000 | 0.6880 | 0.7440 | 1.08x |
|🟡| np.quantile(a, 0.5) (float32) | float32 | 10,000,000 | 64.1920 | 85.6320 | 1.33x |
|✅| np.quantile(a, 0.5) (float64) | float64 | 1,000 | 0.0230 | 0.0020 | 0.10x |
|🟡| np.quantile(a, 0.5) (float64) | float64 | 100,000 | 0.7040 | 0.7070 | 1.00x |
|🟡| np.quantile(a, 0.5) (float64) | float64 | 10,000,000 | 86.1590 | 87.6600 | 1.02x |

### Sorting

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🔴| np.argsort(a) (float32) | float32 | 1,000 | 0.0120 | 0.0690 | 5.88x |
|🔴| np.argsort(a) (float32) | float32 | 100,000 | 1.5580 | 12.9880 | 8.34x |
|🟡| np.argsort(a) (float32) | float32 | 10,000,000 | 1524.6230 | 2861.3200 | 1.88x |
|🔴| np.argsort(a) (float64) | float64 | 1,000 | 0.0100 | 0.0710 | 6.76x |
|🔴| np.argsort(a) (float64) | float64 | 100,000 | 1.4220 | 13.4710 | 9.48x |
|🟡| np.argsort(a) (float64) | float64 | 10,000,000 | 2030.5670 | 3133.5310 | 1.54x |
|🟠| np.argsort(a) (int32) | int32 | 1,000 | 0.0120 | 0.0390 | 3.28x |
|🔴| np.argsort(a) (int32) | int32 | 100,000 | 0.4420 | 10.4040 | 23.54x |
|🔴| np.argsort(a) (int32) | int32 | 10,000,000 | 368.7840 | 2162.0890 | 5.86x |
|🟠| np.argsort(a) (int64) | int64 | 1,000 | 0.0130 | 0.0590 | 4.51x |
|🔴| np.argsort(a) (int64) | int64 | 100,000 | 0.4720 | 12.8930 | 27.34x |
|🟠| np.argsort(a) (int64) | int64 | 10,000,000 | 572.7780 | 2835.7750 | 4.95x |
|✅| np.nonzero(a) (float32) | float32 | 1,000 | 0.0030 | 0.0020 | 0.77x |
|✅| np.nonzero(a) (float32) | float32 | 100,000 | 0.1950 | 0.0850 | 0.44x |
|✅| np.nonzero(a) (float32) | float32 | 10,000,000 | 43.6330 | 18.7020 | 0.43x |
|✅| np.nonzero(a) (float64) | float64 | 1,000 | 0.0030 | 0.0020 | 0.78x |
|✅| np.nonzero(a) (float64) | float64 | 100,000 | 0.1870 | 0.0930 | 0.50x |
|✅| np.nonzero(a) (float64) | float64 | 10,000,000 | 56.0460 | 21.9810 | 0.39x |
|🟡| np.nonzero(a) (int32) | int32 | 1,000 | 0.0020 | 0.0020 | 1.19x |
|✅| np.nonzero(a) (int32) | int32 | 100,000 | 0.1040 | 0.0840 | 0.81x |
|✅| np.nonzero(a) (int32) | int32 | 10,000,000 | 32.4050 | 18.6120 | 0.57x |
|🟡| np.nonzero(a) (int64) | int64 | 1,000 | 0.0020 | 0.0020 | 1.25x |
|✅| np.nonzero(a) (int64) | int64 | 100,000 | 0.1040 | 0.0970 | 0.93x |
|✅| np.nonzero(a) (int64) | int64 | 10,000,000 | 57.7190 | 22.4010 | 0.39x |
|✅| np.searchsorted(a, v) (float32) | float32 | 1,000 | 0.0020 | 0.0000 | 0.01x |
|✅| np.searchsorted(a, v) (float32) | float32 | 100,000 | 0.0240 | 0.0000 | 0.00x |
|✅| np.searchsorted(a, v) (float32) | float32 | 10,000,000 | 22.9510 | 0.0000 | 0.00x |
|✅| np.searchsorted(a, v) (float64) | float64 | 1,000 | 0.0010 | 0.0000 | 0.02x |
|✅| np.searchsorted(a, v) (float64) | float64 | 100,000 | 0.0010 | 0.0000 | 0.02x |
|✅| np.searchsorted(a, v) (float64) | float64 | 10,000,000 | 0.0030 | 0.0000 | 0.01x |
|✅| np.searchsorted(a, v) (int32) | int32 | 1,000 | 0.0020 | 0.0000 | 0.01x |
|✅| np.searchsorted(a, v) (int32) | int32 | 100,000 | 0.0320 | 0.0000 | 0.00x |
|✅| np.searchsorted(a, v) (int32) | int32 | 10,000,000 | 22.8200 | 0.0000 | 0.00x |
|✅| np.searchsorted(a, v) (int64) | int64 | 1,000 | 0.0010 | 0.0000 | 0.02x |
|✅| np.searchsorted(a, v) (int64) | int64 | 100,000 | 0.0010 | 0.0000 | 0.02x |
|✅| np.searchsorted(a, v) (int64) | int64 | 10,000,000 | 0.0030 | 0.0000 | 0.01x |

### LinearAlgebra

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟠| np.dot(a, b) (float64) | float64 | 1,000 | 0.0010 | 0.0030 | 4.40x |
|✅| np.dot(a, b) (float64) | float64 | 100,000 | 0.1110 | 0.0710 | 0.65x |
|🔴| np.dot(a, b) (float64) | float64 | 10,000,000 | 1.2320 | 16.4600 | 13.36x |
|🟠| np.matmul(A, B) (float64) | float64 | 1,000 | 0.0030 | 0.0050 | 2.03x |
|🔴| np.matmul(A, B) (float64) | float64 | 100,000 | 0.6010 | 3.2320 | 5.38x |
|🔴| np.matmul(A, B) (float64) | float64 | 10,000,000 | 0.7190 | 4.2600 | 5.92x |
|🟠| np.outer(a, b) (float64) | float64 | 1,000 | 0.0020 | 0.0050 | 2.36x |
|🟡| np.outer(a, b) (float64) | float64 | 100,000 | 0.0380 | 0.0490 | 1.30x |
|✅| np.outer(a, b) (float64) | float64 | 10,000,000 | 14.5050 | 11.8530 | 0.82x |

### Selection

| | Operation | Type | N | NumPy (ms) | NumSharp (ms) | Ratio |
|:-:|-----------|:----:|----:|----------:|-------------:|------:|
|🟡| np.where(cond) (float64) | float64 | 1,000 | 0.0010 | 0.0010 | 1.61x |
|🟠| np.where(cond) (float64) | float64 | 100,000 | 0.0290 | 0.0600 | 2.06x |
|🟡| np.where(cond) (float64) | float64 | 10,000,000 | 7.4850 | 9.6490 | 1.29x |
|🟡| np.where(cond, a, b) (float64) | float64 | 1,000 | 0.0020 | 0.0020 | 1.18x |
|🟡| np.where(cond, a, b) (float64) | float64 | 100,000 | 0.0410 | 0.0650 | 1.60x |
|✅| np.where(cond, a, b) (float64) | float64 | 10,000,000 | 18.7540 | 14.8530 | 0.79x |
