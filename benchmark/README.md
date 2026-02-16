# NumSharp vs NumPy Performance

**Baseline:** NumPy (N=10M elements)

**Ratio** = NumSharp ÷ NumPy → Lower is better for NumSharp

| | Status | Ratio | Meaning |
|:-:|--------|:-----:|---------|
|✅| Faster | <1.0 | NumSharp beats NumPy |
|🟡| Close | 1-2x | Acceptable parity |
|🟠| Slower | 2-5x | Optimization target |
|🔴| Slow | >5x | Priority fix |
|⚪| Pending | - | C# benchmark not run |

---

**Summary:** 64 ops | ✅ 61 | 🟡 3 | 🟠 0 | 🔴 0 | ⚪ 0

### 🏆 Top 15 Best (NumSharp closest to NumPy)

| | Operation | Type | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|------:|---------:|------:|
|✅| a % 7 (literal) (float64) | float64 | 267.0 | 39.8 | 0.1x |
|✅| a % b (element-wise) (float64) | float64 | 188.5 | 45.7 | 0.2x |
|✅| a % 7 (literal) (int32) | int32 | 49.7 | 13.8 | 0.3x |
|✅| a % b (element-wise) (int32) | int32 | 46.9 | 13.9 | 0.3x |
|✅| a % 7 (literal) (int64) | int64 | 49.2 | 23.9 | 0.5x |
|✅| a % b (element-wise) (int64) | int64 | 48.1 | 24.2 | 0.5x |
|✅| a % b (element-wise) (float32) | float32 | 155.0 | 84.3 | 0.5x |
|✅| a % 7 (literal) (float32) | float32 | 174.6 | 96.7 | 0.6x |
|✅| scalar - a (int32) | int32 | 11.4 | 7.2 | 0.6x |
|✅| a / b (element-wise) (int32) | int32 | 21.7 | 14.1 | 0.7x |
|✅| a - b (element-wise) (int32) | int32 | 11.1 | 7.4 | 0.7x |
|✅| a * a (square) (int32) | int32 | 9.7 | 6.7 | 0.7x |
|✅| a * b (element-wise) (int32) | int32 | 10.3 | 7.4 | 0.7x |
|✅| a + b (element-wise) (int64) | int64 | 20.3 | 14.8 | 0.7x |
|✅| np.add(a, b) (float64) | float64 | 20.3 | 14.9 | 0.7x |

### 🔻 Top 15 Worst (Optimization priorities)

| | Operation | Type | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|------:|---------:|------:|
|🟡| a / scalar (int64) | int64 | 18.7 | 21.7 | 1.2x |
|🟡| a * scalar (float32) | float32 | 7.7 | 8.7 | 1.1x |
|🟡| a * 2 (literal) (float64) | float64 | 16.9 | 18.7 | 1.1x |
|✅| a * 2 (literal) (int32) | int32 | 9.0 | 8.9 | 1.0x |
|✅| scalar - a (int64) | int64 | 15.0 | 14.0 | 0.9x |
|✅| a * 2 (literal) (float32) | float32 | 7.7 | 6.9 | 0.9x |
|✅| a * b (element-wise) (int64) | int64 | 17.0 | 15.1 | 0.9x |
|✅| a + scalar (float32) | float32 | 7.8 | 6.9 | 0.9x |
|✅| a + b (element-wise) (float32) | float32 | 8.5 | 7.5 | 0.9x |
|✅| a * 2 (literal) (int64) | int64 | 15.0 | 13.2 | 0.9x |
|✅| a / b (element-wise) (float32) | float32 | 8.9 | 7.7 | 0.9x |
|✅| a - b (element-wise) (float32) | float32 | 8.6 | 7.5 | 0.9x |
|✅| a - scalar (int64) | int64 | 15.4 | 13.5 | 0.9x |
|✅| a * b (element-wise) (float32) | float32 | 8.6 | 7.4 | 0.9x |
|✅| a * a (square) (int64) | int64 | 15.8 | 13.6 | 0.9x |

---

### Arithmetic

| | Operation | Type | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|------:|---------:|------:|
|✅| a + b (element-wise) (int32) | int32 | 9.5 | 7.5 | 0.8x |
|✅| np.add(a, b) (int32) | int32 | 9.6 | 7.7 | 0.8x |
|✅| a + scalar (int32) | int32 | 8.7 | 6.9 | 0.8x |
|✅| a + 5 (literal) (int32) | int32 | 9.1 | 6.9 | 0.8x |
|✅| a - b (element-wise) (int32) | int32 | 11.1 | 7.4 | 0.7x |
|✅| a - scalar (int32) | int32 | 9.0 | 6.8 | 0.8x |
|✅| scalar - a (int32) | int32 | 11.4 | 7.2 | 0.6x |
|✅| a * b (element-wise) (int32) | int32 | 10.3 | 7.4 | 0.7x |
|✅| a * a (square) (int32) | int32 | 9.7 | 6.7 | 0.7x |
|✅| a * scalar (int32) | int32 | 8.7 | 7.3 | 0.8x |
|✅| a * 2 (literal) (int32) | int32 | 9.0 | 8.9 | 1.0x |
|✅| a / b (element-wise) (int32) | int32 | 21.7 | 14.1 | 0.7x |
|✅| a / scalar (int32) | int32 | 17.5 | 14.4 | 0.8x |
|✅| scalar / a (int32) | int32 | 18.1 | 13.7 | 0.8x |
|✅| a % b (element-wise) (int32) | int32 | 46.9 | 13.9 | 0.3x |
|✅| a % 7 (literal) (int32) | int32 | 49.7 | 13.8 | 0.3x |
|✅| a + b (element-wise) (int64) | int64 | 20.3 | 14.8 | 0.7x |
|✅| np.add(a, b) (int64) | int64 | 18.8 | 14.9 | 0.8x |
|✅| a + scalar (int64) | int64 | 16.1 | 13.7 | 0.8x |
|✅| a + 5 (literal) (int64) | int64 | 16.5 | 13.9 | 0.8x |
|✅| a - b (element-wise) (int64) | int64 | 17.5 | 14.8 | 0.8x |
|✅| a - scalar (int64) | int64 | 15.4 | 13.5 | 0.9x |
|✅| scalar - a (int64) | int64 | 15.0 | 14.0 | 0.9x |
|✅| a * b (element-wise) (int64) | int64 | 17.0 | 15.1 | 0.9x |
|✅| a * a (square) (int64) | int64 | 15.8 | 13.6 | 0.9x |
|✅| a * scalar (int64) | int64 | 15.9 | 13.0 | 0.8x |
|✅| a * 2 (literal) (int64) | int64 | 15.0 | 13.2 | 0.9x |
|✅| a / b (element-wise) (int64) | int64 | 24.0 | 19.2 | 0.8x |
|🟡| a / scalar (int64) | int64 | 18.7 | 21.7 | 1.2x |
|✅| scalar / a (int64) | int64 | 18.8 | 15.4 | 0.8x |
|✅| a % b (element-wise) (int64) | int64 | 48.1 | 24.2 | 0.5x |
|✅| a % 7 (literal) (int64) | int64 | 49.2 | 23.9 | 0.5x |
|✅| a + b (element-wise) (float32) | float32 | 8.5 | 7.5 | 0.9x |
|✅| np.add(a, b) (float32) | float32 | 8.9 | 7.6 | 0.8x |
|✅| a + scalar (float32) | float32 | 7.8 | 6.9 | 0.9x |
|✅| a + 5 (literal) (float32) | float32 | 8.2 | 6.9 | 0.8x |
|✅| a - b (element-wise) (float32) | float32 | 8.6 | 7.5 | 0.9x |
|✅| a - scalar (float32) | float32 | 8.1 | 6.8 | 0.8x |
|✅| scalar - a (float32) | float32 | 8.4 | 7.1 | 0.8x |
|✅| a * b (element-wise) (float32) | float32 | 8.6 | 7.4 | 0.9x |
|✅| a * a (square) (float32) | float32 | 8.4 | 6.5 | 0.8x |
|🟡| a * scalar (float32) | float32 | 7.7 | 8.7 | 1.1x |
|✅| a * 2 (literal) (float32) | float32 | 7.7 | 6.9 | 0.9x |
|✅| a / b (element-wise) (float32) | float32 | 8.9 | 7.7 | 0.9x |
|✅| a / scalar (float32) | float32 | 8.9 | 6.6 | 0.7x |
|✅| scalar / a (float32) | float32 | 8.3 | 6.8 | 0.8x |
|✅| a % b (element-wise) (float32) | float32 | 155.0 | 84.3 | 0.5x |
|✅| a % 7 (literal) (float32) | float32 | 174.6 | 96.7 | 0.6x |
|✅| a + b (element-wise) (float64) | float64 | 20.1 | 15.0 | 0.8x |
|✅| np.add(a, b) (float64) | float64 | 20.3 | 14.9 | 0.7x |
|✅| a + scalar (float64) | float64 | 17.9 | 13.7 | 0.8x |
|✅| a + 5 (literal) (float64) | float64 | 17.9 | 13.6 | 0.8x |
|✅| a - b (element-wise) (float64) | float64 | 18.3 | 15.2 | 0.8x |
|✅| a - scalar (float64) | float64 | 17.2 | 13.5 | 0.8x |
|✅| scalar - a (float64) | float64 | 16.7 | 14.1 | 0.8x |
|✅| a * b (element-wise) (float64) | float64 | 18.6 | 14.7 | 0.8x |
|✅| a * a (square) (float64) | float64 | 17.0 | 13.1 | 0.8x |
|✅| a * scalar (float64) | float64 | 17.3 | 13.8 | 0.8x |
|🟡| a * 2 (literal) (float64) | float64 | 16.9 | 18.7 | 1.1x |
|✅| a / b (element-wise) (float64) | float64 | 18.6 | 15.2 | 0.8x |
|✅| a / scalar (float64) | float64 | 17.0 | 13.7 | 0.8x |
|✅| scalar / a (float64) | float64 | 17.9 | 13.5 | 0.8x |
|✅| a % b (element-wise) (float64) | float64 | 188.5 | 45.7 | 0.2x |
|✅| a % 7 (literal) (float64) | float64 | 267.0 | 39.8 | 0.1x |
