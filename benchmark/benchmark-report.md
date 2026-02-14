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

**Summary:** 3 ops | ✅ 0 | 🟡 0 | 🟠 0 | 🔴 0 | ⚪ 3

### Dispatch

| | Operation | Type | NumPy | NumSharp | Ratio |
|:-:|-----------|:----:|------:|---------:|------:|
|⚪|  | int32 | 0.0 | - | - |
|⚪|  | int32 | 0.0 | - | - |
|⚪|  | int32 | 0.0 | - | - |
