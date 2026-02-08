# #471: Unhandled Exception: System.NotSupportedException: Specified method is not supported.

- **URL:** https://github.com/SciSharp/NumSharp/issues/471
- **State:** OPEN
- **Author:** @KonardAdams
- **Created:** 2021-12-15T16:57:20Z
- **Updated:** 2021-12-15T16:57:20Z

## Description

**I am getting an error at this line of code:** 
 `double theta3_2 = Math.Atan2(a3, d4) - Math.Atan2(K / p3, -np.sqrt(np.power(1 - (K / p3), 2)));`

Unhandled Exception: System.NotSupportedException: Specified method is not supported.
   at NumSharp.Backends.DefaultEngine.Negate(NDArray& nd) in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\Default\Math\Default.Negate.cs:line 119
   at NumSharp.NDArray.op_UnaryNegation(NDArray x) in D:\SciSharp\NumSharp\src\NumSharp.Core\Operations\Elementwise\NDArray.Primitive.cs:line 10
   at Robotics.InverseNP.Main(String[] args) in C:\Users\..........................
