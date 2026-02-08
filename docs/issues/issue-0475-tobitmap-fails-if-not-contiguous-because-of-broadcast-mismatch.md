# #475: ToBitmap fails if not contiguous because of Broadcast mismatch

- **URL:** https://github.com/SciSharp/NumSharp/issues/475
- **State:** OPEN
- **Author:** @ponzis
- **Created:** 2022-02-09T15:51:52Z
- **Updated:** 2022-02-09T15:54:08Z

## Description

When using the   `public static unsafe Bitmap ToBitmap(this NDArray nd, int width, int height, PixelFormat format = PixelFormat.DontCare)` passing a NDArray of shape (1, x, y, 3)  fails due to broadcast mismatch at `(LeftShape, RightShape) = DefaultEngine.Broadcast(lhs.Shape, rhs.Shape);` due to broadcasting with `(x*y*3)` and `(1, x, y, 3)` the work around is to clone the NDArray  so that it is continues or change the shape of the function so that it has a correct shape.
 
