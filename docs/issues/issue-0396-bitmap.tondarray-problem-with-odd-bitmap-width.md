# #396: Bitmap.ToNDArray problem with odd bitmap width

- **URL:** https://github.com/SciSharp/NumSharp/issues/396
- **State:** OPEN
- **Author:** @herrvonregen
- **Created:** 2020-02-06T12:04:28Z
- **Updated:** 2020-02-06T16:44:23Z

## Description

Hi everyone.
The official micorsoft documenation for Bitmap.Stride says:

> The stride is the width of a single row of pixels (a scan line), rounded up to a four-byte boundary.

This could cause an exception here if the loaded bitmap is RGB with an odd width.
`                        Buffer.MemoryCopy(src, dst, bmpData.Stride * image.Height, nd.size);`
`var ret = nd.reshape(1, image.Height, image.Width, bmpData.Stride / bmpData.Width);`

Example:
An RGB image with the size of 227x227 pixels
Stride will be 684. 227*3 = 681 rounded up to four-byte boundary.
Copied data are 155268 bytes.
Reshaped into 277 * 277 * floor(684/227) = 154587 bytes
This will cause an IncorrectShapeException



## Comments

### Comment 1 by @Nucs (2020-02-06T15:44:37Z)

You shouldn't work hard, we provide extensions to Bitmap, see [this wiki article](https://github.com/SciSharp/NumSharp/wiki/Bitmap-Extensions).

### Comment 2 by @herrvonregen (2020-02-06T16:44:23Z)

Yes, I know but the implementation has the described behavior.
Try it with the picture provided below.
![00001](https://user-images.githubusercontent.com/33056845/73958447-39697500-4908-11ea-8628-33dce6582f17.jpg)

