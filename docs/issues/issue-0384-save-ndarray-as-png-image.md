# #384: Save NDArray as png image

- **URL:** https://github.com/SciSharp/NumSharp/issues/384
- **State:** OPEN
- **Author:** @solarflarefx
- **Created:** 2020-01-12T23:54:22Z
- **Updated:** 2020-01-13T00:30:55Z

## Description

I am looking to save an NDArray to an image.  

In Python code, I used the io.imsave() method from skimage.io.

I tried using the approach shown here: https://stackoverflow.com/questions/5113919/how-to-convert-2-d-array-into-image-in-c-sharp

Basically it uses the the Bitmap method in System.Drawing.Bitmap

Is this the correct way to do this?

I tried converting the NDArray to a C# multidimensional array and then using the following type of code:

`Bitmap bitmap;
unsafe
{
    fixed (int* intPtr = &integers[0,0])
    {
        bitmap = new Bitmap(width, height, stride, PixelFormat.Format32bppRgb, new IntPtr(intPtr));
    }
}`

However, I get this error on the "bitmap =" line: System.ArgumentException: 'Parameter is not valid.'

Ultimately I would like to compare the output from my python code to the output from my C# code to ensure that they are doing the same thing.  As I stated, in Python I saved a multidimensional array to png.  My thought was to do the same in C# using NumSharp, and then comparing the output images.

Thanks in advance.

## Comments

### Comment 1 by @Nucs (2020-01-13T00:30:00Z)

Take a look at our [wiki](https://github.com/SciSharp/NumSharp/wiki/Bitmap-Extensions). We provide support for converting/wrapping NDArray to System.Drawing.Bitmap.
If it is for the purpose of only saving then specify argument `copy: false`
