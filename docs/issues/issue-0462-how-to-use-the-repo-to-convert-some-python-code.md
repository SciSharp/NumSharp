# #462: How to use the repo to convert some Python code?

- **URL:** https://github.com/SciSharp/NumSharp/issues/462
- **State:** OPEN
- **Author:** @zydjohnHotmail
- **Created:** 2021-09-21T07:57:34Z
- **Updated:** 2021-12-01T05:58:40Z

## Description

Hello:
I have the following Python code to convert one RGB image to one YUV image, and use numpy to calculate an average of one column.

`import cv2
import numpy as np

img_rgb = cv2.imread('C:/Images/1.PNG')
img_yuv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2YUV)
averageV = np.average(img_yuv[:,:,2])
print(averageV);
`

The Python code works well. Now, I want to change it to use C# code, as I have many other C# programs will need this averageV value.

I have done the following:
1) I created one C# console project with Visual Studio 2019 (target .NET 5.0)
2) I installed necessary NUGET packages:
PM> Install-Package OpenCvSharp4 -Version 4.5.3.20210817
PM> Install-Package NumSharp -Version 0.30.0
3) I have the following C# code:
`using NumSharp;
using OpenCvSharp;
using System;

namespace ConvertRGB2YUV
{
    class Program
    {
        public const string Image1_File = @"C:\Images\1.PNG";

        static void Main(string[] args)
        {
            Mat img_rgb = Cv2.ImRead(Image1_File);
            Mat img_yuv = img_rgb.CvtColor(ColorConversionCodes.RGB2YUV);
            //var averageV = (img_yuv[:,:,2]);
            //averageV = np.average(img_yuv[:,:, 2])
        }
    }
}`

I can run my code, and I can see the image: img_rgb and img_yuv.
But I have no idea on how to write the python corresponding statement:
averageV = np.average(img_yuv[:,:,2])
In NumSharp, the img_yuv[…] simply doesn’t exist.  
In Python, the img_yuv is treated like an array of float numbers.
How I can do this in NumSharp?
Please advise,
Thanks,


## Comments

### Comment 1 by @QingtaoLi1 (2021-12-01T05:58:40Z)

C# doesn't support this kind of indices or slices. I guess you can explore the `NumSharp.Slice` class to reach your target.
