# #508: np.hstack has diffrent effect from python

- **URL:** https://github.com/SciSharp/NumSharp/issues/508
- **State:** OPEN
- **Author:** @xdqa01
- **Created:** 2024-02-17T08:19:50Z
- **Updated:** 2024-02-18T00:28:53Z

## Description

Windows 10@19044.2364
Dotnet@net8.0-windows,wpf
NumSharp@0.30.0
OpenCvSharp4@4.9.0.20240103
Python@3.12


NumSharp np.hstack(img1,img2,img3) 
```csharp
        var image1 = Cv2.ImRead(FirstImagePath).NotNull().ResizeToStandardSize();
        var image2 = Cv2.ImRead(SecondImagePath).NotNull().ResizeToStandardSize();
        var image3 = Cv2.ImRead(ThirdImagePath).NotNull().ResizeToStandardSize();
        var imageArray = np.hstack(image1.ToNDArray(), image2.ToNDArray(), image3.ToNDArray());
        var image = imageArray.ToMat();
        Cv2.ImShow(WindowName, image);
        FourthImageSource = image.ToBitmapSource();
```

print 
- img1
- img2
- img3

python np.hstack(img1,img2,img3) 
```python
    img1 = cv.imread("./static/fllower.jpg")
    img2 = cv.imread("./static/lake.jpg")
    img3 = cv.imread("./static/mountain.jpg")
    img1 = cv.resize(img1, (200, 200))
    img2 = cv.resize(img2, (200, 200))
    img3 = cv.resize(img3, (200, 200))
    imgs = np.hstack([img1, img2, img3])
    cv.imshow("multi_pic", imgs)
```

print 
img1 img2 img3


And,if i use code below,it print like python np.hstack
```csharp
var imageArray = np.dstack(image1.ToNDArray(), image2.ToNDArray(), image3.ToNDArray());
```

