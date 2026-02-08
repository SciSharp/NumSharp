# #440: NDArray.ToBitmap() has critical issue with 24bpp VERTICAL images

- **URL:** https://github.com/SciSharp/NumSharp/issues/440
- **State:** OPEN
- **Author:** @MiroslavKabat
- **Created:** 2021-02-09T00:57:34Z
- **Updated:** 2021-04-14T11:21:57Z
- **Labels:** bug

## Description

            var arr = np.ones(1, 2, 1, 3).astype(NPTypeCode.Byte);
            var bmp = arr.ToBitmap();

            for (int c = 0; c < bmp.Width; c++)
            {
                for (int r = 0; r < bmp.Height; r++)
                {
                    var p = bmp.GetPixel(c, r);
                    Console.WriteLine($"r:{r} c:{c} => ({p.R};{p.G};{p.B})");
                }
            }

            // return
            // r: 0 c: 0 => (1; 1; 1)
            // r: 1 c: 0 => (0; 1; 1) !!!

            // instead of
            // r: 0 c: 0 => (1; 1; 1)
            // r: 1 c: 0 => (1; 1; 1)
