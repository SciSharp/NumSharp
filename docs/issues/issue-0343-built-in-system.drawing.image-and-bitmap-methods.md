# #343: Built-in System.Drawing.Image and Bitmap methods

- **URL:** https://github.com/SciSharp/NumSharp/issues/343
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2019-09-01T13:01:21Z
- **Updated:** 2019-11-05T17:00:36Z
- **Labels:** enhancement
- **Assignees:** @Nucs

## Description

Theres a repeating need for methods to load Image to bitmap, we should provide performant builtin API for that.

EDIT 1:
System.Drawing.Bitmap are now supported by a separate package, [read more](https://github.com/SciSharp/NumSharp/wiki/Bitmap-Extensions).

## Comments

### Comment 1 by @Oceania2018 (2019-09-01T13:06:07Z)

Yes, many people need it. It helps. But that will make NumSharp introduce extra dependency. Or we just add file.read to bytes interface ?

### Comment 2 by @Nucs (2019-09-01T13:09:39Z)

Yes, `System.Drawing.Image` package.
I'm thinking about `new NDArray(System.Drawing.Image)` and `np.array(System.Drawing.Image)` or something of that sort.

### Comment 3 by @sdg002 (2019-09-10T09:02:37Z)

Thanks for all the hard work.
My 2 cents.   It might be worth considering the pros and cons of keeping `System.Drawing.Image` outside of the core `NDArray` through a factory approach. At some later date you might consider introducing a factory class  for images loaded using `ImageSharp` or some other imaging library.  

This approach will not cut down your code, however it might spare end developers from having to add too many NUGET references. Gives them the opportunity to progressively encompass more packages as the needs grow.  E.g. In my company, `System.Drawing` is not really favored because it does not work on Azure functions due to GDI+ restrictions of Azure function sandbox.


### Comment 4 by @Nucs (2019-09-11T20:30:52Z)

@sdg002 Thanks for the note,
I've decided to create separate projects and nuget packages for `NumSharp.Bitmap` (published) and `NumSharp.ImageSharp` (WIP).

### Comment 5 by @Oceania2018 (2019-10-04T11:11:03Z)

Is it possible to integrate OpenCvSharp into NumSharp.ImageSharp?
