# #315: ToString should truncate its output

- **URL:** https://github.com/SciSharp/NumSharp/issues/315
- **State:** OPEN
- **Author:** @thomasd3
- **Created:** 2019-07-18T12:32:09Z
- **Updated:** 2019-07-18T13:56:23Z
- **Labels:** bug, enhancement
- **Assignees:** @Nucs

## Description

On very large arrays, NDArray.ToString() is never coming back (or maybe it would at some point in time).

It would make sense to truncate the ToString() output.

## Comments

### Comment 1 by @Nucs (2019-07-18T12:49:04Z)

I'll add a `DebuggerTypeProxy` that'll truncute beyond certain amount of characters.
After we finish the rework on the NumSharp's backend - It should be much faster.

### Comment 2 by @henon (2019-07-18T13:30:42Z)

I implemented the NDArray.ToString() but at the time I didn't know that Numpy has a way of not showing the middle elements of a huge array. if you create a 100x100 matrix it will show this on the console. the ToString method probably should do the same. See below. Also, current ToString() does not format the elements so that they are equally spaced with a monotype font. 

```
>>> import numpy as np
>>> a=np.arange(10000).reshape(100,100)
>>> a
array([[   0,    1,    2, ...,   97,   98,   99],
       [ 100,  101,  102, ...,  197,  198,  199],
       [ 200,  201,  202, ...,  297,  298,  299],
       ...,
       [9700, 9701, 9702, ..., 9797, 9798, 9799],
       [9800, 9801, 9802, ..., 9897, 9898, 9899],
       [9900, 9901, 9902, ..., 9997, 9998, 9999]])
>>>
```
