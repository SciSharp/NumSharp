# #405: np.argsort not sorting properly

- **URL:** https://github.com/SciSharp/NumSharp/issues/405
- **State:** OPEN
- **Author:** @tk4218
- **Created:** 2020-04-09T18:35:33Z
- **Updated:** 2020-04-09T21:18:51Z

## Description

I am attempting to call np.argsort<Double>() on the following array:
[0.700656592845917, 0.651415288448334, 0.719015657901764] dtype = Double

The expected result should be:
[1, 0, 2] 

However, the actual result I'm getting is:
[0, 2, 1]

This doesn't to produce a result that is sorted in any direction. I pulled in ndarray.argsort locally and  changed line 31 -
from: `var data = Array;`
to:     `var data = ToArray<T>()`
(Also changing the declaration of the function to include `where T : unmanaged `)

And this seemed to work for me. 


## Comments

### Comment 1 by @tk4218 (2020-04-09T21:18:51Z)

After trying things out for awhile, it seems I can get np.argsort to work if I pass a copy of the NDArray to it. It seems like the NDArray is being manipulated while sorting, causing the sort to misbehave.

It seems like np.argsort should be:
`nd.copy().argsort<T>(axis)`

rather than:
`nd.argsort<t>(axis)`
