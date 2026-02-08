# #210: Add numpy.all

- **URL:** https://github.com/SciSharp/NumSharp/issues/210
- **State:** OPEN
- **Author:** @Esther2013
- **Created:** 2019-03-10T04:32:04Z
- **Updated:** 2019-04-05T07:56:30Z
- **Assignees:** @Oceania2018, @KevinMa0207

## Description

Test whether all array elements along a given axis evaluate to True.
`np.all([[True,False],[True,True]])`
https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html

## Comments

### Comment 1 by @henon (2019-04-05T07:56:30Z)

I just implemented this, but without axis support at the moment. The overload with axis is still to be done
