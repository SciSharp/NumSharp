# #375: Slice assignment?

- **URL:** https://github.com/SciSharp/NumSharp/issues/375
- **State:** OPEN
- **Author:** @solarflarefx
- **Created:** 2019-12-07T17:40:06Z
- **Updated:** 2019-12-07T17:48:07Z
- **Labels:** missing feature/s
- **Assignees:** @Nucs

## Description

If you declare an array of a specific size, is there a way to assign slices of arrays?

For example, if x is an NDArray of size (5,2,3,4), is there a way to do something like the following?

NDArray a = some NDArray of size (1,2,3,4)

x[0,:,:,:] = a

## Comments

### Comment 1 by @Nucs (2019-12-07T17:47:58Z)

Related to https://github.com/SciSharp/NumSharp/issues/369 https://github.com/SciSharp/NumSharp/issues/368 https://github.com/SciSharp/NumSharp/issues/366 and is work in progress.
