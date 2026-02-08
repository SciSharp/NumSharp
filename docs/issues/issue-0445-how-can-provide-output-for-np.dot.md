# #445: How can provide output for np.dot?

- **URL:** https://github.com/SciSharp/NumSharp/issues/445
- **State:** OPEN
- **Author:** @bigdimboom
- **Created:** 2021-03-24T02:49:34Z
- **Updated:** 2021-04-23T11:58:36Z
- **Labels:** missing feature/s

## Description

I can't find the impl. of np.dot(inputA, inutB, out preallocatedArray).

Currently I am using np.dot(.....).copyTo(....). But I think it still allocates a tmp memory.  
