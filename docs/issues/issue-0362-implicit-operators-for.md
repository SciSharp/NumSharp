# #362: Implicit operators for >, >=, <, <=

- **URL:** https://github.com/SciSharp/NumSharp/issues/362
- **State:** OPEN
- **Author:** @deepakkumar1984
- **Created:** 2019-10-06T07:49:57Z
- **Updated:** 2019-10-12T00:33:30Z
- **Labels:** help wanted, missing feature/s

## Description

x == 1 gives a result which is NDArray<bool> as expected

x>1 and other operators like >=, <, <= give null result

## Comments

### Comment 1 by @Nucs (2019-10-06T07:51:06Z)

`np.where` is yet to be implemented.

### Comment 2 by @deepakkumar1984 (2019-10-06T08:55:19Z)

I have done some work with np.where in this commit https://github.com/deepakkumar1984/NumSharp/commit/8bfa4f2484fc974e10e250f6b7ef5da0f381b683
the first parameter is a condition for which we need to finish the implementation under https://github.com/SciSharp/NumSharp/tree/master/src/NumSharp.Core/Operations/Elementwise

Most of the code in commented and return as null

### Comment 3 by @Nucs (2019-10-06T09:03:06Z)

I misunderstood [np.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html), I was sure that the first argument is something of a sort of Expression<Func<..., bool>>.
I'll need to implement each (comparing) operator separately and np.where will work then.
I'll open up a separate issue for that. This might take a while as I'm unavailable (sunday is not a day off in my country 😅)

### Comment 4 by @Oceania2018 (2019-10-12T00:03:42Z)

Same thing happened for me, shouldn't return null:
![image](https://user-images.githubusercontent.com/1705364/66691132-c6056e80-ec59-11e9-890e-29e516a00a1f.png)


### Comment 5 by @Nucs (2019-10-12T00:33:30Z)

Null because it is not implemented.
