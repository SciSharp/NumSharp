# #407: np.negative is not working ?

- **URL:** https://github.com/SciSharp/NumSharp/issues/407
- **State:** OPEN
- **Author:** @LordTrololo
- **Created:** 2020-04-21T07:52:19Z
- **Updated:** 2020-04-21T13:56:16Z
- **Assignees:** @Nucs

## Description

Hi,

I have a NDArray of Floats with dimensions {(1, 13, 13, 3, 2)} and size 1014.

`np.negative(myArray)`

works by converting all values to negative. However I think that [official Numpy behaviour](https://numpy.org/doc/stable/reference/generated/numpy.negative.html?highlight=negative#numpy.negative) is to convert positives to negatives and negatives to positives.

Like so:
```
np.negative([1.,-1.])
array([-1.,  1.])

```

What I get is:
```
np.negative([1.,-1.])
array([-1.,  -1.])
```
Did someone experience similar problem ?

BTW, the solution I use to solve the problem is trivial - 
`var negativeArray = myarray*(-1);`
