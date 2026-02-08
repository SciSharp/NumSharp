# #239: np.linalg.norm

- **URL:** https://github.com/SciSharp/NumSharp/issues/239
- **State:** OPEN
- **Author:** @henon
- **Created:** 2019-04-05T10:52:03Z
- **Updated:** 2020-10-09T10:41:13Z
- **Labels:** enhancement

## Description

I am going to port np.linalg.norm(...)

## Comments

### Comment 1 by @Oceania2018 (2019-04-05T15:35:08Z)

It's not same as ?
![image](https://user-images.githubusercontent.com/1705364/55639218-77e54f00-578e-11e9-8320-89364ddc6805.png)


### Comment 2 by @henon (2019-04-05T16:36:31Z)

ok, you are right. that implements the L2 norm I need. I'll add the shortcut np.linalg.norm(...) just for Python  compatibility

### Comment 3 by @henon (2019-04-05T16:44:54Z)

no, after checking the code against the numpy docs, actually, the 2-norm is not the same what has been implemented in normalize(). There is no squaring of coefficients. Is that a bug?

https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.linalg.norm.html

### Comment 4 by @Oceania2018 (2019-04-05T18:02:55Z)

Yes, they're different.

### Comment 5 by @8 (2020-10-09T07:23:27Z)

Hi,

I also noticed that the function is missing in `0.20.5`.
I think one could add a shortcut from `np.linalg.norm(a)` => `np.sqrt(a.dot(a))` instead, which should be equivalent.
Not sure about the overloads though.

Does this make sense to you?

Take care,
Martin


