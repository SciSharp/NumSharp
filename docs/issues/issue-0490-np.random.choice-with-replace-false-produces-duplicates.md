# #490: np.random.choice with replace: false produces duplicates

- **URL:** https://github.com/SciSharp/NumSharp/issues/490
- **State:** OPEN
- **Author:** @GThibeault
- **Created:** 2023-03-06T00:04:52Z
- **Updated:** 2023-03-06T00:04:52Z

## Description

e.g. np.random.choice(71, new Shape(40), **replace: false**) is producing duplicates.

I've tried setting the seed to provide a reproducible example, but that doesn't seem to work either.
