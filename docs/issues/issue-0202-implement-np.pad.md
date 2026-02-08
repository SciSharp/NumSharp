# #202: implement np.pad

- **URL:** https://github.com/SciSharp/NumSharp/issues/202
- **State:** OPEN
- **Author:** @skywalkerisnull
- **Created:** 2019-03-03T04:25:59Z
- **Updated:** 2019-03-10T01:06:49Z
- **Labels:** enhancement
- **Assignees:** @KevinMa0207
- **Milestone:** v0.7.5

## Description

Add a buffer or padding around a numpy array: 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html 

## Comments

### Comment 1 by @Oceania2018 (2019-03-03T15:13:32Z)

```python
a = [1, 2, 3, 4, 5]
np.pad(a, (2,3), 'constant', constant_values=(4, 6))

```
