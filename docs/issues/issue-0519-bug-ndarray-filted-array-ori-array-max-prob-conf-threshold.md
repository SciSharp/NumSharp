# #519: BUG:  NDArray filted_array = ori_array[max_prob > conf_threshold];

- **URL:** https://github.com/SciSharp/NumSharp/issues/519
- **State:** OPEN
- **Author:** @1Zengy
- **Created:** 2024-11-11T09:09:51Z
- **Updated:** 2024-11-11T09:09:51Z

## Description

In the C# code, NDArray filted_array = ori_array[max_prob > conf_threshold]; 
Here, `ori_array` is an `NDArray` with a shape of [1, 8400, 5], where 5 represents the five probabilities of 5 classes. `max_prob` is an `NDArray` with a shape of [1, 8400], indicating the maximum probability among the 5 classes. `conf_threshold` is a float value which represents the threshold of probability.

I would like to delete the objects in the 8400 dimension where the maximum probability is lower than the `conf_threshold`. However, even though I have ensured that `ori_array` and `max_prob` have the same values, the code will randomly return either an exact value with a shape of [n, 5] or a null value with a shape of [0, 5]. Here is my data saved using "np.save()".
[ori_array.json](https://github.com/user-attachments/files/17699927/ori_array.json)
[max_prob.json](https://github.com/user-attachments/files/17699931/max_prob.json)


