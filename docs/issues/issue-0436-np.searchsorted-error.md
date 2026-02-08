# #436: np.searchsorted error!

- **URL:** https://github.com/SciSharp/NumSharp/issues/436
- **State:** OPEN
- **Author:** @wangfeixing
- **Created:** 2021-01-12T09:20:40Z
- **Updated:** 2021-01-12T09:20:40Z

## Description

i use "np.searchsorted" function like the code below:
  List<Double> list1 = new List<double>() { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            List<Double> list2 = new List<double>() { 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1 };
            NDArray array1 = np.array(list1.ToArray());
            NDArray array2 = np.array(list2.ToArray());

            int kk = np.searchsorted(array2, 3.0);

but it reports the error "System.IndexOutOfRangeException",is this a bug?
