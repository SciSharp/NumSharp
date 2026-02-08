# #190: Compressed Sparse Format

- **URL:** https://github.com/SciSharp/NumSharp/issues/190
- **State:** OPEN
- **Author:** @Oceania2018
- **Created:** 2019-01-09T21:45:32Z
- **Updated:** 2019-04-16T16:00:05Z
- **Labels:** enhancement
- **Assignees:** @dotChris90
- **Milestone:** v0.7

## Description

We should implement the compressed ndarray, it is used widely in scikit-learn. The csr_matrix is included in SciPy not NumPy, but I think we should move it into NumSharp. @dotChris90 What do you think of it?

[Compressed Sparse Column Format (CSC)](https://www.scipy-lectures.org/advanced/scipy_sparse/csc_matrix.html)
[Compressed Sparse Row Format (CSR)](https://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html)

https://www.scipy-lectures.org/advanced/scipy_sparse/storage_schemes.html

## Comments

### Comment 1 by @Deep-Blue-2013 (2019-01-09T21:47:51Z)

CSC format saved memory a lot. Looking forward to see this feature.

### Comment 2 by @sebhofer (2019-04-16T13:38:38Z)

Would be great to have an sparse matrix implementation with good performance! Math.Net has some support for sparse matrices, but performance is not great. Also, would be useful to have support for different storage formats (as you already hinted at above).

As an additional reference:  [Here](https://github.com/wo80/mathnet-extensions) is an implementation improving on the performance of Math.Net.

### Comment 3 by @Oceania2018 (2019-04-16T16:00:05Z)

@sebhofer Thanks for the info, it's important to us.
