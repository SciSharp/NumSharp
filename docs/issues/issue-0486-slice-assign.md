# #486: Slice assign

- **URL:** https://github.com/SciSharp/NumSharp/issues/486
- **State:** OPEN
- **Author:** @burungiu
- **Created:** 2023-01-30T10:42:54Z
- **Updated:** 2023-02-28T17:38:50Z

## Description

Hello everyone, there is a way to do assigment as in python?
Example:
`preds[:, :, 0] = preds[:, :, 0] % heatmapWidth`

Thank you

## Comments

### Comment 1 by @bojake (2023-02-28T17:38:50Z)

There is an issue with auto-broadening of scalar operands in the framework.

For instance:

var foo = (array > 12f); // fails
var foo = np.full(12f, array.shape);
var result = (array > foo); // successful

In your "heatmapWidth" operand just create an "np.full(heatmapWidth, preds.shape)" NDArray and use that as the operand. I bet that will work for you.
