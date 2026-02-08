# #413: NDArray Split

- **URL:** https://github.com/SciSharp/NumSharp/issues/413
- **State:** OPEN
- **Author:** @lqdev
- **Created:** 2020-06-01T15:09:12Z
- **Updated:** 2020-06-01T15:09:12Z

## Description

Given the following Python code:

```python
a = [1, 2, 3, 99, 99, 3, 2, 1]
a1, a2, a3 = np.split(a, [3, 5])
print(a1, a2, a3)
```

Translated to F#:

```fsharp
let a = [|1;2;3;99;99;3;2;1|]
let a1,a2,a3 = np.split(a,[|3,5|])
```

When trying to call `split`, I get the following error

```text
typecheck error The field, constructor or member 'split' is not defined
```

Calling `split` on its own without any input arguments, also returns the same error.

Is `split` implemented in NumSharp?
