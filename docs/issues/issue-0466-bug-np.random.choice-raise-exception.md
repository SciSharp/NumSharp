# #466: [Bug] np.random.choice raise Exception

- **URL:** https://github.com/SciSharp/NumSharp/issues/466
- **State:** OPEN
- **Author:** @QingtaoLi1
- **Created:** 2021-11-30T07:54:33Z
- **Updated:** 2021-12-15T05:29:47Z

## Description

Hi, when using `np.random.choice`, I got an Exception with message "Specified method is not supported.", and the stack trace is:

>    at NumSharp.NPTypeCodeExtensions.GetAccumulatingType(NPTypeCode typeCode)
>    at NumSharp.Backends.DefaultEngine.cumsum_elementwise(NDArray& arr, Nullable`1 typeCode)
>    at NumSharp.Backends.DefaultEngine.ReduceCumAdd(NDArray& arr, Nullable`1 axis_, Nullable`1 typeCode)
>    at NumSharp.np.cumsum(NDArray arr, Nullable`1 axis, Nullable`1 typeCode)

## Comments

### Comment 1 by @QingtaoLi1 (2021-11-30T07:57:42Z)

It seems there is no UT for this function in this repo.

### Comment 2 by @QingtaoLi1 (2021-11-30T08:00:22Z)

Besides, another argument `replace` is not used at all in both overrides.

### Comment 3 by @QingtaoLi1 (2021-11-30T08:28:09Z)

I do a simple test on `np.cumsum` since it is in the stack trace:
>         public static void RandomChoice()
>         {
>             var array = np.arange(1, 50265);
>             var arrayDouble = 1.0 / array.astype(np.@double);
>             var cumsum = np.cumsum(arrayDouble, typeCode: arrayDouble.typecode);  // OK
>             var cumsum2 = np.cumsum(arrayDouble);                              // raise Exception mentioned above
>         }
> 

It looks like there's something wrong in the default typeCode. BTW, I'm using the latest NumSharp 0.30.0 version.

### Comment 4 by @UCtreespring (2021-12-14T16:25:46Z)

相同的问题，你解决了吗？

### Comment 5 by @QingtaoLi1 (2021-12-15T02:55:05Z)

目前发现的解决方案就是上面那样自己copy一个，加上typeCode就能跑了

### Comment 6 by @UCtreespring (2021-12-15T05:29:47Z)

是我没有留意到那个“Ok”的注释，按照你的方案，我复制了官方Numsharp-master中的相关方法，添加了相关的typeCode参数，问题解决。非常感谢！
