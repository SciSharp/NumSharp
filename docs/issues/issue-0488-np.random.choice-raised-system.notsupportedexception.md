# #488: np.random.choice raised  System.NotSupportedException

- **URL:** https://github.com/SciSharp/NumSharp/issues/488
- **State:** OPEN
- **Author:** @alvinfebriando
- **Created:** 2023-02-15T06:20:28Z
- **Updated:** 2023-02-27T22:57:51Z

## Description

error on np.random.choice, it said "Specified method isn not supported". Is my usage wrong?

```csharp
var score = np.arange(1,6);
var p = new[] { 0.2, 0.2, 0.2, 0.2, 0.2 };
var result = np.random.choice(score, probabilities: p);
```

Stack trace
```
Unhandled exception. System.NotSupportedException: Specified method is not supported.
   at NumSharp.NPTypeCodeExtensions.GetAccumulatingType(NPTypeCode typeCode)
   at NumSharp.Backends.DefaultEngine.cumsum_elementwise(NDArray& arr, Nullable`1 typeCode)
   at NumSharp.Backends.DefaultEngine.ReduceCumAdd(NDArray& arr, Nullable`1 axis_, Nullable`1 typeCode)
   at NumSharp.np.cumsum(NDArray arr, Nullable`1 axis, Nullable`1 typeCode)
   at NumSharp.NumPyRandom.choice(Int32 a, Shape shape, Boolean replace, Double[] probabilities)
   at NumSharp.NumPyRandom.choice(NDArray arr, Shape shape, Boolean replace, Double[] probabilities)
   at Program.<Main>$(String[] args) 
```

## Comments

### Comment 1 by @bojake (2023-02-27T22:57:51Z)

I created a test case with your exact code and ran it using my dev fork of NumSharp. No errors were produced. Are you using an out dated NumSharp? 
