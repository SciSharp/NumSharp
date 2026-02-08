# #470: Numsharp0.30.0  np.random.choice() method missing cause Exception

- **URL:** https://github.com/SciSharp/NumSharp/issues/470
- **State:** OPEN
- **Author:** @UCtreespring
- **Created:** 2021-12-14T10:52:50Z
- **Updated:** 2021-12-14T10:52:50Z

## Description

System.NotSupportedException
  HResult=0x80131515
  Source=NumSharp
  StackTrace:
   at NumSharp.NPTypeCodeExtensions.GetAccumulatingType(NPTypeCode typeCode)
   at NumSharp.Backends.DefaultEngine.cumsum_elementwise(NDArray& arr, Nullable`1 typeCode)
   at NumSharp.Backends.DefaultEngine.ReduceCumAdd(NDArray& arr, Nullable`1 axis_, Nullable`1 typeCode)
   at NumSharp.NumPyRandom.choice(Int32 a, Shape shape, Boolean replace, Double[] probabilities)
   at NumSharp.NumPyRandom.choice(NDArray arr, Shape shape, Boolean replace, Double[] probabilities)
   at PolicyGradient_TF050.PolicyGradient.ChooseAction(NDArray observation, NDArray actions, NDArray actionmask)

I‘m  new to this ,

Thanks for help.

