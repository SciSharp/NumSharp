# #446: Unable to use np.dot due to "Specified method unsupported" error

- **URL:** https://github.com/SciSharp/NumSharp/issues/446
- **State:** OPEN
- **Author:** @moonlitlyra
- **Created:** 2021-03-27T17:01:11Z
- **Updated:** 2021-06-30T18:55:04Z

## Description

I have recently been working on a unity project involving the genetic algorithm, but have run into an error while trying to use the function np.dot(). NumSharp has been installed using the NuGet client [NuGetForUnity](https://github.com/GlitchEnzo/NuGetForUnity)

The error message is as follows:

NotSupportedException: Specified method is not supported.
NumSharp.NPTypeCodeExtensions.GetAccumulatingType (NumSharp.NPTypeCode typeCode) (at <7807b007e09c46aca587061f8867e538>:0)
NumSharp.Backends.DefaultEngine.ReduceAdd (NumSharp.NDArray& arr, System.Nullable`1[T] axis_, System.Boolean keepdims, System.Nullable`1[T] typeCode, NumSharp.NDArray out) (at <7807b007e09c46aca587061f8867e538>:0)
NumSharp.Backends.DefaultEngine.Sum (NumSharp.NDArray& nd, System.Nullable`1[T] axis, System.Nullable`1[T] typeCode, System.Boolean keepdims) (at <7807b007e09c46aca587061f8867e538>:0)
NumSharp.np.sum (NumSharp.NDArray& a, System.Int32 axis) (at <7807b007e09c46aca587061f8867e538>:0)
NumSharp.Backends.DefaultEngine.Dot (NumSharp.NDArray& left, NumSharp.NDArray& right) (at <7807b007e09c46aca587061f8867e538>:0)
NumSharp.np.dot (NumSharp.NDArray& a, NumSharp.NDArray& b) (at <7807b007e09c46aca587061f8867e538>:0)
NN.FeedForward () (at Assets/scripts/NeuralNetwork.cs:73)
Bot.FixedUpdate () (at Assets/scripts/Bot.cs:70)

Line 73 of the FeedForward script:

73. `NDArray activations = np.dot(layers[i].weights, layers[i].activations);`

Other sections of relevant code:

47. `layer.weights = np.random.rand(3, 2);`
48. `layer.activations = np.zeros(2);`

## Comments

### Comment 1 by @Nucs (2021-04-14T11:16:39Z)

Related to #447 


### Comment 2 by @BlackholeGH (2021-06-27T10:07:04Z)

I seem to have this same issue, in a similar context attempting to use np.dot to do a dot-product for two 1-D numpy arrays containing double values. Assuming we're not both doing something wrong, any workarounds beside calculating the dot product manually?

> System.NotSupportedException
  HResult=0x80131515
  Message=Specified method is not supported.
  Source=NumSharp
  StackTrace:
   at NumSharp.NPTypeCodeExtensions.GetAccumulatingType(NPTypeCode typeCode)
   at NumSharp.Backends.DefaultEngine.sum_elementwise(NDArray arr, Nullable`1 typeCode)
   at NumSharp.Backends.DefaultEngine.ReduceAdd(NDArray& arr, Nullable`1 axis_, Boolean keepdims, Nullable`1 typeCode, NDArray out)
   at NumSharp.Backends.DefaultEngine.Dot(NDArray& left, NDArray& right)


### Comment 3 by @moonlitlyra (2021-06-30T18:55:04Z)

> I seem to have this same issue, in a similar context attempting to use np.dot to do a dot-product for two 1-D numpy arrays containing double values. Assuming we're not both doing something wrong, any workarounds beside calculating the dot product manually?
> 
> > System.NotSupportedException
> > HResult=0x80131515
> > Message=Specified method is not supported.
> > Source=NumSharp
> > StackTrace:
> > at NumSharp.NPTypeCodeExtensions.GetAccumulatingType(NPTypeCode typeCode)
> > at NumSharp.Backends.DefaultEngine.sum_elementwise(NDArray arr, Nullable`1 typeCode)    at NumSharp.Backends.DefaultEngine.ReduceAdd(NDArray& arr, Nullable`1 axis_, Boolean keepdims, Nullable`1 typeCode, NDArray out)
> > at NumSharp.Backends.DefaultEngine.Dot(NDArray& left, NDArray& right)

Hello, thankyou for commenting on this issue. No workarounds as far as I am aware, I ended up having to scrap part of my project sadly. Thanks anyway.
