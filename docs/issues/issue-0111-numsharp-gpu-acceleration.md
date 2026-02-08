# #111: NumSharp GPU acceleration

- **URL:** https://github.com/SciSharp/NumSharp/issues/111
- **State:** OPEN
- **Author:** @Oceania2018
- **Created:** 2018-11-16T18:08:54Z
- **Updated:** 2022-02-26T16:16:26Z
- **Labels:** enhancement, help wanted, further discuss

## Description

We might use [Campy](https://github.com/kaby76/Campy) to accelerate NumSharp.

## Comments

### Comment 1 by @fdncred (2018-11-16T18:25:03Z)

@Oceania2018, I was going to suggest the same thing. We don't want to be writing cuda .cu files. It's not easy. However, my first suggestion would be to parallelize the code where possible. i.e. on `IEnumerables` we could use `.AsParallel()`, on for loops we could use `Parallel.For` and `Parallel.Foreach`. BenchmarkDotNet will be helpful here.

### Comment 2 by @ZhiZe-ZG (2022-02-26T15:11:37Z)

This function is very important in some occasions where accelerated computing is required but neural networks (and torch) are not suitable.

### Comment 3 by @Oceania2018 (2022-02-26T16:16:26Z)

We switched to [Tensorflow.Numpy](https://github.com/SciSharp/TensorFlow.NET/tree/master/src/TensorFlowNET.Core/NumPy).
