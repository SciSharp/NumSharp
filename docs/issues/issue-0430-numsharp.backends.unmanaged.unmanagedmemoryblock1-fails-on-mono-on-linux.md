# #430:  NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock`1 fails on Mono on Linux

- **URL:** https://github.com/SciSharp/NumSharp/issues/430
- **State:** OPEN
- **Author:** @kgoderis
- **Created:** 2020-11-15T18:22:46Z
- **Updated:** 2020-11-15T18:22:46Z

## Description

Mono 5.12.0.301
Linux 4.19.76 

Attempting the "Hello World" Tensortflow.NET  example in a docker environment running Mono yields the following error. Exactly the same code on Mono 6.12.0.93 on Mac OS X however does run flawlessly

2020-11-15 17:53:41.843832: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-11-15 17:53:41.873442: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 2712000000 Hz
2020-11-15 17:53:41.874170: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fc874d557a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-15 17:53:41.874230: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Stacktrace:

  at <unknown> <0xffffffff>
* Assertion at method-to-ir.c:7352, condition `!sig->has_type_parameters' not met

  at NumSharp.Backends.Unmanaged.UnmanagedMemoryBlock`1<byte>.FromArray (byte[],bool) [0x00037] in <6d1fbec37f814ff9b52dec21dc0ebd1a>:0
  at NumSharp.Backends.Unmanaged.ArraySlice.FromArray<byte> (byte[],bool) [0x00000] in <6d1fbec37f814ff9b52dec21dc0ebd1a>:0
  at NumSharp.np.array<byte> (byte[]) [0x00000] in <6d1fbec37f814ff9b52dec21dc0ebd1a>:0
  at Tensorflow.Tensor.GetNDArray (Tensorflow.TF_DataType) [0x00041] in <f58589f3e4134776b3e4cecbf8fb8b86>:0
  at Tensorflow.Tensor.numpy () [0x00007] in <f58589f3e4134776b3e4cecbf8fb8b86>:0
  at Tensorflow.tensor_util.to_numpy_string (Tensorflow.Tensor) [0x00034] in <f58589f3e4134776b3e4cecbf8fb8b86>:0
  at Tensorflow.Eager.EagerTensor.ToString () [0x00016] in <f58589f3e4134776b3e4cecbf8fb8b86>:0

