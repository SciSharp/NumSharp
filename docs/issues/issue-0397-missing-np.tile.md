# #397: Missing np.tile?

- **URL:** https://github.com/SciSharp/NumSharp/issues/397
- **State:** OPEN
- **Author:** @QadiymStewart
- **Created:** 2020-02-12T15:55:12Z
- **Updated:** 2020-06-03T13:42:46Z
- **Assignees:** @Nucs

## Description

Is np.tile implemented in this library as of yet?

## Comments

### Comment 1 by @simonbuehler (2020-05-25T10:25:11Z)

hi,

bumped into the same issue, is there a workaround for something like `np.tile(center, (1, 1, 2 * num_anchors));` or maybe even better a working tile implementation?

### Comment 2 by @QadiymStewart (2020-05-25T14:41:50Z)

> hi,
> 
> bumped into the same issue, is there a workaround for something like `np.tile(center, (1, 1, 2 * num_anchors));` or maybe even better a working tile implementation?

Switched to https://github.com/Quansight-Labs/numpy.net

### Comment 3 by @simonbuehler (2020-05-25T14:56:20Z)

@QadiymStewart thanks for your reply, a pity that tensorflow.net depends on NumSharp so i can't switch  :/  
@Oceania2018 are there plans to implement this like in[NumpyDotNet/shape_base.cs](https://github.com/Quansight-Labs/numpy.net/blob/b6ac4af87e21bd561a022ebe067d322b88273157/src/NumpyDotNet/NumpyDotNet/shape_base.cs#L1623) ?

### Comment 4 by @Oceania2018 (2020-05-25T15:23:08Z)

@simonbuehler Try to use the built-in functions of tensorflow. https://www.tensorflow.org/api_docs/python/tf/tile

### Comment 5 by @simonbuehler (2020-05-25T17:54:25Z)

is there already  a nuget  with a  Tensor -> NDarray  method?

### Comment 6 by @Oceania2018 (2020-05-26T03:38:36Z)

@simonbuehler You can create tensor directly from Tensor -> NDArray and vice verse.

### Comment 7 by @simonbuehler (2020-05-26T08:54:41Z)

just for info, unfortunatly `var center_tiled = tf.tile(temp , temp2);` throws a `NullReferenceException` guess this is a TensorFlow.NET issue, nevertheless a numsharp implementation would be awesome!

```
 	TensorFlow.NET.dll!Tensorflow.Tensor._as_tf_output()	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.ops._create_c_op<object>(Tensorflow.Graph graph, Tensorflow.NodeDef node_def, object[] inputs, Tensorflow.Operation[] control_inputs)	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.Operation.Operation(Tensorflow.NodeDef node_def, Tensorflow.Graph g, Tensorflow.Tensor[] inputs, Tensorflow.TF_DataType[] output_types, Tensorflow.ITensorOrOperation[] control_inputs, Tensorflow.TF_DataType[] input_types, string original_op, Tensorflow.OpDef op_def)	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.Graph.create_op(string op_type, Tensorflow.Tensor[] inputs, Tensorflow.TF_DataType[] dtypes, Tensorflow.TF_DataType[] input_types, string name, System.Collections.Generic.Dictionary<string, Tensorflow.AttrValue> attrs, Tensorflow.OpDef op_def)	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.OpDefLibrary._apply_op_helper.AnonymousMethod__0(Tensorflow.ops.NameScope scope)	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.Binding.tf_with<System.__Canon, System.__Canon>(System.__Canon py, System.Func<System.__Canon, System.__Canon> action)	Unbekannt
 	TensorFlow.NET.dll!Tensorflow.gen_array_ops.tile<Tensorflow.Tensor>(Tensorflow.Tensor input, Tensorflow.Tensor multiples, string name)	Unbekannt

```

### Comment 8 by @simonbuehler (2020-06-03T13:42:46Z)

@Oceania2018 hi, are there any chances that np.tile could be implemented?
