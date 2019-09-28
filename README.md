# NumSharp

[NumPy](https://github.com/numpy/numpy) port to C# targetting .NET Standard.<a href="http://scisharpstack.org"><img src="https://github.com/SciSharp/SciSharp/blob/master/art/scisharp_badge.png" width="200" height="200" align="right" /></a><br>
NumSharp is the fundamental package needed for scientific computing with C#.<br>
[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sci-sharp/community)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/bmaauxd9rx5lsq9i?svg=true)](https://ci.appveyor.com/project/Haiping-Chen/numsharp)
[![codecov](https://codecov.io/gh/SciSharp/NumSharp/branch/master/graph/badge.svg)](https://codecov.io/gh/SciSharp/NumSharp)
[![NuGet](https://img.shields.io/nuget/dt/NumSharp.svg)](https://www.nuget.org/packages/NumSharp)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/en_US)

Is it difficult to translate python machine learning code into C#? Because too many functions can’t be found in the corresponding code in the .NET SDK. 
NumSharp is the C# version of NumPy, which is as consistent as possible with the NumPy programming interface, including function names and parameter locations. By introducing the NumSharp tool library, you can easily convert from python code to C# code.
Here is a comparison code between NumSharp and NumPy (left is python, right is C#):

![comparision](docfx_project/images/python-csharp-comparision.png)

### Bold Features
* Use of Unmanaged Memory and fast unsafe algorithms.
* [Broadcasting](https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html) n-d shapes against each other. ([intro](https://machinelearningmastery.com/broadcasting-with-numpy-arrays/))
* [NDArray Slicing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html) and nested/recusive slicing (`nd["-1, ::2"]["1::3, :, 0"]`)
* Axis iteration and support in all of our implemented functions.
* Full and precise (to numpy) automatic type resolving and conversion (upcasting, downcasting and other cases)
* Non-copy - most cases, similarly to numpy, do not perform copying but returns a view instead.
* Almost non-effort copy-pasting numpy code from python to C#.

### Implemented APIs
The NumPy class is a high-level abstraction of NDArray that allows NumSharp to be used in the same way as Python's NumPy, minimizing API differences caused by programming language features, allowing .NET developers to maximize Utilize a wide range of NumPy code resources to seamlessly translate python code into .NET code.

### Install NumSharp in NuGet
```sh
PM> Install-Package NumSharp
```

### How to use
```cs
using NumSharp;

var nd = np.full(5, 12); //[5, 5, 5 .. 5]
nd = np.zeros(12); //[0, 0, 0 .. 0]
nd = np.arange(12); //[0, 1, 2 .. 11]

// create a matrix
nd = np.zeros((3, 4)); //[0, 0, 0 .. 0]
nd = np.arange(12).reshape(3, 4);

// access data by index
var data = nd[1, 1];

// create a tensor
nd = np.arange(12);

// reshaping
data = nd.reshape(2, -1); //returning ndarray shaped (2, 6)

Shape shape = (2, 3, 2);
data = nd.reshape(shape); //Tuple implicitly casted to Shape
    //or:
nd =   nd.reshape(2, 3, 2);

// slicing tensor
data = nd[":, 0, :"]; //returning ndarray shaped (2, 1, 2)
data = nd[Slice.All, 0, Slice.All]; //equivalent to the line above.

// nd is currently shaped (2, 3, 2)
// get the 2nd vector in the 1st dimension
data = nd[1]; //returning ndarray shaped (3, 2)

// get the 3rd vector in the (axis 1, axis 2) dimension
data = nd[1, 2]; //returning ndarray shaped (2, )

// get flat representation of nd
data = nd.flat; //or nd.flatten() for a copy

// interate ndarray
foreach (object val in nd)
{
    // val can be either boxed value-type or a NDArray.
}

var iter = nd.AsIterator<int>(); //a different T can be used to automatically perform cast behind the scenes.
while (iter.HasNext())
{
    //read
    int val = iter.MoveNext();

    //write
    iter.MoveNextReference() = 123; //set value to the next val
    //note that setting is not supported when calling AsIterator<T>() where T is not the dtype of the ndarray.
}
```

### How to run benchmark
```
C: \> dotnet NumSharp.Benchmark.dll nparange
```

### NumSharp is referenced by
* [dotnet/ML.NET](https://github.com/dotnet/machinelearning)
* [ScipSharp/TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET)
* [ScipSharp/Gym.NET](https://github.com/SciSharp/Gym.NET)
* [ScipSharp/Pandas.NET](https://github.com/SciSharp/Pandas.NET)
* [Oceania2018/Bigtree.MachineLearning](https://github.com/Oceania2018/Bigtree.MachineLearning)
* [Oceania2018/CherubNLP](https://github.com/Oceania2018/CherubNLP)
* [SciSharp/BotSharp](https://github.com/SciSharp/BotSharp)

You might also be interested in NumSharp's sister project [Numpy.NET](https://github.com/SciSharp/Numpy.NET) which provides a more whole implementation of numpy by using [pythonnet](https://github.com/pythonnet/pythonnet) and [behind-the-scenes deployment of python](https://github.com/henon/Python.Included) ([read more](https://henon.wordpress.com/2019/06/05/using-python-libraries-in-net-without-a-python-installation/)).

NumSharp is a member project of [SciSharp.org](https://github.com/SciSharp) which is the .NET based ecosystem of open-source software for mathematics, science, and engineering.

### Regen Templating
Our library contains over 150,000 lines of generated code, mostly for handling different data types without hurting performance.<br>
The templates can be recognized with `#if _REGEN` blocks and are powered by [Regen Templating Engine](https://github.com/Nucs/Regen).
Regen is an external tool (Visual studio extension, [download here](https://github.com/Nucs/Regen/tree/master/releases)) that are generated on demand.
