# NumSharp

NumPy port in C# .NET Standard

[![Join the chat at https://gitter.im/publiclab/publiclab](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/numsharp/Lobby)
![NumSharp](https://ci.appveyor.com/api/projects/status/bmaauxd9rx5lsq9i?svg=true)
![NuGet](https://img.shields.io/nuget/dt/NumSharp.svg)

Is it difficult to translate python machine learning code into C#? Because too many functions can’t be found in the corresponding code in the .Net SDK. NumSharp is the C# version of NumPy, which is as consistent as possible with the NumPy programming interface, including function names and parameter locations. By introducing the NumSharp tool library, you can easily convert from python code to C# code. Here is a comparison code between NumSharp and NumPy (left is python, right is C#):

![comparision](docs/_static/screenshots/python-csharp-comparision.png)

NumSharp has implemented the arange, array, max, min, reshape, normalize, unique interfaces. More and more interfaces will be added to the library gradually. If you want to use .NET to get started with machine learning, NumSharp will be your best tool library.

### Implemented APIs

The NumPy class is a high-level abstraction of NDArray that allows NumSharp to be used in the same way as Python's NumPy, minimizing API differences caused by programming language features, allowing .NET developers to maximize Utilize a wide range of NumPy code resources to seamlessly translate python code into .NET code.

* NumPy
  * absolute
  * amax
  * amin
  * arange
  * array
  * hstack
  * linspace
  * random
    * normal
    * randint
    * randn
    * stardard_normal
  * reshape
  * sin
  * vstack
  * zeros
  
### How to use
```
// init NumPy instance which pesists integer data type
var np = new NumPy<int>();
// create a 2-dimension vector
var n = np.arange(12).reshape(3, 4);

// access data by index
Assert.IsTrue(n[1, 1] == 5);
Assert.IsTrue(n[2, 0] == 8);

// create a 3-dimension vector
n = np.arange(12).reshape(2, 3, 2);
// get the 2nd vector in the 1st dimension
var n1 = n.Vector(1);

Assert.IsTrue(n1[1, 1] == 9);
Assert.IsTrue(n1[2, 1] == 11);

// get the 3rd vector in the (axis 1, axis 2) dimension
var n2 = n.Vector(1, 2);

Assert.IsTrue(n2[0] == 10);
Assert.IsTrue(n2[1] == 11);
```

### Install NumSharp in NuGet
```
PM> Install-Package NumSharp
```

### How to make docs
```
$ pip install sphinx
$ pip install recommonmark
$ cd docs
$ make html
```

### How to run benchmark
```
C: \> dotnet NumSharp.Benchmark.dll
```
Reference the online [documents](https://numsharp.readthedocs.io).

NumSharp is referenced by:
* [Pandas.NET](https://github.com/Oceania2018/Pandas.NET)
* [Bigtree.MachineLearning](https://github.com/Oceania2018/Bigtree.MachineLearning)
* [CherubNLP](https://github.com/Oceania2018/CherubNLP)
* [BotSharp](https://github.com/dotnetcore/BotSharp)

Welcome to fork and pull request to add more APIs, and make reference list longer.
