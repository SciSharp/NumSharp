# NumSharp
NumPy port in C#

Is it difficult to translate python machine learning code into C#? Because too many functions can’t be found in the corresponding code in the .Net SDK. NumSharp is the C# version of NumPy, which is as consistent as possible with the NumPy programming interface, including function names and parameter locations. By introducing the NumSharp tool library, you can easily convert from python code to C# code. Here is a comparison code between NumSharp and NumPy (left is python, right is C#):

![comparision](docs/_static/screenshots/python-csharp-comparision.png)

NumSharp has implemented the arange, array, max, min, reshape, normalize, unique interfaces. More and more interfaces will be added to the library gradually. If you want to use .NET to get started with machine learning, NumSharp will be your best tool library.

## Implemented APIs
* NdArray
  * Arange
  * Array
  * Delete
  * Max
  * Min
  * Normalize
  * Random
  * ReShape
  * Unique
  * Zeros
  
* NdArrayRandom
  * Permutation

## Install NumSharp in NuGet
```
Install-Package NumSharp -Version 0.1.0
```

NumSharp is referenced by:
* [Bigtree.MachineLearning](https://github.com/Oceania2018/Bigtree.MachineLearning)

Welcome to fork and pull request to add more APIs, and make reference list longer.
