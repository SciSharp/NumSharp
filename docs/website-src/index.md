# Welcome to NumSharp

NumSharp is a .NET port of Python's NumPy library, bringing powerful numerical computing to the .NET ecosystem.

## Why NumSharp?

- **NumPy API compatibility** - Feel right at home if you're coming from Python
- **High-performance NDArray** - Multi-dimensional arrays stored efficiently in unmanaged memory
- **Full .NET integration** - Works seamlessly with C#, F#, VB.NET, and other .NET languages
- **Part of the SciSharp ecosystem** - Works alongside TensorFlow.NET, ML.NET, and other ML libraries

## Quick Start

```bash
dotnet add package NumSharp
```

```csharp
using NumSharp;

var a = np.array(new int[] { 1, 2, 3, 4, 5 });
var b = np.arange(5);
var c = a + b;
Console.WriteLine(c);  // [1, 3, 5, 7, 9]
```

## Features

- **Array Creation** - `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, and more
- **Array Manipulation** - Reshape, transpose, concatenate, stack operations
- **Math Operations** - Element-wise arithmetic, broadcasting, linear algebra
- **Slicing & Indexing** - NumPy-style slicing with views, not copies
- **Random Sampling** - Full numpy.random compatibility with seed/state matching
- **File I/O** - Load and save `.npy` and `.npz` files

## Get Started

- [Introduction](docs/intro.md) - Learn about NDArray, Shape, and Storage
- [NumPy Compliance](docs/compliance.md) - Compatibility status and roadmap
- [API Reference](api/index.md) - Full API documentation

## Community

- [GitHub Repository](https://github.com/SciSharp/NumSharp) - Star us, report issues, contribute
- [NuGet Package](https://www.nuget.org/packages/NumSharp) - Latest stable release
