# #383: is there any way to convert NumSharp.NDArray to Numpy.NDarray?

- **URL:** https://github.com/SciSharp/NumSharp/issues/383
- **State:** OPEN
- **Author:** @lelelemonade
- **Created:** 2020-01-06T10:31:01Z
- **Updated:** 2023-02-24T10:15:20Z

## Description

_No description provided._

## Comments

### Comment 1 by @Nucs (2020-01-06T11:15:15Z)

Me and @henon have collaborated just two days ago on a solution for that, 
The trick is to create a python scope, load a script that takes in a pointer-length and returns an np.ndarray
Remember! you have to keep reference to the original NDArray, otherwise the NDArray will be disposed and will release the memory. (@henon is there a tag field we can use in NDarray?)

Note: the code below is psuedo, the rest of the code can be found here: [ConsoleApp9.zip](https://github.com/SciSharp/NumSharp/files/4025804/ConsoleApp9.zip)


*numpy_interop.py* as embedded resource
```python
import numpy as np
import ctypes

def to_numpy(ptr, bytes, dtype, shape):
    return np.frombuffer((ctypes.c_uint8*(bytes)).from_address(ptr), dtype).reshape(shape)

```

*PythonExtensions.cs*
```C#
public static class PythonExtensions {

    ...
    private static PyScope NumpyInteropScope;
    private static PyObject NumpyConverter;

    public static unsafe NDarray ToNumpyNET(this NDArray nd) {
        using (Py.GIL()) {
            if (NumpyInteropScope == null) {
                NumpyInteropScope = Py.CreateScope();
                NumpyInteropScope.ExecResource("numpy_interop.py");
                NumpyConverter = NumpyInteropScope.GetFunction("to_numpy");
            }

            return new NDarray(NumpyConverter.Invoke(new PyLong((long) nd.Unsafe.Address),
                new PyLong(nd.size * nd.dtypesize),
                new PyString(nd.dtype.Name.ToLowerInvariant()),
                new PyTuple(nd.shape.Select(dim => (PyObject) new PyLong(dim)).ToArray())));
        }
    }
}
```

*main.cs*
```C#
using numsharp_np = NumSharp.np;
using numpy_np = Numpy.np;
static void Main(string[] args) {
    var numsharp_nd = numsharp_np.arange(10);
    var numpy_nd = numsharp_nd.ToNumpyNET();
    numpy_nd[1] = (NDarray) 5;
    Debug.Assert(numsharp_nd[1].array_equal(5));
    Console.WriteLine("numsharp_nd[1].array_equal(5) is indeed true!");
}
```

### Comment 2 by @Nucs (2020-01-06T11:35:04Z)

Added some changes that will make sure the NDarray will hold reference to the NDArray as long as the PyObject doesnt change internally.

[ConsoleApp9.zip](https://github.com/SciSharp/NumSharp/files/4025864/ConsoleApp9.zip)



### Comment 3 by @henon (2020-01-06T15:13:53Z)

> Remember! you have to keep reference to the original NDArray, otherwise the NDArray will be disposed and will release the memory. (@henon is there a tag field we can use in NDarray?)
> 

Not yet, but I will add one in Numpy.NET and support this.


### Comment 4 by @davidvct (2023-02-24T04:14:22Z)

Hi, could you provide instruction on how to use the ConsoleApp9.zip? And does it able to convert Numpy array back to NumSharp array?

### Comment 5 by @henon (2023-02-24T10:15:20Z)

@davidvct, if I remember correctly I have added support for copying data to and from Numpy.NET like @Nucs did in this solution. Read this to see how: https://github.com/SciSharp/Numpy.NET#create-a-numpy-array-from-a-c-array-and-vice-versa
