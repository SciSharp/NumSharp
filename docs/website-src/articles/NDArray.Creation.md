 # Array creation

Before we do some fancy numeric stuff or even machine learning we have to clear one thing. 

**How do we generate NDArrays?**

Since NDArray is the key class in SciSharp stack there must be numerous possibilities how to generate this arrays. And yes that’s the case. 

Maybe first of all we should see the dump way – which can be always used but is not too user friendly.  
In this example we access the Storage property directly - one more reason to avoid it. 

**Dump way**

```CSHARP
// first constructor with data type and shape (here 3x3 matrix)
var nd = new NDArray(typeof(double),3,3);
            
// set 9 elements into the storage of this array.
np.Storage.SetData(new double[] {1,2,3,4,5,6,7,8,9});
```

Ok looks not too difficult. But also not too userfriendly. 

We create an empty NDArray with 3x3 shape, fill it with 9 elements.
We followed the row wise matrix layour by default.

So with this 3x3 shaped NDArray we can do matrix multiplication, QR decomposition, SVD, ...

But keep in mind - always be careful with your shape and be sure what you want to do with your elements in this shape. 

**Create by enumeration**

The next example shows the numpy style creation.

```CSHARP
using NumSharp.Core;

// we take the Data / elements from an array 
// we do not need to define the shape here - it is automaticly shaped to 1D
var np1 = np.array(new double[] {1,2,3,4,5,6} );
```

Ok as we can see, this time the array was created without define the shape. 

This is possible since the method expect that the double[] array shall be transformed into a NDArray directly. 

**Create by implicit cast**

Beside this numpy style C# offers its own flavour of creation.

```CSHARP
using NumSharp.Core;

// implicit cast double[] to NDArray - dtype & shape are deduced by array type and shape
NDArray nd = new double[]{1,2,3,4};
```

And for matrix and n dim tensors also work the same.

```CSHARP
using NumSharp.Core;

NDArray nd = new double[,]{{1,2,3},{4,5,6}};
```

Beside the .NET array to NDArray converting there exist different kinds of methods which also exist in numpy.

**Create by given range**

```CSHARP
using NumSharp.Core;

// we simple say "create an array with 10 elements"
// start is expected to be 0 and step 1
var np1 = np.arange(10);

// this time start with 1, step 2 
// and do it as long as smaller than 10
var np2 = np.arange(1,10,2);
```

**Create diagonal matrix**

```CSHARP
using NumSharp.Core;

// simple 5x5 eye matrix
var nd1 = np.eye(5);

// 3x3 eye matrix but elements different diagonal 
nd1 = np.eye(3,1);
```

**Create by linspace**

```CSHARP
using NumSharp.Core;

// create vector with 50 elements, from 4 to 10
// include last element 
// and convert them to double (float64)
var nd1 = np.linspace(4,10, 50, true, np.float64);
```

