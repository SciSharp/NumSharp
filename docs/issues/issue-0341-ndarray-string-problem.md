# #341: NDArray string problem

- **URL:** https://github.com/SciSharp/NumSharp/issues/341
- **State:** OPEN
- **Author:** @lokinfey
- **Created:** 2019-08-26T02:39:33Z
- **Updated:** 2019-08-28T21:45:29Z
- **Labels:** missing feature/s
- **Assignees:** @Nucs

## Description

I try to use
 var bb8List =  new NDArray(typeof(string) , new Shape(bb8Num));

but it show error like this 

Exception has occurred: CLR/System.NotSupportedException
An unhandled exception of type 'System.NotSupportedException' occurred in NumSharp.Core.dll: 'Specified method is not supported.'
   at NumSharp.Backends.Unmanaged.ArraySlice.Allocate(Type elementType, Int32 count, Boolean fillDefault)
   at NumSharp.Backends.UnmanagedStorage.Allocate(Shape shape, Type dtype, Boolean fillZeros)
   at NumSharp.NDArray..ctor(Type dtype, Shape shape)
   at TFDemo.Program.Main(String[] args) in /Users/lokinfey/Desktop/File/Proj/AI/tensorflownet/code/TFDemo/Program.cs:line 61

## Comments

### Comment 1 by @Nucs (2019-08-26T13:02:40Z)

We currently do not support NDArray of string because `System.String` is not an unmanaged object which causes multiple problems with our current backend architecture since strings can have different sizes.
We do plan to implement it ASAP.
