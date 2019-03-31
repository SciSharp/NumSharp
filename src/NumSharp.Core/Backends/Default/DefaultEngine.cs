using NumSharp.Interfaces;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace NumSharp.Backends
{
    /// <summary>
    /// Default Tensor Engine
    ///
    /// Responsible for :
    ///
    ///  - store data type, elements, Shape
    ///  - offers methods for accessing elements depending on shape
    ///  - offers methods for casting elements
    ///  - offers methods for change tensor order
    ///  - GetData always return reference object to the true storage
    ///  - GetData<T> and SetData<T> change dtype and cast storage
    ///  - CloneData always create a clone of storage and return this as reference object
    ///  - CloneData<T> clone storage and cast this clone 
    ///     
    /// </summary>
    public abstract partial class DefaultEngine : ITensorEngine
    {
    }
}