using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp.Backends.Unmanaged
{
    public unsafe class NDIterator<T> : IEnumerable<T>, IDisposable where T : unmanaged
    {
        private int index;
        public readonly IMemoryBlock Block;
        public readonly Shape Shape;
        public readonly IteratorType Type;
        public bool AutoReset;

        public Func<T> MoveNext;
        public Func<bool> HasNext;
        public Action Reset;

        /// <summary>Initializes a new instance of the <see cref="T:System.Object"></see> class.</summary>
        public NDIterator(IMemoryBlock<T> block, Shape shape, bool autoReset = false)
        {
            AutoReset = autoReset;
            Block = block;
            Shape = shape;
            if (shape.IsScalar)
                Type = IteratorType.Scalar;
            else if (shape.NDim == 1)
                Type = IteratorType.Vector;
            else if (shape.NDim == 2)
                Type = IteratorType.Matrix;
            else
                Type = IteratorType.Tensor;

            setDefaults();
        }

        public NDIterator(NDArray arr, bool autoReset = false) : this(arr.Storage, autoReset) { }
        private NDIterator(UnmanagedStorage storage, bool autoReset = false) : this((IMemoryBlock<T>)storage.InternalArray, storage.Shape, autoReset) { }

        protected void setDefaults()
        {
            if (AutoReset)
            {
                autoresetDefault();
                return;
            }

            //non auto-resetting.
            var localBlock = Block;
            Shape shape = Shape;
            if (Shape.IsSliced)
            {
                //Shape is sliced, not auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                    {
                        var hasNext = new Reference<bool>(true);
                        var offset = shape.TransformOffset(0);
                        if (offset != 0)
                        {
                            MoveNext = () =>
                            {
                                hasNext.Value = false;
                                return *((T*)localBlock.Address + offset);
                            };
                        }
                        else
                        {
                            MoveNext = () =>
                            {
                                hasNext.Value = false;
                                return *((T*)localBlock.Address);
                            };
                        }

                        Reset = () => hasNext.Value = true;
                        HasNext = () => hasNext.Value;
                        break;
                    }

                    case IteratorType.Vector:
                    {
                        MoveNext = () => *((T*)localBlock.Address + shape.GetOffset(index++));
                        Reset = () => index = 0;
                        HasNext = () => index < Shape.size;
                        break;
                    }

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                    {
                        var hasNext = new Reference<bool>(true);
                        var iterator = new NDIndexArrayIncrementor(ref shape, _ => hasNext.Value = false);
                        Func<int[], int> getOffset = shape.GetOffset;
                        var index = iterator.Index;
                        
                        MoveNext = () =>
                        {
                            var ret = *((T*)localBlock.Address + getOffset(index));
                            iterator.Next();
                            return ret;
                        };
                        
                        Reset = () =>
                        {
                            iterator.Reset();
                            hasNext.Value = true;
                        };
                        
                        HasNext = () => hasNext.Value;
                        break;
                    }

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            else
            {
                //Shape is not sliced, not auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                        var hasNext = new Reference<bool>(true);
                        MoveNext = () =>
                        {
                            hasNext.Value = false;
                            return *((T*)localBlock.Address);
                        };
                        Reset = () => hasNext.Value = true;
                        HasNext = () => hasNext.Value;
                        break;

                    case IteratorType.Vector:
                        MoveNext = () => *((T*)localBlock.Address + index++);
                        Reset = () => index = 0;
                        HasNext = () => index < Shape.size;
                        break;

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () =>
                        {
                            var ret = *((T*)localBlock.Address + iterator.Offset);
                            iterator.Next();
                            return ret;
                        };
                        Reset = () => iterator.Reset();
                        HasNext = () => iterator.HasNext;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        protected void autoresetDefault()
        {
            var localBlock = Block;
            Shape shape = Shape;
            if (Shape.IsSliced)
            {
                //Shape is sliced, auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                    {
                        var offset = shape.TransformOffset(0);
                        if (offset != 0)
                        {
                            MoveNext = () => *((T*)localBlock.Address + offset);
                        }
                        else
                        {
                            MoveNext = () => *((T*)localBlock.Address);
                        }

                        Reset = () => { };
                        HasNext = () => true;
                        break;
                    }

                    case IteratorType.Vector:
                    {
                        MoveNext = () => *((T*)localBlock.Address + shape.GetOffset(index++));
                        Reset = () => index = 0;
                        HasNext = () =>
                        {
                            if (index < Shape.size) return true;
                            index = 0;
                            return true;
                        };
                        break;
                    }
                    
                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                    {
                        var iterator = new NDIndexArrayIncrementor(ref shape, incr => incr.Reset());
                        Func<int[], int> getOffset = shape.GetOffset;
                        var index = iterator.Index;
                        MoveNext = () =>
                        {
                            var ret = *((T*)localBlock.Address + getOffset(index));
                            if (iterator.Next() == null)
                                iterator.Reset();
                            return ret;
                        };
                        Reset = () => iterator.Reset();
                        HasNext = () => true;
                        break;
                    }

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            else
            {
                //Shape is not sliced, auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                        MoveNext = () => *(T*)localBlock.Address;
                        Reset = () => { };
                        HasNext = () => true;
                        break;
                    case IteratorType.Vector:
                        MoveNext = () => *((T*)localBlock.Address + index++);
                        Reset = () => index = 0;
                        HasNext = () =>
                        {
                            if (index < Shape.size) return true;
                            index = 0;
                            return true;
                        };
                        break;
                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () =>
                        {
                            var ret = *((T*)localBlock.Address + iterator.Offset);
                            iterator.Next();
                            return ret;
                        };
                        HasNext = () =>
                        {
                            if (iterator.HasNext) return true;
                            iterator.Reset();
                            return true;
                        };
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        /// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
        public void Dispose()
        {
            //incase of a cross-reference
            MoveNext = null;
            Reset = null;
            HasNext = null;
        }


        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            var next = MoveNext;
            var hasNext = HasNext;
            
            while (hasNext())
                yield return next();
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator"></see> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>
    ///     Holds a reference to any value.
    /// </summary>
    internal class Reference<T>
    {
        public T Value;

        public Reference(T value)
        {
            Value = value;
        }

        public Reference() { }
    }
}
