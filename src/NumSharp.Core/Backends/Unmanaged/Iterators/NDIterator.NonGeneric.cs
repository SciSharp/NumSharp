using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp.Backends.Unmanaged
{
    public unsafe class NDIterator : IEnumerable<object>, IDisposable
    {
        private int index;
        public readonly IMemoryBlock Block;
        public readonly Shape Shape;
        public readonly IteratorType Type;
        public bool AutoReset;

        public Func<object> MoveNext;
        public Func<bool> HasNext;
        public Action Reset;

        public NDIterator(IMemoryBlock block, Shape shape, bool autoReset = false)
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
        private NDIterator(UnmanagedStorage storage, bool autoReset = false) : this(storage.InternalArray, storage.Shape, autoReset) { }

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


#if _REGEN
        #region Compute
		switch (localBlock.TypeCode)
		{
			%foreach supported_currently_supported,supported_currently_supported_lowercase%
			case NPTypeCode.#1:
			{
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
                                    return *((#2*)localBlock.Address + offset);
                                };
                            }
                            else
                            {
                                MoveNext = () =>
                                {
                                    hasNext.Value = false;
                                    return *((#2*)localBlock.Address);
                                };
                            }

                            Reset = () => hasNext.Value = true;
                            HasNext = () => hasNext.Value;
                            break;
                        }

                        case IteratorType.Vector:
                        {
                            MoveNext = () => *((#2*)localBlock.Address + shape.GetOffset(index++));
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
                                var ret = *((#2*)localBlock.Address + getOffset(index));
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
                                return *((#2*)localBlock.Address);
                            };
                            Reset = () => hasNext.Value = true;
                            HasNext = () => hasNext.Value;
                            break;

                        case IteratorType.Vector:
                            MoveNext = () => *((#2*)localBlock.Address + index++);
                            Reset = () => index = 0;
                            HasNext = () => index < Shape.size;
                            break;

                        case IteratorType.Matrix:
                        case IteratorType.Tensor:
                            var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                            MoveNext = () =>
                            {
                                var ret = *((#2*)localBlock.Address + iterator.Offset);
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
                break;
			}
			%
			default:
				throw new NotSupportedException();
		}
            #endregion
#else

            #region Compute

            switch (localBlock.TypeCode)
            {
                case NPTypeCode.Boolean:
                {
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
                                        return *((bool*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((bool*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((bool*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((bool*)localBlock.Address + getOffset(index));
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
                                    return *((bool*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((bool*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((bool*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Byte:
                {
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
                                        return *((byte*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((byte*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((byte*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((byte*)localBlock.Address + getOffset(index));
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
                                    return *((byte*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((byte*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((byte*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Int16:
                {
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
                                        return *((short*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((short*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((short*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((short*)localBlock.Address + getOffset(index));
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
                                    return *((short*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((short*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((short*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.UInt16:
                {
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
                                        return *((ushort*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((ushort*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((ushort*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((ushort*)localBlock.Address + getOffset(index));
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
                                    return *((ushort*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((ushort*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((ushort*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Int32:
                {
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
                                        return *((int*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((int*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((int*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((int*)localBlock.Address + getOffset(index));
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
                                    return *((int*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((int*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((int*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.UInt32:
                {
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
                                        return *((uint*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((uint*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((uint*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((uint*)localBlock.Address + getOffset(index));
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
                                    return *((uint*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((uint*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((uint*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Int64:
                {
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
                                        return *((long*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((long*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((long*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((long*)localBlock.Address + getOffset(index));
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
                                    return *((long*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((long*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((long*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.UInt64:
                {
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
                                        return *((ulong*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((ulong*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((ulong*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((ulong*)localBlock.Address + getOffset(index));
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
                                    return *((ulong*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((ulong*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((ulong*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Char:
                {
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
                                        return *((char*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((char*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((char*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((char*)localBlock.Address + getOffset(index));
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
                                    return *((char*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((char*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((char*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Double:
                {
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
                                        return *((double*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((double*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((double*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((double*)localBlock.Address + getOffset(index));
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
                                    return *((double*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((double*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((double*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Single:
                {
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
                                        return *((float*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((float*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((float*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((float*)localBlock.Address + getOffset(index));
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
                                    return *((float*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((float*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((float*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                case NPTypeCode.Decimal:
                {
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
                                        return *((decimal*)localBlock.Address + offset);
                                    };
                                }
                                else
                                {
                                    MoveNext = () =>
                                    {
                                        hasNext.Value = false;
                                        return *((decimal*)localBlock.Address);
                                    };
                                }

                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((decimal*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((decimal*)localBlock.Address + getOffset(index));
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
                                    return *((decimal*)localBlock.Address);
                                };
                                Reset = () => hasNext.Value = true;
                                HasNext = () => hasNext.Value;
                                break;

                            case IteratorType.Vector:
                                MoveNext = () => *((decimal*)localBlock.Address + index++);
                                Reset = () => index = 0;
                                HasNext = () => index < Shape.size;
                                break;

                            case IteratorType.Matrix:
                            case IteratorType.Tensor:
                                var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                                MoveNext = () =>
                                {
                                    var ret = *((decimal*)localBlock.Address + iterator.Offset);
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

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
        }

        protected void autoresetDefault()
        {
            var localBlock = Block;
            Shape shape = Shape;

#if _REGEN
        #region Compute
		switch (localBlock.TypeCode)
		{
			%foreach supported_currently_supported,supported_currently_supported_lowercase%
			case NPTypeCode.#1:
			{
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
                                MoveNext = () => *((#2*)localBlock.Address + offset);
                            }
                            else
                            {
                                MoveNext = () => *((#2*)localBlock.Address);
                            }

                            Reset = () => { };
                            HasNext = () => true;
                            break;
                        }

                        case IteratorType.Vector:
                        {
                            MoveNext = () => *((#2*)localBlock.Address + shape.GetOffset(index++));
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
                                var ret = *((#2*)localBlock.Address + getOffset(index));
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
                            MoveNext = () => *(#2*)localBlock.Address;
                            Reset = () => { };
                            HasNext = () => true;
                            break;
                        case IteratorType.Vector:
                            MoveNext = () => *((#2*)localBlock.Address + index++);
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
                                var ret = *((#2*)localBlock.Address + iterator.Offset);
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
            break;
			%
			default:
				throw new NotSupportedException();
		}
            #endregion
#else

            #region Compute

            switch (localBlock.TypeCode)
            {
                case NPTypeCode.Boolean:
                {
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
                                    MoveNext = () => *((bool*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((bool*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((bool*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((bool*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(bool*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((bool*)localBlock.Address + index++);
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
                                    var ret = *((bool*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Byte:
                {
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
                                    MoveNext = () => *((byte*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((byte*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((byte*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((byte*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(byte*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((byte*)localBlock.Address + index++);
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
                                    var ret = *((byte*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Int16:
                {
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
                                    MoveNext = () => *((short*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((short*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((short*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((short*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(short*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((short*)localBlock.Address + index++);
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
                                    var ret = *((short*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.UInt16:
                {
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
                                    MoveNext = () => *((ushort*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((ushort*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((ushort*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((ushort*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(ushort*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((ushort*)localBlock.Address + index++);
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
                                    var ret = *((ushort*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Int32:
                {
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
                                    MoveNext = () => *((int*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((int*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((int*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((int*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(int*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((int*)localBlock.Address + index++);
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
                                    var ret = *((int*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.UInt32:
                {
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
                                    MoveNext = () => *((uint*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((uint*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((uint*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((uint*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(uint*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((uint*)localBlock.Address + index++);
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
                                    var ret = *((uint*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Int64:
                {
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
                                    MoveNext = () => *((long*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((long*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((long*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((long*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(long*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((long*)localBlock.Address + index++);
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
                                    var ret = *((long*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.UInt64:
                {
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
                                    MoveNext = () => *((ulong*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((ulong*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((ulong*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((ulong*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(ulong*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((ulong*)localBlock.Address + index++);
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
                                    var ret = *((ulong*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Char:
                {
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
                                    MoveNext = () => *((char*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((char*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((char*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((char*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(char*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((char*)localBlock.Address + index++);
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
                                    var ret = *((char*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Double:
                {
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
                                    MoveNext = () => *((double*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((double*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((double*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((double*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(double*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((double*)localBlock.Address + index++);
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
                                    var ret = *((double*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Single:
                {
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
                                    MoveNext = () => *((float*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((float*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((float*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((float*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(float*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((float*)localBlock.Address + index++);
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
                                    var ret = *((float*)localBlock.Address + iterator.Offset);
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
                    break;
                case NPTypeCode.Decimal:
                {
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
                                    MoveNext = () => *((decimal*)localBlock.Address + offset);
                                }
                                else
                                {
                                    MoveNext = () => *((decimal*)localBlock.Address);
                                }

                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            }

                            case IteratorType.Vector:
                            {
                                MoveNext = () => *((decimal*)localBlock.Address + shape.GetOffset(index++));
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
                                    var ret = *((decimal*)localBlock.Address + getOffset(index));
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
                                MoveNext = () => *(decimal*)localBlock.Address;
                                Reset = () => { };
                                HasNext = () => true;
                                break;
                            case IteratorType.Vector:
                                MoveNext = () => *((decimal*)localBlock.Address + index++);
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
                                    var ret = *((decimal*)localBlock.Address + iterator.Offset);
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
                    break;
                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif
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
        public IEnumerator<object> GetEnumerator()
        {
            var next = MoveNext;
            var hasNext = HasNext;

            while (hasNext())
                yield return next();
        }

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
