﻿//Generated by Regex Templating Engine at 03/08/2019 23:16:43 UTC
//template source: C:\Users\Eli-PC\Desktop\SciSharp\NumSharp\src\NumSharp.Core\Backends\Iterators\NDIterator.template.cs

using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public unsafe partial class NDIterator<TOut>
    {
        protected void setDefaults_Int64() //Int64 is the input type
        {
            if (AutoReset)
            {
                autoresetDefault_Int64();
                return;
            }

            if (typeof(TOut) == typeof(Int64))
            {
                setDefaults_NoCast();
                return;
            }

            var convert = Converts.FindConverter<Int64, TOut>();

            //non auto-resetting.
            var localBlock = Block;
            Shape shape = Shape;
            if (!Shape.IsContiguous || Shape.ModifiedStrides)
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
                                    return convert(*((Int64*)localBlock.Address + offset));
                                };
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }
                            else
                            {
                                MoveNext = () =>
                                {
                                    hasNext.Value = false;
                                    return convert(*((Int64*)localBlock.Address));
                                };
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }

                            Reset = () => hasNext.Value = true;
                            HasNext = () => hasNext.Value;
                            break;
                        }

                    case IteratorType.Vector:
                        {
                            MoveNext = () => convert(*((Int64*)localBlock.Address + shape.GetOffset(index++)));
                            MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            Reset = () => index = 0;
                            HasNext = () => index < Shape.size;
                            break;
                        }

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        {
                            var hasNext = new Reference<bool>(true);
                            var iterator = new ValueNDCoordinatesIncrementor(ref shape, delegate(ref ValueNDCoordinatesIncrementor _) { hasNext.Value = false; });
                            Func<int[], int> getOffset = shape.GetOffset;
                            var index = iterator.Index;

                            MoveNext = () =>
                            {
                                var ret = convert(*((Int64*)localBlock.Address + getOffset(index)));
                                iterator.Next();
                                return ret;
                            };
                            MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");

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
                            return convert(*((Int64*)localBlock.Address));
                        };
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => hasNext.Value = true;
                        HasNext = () => hasNext.Value;
                        break;

                    case IteratorType.Vector:
                        MoveNext = () => convert(*((Int64*)localBlock.Address + index++));
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => index = 0;
                        HasNext = () => index < Shape.size;
                        break;

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new NDOffsetIncrementor(Shape); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () => convert(*((Int64*)localBlock.Address + iterator.Next()));
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => iterator.Reset();
                        HasNext = () => iterator.HasNext;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        protected void autoresetDefault_Int64()
        {
            if (typeof(TOut) == typeof(Int64))
            {
                autoresetDefault_NoCast();
                return;
            }

            var localBlock = Block;
            Shape shape = Shape;
            var convert = Converts.FindConverter<Int64, TOut>();

            if (!Shape.IsContiguous || Shape.ModifiedStrides)
            {
                //Shape is sliced, auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                        {
                            var offset = shape.TransformOffset(0);
                            if (offset != 0)
                            {
                                MoveNext = () => convert(*((Int64*)localBlock.Address + offset));
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }
                            else
                            {
                                MoveNext = () => convert(*((Int64*)localBlock.Address));
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }

                            Reset = () => { };
                            HasNext = () => true;
                            break;
                        }

                    case IteratorType.Vector:
                        {
                            var size = Shape.size;
                            MoveNext = () =>
                            {
                                var ret = convert(*((Int64*)localBlock.Address + shape.GetOffset(index++)));
                                if (index >= size)
                                    index = 0;
                                return ret;
                            };
                            MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");

                            Reset = () => index = 0;
                            HasNext = () => true;
                            break;
                        }

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        {
                            var iterator = new ValueNDCoordinatesIncrementor(ref shape, delegate(ref ValueNDCoordinatesIncrementor incr) { incr.Reset(); });
                            var index = iterator.Index;
                            Func<int[], int> getOffset = shape.GetOffset;
                            MoveNext = () =>
                            {
                                var ret = convert(*((Int64*)localBlock.Address + getOffset(index)));
                                iterator.Next();
                                return ret;
                            };
                            MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
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
                        MoveNext = () => convert(*(Int64*)localBlock.Address);
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => { };
                        HasNext = () => true;
                        break;
                    case IteratorType.Vector:
                        var size = Shape.size;
                        MoveNext = () =>
                        {
                            var ret = convert(*((Int64*)localBlock.Address + index++));
                            if (index >= size)
                                index = 0;
                            return ret;
                        };
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => index = 0;
                        HasNext = () => true;
                        break;
                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new NDOffsetIncrementorAutoresetting(Shape); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () => convert(*((Int64*)localBlock.Address + iterator.Next()));
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        HasNext = () => true;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }
    }
}
