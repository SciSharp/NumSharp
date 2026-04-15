using System;
using System.Numerics;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public unsafe partial class NDIterator<TOut>
    {
        protected void setDefaults_Complex() //Complex is the input type
        {
            if (AutoReset)
            {
                autoresetDefault_Complex();
                return;
            }

            if (typeof(TOut) == typeof(Complex))
            {
                setDefaults_NoCast();
                return;
            }

            var convert = Converts.FindConverter<Complex, TOut>();

            //non auto-resetting.
            var localBlock = Block;
            Shape shape = Shape;
            if (!Shape.IsContiguous || Shape.offset != 0)
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
                                    return convert(*((Complex*)localBlock.Address + offset));
                                };
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }
                            else
                            {
                                MoveNext = () =>
                                {
                                    hasNext.Value = false;
                                    return convert(*((Complex*)localBlock.Address));
                                };
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }

                            Reset = () => hasNext.Value = true;
                            HasNext = () => hasNext.Value;
                            break;
                        }

                    case IteratorType.Vector:
                        {
                            MoveNext = () => convert(*((Complex*)localBlock.Address + shape.GetOffset(index++)));
                            MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            Reset = () => index = 0;
                            HasNext = () => index < Shape.size;
                            break;
                        }

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        {
                            var hasNext = new Reference<bool>(true);
                            var iterator = new ValueCoordinatesIncrementor(ref shape, delegate(ref ValueCoordinatesIncrementor _) { hasNext.Value = false; });
                            Func<long[], long> getOffset = shape.GetOffset;
                            var index = iterator.Index;

                            MoveNext = () =>
                            {
                                var ret = convert(*((Complex*)localBlock.Address + getOffset(index)));
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
                            return convert(*((Complex*)localBlock.Address));
                        };
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => hasNext.Value = true;
                        HasNext = () => hasNext.Value;
                        break;

                    case IteratorType.Vector:
                        MoveNext = () => convert(*((Complex*)localBlock.Address + index++));
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => index = 0;
                        HasNext = () => index < Shape.size;
                        break;

                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new ValueOffsetIncrementor(Shape); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () => convert(*((Complex*)localBlock.Address + iterator.Next()));
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => iterator.Reset();
                        HasNext = () => iterator.HasNext;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        protected void autoresetDefault_Complex()
        {
            if (typeof(TOut) == typeof(Complex))
            {
                autoresetDefault_NoCast();
                return;
            }

            var localBlock = Block;
            Shape shape = Shape;
            var convert = Converts.FindConverter<Complex, TOut>();

            if (!Shape.IsContiguous || Shape.offset != 0)
            {
                //Shape is sliced, auto-resetting
                switch (Type)
                {
                    case IteratorType.Scalar:
                        {
                            var offset = shape.TransformOffset(0);
                            if (offset != 0)
                            {
                                MoveNext = () => convert(*((Complex*)localBlock.Address + offset));
                                MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                            }
                            else
                            {
                                MoveNext = () => convert(*((Complex*)localBlock.Address));
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
                                var ret = convert(*((Complex*)localBlock.Address + shape.GetOffset(index++)));
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
                            var iterator = new ValueCoordinatesIncrementor(ref shape, delegate(ref ValueCoordinatesIncrementor incr) { incr.Reset(); });
                            var index = iterator.Index;
                            Func<long[], long> getOffset = shape.GetOffset;
                            MoveNext = () =>
                            {
                                var ret = convert(*((Complex*)localBlock.Address + getOffset(index)));
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
                        MoveNext = () => convert(*(Complex*)localBlock.Address);
                        MoveNextReference = () => throw new NotSupportedException("Unable to return references during iteration when casting is involved.");
                        Reset = () => { };
                        HasNext = () => true;
                        break;
                    case IteratorType.Vector:
                        var size = Shape.size;
                        MoveNext = () =>
                        {
                            var ret = convert(*((Complex*)localBlock.Address + index++));
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
                        var iterator = new ValueOffsetIncrementorAutoresetting(Shape); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () => convert(*((Complex*)localBlock.Address + iterator.Next()));
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
