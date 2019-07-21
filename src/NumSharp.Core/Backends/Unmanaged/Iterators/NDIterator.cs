using System;

namespace NumSharp.Backends.Unmanaged
{
    public unsafe class NDIterator<T> where T : unmanaged
    {
        private int index;
        public readonly IMemoryBlock Block;
        public readonly Shape Shape;
        public readonly IteratorType Type;
        public bool AutoReset;

        public Func<T> MoveNext;
        public Func<bool> HasNext;
        public Action Reset;
        public T Current;

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

        protected void setDefaults()
        {
            var localBlock = Block;

            if (AutoReset)
            {
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
                        MoveNext = () => *((T*)localBlock.Address + iterator.Next());
                        Reset = () => iterator.Reset();
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
            else

                switch (Type)
                {
                    case IteratorType.Scalar:
                        MoveNext = () => *(T*)localBlock.Address;
                        Reset = () => index = Shape.size - 1;
                        HasNext = () => index >= 0;
                        break;
                    case IteratorType.Vector:
                        MoveNext = () => *((T*)localBlock.Address + index++);
                        Reset = () => index = 0;
                        HasNext = () => index < Shape.size;
                        break;
                    case IteratorType.Matrix:
                    case IteratorType.Tensor:
                        var iterator = new NDOffsetIncrementor(Shape.dimensions, Shape.strides, Shape.size); //we do not copy the dimensions because there is not risk for the iterator's shape to change.
                        MoveNext = () => *((T*)localBlock.Address + iterator.Next());
                        Reset = () => iterator.Reset();
                        HasNext = () => iterator.HasNext;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
        }


        /// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
        public void Dispose()
        {
            MoveNext = null;
            Reset = null;
        }

        private void Nop() { }
    }
}
