using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public struct ValueOffsetIncrementor
    {
        private readonly ValueCoordinatesIncrementor _incr;
        private readonly int[] _index;
        private bool _hasNext;
        private readonly Shape _shape;

        public ValueOffsetIncrementor(Shape shape)
        {
            this._shape = shape;
            _incr = new ValueCoordinatesIncrementor(shape.dimensions);
            _index = _incr.Index;
            _hasNext = true;
        }

        public ValueOffsetIncrementor(int[] dims) : this(new Shape(dims)) {}

        public bool HasNext => _hasNext;

        public void Reset()
        {
            _incr.Reset();
            _hasNext = true;
        }

        [MethodImpl((MethodImplOptions)512)]
        public int Next()
        {
            if (!_hasNext)
                return -1;

            int offset = 0;
            if (_shape.IsSliced)
            {
                offset = _shape.GetOffset(_index);
            }
            else
            {
                unchecked
                {
                    for (int i = 0; i < _index.Length; i++)
                        offset += _shape.strides[i] * _index[i];
                }
            }

            if (_incr.Next() == null)
                _hasNext = false;

            return offset;
        }
    }

    public struct ValueOffsetIncrementorAutoresetting
    {
        private readonly ValueCoordinatesIncrementor incr;
        private readonly int[] index;
        private readonly Shape shape;

        public ValueOffsetIncrementorAutoresetting(Shape shape)
        {
            this.shape = shape;
            incr = new ValueCoordinatesIncrementor(shape.dimensions, (ref ValueCoordinatesIncrementor incrementor) => incrementor.Reset());
            index = incr.Index;
        }

        public ValueOffsetIncrementorAutoresetting(int[] dims) : this(new Shape(dims)) { }


        public bool HasNext => true;

        public void Reset()
        {
            incr.Reset();
        }

        [MethodImpl((MethodImplOptions)512)]
        public int Next()
        {
            int offset = 0;
            if (shape.IsSliced)
            {
                offset = shape.GetOffset(index);
            }
            else
            {
                unchecked
                {
                    for (int i = 0; i < index.Length; i++)
                        offset += shape.strides[i] * index[i];
                }
            }

            incr.Next();

            return offset;
        }
    }
}
