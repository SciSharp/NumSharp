using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public partial class UnmanagedByteStorage<T> : IEnumerable<T>, ICloneable where T : unmanaged
    {
        public static readonly NPTypeCode TypeCode = typeof(T).GetTypeCode();
        private static readonly InternalBufferManager.PooledBufferManager _scalarPool = ScalarMemoryPool.Instance;

        public const int ParallelLimit = 84999;

        public readonly unsafe T* Address;
        private ArraySlice<T> _array;
        private Shape _shape;

        public Shape Shape
        {
            get => _shape;
            set
            {
                if (_shape.Size != value.Size)
                    throw new Exception("Shape is incorrent for this DArray."); //TODO! missmatch exception
                _shape = value;
            }
        }

        /// <summary>
        ///     The current slice this <see cref="IStorage"/> instance currently represent.
        /// </summary>
        public Slice Slice { get; set; }

        #region Constructors

        static UnmanagedByteStorage()
        {
            //init statics
            var _ = new UnmanagedMemoryBlock<T>();
            var __ = new ArraySlice<T>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="length">The length of this DArray.</param>
        /// <param name="shape">The shape of this DArray.</param>
        /// <param name="assignZeros">Assign all internal storage with <see cref="default(T)"/>.</param>
        public unsafe UnmanagedByteStorage(Shape shape, bool assignZeros)
        {
            _shape = shape;
            _array = new ArraySlice<T>(assignZeros ? new UnmanagedMemoryBlock<T>(shape.Size, default) : new UnmanagedMemoryBlock<T>(shape.Size));
            Address = _array.Address;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="length">The length of this DArray.</param>
        /// <param name="shape">The shape of this DArray.</param>
        /// <param name="assignZeros">Assign all internal storage with <see cref="default(T)"/>.</param>
        public unsafe UnmanagedByteStorage(Shape shape)
        {
            _shape = shape;
            _array = new ArraySlice<T>(new UnmanagedMemoryBlock<T>(shape.Size));
            Address = _array.Address;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="length">The length of this DArray.</param>
        /// <param name="shape">The shape of this DArray.</param>
        /// <param name="assignZeros">Assign all internal storage with <see cref="default(T)"/>.</param>
        /// <param name="fill"></param>
        public unsafe UnmanagedByteStorage(Shape shape, T fill)
        {
            _shape = shape;
            _array = new ArraySlice<T>(new UnmanagedMemoryBlock<T>(shape.Size, fill));
            Address = _array.Address;
        }

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public unsafe UnmanagedByteStorage(T scalar)
        {
            _shape = Shape.Scalar;
            var mem = UnmanagedMemoryBlock<T>.FromPool(1, _scalarPool);
            _array = new ArraySlice<T>(mem);
            *(Address = _array.Address) = scalar;
        }

        public unsafe UnmanagedByteStorage(T[] arr, Shape shape)
        {
            _shape = shape;
            _array = new ArraySlice<T>(UnmanagedMemoryBlock<T>.FromArray(arr));
            Address = _array.Address;
        }

        public unsafe UnmanagedByteStorage(ArraySlice<T> arr, Shape shape)
        {
            _shape = shape;
            _array = arr;
            Address = _array.Address;
        }

        public unsafe UnmanagedByteStorage(UnmanagedMemoryBlock<T> memoryBlock, Shape shape)
        {
            _shape = shape;
            _array = new ArraySlice<T>(memoryBlock);
            Address = _array.Address;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="lengthInSizeOfT"></param>
        /// <param name="shape"></param>
        /// <param name="dispose"></param>
        public unsafe UnmanagedByteStorage(T* ptr, int lengthInSizeOfT, Shape shape, Action dispose)
        {
            _shape = shape;
            _array = new ArraySlice<T>(new UnmanagedMemoryBlock<T>(ptr, lengthInSizeOfT, dispose));
            Address = _array.Address;
        }

        #endregion

        public UnmanagedByteStorage<T> this[params int[] indices]
        {
            [MethodImpl((MethodImplOptions)768)] get => Get(indices);
            [MethodImpl((MethodImplOptions)768)] set => Set(value, indices);
        }

        public int Count
        {
            [MethodImpl((MethodImplOptions)768)] get { return _array.Count; }
        }

        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedByteStorage<T> Get(params int[] indices)
        {
            var (shape, offset) = _shape.GetSubshape(indices);
            return new UnmanagedByteStorage<T>(_array.Slice(offset, shape.Size), shape);
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetScalar(params int[] indices)
        {
#if DEBUG
            var (shape, offset) = _shape.GetSubshape(indices);
            Debug.Assert(shape.IsScalar);
#else
            var (_, offset) = _shape.GetSubshape(indices);
#endif
            return GetIndex(offset);
        }

        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedByteStorage<T> GetCopy(params int[] indices)
        {
            var (shape, offset) = _shape.GetSubshape(indices);
            return new UnmanagedByteStorage<T>(_array.Slice(offset, shape.Size).Clone(), shape);
        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe void Set(T value, params int[] indices)
        {
#if DEBUG
            if (indices.Length != _shape.NDim)
                throw new InvalidOperationException();
#endif
            *(Address + _shape.GetIndexInShape(indices)) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe void Set(UnmanagedByteStorage<T> value, params int[] indices)
        {
            var (shape, offset) = _shape.GetSubshape(indices);
            if (shape != value.Shape)
                throw new Exception("Shape do not match"); //todo! ShapeMissmatchException.

            value._array.CopyTo(new Span<T>(Address + offset, shape.Size));
        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe void SetIndex(T value, int index)
        {
            *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public unsafe T GetIndex(int index)
        {
            return *(Address + index);
        }

        //public override T this[int index0] {
        //    get => _array[index0];
        //    set => _array[index0] = value;
        //}
        //
        //public override T this[int index0, int index1] {
        //    get => throw new NotSupportedException();
        //    set => throw new NotSupportedException();
        //}
        //
        //public override T this[int index0, int index1, int index2] {
        //    get => throw new NotSupportedException();
        //    set => throw new NotSupportedException();
        //}

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            return _enumerate().GetEnumerator();
        }

        public IEnumerable<T> _enumerate()
        {
            var cnt = Count;
            for (int i = 0; i < cnt; i++)
            {
                yield return _array[i];
            }
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator" /> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }


        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="UnmanagedByteStorage{T}"/>.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static unsafe UnmanagedByteStorage<T> Scalar(T val)
        {
            var ret = new UnmanagedByteStorage<T>(UnmanagedMemoryBlock<T>.FromPool(1, _scalarPool), Shape.Scalar);
            *(ret.Address) = val;

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedByteStorage<T> Clone()
        {
            return new UnmanagedByteStorage<T>(_array.Clone(), _shape);
        }

        /// <summary>
        ///     Creates a new <see cref="UnmanagedByteStorage{T}"/> that references to the internal data storage this current object does.
        /// </summary>
        /// <returns></returns>
        /// <remarks>Does not copy data.</remarks>
        public UnmanagedByteStorage<T> CreateAlias()
        {
            return new UnmanagedByteStorage<T>(_array, _shape);
        }

        /// <summary>
        ///     Creates a new <see cref="UnmanagedByteStorage{T}"/> with a specific shape that references to the internal data storage this current object does.
        /// </summary>
        /// <returns></returns>
        /// <remarks>Does not copy data.</remarks>
        public UnmanagedByteStorage<T> CreateAlias(Shape shape)
        {
            return new UnmanagedByteStorage<T>(_array, shape);
        }

        internal void Free()
        {
            try
            {
                _array.DangerousFree();
            }
            catch (Exception) { }
        }


        public override string ToString()
        {
            return ToString(flat: false);
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        public string ToString(bool flat)
        {
            var s = new StringBuilder();
            if (_shape.NDim == 0)
            {
                s.Append(this._array[0].ToString());
            }
            else
            {
                s.Append("array(");
                PrettyPrint(s, flat);
                s.Append(")");
            }

            return s.ToString();
        }

        private void PrettyPrint(StringBuilder s, bool flat = false)
        {
            if (_shape.NDim == 0)
            {
                s.Append($"{this._array[0]}");
                return;
            }

            if (_shape.NDim == 1)
            {
                s.Append("[");
                s.Append(string.Join(", ", this._enumerate().OfType<object>().Select(x => x == null ? "null" : x.ToString())));
                s.Append("]");
                return;
            }

            var size = _shape[0];
            s.Append("[");
            for (int i = 0; i < size; i++)
            {
                var n_minus_one_dim_slice = this[i]; //todo! if theres a slice - use it.
                n_minus_one_dim_slice.PrettyPrint(s, flat);
                if (i < size - 1)
                {
                    s.Append(", ");
                    if (!flat)
                        s.AppendLine();
                }
            }

            s.Append("]");
        }
    }
}
