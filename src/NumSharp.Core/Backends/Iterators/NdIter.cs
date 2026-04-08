using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     Flags for controlling NdIter behavior.
    /// </summary>
    [Flags]
    public enum NdIterFlags
    {
        /// <summary>No special flags.</summary>
        None = 0,

        /// <summary>Track multi-dimensional index during iteration.</summary>
        MultiIndex = 1 << 0,

        /// <summary>Allow read-write access to array elements.</summary>
        ReadWrite = 1 << 1,

        /// <summary>Read-only access (default).</summary>
        ReadOnly = 1 << 2,

        /// <summary>
        ///     Causes iteration to emit contiguous chunks when possible.
        ///     The returned value becomes an array instead of a scalar.
        /// </summary>
        ExternalLoop = 1 << 3,
    }

    /// <summary>
    ///     NumPy-compatible multi-dimensional iterator.
    /// </summary>
    /// <remarks>
    ///     https://numpy.org/doc/stable/reference/generated/numpy.nditer.html
    ///
    ///     Provides efficient iteration over arrays with support for:
    ///     - Multi-index tracking (current N-D coordinates)
    ///     - Read-write access to elements
    ///     - Broadcasting multiple arrays together
    ///     - C-order (row-major) traversal
    /// </remarks>
    /// <example>
    ///     // Basic iteration
    ///     var it = np.nditer(arr);
    ///     while (!it.finished)
    ///     {
    ///         Console.WriteLine(it.value);
    ///         it.iternext();
    ///     }
    ///
    ///     // With multi-index tracking
    ///     var it = np.nditer(arr, NdIterFlags.MultiIndex);
    ///     while (!it.finished)
    ///     {
    ///         Console.WriteLine($"arr{it.multi_index} = {it.value}");
    ///         it.iternext();
    ///     }
    ///
    ///     // Read-write modification
    ///     var it = np.nditer(arr, NdIterFlags.ReadWrite);
    ///     while (!it.finished)
    ///     {
    ///         it.value = it.value * 2;  // Double each element
    ///         it.iternext();
    ///     }
    /// </example>
    public sealed class NdIter : IEnumerable<object>, IDisposable
    {
        private readonly NDArray _array;
        private readonly NdIterFlags _flags;
        private readonly Shape _shape;
        private long _index;
        private bool _finished;
        private readonly long _size;
        private long[]? _multiIndex;
        private ValueCoordinatesIncrementor _coordIter;
        private readonly bool _trackMultiIndex;

        /// <summary>
        ///     Creates a new NdIter for the given array.
        /// </summary>
        /// <param name="array">The array to iterate over.</param>
        /// <param name="flags">Iteration flags.</param>
        public NdIter(NDArray array, NdIterFlags flags = NdIterFlags.None)
        {
            _array = array ?? throw new ArgumentNullException(nameof(array));
            _flags = flags;
            _shape = array.Shape;
            _size = array.size;
            _index = -1; // Will be incremented to 0 on first iternext() or property access
            _finished = _size == 0;

            _trackMultiIndex = (flags & NdIterFlags.MultiIndex) != 0;
            if (_trackMultiIndex)
            {
                _coordIter = new ValueCoordinatesIncrementor(_shape.dimensions);
                _multiIndex = new long[_shape.NDim];
            }

            if ((flags & NdIterFlags.ReadWrite) != 0)
            {
                NumSharpException.ThrowIfNotWriteable(_shape);
            }
        }

        /// <summary>
        ///     Whether iteration is finished.
        /// </summary>
        public bool finished => _finished;

        /// <summary>
        ///     Current flat index into the array.
        /// </summary>
        public long index
        {
            get
            {
                EnsureStarted();
                return _index;
            }
        }

        /// <summary>
        ///     Current multi-dimensional index (requires MultiIndex flag).
        /// </summary>
        /// <exception cref="InvalidOperationException">If MultiIndex flag was not set.</exception>
        public long[] multi_index
        {
            get
            {
                if (!_trackMultiIndex)
                    throw new InvalidOperationException("multi_index requires NdIterFlags.MultiIndex to be set.");
                EnsureStarted();
                return _multiIndex!;
            }
        }

        /// <summary>
        ///     Current value at the iterator position.
        /// </summary>
        /// <exception cref="InvalidOperationException">If iteration hasn't started or is finished.</exception>
        public object value
        {
            get
            {
                EnsureStarted();
                if (_finished)
                    throw new InvalidOperationException("Iterator is finished.");
                return _array.item(_index);
            }
            set
            {
                if ((_flags & NdIterFlags.ReadWrite) == 0)
                    throw new InvalidOperationException("Cannot set value without NdIterFlags.ReadWrite flag.");
                EnsureStarted();
                if (_finished)
                    throw new InvalidOperationException("Iterator is finished.");
                _array.SetAtIndex(value, _index);
            }
        }

        /// <summary>
        ///     Get current value as typed.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <returns>Current value cast to T.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T value_as<T>() where T : unmanaged
        {
            EnsureStarted();
            if (_finished)
                throw new InvalidOperationException("Iterator is finished.");
            return _array.GetAtIndex<T>(_index);
        }

        /// <summary>
        ///     Set current value with typed value.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="newValue">The value to set.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void set_value<T>(T newValue) where T : unmanaged
        {
            if ((_flags & NdIterFlags.ReadWrite) == 0)
                throw new InvalidOperationException("Cannot set value without NdIterFlags.ReadWrite flag.");
            EnsureStarted();
            if (_finished)
                throw new InvalidOperationException("Iterator is finished.");
            _array.SetAtIndex<T>(newValue, _index);
        }

        /// <summary>
        ///     Number of elements to iterate.
        /// </summary>
        public long size => _size;

        /// <summary>
        ///     Shape of the iterated array.
        /// </summary>
        public long[] shape => _shape.dimensions;

        /// <summary>
        ///     Number of dimensions.
        /// </summary>
        public int ndim => _shape.NDim;

        /// <summary>
        ///     Move to the next element.
        /// </summary>
        /// <returns>True if there are more elements, false if finished.</returns>
        public bool iternext()
        {
            if (_finished)
                return false;

            _index++;

            if (_index >= _size)
            {
                _finished = true;
                return false;
            }

            // Update multi-index if tracking
            if (_trackMultiIndex && _index > 0)
            {
                _coordIter.Next();
                Array.Copy(_coordIter.Index, _multiIndex!, _shape.NDim);
            }
            else if (_trackMultiIndex && _index == 0)
            {
                // First element - copy initial coordinates
                Array.Copy(_coordIter.Index, _multiIndex!, _shape.NDim);
            }

            return true;
        }

        /// <summary>
        ///     Reset the iterator to the beginning.
        /// </summary>
        public void reset()
        {
            _index = -1;
            _finished = _size == 0;
            if (_trackMultiIndex)
                _coordIter.Reset();
        }

        /// <summary>
        ///     Ensure iteration has started (auto-start on first access).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void EnsureStarted()
        {
            if (_index < 0 && !_finished)
                iternext();
        }

        /// <summary>
        ///     Indexer for accessing current operand values.
        ///     For single-array iteration, [0] returns the current value.
        /// </summary>
        public object this[int operandIndex]
        {
            get
            {
                if (operandIndex != 0)
                    throw new IndexOutOfRangeException("Single-array NdIter only supports operand index 0.");
                return value;
            }
            set
            {
                if (operandIndex != 0)
                    throw new IndexOutOfRangeException("Single-array NdIter only supports operand index 0.");
                this.value = value;
            }
        }

        #region IEnumerable

        /// <summary>
        ///     Enumerate all values in the array.
        /// </summary>
        public IEnumerator<object> GetEnumerator()
        {
            reset();
            while (iternext())
            {
                yield return value;
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        #endregion

        #region IDisposable

        /// <summary>
        ///     Dispose the iterator (no-op, provided for `using` pattern compatibility).
        /// </summary>
        public void Dispose()
        {
            // No unmanaged resources - just for API compatibility with NumPy's context manager
        }

        #endregion
    }

    /// <summary>
    ///     NumPy-compatible multi-dimensional iterator (typed version).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    public sealed class NdIter<T> : IEnumerable<T>, IDisposable where T : unmanaged
    {
        private readonly NDArray _array;
        private readonly NdIterFlags _flags;
        private readonly Shape _shape;
        private long _index;
        private bool _finished;
        private readonly long _size;
        private long[]? _multiIndex;
        private ValueCoordinatesIncrementor _coordIter;
        private readonly bool _trackMultiIndex;

        /// <summary>
        ///     Creates a new typed NdIter for the given array.
        /// </summary>
        public NdIter(NDArray array, NdIterFlags flags = NdIterFlags.None)
        {
            _array = array ?? throw new ArgumentNullException(nameof(array));
            _flags = flags;
            _shape = array.Shape;
            _size = array.size;
            _index = -1;
            _finished = _size == 0;

            _trackMultiIndex = (flags & NdIterFlags.MultiIndex) != 0;
            if (_trackMultiIndex)
            {
                _coordIter = new ValueCoordinatesIncrementor(_shape.dimensions);
                _multiIndex = new long[_shape.NDim];
            }

            if ((flags & NdIterFlags.ReadWrite) != 0)
            {
                NumSharpException.ThrowIfNotWriteable(_shape);
            }
        }

        /// <summary>Whether iteration is finished.</summary>
        public bool finished => _finished;

        /// <summary>Current flat index.</summary>
        public long index
        {
            get
            {
                EnsureStarted();
                return _index;
            }
        }

        /// <summary>Current multi-dimensional index.</summary>
        public long[] multi_index
        {
            get
            {
                if (!_trackMultiIndex)
                    throw new InvalidOperationException("multi_index requires NdIterFlags.MultiIndex to be set.");
                EnsureStarted();
                return _multiIndex!;
            }
        }

        /// <summary>Current typed value.</summary>
        public T value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                EnsureStarted();
                if (_finished)
                    throw new InvalidOperationException("Iterator is finished.");
                return _array.GetAtIndex<T>(_index);
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if ((_flags & NdIterFlags.ReadWrite) == 0)
                    throw new InvalidOperationException("Cannot set value without NdIterFlags.ReadWrite flag.");
                EnsureStarted();
                if (_finished)
                    throw new InvalidOperationException("Iterator is finished.");
                _array.SetAtIndex<T>(value, _index);
            }
        }

        /// <summary>Number of elements.</summary>
        public long size => _size;

        /// <summary>Shape of array.</summary>
        public long[] shape => _shape.dimensions;

        /// <summary>Number of dimensions.</summary>
        public int ndim => _shape.NDim;

        /// <summary>Move to next element.</summary>
        public bool iternext()
        {
            if (_finished)
                return false;

            _index++;

            if (_index >= _size)
            {
                _finished = true;
                return false;
            }

            // Update multi-index if tracking
            if (_trackMultiIndex && _index > 0)
            {
                _coordIter.Next();
                Array.Copy(_coordIter.Index, _multiIndex!, _shape.NDim);
            }
            else if (_trackMultiIndex && _index == 0)
            {
                // First element - copy initial coordinates
                Array.Copy(_coordIter.Index, _multiIndex!, _shape.NDim);
            }

            return true;
        }

        /// <summary>Reset iterator.</summary>
        public void reset()
        {
            _index = -1;
            _finished = _size == 0;
            if (_trackMultiIndex)
                _coordIter.Reset();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void EnsureStarted()
        {
            if (_index < 0 && !_finished)
                iternext();
        }

        /// <summary>Indexer for operand access.</summary>
        public T this[int operandIndex]
        {
            get
            {
                if (operandIndex != 0)
                    throw new IndexOutOfRangeException("Single-array NdIter only supports operand index 0.");
                return value;
            }
            set
            {
                if (operandIndex != 0)
                    throw new IndexOutOfRangeException("Single-array NdIter only supports operand index 0.");
                this.value = value;
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            reset();
            while (iternext())
            {
                yield return value;
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Dispose() { }
    }
}
