using System;

namespace NumSharp.Backends.Unmanaged
{
#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        public class Unmanaged#1Storage : UnmanagedByteStorage<#2> {
            /// <inheritdoc />
            public Unmanaged#1Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(Shape shape) : base(shape) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(Shape shape, #2 fill) : base(shape, fill) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(#2 scalar) : base(scalar) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(#2[] arr, Shape shape) : base(arr, shape) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(ArraySlice<#2> arr, Shape shape) : base(arr, shape) { }
            /// <inheritdoc />
            public Unmanaged#1Storage(UnmanagedArray<#2> array, Shape shape) : base(array, shape) { }
            /// <inheritdoc />
            public unsafe Unmanaged#1Storage(#2* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
        }

        %
#else
    public class UnmanagedByteStorage : UnmanagedByteStorage<byte>
    {
        /// <inheritdoc />
        public UnmanagedByteStorage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(Shape shape, byte fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(byte scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(byte[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(ArraySlice<byte> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedByteStorage(UnmanagedArray<byte> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedByteStorage(byte* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedInt16Storage : UnmanagedByteStorage<short>
    {
        /// <inheritdoc />
        public UnmanagedInt16Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(Shape shape, short fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(short scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(short[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(ArraySlice<short> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt16Storage(UnmanagedArray<short> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedInt16Storage(short* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedUInt16Storage : UnmanagedByteStorage<ushort>
    {
        /// <inheritdoc />
        public UnmanagedUInt16Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(Shape shape, ushort fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(ushort scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(ushort[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(ArraySlice<ushort> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt16Storage(UnmanagedArray<ushort> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedUInt16Storage(ushort* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedInt32Storage : UnmanagedByteStorage<int>
    {
        /// <inheritdoc />
        public UnmanagedInt32Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(Shape shape, int fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(int scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(int[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(ArraySlice<int> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt32Storage(UnmanagedArray<int> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedInt32Storage(int* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedUInt32Storage : UnmanagedByteStorage<uint>
    {
        /// <inheritdoc />
        public UnmanagedUInt32Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(Shape shape, uint fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(uint scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(uint[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(ArraySlice<uint> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt32Storage(UnmanagedArray<uint> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedUInt32Storage(uint* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedInt64Storage : UnmanagedByteStorage<long>
    {
        /// <inheritdoc />
        public UnmanagedInt64Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(Shape shape, long fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(long scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(long[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(ArraySlice<long> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedInt64Storage(UnmanagedArray<long> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedInt64Storage(long* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedUInt64Storage : UnmanagedByteStorage<ulong>
    {
        /// <inheritdoc />
        public UnmanagedUInt64Storage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(Shape shape, ulong fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(ulong scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(ulong[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(ArraySlice<ulong> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedUInt64Storage(UnmanagedArray<ulong> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedUInt64Storage(ulong* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedCharStorage : UnmanagedByteStorage<char>
    {
        /// <inheritdoc />
        public UnmanagedCharStorage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(Shape shape, char fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(char scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(char[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(ArraySlice<char> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedCharStorage(UnmanagedArray<char> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedCharStorage(char* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedDoubleStorage : UnmanagedByteStorage<double>
    {
        /// <inheritdoc />
        public UnmanagedDoubleStorage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(Shape shape, double fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(double scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(double[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(ArraySlice<double> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedDoubleStorage(UnmanagedArray<double> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedDoubleStorage(double* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedSingleStorage : UnmanagedByteStorage<float>
    {
        /// <inheritdoc />
        public UnmanagedSingleStorage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(Shape shape, float fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(float scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(float[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(ArraySlice<float> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedSingleStorage(UnmanagedArray<float> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedSingleStorage(float* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }

    public class UnmanagedDecimalStorage : UnmanagedByteStorage<decimal>
    {
        /// <inheritdoc />
        public UnmanagedDecimalStorage(Shape shape, bool assignZeros) : base(shape, assignZeros) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(Shape shape) : base(shape) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(Shape shape, decimal fill) : base(shape, fill) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(decimal scalar) : base(scalar) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(decimal[] arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(ArraySlice<decimal> arr, Shape shape) : base(arr, shape) { }

        /// <inheritdoc />
        public UnmanagedDecimalStorage(UnmanagedArray<decimal> array, Shape shape) : base(array, shape) { }

        /// <inheritdoc />
        public unsafe UnmanagedDecimalStorage(decimal* ptr, int lengthInSizeOfT, Shape shape, Action dispose) : base(ptr, lengthInSizeOfT, shape, dispose) { }
    }
#endif
}
