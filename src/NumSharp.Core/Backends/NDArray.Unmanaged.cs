using System;
using System.Globalization;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using CompilerUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Provides an interface for unsafe methods in NDArray.
        /// </summary>
        public unsafe _Unsafe Unsafe => new _Unsafe(this);

        public unsafe struct _Unsafe
        {
            private readonly NDArray _this;

            internal _Unsafe(NDArray @this)
            {
                _this = @this;
            }

            /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
            public ref byte GetPinnableReference()
            {
                if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted)
                    throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");

                unsafe
                {
                    return ref CompilerUnsafe.AsRef<byte>(Address);
                }
            }

            /// <summary>
            ///     Returns the memory address to the start of this block of memory.
            /// </summary>
            /// <returns></returns>
            /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
            public void* Address
            {
                get
                {
                    if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted)
                        throw new InvalidOperationException("Can't return a memory address when NDArray is sliced or broadcasted.");

                    return _this.Address;
                }
            }

            /// <summary>
            ///     Get: Gets internal storage array by calling <see cref="IStorage.GetData"/><br></br>
            ///     Set: Replace internal storage by calling <see cref="IStorage.ReplaceData(System.Array)"/>
            /// </summary>
            /// <remarks>Setting does not replace internal storage array.</remarks>
            internal IArraySlice Array
            {
                get
                {
                    if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted)
                        throw new InvalidOperationException("Can't access a memory address when NDArray is sliced or broadcasted.");

                    return _this.Storage.InternalArray;
                }
            }

            /// <summary>
            ///     Provides access to the internal <see cref="UnmanagedStorage"/>.
            /// </summary>
            public UnmanagedStorage Storage => _this.Storage;

            /// <summary>
            ///     Provides access to the internal <see cref="Shape"/>.
            /// </summary>
            [Obsolete("Please use nd.Shape directly instead of nd.Unsafe.Shape, will be removed in 0.21.0")]
            public Shape Shape => _this.Storage.Shape;

            /// A Span representing this slice.
            /// <remarks>Does not perform copy.</remarks>
            public Span<T> AsSpan<T>()
            {
                return Array.AsSpan<T>();
            }

            /// <summary>
            ///     The size of a single item stored in <see cref="Address"/>.
            /// </summary>
            /// <remarks>Equivalent to <see cref="NPTypeCode.SizeOf"/> extension.</remarks>
            public int ItemLength => Array.ItemLength;

            /// <summary>
            ///     How many bytes are stored in this memory block.
            /// </summary>
            /// <remarks>Calculated by <see cref="Count"/>*<see cref="ItemLength"/></remarks>
            public int BytesLength => ((IConvertible)Array.BytesLength).ToInt32(CultureInfo.InvariantCulture);

            /// <summary>
            ///     How many items are stored in <see cref="Address"/>.
            /// </summary>
            /// <remarks>Not to confuse with <see cref="BytesLength"/></remarks>
            public int Count => (int) Array.Count;

            /// <summary>
            ///     Fills all indexes with <paramref name="value"/>.
            /// </summary>
            /// <param name="value"></param>
            public void Fill(object value)
            {
                Array.Fill(value);
            }

            public T GetIndex<T>(int index) where T : unmanaged
            {
                return Array.GetIndex<T>(index);
            }

            public object GetIndex(int index)
            {
                return Array.GetIndex(index);
            }

            /// <summary>
            ///     Gets pinnable reference of the first item in the memory block storage.
            /// </summary>
            public ref T GetPinnableReference<T>() where T : unmanaged
            {
                return ref Array.GetPinnableReference<T>();
            }

            #region Pinning

            /// <summary>
            ///     Provides the ability to return a pin to the memory address of NDArray.
            /// </summary>
            /// <remarks>Possible only when the ndarray is not sliced.</remarks>
            public unsafe _Pinning Pin => new _Pinning(_this);

            public struct _Pinning
            {
                private readonly NDArray _this;
                internal _Pinning(NDArray @this) => _this = @this;

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
                public unsafe ref T GetPin<T>()
                {
                    if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                    return ref CompilerUnsafe.AsRef<T>(_this.Address);
                }

#if _REGEN1
                #region Compute
		        %foreach supported_dtypes,supported_dtypes_lowercase%
                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref #2 #1
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<#2>(_this.Address);
                    }
                } 

		        %
                #endregion
#else

                #region Compute
                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref bool Boolean
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<bool>(_this.Address);
                    }
                } 

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref byte Byte
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<byte>(_this.Address);
                    }
                } 

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref int Int32
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<int>(_this.Address);
                    }
                } 

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref long Int64
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<long>(_this.Address);
                    }
                } 

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref float Single
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<float>(_this.Address);
                    }
                } 

                /// <exception cref="InvalidOperationException">When this NDArray is a slice.</exception>
		        public unsafe ref double Double
                {
                    get
                    {
                        if (_this.Shape.IsSliced || _this.Shape.IsBroadcasted) throw new InvalidOperationException("Can't pin reference when NDArray is sliced or broadcasted.");
                        return ref CompilerUnsafe.AsRef<double>(_this.Address);
                    }
                } 

                #endregion
#endif
            }

            #endregion
        }
    }
}
