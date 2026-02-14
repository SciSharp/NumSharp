using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
    public partial class NDArray
    {
        /// <summary>
        ///     Throws if this NDArray is not writeable (e.g., broadcast arrays).
        ///     NumPy raises: ValueError: assignment destination is read-only
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ThrowIfNotWriteable()
        {
            NumSharpException.ThrowIfNotWriteable(Shape);
        }

        /// <summary>
        ///     Used to perform selection based on given indices.
        /// </summary>
        /// <param name="dims">The pointer to the dimensions</param>
        /// <param name="ndims">The count of ints in <paramref name="dims"/></param>
        public unsafe NDArray this[int* dims, int ndims]
        {
            get => new NDArray(Storage.GetData(dims, ndims));
            set { ThrowIfNotWriteable(); Storage.GetData(dims, ndims).SetData(value); }
        }

        /// <summary>
        ///     Used to perform selection based on a selection indices.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray this[params NDArray<int>[] selection]
        {
            get => FetchIndices(this, selection.Select(array => (NDArray)array).ToArray(), null, true);
            set
            {
                ThrowIfNotWriteable();
                SetIndices(this, selection, value);
            }
        }

        /// <summary>
        ///     Slice the array with Python slice notation like this: ":, 2:7:1, ..., np.newaxis"
        /// </summary>
        /// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        /// <returns>A sliced view</returns>
        public NDArray this[string slice]
        {
            get => new NDArray(Storage.GetView(Slice.ParseSlices(slice)));
            set { ThrowIfNotWriteable(); Storage.GetView(Slice.ParseSlices(slice)).SetData(value); }
        }


        /// <summary>
        ///     Slice the array with Python slice notation like this: ":, 2:7:1, ..., np.newaxis"
        /// </summary>
        /// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        /// <returns>A sliced view</returns>
        public NDArray this[params Slice[] slice]
        {
            get => new NDArray(Storage.GetView(slice));
            set { ThrowIfNotWriteable(); Storage.GetView(slice).SetData(value); }
        }

        ///// <summary>
        /////     todo: doc
        ///// </summary>
        ///// <param name="slice">A string containing slice notations for every dimension, delimited by comma</param>
        ///// <returns>A sliced view</returns>
        //public NDArray this[params IIndex[] slice] //TODO IIndex is NDArray and 
        //{
        //    get => new NDArray(Storage.GetView(slice)); 
        //    set => Storage.GetView(slice).SetData(value);
        //}


        /// <summary>
        /// Perform slicing, index extraction, masking and indexing all at the same time with mixed index objects
        /// </summary>
        /// <param name="indicesObjects"></param>
        /// <returns></returns>
        public NDArray this[params object[] indicesObjects]
        {
            get
            {
                return this.FetchIndices(indicesObjects);
            }
            set
            {
                ThrowIfNotWriteable();
                SetIndices(indicesObjects, value);
            }
        }
    }
}
