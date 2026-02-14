using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp
{
    public partial struct Shape
    {
        /// <summary>
        ///     Get offset index out of coordinate indices (pointer version).
        ///     NumPy-aligned: offset + sum(indices * strides)
        /// </summary>
        /// <param name="indices">A pointer to the coordinates to turn into linear offset</param>
        /// <param name="ndims">The number of dimensions</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public readonly unsafe int GetOffset(int* indices, int ndims)
        {
            // Scalar case
            if (dimensions.Length == 0)
                return offset + (ndims > 0 ? indices[0] : 0);

            // NumPy formula: offset + sum(indices * strides)
            int off = offset;
            unchecked
            {
                for (int i = 0; i < ndims; i++)
                    off += strides[i] * indices[i];
            }
            return off;
        }

        /// <summary>
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contiguous) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, returned shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public readonly unsafe (Shape Shape, int Offset) GetSubshape(int* dims, int ndims)
        {
            if (ndims == 0)
                return (this, 0);

            int offset;
            var dim = ndims;
            var newNDim = dimensions.Length - dim;
            if (IsBroadcasted)
            {
                var dimsClone = stackalloc int[ndims];
                for (int j = 0; j < ndims; j++)
                    dimsClone[j] = dims[j];

                // NumPy-aligned: compute unreduced dims on the fly (stride=0 means broadcast)
                // Unbroadcast indices (wrap around for broadcast dimensions)
                for (int i = 0; i < dim; i++)
                {
                    int unreducedDim = strides[i] == 0 ? 1 : dimensions[i];
                    dimsClone[i] = dimsClone[i] % unreducedDim;
                }

                // Compute offset using strides
                offset = this.offset;
                for (int i = 0; i < dim; i++)
                    offset += strides[i] * dimsClone[i];

                var retShape = new int[newNDim];
                var retStrides = new int[newNDim];
                for (int i = 0; i < newNDim; i++)
                {
                    retShape[i] = this.dimensions[dim + i];
                    retStrides[i] = this.strides[dim + i];
                }

                // Create result with bufferSize preserved (immutable constructor)
                int bufSize = this.bufferSize > 0 ? this.bufferSize : this.size;
                var result = new Shape(retShape, retStrides, offset, bufSize);
                return (result, offset);
            }

            //compute offset
            offset = GetOffset(dims, ndims);

            // Use bufferSize for bounds checking (NumPy-aligned: no ViewInfo dependency)
            int boundSize = bufferSize > 0 ? bufferSize : size;
            if (offset >= boundSize)
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {boundSize}");

            if (ndims == dimensions.Length)
                return (Scalar, offset);

            //compute subshape
            var innerShape = new int[newNDim];
            for (int i = 0; i < innerShape.Length; i++)
                innerShape[i] = this.dimensions[dim + i];

            //TODO! This is not full support of sliced,
            //TODO! when sliced it usually diverts from this function but it would be better if we add support for sliced arrays too.
            return (new Shape(innerShape), offset);
        }


        /// <summary>
        ///     Translates coordinates with negative indices, e.g:<br></br>
        ///     np.arange(9)[-1] == np.arange(9)[8]<br></br>
        ///     np.arange(9)[-2] == np.arange(9)[7]<br></br>
        /// </summary>
        /// <param name="dimensions">The dimensions these coordinates are targeting</param>
        /// <param name="coords">The coordinates.</param>
        /// <returns>Coordinates without negative indices.</returns>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl((MethodImplOptions)512)]
        public static unsafe void InferNegativeCoordinates(int[] dimensions, int* coords, int coordsCount)
        {
            for (int i = 0; i < coordsCount; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = dimensions[i] + curr;
            }
        }

        /// <summary>
    }
}
