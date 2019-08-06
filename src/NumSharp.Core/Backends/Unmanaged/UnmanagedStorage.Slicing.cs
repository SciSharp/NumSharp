using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Slicing

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedStorage GetView(string slicing_notation) => GetView(Slice.ParseSlices(slicing_notation));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [SuppressMessage("ReSharper", "PossibleInvalidOperationException")]
        public UnmanagedStorage GetView(params Slice[] slices)
        {
            if (slices == null)
                throw new ArgumentNullException(nameof(slices));

            //handle memory slice if possible
            if (!_shape.IsSliced)
            {
                var indices = new int[slices.Length];
                for (var i = 0; i < slices.Length; i++)
                {
                    var inputSlice = slices[i];
                    if (!inputSlice.IsIndex)
                    {
                        //incase it is a trailing :, e.g. [2,2, :] in a shape (3,3,5,5) -> (5,5)
                        if (i == slices.Length - 1 && inputSlice == Slice.All)
                        {
                            Array.Resize(ref indices, indices.Length - 1);
                            goto _getdata;
                        }

                        goto _perform_slice;
                    }

                    indices[i] = inputSlice.Start.Value;
                }

                _getdata:
                return GetData(indices);
            }

            //perform a regular slicing
            _perform_slice:

            // In case the slices selected are all ":"
            // ReSharper disable once ConvertIfStatementToReturnStatement
            if (!_shape.IsRecursive && slices.All(s => Equals(Slice.All, s)))
                return Alias();

            return Alias(_shape.Slice(slices));
        }

        #endregion
    }
}
