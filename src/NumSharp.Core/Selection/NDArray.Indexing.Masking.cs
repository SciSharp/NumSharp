using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.17.0/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        public NDArray this[NDArray<bool> mask]
        {
            get => FetchIndices(this, np.nonzero(mask), null, true);
            set
            {
                if(mask.ndim == 1)
                {
                    for (int i = 0; i < mask.size; i++)
                    {
                        if (mask.GetBoolean(i))
                            this[i] = value;
                    }
                }
                else
                {
                    throw new NotImplementedException("Setter is not implemented yet");
                }
            }
        }
    }
}
