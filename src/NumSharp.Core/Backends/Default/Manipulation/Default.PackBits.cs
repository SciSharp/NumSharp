// =====================================================================================
//  np.packbits / np.unpackbits — NumPy 2.4.2 parity (numpy/_core/src/multiarray/compiled_base.c
//  io_pack / io_unpack / pack_bits / unpack_bits). Both are pure axis transforms over a
//  dtype-agnostic byte kernel (PackBits), so the whole member here is validation +
//  geometry; the bit work lives in Backends/Kernels/PackBitsKernel.cs.
//
//  Layout strategy (matches np.repeat): read through a C-contiguous normalisation of the input
//  (a no-op when already C-contiguous) so the collapsed (n_outer, axisLen, nel) geometry is a
//  simple block walk (Shape.SplitAtAxis), then honour NumPy's F-output rule — the output is
//  F-contiguous iff the post-CheckAxis input is (PyArray_ISFORTRAN(new)) — by converting a C
//  result to F on that rare path. Reading strided/negative-stride/transposed views directly through a per-line NDIter
//  drive was measured at 0.05-0.66x NumPy, hence the normalise-and-block-walk design.
// =====================================================================================
//
//  DefaultEngine's Packbits / Unpackbits. Their layout statements are the array object's: np.ascontiguousarray's
//  NDArray.ascontiguousarray, and np.asfortranarray's F-order conversion (NDArray.asarray with order 'F'; its
//  last statement, a reshape of a 0-d result to (1,), cannot fire: a result is converted only when it has two or more
//  dimensions).
// =====================================================================================

using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // reclaims the ravel/ascontiguousarray/asfortranarray normalisation temps
        public override unsafe NDArray Packbits(NDArray a, int? axis = null, string bitorder = "big")
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy order: bitorder (io_pack) is validated BEFORE the dtype/axis checks in pack_bits.
            bool big = ParsePackOrder(bitorder);

            if (!IsPackable(a.GetTypeCode))
                throw new TypeError("Expected an input array of integer or boolean data type");

            // CheckAxis: axis=null flattens (C order) and packs the 1-D result; a 0-d input is promoted
            // to (1,) and only axis 0/-1 is valid there.
            NDArray nw = ResolveCheckAxis(a, ref axis);
            int packAxis = axis.Value;
            int ndim = nw.ndim;
            bool isFortran = ndim > 1 && nw.Shape.IsFContiguous && !nw.Shape.IsContiguous;

            long axisLen = nw.Shape.dimensions[packAxis];
            long nOut = axisLen == 0 ? 0 : ((axisLen - 1) >> 3) + 1;

            long[] outDims = (long[])nw.Shape.dimensions.Clone();
            outDims[packAxis] = nOut;

            // pack's output axis is ceil(inputAxis/8), so an empty input always yields an empty output
            // (no zero-fill needed). The main path overwrites every output byte, so fillZeros: false.
            var cOut = new NDArray(NPTypeCode.Byte, Shape.Adopt(outDims, 'C'), false);
            if (nw.size == 0 || cOut.size == 0)
                return isFortran ? NDArray.asarray(cOut, null, 'F', null) : cOut;

            // Normalise to C-contiguous so (n_outer, axisLen, nel) is a plain block walk.
            NDArray srcC = nw.Shape.IsContiguous ? nw : NDArray.ascontiguousarray(nw, null);
            srcC.Shape.SplitAtAxis(packAxis, out long nOuter, out long nel);

            // Logical element 0 = base Address + Shape.offset elements: a contiguous slice keeps a
            // non-zero offset into a shared buffer (e.g. a[2:5]), which the whole-array block walk must
            // start from. The fresh output is offset 0.
            byte* srcPtr = (byte*)srcC.Address + srcC.Shape.offset * srcC.dtypesize;
            PackBits.Pack(
                srcPtr, (byte*)cOut.Address,
                nOuter, axisLen, nel, srcC.dtypesize, big);

            return isFortran ? NDArray.asarray(cOut, null, 'F', null) : cOut;
        }

        /// <inheritdoc />
        [NDScoped] // reclaims the ravel/ascontiguousarray/asfortranarray normalisation temps
        public override unsafe NDArray Unpackbits(NDArray a, int? axis = null, long? count = null, string bitorder = "big")
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy order: bitorder (io_unpack) is validated BEFORE dtype/axis/count in unpack_bits.
            bool big = ParseUnpackOrder(bitorder);

            if (a.GetTypeCode != NPTypeCode.Byte)
                throw new TypeError("Expected an input array of unsigned byte data type");

            NDArray nw = ResolveCheckAxis(a, ref axis);
            int unpackAxis = axis.Value;
            int ndim = nw.ndim;
            bool isFortran = ndim > 1 && nw.Shape.IsFContiguous && !nw.Shape.IsContiguous;

            long axisLen = nw.Shape.dimensions[unpackAxis];   // input bytes along the axis
            long full = axisLen * 8;

            // count resolves the output axis length (NumPy: outdims[axis]*=8, then count adjusts).
            long outLen;
            if (count is null)
                outLen = full;
            else
            {
                long cnt = count.Value;
                if (cnt < 0)
                {
                    outLen = full + cnt;
                    if (outLen < 0)
                        throw new ValueError("-count larger than number of elements");
                }
                else
                    outLen = cnt;
            }

            long[] outDims = (long[])nw.Shape.dimensions.Clone();
            outDims[unpackAxis] = outLen;

            long outSize = 1;
            foreach (long d in outDims) outSize *= d;

            // Empty input but a count-forced non-empty output: NumPy's IterAllButAxis never runs its
            // zeroing loop here and leaks UNINITIALISED memory (contradicting its own "add zero padding"
            // docstring). NumSharp returns the documented deterministic zeros — the sole [Misaligned]
            // divergence, and the more correct behaviour. Truly-empty outputs need no fill.
            if (nw.size == 0 || outSize == 0)
            {
                var z = new NDArray(NPTypeCode.Byte, Shape.Adopt(outDims, 'C'), fillZeros: outSize != 0);
                return isFortran ? NDArray.asarray(z, null, 'F', null) : z;
            }

            // Main path overwrites every output byte (full bytes + partial + explicit pad), so fillZeros: false.
            var cOut = new NDArray(NPTypeCode.Byte, Shape.Adopt(outDims, 'C'), false);

            NDArray srcC = nw.Shape.IsContiguous ? nw : NDArray.ascontiguousarray(nw, null);
            srcC.Shape.SplitAtAxis(unpackAxis, out long nOuter, out long nel);

            // Logical element 0 = base Address + Shape.offset (uint8, so offset is in bytes). The fresh
            // output is offset 0.
            byte* srcPtr = (byte*)srcC.Address + srcC.Shape.offset;
            PackBits.Unpack(
                srcPtr, (byte*)cOut.Address,
                nOuter, axisLen, nel, outLen, big);

            return isFortran ? NDArray.asarray(cOut, null, 'F', null) : cOut;
        }

        /// <summary>
        ///     NumPy <c>PyArray_CheckAxis</c> for the pack/unpack family: <c>axis=null</c> ravels the input
        ///     to a C-order 1-D array (and sets the effective axis to 0); a 0-d input is promoted to shape
        ///     <c>(1,)</c> accepting only axis 0/-1; otherwise the axis is normalised (negatives from the end)
        ///     and range-checked.
        /// </summary>
        /// <param name="a">The input array.</param>
        /// <param name="axis">On entry the requested axis (null ⇒ flatten); on return the resolved non-negative axis into the returned array.</param>
        /// <returns>The array to operate on (a C-order 1-D ravel when <paramref name="axis"/> was null, a <c>(1,)</c> promotion for 0-d, else <paramref name="a"/> itself).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        private static NDArray ResolveCheckAxis(NDArray a, ref int? axis)
        {
            if (axis is null)
            {
                // Flatten in C (logical) order regardless of the input's memory layout.
                axis = 0;
                return a.ravel();
            }

            int requested = axis.Value;
            if (a.ndim == 0)
            {
                // A 0-d array has one implicit axis after promotion (PyArray_CheckAxis ravels it first), so NumPy accepts
                // axis 0 or -1 and reports any other against that rank of 1.
                if (requested != 0 && requested != -1)
                    throw new AxisError(requested, 1);
                axis = 0;
                return a.reshape(1);
            }

            int norm = requested < 0 ? requested + a.ndim : requested;
            if (norm < 0 || norm >= a.ndim)
                throw new AxisError(requested, a.ndim);
            axis = norm;
            return a;
        }

        /// <summary>Whether a dtype may be bit-packed — boolean or any integer width (NumPy's <c>ISBOOL || ISINTEGER</c>, plus NumSharp's Char which has no NumPy analog).</summary>
        /// <param name="tc">The input element type code.</param>
        /// <returns>True for boolean/integer/char, false for float/half/decimal/complex.</returns>
        private static bool IsPackable(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Parses <c>packbits</c>'s <c>bitorder</c> (NumPy <c>io_pack</c>): a value beginning with
        ///     <c>"little"</c> ⇒ little, beginning with <c>"big"</c> ⇒ big, <c>null</c> ⇒ big.
        /// </summary>
        /// <param name="bitorder">The requested bit order.</param>
        /// <returns>True for 'big', false for 'little'.</returns>
        /// <exception cref="ValueError">The value is neither a "little" nor "big" prefix.</exception>
        private static bool ParsePackOrder(string bitorder)
        {
            if (bitorder is null)
                return true;
            if (bitorder.StartsWith("little", StringComparison.Ordinal))
                return false;
            if (bitorder.StartsWith("big", StringComparison.Ordinal))
                return true;
            throw new ValueError("'order' must be either 'little' or 'big'");
        }

        /// <summary>
        ///     Parses <c>unpackbits</c>'s <c>bitorder</c> (NumPy <c>io_unpack</c>): only the FIRST character
        ///     is inspected — <c>'l'</c> ⇒ little, <c>'b'</c> ⇒ big, <c>null</c> ⇒ big.
        /// </summary>
        /// <param name="bitorder">The requested bit order.</param>
        /// <returns>True for 'big', false for 'little'.</returns>
        /// <exception cref="ValueError">The value is empty or its first character is not 'l' or 'b'.</exception>
        private static bool ParseUnpackOrder(string bitorder)
        {
            if (bitorder is null)
                return true;
            if (bitorder.Length > 0)
            {
                if (bitorder[0] == 'l') return false;
                if (bitorder[0] == 'b') return true;
            }
            throw new ValueError("'order' must begin with 'l' or 'b'");
        }
    }
}
