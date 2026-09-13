using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================================
        //  np.packbits / np.unpackbits — NumPy 2.4.2 parity (numpy/_core/src/multiarray/compiled_base.c
        //  io_pack / io_unpack / pack_bits / unpack_bits). Both are pure axis transforms over a
        //  dtype-agnostic byte kernel (PackBits), so the whole np.* surface here is validation +
        //  geometry; the bit work lives in Backends/Kernels/PackBitsKernel.cs.
        //
        //  Layout strategy (matches np.repeat): read through a C-contiguous normalisation of the input
        //  (a no-op when already C-contiguous) so the collapsed (n_outer, axisLen, nel) geometry is a
        //  simple block walk, then honour NumPy's F-output rule — the output is F-contiguous iff the
        //  post-CheckAxis input is (PyArray_ISFORTRAN(new)) — by converting a C result to F on that rare
        //  path. Reading strided/negative-stride/transposed views directly through a per-line NDIter
        //  drive was measured at 0.05-0.66x NumPy, hence the normalise-and-block-walk design.
        // =====================================================================================

        /// <summary>
        ///     Packs the elements of a binary-valued array into bits in a uint8 array. Each element
        ///     contributes one bit (1 iff the element is nonzero); the axis is shortened to
        ///     <c>ceil(len/8)</c> and zero-padded to a full trailing byte.
        /// </summary>
        /// <param name="a">Input array. Must be a boolean or integer dtype (float/complex/decimal raise); read in logical order for any memory layout.</param>
        /// <param name="axis">Axis to pack along. <c>null</c> (NumPy <c>None</c>) packs the C-order flattened array and returns a 1-D result.</param>
        /// <param name="bitorder">
        ///     <c>"big"</c> (default) puts the first element in the most-significant bit
        ///     (<c>[.,.,.,.,.,.,1,1] =&gt; 3</c>); <c>"little"</c> reverses it. Matched to NumPy's prefix rule:
        ///     a value beginning with <c>"little"</c> is little, one beginning with <c>"big"</c> is big;
        ///     <c>null</c> is treated as <c>"big"</c>.
        /// </param>
        /// <returns>A uint8 array with the packed bits — same ndim as <paramref name="a"/> (except <c>axis=null</c> ⇒ 1-D), F-contiguous iff <paramref name="a"/> is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> is neither a "big" nor "little" prefix.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not a boolean/integer dtype.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.packbits.html</remarks>
        [NDScoped] // reclaims the ravel/ascontiguousarray/asfortranarray normalisation temps
        public static unsafe NDArray packbits(NDArray a, int? axis = null, string bitorder = "big")
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

            long axisLen = nw.shape[packAxis];
            long nOut = axisLen == 0 ? 0 : ((axisLen - 1) >> 3) + 1;

            long[] outDims = (long[])nw.shape.Clone();
            outDims[packAxis] = nOut;

            // pack's output axis is ceil(inputAxis/8), so an empty input always yields an empty output
            // (no zero-fill needed). The main path overwrites every output byte, so fillZeros: false.
            var cOut = new NDArray(NPTypeCode.Byte, new Shape(outDims, 'C'), false);
            if (nw.size == 0 || cOut.size == 0)
                return isFortran ? np.asfortranarray(cOut) : cOut;

            // Normalise to C-contiguous so (n_outer, axisLen, nel) is a plain block walk.
            NDArray srcC = nw.Shape.IsContiguous ? nw : np.ascontiguousarray(nw);
            ComputeAxisGeometry(srcC.shape, packAxis, out long nOuter, out long nel);

            // Logical element 0 = base Address + Shape.offset elements: a contiguous slice keeps a
            // non-zero offset into a shared buffer (e.g. a[2:5]), which the whole-array block walk must
            // start from. The fresh output is offset 0.
            byte* srcPtr = (byte*)srcC.Address + srcC.Shape.offset * srcC.dtypesize;
            PackBits.Pack(
                srcPtr, (byte*)cOut.Address,
                nOuter, axisLen, nel, srcC.dtypesize, big);

            return isFortran ? np.asfortranarray(cOut) : cOut;
        }

        /// <summary>
        ///     Unpacks each uint8 element of an array into 8 binary-valued (0/1) uint8 elements along an
        ///     axis — the inverse of <see cref="packbits(NDArray,int?,string)"/>.
        /// </summary>
        /// <param name="a">Input array. Must be uint8 (any other dtype raises); read in logical order for any memory layout.</param>
        /// <param name="axis">Axis to unpack along. <c>null</c> (NumPy <c>None</c>) unpacks the C-order flattened array and returns a 1-D result.</param>
        /// <param name="count">
        ///     Number of output elements along <paramref name="axis"/> — undoes packing a non-multiple-of-8
        ///     size. <c>null</c> unpacks everything (<c>8·len</c>); a non-negative value takes exactly that
        ///     many bits (truncating, or zero-padding past the available bits); a negative value trims that
        ///     many bits off the end (and must not exceed the available bits).
        /// </param>
        /// <param name="bitorder">
        ///     <c>"big"</c> (default) or <c>"little"</c>. Matched to NumPy's unpack rule: only the FIRST
        ///     character is inspected (<c>'l'</c> ⇒ little, <c>'b'</c> ⇒ big); <c>null</c> is <c>"big"</c>.
        /// </param>
        /// <returns>A uint8 array of 0/1 values — same ndim as <paramref name="a"/> (except <c>axis=null</c> ⇒ 1-D), F-contiguous iff <paramref name="a"/> is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> does not begin with 'l' or 'b', or <c>-count</c> exceeds the available bits.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not uint8.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html</remarks>
        [NDScoped] // reclaims the ravel/ascontiguousarray/asfortranarray normalisation temps
        public static unsafe NDArray unpackbits(NDArray a, int? axis = null, long? count = null, string bitorder = "big")
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

            long axisLen = nw.shape[unpackAxis];   // input bytes along the axis
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

            long[] outDims = (long[])nw.shape.Clone();
            outDims[unpackAxis] = outLen;

            long outSize = 1;
            foreach (long d in outDims) outSize *= d;

            // Empty input but a count-forced non-empty output: NumPy's IterAllButAxis never runs its
            // zeroing loop here and leaks UNINITIALISED memory (contradicting its own "add zero padding"
            // docstring). NumSharp returns the documented deterministic zeros — the sole [Misaligned]
            // divergence, and the more correct behaviour. Truly-empty outputs need no fill.
            if (nw.size == 0 || outSize == 0)
            {
                var z = new NDArray(NPTypeCode.Byte, new Shape(outDims, 'C'), fillZeros: outSize != 0);
                return isFortran ? np.asfortranarray(z) : z;
            }

            // Main path overwrites every output byte (full bytes + partial + explicit pad), so fillZeros: false.
            var cOut = new NDArray(NPTypeCode.Byte, new Shape(outDims, 'C'), false);

            NDArray srcC = nw.Shape.IsContiguous ? nw : np.ascontiguousarray(nw);
            ComputeAxisGeometry(srcC.shape, unpackAxis, out long nOuter, out long nel);

            // Logical element 0 = base Address + Shape.offset (uint8, so offset is in bytes). The fresh
            // output is offset 0.
            byte* srcPtr = (byte*)srcC.Address + srcC.Shape.offset;
            PackBits.Unpack(
                srcPtr, (byte*)cOut.Address,
                nOuter, axisLen, nel, outLen, big);

            return isFortran ? np.asfortranarray(cOut) : cOut;
        }

        // ============================ helpers ============================

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
                // A 0-d array has one implicit axis after promotion; NumPy accepts axis 0 or -1.
                if (requested != 0 && requested != -1)
                    throw new AxisError(requested, 0);
                axis = 0;
                return a.reshape(1);
            }

            int norm = requested < 0 ? requested + a.ndim : requested;
            if (norm < 0 || norm >= a.ndim)
                throw new AxisError(requested, a.ndim);
            axis = norm;
            return a;
        }

        // ComputeAxisGeometry(long[] dims, int axis, out long n_outer, out long nel) is shared with
        // np.repeat (same all-but-axis collapse: n_outer = product of dims before the axis, nel =
        // product of dims after it — the inner-column count, 1 ⇒ the axis is innermost/contiguous).

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
