using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Return the fractional and integral parts of an array, element-wise — NumPy's two-output
        /// <c>modf</c> ufunc, with the <c>out=</c>/<c>where=</c>/<c>dtype=</c> parameters.
        ///
        /// NumPy behavior (C standard modf; all probed against NumPy 2.4.2):
        /// - modf(1.5) = (0.5, 1.0)
        /// - modf(-2.7) = (-0.7, -2.0)  -- the fractional AND integral parts carry the input's sign
        /// - modf(inf) = (+0.0, +inf), modf(-inf) = (-0.0, -inf)
        /// - modf(nan) = (nan, nan)
        /// - modf(-0.0) = (-0.0, -0.0)  -- the signed zero is preserved on both outputs
        ///
        /// Loop selection (float loops only, <c>e/f/d/g</c>): bool/int8/uint8 to float16,
        /// int16/uint16/char to float32, int32/uint32/int64/uint64 to float64 (NumPy's NEP50
        /// narrowest-float tier, NOT a blanket float64); float16/float32/float64 are preserved, Complex has
        /// NO loop (NumPy's "not supported for the input types" TypeError), and Decimal is a NumSharp
        /// extension (no NumPy analog). An explicit
        /// <paramref name="dtype"/> must itself be one of the float loops or NumPy raises the no-loop
        /// TypeError (dtype=int32/complex128 → "No loop matching …").
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="dtype">Optional loop dtype override; must be Half/Single/Double/Decimal.</param>
        /// <param name="outFrac">Optional provided output for the fractional part (NumPy's <c>out[0]</c>);
        /// same_kind-castable from the loop dtype, else "Cannot cast ufunc 'modf' output 1 …".</param>
        /// <param name="outIntegral">Optional provided output for the integral part (NumPy's <c>out[1]</c>);
        /// same_kind-castable from the loop dtype, else "… output 2 …".</param>
        /// <param name="where">Optional bool write mask; masked-off slots of BOTH provided outputs keep
        /// their prior contents.</param>
        /// <returns>The (fractional, integral) pair — a provided output is returned as-is (reference
        /// identity), an omitted one is a fresh loop-dtype array whose layout follows the input (an
        /// F-contiguous input yields F-contiguous fresh outputs, matching NumPy).</returns>
        /// <exception cref="TypeError">Complex input (no loop), or a non-float <paramref name="dtype"/>.</exception>
        /// <exception cref="ValueError">A read-only provided output.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, an out not
        /// same_kind-castable from the loop dtype, or an output that would have to be broadcast/stretched.</exception>
        public override (NDArray Fractional, NDArray Integral) ModF(
            NDArray nd, DType dtype = null,
            NDArray outFrac = null, NDArray outIntegral = null, NDArray where = null)
        {
            // Loop dtype: promote the input (int/bool/char → float64, floats preserved) or honour a valid
            // dtype= override; a complex input or a non-float dtype= raises here, before any allocation.
            NPTypeCode loopType = ResolveModfLoopType(nd.typecode, dtype);

            // where must be exactly bool (NumPy's 'safe' wheremask rule).
            ValidateWhereMask(where);

            // Read-only + same_kind out-cast validation, in NumPy's operand order (frac = output 1,
            // integral = output 2) so the FIRST offending output is the one reported — and before any
            // compute, so a read-only/bad-cast out fails without touching memory.
            if (outFrac is not null)
            {
                NumSharpException.ThrowIfNotWriteable(outFrac.Shape, "output array");
                ValidateModfOutCast(loopType, outFrac.typecode, 1);
            }
            if (outIntegral is not null)
            {
                NumSharpException.ThrowIfNotWriteable(outIntegral.Shape, "output array");
                ValidateModfOutCast(loopType, outIntegral.typecode, 2);
            }

            // Iteration (= output) shape: the input broadcasts UP to a bigger provided output (which is
            // never itself stretched), and a where mask joins the shape too. With no out/where this is
            // just the input's shape, so the common path allocates exactly what it did before.
            Shape iterShape = ResolveModfIterationShape(nd.Shape, outFrac, outIntegral, where);

            // The OUT-OF-PLACE kernel reads `source[i]` and writes `frac[i]`/`integral[i]` at the SAME
            // linear index, so `source` AND every write target must share ONE memory layout for the logical
            // positions to align — the whole family is made C-CONTIGUOUS (an F-contiguous fresh output is
            // relabelled afterward by np.modf's PreserveFContig). iterShape can carry F/broadcast strides,
            // so a fresh temp is built from its DIMENSIONS only.

            // READ source: a C-contiguous, loop-dtype view in logical order. ascontiguousarray RETURNS THE
            // INPUT when it is already C-contiguous and the loop dtype (the common float path — no copy),
            // and otherwise mints one C-contiguous copy (densifying a broadcast view, casting an integer
            // input, or reordering an F/strided one). It never returns a non-owning view, so a result that
            // is not `readFrom` is a fresh copy to dispose after the kernel.
            // A C-CONTIGUOUS shape with the iteration dims for the source copy + any fresh temp: use
            // iterShape as-is when it is already C-contiguous or 0-d (a scalar is trivially contiguous, and
            // `new Shape(empty)` would wrongly build a 1-D shape), and rebuild from the dimensions only when
            // it carries F-contiguous or broadcast strides (an F input or a broadcast join).
            Shape ccShape = (iterShape.NDim == 0 || iterShape.IsContiguous)
                ? iterShape
                : new Shape((long[])iterShape.dimensions.Clone());

            NDArray broadcastView = SameDims(nd.Shape, iterShape) ? null : np.broadcast_to(nd, ccShape);
            NDArray readFrom = broadcastView ?? nd;

            // A C-contiguous input (including a 0-d scalar and any C-contiguous slice) is already in logical
            // order: read it directly when the dtype matches, else a Cast COPY of a C-contiguous array is
            // itself C-contiguous and preserves the rank (Cast keeps the input layout). Only a non-C
            // layout (F-contiguous, strided, broadcast — all rank >= 1) needs ascontiguousarray, which
            // reorders to C; that call promotes a 0-d to 1-D, but a 0-d is always C-contiguous so it never
            // reaches this branch. A fresh copy is a disposable buffer we own.
            NDArray source;
            bool sourceIsFreshCopy;
            if (readFrom.Shape.IsContiguous)
            {
                sourceIsFreshCopy = loopType != readFrom.typecode;
                source = sourceIsFreshCopy ? Cast(readFrom, loopType, copy: true) : readFrom;
            }
            else
            {
                source = np.ascontiguousarray(readFrom, loopType);
                sourceIsFreshCopy = true;   // a non-contiguous input is always copied by ascontiguousarray
            }

            // WRITE targets: a provided output is written DIRECTLY (no temp, no copy-back) when it is a
            // clean landing zone — C-contiguous, already the loop dtype, exactly the iteration shape, and
            // not masked; otherwise the kernel writes a fresh C-contiguous temp that WriteModfResult
            // casts/masks into the output (or returns as the fresh result when no output was given).
            bool fracDirect = IsDirectModfOut(outFrac, loopType, iterShape, where);
            bool intDirect = IsDirectModfOut(outIntegral, loopType, iterShape, where);

            // When the frac part needs a temp AND we already own a fresh source copy, REUSE the source
            // buffer as the frac target (in-place: the kernel reads source[i] then writes frac[i] to the
            // same slot — safe per element). This keeps the allocation count at two (source-as-frac + the
            // integral temp) instead of three for the promotion/F/strided paths.
            bool fracReusesSource = !fracDirect && sourceIsFreshCopy;
            NDArray fracTarget = fracDirect ? outFrac
                                : fracReusesSource ? source
                                : new NDArray(loopType, ccShape, false);
            NDArray intTarget = intDirect ? outIntegral : new NDArray(loopType, ccShape, false);

            if (fracTarget.size > 0)
                RunModfKernel(loopType, source, fracTarget, intTarget);

            // Route each part: a direct target IS the output (return as-is); a temp goes through
            // WriteModfResult (return the fresh temp, or cast/mask-copy it into a non-direct output — which
            // disposes the temp; when fracTarget==source that also reclaims the source copy).
            NDArray frac = fracDirect ? outFrac : WriteModfResult(fracTarget, outFrac, where);
            NDArray integral = intDirect ? outIntegral : WriteModfResult(intTarget, outIntegral, where);

            // Dispose the read-only transients the kernel only read. The source copy is reclaimed here
            // ONLY when it was not consumed as the frac target (WriteModfResult already handled that case);
            // never dispose source when it aliases the caller's input (readFrom == nd, not a fresh copy).
            if (sourceIsFreshCopy && !fracReusesSource)
                source.Dispose();
            if (broadcastView is not null && !ReferenceEquals(source, broadcastView))
                broadcastView.Dispose();
            return (frac, integral);
        }

        /// <summary>
        /// True when a provided modf output can be written by the kernel DIRECTLY — no temp, no cast, no
        /// masked write-back. That needs: an output that exists, is not masked (a <c>where</c> requires
        /// keeping the prior contents of masked-off slots, so it must go through the masked copy), is
        /// already the loop dtype (no cast), is C-contiguous (the kernel writes linearly), and is exactly
        /// the iteration shape (never stretched — the shape resolver already guaranteed this for a provided
        /// output, so this is a defensive check).
        /// </summary>
        /// <param name="output">The provided output, or null.</param>
        /// <param name="loopType">The loop (compute) dtype.</param>
        /// <param name="iterShape">The iteration (output) shape.</param>
        /// <param name="where">The optional write mask.</param>
        /// <returns>True if the kernel may write straight into <paramref name="output"/>.</returns>
        private static bool IsDirectModfOut(NDArray output, NPTypeCode loopType, Shape iterShape, NDArray where)
            => output is not null
               && where is null
               && output.typecode == loopType
               && output.Shape.IsContiguous
               && SameDims(output.Shape, iterShape);

        /// <summary>
        /// Resolve modf's loop (output) dtype from the input dtype and an optional <c>dtype=</c> override,
        /// matching NumPy's float-only loop set (<c>e/f/d/g</c>).
        /// </summary>
        /// <param name="inputType">The input array's dtype.</param>
        /// <param name="dtype">The optional <c>dtype=</c> override.</param>
        /// <returns>The loop dtype (Half/Single/Double/Decimal).</returns>
        /// <exception cref="TypeError">A complex input (no loop), or a non-float <paramref name="dtype"/>.</exception>
        private static NPTypeCode ResolveModfLoopType(NPTypeCode inputType, DType dtype)
        {
            if (dtype is not null)
            {
                NPTypeCode dt = dtype.GetTypeCode();
                // dtype= must select one of modf's float loops. Anything else (int/uint/char/bool/complex)
                // has no loop to bind — NumPy's verbatim no-loop TypeError.
                if (dt == NPTypeCode.Half || dt == NPTypeCode.Single || dt == NPTypeCode.Double || dt == NPTypeCode.Decimal)
                    return dt;
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc modf");
            }

            switch (inputType)
            {
                // Integer/bool promote to the NARROWEST float that fits the input width (NumPy's NEP50
                // width-based rule for a float-only ufunc — the SAME tier ResolveUnaryFloatReturnType uses
                // for sin/frexp, NOT a blanket float64). Probed 2.4.2: modf(int8)->float16,
                // modf(int16)->float32, modf(int32)->float64.
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                    return NPTypeCode.Half;       // float16
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    return NPTypeCode.Single;     // float32
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return NPTypeCode.Double;     // float64
                // Float family is preserved (float16->float16, float32->float32, float64->float64); Decimal
                // is a NumSharp-only extension with no NumPy analog.
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                    return inputType;
                // Complex has no modf loop (there is no meaningful fractional/integral split of a complex
                // value): NumPy reports it as an unsupported input type.
                case NPTypeCode.Complex:
                    throw new TypeError(
                        "ufunc 'modf' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
                default:
                    return NPTypeCode.Double;
            }
        }

        /// <summary>
        /// Validate that a provided modf output is same_kind-castable from the loop dtype, with NumPy's
        /// verbatim multi-output error text (which numbers the OPERAND: the fractional output is 1, the
        /// integral output is 2 — <c>nin + output_position</c>).
        /// </summary>
        /// <param name="loopType">The loop (compute) dtype.</param>
        /// <param name="outType">The provided output's dtype.</param>
        /// <param name="operandIndex">1 for the fractional output, 2 for the integral output.</param>
        /// <exception cref="ArgumentException">When the cast is not permitted under the same_kind rule.</exception>
        private static void ValidateModfOutCast(NPTypeCode loopType, NPTypeCode outType, int operandIndex)
        {
            if (loopType == outType)
                return;
            if (NDIterCasting.CanCast(loopType, outType, NPY_CASTING.NPY_SAME_KIND_CASTING))
                return;
            throw new ArgumentException(
                $"Cannot cast ufunc 'modf' output {operandIndex} from " +
                $"dtype('{loopType.AsNumpyDtypeName()}') to " +
                $"dtype('{outType.AsNumpyDtypeName()}') with casting rule 'same_kind'");
        }

        /// <summary>
        /// Resolve modf's iteration (= output) shape: the input broadcasts up to each provided output
        /// (which may never itself be stretched) and to the where mask. Reproduces NumPy's broadcast
        /// error texts — the non-broadcastable-output message and the operand-list "operands could not be
        /// broadcast together with shapes …" (input first, then each provided output).
        /// </summary>
        /// <param name="inputShape">The input array's shape.</param>
        /// <param name="outFrac">The optional fractional output.</param>
        /// <param name="outIntegral">The optional integral output.</param>
        /// <param name="where">The optional where mask.</param>
        /// <returns>The iteration shape (the shape of both outputs).</returns>
        /// <exception cref="ArgumentException">On an incompatible or stretched output/mask.</exception>
        private static Shape ResolveModfIterationShape(Shape inputShape, NDArray outFrac, NDArray outIntegral, NDArray where)
        {
            Shape full = inputShape.Clean();

            full = JoinModfOutput(full, outFrac, inputShape, outFrac, outIntegral);
            full = JoinModfOutput(full, outIntegral, inputShape, outFrac, outIntegral);

            if (where is not null)
            {
                Shape withWhere;
                try
                {
                    withWhere = Shape.ResolveReturnShape(full, where.Shape);
                }
                catch (Exception e)
                {
                    throw new ArgumentException(
                        $"operands could not be broadcast together with shapes {ModfOperandShapes(inputShape, outFrac, outIntegral, where)}", e);
                }

                // A provided output cannot be stretched by the mask (both outputs already equal `full`).
                if ((outFrac is not null || outIntegral is not null) && !withWhere.Equals(full))
                    throw new ArgumentException(
                        $"non-broadcastable output operand with shape {NumPyShapeRepr(full)} " +
                        $"doesn't match the broadcast shape {NumPyShapeRepr(withWhere)}");

                full = withWhere;
            }

            return full;
        }

        /// <summary>
        /// Join one provided modf output into the running iteration shape: the inputs may broadcast up to
        /// it, but the output itself must equal the result (NumPy forbids stretching a provided output).
        /// </summary>
        /// <param name="running">The iteration shape so far.</param>
        /// <param name="output">The output to join (no-op when null).</param>
        /// <param name="inputShape">The input shape (for the error operand list).</param>
        /// <param name="outFrac">The fractional output (for the error operand list).</param>
        /// <param name="outIntegral">The integral output (for the error operand list).</param>
        /// <returns>The updated iteration shape.</returns>
        /// <exception cref="ArgumentException">On an incompatible or stretched output.</exception>
        private static Shape JoinModfOutput(Shape running, NDArray output, Shape inputShape, NDArray outFrac, NDArray outIntegral)
        {
            if (output is null)
                return running;

            Shape joined;
            try
            {
                joined = Shape.ResolveReturnShape(running, output.Shape);
            }
            catch (Exception e)
            {
                throw new ArgumentException(
                    $"operands could not be broadcast together with shapes {ModfOperandShapes(inputShape, outFrac, outIntegral, null)}", e);
            }

            if (!joined.Equals(output.Shape))
                throw new ArgumentException(
                    $"non-broadcastable output operand with shape {NumPyShapeRepr(output.Shape)} " +
                    $"doesn't match the broadcast shape {NumPyShapeRepr(joined)}");

            return joined;
        }

        /// <summary>
        /// Build NumPy's operand-shape list for a modf broadcast error: the input shape, then each
        /// provided output shape, then the where shape — space-separated with a trailing space, exactly as
        /// NumPy's iterator prints it.
        /// </summary>
        private static string ModfOperandShapes(Shape inputShape, NDArray outFrac, NDArray outIntegral, NDArray where)
        {
            string s = NumPyShapeRepr(inputShape) + " ";
            if (outFrac is not null) s += NumPyShapeRepr(outFrac.Shape) + " ";
            if (outIntegral is not null) s += NumPyShapeRepr(outIntegral.Shape) + " ";
            if (where is not null) s += NumPyShapeRepr(where.Shape) + " ";
            return s;
        }

        /// <summary>
        /// Run the OUT-OF-PLACE modf kernel: read <paramref name="source"/> (contiguous, loop dtype) once
        /// and write the fractional parts to <paramref name="fracTarget"/> and the integral parts to
        /// <paramref name="intTarget"/> (both contiguous, loop dtype) — NumPy's 3-touch pattern. Any target
        /// may alias <paramref name="source"/> per-element (e.g. a provided out that is the input itself).
        /// Dispatches to the SIMD float/double helper, the scalar Half helper, or the scalar Decimal loop.
        /// </summary>
        /// <param name="loopType">The loop dtype (Half/Single/Double/Decimal).</param>
        /// <param name="source">Contiguous loop-dtype view of the (broadcast) input to read.</param>
        /// <param name="fracTarget">Contiguous loop-dtype buffer receiving the fractional parts.</param>
        /// <param name="intTarget">Contiguous loop-dtype buffer receiving the integral parts.</param>
        private static unsafe void RunModfKernel(NPTypeCode loopType, NDArray source, NDArray fracTarget, NDArray intTarget)
        {
            long len = fracTarget.size;
            byte* src = (byte*)source.Address + (long)source.Shape.offset * source.dtypesize;
            switch (loopType)
            {
                case NPTypeCode.Double:
                    DirectILKernelGenerator.ModfHelper((double*)src, (double*)fracTarget.Address, (double*)intTarget.Address, len);
                    break;
                case NPTypeCode.Single:
                    DirectILKernelGenerator.ModfHelper((float*)src, (float*)fracTarget.Address, (float*)intTarget.Address, len);
                    break;
                case NPTypeCode.Half:
                    DirectILKernelGenerator.ModfHelper((Half*)src, (Half*)fracTarget.Address, (Half*)intTarget.Address, len);
                    break;
                case NPTypeCode.Decimal:
                    ModfDecimal((decimal*)src, (decimal*)fracTarget.Address, (decimal*)intTarget.Address, len);
                    break;
                default:
                    throw new NotSupportedException($"Unexpected modf loop type: {loopType}");
            }
        }

        /// <summary>
        /// Route one modf part into its provided output or return the fresh temp. A provided output takes a
        /// masked/cast write (<see cref="np.copyto"/> — same_kind cast, mask honoured) and is returned as-is
        /// (reference identity), disposing the temp; an omitted output returns the fresh temp unchanged
        /// (the np layer relabels it to the input's F-contiguity via <c>PreserveFContig</c>).
        /// </summary>
        /// <param name="temp">The computed part (contiguous, loop dtype).</param>
        /// <param name="output">The provided output, or null.</param>
        /// <param name="where">The optional write mask.</param>
        /// <returns>The provided output (written) or the fresh temp.</returns>
        private static NDArray WriteModfResult(NDArray temp, NDArray output, NDArray where)
        {
            if (output is null)
                return temp;

            // Provided output: masked, same_kind-cast write from the temp (validated above), then hand back
            // the caller's array. Dispose the transient temp promptly rather than leaving it to the finalizer.
            np.copyto(output, temp, "same_kind", where);
            temp.Dispose();
            return output;
        }

        /// <summary>
        /// Out-of-place scalar modf for decimal (no SIMD, decimal is 128-bit; a NumSharp extension — NumPy
        /// has no decimal dtype). Reads <paramref name="input"/> once and writes fractional/integral to
        /// their own buffers (aliasing input==frac is safe). <see cref="Math.Truncate(decimal)"/> gives the
        /// toward-zero integral part.
        /// </summary>
        /// <param name="input">Source array (read-only unless it aliases an output).</param>
        /// <param name="frac">Destination for the fractional parts.</param>
        /// <param name="integral">Destination for the integral parts.</param>
        /// <param name="size">Number of elements.</param>
        private static unsafe void ModfDecimal(decimal* input, decimal* frac, decimal* integral, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var v = input[i];
                var trunc = Math.Truncate(v);
                integral[i] = trunc;
                frac[i] = v - trunc;
            }
        }
    }
}
