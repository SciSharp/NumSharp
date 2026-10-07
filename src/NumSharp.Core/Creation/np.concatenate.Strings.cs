using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Whether a concatenation involves a fixed-width string dtype — an input, the <c>out</c> array or the requested
        ///     <c>dtype</c> — and so must resolve its result by DESCRIPTOR.
        /// </summary>
        /// <param name="arrays">The (validated) inputs.</param>
        /// <param name="out">The output array, or null.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns>True when the string route applies.</returns>
        private static bool ConcatenateTouchesStrings(NDArray[] arrays, NDArray @out, DType dtype)
        {
            if (dtype?.Meta is StringDTypeMeta || @out?.typecode == NPTypeCode.String)
                return true;
            foreach (var a in arrays)
                if (a.typecode == NPTypeCode.String)
                    return true;
            return false;
        }

        /// <summary>
        ///     <c>np.concatenate</c> when a string dtype is involved — NumPy's <c>PyArray_ConcatenateInto</c> over
        ///     descriptors: the result descriptor is <paramref name="dtype"/> adapted to the inputs
        ///     (<c>PyArray_FindConcatenationDescriptor</c>: an unsized <c>'U'</c> takes the widest input's length), else
        ///     <paramref name="out"/>'s, else the inputs' <c>result_type</c> (<c>U3 + S5 → &lt;U5</c>); every input must cast
        ///     to it under <paramref name="casting"/>; each input is then copied into its slab of the result through the string
        ///     copy route (truncation, NUL padding, <c>S ↔ U</c> ASCII checks).
        /// </summary>
        /// <param name="arrays">The inputs (same rank, matching non-axis extents — already validated).</param>
        /// <param name="axis">The normalized concatenation axis.</param>
        /// <param name="ndim">The inputs' rank.</param>
        /// <param name="resultDims">The result dimensions.</param>
        /// <param name="out">The output array, or null to allocate.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <param name="casting">The casting rule name (validated by the caller).</param>
        /// <returns>The concatenated array (<paramref name="out"/> when given).</returns>
        /// <exception cref="InvalidCastException">An input cannot cast to the result descriptor under <paramref name="casting"/>
        ///     (NumPy's TypeError text: <c>Cannot cast array data from dtype('&lt;U1') to dtype('S3') according to the rule
        ///     'same_kind'</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> has the wrong rank or shape.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> input is not ASCII and the result is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> input is not ASCII and the result is <c>U</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">The cast is allowed but converts between text and numbers for
        ///     an input with elements: a text input parsed into a numeric result (at that input), or a numeric input formatted
        ///     into a text result (reported once every other input passed — formatting cannot fail). An EMPTY input converts
        ///     nothing.</exception>
        /// <remarks>
        ///     <b>Order (NumPy's <c>PyArray_ConcatenateArrays</c>).</b> The <paramref name="out"/> shape first; then each input
        ///     in turn through <c>PyArray_AssignArray</c> — its casting rule, then its conversion — so the FIRST input's rule or
        ///     conversion error wins over a later input's (a non-ASCII <c>S</c> input before a <c>U5</c> input that fails
        ///     <c>'safe'</c> into <c>&lt;U3</c> reports the UnicodeDecodeError). Formatting a number cannot fail and is
        ///     unobservable once the call fails, so a numeric input defers only when every input passed.
        /// </remarks>
        private static NDArray ConcatenateStrings(NDArray[] arrays, int axis, int ndim, long[] resultDims, NDArray @out, DType dtype, string casting)
        {
            var rule = DTypeCasting.ParseCasting((casting ?? string.Empty).ToLowerInvariant());

            DType result;
            if (dtype is not null)
                result = FindConcatenationDescriptor(arrays, dtype);
            else if (@out is not null)
                result = @out.dtype;
            else
                result = DTypePromotion.ResultType(arrays.Cast<object>().ToArray());
            bool resultIsText = result.Meta is StringDTypeMeta;

            NDArray dst;
            if (@out is not null)
            {
                if (@out.ndim != ndim)
                    throw new IncorrectShapeException("Output array has wrong dimensionality");
                for (int j = 0; j < ndim; j++)
                {
                    if (@out.Shape.dimensions[j] != resultDims[j])
                        throw new IncorrectShapeException(
                            "Output array is the wrong shape");
                }
                dst = @out;
            }
            else
            {
                // NumPy's multi-operand KEEPORDER layout vote, as for every dtype.
                var inputShapes = new Shape[arrays.Length];
                for (int k = 0; k < arrays.Length; k++)
                    inputShapes[k] = arrays[k].Shape;
                // The pieces cover the result exactly, so a U result skips the NEEDS_INIT zero fill. (A deferred numeric
                // input leaves its slab unwritten, but then the result is discarded below.)
                dst = NDArray.ForOverwrite(result, Shape.ConcatOutputShape(resultDims, inputShapes));
            }

            try
            {
                var accessor = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    accessor[i] = Slice.All;

                long position = 0;
                DType pendingFormat = null;
                foreach (var src in arrays)
                {
                    // PyArray_AssignArray, input by input: the casting rule (a dtype check, so it runs for an empty input
                    // too), then the conversion.
                    if (!DTypeCasting.CanCastTypeTo(src.dtype, result, rule))
                        throw NDStringCasting.InvalidCast(src.dtype, result, rule, scalar: false);

                    long len = src.Shape.dimensions[axis];
                    if ((src.typecode == NPTypeCode.String) != resultIsText && src.size != 0)
                    {
                        // Parsing text into numbers can fail, so it is reported at this input; formatting numbers cannot,
                        // so a later input's error still wins and the refusal waits for the end.
                        if (!resultIsText)
                            throw NDStringCasting.TextConversion(src.dtype, result);
                        pendingFormat ??= src.dtype;
                        position += len;
                        continue;
                    }
                    if (len == 0)
                        continue;
                    accessor[axis] = new Slice(position, position + len);
                    // The slab view is a fresh wrapper over dst's storage: release it each iteration.
                    using var slab = dst[accessor];
                    NDIter.Copy(slab.Storage, src.Storage);
                    position += len;
                }

                if (pendingFormat is not null)
                    throw NDStringCasting.TextConversion(pendingFormat, result);
                return dst;
            }
            catch
            {
                // A failing input discards the result NumSharp allocated (a caller's out keeps what was written, as in NumPy).
                if (@out is null)
                    dst.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FindConcatenationDescriptor</c>: a SIZED requested descriptor is used as given; an unsized
        ///     string class (<c>'U'</c>, <c>'S'</c>) becomes the common instance of every input cast to it (the widest).
        /// </summary>
        /// <param name="arrays">The inputs.</param>
        /// <param name="requested">The requested dtype.</param>
        /// <returns>The result descriptor.</returns>
        /// <exception cref="TypeError">An input has no cast to the requested class.</exception>
        private static DType FindConcatenationDescriptor(NDArray[] arrays, DType requested)
        {
            if (!(requested.itemsize == 0 && requested.Meta is StringDTypeMeta))
                return requested;
            return DTypePromotion.CastToDTypeAndPromoteDescriptors(arrays.Select(a => a.dtype).ToArray(), requested.Meta);
        }
    }
}
