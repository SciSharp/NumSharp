using System;
using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics.Tensors;

namespace NumSharp.Interop.Tensors
{
    // The class's documentation is on its main declaration (NDArrayTensorsInterop.cs, "Text" paragraph); this part holds
    // the text verbs. A second <summary> on a partial would be concatenated into the shipped XML docs.
    public static partial class NDArrayTensorsInterop
    {
        // ===================================  text  ===================================
        //
        // A Tensor<string> holds managed string objects; NumSharp's text is UCS-4 code units (U), bytes (S) or a UTF-8 arena
        // (StringDType) in unmanaged memory, so no string can be shared in either direction. Under the bridges' Auto rule —
        // a view when the layout allows it, otherwise a copy — text always takes the copy, so it crosses through To... verbs
        // only; the view verbs (AsTensorSpan<T> / ToTensor<T> over unmanaged T) refuse it and name ToStringTensor.

        /// <summary>
        ///     Copy a NumSharp text array — fixed-width <c>U</c> / <c>S</c> or <c>StringDType</c> — into an independent, dense
        ///     <see cref="Tensor{T}"/> of <see cref="string"/>, in logical (C) order. Always a copy: no string can be shared
        ///     between NumSharp's text storage and a managed <c>Tensor&lt;string&gt;</c>.
        /// </summary>
        /// <remarks>
        ///     Each element is the text numpy's <c>tolist()</c> gives: a <c>U</c> element without its trailing NULs; an
        ///     <c>S</c> element decoded as UTF-8, strictly (<c>np.strings.decode</c>'s default); a <c>StringDType</c> element
        ///     as its text and an NA as <c>null</c> — except an NA object that is itself a string (<c>na_object="NA"</c>), which
        ///     crosses as that string, as numpy's <c>tolist()</c> returns it. The shape follows <see cref="ToTensor{T}(NDArray)"/>:
        ///     a 0-d array becomes the single-element vector <c>[1]</c> (the BCL has no rank 0), an empty array keeps its shape.
        /// </remarks>
        /// <param name="source">The text array; any layout (strided, transposed, reversed and broadcast views included).</param>
        /// <returns>A new dense <c>Tensor&lt;string&gt;</c> with the array's shape and texts.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not a text array; <see cref="ToTensor{T}(NDArray)"/>
        ///     copies a numeric or boolean one.</exception>
        /// <exception cref="NotSupportedException">The array has more elements than a managed array holds
        ///     (<see cref="Array.MaxLength"/>); refused before anything is read.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> element is not valid UTF-8 (numpy's <c>UnicodeDecodeError</c> text).</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static Tensor<string> ToStringTensor(this NDArray source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            DType dtype = source.dtype;
            NPTypeCode code = dtype.GetTypeCode();
            if (code != NPTypeCode.String)
                throw new ArgumentException(
                    $"ToStringTensor copies a text array (U, S or StringDType); this is a NumSharp {code} array, which " +
                    $"ToTensor<{ToTensorElementClrType(dtype).Name}>() copies.", nameof(source));

            Shape shape = source.Shape;
            if (shape.Size > Array.MaxLength)
                throw new NotSupportedException(
                    $"ToStringTensor: a managed string[] holds at most {Array.MaxLength} elements (Array.MaxLength), the array has {shape.Size}.");

            // A 0-d scalar needs explicit lengths [1]: empty lengths would make a length-0 tensor and drop the text.
            nint[] lengths = shape.NDim == 0 ? new nint[] { 1 } : NIntLengths(shape);
            return Tensor.Create(Texts(source), lengths);
        }

        /// <summary>
        ///     The array's texts in logical (C) order, one per element, as <see cref="ToStringTensor"/> documents them.
        /// </summary>
        /// <param name="source">A text array.</param>
        /// <returns>The texts (null for a StringDType NA whose object is not a string).</returns>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> element is not valid UTF-8.</exception>
        private static string[] Texts(NDArray source)
        {
            switch (source.dtype.kind)
            {
                case 'U':
                    // AsStringArray reads any layout in logical order and drops each element's trailing NULs (getitem's rule).
                    return NDArray.AsStringArray(source);
                case 'S':
                {
                    // An S element holds bytes; its text is their UTF-8 decode, raising numpy's error for a sequence that is
                    // not UTF-8 rather than inventing replacement characters.
                    using NDArray decoded = np.strings.decode(source);
                    return NDArray.AsStringArray(decoded);
                }
                default:
                {
                    // StringDType: tolist() of the C-order copy gives each element's text, or the NA object for an NA.
                    // AsStringArray refuses a non-string NA (a float NaN); here every NA object that is not a string —
                    // None, NaN — becomes null, the one "missing" a string tensor can hold.
                    using NDArray flat = source.flatten();
                    return ((IEnumerable)flat.tolist()).Cast<object>().Select(o => o as string).ToArray();
                }
            }
        }

        /// <summary>
        ///     Copy a <see cref="Tensor{T}"/> of <see cref="string"/> into a NumSharp <c>StringDType(na_object=None)</c> array of
        ///     its shape — numpy's <c>np.array(texts, dtype=StringDType(na_object=None))</c>: every text kept whole (any
        ///     length; embedded NULs and astral characters included), a <c>null</c> element as the NA. Always a copy.
        /// </summary>
        /// <param name="tensor">The tensor to copy.</param>
        /// <returns>A new <c>StringDType(na_object=None)</c> array with the tensor's lengths and texts in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">The tensor has more elements than a managed array holds.</exception>
        /// <exception cref="UnicodeEncodeError">A text holds a lone surrogate, which has no UTF-8 form (numpy's
        ///     <c>UnicodeEncodeError</c>: surrogates not allowed).</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static NDArray ToNDArray(this Tensor<string> tensor)
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            return tensor.AsReadOnlyTensorSpan().ToNDArray();
        }

        /// <summary>
        ///     Copy a <see cref="ReadOnlyTensorSpan{T}"/> of <see cref="string"/> into a NumSharp
        ///     <c>StringDType(na_object=None)</c> array, read in the span's logical (C) order (a sliced or strided span
        ///     included), a <c>null</c> element as the NA.
        /// </summary>
        /// <remarks>
        ///     A rank-0 span (<c>ReadOnlyTensorSpan&lt;string&gt;.Empty</c>, a <c>default</c> span) holds no element and comes
        ///     back as the empty vector <c>(0,)</c>, as the numeric import does (see <see cref="ImportShape"/>).
        /// </remarks>
        /// <param name="span">The span to copy.</param>
        /// <returns>A new <c>StringDType(na_object=None)</c> array with the span's lengths and texts.</returns>
        /// <exception cref="NotSupportedException">The span has more elements than a managed array holds.</exception>
        /// <exception cref="UnicodeEncodeError">A text holds a lone surrogate.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static NDArray ToNDArray(this ReadOnlyTensorSpan<string> span)
        {
            Shape shape = ImportShape(span.Lengths, span.FlattenedLength);
            long count = shape.Size;
            if (count > Array.MaxLength)
                throw new NotSupportedException(
                    $"ToNDArray: a managed string[] holds at most {Array.MaxLength} elements (Array.MaxLength); the span has {count}.");

            var texts = new string[count];
            if (count > 0)
                span.FlattenTo(texts);
            // np.array packs each text with StringDType's setitem (NumPy's PyArray_Pack), a null as the NA; string[] passes
            // as object[] by array covariance, so the texts are not copied again.
            NDArray flat = np.array((object[])texts, np.dtypes.StringDType((object)null));
            return shape.NDim == 1 ? flat : flat.reshape(shape);
        }

        /// <summary><see cref="ToNDArray(ReadOnlyTensorSpan{string})"/> for the mutable <see cref="TensorSpan{T}"/>.</summary>
        /// <param name="span">The span to copy.</param>
        /// <returns>A new <c>StringDType(na_object=None)</c> array with the span's lengths and texts.</returns>
        /// <exception cref="NotSupportedException">The span has more elements than a managed array holds.</exception>
        /// <exception cref="UnicodeEncodeError">A text holds a lone surrogate.</exception>
        [Experimental("NUMSHARP_TENSORS")]
        public static NDArray ToNDArray(this TensorSpan<string> span)
            => span.AsReadOnlyTensorSpan().ToNDArray();
    }
}
