using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // =====================  text: NumSharp fixed-width 'U' / 'S' arrays  <->  ORT string tensors  =====================
        //
        // ##################################################################################################################
        // # TODO (variable-width strings) - READ THIS BEFORE CHANGING ANYTHING HERE.                                        #
        // #                                                                                                                #
        // # ONNX Runtime strings are VARIABLE-width. NumSharp has only NumPy's FIXED-width string dtypes today ('U',       #
        // # UCS-4 code points, and 'S', bytes). NumPy's variable-width StringDType (NEP 55, np.dtypes.StringDType, kind    #
        // # 'T') is phase 3 of docs/plans/strings-phase2.md and does not exist in NumSharp yet. Until it does, a string   #
        // # tensor is read into a 'U' array whose width is the tensor's LONGEST string, which costs three things:          #
        // #                                                                                                                #
        // #   1. A string that ENDS in NUL characters loses them: fixed-width storage pads with NUL and strips the         #
        // #      padding on read, so "ab\0" comes back as "ab". (A NUL inside a string survives.)                          #
        // #   2. One long string widens EVERY element: 4 bytes per code point of the longest string, per element.          #
        // #   3. Nothing is shared: every crossing, in either direction, is a copy. ORT keeps each string as a separate    #
        // #      native object, so even with StringDType only the import could ever avoid a copy.                          #
        // #                                                                                                                #
        // # When StringDType lands: map TensorElementType.String <-> StringDType in TryFromTensorElementType /             #
        // # TryToTensorElementType, read into it in TextArray, and write from it in TextToOrtValue. Keep the fixed-width   #
        // # paths: a caller's 'U' / 'S' array must still cross.                                                            #
        // #                                                                                                                #
        // # Also open: NumSharp cannot convert numbers to text or text to numbers yet (astype between a string dtype and  #
        // # a numeric one throws), so the Run tier feeds a string model input from a TEXT array only.                      #
        // ##################################################################################################################

        /// <summary>
        ///     The dtype an ORT string tensor reads back as: the UNSIZED fixed-width Unicode descriptor
        ///     (<c>np.dtype("U")</c>). An array read from a tensor gets a sized one (<c>&lt;U5</c>), the width of its longest
        ///     string. <see cref="NPTypeCode.String"/> itself cannot be a <see cref="DType"/>: the storage lane names neither
        ///     the class (<c>U</c> / <c>S</c>) nor a width, and converting it throws.
        /// </summary>
        internal static readonly DType TextDType = np.dtype("U");

        /// <summary>
        ///     Is <paramref name="array"/> a text array (NumSharp's fixed-width <c>U</c> or <c>S</c> dtype)? Every verb asks
        ///     this BEFORE it touches the dtype maps with the array's type code: <see cref="NPTypeCode.String"/> does not
        ///     convert to a <see cref="DType"/>, so the numeric path would fail on its first line with NumSharp's own
        ///     "asked for a dtype by storage lane" error.
        /// </summary>
        /// <param name="array">Any array.</param>
        /// <returns><c>true</c> for a <c>U</c> or <c>S</c> array.</returns>
        internal static bool IsText(NDArray array) => array.typecode == NPTypeCode.String;

        /// <summary>
        ///     Copy a text array into a new ORT string tensor of the same shape, in logical (C) order. A <c>U</c> element
        ///     crosses as its text (ORT stores UTF-8); an <c>S</c> element crosses as its bytes, which ORT takes as UTF-8
        ///     without checking them (what Python's <c>onnxruntime</c> does with a bytes array).
        /// </summary>
        /// <remarks>
        ///     <para>Any layout: the elements are read through the array's strides. The array's buffer is referenced for the
        ///     duration of the read, so a released buffer is refused instead of being read.</para>
        ///     <para>Lossy edges, both outside what a model normally sees: a lone UTF-16 surrogate in a <c>U</c> element has
        ///     no UTF-8 form and becomes U+FFFD; and see the TODO at the top of this file for trailing NULs.</para>
        /// </remarks>
        /// <param name="source">A <c>U</c> or <c>S</c> array of any shape and layout.</param>
        /// <param name="verb">The public verb, named in the refusal.</param>
        /// <returns>A new string tensor ONNX Runtime owns; the caller disposes it.</returns>
        /// <exception cref="NotSupportedException">The array holds more strings than ORT's string tensor API can index (<see cref="int.MaxValue"/>).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static OrtValue TextToOrtValue(NDArray source, string verb)
        {
            Shape shape = source.Shape;
            long count = shape.Size;
            if (count > int.MaxValue)
                throw new NotSupportedException(
                    $"{verb}: ONNX Runtime's string tensor API addresses elements with an Int32 index, and the array holds {count} strings.");
            long[] dims = LongDims(shape);
            bool unicode = source.dtype.kind == 'U';

            IArraySlice pin = Pin(source);   // the liveness check, and what keeps the buffer alive while it is read
            try
            {
                OrtValue value = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, dims);
                try
                {
                    if (unicode)
                    {
                        // One decode of the whole array in logical order (it reads a C-contiguous array as one run).
                        string[] items = NDArray.AsStringArray(source);
                        for (int i = 0; i < items.Length; i++)
                        {
                            if (items[i].Length != 0)   // the tensor starts as all empty strings
                                value.StringTensorSetElementAt(items[i].AsSpan(), i);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < count; i++)
                        {
                            // An S element reads back as its bytes with the NUL padding stripped. ORT's byte overload
                            // stores them as they are: the tensor's "UTF-8" is whatever the array holds.
                            var bytes = (byte[])source.Storage.GetAtIndex(i);
                            if (bytes.Length != 0)
                                value.StringTensorSetElementAt(new ReadOnlySpan<byte>(bytes), i);
                        }
                    }

                    return value;
                }
                catch
                {
                    value.Dispose();
                    throw;
                }
            }
            finally
            {
                pin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     <see cref="AsOrtValue"/> for a text array: a handle over a new ORT string tensor. A text tensor is always a
        ///     copy, so the handle reports <see cref="OrtTensor.IsView"/> <c>false</c>, holds no NumSharp buffer, and its
        ///     <see cref="OrtTensor.Source"/> is the array the strings were copied from.
        /// </summary>
        /// <param name="source">A <c>U</c> or <c>S</c> array.</param>
        /// <param name="ownsSource"><c>true</c> when <paramref name="source"/> is a temporary the handle should dispose.</param>
        /// <param name="mode">The requested mode (already validated); <see cref="OnnxCodecMode.View"/> is refused.</param>
        /// <returns>A handle owning the string tensor.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="OnnxCodecMode.View"/>: text cannot be shared.</exception>
        /// <exception cref="NotSupportedException">The array holds more strings than ORT's string tensor API can index.</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static OrtTensor TextAsOrtValue(NDArray source, bool ownsSource, OnnxCodecMode mode)
        {
            if (mode == OnnxCodecMode.View)
                throw new InvalidOperationException(TextViewMessage(nameof(AsOrtValue)));
            long[] dims = LongDims(source.Shape);
            OrtValue value = TextToOrtValue(source, nameof(AsOrtValue));
            try
            {
                // No slice, no data pointer: the strings live in ORT's memory, so there is no NumSharp buffer to keep alive
                // and the import verbs must not alias one (OrtTensor.IsNumSharpMemory is false).
                return new OrtTensor(value, source, slice: null, data: IntPtr.Zero, byteLength: 0, ownsSource, isView: false, TensorElementType.String, dims);
            }
            catch
            {
                value.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Read an ORT string tensor into a new, owning <c>U</c> array of the tensor's shape (see the TODO at the top
        ///     of this file for what the fixed width costs).
        /// </summary>
        /// <param name="value">A string tensor.</param>
        /// <param name="dims">The tensor's shape (already read by the caller).</param>
        /// <returns>A new owning C-contiguous <c>U</c> array; its width is the longest string's length in code points (at least 1).</returns>
        /// <exception cref="ArgumentException">The shape has a negative (symbolic) dimension.</exception>
        internal static NDArray TextToNDArray(OrtValue value, long[] dims)
        {
            ShapeFrom(dims);   // refuses a negative dimension with the import verbs' message, before anything is read
            // ORT decodes every element from UTF-8 (a byte sequence that is not UTF-8 becomes U+FFFD).
            string[] items = value.GetStringTensorAsArray();
            GC.KeepAlive(value);
            return TextArray(items, dims);
        }

        /// <summary>
        ///     Build a new owning <c>U</c> array of a given shape from its strings in row-major order — NumPy's
        ///     <c>np.array(list_of_str).reshape(shape)</c>, but the result owns its buffer.
        /// </summary>
        /// <param name="items">The strings in row-major order; <c>null</c> entries (a caller's own <c>Tensor&lt;string&gt;</c> may hold them) count as empty, as ORT itself treats them.</param>
        /// <param name="dims">The shape; its element count is <paramref name="items"/>' length (an empty array for a 0-d shape is not allowed: a 0-d tensor has one element).</param>
        /// <returns>The array: 0-d for an empty <paramref name="dims"/>, otherwise of that shape.</returns>
        internal static NDArray TextArray(string[] items, long[] dims)
        {
            for (int i = 0; i < items.Length; i++)
                items[i] ??= string.Empty;   // np.array refuses a null entry (Python's None), ORT reads it as ""

            if (dims.Length == 0)
                return NDArray.Scalar(items[0]);   // the 0-d <U{n} array

            NDArray flat = np.array(items);        // 1-D, owning; the width is the longest value's (at least 1)
            if (dims.Length == 1)
                return flat;
            try
            {
                // The same element count, so this is NumPy's in-place reshape: no copy, and the array keeps owning its
                // buffer (a reshape VIEW of it would report owndata == False and leave the 1-D owner to its finalizer).
                flat.resize(dims);
                return flat;
            }
            catch
            {
                flat.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Read a legacy <see cref="Tensor{T}"/> of strings (inside a <see cref="NamedOnnxValue"/>: a string output of
        ///     the legacy <c>session.Run</c>, or a caller's own tensor) into a new owning <c>U</c> array, in the tensor's
        ///     LOGICAL order: a column-major tensor is read through its own indexer, as the numeric verbs transpose one.
        /// </summary>
        /// <param name="tensor">The string tensor; row- or column-major, dense or not.</param>
        /// <returns>A new owning <c>U</c> array of the tensor's shape.</returns>
        internal static NDArray TextTensorToNDArray(Tensor<string> tensor)
        {
            long[] dims = ToLongDims(tensor.Dimensions);
            var items = new string[tensor.Length];
            if (!tensor.IsReversedStride || dims.Length < 2)
            {
                // Row-major (or 0-/1-D, where both orders agree): the flat index IS the row-major position.
                for (int i = 0; i < items.Length; i++)
                    items[i] = tensor.GetValue(i);
            }
            else
            {
                // Column-major: walk the multi-index in row-major order and let the tensor's indexer apply its strides.
                var index = new int[dims.Length];
                for (int flat = 0; flat < items.Length; flat++)
                {
                    items[flat] = tensor[index];
                    for (int d = index.Length - 1; d >= 0; d--)
                    {
                        if (++index[d] < dims[d])
                            break;
                        index[d] = 0;
                    }
                }
            }

            return TextArray(items, dims);
        }

        /// <summary>The refusal of <see cref="OnnxCodecMode.View"/> for text, in either direction.</summary>
        /// <param name="verb">The verb refusing (named in the message).</param>
        /// <returns>The message, naming the two modes that work.</returns>
        internal static string TextViewMessage(string verb) =>
            $"{verb}(mode: OnnxCodecMode.View): text crosses between NumSharp and ONNX Runtime only as a copy. ONNX Runtime keeps every " +
            "string as a separate object in its own memory, and NumSharp keeps fixed-width code points, so the two cannot share a buffer. " +
            "Pass OnnxCodecMode.Auto or OnnxCodecMode.Copy.";

        /// <summary>The refusal of the two <see cref="DenseTensor{T}"/> export verbs for a text array.</summary>
        /// <param name="verb">The verb refusing (named in the message).</param>
        /// <returns>The message, naming the verbs that take text and the way to build the legacy tensor by hand.</returns>
        internal static string TextDenseTensorMessage(string verb) =>
            $"{verb}: a text array crosses as an ORT string tensor, and this verb builds a DenseTensor<T> of an unmanaged T, which a string " +
            "is not. Use ToOrtValue / AsOrtValue (a string OrtValue), or build the legacy tensor yourself: " +
            "new DenseTensor<string>(NDArray.AsStringArray(nd), dimensions).";
    }
}
