using System;
using Microsoft.ML.OnnxRuntime;

namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // ===========================  non-tensor OrtValues  ===========================
        //
        // A model output is not always a dense tensor. scikit-learn classifiers exported through skl2onnx
        // emit a ZipMap — a SEQUENCE of MAPs (probability keyed by class label); other graphs emit a plain
        // sequence of tensors. NumSharp has no map or sequence type, so these come back as NumSharp-shaped
        // pieces (one NDArray per tensor part) through the readers below rather than through ToNDArray, which
        // needs a dense tensor. A STRING part (string class labels, a sequence of string tensors) comes back as
        // a text array (fixed-width U, a copy; see NDArrayOnnxInterop.Text.cs).
        //
        // ONE copy per element, made by ORT: GetValue(i, allocator) hands out a COPY of the element in the allocator's
        // memory, and each array a reader returns is a view that OWNS that copy (AsNDArray(ownsValue: true)); it disposes
        // the element when the last NumSharp view over it dies. The readers used to copy that copy again and dispose the
        // element at once (two copies per element). Consequences for a caller: the arrays report flags.owndata == False,
        // they count in LiveImports until they are disposed (or collected), and the source OrtValue may still be disposed
        // the moment the reader returns, because the elements do not point into it. A text part is the exception: it
        // cannot be viewed, so it is an ordinary owning array and ORT's copy of it is disposed at once.

        /// <summary>
        ///     Read a SEQUENCE <see cref="OrtValue"/> whose elements are dense tensors into one <see cref="NDArray"/> per
        ///     element — e.g. a <c>SequenceConstruct</c> / <c>SplitToSequence</c> output. Each array is a C-contiguous
        ///     view that owns ORT's copy of its element (dtype from the tensor, as <see cref="ToNDArray(OrtValue)"/>
        ///     maps it), so the sequence can be disposed as soon as this returns.
        /// </summary>
        /// <param name="sequence">A sequence OrtValue whose elements are dense tensors.</param>
        /// <returns>One array per element, in sequence order; dispose each one to free ORT's copy of its element.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sequence"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     Not a sequence, or an element is not a dense tensor (a sequence of MAPs — the ZipMap shape — is
        ///     read with <see cref="ToMaps"/> instead), or an element's element type has no NumSharp dtype.
        /// </exception>
        /// <remarks>
        ///     All or nothing: when an element fails half-way through, the arrays already read are disposed before the
        ///     exception propagates — the caller never sees them, so nobody else could give their buffers back.
        /// </remarks>
        public static NDArray[] ToNDArrays(this OrtValue sequence)
        {
            if (sequence is null)
                throw new ArgumentNullException(nameof(sequence));
            if (sequence.OnnxType != OnnxValueType.ONNX_TYPE_SEQUENCE)
                throw new NotSupportedException(
                    $"ToNDArrays needs a sequence OrtValue, but this value is {sequence.OnnxType}. A dense tensor is read with " +
                    "ToNDArray / AsNDArray; a single map with ToMap; a sequence of maps (ZipMap output) with ToMaps.");

            int count = sequence.GetValueCount();
            var result = new NDArray[count];
            OrtAllocator allocator = OrtAllocator.DefaultInstance;
            try
            {
                for (int i = 0; i < count; i++)
                {
                    OrtValue element = sequence.GetValue(i, allocator);
                    try
                    {
                        if (element.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
                            throw new NotSupportedException(
                                $"ToNDArrays: sequence element {i} is {element.OnnxType}, not a dense tensor. A sequence of maps " +
                                "(e.g. a scikit-learn ZipMap output) is read with ToMaps.");
                        result[i] = element.AsNDArray(ownsValue: true);   // the array owns ORT's copy from here on
                    }
                    catch
                    {
                        // A throw never transfers ownership (AsNDArray leaves the value untouched), so the element is
                        // still this method's to free.
                        element.Dispose();
                        throw;
                    }
                }
                return result;
            }
            catch
            {
                // The arrays read so far never reach the caller: give their buffers back now instead of to the finalizer.
                foreach (NDArray made in result)
                    made?.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Read a MAP <see cref="OrtValue"/> as two parallel arrays: <c>keys</c> and the <c>values</c> at the same
        ///     positions (ORT stores a map as a keys tensor + a values tensor). The map can be disposed as soon as this
        ///     returns. The dtypes follow the map: an int64 side is a 1-D view that owns ORT's copy of its tensor; a
        ///     STRING side (string class labels as keys, or string values) is a text array (fixed-width <c>U</c>).
        /// </summary>
        /// <param name="map">A map OrtValue.</param>
        /// <returns>The keys and the values, each a 1-D array; dispose both.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="map"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a map, or its keys or its values are of a type NumSharp lacks (BFloat16).</exception>
        /// <remarks>
        ///     A failure converting the values disposes the keys array already made, so a refused map leaves no array
        ///     behind, and ORT's copies of both tensors are freed. String-keyed and string-valued maps used to be refused
        ///     (NumSharp had no text dtype); they are read now.
        /// </remarks>
        public static (NDArray keys, NDArray values) ToMap(this OrtValue map)
        {
            if (map is null)
                throw new ArgumentNullException(nameof(map));
            if (map.OnnxType != OnnxValueType.ONNX_TYPE_MAP)
                throw new NotSupportedException($"ToMap needs a map OrtValue, but this value is {map.OnnxType}.");

            OrtAllocator allocator = OrtAllocator.DefaultInstance;
            // Each local is set to null once an array owns the value, so the finally below disposes exactly the values
            // whose ownership did not transfer (a refusal, a failure half-way).
            OrtValue keys = null;
            OrtValue values = null;
            NDArray keyArray = null;
            try
            {
                keys = map.GetValue(0, allocator);     // element 0 = keys tensor (ORT's copy)
                values = map.GetValue(1, allocator);   // element 1 = values tensor (ORT's copy)

                // Owning either way: a numeric tensor becomes a view that owns ORT's copy, a string tensor becomes a text
                // array and ORT's copy is disposed at once (AsNDArray's default mode copies what it cannot view).
                keyArray = keys.AsNDArray(ownsValue: true);
                keys = null;
                NDArray valueArray = values.AsNDArray(ownsValue: true);
                values = null;
                return (keyArray, valueArray);
            }
            catch
            {
                // The keys were converted first; a values failure (e.g. a BFloat16 value tensor) must not strand them.
                // Disposing the array disposes the keys value it owns.
                keyArray?.Dispose();
                throw;
            }
            finally
            {
                keys?.Dispose();
                values?.Dispose();
            }
        }

        /// <summary>
        ///     Read a SEQUENCE of MAPs — the scikit-learn / ZipMap classifier output
        ///     (<c>sequence(map(int64, float))</c>) — as one (keys, values) pair per element. Each pair is what
        ///     <see cref="ToMap"/> returns; typically every map shares the same class-label keys, so
        ///     <c>result[i].values</c> is sample <c>i</c>'s per-class scores.
        /// </summary>
        /// <param name="sequenceOfMaps">A sequence OrtValue whose elements are maps.</param>
        /// <returns>One (keys, values) pair per element, in sequence order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="sequenceOfMaps"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a sequence, an element is not a map, or a map is refused by <see cref="ToMap"/>.</exception>
        /// <remarks>All or nothing, like <see cref="ToNDArrays"/>: a failing element disposes the pairs already read.</remarks>
        public static (NDArray keys, NDArray values)[] ToMaps(this OrtValue sequenceOfMaps)
        {
            if (sequenceOfMaps is null)
                throw new ArgumentNullException(nameof(sequenceOfMaps));
            if (sequenceOfMaps.OnnxType != OnnxValueType.ONNX_TYPE_SEQUENCE)
                throw new NotSupportedException($"ToMaps needs a sequence OrtValue, but this value is {sequenceOfMaps.OnnxType}.");

            int count = sequenceOfMaps.GetValueCount();
            var result = new (NDArray keys, NDArray values)[count];
            OrtAllocator allocator = OrtAllocator.DefaultInstance;
            try
            {
                for (int i = 0; i < count; i++)
                {
                    using OrtValue element = sequenceOfMaps.GetValue(i, allocator);
                    if (element.OnnxType != OnnxValueType.ONNX_TYPE_MAP)
                        throw new NotSupportedException(
                            $"ToMaps: sequence element {i} is {element.OnnxType}, not a map. A sequence of tensors is read with ToNDArrays.");
                    result[i] = element.ToMap();
                }
                return result;
            }
            catch
            {
                // The pairs read so far never reach the caller: give their buffers back now.
                foreach ((NDArray keys, NDArray values) in result)
                {
                    keys?.Dispose();
                    values?.Dispose();
                }
                throw;
            }
        }

        /// <summary>
        ///     The keys tensor of a MAP <see cref="OrtValue"/> as a standalone value, for a caller that wants ORT's own
        ///     value (for a string-keyed map, <see cref="ReadStringTensor"/> gives the keys as a <c>string[]</c>;
        ///     <see cref="ToMap"/> gives them as a text array).
        /// </summary>
        /// <param name="map">A map OrtValue.</param>
        /// <returns>ORT's copy of the keys tensor; the caller disposes it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="map"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a map.</exception>
        public static OrtValue GetMapKeys(this OrtValue map)
        {
            if (map is null)
                throw new ArgumentNullException(nameof(map));
            if (map.OnnxType != OnnxValueType.ONNX_TYPE_MAP)
                throw new NotSupportedException($"GetMapKeys needs a map OrtValue, but this value is {map.OnnxType}.");
            return map.GetValue(0, OrtAllocator.DefaultInstance);
        }

        /// <summary>
        ///     Read a STRING tensor <see cref="OrtValue"/> as a flat <see cref="string"/>[] in row-major order: the
        ///     strings exactly as ONNX Runtime decodes them, with no fixed width (a string that ends in NUL characters
        ///     keeps them, which the text array <see cref="ToNDArray(OrtValue)"/> returns cannot). The tensor's shape is
        ///     <c>value.GetTensorTypeAndShape().Shape</c>.
        /// </summary>
        /// <param name="tensor">A string tensor.</param>
        /// <returns>The strings in row-major order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tensor"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a tensor, or not a string tensor (a numeric tensor reads with <see cref="ToNDArray(OrtValue)"/>).</exception>
        public static string[] ReadStringTensor(this OrtValue tensor)
        {
            if (tensor is null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
                throw new NotSupportedException($"ReadStringTensor needs a tensor OrtValue, but this value is {tensor.OnnxType}.");
            if (!tensor.GetTensorTypeAndShape().IsString)
                throw new NotSupportedException(
                    "ReadStringTensor is for STRING tensors only; a numeric tensor reads back as an NDArray via ToNDArray / AsNDArray.");
            return tensor.GetStringTensorAsArray();
        }
    }
}
