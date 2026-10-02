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
        // sequence of tensors, and STRING tensors carry labels/text. NumSharp has no map/sequence/string
        // dtype, so these come back as NumSharp-shaped pieces (NDArrays for the tensor parts, string[] for
        // strings) through the readers below rather than through ToNDArray, which needs a dense tensor.
        //
        // All readers COPY (like ToNDArray): the intermediate element OrtValues that GetValue(i, allocator)
        // hands out are disposed inside, so nothing here holds an ORT lease — the returned NDArrays own
        // their buffers and the source OrtValue may be disposed the moment the reader returns.

        /// <summary>
        ///     Read a SEQUENCE <see cref="OrtValue"/> whose elements are dense tensors into owning
        ///     <see cref="NDArray"/> copies — e.g. a <c>SequenceConstruct</c> / <c>SplitToSequence</c> output.
        ///     Each element follows <see cref="ToNDArray(OrtValue)"/> (dtype from the tensor, C-contiguous copy).
        /// </summary>
        /// <exception cref="NotSupportedException">
        ///     Not a sequence, or an element is not a dense tensor (a sequence of MAPs — the ZipMap shape — is
        ///     read with <see cref="ToMaps"/> instead).
        /// </exception>
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
            for (int i = 0; i < count; i++)
            {
                using OrtValue element = sequence.GetValue(i, allocator);
                if (element.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
                    throw new NotSupportedException(
                        $"ToNDArrays: sequence element {i} is {element.OnnxType}, not a dense tensor. A sequence of maps " +
                        "(e.g. a scikit-learn ZipMap output) is read with ToMaps.");
                result[i] = element.ToNDArray();
            }
            return result;
        }

        /// <summary>
        ///     Read a MAP <see cref="OrtValue"/> as two parallel owning arrays: <paramref name="keys"/> and the
        ///     <paramref name="values"/> at the same positions (ORT stores a map as a keys tensor + a values
        ///     tensor). Key dtype follows the map (int64 or — for a string-keyed map — refused, since NumSharp
        ///     has no string dtype: read those with <see cref="ReadStringTensor"/> on <see cref="GetMapKeys"/>).
        /// </summary>
        /// <exception cref="NotSupportedException">Not a map (or a string-keyed map — its keys have no NumSharp dtype).</exception>
        public static (NDArray keys, NDArray values) ToMap(this OrtValue map)
        {
            if (map is null)
                throw new ArgumentNullException(nameof(map));
            if (map.OnnxType != OnnxValueType.ONNX_TYPE_MAP)
                throw new NotSupportedException($"ToMap needs a map OrtValue, but this value is {map.OnnxType}.");

            OrtAllocator allocator = OrtAllocator.DefaultInstance;
            using OrtValue keys = map.GetValue(0, allocator);     // element 0 = keys tensor
            using OrtValue values = map.GetValue(1, allocator);   // element 1 = values tensor
            if (keys.GetTensorTypeAndShape().IsString)
                throw new NotSupportedException(
                    "ToMap: this map has STRING keys, which have no NumSharp dtype. Read the keys with " +
                    "GetMapKeys(map).ReadStringTensor() and the values with ToMap-of-the-values, or use the OrtValue API directly.");
            return (keys.ToNDArray(), values.ToNDArray());
        }

        /// <summary>
        ///     Read a SEQUENCE of MAPs — the scikit-learn / ZipMap classifier output
        ///     (<c>sequence(map(int64, float))</c>) — as one (keys, values) pair per element. Each pair is what
        ///     <see cref="ToMap"/> returns; typically every map shares the same class-label keys, so
        ///     <c>result[i].values</c> is sample <c>i</c>'s per-class scores.
        /// </summary>
        /// <exception cref="NotSupportedException">Not a sequence, or an element is not a map.</exception>
        public static (NDArray keys, NDArray values)[] ToMaps(this OrtValue sequenceOfMaps)
        {
            if (sequenceOfMaps is null)
                throw new ArgumentNullException(nameof(sequenceOfMaps));
            if (sequenceOfMaps.OnnxType != OnnxValueType.ONNX_TYPE_SEQUENCE)
                throw new NotSupportedException($"ToMaps needs a sequence OrtValue, but this value is {sequenceOfMaps.OnnxType}.");

            int count = sequenceOfMaps.GetValueCount();
            var result = new (NDArray keys, NDArray values)[count];
            OrtAllocator allocator = OrtAllocator.DefaultInstance;
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

        /// <summary>The keys tensor of a MAP <see cref="OrtValue"/> as a standalone value — for a string-keyed map, feed it to <see cref="ReadStringTensor"/>.</summary>
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
        ///     Read a STRING tensor <see cref="OrtValue"/> as a flat <see cref="string"/>[] in row-major order —
        ///     NumSharp has no string/object dtype, so string outputs (label heads, tokenizers) come back as a
        ///     managed array rather than an <see cref="NDArray"/>. Reshape it yourself against the tensor's shape
        ///     if it is multi-dimensional (<c>value.GetTensorTypeAndShape().Shape</c>).
        /// </summary>
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
