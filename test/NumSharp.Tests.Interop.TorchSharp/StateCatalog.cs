using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     Every array and tensor STATE the state-matrix suites cross, rebuilt by name with the same recipe
    ///     <c>test/oracle/gen_torch_interop_states.py</c> used with NumPy and PyTorch, plus what the bridge is expected to do
    ///     with each one.
    /// </summary>
    /// <remarks>
    ///     <para><b>Array states</b> cover what decides an export: every NumPy flag (C / F contiguity, OWNDATA, WRITEABLE,
    ///     ALIGNED), offsets, steps, negative strides (also on a length-1 axis, where they do not matter), broadcasts over
    ///     writeable and over read-only memory, 0-d and empty arrays, rank 8, overlapping windows, reinterpreting views
    ///     (<c>view(dtype)</c>, a complex array's <c>.real</c> / <c>.imag</c>), memory NumSharp does not own
    ///     (<c>np.frombuffer</c>) and the matrix subclass.</para>
    ///     <para><b>Tensor states</b> cover what decides an import: contiguous, transposed, permuted, narrowed, stepped and
    ///     selected views (storage offsets), 0-d tensors and 0-d views, expanded (stride-0) and overlapping layouts, channels-last,
    ///     a length-1 axis with an odd stride, rank 8, empty tensors and views, tensors that require grad (leaf and not), pending
    ///     conjugate and negative bits, and the dtypes NumSharp widens or names by ordinal.</para>
    ///     <para>Every recipe leaves its intermediates to the caller's TorchSharp <c>DisposeScope</c> (tensors) or to the GC
    ///     (arrays: a view keeps its base alive through its own reference).</para>
    /// </remarks>
    public static class StateCatalog
    {
        // ===================================  dtypes  ===================================

        /// <summary>The ten dtypes NumSharp and torch share, by NumPy name (the fixture's keys).</summary>
        public static readonly string[] SharedDtypes =
            { "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex128" };

        /// <summary>The NumSharp dtype of a NumPy dtype name.</summary>
        /// <param name="name">A name from the fixture.</param>
        /// <returns>The dtype.</returns>
        public static DType DTypeOf(string name) => np.dtype(name);

        /// <summary>The torch dtype of a torch dtype name from the fixture (libtorch's unsigned types by c10 ordinal).</summary>
        /// <param name="name">A name from the fixture.</param>
        /// <returns>The dtype.</returns>
        /// <exception cref="ArgumentException">A name the catalogue does not know.</exception>
        public static ScalarType ScalarTypeOf(string name) => name switch
        {
            "bool" => ScalarType.Bool,
            "uint8" => ScalarType.Byte,
            "int8" => ScalarType.Int8,
            "int16" => ScalarType.Int16,
            "int32" => ScalarType.Int32,
            "int64" => ScalarType.Int64,
            "float16" => ScalarType.Float16,
            "float32" => ScalarType.Float32,
            "float64" => ScalarType.Float64,
            "complex64" => ScalarType.ComplexFloat32,
            "complex128" => ScalarType.ComplexFloat64,
            "bfloat16" => ScalarType.BFloat16,
            "uint16" => (ScalarType)27,
            "uint32" => (ScalarType)28,
            "uint64" => (ScalarType)29,
            _ => throw new ArgumentException($"unknown torch dtype '{name}'", nameof(name)),
        };

        /// <summary>
        ///     Whether the loaded libtorch can make tensors of a fixture dtype: its unsigned 16/32/64-bit types exist from
        ///     libtorch 2.3 (TorchSharp 0.103) on.
        /// </summary>
        /// <param name="name">A torch dtype name from the fixture.</param>
        /// <returns><c>false</c> for uint16 / uint32 / uint64 on an older TorchSharp.</returns>
        public static bool TorchHas(string name)
            => !(name is "uint16" or "uint32" or "uint64") || typeof(torch).Assembly.GetName().Version.Minor >= 103;

        // ===================================  array states  ===================================

        /// <summary>
        ///     A fresh, OWNING, C-contiguous array of the dtype and shape: 0, 1, 2, ... (a bool alternates, a complex value
        ///     carries the imaginary part i + 0.5) - the generator's <c>ramp</c>.
        /// </summary>
        /// <param name="dtype">A NumPy dtype name.</param>
        /// <param name="shape">The dimensions (none for 0-d).</param>
        /// <returns>The array.</returns>
        public static NDArray Ramp(string dtype, params long[] shape)
        {
            long n = 1;
            foreach (long d in shape)
                n *= d;
            NDArray flat;
            if (dtype == "complex128")
            {
                var values = new Complex[n];
                for (long i = 0; i < n; i++)
                    values[i] = new Complex(i, i + 0.5);
                flat = np.array(values);
            }
            else
            {
                using NDArray ints = np.arange(n);
                using NDArray alternating = dtype == "bool" ? ints % 2 : null;
                flat = (alternating ?? ints).astype(DTypeOf(dtype), copy: true);
            }
            using (flat)
            using (NDArray shaped = shape.Length == 0 ? flat.reshape(new Shape()) : flat.reshape(shape))
                return shaped.copy();   // the copy owns its buffer, as the generator's astype does
        }

        /// <summary>Clears WRITEABLE on an array (NumPy's <c>a.flags.writeable = False</c>).</summary>
        /// <param name="nd">The array.</param>
        /// <returns>The same array, read-only.</returns>
        private static NDArray ReadOnly(NDArray nd)
        {
            nd.setflags(write: false);
            return nd;
        }

        /// <summary>
        ///     Four elements of the dtype at an address its alignment forbids, viewed from a fresh uint8 buffer - NumPy's state
        ///     <c>raw[1:...].view(dt)</c> (ALIGNED false).
        /// </summary>
        /// <remarks>
        ///     The byte offset is found at run time: NumPy's allocator aligns every buffer, so offset 1 is misaligned there, but
        ///     NumSharp's guard-page mode (<c>NUMSHARP_DEBUG_GUARD_PAGES=1</c>) ends each buffer at a page boundary instead, which
        ///     made one byte in exactly aligned for every dtype and turned the state into another one.
        /// </remarks>
        /// <param name="dtype">A NumPy dtype name wider than one byte.</param>
        /// <returns>A writeable, contiguous view whose data address is not a multiple of the itemsize.</returns>
        private static unsafe NDArray Misaligned(string dtype)
        {
            int size = ItemSize(dtype);
            NDArray raw = np.zeros(new Shape(9 * size), np.uint8);
            long address = (long)raw.Storage.InternalArray.Address;
            int k = 1;
            while ((address + k) % size == 0)
                k++;
            return raw[$"{k}:{k + 4 * size}"].view(DTypeOf(dtype));
        }

        /// <summary>The element size of a NumPy dtype name.</summary>
        /// <param name="dtype">The name.</param>
        /// <returns>Its itemsize in bytes.</returns>
        private static int ItemSize(string dtype) => DTypeOf(dtype).itemsize;

        /// <summary>The array state recipes, by the generator's names: each builds the state for a dtype name.</summary>
        public static readonly IReadOnlyDictionary<string, Func<string, NDArray>> ArrayStates = new Dictionary<string, Func<string, NDArray>>(StringComparer.Ordinal)
        {
            ["c_1d"] = d => Ramp(d, 7),
            ["c_2d"] = d => Ramp(d, 4, 5),
            ["c_3d"] = d => Ramp(d, 2, 3, 4),
            ["f_2d_owner"] = d => np.asfortranarray(Ramp(d, 4, 5)),
            ["f_3d_view"] = d => Ramp(d, 2, 3, 4).transpose(2, 1, 0),
            ["transposed_2d"] = d => Ramp(d, 4, 5).T,
            ["transposed_3d"] = d => Ramp(d, 2, 3, 4).transpose(2, 0, 1),
            ["offset_1d"] = d => Ramp(d, 10)["2:7"],
            ["stepped_1d"] = d => Ramp(d, 10)["::2"],
            ["stepped_cols_2d"] = d => Ramp(d, 4, 6)[":, ::2"],
            ["stepped_rows_2d"] = d => Ramp(d, 6, 4)["::2, :"],
            ["stepped_offset_3d"] = d => Ramp(d, 4, 5, 6)["1:, 1::2, 2:"],
            ["sliced_composed"] = d => Ramp(d, 6, 4)["1:5"].T,
            ["negstride_1d"] = d => Ramp(d, 8)["::-1"],
            ["negstride_2d"] = d => Ramp(d, 4, 5)["::-1, ::-1"],
            ["negstride_inner_2d"] = d => Ramp(d, 4, 5)[":, ::-1"],
            ["negstride_len1_axis"] = d => Ramp(d, 3, 1, 4)[":, ::-1, :"],
            ["singleton_3d"] = d => Ramp(d, 3, 1, 4),
            ["newaxis"] = d => np.expand_dims(Ramp(d, 5), 0),
            ["reshape_view"] = d => Ramp(d, 24).reshape(4, 6),
            ["highrank_8d"] = d => Ramp(d, 2, 2, 2, 2, 2, 2, 2, 2),
            ["highrank_8d_permuted"] = d => Ramp(d, 2, 2, 2, 2, 2, 2, 2, 2).transpose(1, 0, 3, 2, 5, 4, 7, 6),
            ["zerod_owner"] = d => np.zeros(new Shape(), DTypeOf(d)),
            ["zerod_view"] = d => Ramp(d, 24)["23:24"].reshape(new Shape()),
            ["one_element_1d"] = d => Ramp(d, 1),
            ["empty_2d"] = d => np.zeros(new Shape(0, 3), DTypeOf(d)),
            ["empty_view"] = d => Ramp(d, 4, 6)["2:2"],
            ["empty_composed"] = d => np.zeros(new Shape(0, 4), DTypeOf(d))["::2, :"],
            ["broadcast_row"] = d => np.broadcast_to(Ramp(d, 5), new Shape(4, 5)),
            ["broadcast_col"] = d => np.broadcast_to(np.expand_dims(Ramp(d, 4), 1), new Shape(4, 5)),
            ["broadcast_strided_base"] = d => np.broadcast_to(Ramp(d, 10)["::2"], new Shape(3, 5)),
            ["broadcast_of_readonly"] = d => np.broadcast_to(ReadOnly(Ramp(d, 6)), new Shape(2, 6)),
            ["readonly_owner"] = d => ReadOnly(Ramp(d, 6)),
            ["readonly_view"] = d => ReadOnly(Ramp(d, 6))["1:"],
            ["aligned_flag_cleared"] = d =>
            {
                NDArray a = Ramp(d, 6);
                a.setflags(align: false);
                return a;
            },
            ["misaligned"] = d => Misaligned(d),
            ["overlapping_windows"] = d => np.ndarray(new Shape(3, 4), DTypeOf(d), Ramp(d, 10), 0, new long[] { ItemSize(d), ItemSize(d) }),
            ["matrix"] = d => np.asmatrix(Ramp(d, 2, 3)),
            ["matrix_T"] = d => np.asmatrix(Ramp(d, 2, 3)).T,
            ["frombuffer"] = d => np.frombuffer(new byte[6 * ItemSize(d)], DTypeOf(d)),
            ["complex_real"] = d => Ramp("complex128", 6).real,
            ["complex_imag"] = d => Ramp("complex128", 6).imag,
            ["complex_8mod16"] = d => Ramp("float64", 10)["1:9"].view(np.complex128),
            ["view_int64_of_float64"] = d => Ramp("float64", 6).view(np.int64),
            ["view_uint8_of_float64"] = d => Ramp("float64", 3).view(np.uint8),
        };

        /// <summary>
        ///     What keeps the bridge from sharing an array state's memory with torch - the reason its View refusal names - or
        ///     null when the bridge shares it (Auto and View) - decided from the array itself, by the bridge's documented rules.
        /// </summary>
        /// <remarks>
        ///     Where this differs from <c>torch.from_numpy</c> on purpose: a read-only array (from_numpy shares it with a warning),
        ///     a broadcast over read-only memory (the same), misaligned data (from_numpy shares any address) and complex128 data
        ///     off a 16-byte boundary (libtorch declares <c>c10::complex&lt;double&gt;</c> alignas(16)) are copied; a negative
        ///     stride on a length-1 axis is shared (from_numpy refuses it, although the axis is never stepped).
        /// </remarks>
        /// <param name="nd">The array.</param>
        /// <returns>The substring the refusal message carries, or null.</returns>
        public static unsafe string ExportBlocker(NDArray nd)
        {
            Shape shape = nd.Shape;
            for (int i = 0; i < shape.NDim; i++)
                if (shape.dimensions[i] > 1 && shape.strides[i] < 0)
                    return "negative stride";
            if (!shape.IsWriteable)
            {
                if (!shape.IsBroadcasted)
                    return "is read-only";
                if (nd.Storage.BaseStorage is null || !nd.Storage.BaseStorage.Shape.IsWriteable)
                    return "broadcast of read-only memory";
            }
            long align = nd.typecode == NPTypeCode.Complex ? 16 : nd.dtypesize;
            long address = (long)((byte*)nd.Storage.InternalArray.Address + shape.offset * nd.dtypesize);
            if (nd.size > 0 && address % align != 0)
                return "not aligned";
            return null;
        }

        // ===================================  tensor states  ===================================

        /// <summary>
        ///     A fresh contiguous tensor of the dtype and shape: 0, 1, 2, ... (a bool alternates, a complex value carries the
        ///     imaginary part i + 0.5) - the generator's <c>tramp</c>. Its intermediates join the caller's DisposeScope.
        /// </summary>
        /// <param name="dtype">The torch dtype.</param>
        /// <param name="shape">The dimensions (none for 0-d).</param>
        /// <returns>The tensor.</returns>
        public static Tensor TorchRamp(ScalarType dtype, params long[] shape)
        {
            long n = 1;
            foreach (long d in shape)
                n *= d;
            Tensor t = torch.arange(n, dtype: ScalarType.Float64);
            if (dtype == ScalarType.Bool)
                t = t.remainder(2);
            if (dtype == ScalarType.ComplexFloat32 || dtype == ScalarType.ComplexFloat64)
                t = torch.complex(t, t + 0.5);
            return t.reshape(shape).to_type(dtype).contiguous();
        }

        /// <summary>The tensor state recipes, by the generator's names: each builds the state for a torch dtype.</summary>
        public static readonly IReadOnlyDictionary<string, Func<ScalarType, Tensor>> TensorStates = new Dictionary<string, Func<ScalarType, Tensor>>(StringComparer.Ordinal)
        {
            ["contiguous_3d"] = d => TorchRamp(d, 2, 3, 4),
            ["transposed_2d"] = d => TorchRamp(d, 4, 5).t(),
            ["permuted_3d"] = d => TorchRamp(d, 2, 3, 4).permute(2, 0, 1),
            ["narrowed"] = d => TorchRamp(d, 4, 6).narrow(1, 1, 3),
            ["stepped"] = d => TorchRamp(d, 4, 6)[TensorIndex.Colon, TensorIndex.Slice(null, null, 2)],
            ["stepped_offset_3d"] = d => TorchRamp(d, 4, 5, 6)[TensorIndex.Slice(1), TensorIndex.Slice(1, null, 2), TensorIndex.Slice(2)],
            ["select"] = d => TorchRamp(d, 3, 4)[1],
            ["zerod_select"] = d => TorchRamp(d, 3, 4)[1, 2],
            ["zerod_tensor"] = d => TorchRamp(d),
            ["one_element"] = d => TorchRamp(d, 1),
            ["unsqueezed"] = d => TorchRamp(d, 5).unsqueeze(0),
            ["expanded_row"] = d => TorchRamp(d, 5).expand(new long[] { 4, 5 }),
            ["expanded_col"] = d => TorchRamp(d, 4).unsqueeze(1).expand(new long[] { 4, 5 }),
            ["overlapping_windows"] = d => TorchRamp(d, 10).as_strided(new long[] { 3, 4 }, new long[] { 1, 1 }, 0L),
            ["channels_last"] = d => TorchRamp(d, 2, 3, 4, 5).permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2),
            ["size1_odd_stride"] = d => TorchRamp(d, 2, 3).unsqueeze(1).as_strided(new long[] { 2, 1, 3 }, new long[] { 3, 100, 1 }, 0L),
            ["highrank_8d_permuted"] = d => TorchRamp(d, 2, 2, 2, 2, 2, 2, 2, 2).permute(1, 0, 3, 2, 5, 4, 7, 6),
            ["empty_3d"] = d => TorchRamp(d, 2, 0, 3),
            ["empty_view"] = d => TorchRamp(d, 4, 6)[TensorIndex.Colon, TensorIndex.Slice(3, 3)],
            ["requires_grad_leaf"] = d => TorchRamp(d, 2, 3).requires_grad_(),
            ["requires_grad_nonleaf"] = d => TorchRamp(d, 2, 3).requires_grad_() * 2,
            ["conj"] = d => TorchRamp(d, 2, 3).conj(),
            ["conj_imag_neg"] = d => TorchRamp(d, 2, 3).conj().imag,
            ["transposed_widened"] = d => TorchRamp(d, 3, 4).t(),
        };

        /// <summary>
        ///     What keeps the bridge from viewing a tensor state's own memory - the reason its View refusal names - or null when
        ///     the bridge views it (Auto and View). The bridge's rules: a pending conjugation or negation, and a dtype NumSharp
        ///     must widen, need a copy; a CPU tensor of any strides does not.
        /// </summary>
        /// <param name="t">The tensor.</param>
        /// <returns>The substring the refusal message carries, or null.</returns>
        public static string ImportBlocker(Tensor t)
        {
            if (t.is_conj())
                return "pending conjugation";
            if (t.is_neg())
                return "pending negation";
            if (t.dtype == ScalarType.ComplexFloat32 || t.dtype == ScalarType.BFloat16)
                return "widened";
            return null;
        }

        /// <summary>
        ///     The flags of an owning, C-contiguous copy of the given shape: NumPy's C rule, and F-contiguous only when at most one
        ///     dimension exceeds 1 (or nothing is held).
        /// </summary>
        /// <param name="shape">The copy's dimensions.</param>
        /// <returns>The flags NumPy gives such a fresh array.</returns>
        public static StateOracle.Flags OwningCopyFlags(long[] shape)
        {
            bool empty = shape.Any(d => d == 0);
            bool f = empty || shape.Count(d => d > 1) <= 1;
            return new StateOracle.Flags(C: true, F: f, O: true, W: true, A: true);
        }
    }
}
