using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     <see cref="DTypeCasting.GetDTypeTransferFunction"/> (NumPy's <c>PyArray_GetDTypeTransferFunction</c>) and the
    ///     strided loops of the storage classes' casting impls (<see cref="ArrayMethod.get_strided_loop"/>): every loop, run
    ///     over NumPy's raw assignment walk (<see cref="UnmanagedStorage.RawArrayAssignArray"/>), must give the bytes - or the
    ///     exception - <c>astype</c> gives for the same pair, because the callers that take their loop from
    ///     <see cref="DTypeCasting"/> instead of choosing a converter by name must not change a value.
    /// </summary>
    [TestClass]
    public class DTypeTransferFunctionTests
    {
        /// <summary>The source arrays: numbers of every NumPy dtype (special values in the floats), and text that parses,
        ///     text that does not, the empty string and non-ASCII text, in each text class.</summary>
        private static IEnumerable<(string Label, NDArray Array)> Sources()
        {
            var plain = new[] { 3.0, -1.0, 2.0, 0.0, 5.0, -4.0, 1.0, 7.0 };
            var special = new[] { 3.5, double.PositiveInfinity, -0.0, double.NaN, 2.25, -1.5, double.NegativeInfinity, 1e300 };
            foreach (var dt in new[] { "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16",
                         "float32", "float64", "complex128" })
            {
                yield return (dt + "/plain", np.array(plain).astype(np.dtype(dt)));
                if (dt.StartsWith("float") || dt == "complex128")
                    yield return (dt + "/special", np.array(special).astype(np.dtype(dt)));
            }
            var parses = new[] { "3", "-1", "2", "0", "5", "-4", "1", "7" };
            var mixed = new[] { "3.5", "x", "", "nan", "-1", "ab", "1e3", "inf" };
            var ascii = new[] { "abc", "", "Hello", "a b", "12", "-", "zz", "q" };
            var nonAscii = new[] { "ok", "é", "x" };
            foreach (var cls in new[] { "U", "S" })
            {
                foreach (var (label, texts) in new[] { ("parses", parses), ("mixed", mixed), ("ascii", ascii) })
                    yield return ($"{cls}5/{label}", np.array(texts).astype(np.dtype(cls + "5")));
            }
            yield return ("U5/nonascii", np.array(nonAscii).astype(np.dtype("U5")));
            foreach (var (label, texts) in new[] { ("parses", parses), ("mixed", mixed), ("ascii", ascii), ("nonascii", nonAscii) })
                yield return ($"T/{label}", np.array(texts).astype(np.dtypes.StringDType()));
        }

        /// <summary>The destination dtypes: every NumPy number, bool, the two text classes at three widths, StringDType.</summary>
        private static readonly string[] Targets =
        {
            "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64",
            "complex128", "U5", "U3", "U8", "S5", "S3", "S8", "T",
        };

        /// <summary>The descriptor of a <see cref="Targets"/> label.</summary>
        private static DType Target(string label) => label == "T" ? np.dtypes.StringDType() : np.dtype(label);

        /// <summary>
        ///     The outcome of a conversion as one comparable line: the array's dtype and its repr (the values, StringDType's
        ///     included), or the exception's type and message.
        /// </summary>
        private static string Outcome(Func<NDArray> convert)
        {
            try
            {
                using var result = convert();
                return result.dtype + " " + result.ToString(true);
            }
            catch (Exception e)
            {
                return "EXC " + e.GetType().Name + ": " + e.Message;
            }
        }

        /// <summary>
        ///     The conversion through the transfer function: a fresh destination of the source's shape, the loop of
        ///     <see cref="DTypeCasting.GetDTypeTransferFunction"/> run over the raw walk.
        /// </summary>
        private static NDArray ThroughLoop(NDArray src, DType to)
        {
            var dst = np.empty(new Shape((long[])src.Shape.dimensions.Clone()), to);
            try
            {
                var cast = DTypeCasting.GetDTypeTransferFunction(true, src.Storage.ItemBytes, dst.Storage.ItemBytes, src.dtype, dst.dtype,
                    false, out _,
                    src.Storage.IsVString ? src.Storage.VStringSlice.Allocator : null,
                    dst.Storage.IsVString ? dst.Storage.VStringSlice.Allocator : null);
                UnmanagedStorage.RawArrayAssignArray(dst.Storage, src.Storage, null, cast.func);
                return dst;
            }
            catch
            {
                dst.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Every source against every target: the loop gives astype's result, value for value, or astype's exception
        ///     (type and message). This is the contract the batch's callers rely on when they stop choosing a converter by
        ///     the dtypes' names.
        /// </summary>
        [TestMethod]
        public void EveryStorageLoop_GivesAstypesOutcome()
        {
            var mismatches = new List<string>();
            int compared = 0;
            foreach (var (label, src) in Sources())
            {
                foreach (var t in Targets)
                {
                    var to = Target(t);
                    string expected = Outcome(() => src.astype(to));
                    string actual = Outcome(() => ThroughLoop(src, to));
                    compared++;
                    if (expected != actual)
                        mismatches.Add($"{label} -> {t}:\n  astype: {expected}\n  loop:   {actual}");
                }
                src.Dispose();
            }
            compared.Should().BeGreaterThan(500);
            mismatches.Should().BeEmpty(string.Join("\n", mismatches.Take(10)));
        }

        /// <summary>
        ///     A loop runs over any stride: a reversed and a stepped source give the bytes of their contiguous copies (the text
        ///     loops are called per run by the iterator's buffers, whose array side has the operand's own strides).
        /// </summary>
        [TestMethod]
        public void TextLoops_RunOverAnyStride()
        {
            var texts = np.array(new[] { "3", "-1", "2", "0", "5", "-4", "1", "7" });
            foreach (var view in new[] { texts["::-1"], texts["::2"], texts["1::3"] })
            {
                using var dense = view.copy();
                foreach (var t in new[] { "float64", "int16", "bool", "S3", "U8", "T" })
                    Outcome(() => ThroughLoop(view, Target(t))).Should().Be(Outcome(() => ThroughLoop(dense, Target(t))), t);
            }
        }

        /// <summary>
        ///     The descriptors resolved for the loop are the given ones (NumSharp's storage is native, so NumPy's multi-step
        ///     wrapping is never needed), and the context carries the StringDType sides' allocators.
        /// </summary>
        [TestMethod]
        public void CastInfo_HoldsTheGivenDescriptorsAndAllocators()
        {
            var u5 = np.dtype("U5");
            var info = DTypeCasting.GetDTypeTransferFunction(true, 20, 8, u5, np.float64, false, out var flags);
            info.descriptors[0].Should().BeSameAs(u5);
            info.descriptors[1].Should().BeSameAs(np.float64);
            info.context.method.Should().BeSameAs(DTypeCasting.GetCastingImpl(u5.Meta, np.float64.Meta));
            info.context.Allocators.Should().BeNull();
            flags.Should().Be(info.context.method.flags);

            using var t = np.array(new[] { "a" }).astype(np.dtypes.StringDType());
            var allocator = t.Storage.VStringSlice.Allocator;
            var toText = DTypeCasting.GetDTypeTransferFunction(true, 16, 20, t.dtype, u5, false, out _, allocator, null);
            toText.context.Allocators.Should().Equal(allocator, null);
        }

        /// <summary>
        ///     The refusals: a datetime cast has no storage before Stage C (its impl keeps the base loop, which raises), and a
        ///     StringDType side without its allocator cannot be read.
        /// </summary>
        [TestMethod]
        public void Refusals()
        {
            new Action(() => DTypeCasting.GetDTypeTransferFunction(true, 8, 8, np.dtype("M8[s]"), np.int64, false, out _))
                .Should().Throw<NotSupportedException>().WithMessage("*Stage C*");
            new Action(() => DTypeCasting.GetDTypeTransferFunction(true, 16, 8, np.dtypes.StringDType(), np.float64, false, out _))
                .Should().Throw<ArgumentException>().WithMessage("*allocator*");
            new Action(() => DTypeCasting.GetDTypeTransferFunction(true, 8, 8, null, np.int64, false, out _))
                .Should().Throw<ArgumentNullException>();
        }
    }
}
