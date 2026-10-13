using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     Every internal promotion engine against the one table the <c>dtype_text</c> oracle tier gates against NumPy
    ///     (<c>np._nptypemap_arr_arr</c>), and the multi-type promotions against NumPy 2.4.2's order-independent
    ///     <c>np.result_type</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Three engines carried rules of their own: <c>NDIterCasting.PromoteTypes</c> (the iterator's COMMON_DTYPE and VIRTUAL
    ///     operands) differed on 30 of the 225 pairs — a float beside a wider integer kept the float (int32 + float32 gave
    ///     float32), bool + int8 gave int16 —, <c>NDExprTypeRules.PromoteStrong</c> (np.evaluate, np.select, np.choose, np.r_)
    ///     typed char beside bool or uint8 as uint16, and the set promotion behind <c>np._FindCommonType</c> kept only the
    ///     LAST pair of a list of four or more types.
    ///     </para>
    ///     <para>
    ///     NumPy's <c>result_type</c> of three or more types is not a left fold of the pairwise promotion:
    ///     <c>int8, uint8, float16</c> is float16 in every order (the highest DType class promotes each other one), where
    ///     folding gives <c>int16</c> then <c>float32</c>. 10 of the 2,197 three-type combinations of NumPy's 13 dtypes differ.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class PromotionEngineParityTests
    {
        private static readonly NPTypeCode[] Codes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
            NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        [TestMethod]
        public void PairEngines_AgreeWithTheTable_OnEveryPair()
        {
            var bad = new List<string>();
            foreach (var a in Codes)
                foreach (var b in Codes)
                {
                    var table = np._nptypemap_arr_arr[(a, b)];
                    if (NDIterCasting.PromoteTypes(a, b) != table)
                        bad.Add($"NDIterCasting.PromoteTypes({a}, {b}) = {NDIterCasting.PromoteTypes(a, b)}, table {table}");
                    if (NDExprTypeRules.PromoteStrong(a, b) != table)
                        bad.Add($"NDExprTypeRules.PromoteStrong({a}, {b}) = {NDExprTypeRules.PromoteStrong(a, b)}, table {table}");
                    if (np._FindCommonType_Array(a, b) != table)
                        bad.Add($"np._FindCommonType_Array({a}, {b}) = {np._FindCommonType_Array(a, b)}, table {table}");
                }
            bad.Should().BeEmpty();
        }

        [TestMethod]
        public void PairEngines_ProbedCases()
        {
            // np.promote_types: int32 + float32 is float64, int16 + float16 float32, bool + int8 int8.
            NDIterCasting.PromoteTypes(NPTypeCode.Int32, NPTypeCode.Single).Should().Be(NPTypeCode.Double);
            NDIterCasting.PromoteTypes(NPTypeCode.Int16, NPTypeCode.Half).Should().Be(NPTypeCode.Single);
            NDIterCasting.PromoteTypes(NPTypeCode.Boolean, NPTypeCode.SByte).Should().Be(NPTypeCode.SByte);
            // char (a uint16 in NumSharp's rules) keeps its own dtype where the uint16 answer is its own width.
            NDExprTypeRules.PromoteStrong(NPTypeCode.Char, NPTypeCode.Boolean).Should().Be(NPTypeCode.Char);
            NDExprTypeRules.PromoteStrong(NPTypeCode.Byte, NPTypeCode.Char).Should().Be(NPTypeCode.Char);
            NDExprTypeRules.PromoteStrong(NPTypeCode.Char, NPTypeCode.UInt16).Should().Be(NPTypeCode.UInt16);
            NDExprTypeRules.PromoteStrong(NPTypeCode.Char, NPTypeCode.SByte).Should().Be(NPTypeCode.Int32);
        }

        [TestMethod]
        public void SetPromotion_IsNumPysOrderIndependentResultType()
        {
            // np.result_type over each permutation (probed): one answer per set.
            void Same(NPTypeCode expected, params NPTypeCode[] set)
            {
                foreach (var perm in Permutations(set))
                {
                    np._FindCommonType_Array(perm).Should().Be(expected, string.Join(", ", perm));
                    NDIterCasting.FindCommonDtype(perm.Select(c => np.zeros(new Shape(1), c)).ToArray(), perm.Length)
                        .Should().Be(expected, string.Join(", ", perm));
                }
            }

            Same(NPTypeCode.Half, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Half);
            Same(NPTypeCode.Single, NPTypeCode.SByte, NPTypeCode.UInt16, NPTypeCode.Half);
            Same(NPTypeCode.Single, NPTypeCode.SByte, NPTypeCode.UInt16, NPTypeCode.Single);
            Same(NPTypeCode.Single, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Half);
            Same(NPTypeCode.Double, NPTypeCode.SByte, NPTypeCode.SByte, NPTypeCode.Double, NPTypeCode.SByte);
            Same(NPTypeCode.Int16, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Boolean, NPTypeCode.SByte);
        }

        [TestMethod]
        public void SetPromotion_FourOrMoreTypes_KeepsEveryPair()
        {
            // The old fold promoted two entries at a time and kept only the last pair's result.
            np._FindCommonType_Array(NPTypeCode.SByte, NPTypeCode.SByte, NPTypeCode.Double, NPTypeCode.SByte).Should().Be(NPTypeCode.Double);
            np._FindCommonType_Array(NPTypeCode.Complex, NPTypeCode.Byte, NPTypeCode.Byte, NPTypeCode.Byte, NPTypeCode.Byte)
                .Should().Be(NPTypeCode.Complex);
            np.find_common_type(new[] { "i1", "i1", "f8", "i1" }).Should().Be(np.float64);
        }

        [TestMethod]
        public void SetPromotion_RefusesTheStringLane()
        {
            Action act = () => np._FindCommonType_Array(NPTypeCode.Int32, NPTypeCode.String, NPTypeCode.Double);
            act.Should().Throw<NotSupportedException>().WithMessage("This operation is not supported for fixed-width string arrays*");
        }

        [TestMethod]
        public void NDIter_VirtualOperand_TakesTheOrderIndependentCommonDtype()
        {
            // A VIRTUAL operand (without ALLOCATE) discards its requested dtype for np.result_type over the real operands,
            // which is float16 for int8, uint8, float16 in every order (a left fold of the pairwise promotion gave float32
            // for this order).
            foreach (var order in new[]
                     {
                         new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Half },
                         new[] { NPTypeCode.Half, NPTypeCode.SByte, NPTypeCode.Byte },
                         new[] { NPTypeCode.Byte, NPTypeCode.Half, NPTypeCode.SByte },
                     })
            {
                var ops = new NDArray[] { null, np.zeros(new Shape(2), order[0]), np.zeros(new Shape(2), order[1]), np.zeros(new Shape(2), order[2]) };
                using var it = NDIterRef.MultiNew(4, ops,
                    NDIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
                    new[]
                    {
                        NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.VIRTUAL, NDIterPerOpFlags.READONLY,
                        NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                    },
                    new[] { NPTypeCode.Int32, NPTypeCode.Empty, NPTypeCode.Empty, NPTypeCode.Empty });
                ops[0].typecode.Should().Be(NPTypeCode.Half, string.Join(", ", order));
            }

            // int32 beside float32 is float64 (the iterator's old pairwise rule kept the float: float32).
            var pair = new NDArray[] { null, np.zeros(new Shape(2), NPTypeCode.Int32), np.zeros(new Shape(2), NPTypeCode.Single) };
            using (var it = NDIterRef.MultiNew(3, pair,
                       NDIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
                       new[] { NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.VIRTUAL, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY }))
            {
                pair[0].typecode.Should().Be(NPTypeCode.Double);
            }
        }

        /// <summary>Every ordering of <paramref name="items"/>.</summary>
        /// <param name="items">The items.</param>
        /// <returns>The permutations.</returns>
        private static IEnumerable<NPTypeCode[]> Permutations(NPTypeCode[] items)
        {
            if (items.Length <= 1)
            {
                yield return items;
                yield break;
            }
            for (int i = 0; i < items.Length; i++)
            {
                var rest = items.Where((_, j) => j != i).ToArray();
                foreach (var tail in Permutations(rest))
                    yield return new[] { items[i] }.Concat(tail).ToArray();
            }
        }
    }
}
