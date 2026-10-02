using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The C# spellings of NumPy's array_like arguments across the <c>numpy.polynomial</c> package — what the byte
    ///     oracles (<c>polycalc</c> / <c>polyseries</c> / <c>polyeval</c>.jsonl, which replay only <c>object[]</c>,
    ///     <c>ValueTuple</c>, Python scalars and NDArrays) cannot express: jagged arrays, <c>List&lt;T&gt;</c> and LINQ,
    ///     <c>NDArray[]</c>, <c>Memory&lt;T&gt;</c>, multi-dimensional <c>object[,]</c>, <c>BigInteger[]</c>,
    ///     <c>System.Tuple</c>, the 8-item <c>ValueTuple</c> (whose last item nests in TRest), and the argument-order facts
    ///     of the sequence conversion (NDPolySequence). Every expected value is NumPy 2.4.2's, the Python spelling next to
    ///     it.
    /// </summary>
    [TestClass]
    public class PolynomialArgumentKindsTests
    {
        private static readonly ChebyshevModule C = np.polynomial.chebyshev;
        private static readonly PowerSeriesModule P = np.polynomial.polynomial;
        private static readonly PolyUtilsModule U = np.polynomial.polyutils;

        /// <summary>
        ///     Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump — bit parity.
        /// </summary>
        /// <param name="actual">The result.</param>
        /// <param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="hex">NumPy's <c>tobytes().hex()</c> of the C-order copy.</param>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex)
        {
            actual.dtype.name.Should().Be(dtype);
            actual.shape.Should().Equal(shape);
            Hex(actual).Should().Be(hex);
        }

        /// <summary>The logical-C-order bytes of <paramref name="a"/> as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex text.</returns>
        private static string Hex(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }

        /// <summary>A Python list: <c>params</c> wraps the items (a ONE-item list of a list must be spelled explicitly —
        ///     <c>L(someObjectArray)</c> would hand the array itself over as the params array).</summary>
        /// <param name="items">The items.</param>
        /// <returns>The list.</returns>
        private static object[] L(params object[] items) => items;

        /// <summary>NumPy's <c>[1.0, 2.0, 3.0]</c> series.</summary>
        private static NDArray C64() => np.array(new[] { 1.0, 2.0, 3.0 });

        // =========================================================================================================
        //  C# collection spellings of a coefficient series
        // =========================================================================================================

        /// <summary>
        ///     A jagged <c>double[][]</c> is a Python list of ndarrays, stacked like <c>np.array([a, b, c])</c> — a 2-D
        ///     series (<c>chebder(np.array([[1.,2.],[3.,4.],[5.,6.]]))</c> and its <c>axis=1</c> form).
        /// </summary>
        [TestMethod]
        public void JaggedArray_IsAListOfArrays()
        {
            var jag = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 }, new[] { 5.0, 6.0 } };
            AssertBytes(C.chebder(jag, 1), "float64", new long[] { 2, 2 }, "0000000000000840000000000000104000000000000034400000000000003840");
            AssertBytes(C.chebder(jag, 1, axis: 1), "float64", new long[] { 3, 1 }, "000000000000004000000000000010400000000000001840");
        }

        /// <summary>
        ///     A <c>List&lt;int&gt;</c> / LINQ enumerable is a Python list of Python numbers (C# ints are Python ints —
        ///     int64 — then the series converts to float64): <c>chebder([1, 2, 3])</c>, <c>chebint([1.0, 4.0, 9.0])</c>.
        /// </summary>
        [TestMethod]
        public void ListAndEnumerable_ArePythonLists()
        {
            AssertBytes(C.chebder(new List<int> { 1, 2, 3 }), "float64", new long[] { 2 }, "00000000000000400000000000002840");
            AssertBytes(C.chebint(Enumerable.Range(1, 3).Select(i => (double)(i * i))), "float64", new long[] { 4 },
                "000000000000f03f0000000000000cc0000000000000f03f000000000000f83f");
        }

        /// <summary>
        ///     An <c>NDArray[]</c> is a Python list of ndarrays; two float32 rows stack into a float32 series
        ///     (<c>chebint((f32([1,2]), f32([3,4])))</c>).
        /// </summary>
        [TestMethod]
        public void NDArrayArray_StacksLikeAList()
        {
            var rows = new[] { np.array(new[] { 1f, 2f }), np.array(new[] { 3f, 4f }) };
            AssertBytes(C.chebint(rows, 1), "float32", new long[] { 3, 2 }, "0000403f0000803f0000803f000000400000403f0000803f");
        }

        /// <summary>
        ///     A <c>Memory&lt;float&gt;</c> is an ndarray of its element dtype, like a <c>float[]</c>
        ///     (<c>chebder(np.array([1, 2, 3], np.float32))</c>).
        /// </summary>
        [TestMethod]
        public void MemoryOfT_IsAnNdarray()
        {
            var mem = new Memory<float>(new[] { 1f, 2f, 3f });
            AssertBytes(C.chebder(mem), "float32", new long[] { 2 }, "0000004000004041");
        }

        /// <summary>
        ///     A multi-dimensional <c>object[,]</c> is a list of rows of Python numbers — dtype discovered over every item
        ///     (<c>chebder([[1, 2.5], [3, 4]], axis=1)</c> is float64).
        /// </summary>
        [TestMethod]
        public void MultiDimensionalObjectArray_IsAListOfRows()
        {
            var rows = new object[,] { { 1, 2.5 }, { 3, 4 } };
            AssertBytes(C.chebder(rows, 1, axis: 1), "float64", new long[] { 2, 1 }, "00000000000004400000000000001040");
        }

        /// <summary>
        ///     A <c>BigInteger[]</c> is a list of Python ints: <c>[2**63, -1]</c> discovers uint64 and int64, promoted to
        ///     float64; a Python int past uint64 would make NumPy an OBJECT array, a dtype NumSharp does not have.
        /// </summary>
        [TestMethod]
        public void BigIntegerArray_IsAListOfPythonInts()
        {
            AssertBytes(C.chebder(new[] { BigInteger.Pow(2, 63), BigInteger.MinusOne }), "float64", new long[] { 1 }, "000000000000f0bf");
            Action objectArray = () => C.chebder(new[] { BigInteger.Pow(2, 70), BigInteger.One });
            objectArray.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     A <c>System.Tuple</c> (like a <c>ValueTuple</c>) is a Python tuple — a sequence, not a scalar
        ///     (<c>chebint((1.0, 2.0, 3.0))</c>); an 8-item <c>ValueTuple</c>, whose 8th item C# nests in TRest, is still
        ///     8 coefficients (<c>polyder((1, 2, 3, 4, 5, 6, 7, 8))</c>).
        /// </summary>
        [TestMethod]
        public void Tuples_AreSequences()
        {
            AssertBytes(C.chebint(Tuple.Create(1.0, 2.0, 3.0)), "float64", new long[] { 4 },
                "000000000000e03f000000000000e0bf000000000000e03f000000000000e03f");
            AssertBytes(P.polyder((1, 2, 3, 4, 5, 6, 7, 8)), "float64", new long[] { 7 },
                "00000000000000400000000000001840000000000000284000000000000034400000000000003e4000000000000045400000000000004c40");
        }

        // =========================================================================================================
        //  np.array's coercion walk: empty sequences, ragged input, refused items
        // =========================================================================================================

        /// <summary>
        ///     An empty sequence contributes NO dtype (<c>chebder([np.zeros(0, np.float32), []])</c> is float32 (1, 0)),
        ///     and it ends the dims at its depth — so <c>[np.zeros((0, 3)), []]</c> is a (2, 0) array the (0, 3) item cannot
        ///     be assigned into (NumPy's broadcast text).
        /// </summary>
        [TestMethod]
        public void EmptySequence_ContributesNoDtype_AndEndsTheDims()
        {
            AssertBytes(C.chebder(L(np.zeros(new Shape(0), NPTypeCode.Single), L())), "float32", new long[] { 1, 0 }, "");
            Action shrink = () => C.chebder(L(np.zeros(new Shape(0, 3)), L()));
            shrink.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (0,3) into shape (0,)");
        }

        /// <summary>
        ///     A ragged input reports the shape the walk still AGREED on (<c>[[[1], [1, 2]], [[1], [1, 2]]]</c> is
        ///     "(2, 2)", not the inner list's "(2,)"), and raggedness wins over an item NumSharp cannot hold
        ///     (<c>[[1], "ab"]</c> is the ValueError, as in NumPy; <c>[1, "ab"]</c> — a str array in NumPy — is refused).
        /// </summary>
        [TestMethod]
        public void RaggedInput_ReportsNumPysShape_BeforeARefusedItem()
        {
            Action deep = () => C.chebder(L(L(new object[] { 1 }, L(1, 2)), L(new object[] { 1 }, L(1, 2))));
            deep.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence. The requested array has an " +
                                                          "inhomogeneous shape after 2 dimensions. The detected shape was (2, 2) + inhomogeneous part.");
            Action raggedStr = () => C.chebder(L(new object[] { 1 }, "ab"));
            raggedStr.Should().Throw<ValueError>().WithMessage("*after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            Action str = () => C.chebder(L(1, "ab"));
            str.Should().Throw<NotSupportedException>();
        }

        // =========================================================================================================
        //  k / lbnd / scl
        // =========================================================================================================

        /// <summary>
        ///     A tuple <c>k</c> is iterated like a list (<c>chebint(c, 2, k=(1, 2))</c>); a tuple <c>lbnd</c> / <c>scl</c> is
        ///     NOT a scalar for an integral (<c>np.ndim((0,))</c> is 1), while a derivative's <c>scl</c> may be any array
        ///     (<c>chebder(c, scl=(2.0,))</c> broadcasts it).
        /// </summary>
        [TestMethod]
        public void TupleArguments_FollowNumPysSequenceRules()
        {
            AssertBytes(C.chebint(C64(), 2, (1, 2)), "float64", new long[] { 5 },
                "000000000000fb3f000000000000f43f000000000000d0bf555555555555b53f000000000000b03f");
            Action lbnd = () => C.chebint(C64(), lbnd: ValueTuple.Create(0));
            lbnd.Should().Throw<ValueError>().WithMessage("lbnd must be a scalar.");
            Action scl = () => C.chebint(C64(), scl: ValueTuple.Create(2.0));
            scl.Should().Throw<ValueError>().WithMessage("scl must be a scalar.");
            AssertBytes(C.chebder(C64(), scl: ValueTuple.Create(2.0)), "float64", new long[] { 2 }, "00000000000010400000000000003840");
        }

        /// <summary>
        ///     np.ndim of a ragged lbnd is np.asarray's ValueError, raised by NumPy's scalar check itself.
        /// </summary>
        [TestMethod]
        public void RaggedLbnd_RaisesAtTheScalarCheck()
        {
            Action ragged = () => C.chebint(C64(), lbnd: L(new object[] { 1 }, L(1, 2)));
            ragged.Should().Throw<ValueError>().WithMessage("*The detected shape was (2,) + inhomogeneous part.");
        }

        /// <summary>
        ///     A Python int past int64 as lbnd (<see cref="BigInteger"/>) is weak like any Python int — NumPy's value
        ///     (<c>chebint(c, lbnd=2**70)</c>) — and past float64 it is NumPy's OverflowError.
        /// </summary>
        [TestMethod]
        public void BigIntegerLbnd_IsAPythonInt()
        {
            AssertBytes(C.chebint(C64(), lbnd: BigInteger.Pow(2, 70)), "float64", new long[] { 4 },
                "00000000000020cd000000000000e0bf000000000000e03f000000000000e03f");
            Action overflow = () => C.chebint(C64(), lbnd: BigInteger.Pow(2, 1030));
            overflow.Should().Throw<OverflowException>().WithMessage("int too large to convert to float");
        }

        /// <summary>
        ///     A str lbnd passes NumPy's scalar check (<c>np.ndim('a')</c> is 0) and fails only where NumPy EVALUATES at it:
        ///     never for <c>m == 0</c>, never in the one-coefficient zero branch (<c>chebint([0.0], lbnd='a')</c> is
        ///     <c>[0.]</c>); NumPy's str-dtype error is NumSharp's NotSupportedException.
        /// </summary>
        [TestMethod]
        public void StrLbnd_FailsOnlyWhenEvaluated()
        {
            AssertBytes(C.chebint(C64(), 0, lbnd: "a"), "float64", new long[] { 3 }, "000000000000f03f00000000000000400000000000000840");
            AssertBytes(C.chebint(new object[] { 0.0 }, 1, lbnd: "a"), "float64", new long[] { 1 }, "0000000000000000");
            Action evaluated = () => C.chebint(C64(), 1, lbnd: "a");
            evaluated.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     An integration constant is converted only when its order USES it, after that order's lbnd evaluation — so
        ///     a ragged constant loses to the order check (<c>m=-1</c>) and to an lbnd that overflows (NumPy's order).
        /// </summary>
        [TestMethod]
        public void IntegrationConstants_AreConvertedWhenUsed()
        {
            var c = np.ones(new Shape(3, 2));
            object raggedK = new object[] { L(new object[] { 1 }, L(1, 2)) };
            Action mneg = () => C.chebint(c, -1, raggedK);
            mneg.Should().Throw<ValueError>().WithMessage("The order of integration must be non-negative");
            Action lbndFirst = () => C.chebint(c, 1, raggedK, lbnd: BigInteger.Pow(2, 1030));
            lbndFirst.Should().Throw<OverflowException>().WithMessage("int too large to convert to float");
            Action used = () => C.chebint(c, 1, raggedK);
            used.Should().Throw<ValueError>().WithMessage("*inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
        }

        // =========================================================================================================
        //  The shared conversion in the sibling families (U1 polyutils, U3 {p}val)
        // =========================================================================================================

        /// <summary>
        ///     <c>as_series</c> iterates a typed C# array as the ndarray it is: NumPy SCALARS of its dtype, so a
        ///     <c>float[]</c> gives float32 series (<c>as_series(np.array([1, 2], np.float32))</c>), not the float64 of
        ///     boxed Python floats; a tuple of arrays is iterated like a list (and trimmed).
        /// </summary>
        [TestMethod]
        public void AsSeries_IteratesTypedArraysAndTuples()
        {
            var f32 = U.as_series(new[] { 1f, 2f });
            f32.Length.Should().Be(2);
            AssertBytes(f32[0], "float32", new long[] { 1 }, "0000803f");
            AssertBytes(f32[1], "float32", new long[] { 1 }, "00000040");
            var tup = U.as_series((np.array(new[] { 1.0, 2.0, 0.0 }), np.array(new[] { 3.0, 4.0 })));
            AssertBytes(tup[0], "float64", new long[] { 2 }, "000000000000f03f0000000000000040");
            AssertBytes(tup[1], "float64", new long[] { 2 }, "00000000000008400000000000001040");
        }

        /// <summary>
        ///     <c>{p}val</c>'s x: a <see cref="BigInteger"/> is a weak Python int (<c>chebval(2**70, c)</c>), and a nested
        ///     list is <c>np.asarray(x)</c> of it — a 2-D x (<c>chebval([[0.5, 1.0], [2.0, 3.0]], c)</c>).
        /// </summary>
        [TestMethod]
        public void Val_TakesBigIntegerAndNestedListX()
        {
            AssertBytes(C.chebval(BigInteger.Pow(2, 70), C64()), "float64", Array.Empty<long>(), "000000000000d848");
            AssertBytes(C.chebval(L(L(0.5, 1.0), L(2.0, 3.0)), C64()), "float64", new long[] { 2, 2 },
                "000000000000e03f00000000000018400000000000003a400000000000004d40");
        }

        /// <summary>
        ///     <c>{p}val</c>'s c is array_like too (<c>c = np.array(c, ndmin=1)</c>): a mixed list (<c>polyval(0.5, [1, 2.5])</c>),
        ///     a tuple, a nested list (a 2-D series, <c>chebval([0.5, 1.0], [[1, 2], [3, 4], [0.5, -1]])</c>) and a NumPy
        ///     scalar, which keeps its dtype (<c>polyval(0.5, np.float16(3))</c> is float16). An <c>object[]</c> c used to
        ///     compile — through NumSharp's implicit array conversion — and then fail at run time.
        /// </summary>
        [TestMethod]
        public void Val_TakesArrayLikeCoefficients()
        {
            AssertBytes(P.polyval(0.5, L(1, 2.5)), "float64", Array.Empty<long>(), "0000000000000240");
            AssertBytes(P.polyval(0.5, (1, 2.5)), "float64", Array.Empty<long>(), "0000000000000240");
            AssertBytes(C.chebval(L(0.5, 1.0), L(L(1, 2), L(3, 4), L(0.5, -1))), "float64", new long[] { 2, 2 },
                "0000000000000240000000000000124000000000000012400000000000001440");
            AssertBytes(P.polyval(0.5, (Half)3), "float16", Array.Empty<long>(), "0042");
        }

        /// <summary>
        ///     NumPy converts c BEFORE x, so of two ragged arguments c's error is the one reported — distinguishable here
        ///     because the two are ragged at different depths (<c>polyval([[[1], [1, 2]]], [[1], [1, 2]])</c>).
        /// </summary>
        [TestMethod]
        public void Val_RaggedCoefficientsAreReportedBeforeARaggedX()
        {
            // [[[1], [1, 2]]] is a ONE-item list of a list: spelled explicitly, because L(x) of a lone object[] x IS x (the
            // params array itself), which would silently drop a nesting level.
            object raggedDeep = new object[] { L(L(1), L(1, 2)) };
            Action both = () => P.polyval(raggedDeep, L(L(1), L(1, 2)));
            both.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence. The requested array has an " +
                                                          "inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            Action xOnly = () => P.polyval(raggedDeep, L(1, 2));
            xOnly.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence. The requested array has an " +
                                                           "inhomogeneous shape after 2 dimensions. The detected shape was (1, 2) + inhomogeneous part.");
        }

        /// <summary>
        ///     <c>{p}val2d</c> / <c>{p}val3d</c> ordinates are <c>np.asanyarray</c>'d first: a tuple is an array, a Python int a
        ///     STRONG int64 0-d array (so a float32 series evaluates in float64 — <c>polyval2d(2, 3, c.astype(np.float32))</c>),
        ///     while <c>{p}grid2d</c> hands the same scalars to <c>{p}val</c> unconverted (weak: float32). The shape check runs
        ///     before c is converted, so mismatched ordinates win over a ragged c.
        /// </summary>
        [TestMethod]
        public void ValNd_OrdinatesAreArrayLike()
        {
            var c2 = np.array(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
            AssertBytes(P.polyval2d((0.5, 1.0), (1.0, 2.0), c2), "float64", new long[] { 2 }, "0000000000001a400000000000003040");
            AssertBytes(P.polyval2d(2, 3, c2), "float64", Array.Empty<long>(), "0000000000804240");
            using var c32 = c2.astype(np.float32);
            AssertBytes(P.polyval2d(2, 3, c32), "float64", Array.Empty<long>(), "0000000000804240");
            AssertBytes(P.polygrid2d(2, 3, c32), "float32", Array.Empty<long>(), "00001442");
            Action shapesFirst = () => P.polyval2d(L(0.5, 1.0), L(1.0, 2.0, 3.0), L(L(1), L(1, 2)));
            shapesFirst.Should().Throw<ValueError>().WithMessage("x, y are incompatible");
        }

        /// <summary>
        ///     A C# <c>object[]</c> of points binds <c>mapdomain</c>'s LIST overload (its <see cref="NDArray"/> return type
        ///     is the compile-time proof) — it used to bind the NDArray overload through the implicit array conversion and
        ///     fail at run time. <c>mapdomain([0.5, 1.5], (0, 2), (-1, 1))</c> and a nested list with list domains.
        /// </summary>
        [TestMethod]
        public void MapDomain_ListBindsTheListOverload()
        {
            NDArray tuples = U.mapdomain(L(0.5, 1.5), (0, 2), (-1, 1));
            AssertBytes(tuples, "float64", new long[] { 2 }, "000000000000e0bf000000000000e03f");
            NDArray lists = U.mapdomain(L(L(0.5), L(1.5)), L(0, 2), L(-1, 1));
            AssertBytes(lists, "float64", new long[] { 2, 1 }, "000000000000e0bf000000000000e03f");
        }

        /// <summary>
        ///     <c>trimseq</c> of a Python sequence returns the same KIND: the list itself when nothing trails
        ///     (<c>trimseq(r) is r</c>), else a new list of the kept items (<c>trimseq([1, 2, 0])</c> is <c>[1, 2]</c>, an
        ///     <c>NDArray[]</c> staying one), a tuple's slice as a tuple (<c>trimseq((1, 0, 0))</c> is <c>(1,)</c>), NumPy's
        ///     truth-value error for a tested array item of several elements, and <c>len()</c>'s TypeError for None or a
        ///     number.
        /// </summary>
        [TestMethod]
        public void TrimSeq_OfSequencesKeepsTheKind()
        {
            var keep = L(1, 2);
            U.trimseq(keep).Should().BeSameAs(keep);
            var trimmed = U.trimseq(L(1, 2, 0));
            trimmed.Should().Equal(1, 2);
            U.trimseq(L(0, 0)).Should().Equal(0);

            var a = np.array(new[] { 1.0 });
            var z = np.array(new[] { 0.0 });
            var arrays = U.trimseq(new[] { a, z });
            arrays.Should().BeOfType<NDArray[]>();
            arrays.Length.Should().Be(1);
            arrays[0].Should().BeSameAs(a);

            var tuple = U.trimseq((object)(1, 0, 0));
            tuple.Should().BeAssignableTo<System.Runtime.CompilerServices.ITuple>();
            var t = (System.Runtime.CompilerServices.ITuple)tuple;
            t.Length.Should().Be(1);
            t[0].Should().Be(1);

            Action ambiguous = () => U.trimseq(L(np.array(new[] { 1.0, 2.0 }), 0));
            ambiguous.Should().Throw<ValueError>().WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            Action none = () => U.trimseq((object[])null);
            none.Should().Throw<TypeError>().WithMessage("object of type 'NoneType' has no len()");
            Action number = () => U.trimseq((object)5);
            number.Should().Throw<TypeError>().WithMessage("object of type 'int' has no len()");
        }
    }
}
