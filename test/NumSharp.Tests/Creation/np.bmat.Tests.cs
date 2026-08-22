using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    /// Tests for np.bmat — verified 1-to-1 against NumPy 2.4.2. Builds a 2-D array from blocks: the
    /// nested form [[A,B],[C,D]] joins each row along the last axis and stacks the rows along axis 0;
    /// the flat form [A,B] is a single side-by-side row; a single array is copied and coerced to 2-D;
    /// the string form "A,B; C,D" resolves each token to a block through a name dictionary. bmat is
    /// pure block assembly (concatenation) — it performs no matrix multiplication. NumSharp returns a
    /// plain 2-D NDArray (no matrix subclass).
    /// </summary>
    [TestClass]
    public class np_bmat_Tests
    {
        // NumPy 2.4.2 fixtures:
        //   A = [[1,1],[1,1]]  B = [[2,2],[2,2]]  C = [[3,4],[5,6]]  D = [[7,8],[9,0]]
        private static NDArray A() => np.array(new long[,] { { 1, 1 }, { 1, 1 } });
        private static NDArray B() => np.array(new long[,] { { 2, 2 }, { 2, 2 } });
        private static NDArray C() => np.array(new long[,] { { 3, 4 }, { 5, 6 } });
        private static NDArray D() => np.array(new long[,] { { 7, 8 }, { 9, 0 } });

        // ─── nested [[A,B],[C,D]] form ──────────────────────────────────────

        [TestMethod]
        public void Nested_TwoByTwoBlocks_TilesMatrix()
        {
            // np.bmat([[A,B],[C,D]]) -> [[1,1,2,2],[1,1,2,2],[3,4,7,8],[5,6,9,0]]
            var m = np.bmat(new[] { new[] { A(), B() }, new[] { C(), D() } });
            m.shape.Should().Equal(new long[] { 4, 4 });
            m.typecode.Should().Be(NPTypeCode.Int64);
            var expected = np.array(new long[,]
            {
                { 1, 1, 2, 2 },
                { 1, 1, 2, 2 },
                { 3, 4, 7, 8 },
                { 5, 6, 9, 0 },
            });
            np.array_equal(m, expected).Should().BeTrue();
        }

        [TestMethod]
        public void Nested_SingleRow_ConcatenatesHorizontally()
        {
            // np.bmat([[A,B]]) -> (2,4)
            var m = np.bmat(new[] { new[] { A(), B() } });
            m.shape.Should().Equal(new long[] { 2, 4 });
            np.array_equal(m, np.array(new long[,] { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } })).Should().BeTrue();
        }

        [TestMethod]
        public void Nested_ThreeRows_StacksVertically()
        {
            // np.bmat([[A],[B],[A]]) -> (6,2)
            var m = np.bmat(new[] { new[] { A() }, new[] { B() }, new[] { A() } });
            m.shape.Should().Equal(new long[] { 6, 2 });
        }

        [TestMethod]
        public void Nested_OneDBlocks_CoercesToRow()
        {
            // np.bmat([[u],[v]]) with u=(3,), v=(2,) -> (1,5)
            var u = np.array(new long[] { 1, 2, 3 });
            var v = np.array(new long[] { 4, 5 });
            var m = np.bmat(new[] { new[] { u }, new[] { v } });
            m.shape.Should().Equal(new long[] { 1, 5 });
            np.array_equal(m, np.array(new long[,] { { 1, 2, 3, 4, 5 } })).Should().BeTrue();
        }

        // ─── flat [A,B] form ────────────────────────────────────────────────

        [TestMethod]
        public void Flat_TwoDBlocks_ConcatenatesHorizontally()
        {
            // np.bmat([A,B]) -> (2,4)
            var m = np.bmat(new[] { A(), B() });
            m.shape.Should().Equal(new long[] { 2, 4 });
            np.array_equal(m, np.array(new long[,] { { 1, 1, 2, 2 }, { 1, 1, 2, 2 } })).Should().BeTrue();
        }

        [TestMethod]
        public void Flat_OneDBlocks_CoercesToRow()
        {
            // np.bmat([u,v]) -> (1,5)
            var u = np.array(new long[] { 1, 2, 3 });
            var v = np.array(new long[] { 4, 5 });
            var m = np.bmat(new[] { u, v });
            m.shape.Should().Equal(new long[] { 1, 5 });
            np.array_equal(m, np.array(new long[,] { { 1, 2, 3, 4, 5 } })).Should().BeTrue();
        }

        // ─── single-array form (copies, coerces to 2-D) ─────────────────────

        [TestMethod]
        public void Single_TwoD_Unchanged()
        {
            var m = np.bmat(A());
            m.shape.Should().Equal(new long[] { 2, 2 });
            np.array_equal(m, A()).Should().BeTrue();
        }

        [TestMethod]
        public void Single_Copies_DoesNotAliasInput()
        {
            // NumPy's bmat(ndarray) is matrix(obj) with copy=True -> the result must not alias obj.
            var a = A();
            var m = np.bmat(a);
            m[0, 0] = 99L;
            a.GetValue<long>(0, 0).Should().Be(1L);
        }

        [TestMethod]
        public void Single_OneD_Becomes_1xN()
        {
            np.bmat(np.array(new long[] { 1, 2, 3 })).shape.Should().Equal(new long[] { 1, 3 });
        }

        [TestMethod]
        public void Single_ZeroD_Becomes_1x1()
        {
            var m = np.bmat(NDArray.Scalar(5.0));
            m.shape.Should().Equal(new long[] { 1, 1 });
            m.GetValue<double>(0, 0).Should().Be(5.0);
        }

        // ─── string form ────────────────────────────────────────────────────

        [TestMethod]
        public void String_CommaSeparated_ResolvesBlocks()
        {
            var dict = new Dictionary<string, NDArray> { ["A"] = A(), ["B"] = B(), ["C"] = C(), ["D"] = D() };
            var m = np.bmat("A,B; C,D", dict);
            m.shape.Should().Equal(new long[] { 4, 4 });
            np.array_equal(m, np.bmat(new[] { new[] { A(), B() }, new[] { C(), D() } })).Should().BeTrue();
        }

        [TestMethod]
        public void String_WhitespaceSeparated_ResolvesBlocks()
        {
            var dict = new Dictionary<string, NDArray> { ["A"] = A(), ["B"] = B(), ["C"] = C(), ["D"] = D() };
            np.bmat("A B; C D", dict).shape.Should().Equal(new long[] { 4, 4 });
        }

        [TestMethod]
        public void String_LdictTriedBeforeGdict()
        {
            // Resolution order: ldict first, then gdict (NumPy's _from_string).
            var ldict = new Dictionary<string, NDArray> { ["X"] = A() };
            var gdict = new Dictionary<string, NDArray> { ["X"] = D() };
            np.array_equal(np.bmat("X", ldict, gdict), A()).Should().BeTrue();
        }

        [TestMethod]
        public void String_FallsBackToGdict()
        {
            var gdict = new Dictionary<string, NDArray> { ["A"] = A(), ["B"] = B() };
            np.bmat("A B", null, gdict).shape.Should().Equal(new long[] { 2, 4 });
        }

        [TestMethod]
        public void String_MissingName_RaisesNameError() =>
            ((Action)(() => np.bmat("Z", new Dictionary<string, NDArray>())))
                .Should().Throw<NameError>().WithMessage("name 'Z' is not defined");

        [TestMethod]
        public void String_NumericLiteralTreatedAsName_RaisesNameError() =>
            // Unlike np.asmatrix, bmat's string form resolves tokens as NAMES, so "1" is undefined.
            ((Action)(() => np.bmat("1 2; 3 4", new Dictionary<string, NDArray>())))
                .Should().Throw<NameError>().WithMessage("name '1' is not defined");

        [TestMethod]
        public void String_NoDictionary_Raises() =>
            ((Action)(() => np.bmat("A B", null, null)))
                .Should().Throw<ArgumentException>();

        // ─── dtype promotion ────────────────────────────────────────────────

        [TestMethod]
        public void MixedDtype_PromotesToDouble()
        {
            var af = np.array(new double[,] { { 1.5, 2.5 } });
            var bi = np.array(new long[,] { { 3, 4 } });
            var m = np.bmat(new[] { new[] { af, bi } });
            m.typecode.Should().Be(NPTypeCode.Double);
            np.array_equal(m, np.array(new double[,] { { 1.5, 2.5, 3.0, 4.0 } })).Should().BeTrue();
        }

        // ─── plain (non-"matrix") ndarray blocks still work ─────────────────

        [TestMethod]
        public void PlainNdarrayBlocks_Tile()
        {
            var p = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
            var q = np.array(new long[,] { { 5, 6 }, { 7, 8 } });
            var m = np.bmat(new[] { new[] { p, q } });
            np.array_equal(m, np.array(new long[,] { { 1, 2, 5, 6 }, { 3, 4, 7, 8 } })).Should().BeTrue();
        }

        // ─── error parity ───────────────────────────────────────────────────

        [TestMethod]
        public void RaggedWidth_Raises() =>
            // row0 width 4, row1 width 2 -> concatenate axis 0 mismatch (NumPy ValueError; house type).
            ((Action)(() => np.bmat(new[] { new[] { A(), B() }, new[] { A() } })))
                .Should().Throw<IncorrectShapeException>();

        [TestMethod]
        public void HeightMismatchInRow_Raises()
        {
            var tall = np.array(new long[,] { { 1 }, { 2 }, { 3 } }); // (3,1)
            ((Action)(() => np.bmat(new[] { new[] { A(), tall } })))
                .Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void EmptyFlat_Raises() =>
            ((Action)(() => np.bmat(new NDArray[0]))).Should().Throw<ArgumentException>();

        [TestMethod]
        public void EmptyNested_Raises() =>
            ((Action)(() => np.bmat(new NDArray[0][]))).Should().Throw<ArgumentException>();

        // ─── null arguments ─────────────────────────────────────────────────

        [TestMethod]
        public void NullNestedInput_Throws() =>
            ((Action)(() => np.bmat((NDArray[][])null))).Should().Throw<ArgumentNullException>();

        [TestMethod]
        public void NullFlatInput_Throws() =>
            ((Action)(() => np.bmat((NDArray[])null))).Should().Throw<ArgumentNullException>();

        [TestMethod]
        public void NullSingleInput_Throws() =>
            ((Action)(() => np.bmat((NDArray)null))).Should().Throw<ArgumentNullException>();

        [TestMethod]
        public void NullStringInput_Throws() =>
            ((Action)(() => np.bmat((string)null, new Dictionary<string, NDArray>())))
                .Should().Throw<ArgumentNullException>();
    }
}
