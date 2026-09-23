using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Tests for np.real_if_close — "if all imaginary parts are within tol of zero, return the real
    /// parts". All expected values probed against NumPy 2.4.2 (numpy/lib/_type_check_impl.py:
    /// <c>if all(absolute(a.imag) &lt; tol): a = a.real</c>). The collapse yields a float64 real-lane
    /// VIEW (== NumPy's <c>a.real</c>); otherwise the complex array is returned unchanged. tol &gt; 1 is
    /// interpreted in float64-eps multiples, tol &lt;= 1 as an absolute tolerance, tol &lt;= 0 collapses
    /// nothing (a magnitude is never strictly &lt; 0). NaN/inf imaginary parts and the strict <c>&lt;</c>
    /// boundary prevent collapse; an empty complex array collapses (vacuous <c>all</c>).
    /// </summary>
    [TestClass]
    public class np_real_if_close_Test
    {
        private const double Eps = 2.220446049250313e-16; // finfo(float64).eps

        private static Complex C(double re, double im) => new Complex(re, im);

        // ---- Collapse: complex -> float64 real lane -------------------------------------------

        [TestMethod]
        public void Collapse_DefaultTol_ReturnsFloat64Reals()
        {
            // np.real_if_close([1+1e-15j, 2+1e-16j]) -> array([1., 2.]) (default tol=100 -> 2.22e-14).
            var r = np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 1e-16) }));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, 2.0 }));
        }

        [TestMethod]
        public void Collapse_LargeTol_MachineEpsilonMultiples()
        {
            // NumPy docstring example: tol=1000 -> 1000*eps ~ 2.22e-13; 4e-14 and 3e-15 both below.
            var r = np.real_if_close(np.array(new[] { C(2.1, 4e-14), C(5.2, 3e-15) }), tol: 1000);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 2.1, 5.2 }));
        }

        [TestMethod]
        public void NoCollapse_ImagOutOfBand_StaysComplexUnchanged()
        {
            // NumPy docstring example: 4e-13 > 1000*eps (2.22e-13) -> stays complex, unchanged.
            var a = np.array(new[] { C(2.1, 4e-13), C(5.2, 3e-15) });
            var r = np.real_if_close(a, tol: 1000);
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            // NumPy returns the SAME object when it does not collapse.
            Assert.IsTrue(ReferenceEquals(a, r));
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<Complex>(), new[] { C(2.1, 4e-13), C(5.2, 3e-15) }));
        }

        // ---- tol modes -------------------------------------------------------------------------

        [TestMethod]
        public void TolLessEqualOne_IsAbsoluteTolerance()
        {
            // tol=0.5 <= 1 -> absolute: 0.3, 0.4 < 0.5 -> collapse.
            var collapse = np.real_if_close(np.array(new[] { C(1, 0.3), C(2, 0.4) }), tol: 0.5);
            Assert.AreEqual(NPTypeCode.Double, collapse.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(collapse.Data<double>(), new[] { 1.0, 2.0 }));

            // tol=0.35 -> 0.4 >= 0.35 -> stays complex.
            var stay = np.real_if_close(np.array(new[] { C(1, 0.3), C(2, 0.4) }), tol: 0.35);
            Assert.AreEqual(NPTypeCode.Complex, stay.typecode);
        }

        [TestMethod]
        public void TolExactlyOne_IsAbsolute_NotEpsMultiplied()
        {
            // tol=1 is NOT > 1, so it is an absolute tolerance of 1.0 (not 1*eps).
            var r = np.real_if_close(np.array(new[] { C(1, 0.5), C(2, 0.9) }), tol: 1);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, 2.0 }));
        }

        [TestMethod]
        public void TolZeroOrNegative_CollapsesNothing_EvenForExactZeroImag()
        {
            // |imag| >= 0 is never strictly < 0, so tol <= 0 never collapses — even for exact-zero imag.
            foreach (var tol in new[] { 0.0, -5.0 })
            {
                var r = np.real_if_close(np.array(new[] { C(1, 0), C(2, 0) }), tol: tol);
                Assert.AreEqual(NPTypeCode.Complex, r.typecode, $"tol={tol}");
            }
        }

        [TestMethod]
        public void Boundary_ImagExactlyEqualsTol_DoesNotCollapse()
        {
            // The comparison is STRICT (< tol), so imag == the resolved tol (eps*100) stays complex.
            var r = np.real_if_close(np.array(new[] { C(1, Eps * 100) }), tol: 100);
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
        }

        // ---- NaN / inf imaginary parts ---------------------------------------------------------

        [TestMethod]
        public void NanImag_DoesNotCollapse()
        {
            var r = np.real_if_close(np.array(new[] { C(1, double.NaN), C(2, 0) }), tol: 1000);
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
        }

        [TestMethod]
        public void InfImag_DoesNotCollapse()
        {
            var r = np.real_if_close(np.array(new[] { C(1, double.PositiveInfinity), C(2, 0) }), tol: 1000);
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            var r2 = np.real_if_close(np.array(new[] { C(1, double.NegativeInfinity), C(2, 0) }), tol: 1000);
            Assert.AreEqual(NPTypeCode.Complex, r2.typecode);
        }

        [TestMethod]
        public void NanReal_TinyImag_Collapses_KeepingNanReal()
        {
            // The band test is on the IMAGINARY part only — a NaN REAL part collapses fine, and the
            // result carries that NaN in the real lane.
            var r = np.real_if_close(np.array(new[] { C(double.NaN, 1e-15), C(2, 0) }), tol: 1000);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            var vals = r.Data<double>();
            Assert.IsTrue(double.IsNaN(vals[0]));
            Assert.AreEqual(2.0, vals[1]);
        }

        // ---- Signed zero ------------------------------------------------------------------------

        [TestMethod]
        public void NegativeZeroImag_Collapses()
        {
            // |-0.0| == 0.0 < tol -> collapses.
            var r = np.real_if_close(np.array(new[] { C(1, -0.0), C(2, -0.0) }));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, 2.0 }));
        }

        // ---- Non-complex input returned unchanged ----------------------------------------------

        [TestMethod]
        public void NonComplex_ReturnedUnchanged_SameInstance()
        {
            var ai = np.array(new[] { 1, 2, 3 });
            Assert.IsTrue(ReferenceEquals(ai, np.real_if_close(ai)));

            var af = np.array(new[] { 1.5, 2.5 });
            Assert.IsTrue(ReferenceEquals(af, np.real_if_close(af)));

            var ab = np.array(new[] { true, false });
            Assert.IsTrue(ReferenceEquals(ab, np.real_if_close(ab)));
        }

        // ---- Empty / 0-d edge shapes -----------------------------------------------------------

        [TestMethod]
        public void EmptyComplex_Collapses_VacuousAll()
        {
            // np.all([]) is True, so an empty complex array collapses to an empty float64 array.
            var r = np.real_if_close(np.zeros(new Shape(0), NPTypeCode.Complex), tol: 1000);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(0, r.size);
            r.shape.Should().Equal(0);

            var r2 = np.real_if_close(np.zeros(new Shape(0, 3), NPTypeCode.Complex));
            Assert.AreEqual(NPTypeCode.Double, r2.typecode);
            r2.shape.Should().Equal(0, 3);
        }

        [TestMethod]
        public void ScalarZeroD_CollapseAndNoCollapse()
        {
            var close = np.real_if_close(NDArray.Scalar(C(1, 1e-15)));
            Assert.AreEqual(NPTypeCode.Double, close.typecode);
            Assert.AreEqual(0, close.ndim);
            Assert.AreEqual(1.0, close.GetDouble());

            var far = np.real_if_close(NDArray.Scalar(C(1, 0.5)));
            Assert.AreEqual(NPTypeCode.Complex, far.typecode);
            Assert.AreEqual(0, far.ndim);
        }

        // ---- Multi-dimensional + non-contiguous layouts ----------------------------------------

        [TestMethod]
        public void TwoD_Collapse_PreservesShape()
        {
            var r = np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 1e-16), C(3, 0), C(4, 1e-15) }).reshape(2, 2));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            r.shape.Should().Equal(2, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(r.flatten().Data<double>(), new[] { 1.0, 2.0, 3.0, 4.0 }));
        }

        [TestMethod]
        public void Transposed_Collapse_LogicalRealValues()
        {
            // reshape(2,3) reals [[1,2,3],[4,5,6]], transposed -> [[1,4],[2,5],[3,6]].
            var big = np.array(new[] { C(1, 1e-15), C(2, 1e-16), C(3, 0), C(4, 1e-15), C(5, 0), C(6, 1e-15) }).reshape(2, 3);
            var r = np.real_if_close(big.T);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            r.shape.Should().Equal(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(r.flatten().Data<double>(), new[] { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }));
        }

        [TestMethod]
        public void NegativeStride_Collapse_ReversedReals()
        {
            var r = np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 0), C(3, 1e-16) })["::-1"]);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 3.0, 2.0, 1.0 }));
        }

        [TestMethod]
        public void StridedView_ScansOnlyItsOwnElements()
        {
            // The ::2 view addresses only the tiny-imag elements; the skipped elements carry a large
            // imaginary part (0.9) that must NOT affect the collapse decision.
            var interleaved = np.array(new[]
            {
                C(1, 1e-15), C(9, 0.9), C(2, 1e-16), C(9, 0.9), C(3, 0), C(9, 0.9)
            });
            var r = np.real_if_close(interleaved["::2"]);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, 2.0, 3.0 }));
        }

        // ---- View semantics: the collapse result shares memory with the input (like a.real) -----

        [TestMethod]
        public void Collapse_ReturnsWriteThroughViewOfRealLane()
        {
            var a = np.array(new[] { C(1, 1e-15), C(2, 1e-16) });
            var r = np.real_if_close(a);
            r.SetData(99.0, 0);                       // write through the real-lane view
            Assert.AreEqual(99.0, a.GetComplex(0).Real); // the source's real part changed
            Assert.AreEqual(1e-15, a.GetComplex(0).Imaginary); // imaginary untouched
        }

        // ---- Longer arrays exercise the SIMD deinterleave body (>= 8 elements) ------------------

        [TestMethod]
        public void LongArray_SimdBody_CollapseAndNoCollapse()
        {
            var reals = Enumerable.Range(0, 20).Select(i => (double)(i + 1)).ToArray();
            var tiny = np.array(reals.Select(x => C(x, 1e-15)).ToArray());
            var rc = np.real_if_close(tiny);
            Assert.AreEqual(NPTypeCode.Double, rc.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(rc.Data<double>(), reals));

            // One out-of-band element deep in the array -> the whole array stays complex.
            var vals = reals.Select(x => C(x, 1e-15)).ToArray();
            vals[13] = C(14, 0.5);
            var rs = np.real_if_close(np.array(vals));
            Assert.AreEqual(NPTypeCode.Complex, rs.typecode);
        }

        // ---- Validation-hardening: edges the 360-case oracle tier does not reach (all probed 2.4.2) --

        [TestMethod]
        public void TolPositiveInfinity_CollapsesEveryFiniteImag()
        {
            // tol=+inf is > 1, so tol = eps*inf = +inf; every FINITE |imag| < inf -> collapse, even a
            // large imaginary part (probed against NumPy 2.4.2).
            var r = np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 5.0), C(3, -100.0) }), tol: double.PositiveInfinity);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(r.Data<double>(), new[] { 1.0, 2.0, 3.0 }));

            // ...but an infinite imaginary part is NOT < inf, so it still does not collapse.
            var r2 = np.real_if_close(np.array(new[] { C(1, double.PositiveInfinity), C(2, 0) }), tol: double.PositiveInfinity);
            Assert.AreEqual(NPTypeCode.Complex, r2.typecode);
        }

        [TestMethod]
        public void TolNaN_And_TolSubnormal_DoNotCollapse()
        {
            // tol=NaN: NaN > 1 is false -> tol stays NaN -> |imag| < NaN is always false -> no collapse.
            Assert.AreEqual(NPTypeCode.Complex,
                np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 0) }), tol: double.NaN).typecode);

            // tol = smallest subnormal (<= 1 -> absolute): 1e-15 >> 5e-324, so nothing is within band.
            Assert.AreEqual(NPTypeCode.Complex,
                np.real_if_close(np.array(new[] { C(1, 1e-15), C(2, 0) }), tol: double.Epsilon).typecode);
        }

        [TestMethod]
        public void ImagBoundary_JustUnderCollapses_JustOverDoesNot()
        {
            // The resolved tol at tol=100 is eps*100; strict `<` splits the two adjacent doubles.
            double tol100 = Eps * 100.0;
            var under = np.real_if_close(np.array(new[] { C(1, System.Math.BitDecrement(tol100)), C(2, 1e-16) }), tol: 100);
            Assert.AreEqual(NPTypeCode.Double, under.typecode);
            var over = np.real_if_close(np.array(new[] { C(1, System.Math.BitIncrement(tol100)), C(2, 1e-16) }), tol: 100);
            Assert.AreEqual(NPTypeCode.Complex, over.typecode);
        }

        [TestMethod]
        public void SpecialRealParts_SurviveCollapse_BitExact()
        {
            // The band test is on the imaginary part only, so a NaN / +-inf / -0.0 REAL part collapses
            // and survives into the float64 real lane exactly (probed against NumPy 2.4.2).
            var r = np.real_if_close(np.array(new[]
            {
                C(1.5, 1e-15), C(-0.0, 1e-15), C(double.NaN, 1e-15),
                C(double.PositiveInfinity, 1e-15), C(double.NegativeInfinity, 1e-15)
            }), tol: 100);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            var v = r.Data<double>();
            Assert.AreEqual(1.5, v[0]);
            Assert.IsTrue(v[1] == 0.0 && double.IsNegative(v[1]));   // -0.0 sign bit preserved
            Assert.IsTrue(double.IsNaN(v[2]));
            Assert.IsTrue(double.IsPositiveInfinity(v[3]));
            Assert.IsTrue(double.IsNegativeInfinity(v[4]));
        }

        [TestMethod]
        public void Rank3_And_Rank4_CollapseAndNoCollapse()
        {
            foreach (var rank in new[] { 3, 4 })
            {
                int n = rank == 3 ? 8 : 16;
                var tiny = Enumerable.Range(0, n).Select(i => C(i + 1, 1e-15)).ToArray();
                var dims = rank == 3 ? new[] { 2, 2, 2 } : new[] { 2, 2, 2, 2 };
                var rc = np.real_if_close(np.array(tiny).reshape(dims));
                Assert.AreEqual(NPTypeCode.Double, rc.typecode, $"rank {rank} collapse");
                Assert.AreEqual(string.Join(",", dims), string.Join(",", rc.shape), $"rank {rank} shape");

                var big = (Complex[])tiny.Clone();
                big[n / 2] = C(big[n / 2].Real, 0.5);
                var rs = np.real_if_close(np.array(big).reshape(dims));
                Assert.AreEqual(NPTypeCode.Complex, rs.typecode, $"rank {rank} no-collapse");
            }
        }

        [TestMethod]
        public void FContiguous_And_OffsetSlice_Collapse()
        {
            var src = np.array(Enumerable.Range(0, 12).Select(i => C(i + 1, 1e-15)).ToArray()).reshape(3, 4);

            // Genuine F-contiguous (asfortranarray, not a transpose) still takes the dense scan path.
            var rf = np.real_if_close(np.asfortranarray(src));
            Assert.AreEqual(NPTypeCode.Double, rf.typecode);
            rf.shape.Should().Equal(3, 4);

            // Positive-offset slice (offset != 0) — the scan honours Shape.offset.
            var flat = np.array(Enumerable.Range(0, 12).Select(i => C(i + 1, 1e-15)).ToArray());
            var ro = np.real_if_close(flat["2:9"]);
            Assert.AreEqual(NPTypeCode.Double, ro.typecode);
            Assert.IsTrue(Enumerable.SequenceEqual(ro.Data<double>(), new[] { 3.0, 4, 5, 6, 7, 8, 9 }));
        }

        [TestMethod]
        public void BroadcastReadOnly_Collapse_ReturnsReadOnlyView()
        {
            // A broadcast complex view is read-only; NumPy's a.real of it is read-only too, so the
            // collapse result must stay non-writeable (matching NumPy).
            var b = np.broadcast_to(np.array(new[] { C(1.5, 1e-15) }), new Shape(5));
            var r = np.real_if_close(b);
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.IsFalse(r.Shape.IsWriteable);
        }

        [TestMethod]
        public void DoesNotMutateInput()
        {
            // real_if_close itself never writes to the array (collapse returns a read-through view).
            var a = np.array(new[] { C(1.5, 1e-15), C(2.5, 3e-15) });
            var before0 = a.GetComplex(0);
            var before1 = a.GetComplex(1);
            np.real_if_close(a);
            Assert.AreEqual(before0, a.GetComplex(0));
            Assert.AreEqual(before1, a.GetComplex(1));
        }
    }
}
