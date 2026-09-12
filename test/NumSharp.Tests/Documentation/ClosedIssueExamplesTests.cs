using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using NumSharp.Backends.Iteration;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Backing tests for the closing comments posted on the GitHub issues resolved by 0.70.0
    ///     (#628 and the feature PRs it bundles). Each test mirrors the example shown in the
    ///     corresponding issue's closing comment and asserts the result against NumPy 2.4.2's
    ///     observed output, so a closed issue's claim cannot silently regress. The expected values
    ///     were captured by running each snippet against this branch.
    /// </summary>
    [TestClass]
    public class ClosedIssueExamplesTests
    {
        // OpenBLAS auto-install is disabled for this assembly; the factorisation tests enable the
        // backend per-test and restore the absent state in a finally, matching the engine tests.
        private static bool TryEnableOpenBlas()
        {
            try { OpenBlasEngine.Enable(); } catch { return false; }
            return OpenBlasEngine.Enabled;
        }

        // ── Linear algebra ───────────────────────────────────────────────────────────────────────

        // Issue #105 - np.vdot
        [TestMethod]
        public void Issue105_Vdot_MatchesNumPy()
        {
            var a = np.array(new[] { 1.0, 2, 3 });
            var b = np.array(new[] { 4.0, 5, 6 });
            np.vdot(a, b).GetDouble(0).Should().Be(32.0);
        }

        // Issue #106 - np.inner
        [TestMethod]
        public void Issue106_Inner_MatchesNumPy()
        {
            var a = np.array(new[] { 1.0, 2, 3 });
            var b = np.array(new[] { 0.0, 1, 0 });
            np.inner(a, b).GetDouble(0).Should().Be(2.0);
        }

        // Issue #108 - np.tensordot
        [TestMethod]
        public void Issue108_Tensordot_MatchesNumPy()
        {
            var a = np.arange(6).reshape(2, 3);
            var b = np.arange(12).reshape(3, 4);
            var expected = np.array(new long[,] { { 20, 23, 26, 29 }, { 56, 68, 80, 92 } });
            np.array_equal(np.tensordot(a, b, axes: 1), expected).Should().BeTrue();
        }

        // Issue #239 / #441 / #485 - np.linalg.norm
        [TestMethod]
        public void Issue239_Norm_MatchesNumPy()
        {
            np.linalg.norm(np.array(new[] { 3.0, 4 })).GetDouble(0).Should().Be(5.0);
            np.linalg.norm(np.eye(3)).GetDouble(0).Should().BeApproximately(System.Math.Sqrt(3), 1e-9);
        }

        // Issue #445 - np.dot with out=
        [TestMethod]
        public void Issue445_DotOut_MatchesNumPy()
        {
            var a = np.array(new[,] { { 1.0, 2 }, { 3, 4 } });
            var b = np.array(new[,] { { 5.0, 6 }, { 7, 8 } });
            var outBuf = np.zeros(new Shape(2, 2));
            np.dot(a, b, @out: outBuf);
            np.allclose(outBuf, np.array(new[,] { { 19.0, 22 }, { 43, 50 } })).Should().BeTrue();
        }

        // Issue #621 - np.linalg umbrella (solve; managed LU, no backend required)
        [TestMethod]
        public void Issue621_Solve_MatchesNumPy()
        {
            var A = np.array(new[,] { { 3.0, 1 }, { 1, 2 } });
            var b = np.array(new[] { 9.0, 8 });
            np.allclose(np.linalg.solve(A, b), np.array(new[] { 2.0, 3 })).Should().BeTrue();
        }

        // Issue #427 - np.matmul
        [TestMethod]
        public void Issue427_Matmul_MatchesNumPy()
        {
            var a = np.array(new[,] { { 1.0, 2 }, { 3, 4 } });
            var b = np.array(new[,] { { 5.0, 6 }, { 7, 8 } });
            np.allclose(np.matmul(a, b), np.array(new[,] { { 19.0, 22 }, { 43, 50 } })).Should().BeTrue();
        }

        // Issue #454 - np.linalg.lstsq (OpenBLAS)
        [TestMethod]
        public void Issue454_Lstsq_MatchesNumPy()
        {
            if (!TryEnableOpenBlas()) { Assert.Inconclusive("OpenBLAS backend not available on this host."); return; }
            try
            {
                var A = np.array(new[,] { { 0.0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } });
                var y = np.array(new[] { -1.0, 0.2, 0.9, 2.1 });
                np.allclose(np.linalg.lstsq(A, y).Solution, np.array(new[] { 1.0, -0.95 })).Should().BeTrue();
            }
            finally { OpenBlasEngine.Disable(); }
        }

        // Issue #497 - np.linalg.pinv (OpenBLAS)
        [TestMethod]
        public void Issue497_Pinv_MatchesNumPy()
        {
            if (!TryEnableOpenBlas()) { Assert.Inconclusive("OpenBLAS backend not available on this host."); return; }
            try
            {
                var a = np.array(new[,] { { 1.0, 2 }, { 3, 4 }, { 5, 6 } });
                var p = np.linalg.pinv(a);
                np.allclose(np.matmul(np.matmul(a, p), a), a).Should().BeTrue();
            }
            finally { OpenBlasEngine.Disable(); }
        }

        // Issue #472 - np.linalg.matrix_rank (OpenBLAS)
        [TestMethod]
        public void Issue472_MatrixRank_MatchesNumPy()
        {
            if (!TryEnableOpenBlas()) { Assert.Inconclusive("OpenBLAS backend not available on this host."); return; }
            try
            {
                np.array_equal(np.linalg.matrix_rank(np.array(new[,] { { 1.0, 2 }, { 2, 4 } })), NDArray.Scalar(1)).Should().BeTrue();
                np.array_equal(np.linalg.matrix_rank(np.eye(3)), NDArray.Scalar(3)).Should().BeTrue();
            }
            finally { OpenBlasEngine.Disable(); }
        }

        // ── FFT ──────────────────────────────────────────────────────────────────────────────────

        // Issue #114 - np.fft.fft
        [TestMethod]
        public void Issue114_FftFft_MatchesNumPy()
        {
            var s = np.fft.fft(np.array(new[] { 0.0, 1, 0, -1 }));
            np.allclose(np.real(s), np.zeros(new Shape(4))).Should().BeTrue();
            np.allclose(np.imag(s), np.array(new[] { 0.0, -2, 0, 2 })).Should().BeTrue();
        }

        // ── Creation / manipulation / indexing / sorting / sets ───────────────────────────────────

        // Issue #220 - np.flip
        [TestMethod]
        public void Issue220_Flip_MatchesNumPy()
        {
            var m = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            np.array_equal(np.flip(m), np.array(new[,] { { 6, 5, 4 }, { 3, 2, 1 } })).Should().BeTrue();
            np.array_equal(np.flip(m, axis: 1), np.array(new[,] { { 3, 2, 1 }, { 6, 5, 4 } })).Should().BeTrue();
        }

        // Issue #221 - np.rot90
        [TestMethod]
        public void Issue221_Rot90_MatchesNumPy()
        {
            var m = np.arange(4).reshape(2, 2);
            np.array_equal(np.rot90(m, k: 1), np.array(new[,] { { 1, 3 }, { 0, 2 } })).Should().BeTrue();
        }

        // Issue #450 - np.diag (build and extract)
        [TestMethod]
        public void Issue450_Diag_MatchesNumPy()
        {
            var m = np.diag(np.array(new[] { 1, 2, 3 }));
            np.array_equal(m, np.array(new[,] { { 1, 0, 0 }, { 0, 2, 0 }, { 0, 0, 3 } })).Should().BeTrue();
            np.array_equal(np.diag(m), np.array(new[] { 1, 2, 3 })).Should().BeTrue();
        }

        // Issue #622 - set routines
        [TestMethod]
        public void Issue622_Sets_MatchNumPy()
        {
            np.array_equal(np.isin(np.array(new[] { 1, 2, 3, 4 }), np.array(new[] { 2, 4 })),
                           np.array(new[] { false, true, false, true })).Should().BeTrue();
            np.array_equal(np.intersect1d(np.array(new[] { 1, 2, 3 }), np.array(new[] { 2, 3, 4 })),
                           np.array(new[] { 2, 3 })).Should().BeTrue();
        }

        // Issue #623 - sorting & searching
        [TestMethod]
        public void Issue623_Sorting_MatchesNumPy()
        {
            var a = np.array(new[] { 5, 3, 1, 4, 2 });
            np.array_equal(np.sort(np.partition(a, 2)), np.sort(a)).Should().BeTrue();
            np.nanargmax(np.array(new[] { 1.0, double.NaN, 3 })).Should().Be(2L);
        }

        // Issue #624 - np.take_along_axis
        [TestMethod]
        public void Issue624_TakeAlongAxis_MatchesNumPy()
        {
            var a = np.array(new[,] { { 3, 1, 2 }, { 6, 5, 4 } });
            var idx = np.argsort(a, axis: 1);
            np.array_equal(np.take_along_axis(a, idx, axis: 1), np.sort(a, axis: 1)).Should().BeTrue();
        }

        // Issue #546 - F-order layout support
        [TestMethod]
        public void Issue546_Fortran_MatchesNumPy()
        {
            var f = np.asfortranarray(np.arange(6).reshape(2, 3));
            f.flags.f_contiguous.Should().BeTrue();
        }

        // Issue #610 - axis reductions preserve F-contiguity
        [TestMethod]
        public void Issue610_FortranPreserved_MatchesNumPy()
        {
            var f = np.asfortranarray(np.ones(new Shape(4, 3)));
            np.sum(f, axis: 0).flags.f_contiguous.Should().BeTrue();
        }

        // ── Iteration ────────────────────────────────────────────────────────────────────────────

        // Issue #351 - typed np.nditer<T>
        [TestMethod]
        public void Issue351_NditerTyped_MatchesNumPy()
        {
            var a = np.array(new[] { 1.0, 2, 3, 4 });
            foreach (ref double x in np.nditer<double>(a, writeable: true))
                x *= 2;
            np.array_equal(a, np.array(new[] { 2.0, 4, 6, 8 })).Should().BeTrue();
        }

        // Issue #363 - np.nested_iters (per-axis-subset nested loops)
        [TestMethod]
        public void Issue363_NestedIters_MatchesNumPy()
        {
            var iters = np.nested_iters(np.arange(6).reshape(2, 3), new[] { new[] { 0 }, new[] { 1 } });
            iters.Length.Should().Be(2);
        }

        // ── Core / performance / architecture ─────────────────────────────────────────────────────

        // Issue #541 - np.evaluate fusion (IL emission)
        [TestMethod]
        public void Issue541_Evaluate_MatchesNumPy()
        {
            var a = np.array(new[] { 1.0, 2, 3 });
            var b = np.array(new[] { 4.0, 5, 6 });
            np.allclose(np.evaluate((NDExpr)a * b + 2), np.array(new[] { 6.0, 12, 20 })).Should().BeTrue();
        }

        // Issue #585 - IL cast kernels (astype)
        [TestMethod]
        public void Issue585_Astype_MatchesNumPy()
        {
            var f = np.arange(5).astype(np.float32);
            f.dtype.Should().Be(typeof(float));
            np.array_equal(f, np.array(new float[] { 0, 1, 2, 3, 4 })).Should().BeTrue();
        }

        // Issue #586 - IL per-chunk iteration (nditer_chunks)
        [TestMethod]
        public void Issue586_NditerChunks_MatchesNumPy()
        {
            var a = np.array(new[] { 1.0, 2, 3, 4 });
            foreach (Span<double> chunk in np.nditer_chunks<double>(a, writeable: true))
                for (int i = 0; i < chunk.Length; i++)
                    chunk[i] += 10;
            np.array_equal(a, np.array(new[] { 11.0, 12, 13, 14 })).Should().BeTrue();
        }

        // Issue #576 - SIMD argmax + axis reductions
        [TestMethod]
        public void Issue576_ArgmaxAndAxis_MatchNumPy()
        {
            np.argmax(np.array(new[] { 1.0, 9, 3, 2 })).Should().Be(1L);
            var m = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            np.array_equal(np.sum(m, axis: 0), np.array(new[] { 5, 7, 9 })).Should().BeTrue();
        }

        // ── File I/O ─────────────────────────────────────────────────────────────────────────────

        // Issue #517 - loading a scalar .npy
        [TestMethod]
        public void Issue517_SaveLoadScalar_MatchesNumPy()
        {
            var path = Path.Combine(Path.GetTempPath(), "ns_scalar_" + Guid.NewGuid().ToString("N") + ".npy");
            try
            {
                np.save(path, np.array(5.0));
                var x = np.load_npy(path);
                x.ndim.Should().Be(0);
                x.GetDouble(0).Should().Be(5.0);
            }
            finally { try { File.Delete(path); } catch (IOException) { } }
        }

        // Issue #553 - default_rng reproducible streams (NEP-19)
        [TestMethod]
        public void Issue553_DefaultRng_IsDeterministic()
        {
            var a = np.random.default_rng(1234).standard_normal(new Shape(3));
            var b = np.random.default_rng(1234).standard_normal(new Shape(3));
            np.array_equal(a, b).Should().BeTrue();
        }

        // ── Already covered above and retained from the first batch ────────────────────────────────

        // Issue #438 - np.linalg.inv (managed LU, no backend required)
        [TestMethod]
        public void Issue438_Inv_MatchesNumPy()
        {
            var a = np.array(new double[,] { { 4.0, 7 }, { 2, 6 } });
            var expected = np.array(new double[,] { { 0.6, -0.7 }, { -0.2, 0.4 } });
            np.allclose(np.linalg.inv(a), expected).Should().BeTrue();
        }

        // Issue #591 - ndim beyond NumPy's NPY_MAXDIMS = 64
        [TestMethod]
        public void Issue591_HighNdim_ExceedsNumPyCap()
        {
            var dims = Enumerable.Repeat(1, 100).Append(2).ToArray();
            var a = np.ones(new Shape(dims));
            a.ndim.Should().Be(101);
            np.sum(a, axis: 70).ndim.Should().Be(100);
        }

        // ── Second wave (approved batch) ───────────────────────────────────────────────────────────

        // Issue #613 - NDArray is IDisposable; NDScope reclaims transients and Returns the survivor
        private static NDArray ComputeInScope()
        {
            using var s = NDScope.Open();
            var a = np.arange(5);
            var doubled = a * 2;          // transient - returned to the pool at scope exit
            return s.Returns(doubled);    // survives
        }

        [TestMethod]
        public void Issue613_NDScope_ReturnsSurvivor()
        {
            (np.arange(3) is IDisposable).Should().BeTrue();

            var x = np.arange(3);
            x.Dispose();
            new Action(() => x.Dispose()).Should().NotThrow("Dispose is idempotent");

            using var r = ComputeInScope();
            np.array_equal(r, np.array(new[] { 0, 2, 4, 6, 8 })).Should().BeTrue();
        }

        // Issue #386 - np.loadtxt reads CSV (delimiter/skiprows/usecols/unpack/lines)
        [TestMethod]
        public void Issue386_Loadtxt_Csv()
        {
            var p = Path.Combine(Path.GetTempPath(), "ns_csv_" + Guid.NewGuid().ToString("N") + ".csv");
            try
            {
                File.WriteAllText(p, "99,99,99\n1,2,3\n4,5,6\n");   // first row is a header to skip
                np.array_equal(np.loadtxt(p, delimiter: ",", skiprows: 1),
                               np.array(new[,] { { 1.0, 2, 3 }, { 4, 5, 6 } })).Should().BeTrue();
                np.array_equal(np.loadtxt(p, delimiter: ",", skiprows: 1, usecols: new[] { 0, 2 }),
                               np.array(new[,] { { 1.0, 3 }, { 4, 6 } })).Should().BeTrue();
                np.array_equal(np.loadtxt(p, delimiter: ",", skiprows: 1, unpack: true),
                               np.array(new[,] { { 1.0, 4 }, { 2, 5 }, { 3, 6 } })).Should().BeTrue();
                np.array_equal(np.loadtxt(new List<string> { "10,20", "30,40" }, delimiter: ","),
                               np.array(new[,] { { 10.0, 20 }, { 30, 40 } })).Should().BeTrue();
            }
            finally { try { File.Delete(p); } catch (IOException) { } }
        }

        // Issue #483 - stack / array from a List<NDArray>
        [TestMethod]
        public void Issue483_StackAndArray_FromList()
        {
            var list = new List<NDArray> { np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }) };

            var s0 = np.stack(list.ToArray());
            s0.shape.Should().Equal(new long[] { 2, 3 });
            np.array_equal(s0, np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } })).Should().BeTrue();

            np.stack(list.ToArray(), axis: 1).shape.Should().Equal(new long[] { 3, 2 });
            np.array(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } }).shape.Should().Equal(new long[] { 2, 3 });
        }

        // Issue #390 - np.frombuffer over a byte buffer (dtype / count / offset)
        [TestMethod]
        public void Issue390_Frombuffer_Bytes()
        {
            byte[] raw = new byte[24];
            BitConverter.GetBytes(1.0).CopyTo(raw, 0);
            BitConverter.GetBytes(2.0).CopyTo(raw, 8);
            BitConverter.GetBytes(3.0).CopyTo(raw, 16);
            np.array_equal(np.frombuffer(raw, "float64"), np.array(new[] { 1.0, 2, 3 })).Should().BeTrue();
            np.array_equal(np.frombuffer(raw, "float64", count: 2, offset: 8), np.array(new[] { 2.0, 3 })).Should().BeTrue();

            byte[] ints = new byte[8];
            BitConverter.GetBytes(7).CopyTo(ints, 0);
            BitConverter.GetBytes(9).CopyTo(ints, 4);
            np.array_equal(np.frombuffer(ints, "int32"), np.array(new[] { 7, 9 })).Should().BeTrue();
        }

        // Issue #493 - N-D arrays print with their real shape, not flattened to 1-D
        [TestMethod]
        public void Issue493_Printing_NdNotFlattened()
        {
            var s = np.arange(24).reshape(2, 3, 4).ToString();
            s.Should().Contain("[[[");
            s.Should().Contain("\n");
            np.arange(6).reshape(2, 3).ToString().Should().Contain("\n");   // 2-D is multi-line
            np.arange(3).ToString().Should().NotContain("[[");              // 1-D stays flat
        }

        // Issue #211 - np.interp (interior / clamp / left / period)
        [TestMethod]
        public void Issue211_Interp()
        {
            var xp = np.array(new[] { 1.0, 2, 3 });
            var fp = np.array(new[] { 10.0, 20, 30 });
            np.interp(np.array(new[] { 2.5 }), xp, fp).GetDouble(0).Should().BeApproximately(25.0, 1e-9);
            np.interp(np.array(new[] { 0.5 }), xp, fp).GetDouble(0).Should().BeApproximately(10.0, 1e-9);   // clamp low
            np.interp(np.array(new[] { 3.5 }), xp, fp).GetDouble(0).Should().BeApproximately(30.0, 1e-9);   // clamp high
            np.interp(np.array(new[] { 0.5 }), xp, fp, left: -1).GetDouble(0).Should().BeApproximately(-1.0, 1e-9);
            np.interp(np.array(new[] { 7.0 }), xp, fp, period: 6).GetDouble(0).Should().BeApproximately(10.0, 1e-9);
        }

        // Issue #326 - np.load(mmap_mode:) read-only and read-write mappings
        [TestMethod]
        public void Issue326_Mmap_ReadOnlyAndReadWrite()
        {
            var p = Path.Combine(Path.GetTempPath(), "ns_mmap_" + Guid.NewGuid().ToString("N") + ".npy");
            try
            {
                np.save(p, np.arange(5));

                var r = (NDArray)np.load(p, mmap_mode: "r");
                r.flags.writeable.Should().BeFalse();
                np.array_equal(r, np.array(new[] { 0, 1, 2, 3, 4 })).Should().BeTrue();
                r.Dispose();

                var rw = (NDArray)np.load(p, mmap_mode: "r+");
                rw.flags.writeable.Should().BeTrue();
                rw.Dispose();
            }
            finally { try { File.Delete(p); } catch (IOException) { } }
        }

        // Issue #577 - floor / ceil / trunc / round (half-to-even)
        [TestMethod]
        public void Issue577_FloorCeilTruncRound()
        {
            np.array_equal(np.floor(np.array(new[] { 1.5f, 2.5f, -1.5f })), np.array(new[] { 1f, 2f, -2f })).Should().BeTrue();
            np.array_equal(np.ceil(np.array(new[] { 1.1f, 2.9f })), np.array(new[] { 2f, 3f })).Should().BeTrue();
            np.array_equal(np.trunc(np.array(new[] { 1.9, -1.9 })), np.array(new[] { 1.0, -1 })).Should().BeTrue();
            np.array_equal(np.around(np.array(new[] { 1.5, 2.5, 3.5 })), np.array(new[] { 2.0, 2, 4 })).Should().BeTrue();
        }

        // Issue #496 - polyfit (1-D) and a polynomial surface via lstsq (OpenBLAS)
        [TestMethod]
        public void Issue496_Polyfit_And_SurfaceLstsq()
        {
            if (!TryEnableOpenBlas()) { Assert.Inconclusive("OpenBLAS backend not available on this host."); return; }
            try
            {
                NDArray coeffs = np.polyfit(np.array(new[] { 0.0, 1, 2, 3 }), np.array(new[] { 1.0, 3, 5, 7 }), 1);
                np.allclose(coeffs, np.array(new[] { 2.0, 1 })).Should().BeTrue();

                var A = np.array(new[,] { { 1.0, 0, 0 }, { 1, 1, 0 }, { 1, 0, 1 }, { 1, 1, 1 } });
                var z = np.array(new[] { 1.0, 3, 4, 6 });
                np.allclose(np.linalg.lstsq(A, z).Solution, np.array(new[] { 1.0, 2, 3 })).Should().BeTrue();
            }
            finally { OpenBlasEngine.Disable(); }
        }

        // Issue #510 - save a nested dict with savez: (1) flatten keys, (2) JSON manifest entry
        [TestMethod]
        public void Issue510_SaveNestedDict_TwoWays()
        {
            var d1 = np.array(new[] { 1.0, 2 });
            var d2 = np.array(new[] { 3.0, 4 });

            // Solution 1 - flatten keys with "/"
            var p1 = Path.Combine(Path.GetTempPath(), "ns_nest1_" + Guid.NewGuid().ToString("N") + ".npz");
            try
            {
                np.savez(p1, new Dictionary<string, NDArray> { ["port1/data"] = d1, ["port2/data"] = d2 });
                using var z = np.load_npz(p1);
                np.array_equal(z["port1/data"], d1).Should().BeTrue();
                np.array_equal(z["port2/data"], d2).Should().BeTrue();
            }
            finally { try { File.Delete(p1); } catch (IOException) { } }

            // Solution 2 - flat arrays + a JSON structure manifest stored inside the npz
            var p2 = Path.Combine(Path.GetTempPath(), "ns_nest2_" + Guid.NewGuid().ToString("N") + ".npz");
            try
            {
                var flat = new Dictionary<string, NDArray> { ["a0"] = d1, ["a1"] = d2 };
                string json = "{\"port1\":{\"data\":\"a0\"},\"port2\":{\"data\":\"a1\"}}";
                flat["__structure__"] = np.frombuffer(Encoding.UTF8.GetBytes(json), NPTypeCode.Byte);
                np.savez(p2, flat);

                using var z = np.load_npz(p2);
                string back = Encoding.UTF8.GetString(z["__structure__"].ToArray<byte>());
                var root = JsonDocument.Parse(back).RootElement;
                np.array_equal(z[root.GetProperty("port1").GetProperty("data").GetString()], d1).Should().BeTrue();
                np.array_equal(z[root.GetProperty("port2").GetProperty("data").GetString()], d2).Should().BeTrue();
            }
            finally { try { File.Delete(p2); } catch (IOException) { } }
        }

        // ── Third wave (approved batch) ────────────────────────────────────────────────────────────

        // Issue #116 - OpenBLAS is the pluggable provider (MKL alternative)
        [TestMethod]
        public void Issue116_OpenBlasProvider()
        {
            if (!TryEnableOpenBlas()) { Assert.Inconclusive("OpenBLAS backend not available on this host."); return; }
            try
            {
                OpenBlasEngine.Enabled.Should().BeTrue();
                var a = np.array(new[,] { { 1.0, 2 }, { 3, 4 } });
                var b = np.array(new[,] { { 5.0, 6 }, { 7, 8 } });
                np.allclose(np.matmul(a, b), np.array(new[,] { { 19.0, 22 }, { 43, 50 } })).Should().BeTrue();
            }
            finally { OpenBlasEngine.Disable(); }
        }

        // Issue #340 - int->long: strides in bytes (npy_intp), size/strides are long
        [TestMethod]
        public void Issue340_LongIndexingSurface()
        {
            var a = np.arange(6).reshape(2, 3);           // int64, C-order
            a.strides.Should().Equal(new long[] { 24, 8 });   // BYTES per axis
            (a.strides is long[]).Should().BeTrue();
        }

        // Issue #587 - IL-emitted kernels drive elementwise + cast across dtypes
        [TestMethod]
        public void Issue587_ILKernels_CrossDtype()
        {
            var f = np.arange(5).astype(np.float32);
            f.dtype.Should().Be(typeof(float));
            np.array_equal(f, np.array(new float[] { 0, 1, 2, 3, 4 })).Should().BeTrue();

            var g = (np.arange(5) * 2).astype(np.int16);
            np.array_equal(g, np.array(new short[] { 0, 2, 4, 6, 8 })).Should().BeTrue();
        }

        // Issue #465 - bridge to another array lib via the byte buffer (tobytes -> frombuffer)
        [TestMethod]
        public void Issue465_BufferBridge()
        {
            var a = np.array(new[] { 1.0, 2, 3 });
            np.array_equal(np.frombuffer(a.tobytes(), "float64"), a).Should().BeTrue();
        }

        // Issue #559 - default_rng / PCG64 reproducible streams
        [TestMethod]
        public void Issue559_DefaultRng_Integers()
        {
            var a = np.random.default_rng(7).integers(0, 100, new Shape(5));
            var b = np.random.default_rng(7).integers(0, 100, new Shape(5));
            np.array_equal(a, b).Should().BeTrue();
        }

        // Issue #552 - NEP21: take_along_axis + fancy-set into a non-contiguous destination
        [TestMethod]
        public void Issue552_AdvancedIndexing()
        {
            var a = np.arange(6).reshape(2, 3).T;   // (3,2) non-contiguous view
            a[np.array(new[] { 0, 2 })] = np.array(new[,] { { 100, 101 }, { 102, 103 } });
            np.array_equal(a, np.array(new[,] { { 100, 101 }, { 1, 4 }, { 102, 103 } })).Should().BeTrue();

            var m = np.array(new[,] { { 3, 1, 2 }, { 6, 5, 4 } });
            np.array_equal(np.take_along_axis(m, np.argsort(m, axis: 1), axis: 1), np.sort(m, axis: 1)).Should().BeTrue();
        }

        // ── Fourth wave (verify-then-close) ────────────────────────────────────────────────────────

        // Issue #369 - slicing (basic / ellipsis / negative / step) works
        [TestMethod]
        public void Issue369_Slicing_Works()
        {
            var m = np.arange(24).reshape(2, 3, 4);
            m["1"].shape.Should().Equal(new long[] { 3, 4 });
            m[":, 0"].shape.Should().Equal(new long[] { 2, 4 });
            np.array_equal(m["..., -1"], np.array(new[,] { { 3, 7, 11 }, { 15, 19, 23 } })).Should().BeTrue();
            np.array_equal(m["1, 1:3, ::2"], np.array(new[,] { { 16, 18 }, { 20, 22 } })).Should().BeTrue();
        }

        // Issue #484 - loading a float64 matrix .npy (the class of the file attached to the issue)
        [TestMethod]
        public void Issue484_LoadFloat64Matrix_Npy()
        {
            var p = Path.Combine(Path.GetTempPath(), "ns_484_" + Guid.NewGuid().ToString("N") + ".npy");
            try
            {
                var original = np.arange(56 * 56).astype(np.float64).reshape(56, 56);
                np.save(p, original);
                var loaded = np.load_npy(p);
                loaded.dtype.Should().Be(typeof(double));
                loaded.shape.Should().Equal(new long[] { 56, 56 });
                np.array_equal(loaded, original).Should().BeTrue();
            }
            finally { try { File.Delete(p); } catch (IOException) { } }
        }
    }
}
