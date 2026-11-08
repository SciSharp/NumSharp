using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyAlgebra.Bases.cs — the per-basis algorithms of numpy.polynomial's series algebra (plan U2)
// =============================================================================
//
// NumPy 2.4.2's per-basis Python, statement for statement, over the engine of NDPolyAlgebra.cs:
//   * the three-term-recurrence products legmul / lagmul / hermmul / hermemul (a backward recurrence over SERIES);
//   * polydiv's synthetic division and chebdiv's z-series division (chebyshev._zseries_div);
//   * the ten basis conversions: X2poly (a recurrence over power-series ops) and poly2X (Horner over X's ops).
// Each method's comments quote the NumPy source it replays. Every allocation goes to the call's arena; each method
// hands its result back through Keep, so only live series survive a Python-level iteration.
//
// =============================================================================

namespace NumSharp
{
    internal static unsafe partial class NDPolyAlgebra
    {
        // =============================================================================================
        //  legmul / lagmul / hermmul / hermemul
        // =============================================================================================

        /// <summary>
        ///     The four recurrence products after as_series (<paramref name="c1"/>, <paramref name="c2"/> trimmed, one dtype t):
        /// <code>
        /// if len(c1) > len(c2): c = c2; xs = c1
        /// else:                 c = c1; xs = c2
        /// if len(c) == 1:   c0 = c[0] * xs; c1 = 0
        /// elif len(c) == 2: c0 = c[0] * xs; c1 = c[1] * xs
        /// else:
        ///     nd = len(c); c0 = c[-2] * xs; c1 = c[-1] * xs
        ///     for i in range(3, len(c) + 1):
        ///         tmp = c0; nd = nd - 1
        ///         c0 = Xsub(c[-i] * xs, STEP0(c1, nd))
        ///         c1 = Xadd(tmp, STEP1(c1, nd))
        /// return FINAL(c0, c1)
        /// </code>
        ///     with, per basis (NumPy's exact spellings):
        /// <code>
        ///            STEP0                   STEP1                                         FINAL
        /// legmul     (c1*(nd-1))/nd          (legmulx(c1)*(2*nd-1))/nd                     legadd(c0, legmulx(c1))
        /// lagmul     (c1*(nd-1))/nd          lagsub((2*nd-1)*c1, lagmulx(c1))/nd           lagadd(c0, lagsub(c1, lagmulx(c1)))
        /// hermmul    c1*(2*(nd-1))           hermmulx(c1)*2                                hermadd(c0, hermmulx(c1)*2)
        /// hermemul   c1*(nd-1)               hermemulx(c1)                                 hermeadd(c0, hermemulx(c1))
        /// </code>
        ///     The Python int <c>c1 = 0</c> of the one-term branch reaches FINAL only through as_series (inside Xmulx / Xsub),
        ///     where it is the float64 <c>[0.]</c> — so a float16/float32 product with a one-term factor is float64.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">Legendre, Laguerre, Hermite or HermiteE.</param>
        /// <param name="c1">First factor (as_series'd).</param>
        /// <param name="c2">Second factor (as_series'd, same dtype).</param>
        /// <returns>The product (fresh; a trimseq slice when trailing zeros were trimmed).</returns>
        private static PolySer RecurrenceMul(PolyArena a, PolyBasis basis, in PolySer c1, in PolySer c2)
        {
            var mark = a.Position;
            PolySer c, xs;
            if (c1.N > c2.N) { c = c2; xs = c1; }
            else { c = c1; xs = c2; }
            long n = c.N;
            PolySer p0, p1;   // NumPy's c0 and c1 (renamed: c1 is also the argument)
            if (n == 1)
            {
                p0 = ScalarOp(BinaryOp.Multiply, c.At(0), xs, New(a, xs.N, xs.T).P);
                p1 = PythonZero(a);
            }
            else if (n == 2)
            {
                p0 = ScalarOp(BinaryOp.Multiply, c.At(0), xs, New(a, xs.N, xs.T).P);
                p1 = ScalarOp(BinaryOp.Multiply, c.At(1), xs, New(a, xs.N, xs.T).P);
            }
            else
            {
                long nd = n;
                p0 = ScalarOp(BinaryOp.Multiply, c.At(n - 2), xs, New(a, xs.N, xs.T).P);
                p1 = ScalarOp(BinaryOp.Multiply, c.At(n - 1), xs, New(a, xs.N, xs.T).P);
                var loop = a.Position;
                for (long i = 3; i <= n; i++)
                {
                    var tmp = p0;
                    nd--;
                    var ck = ScalarOp(BinaryOp.Multiply, c.At(n - i), xs, New(a, xs.N, xs.T).P);   // c[-i] * xs
                    PolySer s0, s1;
                    switch (basis)
                    {
                        case PolyBasis.Legendre:
                        {
                            s0 = OpInt(BinaryOp.Multiply, p1, nd - 1, New(a, p1.N, p1.T).P);            // c1 * (nd - 1)
                            s0 = OpInt(BinaryOp.Divide, s0, nd, s0.P);                                   //   ... / nd
                            var mx = Mulx(a, basis, p1);                                                 // legmulx(c1)
                            s1 = OpInt(BinaryOp.Multiply, mx, 2 * nd - 1, mx.P);                         //   ... * (2*nd - 1)
                            s1 = OpInt(BinaryOp.Divide, s1, nd, s1.P);                                   //   ... / nd
                            break;
                        }
                        case PolyBasis.Laguerre:
                        {
                            s0 = OpInt(BinaryOp.Multiply, p1, nd - 1, New(a, p1.N, p1.T).P);            // c1 * (nd - 1)
                            s0 = OpInt(BinaryOp.Divide, s0, nd, s0.P);                                   //   ... / nd
                            var k = IntOp(BinaryOp.Multiply, 2 * nd - 1, p1, New(a, p1.N, p1.T).P);      // (2*nd - 1) * c1
                            var mx = Mulx(a, basis, p1);                                                 // lagmulx(c1)
                            s1 = AddSub(a, k, mx, subtract: true);                                       // lagsub(...)
                            s1 = OpInt(BinaryOp.Divide, s1, nd, New(a, s1.N, s1.T).P);                   //   ... / nd
                            break;
                        }
                        case PolyBasis.Hermite:
                        {
                            s0 = OpInt(BinaryOp.Multiply, p1, 2 * (nd - 1), New(a, p1.N, p1.T).P);      // c1 * (2*(nd - 1))
                            var mx = Mulx(a, basis, p1);                                                 // hermmulx(c1)
                            s1 = OpInt(BinaryOp.Multiply, mx, 2, mx.P);                                  //   ... * 2
                            break;
                        }
                        default:   // HermiteE
                        {
                            s0 = OpInt(BinaryOp.Multiply, p1, nd - 1, New(a, p1.N, p1.T).P);            // c1 * (nd - 1)
                            s1 = Mulx(a, basis, p1);                                                     // hermemulx(c1)
                            break;
                        }
                    }
                    p0 = AddSub(a, ck, s0, subtract: true);    // c0 = Xsub(c[-i] * xs, STEP0)
                    p1 = AddSub(a, tmp, s1, subtract: false);  // c1 = Xadd(tmp, STEP1)
                    Keep(a, loop, ref p0, ref p1);
                }
            }

            // FINAL(c0, c1)
            PolySer r;
            var fm = Mulx(a, basis, p1);
            switch (basis)
            {
                case PolyBasis.Laguerre:
                    r = AddSub(a, p0, AddSub(a, p1, fm, subtract: true), subtract: false);   // lagadd(c0, lagsub(c1, lagmulx(c1)))
                    break;
                case PolyBasis.Hermite:
                    r = AddSub(a, p0, OpInt(BinaryOp.Multiply, fm, 2, fm.P), subtract: false); // hermadd(c0, hermmulx(c1) * 2)
                    break;
                default:
                    r = AddSub(a, p0, fm, subtract: false);                                   // Xadd(c0, Xmulx(c1))
                    break;
            }
            Keep(a, mark, ref r);
            return r;
        }

        // =============================================================================================
        //  polydiv / chebdiv
        // =============================================================================================

        /// <summary>
        ///     polydiv past its short cuts (lc1 ≥ lc2 ≥ 2) — synthetic division in place on the as_series copy of c1:
        /// <code>
        /// dlen = lc1 - lc2; scl = c2[-1]; c2 = c2[:-1] / scl
        /// i = dlen; j = lc1 - 1
        /// while i >= 0:
        ///     c1[i:j] -= c2 * c1[j]; i -= 1; j -= 1
        /// return c1[j + 1:] / scl, pu.trimseq(c1[:j + 1])
        /// </code>
        ///     The remainder is a VIEW of the updated c1 (<c>c1[:j+1]</c>, j = lc2 - 2), as NumPy returns it.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="c1">Dividend (the as_series copy; updated in place, as NumPy updates its copy).</param>
        /// <param name="c2">Divisor (as_series copy).</param>
        /// <returns>(quotient, remainder).</returns>
        private static (PolySer quo, PolySer rem) PolyDiv(PolyArena a, in PolySer c1, in PolySer c2)
        {
            long lc1 = c1.N, lc2 = c2.N, dlen = lc1 - lc2;
            ulong* scl = stackalloc ulong[2];
            CastElement(c2.At(lc2 - 1), c2.T, c2.T, (byte*)scl);                                 // scl = c2[-1]
            var d = OpScalar(BinaryOp.Divide, c2.Sub(0, lc2 - 1), (byte*)scl, New(a, lc2 - 1, c2.T).P);   // c2[:-1] / scl
            var tmp = New(a, lc2 - 1, c1.T);
            long i = dlen, j = lc1 - 1;
            while (i >= 0)
            {
                // c1[i:j] -= c2 * c1[j]: the product is a fresh array (c1[j] lies outside c1[i:j]), then the in-place subtract.
                OpScalar(BinaryOp.Multiply, d, c1.At(j), tmp.P);
                House(BinaryOp.Subtract, ExecutionPath.SimdFull, c1.T, c1.At(i), tmp.P, c1.At(i), lc2 - 1);
                i--;
                j--;
            }
            var quo = OpScalar(BinaryOp.Divide, c1.Sub(j + 1, lc1 - (j + 1)), (byte*)scl, New(a, lc1 - (j + 1), c1.T).P);
            // pu.trimseq(c1[:j + 1]) — c1[:j+1] is a view of c1, and so is whatever trimseq returns.
            var head = new PolySer(c1.P, j + 1, c1.T, lc1, true);
            var rem = TrimSeq(head);
            return (quo, new PolySer(rem.P, rem.N, rem.T, lc1, true));
        }

        /// <summary>
        ///     chebdiv past its short cuts: <c>z1 = c2z(c1); z2 = c2z(c2); quo, rem = _zseries_div(z1, z2);
        ///     return trimseq(z2c(quo)), trimseq(z2c(rem))</c>.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="c1">Dividend (lc1 ≥ lc2 ≥ 2).</param>
        /// <param name="c2">Divisor.</param>
        /// <returns>(quotient, remainder).</returns>
        private static (PolySer quo, PolySer rem) ChebDiv(PolyArena a, in PolySer c1, in PolySer c2)
        {
            var mark = a.Position;
            var (zq, zr) = ZSeriesDiv(a, CToZ(a, c1), CToZ(a, c2));
            var quo = TrimSeq(ZToC(a, zq));
            var rem = TrimSeq(ZToC(a, zr));
            Keep(a, mark, ref quo, ref rem);
            return (quo, rem);
        }

        /// <summary>
        ///     <c>chebyshev._zseries_div(z1, z2)</c> for the lengths chebdiv reaches (lc1 ≥ lc2 ≥ 3):
        /// <code>
        /// z1 = z1.copy(); z2 = z2.copy(); dlen = lc1 - lc2
        /// scl = z2[0]; z2 /= scl
        /// quo = np.empty(dlen + 1); i = 0; j = dlen
        /// while i &lt; j:
        ///     r = z1[i]; quo[i] = z1[i]; quo[dlen - i] = r
        ///     tmp = r * z2; z1[i:i + lc2] -= tmp; z1[j:j + lc2] -= tmp
        ///     i += 1; j -= 1
        /// r = z1[i]; quo[i] = r; tmp = r * z2; z1[i:i + lc2] -= tmp
        /// quo /= scl
        /// rem = z1[i + 1:i - 1 + lc2].copy()
        /// </code>
        ///     Every update is the array op NumPy runs, in its order (the two subtractions of one step may overlap).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="z1In">Dividend z-series.</param>
        /// <param name="z2In">Divisor z-series.</param>
        /// <returns>(quotient z-series, remainder z-series), fresh.</returns>
        private static (PolySer quo, PolySer rem) ZSeriesDiv(PolyArena a, in PolySer z1In, in PolySer z2In)
        {
            var z1 = Copy(a, z1In);
            var z2 = Copy(a, z2In);
            long lc1 = z1.N, lc2 = z2.N, dlen = lc1 - lc2;
            ulong* scl = stackalloc ulong[2];
            ulong* r = stackalloc ulong[2];
            CastElement(z2.At(0), z2.T, z2.T, (byte*)scl);                        // scl = z2[0] (a NumPy scalar: a copy)
            OpScalar(BinaryOp.Divide, z2, (byte*)scl, z2.P);                        // z2 /= scl
            var quo = New(a, dlen + 1, z1.T);
            var tmp = New(a, lc2, z1.T);
            long i = 0, j = dlen;
            while (i < j)
            {
                CastElement(z1.At(i), z1.T, z1.T, (byte*)r);                        // r = z1[i]
                CastElement(z1.At(i), z1.T, z1.T, quo.At(i));                       // quo[i] = z1[i]
                CastElement((byte*)r, z1.T, z1.T, quo.At(dlen - i));                // quo[dlen - i] = r
                ScalarOp(BinaryOp.Multiply, (byte*)r, z2, tmp.P);                   // tmp = r * z2
                House(BinaryOp.Subtract, ExecutionPath.SimdFull, z1.T, z1.At(i), tmp.P, z1.At(i), lc2);   // z1[i:i+lc2] -= tmp
                House(BinaryOp.Subtract, ExecutionPath.SimdFull, z1.T, z1.At(j), tmp.P, z1.At(j), lc2);   // z1[j:j+lc2] -= tmp
                i++;
                j--;
            }
            CastElement(z1.At(i), z1.T, z1.T, (byte*)r);                            // r = z1[i]
            CastElement((byte*)r, z1.T, z1.T, quo.At(i));                           // quo[i] = r
            ScalarOp(BinaryOp.Multiply, (byte*)r, z2, tmp.P);                       // tmp = r * z2
            House(BinaryOp.Subtract, ExecutionPath.SimdFull, z1.T, z1.At(i), tmp.P, z1.At(i), lc2);       // z1[i:i+lc2] -= tmp
            OpScalar(BinaryOp.Divide, quo, (byte*)scl, quo.P);                      // quo /= scl
            var rem = Copy(a, z1.Sub(i + 1, lc2 - 2));                              // z1[i + 1:i - 1 + lc2].copy()
            return (quo, rem);
        }

        // =============================================================================================
        //  X2poly / poly2X
        // =============================================================================================

        /// <summary>
        ///     <c>X2poly(c)</c> after as_series — a recurrence over POWER-series ops (<c>polyadd</c>, <c>polysub</c>,
        ///     <c>polymulx</c>):
        /// <code>
        /// n = len(c); short series returned as is (cheb/leg: n &lt; 3; lag: n == 1; herm: n == 1, and n == 2 after c[1] *= 2;
        ///                                            herme: n &lt;= 2)
        /// c0 = c[-2]; c1 = c[-1]
        /// for i in range(n - 1, 1, -1):
        ///     tmp = c0
        ///     cheb:  c0 = polysub(c[i - 2], c1);               c1 = polyadd(tmp, polymulx(c1) * 2)
        ///     leg:   c0 = polysub(c[i - 2], (c1*(i-1))/i);     c1 = polyadd(tmp, (polymulx(c1)*(2*i-1))/i)
        ///     lag:   c0 = polysub(c[i - 2], (c1*(i-1))/i);     c1 = polyadd(tmp, polysub((2*i-1)*c1, polymulx(c1))/i)
        ///     herm:  c0 = polysub(c[i - 2], c1*(2*(i-1)));     c1 = polyadd(tmp, polymulx(c1) * 2)
        ///     herme: c0 = polysub(c[i - 2], c1*(i-1));         c1 = polyadd(tmp, polymulx(c1))
        /// return cheb/leg/herme: polyadd(c0, polymulx(c1)); lag: polyadd(c0, polysub(c1, polymulx(c1)));
        ///        herm: polyadd(c0, polymulx(c1) * 2)
        /// </code>
        ///     c0 and c1 start as NumPy SCALARS (<c>c[-2]</c>, <c>c[-1]</c>), so the first step's <c>(c1*(i-1))/i</c> is
        ///     scalarmath; as a one-element array op it is the same arithmetic (a real multiplier makes the naive and fused
        ///     complex products agree, and both divisions are CDOUBLE_divide), and as_series makes the scalar a one-element
        ///     array anyway.
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The source basis.</param>
        /// <param name="c">The series (as_series copy).</param>
        /// <returns>The power series.</returns>
        private static PolySer ToPowerCore(PolyArena a, PolyBasis basis, in PolySer c)
        {
            long n = c.N;
            switch (basis)
            {
                case PolyBasis.Chebyshev or PolyBasis.Legendre when n < 3:
                case PolyBasis.Laguerre when n == 1:
                case PolyBasis.Hermite when n == 1:
                case PolyBasis.HermiteE when n <= 2:
                    return c;
                case PolyBasis.Hermite when n == 2:
                {
                    // c[1] *= 2 — scalarmath on the element, stored back into the as_series copy (returned as is).
                    ulong* two = stackalloc ulong[2];
                    WeakInt(2, c.T, (byte*)two);
                    DirectILKernelGenerator.GetPolyScalarBinaryKernel(BinaryOp.Multiply, c.T, PolyComplexProduct.Naive)(c.At(1), (byte*)two, c.At(1));
                    return c;
                }
            }

            var mark = a.Position;
            PolySer p0 = c.Sub(n - 2, 1), p1 = c.Sub(n - 1, 1);   // c0 = c[-2]; c1 = c[-1]
            var loop = a.Position;
            for (long i = n - 1; i >= 2; i--)
            {
                var tmp = p0;
                var ci = c.Sub(i - 2, 1);                          // c[i - 2]
                PolySer s0, s1;
                switch (basis)
                {
                    case PolyBasis.Chebyshev:
                    {
                        s0 = p1;                                                                     // c1
                        var mx = Mulx(a, PolyBasis.Power, p1);                                       // polymulx(c1)
                        s1 = OpInt(BinaryOp.Multiply, mx, 2, mx.P);                                  //   ... * 2
                        break;
                    }
                    case PolyBasis.Legendre:
                    {
                        s0 = OpInt(BinaryOp.Multiply, p1, i - 1, New(a, p1.N, p1.T).P);             // c1 * (i - 1)
                        s0 = OpInt(BinaryOp.Divide, s0, i, s0.P);                                    //   ... / i
                        var mx = Mulx(a, PolyBasis.Power, p1);                                       // polymulx(c1)
                        s1 = OpInt(BinaryOp.Multiply, mx, 2 * i - 1, mx.P);                          //   ... * (2*i - 1)
                        s1 = OpInt(BinaryOp.Divide, s1, i, s1.P);                                    //   ... / i
                        break;
                    }
                    case PolyBasis.Laguerre:
                    {
                        s0 = OpInt(BinaryOp.Multiply, p1, i - 1, New(a, p1.N, p1.T).P);             // c1 * (i - 1)
                        s0 = OpInt(BinaryOp.Divide, s0, i, s0.P);                                    //   ... / i
                        var k = IntOp(BinaryOp.Multiply, 2 * i - 1, p1, New(a, p1.N, p1.T).P);       // (2*i - 1) * c1
                        var mx = Mulx(a, PolyBasis.Power, p1);                                       // polymulx(c1)
                        s1 = AddSub(a, k, mx, subtract: true);                                       // polysub(...)
                        s1 = OpInt(BinaryOp.Divide, s1, i, New(a, s1.N, s1.T).P);                    //   ... / i
                        break;
                    }
                    case PolyBasis.Hermite:
                    {
                        s0 = OpInt(BinaryOp.Multiply, p1, 2 * (i - 1), New(a, p1.N, p1.T).P);       // c1 * (2*(i - 1))
                        var mx = Mulx(a, PolyBasis.Power, p1);                                       // polymulx(c1)
                        s1 = OpInt(BinaryOp.Multiply, mx, 2, mx.P);                                  //   ... * 2
                        break;
                    }
                    default:   // HermiteE
                    {
                        s0 = OpInt(BinaryOp.Multiply, p1, i - 1, New(a, p1.N, p1.T).P);             // c1 * (i - 1)
                        s1 = Mulx(a, PolyBasis.Power, p1);                                           // polymulx(c1)
                        break;
                    }
                }
                p0 = AddSub(a, ci, s0, subtract: true);     // c0 = polysub(c[i - 2], STEP0)
                p1 = AddSub(a, tmp, s1, subtract: false);   // c1 = polyadd(tmp, STEP1)
                Keep(a, loop, ref p0, ref p1);
            }

            PolySer r;
            var fm = Mulx(a, PolyBasis.Power, p1);
            switch (basis)
            {
                case PolyBasis.Laguerre:
                    r = AddSub(a, p0, AddSub(a, p1, fm, subtract: true), subtract: false);   // polyadd(c0, polysub(c1, polymulx(c1)))
                    break;
                case PolyBasis.Hermite:
                    r = AddSub(a, p0, OpInt(BinaryOp.Multiply, fm, 2, fm.P), subtract: false); // polyadd(c0, polymulx(c1) * 2)
                    break;
                default:
                    r = AddSub(a, p0, fm, subtract: false);                                   // polyadd(c0, polymulx(c1))
                    break;
            }
            Keep(a, mark, ref r);
            return r;
        }

        /// <summary>
        ///     <c>poly2X(pol)</c> after as_series — Horner's scheme over X's own ops:
        /// <code>
        /// res = 0
        /// for i in range(deg, -1, -1):          (poly2lag: for p in pol[::-1])
        ///     res = Xadd(Xmulx(res), pol[i])
        /// return res
        /// </code>
        ///     <c>res = 0</c> is the Python int, so the first <c>Xmulx(0)</c> is as_series' float64 <c>[0.]</c> and the result
        ///     float64 for any real series (complex128 for a complex one).
        /// </summary>
        /// <param name="a">The arena.</param>
        /// <param name="basis">The target basis.</param>
        /// <param name="pol">The power series (as_series copy).</param>
        /// <returns>The basis series.</returns>
        private static PolySer FromPowerCore(PolyArena a, PolyBasis basis, in PolySer pol)
        {
            var mark = a.Position;
            PolySer res = PythonZero(a);
            var loop = a.Position;
            for (long i = pol.N - 1; i >= 0; i--)
            {
                var mx = Mulx(a, basis, res);
                res = AddSub(a, mx, pol.Sub(i, 1), subtract: false);   // Xadd(Xmulx(res), pol[i])
                Keep(a, loop, ref res);
            }
            Keep(a, mark, ref res);
            return res;
        }
    }
}
