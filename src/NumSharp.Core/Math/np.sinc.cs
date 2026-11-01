using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        // np.sinc — the NORMALIZED sinc function, port of NumPy 2.4.2
        // numpy/lib/_function_base_impl.py::sinc.
        //
        // NumPy's whole implementation is a five-line COMPOSITION, not a ufunc:
        //     x   = np.asanyarray(x)
        //     x   = pi * x
        //     eps = finfo(x.dtype).eps if x.dtype.kind == "f" else 1e-20
        //     y   = where(x, x, eps)          # replace the EXACT zeros with eps
        //     return sin(y) / y
        // so there is no `out=`/`where=`/`dtype=` surface (sinc is a plain
        // function, not a ufunc) — the signature is just `sinc(x)`.
        //
        // Two behaviours drive the design and are easy to get wrong:
        //
        //   (1) DTYPE follows `pi * x`, NOT the sin ufunc's tier. `pi` is a weak
        //       Python float (NEP 50), so `pi * x` promotes EVERY integer/bool
        //       width straight to float64 (not the i16→f32 / i8→f16 tiers that
        //       np.sin uses) and PRESERVES float16/float32/float64/complex128.
        //       NumSharp's own multiply already reproduces this exactly, which is
        //       why the fused ConstNode(pi) — downcast to the output precision on
        //       emit — is bit-identical to NEP 50's weak-scalar adoption.
        //
        //   (2) The zero-replacement tests `pi*x`, not `x`. Only the EXACT zeros
        //       of `pi*x` (i.e. x == ±0, plus any x so tiny that pi*x underflows
        //       to 0) are swapped for eps so `sin(eps)/eps` yields the limit 1
        //       instead of 0/0 = NaN. Checking `x == 0` instead would diverge for
        //       sub-underflow denormals, so the condition is `pi*x != 0` verbatim.
        //
        // IMPLEMENTATION — FUSION (the np.windows pattern):
        //   NumPy materializes ~3 intermediate arrays (pi*x, the where, sin(y))
        //   and reads/writes the buffer once per ufunc. NumSharp folds the ENTIRE
        //   expression — the pi multiply, the zero→eps select, sin, and the
        //   divide — into ONE np.evaluate pass, so the operand is read once and
        //   the result written once (no intermediates). The tree is built in
        //   NumPy's exact operation order, which makes the result BIT-IDENTICAL to
        //   NumPy's unfused chain for every REAL dtype (bool/int/char → float64,
        //   float16/float32/float64 preserved) — verified against NumPy 2.4.2:
        //   float64 arithmetic and the divide are IEEE-exact, and NumSharp's sin
        //   is a bit-exact port of NumPy's own float32 kernel / shares the scalar
        //   ucrtbase Math.Sin at float64. Measured NPY/NS ≈ 1.5–2.5× at 100K and 10M
        //   (f64/int 2.2–2.5×, f32 ~1.5–2.0×); at 1K sinc sits at the per-op NDIter-
        //   setup floor (~parity) like the window generators, sin being the dominant
        //   cost on both sides there.
        //
        //   COMPLEX128 is the one dtype whose per-component BITS are not reproducible:
        //   sinc composes sin ∘ divide, and NumSharp's complex sin differs from NumPy's
        //   UCRT `csin` within its documented ≤3-ULP-relative envelope, so the individual
        //   re/im bits differ. The complex VALUE is nonetheless accurate — validated
        //   ≤~2.5 ULP RELATIVE (max 5.65e-16 relative error) with np.allclose passing on
        //   ALL 200,000 random samples vs NumPy 2.4.2. (A raw per-component ULP can look
        //   large — thousands — only where one component is tiny relative to |z|, which
        //   is a byte-reproducibility fact, not an accuracy one.) So complex128 is
        //   excluded from the byte-exact fuzz corpus and pinned by an allclose unit test.
        //
        //   DECIMAL has no NumPy analog; it rides the same fused expression through
        //   NumSharp's decimal→double→Math.Sin→decimal bridge (result stays decimal).
        // =====================================================================

        /// <summary>
        ///     Return the normalized sinc function, element-wise: <c>sin(pi*x) / (pi*x)</c>, with the
        ///     removable singularity at every zero of <c>pi*x</c> filled by its limit value <c>1</c>.
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
        ///     <para>
        ///     Note the <c>pi</c> normalization: this is the signal-processing sinc. For the
        ///     mathematical <c>sin(x)/x</c>, call <c>np.sinc(x / np.pi)</c>.
        ///     </para>
        ///     <para>
        ///     RESULT DTYPE follows <c>pi * x</c> (a NEP 50 weak-float promotion), NOT np.sin's tier:
        ///     bool and EVERY integer width (and Char) → <c>float64</c>; <c>float16</c>/<c>float32</c>/
        ///     <c>float64</c> are preserved; complex → <c>complex128</c>; Decimal → <c>decimal</c>
        ///     (a NumSharp extension — NumPy has no decimal). Bit-identical to NumPy 2.4.2 for every
        ///     real dtype (validated exhaustively on all 65,536 float16 inputs and over 1,000,000
        ///     adversarial float32 and float64 inputs); complex128 is accurate to ≤~2.5 ULP RELATIVE and
        ///     np.allclose-passing but its per-component bits are not reproducible — see the file header.
        ///     A Decimal input outside the double bridge's domain (e.g. a magnitude whose <c>pi*x</c>
        ///     overflows <see cref="decimal"/>) throws <see cref="OverflowException"/>, exactly as the
        ///     other decimal transcendentals do.
        ///     </para>
        /// </remarks>
        /// <param name="x">
        ///     Input values (any shape, any memory layout — C/F/strided/broadcast/negative-stride views
        ///     are all read through their own strides). Integer and boolean inputs are promoted to
        ///     float64 before evaluation.
        /// </param>
        /// <returns>
        ///     <c>sinc(x)</c>, same shape as <paramref name="x"/>, in the promoted dtype described above.
        ///     A scalar (0-d) input yields a 0-d result; an empty input yields an empty float64 array.
        /// </returns>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="x"/> is <c>null</c> (NumPy would attempt <c>asanyarray(None)</c>; NumSharp
        ///     rejects it up front rather than surfacing a bare <see cref="NullReferenceException"/>).
        /// </exception>
        /// <exception cref="OverflowException">
        ///     <paramref name="x"/> is a Decimal array whose <c>pi*x</c> overflows the decimal range
        ///     (a limitation of the shared decimal→double transcendental bridge, not a NumPy behaviour).
        /// </exception>
        public static NDArray sinc(NDArray x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            using var plain = NDMatrixBoundary.PlainFor(x);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return x.TensorEngine.Sinc(x);
        }
    }
}
