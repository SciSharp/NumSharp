using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Angle(NDArray, bool)"/>, NumPy's <c>np.angle</c> body:
        ///     <c>arctan2(z.imag, z.real)</c> for a complex input and <c>arctan2(0, z)</c> (a weak Python int 0) for a real
        ///     one, scaled by <c>180 / pi</c> in place when <paramref name="deg"/> is set. The facade applies the
        ///     <c>numpy.matrix</c> rule first and calls this for everything else.
        /// </summary>
        /// <param name="z">The input (any dtype; text is refused as NumPy's <c>arctan2</c> refuses it).</param>
        /// <param name="deg">Return degrees instead of radians.</param>
        /// <returns>The phase angle: float64 for complex128, else NumPy's tier of <c>arctan2(0, z)</c>
        ///     (<see cref="AngleRealTier"/>).</returns>
        /// <exception cref="TypeError">Text: <c>arctan2</c> has no string loop (the <c>StringDType</c> no-loop text naming
        ///     the weak <c>_PyLongDType</c>, the legacy resolver's text for <c>S</c> / <c>U</c>).</exception>
        /// <remarks>
        ///     <para>Moved from <c>np.angle</c> by the np-layer dtype plan's P0 pilot, statement for statement; the facade's
        ///     <c>np.arctan2</c> / <c>np.multiply</c> calls became the engine members they forward to (the operands here are
        ///     never matrices or chararrays: the facade decides the class first, and the lanes, casts and scalars below are
        ///     plain arrays). The plan's phase PE (E1) made it this override of a <c>TensorEngine</c> member; it was an
        ///     internal static of DefaultEngine.</para>
        ///     <para>The real path computes <c>arctan2</c> in float64 and casts to the tier: its only values are +0, pi and
        ///     NaN, so the cast is exact and matches NumPy computing at the tier (verified against NumPy 2.4.2).</para>
        /// </remarks>
        [NDScoped] // the lanes, the float64 cast, the 0-d zero and the float64 angle before the tier cast are this call's own
        public override NDArray Angle(NDArray z, bool deg = false)
        {
            // NumPy's angle of a non-complex input is arctan2(0, z), which has no string loop: the Python int 0 is the
            // weak _PyLongDType in the StringDType refusal, the fixed-width classes take the legacy resolver's text.
            if (Strings.NDStringUfuncs.IsVString(z))
                throw Strings.NDStringUfuncs.VStringNoLoop("arctan2", new[] { DTypeRegistry.PyLong.ToString(), z.dtype.Meta.ToString() });
            if (Strings.NDStringUfuncs.IsString(z))
                throw Strings.NDStringUfuncs.NotSupportedForInputTypes("arctan2");
            NDArray a;
            if (z.typecode == NPTypeCode.Complex)
            {
                // NumPy: a = arctan2(z.imag, z.real). Both lanes are float64 VIEWS onto the Complex
                // storage (the FFT driver's strided complex->real read), so the product is float64 and
                // bit-identical to NumPy's arctan2 (verified against NumPy 2.4.2).
                var zimag = new NDArray(z.Storage.AliasComplexLane(imaginary: true));
                var zreal = new NDArray(z.Storage.AliasComplexLane(imaginary: false));
                a = zreal.TensorEngine.ATan2(zimag, zreal);
            }
            else
            {
                // NumPy: a = arctan2(0, z) with a WEAK Python int 0. Its only possible values are
                // {+0, pi, nan} (the sign of z decides), so computing arctan2 in float64 then casting to
                // NumPy's angle tier is bit-exact with NumPy computing directly at that tier (verified);
                // this also reuses the existing arctan2 kernel per NumPy's own composition.
                var x = z.astype(NPTypeCode.Double);
                a = x.TensorEngine.ATan2(NewScalar(0.0), x);
                NPTypeCode tier = AngleRealTier(z.typecode);
                if (tier != NPTypeCode.Double)
                    a = a.astype(tier);
            }

            if (deg)
            {
                // NumPy: `a *= 180 / pi`, an in-place multiply at a's own dtype by the WEAK double
                // 180/pi. Materialize that weak scalar IN the resolved angle tier and force the same
                // loop dtype. The ordinary `NDArray * double` operator happens to preserve the tier
                // for non-scalar arrays, but promotes a 0-D Half/Single result to Double — exactly the
                // shape-dependent dtype bug the differential oracle caught.
                using var scale = NewScalar(180.0 / Math.PI, a.typecode);
                a = a.TensorEngine.Multiply(a, scale, a.typecode);
            }

            return a;
        }

        /// <summary>
        ///     NumPy's per-dtype float tier for the REAL-input angle path, i.e. the dtype of
        ///     <c>arctan2(0, z)</c> where <c>0</c> is a weak Python int (probed against NumPy 2.4.2).
        ///     Note <c>bool</c> resolves to float64 (unlike the ordinary unary-math tier's float16).
        ///     Char (uint16-like) follows uint16 -&gt; float32; Decimal (no NumPy analog) -&gt; float64.
        ///     The fused expression engine's <c>UnaryNode</c> (np.angle's NDExpr node) resolves its real-input result
        ///     dtype through this same table, not a copy.
        /// </summary>
        /// <param name="tc">The input's type code (any real dtype).</param>
        /// <returns>float16 for int8 / uint8 / float16, float32 for int16 / uint16 / char / float32, float64 otherwise.</returns>
        internal static NPTypeCode AngleRealTier(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Half:
                    return NPTypeCode.Half;
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Single:
                    return NPTypeCode.Single;
                default:
                    // Boolean, Int32/UInt32/Int64/UInt64, Double, Decimal.
                    return NPTypeCode.Double;
            }
        }
    }
}
