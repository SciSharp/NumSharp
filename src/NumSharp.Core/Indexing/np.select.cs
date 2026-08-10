using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return an array drawn from elements in <paramref name="choicelist"/>, depending on
        ///     <paramref name="condlist"/>. The output at position <c>m</c> is the <c>m</c>-th element
        ///     of the array in <paramref name="choicelist"/> where the <c>m</c>-th element of the
        ///     corresponding array in <paramref name="condlist"/> is <c>true</c>. When multiple
        ///     conditions are satisfied, the FIRST one encountered in <paramref name="condlist"/> wins;
        ///     positions where every condition is <c>false</c> take <paramref name="default"/>.
        /// </summary>
        /// <param name="condlist">
        ///     The conditions that determine which array in <paramref name="choicelist"/> each output
        ///     element is taken from. Must be boolean arrays and the same length as
        ///     <paramref name="choicelist"/>. All conditions are broadcast against each other.
        /// </param>
        /// <param name="choicelist">
        ///     The arrays the output elements are drawn from. Each entry is either an
        ///     <see cref="NDArray"/> (strong dtype) or a boxed C# scalar. As in NumPy (NEP50), an
        ///     <c>int</c>/<c>float</c>/<c>double</c>/<see cref="Complex"/> literal is a <em>weak</em>
        ///     scalar that adopts the other operands' dtype, while <c>bool</c>, <c>char</c>,
        ///     <see cref="Half"/>, <c>decimal</c>, arrays and every <see cref="NDArray"/> are strong.
        ///     An <see cref="NDArray"/>[] binds here directly via array covariance; scalar choices need
        ///     an explicit <c>new object[] { … }</c>. All choices AND <paramref name="default"/> are
        ///     broadcast against each other.
        /// </param>
        /// <param name="default">
        ///     The value inserted where all conditions are <c>false</c>. <c>null</c> (the C# default)
        ///     is NumPy's <c>default=0</c> — a weak python int. Accepts a scalar or an
        ///     <see cref="NDArray"/> (which participates in the choice broadcast shape).
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array whose dtype is <see cref="result_type"/> of every choice and
        ///     the default (NEP50), and whose shape is the broadcast of the conditions against the
        ///     choices.
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="condlist"/> and <paramref name="choicelist"/> differ in length, or
        ///     <paramref name="condlist"/> is empty.
        /// </exception>
        /// <exception cref="TypeError">A condition is not a boolean array.</exception>
        /// <exception cref="IncorrectShapeException">
        ///     The conditions, the choices, or the two groups against each other cannot be broadcast
        ///     (NumSharp's house form of NumPy's broadcast <c>ValueError</c>).
        /// </exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.select</c> (<c>numpy/lib/_function_base_impl.py</c>): fill the
        ///     result with the default, then <see cref="copyto"/> each choice onto it under its
        ///     condition mask in REVERSE order so the first matching condition takes precedence — the
        ///     same composition NumPy uses, so every masked write rides NumSharp's SIMD masked-cast
        ///     kernel. https://numpy.org/doc/stable/reference/generated/numpy.select.html
        /// </remarks>
        public static NDArray select(NDArray[] condlist, object[] choicelist, object @default = null)
        {
            if (condlist is null) throw new ArgumentNullException(nameof(condlist));
            if (choicelist is null) throw new ArgumentNullException(nameof(choicelist));

            // NumPy's order: the length check precedes the empty check, so select([], [x])
            // is a length error while select([], []) is the empty-condition error.
            if (condlist.Length != choicelist.Length)
                throw new ValueError("list of cases must be same length as list of conditions");
            if (condlist.Length == 0)
                throw new ValueError("select with an empty condition list is not possible");

            int n = condlist.Length;

            // Classify each choice plus the default, resolving the NEP50 result dtype while
            // materialising every entry to an NDArray for the broadcast/copy below. The default
            // occupies the final slot [n] — NumPy appends it to choicelist before promoting so it
            // both fixes the dtype and joins the choice broadcast shape.
            var mats = new NDArray[n + 1];
            bool anyStrong = false;
            bool anyBeyondInt64 = false;
            int weakRank = 0; // 0 none, 1 int, 2 float, 3 complex (ordered by NEP50 dominance)
            NPTypeCode strong = NPTypeCode.Empty;

            for (int i = 0; i <= n; i++)
            {
                object item = i < n ? choicelist[i] : @default;
                NDArray mat;
                switch (item)
                {
                    case null:
                        // A null CHOICE is a caller bug (NumPy would build an object array, a dtype
                        // NumSharp lacks). A null DEFAULT is NumPy's default=0 — a weak python int.
                        if (i < n)
                            throw new ArgumentNullException($"choicelist[{i}]",
                                "choicelist entries must not be null.");
                        mat = NDArray.Scalar(0);
                        weakRank = Math.Max(weakRank, 1);
                        break;

                    case NDArray a:
                        mat = a;
                        strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, a.GetTypeCode) : a.GetTypeCode;
                        anyStrong = true;
                        break;

                    // Weak (NEP50 "python literal") scalars — the eight C# integer primitives, the two
                    // binary floats and Complex. These adopt the other operands' dtype instead of
                    // forcing promotion by their own width/value (int8 array + 1000 stays int8).
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 1);
                        if (item is ulong u && u > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case float or double:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 2);
                        break;

                    case Complex:
                        mat = asanyarray(item);
                        weakRank = Math.Max(weakRank, 3);
                        break;

                    default:
                        // Strong by NumPy's rule `type(x) in (int, float, complex) else asarray(x)`:
                        // bool, char, Half, decimal, C# arrays and collections all take the asarray
                        // branch and contribute their concrete dtype.
                        mat = asanyarray(item);
                        strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, mat.GetTypeCode) : mat.GetTypeCode;
                        anyStrong = true;
                        break;
                }

                mats[i] = mat;
            }

            NPTypeCode dtype = ResolveSelectDtype(anyStrong, strong, weakRank, anyBeyondInt64);

            // The conditions and the choices are broadcast in two SEPARATE groups (NumPy makes two
            // independent broadcast_arrays calls). We only need each GROUP'S shape here — to size the
            // result and to validate mutual compatibility exactly where NumPy's two calls do — because
            // copyto broadcasts each operand to the result shape itself below. Materialising the
            // broadcast VIEWS is therefore skipped: it would allocate needlessly AND, for a group of
            // one, route through broadcast_arrays's single-array path (Shape.Clean(), which resets a
            // non-contiguous operand's strides to contiguous and would read a transposed/strided/
            // reversed condition through the wrong elements).
            Shape scShape = CommonBroadcastShape(condlist);   // validates conditions mutually
            Shape schShape = CommonBroadcastShape(mats);       // validates choices + default mutually

            // Conditions must be boolean — checked after the group broadcasts, reporting the index.
            for (int i = 0; i < n; i++)
                if (condlist[i].GetTypeCode != NPTypeCode.Boolean)
                    throw new TypeError($"invalid entry {i} in condlist: should be boolean ndarray");

            // Result shape = broadcast(condition-shape, choice-shape). NumPy special-cases all-scalar
            // choices to skip this call, but broadcast(S, ()) == S so the general call is identical.
            var (commonLeft, _) = Shape.Broadcast(scShape, schShape);
            var result = new NDArray(dtype, new Shape((long[])commonLeft.dimensions.Clone()), false);

            // Burn the default across the whole result (unsafe cast, matching np.full's copyto), then
            // overlay each choice where its condition is true — LAST choice first, so the FIRST
            // matching condition ends up winning each position. copyto broadcasts each operand
            // (and mask) to the result shape, preserving its own strides. This is exactly NumPy's
            // own structure (np.full + reverse copyto); a fused single-pass kernel was prototyped
            // and MEASURED SLOWER (SIMD bool-mask expansion has no early-out, and the scalar
            // reverse-overwrite evaluates every condition), so the composition — at parity with
            // NumPy at scale, overhead-bound for tiny arrays — is what ships.
            copyto(result, mats[n], casting: "unsafe");
            for (int i = n - 1; i >= 0; i--)
                copyto(result, mats[i], casting: "same_kind", @where: condlist[i]);

            return result;
        }

        /// <summary>
        ///     The common broadcast shape of a group of operands (NumPy's <c>broadcast_arrays</c>
        ///     shape). Only the DIMENSIONS are consumed by the caller, so the single-operand path's
        ///     stride reset is harmless; every returned element carries the common dimensions.
        /// </summary>
        private static Shape CommonBroadcastShape(NDArray[] arrays)
        {
            var shapes = new Shape[arrays.Length];
            for (int i = 0; i < arrays.Length; i++)
                shapes[i] = arrays[i].Shape;
            return Shape.Broadcast(shapes)[0];
        }

        /// <summary>
        ///     NumPy's <c>result_type(*choices, default)</c> folded with NEP50 weak-scalar rules —
        ///     the same resolution <see cref="AxisConcatenator"/> uses, minus the weak-bool kind
        ///     (select treats python bool as a strong asarray) and the weak-int range check (select
        ///     wraps an out-of-range default via <see cref="copyto"/> rather than raising).
        /// </summary>
        private static NPTypeCode ResolveSelectDtype(bool anyStrong, NPTypeCode strong, int weakRank, bool anyBeyondInt64)
        {
            if (!anyStrong)
            {
                // All weak literals — the NEP50 defaults. A weak int past int64 lifts the default to
                // uint64 (NumPy: result_type(2**63) and result_type(2**64-1) are both uint64).
                return weakRank switch
                {
                    3 => NPTypeCode.Complex,
                    2 => NPTypeCode.Double,
                    _ => anyBeyondInt64 ? NPTypeCode.UInt64 : NPTypeCode.Int64,
                };
            }

            return weakRank switch
            {
                // No literal.
                0 => strong,
                // A weak int adopts the strong dtype, except bool which it lifts to the default int.
                1 => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                // A weak float keeps a float/decimal/complex width, else forces the default float.
                2 => NDExprTypeRules.IsFloatKind(strong) || strong == NPTypeCode.Decimal || strong == NPTypeCode.Complex
                    ? strong
                    : NPTypeCode.Double,
                // NumSharp has a single complex width, so complex64 collapses onto Complex.
                _ => NPTypeCode.Complex,
            };
        }
    }
}
