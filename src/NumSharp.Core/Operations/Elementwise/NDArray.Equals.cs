using System;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Determines if NDArray data is same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            unsafe
            {
                if (obj is null)
                    return false;

                if (ReferenceEquals(this, obj))
                    return true;

                // Using this comparison allows less restrictive semantics,
                // like comparing a scalar to an array
                // we can use unmanaged access because the result of == op is never a slice.
                var results = (this == obj);
                var len = results.size;
                var addr = results.Address;

                for (long i = 0; i < len; i++)
                    if (!addr[i])
                        return false;

                return true;
            }
        }

        /// <summary>
        /// Element-wise equal comparison (==).
        /// Supports all 12 dtypes and broadcasting.
        /// </summary>
        public static NDArray<bool> operator ==(NDArray lhs, NDArray rhs)
        {
            if (lhs is null && rhs is null)
                return Scalar<bool>(true).MakeGeneric<bool>();

            if (lhs is null || rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // NumPy: empty array comparison returns empty array, not scalar
            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            return lhs.TensorEngine.Compare(lhs, rhs).AsGeneric<bool>();
        }

        /// <summary>
        /// Element-wise equal comparison with scalar (==).
        /// </summary>
        // Scope: np.asanyarray(rhs) mints a temp for a scalar/array-like operand, and the
        // null/empty early returns strand an untyped Scalar/empty wrapper under MakeGeneric —
        // [NDScoped] reclaims both while yielding the typed result (an NDArray-input passthrough
        // is never tracked, rule R2). The NDArray×NDArray operator above owns no temp (the engine
        // returns a bool-typed result AsGeneric passes through) and stays unscoped on that hot path.
        [NDScoped]
        public static NDArray<bool> operator ==(NDArray lhs, object rhs)
        {
            if (rhs is null)
                return Scalar<bool>(ReferenceEquals(lhs, null)).MakeGeneric<bool>();

            if (lhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            // NumPy: empty array comparison returns empty array, not scalar
            if (lhs.Shape.IsEmpty || lhs.size == 0)
                return np.empty(lhs.Shape, NPTypeCode.Boolean).MakeGeneric<bool>();

            return lhs == np.asanyarray(rhs);
        }

        /// <summary>
        /// Element-wise equal comparison with scalar on left (==).
        /// </summary>
        [NDScoped]
        public static NDArray<bool> operator ==(object lhs, NDArray rhs)
        {
            if (lhs is null)
                return Scalar<bool>(ReferenceEquals(rhs, null)).MakeGeneric<bool>();

            if (rhs is null)
                return Scalar<bool>(false).MakeGeneric<bool>();

            return np.asanyarray(lhs) == rhs;
        }

        /// NumPy signature: numpy.equal(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'equal'>
        /// <summary>
        /// Compare two NDArrays element wise
        /// </summary>
        /// <param name="np2">NDArray to compare with</param>
        /// <returns>NDArray with result of each element compare</returns>
        private NDArray<bool> equal(NDArray np2)
        {
            return this == np2;
        }

        /// <summary>
        ///     True if this array and <paramref name="rhs"/> have the same shape and all elements are equal.
        /// </summary>
        /// <param name="rhs">The array to compare against.</param>
        /// <param name="equal_nan">
        ///     When <c>false</c> (the NumPy default) NaN never compares equal — an array holding a NaN is NOT
        ///     equal even to itself, so <c>a.array_equal(a)</c> is <c>false</c> when <c>a</c> contains a NaN.
        ///     When <c>true</c>, NaNs at matching positions are treated as equal, and for a complex dtype a
        ///     value counts as NaN when EITHER component is NaN.
        /// </param>
        /// <returns>Returns True if the arrays are equal.</returns>
        /// <remarks>
        ///     Faithful port of NumPy 2.4.2's <c>array_equal</c>. Shape must match EXACTLY (no broadcasting;
        ///     see <see cref="np.array_equiv(NDArray,NDArray)"/> for the broadcasting variant), and the shape
        ///     check runs FIRST. The default path is <c>all(this == rhs)</c> — deliberately WITHOUT a
        ///     same-reference short-circuit, because NaN ≠ NaN means an array with a NaN is not equal to itself
        ///     here; a same-reference short-circuit is only valid under <paramref name="equal_nan"/>. Dtypes
        ///     that cannot hold NaN (bool / all integer widths / char / decimal) skip the NaN machinery: their
        ///     <c>isnan</c> is uniformly false, so the fast path yields the identical result NumPy would (this
        ///     also sidesteps calling <c>isnan</c> on Char/Decimal, which have no NumPy analog).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        /// </remarks>
        public bool array_equal(NDArray rhs, bool equal_nan = false)
        {
            // A null operand can never match a real array.
            if (rhs is null)
                return false;

            // NumPy checks shape equality before anything else; differing shapes are never equal
            // (array_equal does NOT broadcast — that is array_equiv's job).
            if (Shape != rhs.Shape)
                return false;

            // Default (equal_nan == false): pure element-wise equality reduced with all(). NaN != NaN,
            // so an array containing NaN is not equal even to itself — which is exactly why NO
            // reference/storage short-circuit is taken here (one would wrongly report such arrays equal).
            if (!equal_nan)
            {
                using var eq = this == rhs;
                return np.all(eq);
            }

            // equal_nan == true from here on.
            // Same object: NaN compares equal to itself, so an array always equals itself.
            if (ReferenceEquals(this, rhs))
                return true;

            // Dtypes that cannot hold NaN take the plain comparison: their isnan is all-false, so the
            // NaN-aware branch below would reduce to exactly this. Result-identical to NumPy's structure.
            if (!CanHoldNaN(this) && !CanHoldNaN(rhs))
            {
                using var eq = this == rhs;
                return np.all(eq);
            }

            // NaN-aware path: NaN must occur at the SAME positions in both operands...
            using var a1nan = np.isnan(this);
            using var a2nan = np.isnan(rhs);
            using var nanPositionsEq = a1nan == a2nan;
            if (!np.all(nanPositionsEq))
                return false;

            // ...and every finite slot must be equal, while a NaN slot passes (a1nan is true there, and
            // the positions were just proven to coincide). This `all((this == rhs) | isnan(this))` form is
            // algebraically identical to NumPy's `all(this[~a1nan] == rhs[~a1nan])` once the positions
            // agree, but needs no gather/copy and handles empty / 0-d inputs uniformly.
            using var elementsEq = this == rhs;
            using var okOrNan = elementsEq | a1nan;
            return np.all(okOrNan);
        }

        // True only for the dtypes whose values can be NaN (the float family plus complex). Every other
        // NumSharp dtype — bool, the signed/unsigned integer widths, char and decimal — cannot, so
        // array_equal's equal_nan path safely skips isnan for them.
        private static bool CanHoldNaN(NDArray a)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    return true;
                default:
                    return false;
            }
        }

    }
}
