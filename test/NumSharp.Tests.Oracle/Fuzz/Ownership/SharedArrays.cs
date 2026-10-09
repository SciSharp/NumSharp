using System;
using System.Collections.Generic;
using System.Reflection;

namespace NumSharp.Tests.Fuzz.Ownership
{
    /// <summary>
    ///     The process-wide arrays NumSharp hands out but no caller owns — <c>np.ma.nomask</c>, the masked constant's data
    ///     and mask, and the 24 <c>numpy.polynomial</c> module constants (<c>polydomain</c>, <c>chebzero</c>, …) — watched by
    ///     one ledger that every ownership check consults.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     These arrays are the purest "data we do not own": every caller of every op may be holding them, ops hand them back
    ///     AS results (<c>np.ma.getmask</c> of an unmasked array is <c>nomask</c> itself), and NumPy has them as ordinary module
    ///     attributes. An op that disposes or releases one damages the whole process: <c>nomask</c>'s buffer would go back to
    ///     the pool and the next allocation of its size would silently rewrite it.
    ///     </para>
    ///     <para>
    ///     Their reference counts legitimately rise while an op holds a view of one, so only the over-disposal kinds count
    ///     here (<see cref="CheckOverDisposal"/>). The snapshot is taken once and re-taken only after a finding
    ///     (<see cref="Rebase"/>): a damaged array stays damaged, and without the re-take every later case would be blamed
    ///     for it too.
    ///     </para>
    /// </remarks>
    internal static class SharedArrays
    {
        /// <summary>The current snapshot of every shared array (built on first use).</summary>
        private static OwnershipLedger _ledger;

        /// <summary>Guards the snapshot (the sweeps run single-threaded; the lock only makes a stray parallel test safe).</summary>
        private static readonly object Gate = new();

        /// <summary>
        ///     The numpy.polynomial modules and their constant prefixes: <c>{prefix}domain</c>, <c>{prefix}zero</c>,
        ///     <c>{prefix}one</c> and <c>{prefix}x</c> on each.
        /// </summary>
        private static readonly (Func<object> module, string name, string prefix)[] PolynomialModules =
        {
            (() => np.polynomial.polynomial, "polynomial", "poly"),
            (() => np.polynomial.chebyshev, "chebyshev", "cheb"),
            (() => np.polynomial.legendre, "legendre", "leg"),
            (() => np.polynomial.laguerre, "laguerre", "lag"),
            (() => np.polynomial.hermite, "hermite", "herm"),
            (() => np.polynomial.hermite_e, "hermite_e", "herme"),
        };

        /// <summary>
        ///     Every shared array with its name. Reading a polynomial constant creates it on first access (as NumPy's import
        ///     does), so the list is complete from the first call on.
        /// </summary>
        /// <returns>The arrays and their names.</returns>
        /// <exception cref="InvalidOperationException">A polynomial module no longer has a constant the list names (a
        ///     renamed member: update this list rather than watch fewer arrays).</exception>
        public static IEnumerable<(NDArray Array, string Role)> All()
        {
            yield return (np.ma.nomask, "shared np.ma.nomask");
            yield return (np.ma.NDMasked._data, "shared np.ma.masked._data");
            if (np.ma.NDMasked._mask is not null)
                yield return (np.ma.NDMasked._mask, "shared np.ma.masked._mask");

            foreach (var (module, name, prefix) in PolynomialModules)
            {
                object m = module();
                foreach (var suffix in new[] { "domain", "zero", "one", "x" })
                {
                    var p = m.GetType().GetProperty(prefix + suffix, BindingFlags.Instance | BindingFlags.Public)
                            ?? throw new InvalidOperationException($"np.polynomial.{name} has no {prefix}{suffix}: update SharedArrays");
                    yield return ((NDArray)p.GetValue(m), $"shared np.polynomial.{name}.{prefix}{suffix}");
                }
            }
        }

        /// <summary>The shared arrays by reference, for <see cref="Contains"/> (built on first use).</summary>
        private static HashSet<object> _set;

        /// <summary>
        ///     True when <paramref name="o"/> is one of the shared arrays — an object a result disposer must never dispose,
        ///     because no caller owns it (an op may hand one back AS its result: <c>np.polynomial.polynomial.polydomain</c>
        ///     returns the module constant itself).
        /// </summary>
        /// <param name="o">A result part.</param>
        /// <returns>Whether it is shared.</returns>
        /// <exception cref="InvalidOperationException">A polynomial constant cannot be resolved (see <see cref="All"/>).</exception>
        public static bool Contains(object o)
        {
            if (o is not NDArray)
                return false;
            HashSet<object> set;
            lock (Gate)
            {
                if (_set is null)
                {
                    _set = new HashSet<object>(ReferenceEqualityComparer.Instance);
                    foreach (var (a, _) in All())
                        _set.Add(a);
                }
                set = _set;
            }
            return set.Contains(o);
        }

        /// <summary>
        ///     The first over-disposal any shared array suffered since the snapshot, or null when all are intact.
        /// </summary>
        /// <returns>The finding, or null.</returns>
        /// <exception cref="InvalidOperationException">The ARC probe cannot read a slice.</exception>
        public static OwnershipFinding CheckOverDisposal()
        {
            OwnershipLedger ledger;
            lock (Gate)
                ledger = _ledger ??= OwnershipLedger.Of(All());

            foreach (var f in ledger.Check())
                if (f.IsOverDisposal)
                    return f;
            return null;
        }

        /// <summary>
        ///     Re-takes the snapshot (after a finding was recorded, so the damage is not blamed on every later case).
        /// </summary>
        /// <exception cref="InvalidOperationException">The ARC probe cannot read a slice.</exception>
        public static void Rebase()
        {
            lock (Gate)
                _ledger = OwnershipLedger.Of(All());
        }
    }
}
