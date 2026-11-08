using System;
using System.Collections.Generic;
using System.Reflection;

namespace NumSharp.Tests.Fuzz.Ownership
{
    /// <summary>
    ///     The process-wide arrays NumSharp hands out but no caller owns — <c>np.ma.nomask</c>, the masked constant's data
    ///     and mask, the 24 <c>numpy.polynomial</c> module constants (<c>polydomain</c>, <c>chebzero</c>, …) and the 12
    ///     class attributes of its six series classes (<c>NDChebyshev.Class.domain</c>, …, which <c>p.domain</c> of a series
    ///     built without one returns itself) — watched by one ledger that every ownership check consults.
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
        ///     The six series classes' class objects (<c>NDChebyshev.Class</c>, …), whose <c>domain</c> / <c>window</c> class
        ///     attributes are shared arrays. Named in code, not by reflection: a renamed class object fails the build here
        ///     instead of shrinking the watched list.
        /// </summary>
        /// <returns>Each class object with its CLR name, for the role text.</returns>
        private static IEnumerable<(NDPolyClass Class, string Name)> PolynomialClasses()
        {
            yield return (NDPolynomial.Class, nameof(NDPolynomial));
            yield return (NDChebyshev.Class, nameof(NDChebyshev));
            yield return (NDLegendre.Class, nameof(NDLegendre));
            yield return (NDLaguerre.Class, nameof(NDLaguerre));
            yield return (NDHermite.Class, nameof(NDHermite));
            yield return (NDHermiteE.Class, nameof(NDHermiteE));
        }

        /// <summary>
        ///     Every shared array with its name. Reading a polynomial constant or class attribute creates it on first access
        ///     (as NumPy's import does), so the list is complete from the first call on. A class attribute assigned later
        ///     (<c>NDChebyshev.Class.domain = ...</c>) is a new array the list does not watch: no oracle case assigns one.
        /// </summary>
        /// <returns>The arrays and their names.</returns>
        /// <exception cref="InvalidOperationException">A polynomial module no longer has a constant the list names (a renamed
        ///     member: update this list rather than watch fewer arrays). The class attributes are named in code
        ///     (<see cref="PolynomialClasses"/>), so a renamed one fails the build instead.</exception>
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

            // The class attributes as they are now: an attribute assigned since the snapshot is a different array, which a
            // Rebase() re-takes (see the summary).
            foreach (var (cls, name) in PolynomialClasses())
            {
                yield return (cls.domain, $"shared {name}.Class.domain");
                yield return (cls.window, $"shared {name}.Class.window");
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
