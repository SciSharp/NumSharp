using System.Threading.Tasks;

namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    // Declaration-site INHERITANCE of the scope attributes: [NDScoped]/[NDScopedAsync]/[NDScopedCovered]
    // on a virtual, abstract or interface member is a contract every override and implementation
    // inherits (unless it carries a scope-family attribute of its own). Two consequences are asserted
    // here, exactly as the weaver applies them:
    //   * an inheriting override is EXEMPT from NDW012 — its dropped temporaries are reclaimed by the
    //     woven scope, so an untagged `var t = a + 1.0;` in an override must stay clean, while the SAME
    //     body under an unscoped base declaration (the controls) draws NDW012;
    //   * the target gate reads the declaration's attribute — an abstract [NDScoped] declaration is NOT
    //     NDW005, and a hidden egress on a scoped declaration is reported on the declaration AND on
    //     every override that inherits it (the weaver rejects both).
    // Tags sit where each analyzer reports: NDW012 on the leaking statement, gate codes on the
    // METHOD declaration line.
    public abstract class InheritanceBase
    {
        private static NDArray _a;

        // IH-1: abstract / property / async / virtual declarations carrying the contract — none is NDW005.
        [NDScoped]
        public abstract NDArray Compute(NDArray a);

        [NDScopedAsync]
        public abstract Task<NDArray> ComputeAsync(NDArray a);

        [NDScoped]
        public abstract NDArray Value { get; }

        [NDScoped]
        public virtual NDArray Virtual(NDArray a)
        {
            var t = a + 1.0;                                // own [NDScoped] -> exempt
            return a.copy();
        }

        // IH-2: an UNSCOPED virtual — the control: its overrides inherit nothing.
        public virtual NDArray Unscoped(NDArray a) => a.copy();

        // IH-3: [NDScopedCovered] on a virtual is inherited too (the override is covered, not woven).
        [NDScopedCovered]
        public virtual NDArray CoveredBase(NDArray a)
        {
            var t = a + 1.0;                                // own [NDScopedCovered] -> exempt
            return a.copy();
        }

        // IH-4: the gate follows the declaration — a hidden egress is reported HERE and on every
        // override that inherits the attribute.
        [NDScoped]
        public virtual void RefEgress(ref NDArray a)        // [NDW002]  the contract itself has a hidden egress
        {
            a = a + 1.0;
        }

        protected static NDArray Field => _a;
    }

    public class InheritanceDerived : InheritanceBase
    {
        public override NDArray Compute(NDArray a)
        {
            var t = a + 1.0;                                // inherits [NDScoped] from the abstract declaration -> exempt
            return a.copy();
        }

        public override async Task<NDArray> ComputeAsync(NDArray a)
        {
            var t = a + 1.0;                                // inherits [NDScopedAsync] -> exempt
            await Task.Yield();
            return a.copy();
        }

        public override NDArray Value
        {
            get
            {
                var t = Field + 1.0;                        // inherits the base PROPERTY's [NDScoped] (getter rule) -> exempt
                return Field.copy();
            }
        }

        public override NDArray Virtual(NDArray a)
        {
            var t = a * 2.0;                                // inherits [NDScoped] from the virtual -> exempt
            return a.copy();
        }

        public override NDArray Unscoped(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  the control: the base is unscoped, nothing inherited
            return a.copy();
        }

        public override NDArray CoveredBase(NDArray a)
        {
            var t = a + 1.0;                                // inherits [NDScopedCovered] -> exempt
            return a.copy();
        }

        public override void RefEgress(ref NDArray a)       // [NDW002]  inherited [NDScoped]: the same rejection, at the override
        {
            a = a * 2.0;
        }
    }

    // IH-5: explicit attributes on the override WIN — [NDScopedCovered] opts out of the inherited weave
    // (still exempt from NDW012, on the covered author's assertion); a re-stated [NDScoped] is just its own.
    public class InheritanceOptOut : InheritanceBase
    {
        [NDScopedCovered]
        public override NDArray Compute(NDArray a)
        {
            var t = a + 1.0;                                // own [NDScopedCovered] wins over the inherited [NDScoped] -> exempt
            return a.copy();
        }

        [NDScopedAsync]
        public override async Task<NDArray> ComputeAsync(NDArray a)
        {
            var t = a + 1.0;                                // re-stated [NDScopedAsync] -> exempt
            await Task.Yield();
            return a.copy();
        }

        public override NDArray Value => Field.copy();
    }

    // IH-6: the contract crosses a level that does NOT declare the slot, and a generic instantiation:
    // GenericBase<T>.Map(T, NDArray) is matched by LeafGeneric.Map(int, NDArray) two levels down.
    public abstract class GenericBase<T>
    {
        [NDScoped]
        public virtual NDArray Map(T x, NDArray a)
        {
            var t = a + 1.0;                                // own [NDScoped] -> exempt
            return a.copy();
        }

        public virtual NDArray Plain(T x, NDArray a) => a.copy();
    }

    public class MidGeneric : GenericBase<int>
    {
    }

    public class LeafGeneric : MidGeneric
    {
        public override NDArray Map(int x, NDArray a)
        {
            var t = a + x;                                  // inherits [NDScoped] through MidGeneric from GenericBase<int> -> exempt
            return a.copy();
        }

        public override NDArray Plain(int x, NDArray a)
        {
            var t = a + x;                                  // [NDW012]  the control: GenericBase<T>.Plain is unscoped
            return a.copy();
        }
    }

    // IH-7: interfaces — a scoped interface member reaches its implicit implementation, its explicit
    // implementation, and an implementation of a DERIVED interface (the base interface's member).
    public interface IScopedOp
    {
        [NDScoped]
        NDArray Apply(NDArray a);

        [NDScoped]
        NDArray Current { get; }

        NDArray Plain(NDArray a);
    }

    public interface IScopedOp2 : IScopedOp
    {
    }

    public class ImplicitImpl : IScopedOp2
    {
        private static NDArray _a;

        public NDArray Apply(NDArray a)
        {
            var t = a + 1.0;                                // implicitly implements IScopedOp.Apply (via IScopedOp2) -> exempt
            return a.copy();
        }

        public NDArray Current
        {
            get
            {
                var t = _a + 1.0;                           // implements the interface PROPERTY's [NDScoped] -> exempt
                return _a.copy();
            }
        }

        public NDArray Plain(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  the control: IScopedOp.Plain is unscoped
            return a.copy();
        }
    }

    public class ExplicitImpl : IScopedOp
    {
        private static NDArray _a;

        NDArray IScopedOp.Apply(NDArray a)
        {
            var t = a + 1.0;                                // explicitly implements the scoped member -> exempt
            return a.copy();
        }

        NDArray IScopedOp.Current
        {
            get
            {
                var t = _a + 1.0;                           // explicitly implements the scoped property -> exempt
                return _a.copy();
            }
        }

        NDArray IScopedOp.Plain(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  the control: the unscoped member, explicitly
            return a.copy();
        }
    }

    // IH-8: same NAME, no relation — a method that neither overrides nor implements inherits nothing.
    public class UnrelatedSameName
    {
        public NDArray Apply(NDArray a)
        {
            var t = a + 1.0;                                // [NDW012]  not an implementation of IScopedOp.Apply
            return a.copy();
        }
    }
}
