using System;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The stack frames the test-time subclass caller audit (<c>NDSubclass.Audit</c>, switched on by the oracle's chararray
    ///     and matrix operand replays) depends on. The audit walks the stack up from a woven subclass path and stops at an
    ///     override of a <c>[SubclassAware]</c> member: NumPy's own class-reading code, which hands its subclass object on by
    ///     design (np.nanmax of a chararray is <c>np.amax</c> of the chararray). Optimized code can take such a frame off the
    ///     stack: an inlined method has no frame, and a call in tail position can become a jump that replaces its caller's.
    ///     The walk then reaches the public function above it and records it as a library caller - a red gate whose outcome
    ///     depends on when tiered compilation optimizes the member (PR #633's CI, 2026-11-07: <c>NDCharArray.NanMax</c>
    ///     tail-jumped to <c>np.&lt;amax&gt;subclass</c> on ubuntu net10.0, and the audit reported <c>np.nanmax -&gt; amax</c>).
    /// </summary>
    [TestClass]
    public class SubclassAuditFrameTests
    {
        /// <summary>
        ///     Every override of a <c>[SubclassAware]</c> virtual member in NumSharp.Core keeps its frame:
        ///     <see cref="MethodImplAttributes.NoInlining"/> (no caller absorbs it, guarded devirtualization included) and
        ///     <see cref="MethodImplAttributes.NoOptimization"/> (it inlines nothing and makes no tail call).
        /// </summary>
        /// <exception cref="AssertFailedException">An override lacks either flag, or fewer overrides than the known four were
        ///     found (the reflection walk no longer sees the attribute, which would make the check vacuous).</exception>
        /// <remarks>Deterministic, unlike the audit itself: it reads the flags from metadata, so it fails on every host and in
        ///     every tiering state, where the audit's failure needed a tier-1 compile to land mid-replay.</remarks>
        [TestMethod]
        public void SubclassAwareOverrides_KeepTheirFrame()
        {
            // System.Reflection.Assembly spelled out: NumSharp.Tests.Assembly is a namespace of this project.
            System.Reflection.Assembly core = typeof(NDArray).Assembly;
            // Internal to NumSharp.Core and inert at run time but for the audit: resolved by name, as the audit's reflection
            // reads it.
            Type aware = core.GetType("NumSharp.SubclassAwareAttribute", throwOnError: true);
            const BindingFlags declared = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
            const MethodImplAttributes keepFrame = MethodImplAttributes.NoInlining | MethodImplAttributes.NoOptimization;

            // An override is a method whose base definition is declared by another type; the attribute sits on that base
            // declaration (NDArray.NanMax), never on the overrides, exactly as NDSubclass.IsSubclassAware resolves it.
            var overrides = core.GetTypes()
                .SelectMany(t => t.GetMethods(declared))
                .Where(m => m.GetBaseDefinition().DeclaringType != m.DeclaringType
                            && m.GetBaseDefinition().IsDefined(aware, false))
                .ToList();

            overrides.Count.Should().BeGreaterThanOrEqualTo(4,
                "NDCharArray and NDMatrix both override NanMax and NanMin; fewer means the walk lost the attribute");
            var missing = overrides
                .Where(m => (m.MethodImplementationFlags & keepFrame) != keepFrame)
                .Select(m => $"{m.DeclaringType!.FullName}.{m.Name} ({m.MethodImplementationFlags})")
                .ToList();
            missing.Should().BeEmpty(
                "the subclass caller audit stops at these frames, so each needs " +
                "[MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.NoOptimization)]");
        }
    }
}
