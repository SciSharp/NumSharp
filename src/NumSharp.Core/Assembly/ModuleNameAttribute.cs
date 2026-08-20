using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a public type as the C# host of a NumPy module surface — the type whose public
    ///     members ARE that module's functions. <c>np</c> itself carries <c>"np"</c>,
    ///     <see cref="NDArray"/> carries <c>"ndarray"</c>, and each function-namespace facade
    ///     carries its dotted Python path (<c>"np.random"</c> on <see cref="NumPyRandom"/>,
    ///     <c>"np.fft"</c> on <see cref="FourierModule"/>, <c>"np.linalg"</c> on the nested
    ///     <c>np.linalg</c> class).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Consumed by <c>coverage/NumSharp.Tools.ApiInventory</c>, which discovers the surfaces to
    ///     reflect by scanning for this attribute instead of hardcoding a type list — so a NEW module
    ///     facade (a hypothetical <c>np.strings</c>, a polynomial module) becomes part of the NumPy
    ///     coverage artifact by annotation alone, with no tool or generator edit. Before this
    ///     existed the inventory hardcoded three types and silently mis-reported the whole
    ///     <c>np.fft</c> and <c>np.linalg</c> surfaces as missing.
    ///     </para>
    ///     <para>
    ///     Apply it to the type that DECLARES the module's functions, not to the property that
    ///     exposes it (<c>np.fft</c> the property is just the reachability path; the functions live
    ///     on <see cref="FourierModule"/>). Single-object DSL exports — <c>np.r_</c>, <c>np.s_</c>,
    ///     <c>np.mgrid</c> — take NO attribute: NumPy exports each as ONE object, so the property on
    ///     <c>np</c> is already the whole coverage row, and annotating their classes would wrongly
    ///     count every indexer overload as a module function.
    ///     </para>
    ///     <para>
    ///     <see cref="AttributeUsageAttribute.Inherited"/> is FALSE and load-bearing:
    ///     <c>NDArray&lt;T&gt;</c> derives from <see cref="NDArray"/> and must not be swept up as a
    ///     second <c>"ndarray"</c> host.
    ///     </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, AllowMultiple = false, Inherited = false)]
    public sealed class ModuleNameAttribute : Attribute
    {
        /// <summary>The NumPy-side module path: <c>"np"</c>, <c>"ndarray"</c>, <c>"np.random"</c>, …</summary>
        public string Name { get; }

        public ModuleNameAttribute(string name)
            => Name = name ?? throw new ArgumentNullException(nameof(name));
    }
}
