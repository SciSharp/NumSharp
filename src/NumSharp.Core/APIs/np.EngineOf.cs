using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The engine a facade hands its call to when NumPy's statements must see a null operand: the operand's own
        ///     engine, or the default engine for a null operand.
        /// </summary>
        /// <param name="a">The operand whose engine computes the call, or null.</param>
        /// <returns><paramref name="a"/>'s engine, or <see cref="BackendFactory.GetEngine"/>'s default engine when
        ///     <paramref name="a"/> is null.</returns>
        /// <remarks>
        ///     A facade whose engine member reports a null operand itself (as <see cref="ArgumentNullException"/>, where
        ///     NumPy's order first reads it: np.quantile checks q and the method before the data, np.gradient refuses a null
        ///     <c>f</c> in its body) must not fail at the dispatch with a <see cref="System.NullReferenceException"/>.
        ///     Such facades dispatch through this method; a facade that checks the operand for null itself calls
        ///     <c>a.TensorEngine</c> directly. <c>np.evaluate</c> resolves its engine the same way. Added by the np-layer
        ///     dtype plan's phase PE, when the bodies of NumPy's Python functions became engine members.
        /// </remarks>
        internal static TensorEngine EngineOf(NDArray a) => a?.TensorEngine ?? BackendFactory.GetEngine();
    }
}
