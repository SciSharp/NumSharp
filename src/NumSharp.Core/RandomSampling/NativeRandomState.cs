using System;

namespace NumSharp
{
    /// <summary>
    ///     Represents the stored state of <see cref="MT19937"/> random number generator.
    ///     This format is compatible with NumPy's random state.
    /// </summary>
    /// <remarks>
    ///     NumPy state format:
    ///     ('MT19937', array([...624...]), pos, has_gauss, cached_gaussian)
    /// </remarks>
    public struct NativeRandomState
    {
        /// <summary>
        ///     Algorithm identifier. Always "MT19937" for NumPy-compatible state.
        /// </summary>
        public string Algorithm;

        /// <summary>
        ///     The MT19937 state array (624 uint32 values).
        /// </summary>
        public uint[] Key;

        /// <summary>
        ///     Current position in the state array (0-624).
        /// </summary>
        public int Pos;

        /// <summary>
        ///     Whether there is a cached Gaussian value (0 or 1).
        /// </summary>
        public int HasGauss;

        /// <summary>
        ///     The cached Gaussian value from Box-Muller transform.
        /// </summary>
        public double CachedGaussian;

        /// <summary>
        ///     Creates a new NativeRandomState with default values.
        /// </summary>
        public NativeRandomState(uint[] key, int pos, int hasGauss = 0, double cachedGaussian = 0.0)
        {
            Algorithm = "MT19937";
            Key = key;
            Pos = pos;
            HasGauss = hasGauss;
            CachedGaussian = cachedGaussian;
        }

        /// <summary>
        ///     Backward compatibility constructor for legacy byte[] state.
        ///     This will throw if called with old-format state.
        /// </summary>
        [Obsolete("Legacy Randomizer state format is no longer supported. Use new MT19937-based state.")]
        public NativeRandomState(byte[] state)
        {
            throw new NotSupportedException(
                "Legacy Randomizer state format is no longer supported. " +
                "NumSharp now uses MT19937 for NumPy compatibility. " +
                "Please re-seed your generator with np.random.seed().");
        }
    }
}
