namespace NumSharp
{
    /// <summary>
    ///     A class that serves as numpy.random.RandomState in python.
    /// </summary>
    /// <remarks>https://docs.scipy.org/doc/numpy-1.16.1/reference/routines.random.html</remarks>
    public partial class NumPyRandom
    {
        protected internal Randomizer randomizer;

        public int Seed { get; set; }

        #region Constructors

        protected internal NumPyRandom(Randomizer randomizer)
        {
            this.randomizer = randomizer;
        }

        protected internal NumPyRandom(NativeRandomState nativeRandomState)
        {
            set_state(nativeRandomState);
        }

        protected internal NumPyRandom(int seed) : this(new Randomizer(seed)) { }

        protected internal NumPyRandom() : this(new Randomizer()) { }

        #endregion

        #region RandomState

        /// <summary>
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState()
        {
            return new NumPyRandom();
        }

        /// <summary>
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState(int seed)
        {
            return new NumPyRandom(seed);
        }

        /// <summary>
        ///     Returns a new instance of <see cref="NumPyRandom"/>.
        /// </summary>
        public NumPyRandom RandomState(NativeRandomState state)
        {
            return new NumPyRandom(state);
        }

        #endregion

        /// <summary>
        ///     Seeds the generator.
        ///     It can be called again to re-seed the generator.
        /// </summary>
        public void seed(int seed)
        {
            randomizer = new Randomizer(seed);
        }

        /// <summary>
        ///     Set the internal state of the generator from a <see cref="NumPyRandom"/>.
        ///     for use if one has reason to manually (re-)set the internal state of the pseudo-random number generating algorithm.
        /// </summary>
        /// <param name="nativeRandomState">The state to restore onto this <see cref="NumPyRandom"/></param>
        public void set_state(NativeRandomState nativeRandomState)
        {
            randomizer = nativeRandomState.Restore();
        }

        /// <summary>
        ///     Return a <see cref="NumPyRandom"/> representing the internal state of the generator.
        /// </summary>
        /// <returns></returns>
        public NativeRandomState get_state()
        {
            return randomizer.Save();
        }
    }
}
