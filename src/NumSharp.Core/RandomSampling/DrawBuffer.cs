namespace NumSharp
{
    /// <summary>
    ///     A read-ahead buffer of 64-bit draws for a sampler that consumes a VARIABLE number of draws per output
    ///     (rejection samplers such as the ziggurat), refilled through <see cref="BitGenerator.FillUInt64"/> without ever
    ///     drawing a word the per-draw sampler would not have drawn.
    /// </summary>
    /// <remarks>
    ///     The no-overdraw contract: before producing output <c>i</c> of <c>n</c>, the caller sets <see cref="Owed"/> to
    ///     <c>n - i</c>. A refill then draws at most <see cref="Owed"/> words — safe because every output consumes AT LEAST
    ///     one draw, so at any refill point the remaining outputs are certain to consume that many more. When the fill
    ///     ends the buffer is empty, and the engine sits exactly where the per-draw loop leaves it. A sampler that can
    ///     finish an output WITHOUT drawing (a degenerate parameter) must not use this buffer.
    ///     <para>
    ///     A 64-bit draw is NumPy's <c>next_uint64</c>; <see cref="NextDouble"/> converts one such unit with the engine's
    ///     <see cref="BitGenerator.UnitToDouble"/>, which equals <c>next_double</c> for every NumPy engine (one word for the
    ///     64-bit generators, the same two 32-bit words for MT19937) — so a sampler mixing the two sees NumPy's stream.
    ///     </para>
    ///     The caller holds the bit generator's lock for the whole fill and provides the storage (usually <c>stackalloc</c>).
    /// </remarks>
    internal unsafe ref struct DrawBuffer64
    {
        /// <summary>The draws a refill may pull at most (2 KB, L1-resident).</summary>
        internal const int Capacity = 256;

        private readonly BitGenerator _bg;
        private readonly ulong* _buf;
        private readonly int _capacity;
        private int _pos;
        private int _avail;

        /// <summary>
        ///     The number of outputs still owed, the current one included — the most a refill may draw. Set it before each
        ///     output; 1 makes the buffer behave exactly like per-draw calls.
        /// </summary>
        internal long Owed;

        /// <summary>Creates an empty buffer over <paramref name="bg"/>.</summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="storage">At least <paramref name="capacity"/> words of scratch.</param>
        /// <param name="capacity">The storage size in words (at least 1).</param>
        internal DrawBuffer64(BitGenerator bg, ulong* storage, int capacity)
        {
            _bg = bg;
            _buf = storage;
            _capacity = capacity;
            _pos = 0;
            _avail = 0;
            Owed = 1;
        }

        /// <summary>
        ///     Resumes a read-ahead a bulk fill has been consuming inline: the words at <c>[pos, avail)</c> of
        ///     <paramref name="storage"/> are still unread, and the fill owes <paramref name="owed"/> more outputs.
        /// </summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="storage">The fill's scratch buffer.</param>
        /// <param name="capacity">Its size in words.</param>
        /// <param name="pos">The index of the next unread word.</param>
        /// <param name="avail">The number of words the last refill drew.</param>
        /// <param name="owed">The outputs still owed, the current one included.</param>
        /// <remarks>Read <see cref="Position"/> / <see cref="Available"/> back afterwards to continue inline.</remarks>
        internal DrawBuffer64(BitGenerator bg, ulong* storage, int capacity, int pos, int avail, long owed)
        {
            _bg = bg;
            _buf = storage;
            _capacity = capacity;
            _pos = pos;
            _avail = avail;
            Owed = owed;
        }

        /// <summary>The index of the next unread word.</summary>
        internal int Position => _pos;

        /// <summary>The number of words the last refill drew (the read position's bound).</summary>
        internal int Available => _avail;

        /// <summary>The next <c>next_uint64</c> draw.</summary>
        /// <returns>The word.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal ulong NextUInt64()
        {
            if (_pos == _avail)
                Refill();
            return _buf[_pos++];
        }

        /// <summary>The next <c>next_double</c> draw (one unit, converted by the engine's rule).</summary>
        /// <returns>A double in <c>[0, 1)</c>.</returns>
        internal double NextDouble() => _bg.UnitToDouble(NextUInt64());

        /// <summary>Draws the next batch: <c>min(capacity, Owed)</c> words (at least one).</summary>
        private void Refill()
        {
            long want = Owed < _capacity ? Owed : _capacity;
            _avail = want < 1 ? 1 : (int)want;
            _bg.FillUInt64(_buf, _avail);
            _pos = 0;
        }
    }

    /// <summary>
    ///     The 32-bit twin of <see cref="DrawBuffer64"/> for the float32 samplers: draws are NumPy's <c>next_uint32</c>
    ///     (buffered halves included, via <see cref="BitGenerator.FillUInt32"/>), and <see cref="NextFloat"/> is
    ///     <c>next_float</c> = <c>(next_uint32 &gt;&gt; 8) * 2**-24</c> for every engine. Same no-overdraw contract.
    /// </summary>
    internal unsafe ref struct DrawBuffer32
    {
        /// <summary>The draws a refill may pull at most (2 KB, L1-resident).</summary>
        internal const int Capacity = 512;

        private readonly BitGenerator _bg;
        private readonly uint* _buf;
        private readonly int _capacity;
        private int _pos;
        private int _avail;

        /// <summary>The number of outputs still owed, the current one included (see <see cref="DrawBuffer64.Owed"/>).</summary>
        internal long Owed;

        /// <summary>Creates an empty buffer over <paramref name="bg"/>.</summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="storage">At least <paramref name="capacity"/> words of scratch.</param>
        /// <param name="capacity">The storage size in words (at least 1).</param>
        internal DrawBuffer32(BitGenerator bg, uint* storage, int capacity)
        {
            _bg = bg;
            _buf = storage;
            _capacity = capacity;
            _pos = 0;
            _avail = 0;
            Owed = 1;
        }

        /// <summary>Resumes an inline read-ahead (see the <see cref="DrawBuffer64"/> twin).</summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="storage">The fill's scratch buffer.</param>
        /// <param name="capacity">Its size in words.</param>
        /// <param name="pos">The index of the next unread word.</param>
        /// <param name="avail">The number of words the last refill drew.</param>
        /// <param name="owed">The outputs still owed, the current one included.</param>
        internal DrawBuffer32(BitGenerator bg, uint* storage, int capacity, int pos, int avail, long owed)
        {
            _bg = bg;
            _buf = storage;
            _capacity = capacity;
            _pos = pos;
            _avail = avail;
            Owed = owed;
        }

        /// <summary>The index of the next unread word.</summary>
        internal int Position => _pos;

        /// <summary>The number of words the last refill drew (the read position's bound).</summary>
        internal int Available => _avail;

        /// <summary>The next <c>next_uint32</c> draw.</summary>
        /// <returns>The word.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal uint NextUInt32()
        {
            if (_pos == _avail)
                Refill();
            return _buf[_pos++];
        }

        /// <summary>The next <c>next_float</c> draw.</summary>
        /// <returns>A float in <c>[0, 1)</c>.</returns>
        internal float NextFloat() => (int)(NextUInt32() >> 8) * (1.0f / 16777216.0f);

        /// <summary>Draws the next batch: <c>min(capacity, Owed)</c> words (at least one).</summary>
        private void Refill()
        {
            long want = Owed < _capacity ? Owed : _capacity;
            _avail = want < 1 ? 1 : (int)want;
            _bg.FillUInt32(_buf, _avail);
            _pos = 0;
        }
    }
    /// <summary>
    ///     A read-ahead buffer of ready-made <c>next_double</c> draws for the samplers that consume ONLY doubles (NumPy's
    ///     <c>distributions.c</c> / <c>legacy-distributions.c</c> rejection samplers), refilled through the engine's
    ///     vectorized <see cref="BitGenerator.FillDouble"/> — so a read costs a load, with no per-draw conversion or virtual
    ///     call (the <see cref="DrawBuffer64"/> route converts every unit through <see cref="BitGenerator.UnitToDouble"/>).
    /// </summary>
    /// <remarks>
    ///     The no-overdraw contract is <see cref="DrawBuffer64"/>'s, counted in doubles (each <c>next_double</c> consumes
    ///     exactly one engine unit, so doubles and units are one-to-one): before producing an output the caller sets
    ///     <see cref="Owed"/> to a LOWER BOUND of the doubles the rest of the fill will draw, and a refill draws at most that
    ///     many. When the fill ends the buffer is empty and the engine sits exactly where the per-draw loop leaves it. With a
    ///     capacity of 1 the buffer IS the per-draw call sequence, which is how the scalar paths and the parameter values that
    ///     can finish an output without drawing use it. The caller holds the bit generator's lock and provides the storage.
    /// </remarks>
    internal unsafe ref struct DrawBufferDouble
    {
        /// <summary>The draws a refill may pull at most (2 KB, L1-resident).</summary>
        internal const int Capacity = 256;

        private readonly BitGenerator _bg;
        private readonly double* _buf;
        private readonly int _capacity;
        private int _pos;
        private int _avail;

        /// <summary>
        ///     A lower bound of the doubles the fill will still draw, the current output's included — the most a refill may
        ///     draw. Set it before each output (see the sampler's accounting); 1 behaves exactly like per-draw calls.
        /// </summary>
        internal long Owed;

        /// <summary>Creates an empty buffer over <paramref name="bg"/>.</summary>
        /// <param name="bg">The bit generator (lock held by the caller).</param>
        /// <param name="storage">At least <paramref name="capacity"/> doubles of scratch.</param>
        /// <param name="capacity">The storage size in doubles (at least 1; 1 = per-draw).</param>
        internal DrawBufferDouble(BitGenerator bg, double* storage, int capacity)
        {
            _bg = bg;
            _buf = storage;
            _capacity = capacity;
            _pos = 0;
            _avail = 0;
            Owed = 1;
        }

        /// <summary>The next <c>next_double</c> draw.</summary>
        /// <returns>A double in <c>[0, 1)</c>, bit-identical to the engine's own <c>next_double</c>.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal double NextDouble()
        {
            if (_pos == _avail)
                Refill();
            return _buf[_pos++];
        }

        /// <summary>Draws the next batch: <c>min(capacity, Owed)</c> doubles (at least one).</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private void Refill()
        {
            long want = Owed < _capacity ? Owed : _capacity;
            _avail = want < 1 ? 1 : (int)want;
            _bg.FillDouble(_buf, _avail);
            _pos = 0;
        }
    }
}
