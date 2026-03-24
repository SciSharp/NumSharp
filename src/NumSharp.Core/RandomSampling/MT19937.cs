using System;

namespace NumSharp
{
    /// <summary>
    ///     Mersenne Twister MT19937 pseudo-random number generator.
    ///     This implementation matches NumPy's MT19937 exactly, producing
    ///     identical sequences for the same seed.
    /// </summary>
    /// <remarks>
    ///     Based on the original C implementation by Takuji Nishimura and Makoto Matsumoto.
    ///     http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
    ///
    ///     NumPy reference:
    ///     https://github.com/numpy/numpy/blob/main/numpy/random/src/mt19937/
    /// </remarks>
    public sealed class MT19937 : ICloneable
    {
        // Period parameters
        private const int N = 624;
        private const int M = 397;
        private const uint MATRIX_A = 0x9908b0dfU;    // Constant vector a
        private const uint UPPER_MASK = 0x80000000U;  // Most significant w-r bits
        private const uint LOWER_MASK = 0x7fffffffU;  // Least significant r bits

        // Tempering parameters
        private const uint TEMPERING_MASK_B = 0x9d2c5680U;
        private const uint TEMPERING_MASK_C = 0xefc60000U;

        // State array
        private readonly uint[] _key = new uint[N];
        private int _pos;

        /// <summary>
        ///     Gets the internal state array (for serialization).
        /// </summary>
        public uint[] Key => _key;

        /// <summary>
        ///     Gets the current position in the state array.
        /// </summary>
        public int Pos => _pos;

        /// <summary>
        ///     Initializes a new instance with a time-based seed.
        /// </summary>
        public MT19937()
        {
            Seed((uint)Environment.TickCount);
        }

        /// <summary>
        ///     Initializes a new instance with the specified seed.
        /// </summary>
        /// <param name="seed">The seed value.</param>
        public MT19937(uint seed)
        {
            Seed(seed);
        }

        /// <summary>
        ///     Initializes a new instance with the specified seed.
        /// </summary>
        /// <param name="seed">The seed value (converted to uint).</param>
        public MT19937(int seed)
        {
            Seed((uint)seed);
        }

        /// <summary>
        ///     Seeds the generator with a single integer.
        ///     This matches NumPy's seeding algorithm exactly.
        /// </summary>
        /// <param name="seed">The seed value.</param>
        public void Seed(uint seed)
        {
            _key[0] = seed;
            for (int i = 1; i < N; i++)
            {
                // This uses the same algorithm as NumPy/C MT19937
                _key[i] = 1812433253U * (_key[i - 1] ^ (_key[i - 1] >> 30)) + (uint)i;
            }
            _pos = N; // Force generation on first call
        }

        /// <summary>
        ///     Seeds the generator with an array of integers.
        ///     This matches NumPy's init_by_array function exactly.
        /// </summary>
        /// <param name="initKey">Array of seed values.</param>
        public void SeedByArray(uint[] initKey)
        {
            if (initKey == null || initKey.Length == 0)
            {
                Seed(0);
                return;
            }

            // First, seed with 19650218 (NumPy's magic number)
            Seed(19650218U);

            int i = 1;
            int j = 0;
            int k = N > initKey.Length ? N : initKey.Length;

            for (; k > 0; k--)
            {
                // Non-linear mixing
                _key[i] = (_key[i] ^ ((_key[i - 1] ^ (_key[i - 1] >> 30)) * 1664525U)) + initKey[j] + (uint)j;
                i++;
                j++;
                if (i >= N)
                {
                    _key[0] = _key[N - 1];
                    i = 1;
                }
                if (j >= initKey.Length)
                    j = 0;
            }

            for (k = N - 1; k > 0; k--)
            {
                _key[i] = (_key[i] ^ ((_key[i - 1] ^ (_key[i - 1] >> 30)) * 1566083941U)) - (uint)i;
                i++;
                if (i >= N)
                {
                    _key[0] = _key[N - 1];
                    i = 1;
                }
            }

            // MSB is 1; assuring non-zero initial array
            _key[0] = 0x80000000U;
            _pos = N; // Force generation on first call
        }

        /// <summary>
        ///     Generates 624 new random numbers (the "twist" operation).
        /// </summary>
        private void Generate()
        {
            uint y;
            uint[] mag01 = { 0x0U, MATRIX_A };

            int kk;
            for (kk = 0; kk < N - M; kk++)
            {
                y = (_key[kk] & UPPER_MASK) | (_key[kk + 1] & LOWER_MASK);
                _key[kk] = _key[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; kk++)
            {
                y = (_key[kk] & UPPER_MASK) | (_key[kk + 1] & LOWER_MASK);
                _key[kk] = _key[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (_key[N - 1] & UPPER_MASK) | (_key[0] & LOWER_MASK);
            _key[N - 1] = _key[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];

            _pos = 0;
        }

        /// <summary>
        ///     Returns a random unsigned 32-bit integer.
        /// </summary>
        /// <returns>A random uint in [0, 2^32).</returns>
        public uint NextUInt32()
        {
            if (_pos >= N)
                Generate();

            uint y = _key[_pos++];

            // Tempering
            y ^= (y >> 11);
            y ^= (y << 7) & TEMPERING_MASK_B;
            y ^= (y << 15) & TEMPERING_MASK_C;
            y ^= (y >> 18);

            return y;
        }

        /// <summary>
        ///     Returns a random double in [0, 1) with 53-bit precision.
        ///     This matches NumPy's random_standard_uniform exactly.
        /// </summary>
        /// <returns>A random double in [0, 1).</returns>
        public double NextDouble()
        {
            // NumPy uses 53-bit precision for doubles
            // Take high 27 bits from first call, high 26 bits from second call
            uint a = NextUInt32() >> 5;   // 27 bits
            uint b = NextUInt32() >> 6;   // 26 bits

            // Combine to form 53-bit integer and divide by 2^53
            return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
        }

        /// <summary>
        ///     Returns a random signed 32-bit integer.
        /// </summary>
        /// <returns>A random int in [0, Int32.MaxValue].</returns>
        public int NextInt()
        {
            return (int)(NextUInt32() >> 1);
        }

        /// <summary>
        ///     Returns a random long in [0, Int64.MaxValue].
        /// </summary>
        /// <returns>A random long.</returns>
        public long NextLong()
        {
            // Combine two 32-bit values, mask off sign bit
            return (long)(((ulong)NextUInt32() << 32) | NextUInt32()) & long.MaxValue;
        }

        /// <summary>
        ///     Returns a random long in [low, high) using NumPy's algorithm.
        ///     NumPy uses: floor(nextDouble() * range) + low for small ranges.
        ///     This matches NumPy's legacy RandomState.randint() exactly.
        /// </summary>
        /// <param name="low">The inclusive lower bound.</param>
        /// <param name="high">The exclusive upper bound.</param>
        /// <returns>A random long in [low, high).</returns>
        public long NextLongNumPy(long low, long high)
        {
            if (low >= high)
                return low;

            // NumPy's legacy randint uses: floor(random_double() * range) + low
            // This is simpler than rejection sampling and matches NumPy exactly
            long range = high - low;
            return (long)(NextDouble() * range) + low;
        }

        /// <summary>
        ///     Returns a random integer in [0, maxValue).
        ///     Uses rejection sampling for unbiased results (matches NumPy).
        /// </summary>
        /// <param name="maxValue">The exclusive upper bound.</param>
        /// <returns>A random int in [0, maxValue).</returns>
        public int Next(int maxValue)
        {
            if (maxValue <= 0)
                return 0;

            // For small ranges, use rejection sampling to avoid bias
            uint range = (uint)maxValue;

            // Find the smallest power of 2 >= range
            uint mask = range - 1;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;

            uint result;
            do
            {
                result = NextUInt32() & mask;
            } while (result >= range);

            return (int)result;
        }

        /// <summary>
        ///     Returns a random integer in [minValue, maxValue).
        ///     Uses rejection sampling for unbiased results.
        /// </summary>
        /// <param name="minValue">The inclusive lower bound.</param>
        /// <param name="maxValue">The exclusive upper bound.</param>
        /// <returns>A random int in [minValue, maxValue).</returns>
        public int Next(int minValue, int maxValue)
        {
            if (minValue >= maxValue)
                return minValue;

            return minValue + Next(maxValue - minValue);
        }

        /// <summary>
        ///     Returns a random long in [0, maxValue).
        ///     Uses rejection sampling for unbiased results.
        /// </summary>
        /// <param name="maxValue">The exclusive upper bound.</param>
        /// <returns>A random long in [0, maxValue).</returns>
        public long NextLong(long maxValue)
        {
            if (maxValue <= 0)
                return 0;

            ulong range = (ulong)maxValue;

            // Find the smallest power of 2 >= range
            ulong mask = range - 1;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;

            ulong result;
            do
            {
                // Combine two 32-bit values for 64-bit range
                result = ((ulong)NextUInt32() << 32) | NextUInt32();
                result &= mask;
            } while (result >= range);

            return (long)result;
        }

        /// <summary>
        ///     Returns a random long in [minValue, maxValue).
        ///     Uses rejection sampling for unbiased results.
        /// </summary>
        /// <param name="minValue">The inclusive lower bound.</param>
        /// <param name="maxValue">The exclusive upper bound.</param>
        /// <returns>A random long in [minValue, maxValue).</returns>
        public long NextLong(long minValue, long maxValue)
        {
            if (minValue >= maxValue)
                return minValue;

            return minValue + NextLong(maxValue - minValue);
        }

        /// <summary>
        ///     Fills a byte array with random bytes.
        /// </summary>
        /// <param name="buffer">The array to fill.</param>
        public void NextBytes(byte[] buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            int i = 0;
            while (i + 4 <= buffer.Length)
            {
                uint r = NextUInt32();
                buffer[i++] = (byte)r;
                buffer[i++] = (byte)(r >> 8);
                buffer[i++] = (byte)(r >> 16);
                buffer[i++] = (byte)(r >> 24);
            }

            if (i < buffer.Length)
            {
                uint r = NextUInt32();
                while (i < buffer.Length)
                {
                    buffer[i++] = (byte)r;
                    r >>= 8;
                }
            }
        }

        /// <summary>
        ///     Sets the internal state from serialized state data.
        /// </summary>
        /// <param name="key">The state array (must be length 624).</param>
        /// <param name="pos">The position in the state array (0-624).</param>
        public void SetState(uint[] key, int pos)
        {
            if (key == null || key.Length != N)
                throw new ArgumentException($"Key array must be length {N}", nameof(key));
            if (pos < 0 || pos > N)
                throw new ArgumentOutOfRangeException(nameof(pos), $"Position must be in [0, {N}]");

            Array.Copy(key, _key, N);
            _pos = pos;
        }

        /// <summary>
        ///     Creates a deep copy of this generator.
        /// </summary>
        /// <returns>A new MT19937 instance with identical state.</returns>
        public MT19937 Clone()
        {
            var clone = new MT19937();
            Array.Copy(_key, clone._key, N);
            clone._pos = _pos;
            return clone;
        }

        object ICloneable.Clone() => Clone();
    }
}
