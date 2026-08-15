using System.Collections.Concurrent;

// =============================================================================
// Port of pocketfft's pocketfft_c<T0> / pocketfft_r<T0> plan dispatch
// (pocketfft_hdronly.h lines 2515-2583): choose the mixed-radix FFTPACK plan
// when largest_prime_factor(n)^2 <= n, else cost-compare it against Bluestein.
// Plus a bounded per-length plan cache keyed by (len, isReal) — pocketfft's
// numpy build has POCKETFFT_CACHE_SIZE==0 (no cache, a fresh plan per call), so
// caching only saves the (deterministic) twiddle setup and never changes bits.
// =============================================================================

namespace NumSharp.Fourier
{
    /// <summary>Flexible complex 1-D transform: FFTPACK mixed-radix or Bluestein.</summary>
    public sealed unsafe class PocketFFTComplexPlan
    {
        private readonly Cfftp packplan;
        private readonly Fftblue blueplan;
        public long Length { get; }

        public PocketFFTComplexPlan(long length)
        {
            Length = length;
            long tmp = (length < 50) ? 0 : PocketFFTUtil.LargestPrimeFactor(length);
            if (tmp * tmp <= length)
            {
                packplan = new Cfftp(length);
                return;
            }
            double comp1 = PocketFFTUtil.CostGuess(length);
            double comp2 = 2 * PocketFFTUtil.CostGuess(PocketFFTUtil.GoodSizeCmplx(2 * length - 1));
            comp2 *= 1.5; // fudge factor that appears to give good overall performance
            if (comp2 < comp1)
                blueplan = new Fftblue(length);
            else
                packplan = new Cfftp(length);
        }

        public void Exec(Cmplx* c, double fct, bool fwd)
        {
            if (packplan != null) packplan.Exec(c, fct, fwd);
            else blueplan.Exec(c, fct, fwd);
        }
    }

    /// <summary>Flexible real 1-D transform: FFTPACK mixed-radix or Bluestein.</summary>
    public sealed unsafe class PocketFFTRealPlan
    {
        private readonly Rfftp packplan;
        private readonly Fftblue blueplan;
        public long Length { get; }

        public PocketFFTRealPlan(long length)
        {
            Length = length;
            long tmp = (length < 50) ? 0 : PocketFFTUtil.LargestPrimeFactor(length);
            if (tmp * tmp <= length)
            {
                packplan = new Rfftp(length);
                return;
            }
            double comp1 = 0.5 * PocketFFTUtil.CostGuess(length);
            double comp2 = 2 * PocketFFTUtil.CostGuess(PocketFFTUtil.GoodSizeCmplx(2 * length - 1));
            comp2 *= 1.5; // fudge factor that appears to give good overall performance
            if (comp2 < comp1)
                blueplan = new Fftblue(length);
            else
                packplan = new Rfftp(length);
        }

        // r2hc == fwd (forward real->halfcomplex); backward is halfcomplex->real.
        public void Exec(double* c, double fct, bool r2hc)
        {
            if (packplan != null) packplan.Exec(c, fct, r2hc);
            else blueplan.ExecR(c, fct, r2hc);
        }
    }

    /// <summary>Bounded plan cache. Plans are immutable + deterministic, so sharing them across
    /// transforms of the same length is safe and reproduces numpy bit-for-bit (only the twiddle
    /// setup is saved). Bounded so a program touching many distinct lengths cannot leak.</summary>
    public static class PocketFFTPlanCache
    {
        private const int MaxEntries = 64;
        private static readonly ConcurrentDictionary<long, PocketFFTComplexPlan> _cplx = new();
        private static readonly ConcurrentDictionary<long, PocketFFTRealPlan> _real = new();

        public static PocketFFTComplexPlan GetComplex(long len)
        {
            if (_cplx.TryGetValue(len, out var p)) return p;
            p = new PocketFFTComplexPlan(len);
            if (_cplx.Count >= MaxEntries) _cplx.Clear();
            _cplx[len] = p;
            return p;
        }

        public static PocketFFTRealPlan GetReal(long len)
        {
            if (_real.TryGetValue(len, out var p)) return p;
            p = new PocketFFTRealPlan(len);
            if (_real.Count >= MaxEntries) _real.Clear();
            _real[len] = p;
            return p;
        }
    }
}
