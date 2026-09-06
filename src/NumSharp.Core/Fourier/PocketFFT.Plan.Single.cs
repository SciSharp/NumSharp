using System.Collections.Concurrent;

// =============================================================================
// SINGLE-PRECISION plan dispatch + cache (companion to PocketFFT.Plan.cs). The
// plan SELECTION (mixed-radix FFTPACK vs Bluestein) is identical to the double
// engine — it depends only on the transform LENGTH, not the arithmetic type, so
// a float32 transform of a given length picks the same algorithm numpy's
// complex64 path does. Only the codelet arithmetic differs (float).
// =============================================================================

namespace NumSharp.Fourier
{
    /// <summary>Single-precision flexible complex 1-D transform: FFTPACK mixed-radix or Bluestein.</summary>
    public sealed unsafe class PocketFFTComplexPlanF
    {
        private readonly CfftpF packplan;
        private readonly FftblueF blueplan;
        public long Length { get; }

        public PocketFFTComplexPlanF(long length)
        {
            Length = length;
            long tmp = (length < 50) ? 0 : PocketFFTUtil.LargestPrimeFactor(length);
            if (tmp * tmp <= length)
            {
                packplan = new CfftpF(length);
                return;
            }
            double comp1 = PocketFFTUtil.CostGuess(length);
            double comp2 = 2 * PocketFFTUtil.CostGuess(PocketFFTUtil.GoodSizeCmplx(2 * length - 1));
            comp2 *= 1.5; // fudge factor that appears to give good overall performance
            if (comp2 < comp1)
                blueplan = new FftblueF(length);
            else
                packplan = new CfftpF(length);
        }

        public void Exec(CmplxF* c, float fct, bool fwd)
        {
            if (packplan != null) packplan.Exec(c, fct, fwd);
            else blueplan.Exec(c, fct, fwd);
        }
    }

    /// <summary>Single-precision flexible real 1-D transform: FFTPACK mixed-radix or Bluestein.</summary>
    public sealed unsafe class PocketFFTRealPlanF
    {
        private readonly RfftpF packplan;
        private readonly FftblueF blueplan;
        public long Length { get; }

        public PocketFFTRealPlanF(long length)
        {
            Length = length;
            long tmp = (length < 50) ? 0 : PocketFFTUtil.LargestPrimeFactor(length);
            if (tmp * tmp <= length)
            {
                packplan = new RfftpF(length);
                return;
            }
            double comp1 = 0.5 * PocketFFTUtil.CostGuess(length);
            double comp2 = 2 * PocketFFTUtil.CostGuess(PocketFFTUtil.GoodSizeCmplx(2 * length - 1));
            comp2 *= 1.5; // fudge factor that appears to give good overall performance
            if (comp2 < comp1)
                blueplan = new FftblueF(length);
            else
                packplan = new RfftpF(length);
        }

        // r2hc == fwd (forward real->halfcomplex); backward is halfcomplex->real.
        public void Exec(float* c, float fct, bool r2hc)
        {
            if (packplan != null) packplan.Exec(c, fct, r2hc);
            else blueplan.ExecR(c, fct, r2hc);
        }
    }

    /// <summary>Bounded single-precision plan cache (companion to <see cref="PocketFFTPlanCache"/>,
    /// with an independent budget so float and double plans never evict one another). Plans are
    /// immutable + deterministic, so sharing them is bit-neutral.</summary>
    public static class PocketFFTPlanCacheF
    {
        private const int MaxEntries = 64;
        private static readonly ConcurrentDictionary<long, PocketFFTComplexPlanF> _cplx = new();
        private static readonly ConcurrentDictionary<long, PocketFFTRealPlanF> _real = new();

        public static PocketFFTComplexPlanF GetComplex(long len)
        {
            if (_cplx.TryGetValue(len, out var p)) return p;
            p = new PocketFFTComplexPlanF(len);
            if (_cplx.Count >= MaxEntries) _cplx.Clear();
            _cplx[len] = p;
            return p;
        }

        public static PocketFFTRealPlanF GetReal(long len)
        {
            if (_real.TryGetValue(len, out var p)) return p;
            p = new PocketFFTRealPlanF(len);
            if (_real.Count >= MaxEntries) _real.Clear();
            _real[len] = p;
            return p;
        }
    }
}
