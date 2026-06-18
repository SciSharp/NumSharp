#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.IO;

// Validate a PairwiseFold replica reproduces NumPy's pairwise_sum BIT-FOR-BIT.
// Generates deterministic data (LCG), writes raw bytes for the Python side to
// np.fromfile, computes PairwiseFold, prints the result's raw bit pattern.
// pairwise_parity.py reads the same bytes, runs np.add.reduce, prints its bits.
// A bit-identical match across all sizes proves exact float32/float64 parity.

string dir = Path.GetTempPath();
int[] sizes = { 1, 7, 8, 9, 100, 128, 129, 200, 256, 1000, 3163, 10000, 100000 };

// deterministic LCG in [-50, 50], representable in both f64 and f32
static double Next(ref ulong s)
{
    s = s * 6364136223846793005UL + 1442695040888963407UL;
    double u = (s >> 11) * (1.0 / (1UL << 53)); // [0,1)
    return (u - 0.5) * 100.0;
}

foreach (int n in sizes)
{
    ulong s = 0x1234567 ^ (ulong)n;
    var d = new double[n];
    var f = new float[n];
    for (int i = 0; i < n; i++) { double v = Math.Round(Next(ref s), 4); d[i] = v; f[i] = (float)v; }

    File.WriteAllBytes(Path.Combine(dir, $"pw_f64_{n}.bin"), DoubleBytes(d));
    File.WriteAllBytes(Path.Combine(dir, $"pw_f32_{n}.bin"), FloatBytes(f));

    double rd; unsafe { fixed (double* p = d) rd = PairwiseD(p, n, 1); }
    float rf; unsafe { fixed (float* p = f) rf = PairwiseF(p, n, 1); }

    Console.WriteLine($"f64 {n} {BitConverter.DoubleToInt64Bits(rd):X16}");
    Console.WriteLine($"f32 {n} {BitConverter.SingleToInt32Bits(rf):X8}");
}

static byte[] DoubleBytes(double[] a) { var b = new byte[a.Length * 8]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }
static byte[] FloatBytes(float[] a) { var b = new byte[a.Length * 4]; Buffer.BlockCopy(a, 0, b, 0, b.Length); return b; }

// ---- NumPy pairwise_sum replica (loops_utils.h.src), stride in ELEMENTS ----
static unsafe double PairwiseD(double* a, long n, long stride)
{
    if (n < 8)
    {
        double res = -0.0;
        for (long i = 0; i < n; i++) res += a[i * stride];
        return res;
    }
    if (n <= 128)
    {
        double* r = stackalloc double[8];
        for (int k = 0; k < 8; k++) r[k] = a[k * stride];
        long i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[(i + k) * stride];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i * stride];
        return res;
    }
    long n2 = n / 2; n2 -= n2 % 8;
    return PairwiseD(a, n2, stride) + PairwiseD(a + n2 * stride, n - n2, stride);
}

static unsafe float PairwiseF(float* a, long n, long stride)
{
    if (n < 8)
    {
        float res = -0.0f;
        for (long i = 0; i < n; i++) res += a[i * stride];
        return res;
    }
    if (n <= 128)
    {
        float* r = stackalloc float[8];
        for (int k = 0; k < 8; k++) r[k] = a[k * stride];
        long i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[(i + k) * stride];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i * stride];
        return res;
    }
    long n2 = n / 2; n2 -= n2 % 8;
    return PairwiseF(a, n2, stride) + PairwiseF(a + n2 * stride, n - n2, stride);
}
