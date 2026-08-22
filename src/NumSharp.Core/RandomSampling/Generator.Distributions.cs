using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // Bit-exact log1p: Math.Log matches ucrtbase log on win-amd64 (NumPy's npy_log = log), and
        // the Kahan/Goldberg correction recovers the log1p precision the raw Math.Log(1+x) loses.
        // NumPy's npy_log1p is `#define npy_log1p log1p` (the CRT), so this reproduces it bit-for-bit
        // (verified 0-diff over 300k values), which is what keeps the ziggurat tail byte-exact.
        internal static double Log1p(double x)
        {
            double u = 1.0 + x;
            if (u == 1.0)
                return x;
            double y = Math.Log(u);
            if (u > 2.0)
                return y;
            return y - ((u - 1.0) - x) / u;
        }

        // ---- standard normal (ziggurat) : numpy random_standard_normal ----

        internal double NextStandardNormal()
        {
            for (;;)
            {
                ulong r = _bitGenerator.NextUInt64();
                int idx = (int)(r & 0xff);
                r >>= 8;
                int sign = (int)(r & 0x1);
                ulong rabs = (r >> 1) & 0x000fffffffffffffUL;
                double x = rabs * ZigguratTables.wi_double[idx];
                if ((sign & 0x1) != 0)
                    x = -x;
                if (rabs < ZigguratTables.ki_double[idx])
                    return x; // 99.3% of the time
                if (idx == 0)
                {
                    for (;;)
                    {
                        double xx = -ZigguratTables.ziggurat_nor_inv_r * Log1p(-_bitGenerator.NextDouble());
                        double yy = -Log1p(-_bitGenerator.NextDouble());
                        if (yy + yy > xx * xx)
                            return ((rabs >> 8) & 0x1) != 0
                                ? -(ZigguratTables.ziggurat_nor_r + xx)
                                : ZigguratTables.ziggurat_nor_r + xx;
                    }
                }
                else
                {
                    if (((ZigguratTables.fi_double[idx - 1] - ZigguratTables.fi_double[idx]) * _bitGenerator.NextDouble()
                         + ZigguratTables.fi_double[idx]) < Math.Exp(-0.5 * x * x))
                        return x;
                }
            }
        }

        // ---- standard exponential (ziggurat) : numpy random_standard_exponential ----

        internal double NextStandardExponential()
        {
            ulong ri = _bitGenerator.NextUInt64();
            ri >>= 3;
            int idx = (int)(ri & 0xFF);
            ri >>= 8;
            double x = ri * ZigguratTables.we_double[idx];
            if (ri < ZigguratTables.ke_double[idx])
                return x; // 98.9% of the time
            return StandardExponentialUnlikely(idx, x);
        }

        private double StandardExponentialUnlikely(int idx, double x)
        {
            if (idx == 0)
                return ZigguratTables.ziggurat_exp_r - Log1p(-_bitGenerator.NextDouble());
            if ((ZigguratTables.fe_double[idx - 1] - ZigguratTables.fe_double[idx]) * _bitGenerator.NextDouble()
                + ZigguratTables.fe_double[idx] < Math.Exp(-x))
                return x;
            return NextStandardExponential();
        }

        // numpy random_standard_exponential_inv : the method='inv' inverse-CDF sampler.
        internal double NextStandardExponentialInv()
        {
            return -Log1p(-_bitGenerator.NextDouble());
        }

        // float32 Kahan log1p, mirroring the double form (MathF.Log ~ ucrtbase logf on win-amd64).
        internal static float Log1pF(float x)
        {
            float u = 1.0f + x;
            if (u == 1.0f)
                return x;
            float y = MathF.Log(u);
            if (u > 2.0f)
                return y;
            return y - ((u - 1.0f) - x) / u;
        }

        // ---- float32 ziggurat : numpy random_standard_normal_f / _exponential_f ----

        internal float NextStandardNormalF()
        {
            for (;;)
            {
                uint r = _bitGenerator.NextUInt32();
                int idx = (int)(r & 0xff);
                int sign = (int)((r >> 8) & 0x1);
                uint rabs = (r >> 9) & 0x007fffff; // 23-bit mantissa mask
                float x = rabs * ZigguratTables.wi_float[idx];
                if ((sign & 0x1) != 0)
                    x = -x;
                if (rabs < ZigguratTables.ki_float[idx])
                    return x;
                if (idx == 0)
                {
                    for (;;)
                    {
                        float xx = -ZigguratTables.ziggurat_nor_inv_r_f * Log1pF(-_bitGenerator.NextFloat());
                        float yy = -Log1pF(-_bitGenerator.NextFloat());
                        if (yy + yy > xx * xx)
                            return ((rabs >> 8) & 0x1) != 0
                                ? -(ZigguratTables.ziggurat_nor_r_f + xx)
                                : ZigguratTables.ziggurat_nor_r_f + xx;
                    }
                }
                else
                {
                    // NumPy uses double exp here (comparison promotes the float LHS to double).
                    if (((ZigguratTables.fi_float[idx - 1] - ZigguratTables.fi_float[idx]) * _bitGenerator.NextFloat()
                         + ZigguratTables.fi_float[idx]) < Math.Exp(-0.5 * x * x))
                        return x;
                }
            }
        }

        internal float NextStandardExponentialF()
        {
            uint ri = _bitGenerator.NextUInt32();
            ri >>= 1;
            int idx = (int)(ri & 0xFF);
            ri >>= 8;
            float x = ri * ZigguratTables.we_float[idx];
            if (ri < ZigguratTables.ke_float[idx])
                return x;
            return StandardExponentialUnlikelyF(idx, x);
        }

        private float StandardExponentialUnlikelyF(int idx, float x)
        {
            if (idx == 0)
                return ZigguratTables.ziggurat_exp_r_f - Log1pF(-_bitGenerator.NextFloat());
            if ((ZigguratTables.fe_float[idx - 1] - ZigguratTables.fe_float[idx]) * _bitGenerator.NextFloat()
                + ZigguratTables.fe_float[idx] < MathF.Exp(-x)) // NumPy uses expf here
                return x;
            return NextStandardExponentialF();
        }

        // numpy random_standard_exponential_inv_fill_f : double log1p of the float draw, stored as float.
        internal float NextStandardExponentialInvF()
        {
            return (float)(-Log1p(-(double)_bitGenerator.NextFloat()));
        }

        // ---- standard gamma : numpy random_standard_gamma ----

        internal double NextStandardGamma(double shape)
        {
            if (shape == 1.0)
                return NextStandardExponential();
            if (shape == 0.0)
                return 0.0;
            if (shape < 1.0)
            {
                for (;;)
                {
                    double U = _bitGenerator.NextDouble();
                    double V = NextStandardExponential();
                    if (U <= 1.0 - shape)
                    {
                        double X = Math.Pow(U, 1.0 / shape);
                        if (X <= V)
                            return X;
                    }
                    else
                    {
                        double Y = -Math.Log((1.0 - U) / shape);
                        double X = Math.Pow(1.0 - shape + shape * Y, 1.0 / shape);
                        if (X <= V + Y)
                            return X;
                    }
                }
            }
            else
            {
                double b = shape - 1.0 / 3.0;
                double c = 1.0 / Math.Sqrt(9.0 * b);
                for (;;)
                {
                    double X, V;
                    do
                    {
                        X = NextStandardNormal();
                        V = 1.0 + c * X;
                    } while (V <= 0.0);

                    V = V * V * V;
                    double U = _bitGenerator.NextDouble();
                    if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                        return b * V;
                    if (Math.Log(U) < 0.5 * X * X + b * (1.0 - V + Math.Log(V)))
                        return b * V;
                }
            }
        }
    }
}
