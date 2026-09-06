using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Initializers
{
    /// <summary>Which fan feeds the variance denominator.</summary>
    public enum FanMode
    {
        FanIn,
        FanOut,
        FanAvg,
    }

    /// <summary>The sampling distribution VarianceScaling draws from.</summary>
    public enum ScalingDistribution
    {
        /// <summary>
        /// Gaussian truncated at ±2 stddev, with the stddev pre-divided by
        /// 0.87962566103423978 (the std of a unit normal truncated to [-2, 2])
        /// so the RESULTING draw has the requested variance — exactly what
        /// Keras does for glorot_normal / he_normal / lecun_normal.
        /// </summary>
        TruncatedNormal,

        /// <summary>Plain Gaussian, stddev = sqrt(scale/n) (PyTorch-style *_normal).</summary>
        UntruncatedNormal,

        /// <summary>Uniform over [-limit, limit], limit = sqrt(3*scale/n).</summary>
        Uniform,
    }

    /// <summary>
    /// The Keras VarianceScaling workhorse: draws with variance scale/n where
    /// n is fan_in / fan_out / their average. Glorot/Xavier, He and LeCun
    /// initializers below are thin parameterizations of this class.
    /// </summary>
    public class VarianceScaling : BaseInitializer
    {
        // scipy.stats.truncnorm.std(a=-2, b=2, loc=0, scale=1) — Keras's constant.
        private const double TruncatedStdCorrection = 0.87962566103423978;

        public double Scale { get; }
        public FanMode Mode { get; }
        public ScalingDistribution Distribution { get; }

        public VarianceScaling(
            double scale = 1.0,
            FanMode mode = FanMode.FanIn,
            ScalingDistribution distribution = ScalingDistribution.TruncatedNormal,
            string name = "variance_scaling") : base(name)
        {
            if (scale <= 0)
                throw new ArgumentOutOfRangeException(nameof(scale), "scale must be positive.");
            Scale = scale;
            Mode = mode;
            Distribution = distribution;
        }

        public override NDArray Initialize(Shape shape)
        {
            var (fanIn, fanOut) = ComputeFans(shape);
            double n = Mode switch
            {
                FanMode.FanIn => Math.Max(1.0, fanIn),
                FanMode.FanOut => Math.Max(1.0, fanOut),
                _ => Math.Max(1.0, (fanIn + fanOut) / 2.0),
            };

            switch (Distribution)
            {
                case ScalingDistribution.TruncatedNormal:
                {
                    double stddev = Math.Sqrt(Scale / n) / TruncatedStdCorrection;
                    return DrawTruncatedNormal(shape, stddev);
                }
                case ScalingDistribution.UntruncatedNormal:
                {
                    double stddev = Math.Sqrt(Scale / n);
                    return np.random.normal(0.0, stddev, shape).astype(NPTypeCode.Single);
                }
                default:
                {
                    double limit = Math.Sqrt(3.0 * Scale / n);
                    return np.random.uniform(-limit, limit, shape).astype(NPTypeCode.Single);
                }
            }
        }

        /// <summary>
        /// Gaussian truncated to ±2*stddev by rejection: bulk-draw the tensor,
        /// then redraw the ~4.6% of entries outside the bound until all are
        /// inside. All draws come from np.random, so the result is fully
        /// deterministic under np.random.seed.
        /// </summary>
        private static unsafe NDArray DrawTruncatedNormal(Shape shape, double stddev)
        {
            NDArray nd = np.random.normal(0.0, stddev, shape);  // float64, fresh contiguous
            double bound = 2.0 * stddev;
            double* p = (double*)nd.Unsafe.Address;
            long size = nd.size;
            for (long i = 0; i < size; i++)
            {
                while (Math.Abs(p[i]) > bound)
                    p[i] = np.random.normal(0.0, stddev, new Shape(1)).GetDouble(0);
            }

            return nd.astype(NPTypeCode.Single);
        }
    }

    /// <summary>Glorot/Xavier uniform (Keras Dense default): scale 1, fan_avg, uniform.</summary>
    public class GlorotUniform : VarianceScaling
    {
        public GlorotUniform() : base(1.0, FanMode.FanAvg, ScalingDistribution.Uniform, "glorot_uniform") { }
    }

    /// <summary>Glorot/Xavier normal: scale 1, fan_avg, truncated normal.</summary>
    public class GlorotNormal : VarianceScaling
    {
        public GlorotNormal() : base(1.0, FanMode.FanAvg, ScalingDistribution.TruncatedNormal, "glorot_normal") { }
    }

    /// <summary>He uniform (for ReLU family): scale 2, fan_in, uniform.</summary>
    public class HeUniform : VarianceScaling
    {
        public HeUniform() : base(2.0, FanMode.FanIn, ScalingDistribution.Uniform, "he_uniform") { }
    }

    /// <summary>He normal (for ReLU family): scale 2, fan_in, truncated normal.</summary>
    public class HeNormal : VarianceScaling
    {
        public HeNormal() : base(2.0, FanMode.FanIn, ScalingDistribution.TruncatedNormal, "he_normal") { }
    }

    /// <summary>LeCun uniform: scale 1, fan_in, uniform.</summary>
    public class LecunUniform : VarianceScaling
    {
        public LecunUniform() : base(1.0, FanMode.FanIn, ScalingDistribution.Uniform, "lecun_uniform") { }
    }

    /// <summary>LeCun normal (pair with SELU): scale 1, fan_in, truncated normal.</summary>
    public class LecunNormal : VarianceScaling
    {
        public LecunNormal() : base(1.0, FanMode.FanIn, ScalingDistribution.TruncatedNormal, "lecun_normal") { }
    }
}
