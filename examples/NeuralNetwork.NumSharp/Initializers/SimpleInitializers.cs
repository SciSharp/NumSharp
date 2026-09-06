using System;
using NumSharp;
using NumSharp.Backends;

namespace NeuralNetwork.NumSharp.Initializers
{
    /// <summary>All-zeros initializer (the standard bias default).</summary>
    public class Zeros : BaseInitializer
    {
        public Zeros() : base("zeros") { }

        public override NDArray Initialize(Shape shape)
            => np.zeros(shape, NPTypeCode.Single);
    }

    /// <summary>All-ones initializer (e.g. BatchNorm gamma).</summary>
    public class Ones : BaseInitializer
    {
        public Ones() : base("ones") { }

        public override NDArray Initialize(Shape shape)
            => np.ones(shape, NPTypeCode.Single);
    }

    /// <summary>Constant-fill initializer.</summary>
    public class Constant : BaseInitializer
    {
        public float Value { get; }

        public Constant(float value) : base("constant")
        {
            Value = value;
        }

        public override NDArray Initialize(Shape shape)
            => np.full(shape, Value, typeof(float));
    }

    /// <summary>
    /// Untruncated Gaussian draw. Keras defaults: mean 0, stddev 0.05.
    /// </summary>
    public class RandomNormal : BaseInitializer
    {
        public float Mean { get; }
        public float Stddev { get; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f) : base("random_normal")
        {
            Mean = mean;
            Stddev = stddev;
        }

        public override NDArray Initialize(Shape shape)
            => np.random.normal(Mean, Stddev, shape).astype(NPTypeCode.Single);
    }

    /// <summary>
    /// Uniform draw over [minval, maxval). Keras defaults: [-0.05, 0.05).
    /// </summary>
    public class RandomUniform : BaseInitializer
    {
        public float MinVal { get; }
        public float MaxVal { get; }

        public RandomUniform(float minval = -0.05f, float maxval = 0.05f) : base("random_uniform")
        {
            MinVal = minval;
            MaxVal = maxval;
        }

        public override NDArray Initialize(Shape shape)
            => np.random.uniform(MinVal, MaxVal, shape).astype(NPTypeCode.Single);
    }
}
