#:project ../NeuralNetwork.NumSharp.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// P0 + P2 verification for NeuralNetwork.NumSharp (86 checks).
// Run from THIS directory (file-based apps need a csproj-free CWD):
//   cd examples/NeuralNetwork.NumSharp/tests && dotnet run verify_p0_p2.cs
// #:project resolves relative to this file.
// Sections: P0 behavior fixes, activation values+FD gradients, loss values+FD
// gradients (NumPy 2.4.2 reference constants), metrics, initializers.
using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.NumSharp;
using NeuralNetwork.NumSharp.Activations;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Initializers;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.Metrics;
using NeuralNetwork.NumSharp.MnistMlp;
using NeuralNetwork.NumSharp.Optimizers;
using NumSharp;
using NumSharp.Backends;

int pass = 0, fail = 0;
void Check(string name, bool ok, string detail = "")
{
    if (ok) { pass++; }
    else { fail++; Console.WriteLine($"  FAIL {name} {detail}"); }
}
void CheckClose(string name, float actual, float expected, float tol = 1e-5f)
    => Check(name, Math.Abs(actual - expected) <= tol * Math.Max(1f, Math.Abs(expected)),
             $"actual={actual} expected={expected}");
void CheckArr(string name, NDArray actual, float[] expected, float tol = 1e-5f)
{
    var flat = np.reshape(actual, new Shape(actual.size)).astype(NPTypeCode.Single);
    for (int i = 0; i < expected.Length; i++)
    {
        float a = flat.GetSingle(i);
        if (Math.Abs(a - expected[i]) > tol * Math.Max(1f, Math.Abs(expected[i])))
        { Check(name, false, $"[{i}] actual={a} expected={expected[i]}"); return; }
    }
    Check(name, true);
}

NDArray X() => np.array(new float[] { -3, -1.5f, -0.5f, 0, 0.5f, 1.5f, 3 });

// ============================ P0 =============================
Console.WriteLine("--- P0 fixes ---");

// astype(copy:false) no longer mutates; same-dtype returns self
{
    var a = np.ones(new Shape(3, 3), np.int32);
    var b = a.astype(np.int64, false);
    Check("astype conv returns new", !ReferenceEquals(b, a) && b.typecode == NPTypeCode.Int64);
    Check("astype input untouched", a.typecode == NPTypeCode.Int32);
    var c = a.astype(np.int32, false);
    Check("astype same-dtype returns self", ReferenceEquals(c, a));
}

// np.allclose / np.where no longer corrupt operand dtypes
{
    var x = np.array(new float[] { 1f, 2f, 3f });
    var y = np.array(new float[] { 1f, 2f, 3.0000001f });
    bool close = np.allclose(x, y);
    Check("allclose true", close);
    Check("allclose keeps x float32", x.typecode == NPTypeCode.Single);
    Check("allclose keeps y float32", y.typecode == NPTypeCode.Single);
    var cond = np.array(new int[] { 1, 0, 2 });
    var _ = np.where(cond, np.array(new double[] { 1, 1, 1 }), np.array(new double[] { 2, 2, 2 }));
    Check("where keeps cond int32", cond.typecode == NPTypeCode.Int32);
}

// activation resolver: softmax registered, unknown throws, ""/linear → null
{
    Check("Get softmax", BaseActivation.Get("softmax") is Softmax);
    Check("Get RELU case-insensitive", BaseActivation.Get("ReLU") is ReLU);
    Check("Get '' null", BaseActivation.Get("") == null);
    Check("Get linear null", BaseActivation.Get("linear") == null);
    bool threw = false;
    try { BaseActivation.Get("rleu"); } catch (ArgumentException) { threw = true; }
    Check("Get unknown throws", threw);
}

// fused layer string ctor
{
    var f = new FullyConnectedFused(4, 3, "relu");
    Check("fused string relu", f.Activation == FusedActivation.ReLU);
    var g = new FullyConnectedFused(4, 3, "");
    Check("fused string none", g.Activation == FusedActivation.None);
    bool threw = false;
    try { new FullyConnectedFused(4, 3, "sigmoid"); } catch (ArgumentException) { threw = true; }
    Check("fused string unsupported throws", threw);
}

// SGD decay: lr_t from base rate, no compounding across calls
{
    var layer = new ParamStub(1f);
    var sgd = new SGD(lr: 0.1f, momentum: 0f, decayRate: 0.5f);
    layer.Grads["w"] = np.array(new float[] { 1f });
    sgd.Update(2, layer);   // lr = 0.1/(1+0.5*2) = 0.05 → w = 0.95
    CheckClose("sgd decay first", layer.Parameters["w"].GetSingle(0), 0.95f);
    layer.Grads["w"] = np.array(new float[] { 1f });
    sgd.Update(2, layer);   // SAME iteration → same lr → w = 0.90 (old code compounded)
    CheckClose("sgd decay no compounding", layer.Parameters["w"].GetSingle(0), 0.90f);
    CheckClose("sgd base lr unchanged", sgd.LearningRate, 0.1f);
}

// Adam decay: base LR never mutated
{
    var layer = new ParamStub(1f);
    var adam = new Adam(lr: 0.1f, decayRate: 0.5f);
    layer.Grads["w"] = np.array(new float[] { 1f });
    adam.Update(1, layer);
    CheckClose("adam base lr unchanged", adam.LearningRate, 0.1f);
}

// Evaluate scores the partial final batch
{
    var echo = new EchoLayer();
    // 10 samples, 3 classes; row argmax == label for all but the last row.
    var xs = np.zeros(new Shape(10, 3), NPTypeCode.Single);
    var lbl = new byte[10];
    for (int i = 0; i < 10; i++)
    {
        int cls = i % 3;
        lbl[i] = (byte)cls;
        xs[$"{i}:{i + 1}"][$":, {cls}:{cls + 1}"] = (NDArray)1f;
    }
    lbl[9] = (byte)((9 % 3 + 1) % 3);  // make the LAST sample (partial batch) wrong
    var labels = np.array(lbl);
    float acc = MlpTrainer.Evaluate(new List<BaseLayer> { echo }, xs, labels, batchSize: 4);
    CheckClose("Evaluate partial batch 9/10", acc, 0.9f);
}

// ============================ P2: activations =============================
Console.WriteLine("--- P2 activations (values vs NumPy + finite-difference gradients) ---");

void ActCase(string name, Func<BaseActivation> make, float[] fwd, float[] der, int kinkIndex = -1)
{
    var act = make();
    act.Forward(X());
    CheckArr($"{name} forward", act.Output, fwd, 1e-4f);
    act.Backward(np.ones(new Shape(7), NPTypeCode.Single));
    CheckArr($"{name} backward", act.InputGrad, der, 1e-3f);

    // finite differences, h=1e-2 (float32 forward, central diff)
    const float h = 1e-2f;
    var xp = X() + h; var xm = X() - h;
    var a1 = make(); a1.Forward(xp);
    var a2 = make(); a2.Forward(xm);
    for (int i = 0; i < 7; i++)
    {
        // central differences straddle the derivative kink at x=0 for
        // leaky_relu/selu and report the average of the one-sided slopes —
        // the analytic value there is checked against NumPy above instead.
        if (i == kinkIndex) continue;
        float fd = (a1.Output.GetSingle(i) - a2.Output.GetSingle(i)) / (2 * h);
        float an = act.InputGrad.GetSingle(i);
        if (Math.Abs(fd - an) > 2e-2f * Math.Max(1f, Math.Abs(fd)))
        { Check($"{name} FD[{i}]", false, $"fd={fd} analytic={an}"); return; }
    }
    Check($"{name} FD", true);
}

ActCase("tanh", () => new Tanh(),
    new[] { -0.99505475f, -0.90514825f, -0.46211716f, 0f, 0.46211716f, 0.90514825f, 0.99505475f },
    new[] { 0.0098660372f, 0.18070664f, 0.78644773f, 1f, 0.78644773f, 0.18070664f, 0.0098660372f });
ActCase("leaky_relu", () => new LeakyReLU(),
    new[] { -0.9f, -0.45f, -0.15f, 0f, 0.5f, 1.5f, 3f },
    new[] { 0.3f, 0.3f, 0.3f, 0.3f, 1f, 1f, 1f }, kinkIndex: 3);
ActCase("elu", () => new ELU(),
    new[] { -0.95021293f, -0.77686984f, -0.39346934f, 0f, 0.5f, 1.5f, 3f },
    new[] { 0.049787068f, 0.22313016f, 0.60653066f, 1f, 1f, 1f, 1f });
ActCase("gelu", () => new GELU(),
    new[] { -0.0036373921f, -0.10042842f, -0.15428599f, 0f, 0.34571401f, 1.3995716f, 2.9963626f },
    new[] { -0.011584167f, -0.12771079f, 0.1326301f, 0.5f, 0.8673699f, 1.1277108f, 1.0115842f });
ActCase("silu", () => new SiLU(),
    new[] { -0.14227762f, -0.27363829f, -0.18877033f, 0f, 0.31122967f, 1.2263617f, 2.8577224f },
    new[] { -0.088104106f, -0.041294154f, 0.26003881f, 0.5f, 0.73996119f, 1.0412942f, 1.0881041f });
ActCase("softplus", () => new Softplus(),
    new[] { 0.048587352f, 0.20141328f, 0.47407698f, 0.69314718f, 0.97407698f, 1.7014133f, 3.0485874f },
    new[] { 0.047425873f, 0.18242552f, 0.37754067f, 0.5f, 0.62245933f, 0.81757448f, 0.95257413f });
ActCase("selu", () => new SELU(),
    new[] { -1.6705687f, -1.3658144f, -0.69175819f, 0f, 0.52535049f, 1.5760515f, 3.152103f },
    new[] { 0.087530612f, 0.39228499f, 1.0663412f, 1.7580993f, 1.050701f, 1.050701f, 1.050701f }, kinkIndex: 3);

// leaky FD at x=0 has the kink — exclude by checking resolver-level names instead
Check("resolver tanh..selu", BaseActivation.Get("tanh") is Tanh && BaseActivation.Get("leaky_relu") is LeakyReLU
    && BaseActivation.Get("elu") is ELU && BaseActivation.Get("gelu") is GELU
    && BaseActivation.Get("swish") is SiLU && BaseActivation.Get("softplus") is Softplus
    && BaseActivation.Get("selu") is SELU);

// ============================ P2: losses =============================
Console.WriteLine("--- P2 losses (values vs NumPy + finite-difference gradients) ---");

void LossFD(string name, BaseCost cost, NDArray preds, NDArray labels)
{
    var analytic = cost.Backward(preds, labels);
    const float h = 1e-2f;
    int rows = (int)preds.shape[0], cols = (int)preds.shape[1];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            var pp = preds.copy(); pp[$"{i}:{i + 1}"][$":, {j}:{j + 1}"] = (NDArray)(preds.GetSingle(i, j) + h);
            var pm = preds.copy(); pm[$"{i}:{i + 1}"][$":, {j}:{j + 1}"] = (NDArray)(preds.GetSingle(i, j) - h);
            float fd = ((float)cost.Forward(pp, labels) - (float)cost.Forward(pm, labels)) / (2 * h);
            float an = analytic.GetSingle(i, j);
            if (Math.Abs(fd - an) > 3e-2f * Math.Max(1f, Math.Abs(fd)))
            { Check($"{name} FD[{i},{j}]", false, $"fd={fd} analytic={an}"); return; }
        }
    Check($"{name} FD", true);
}

{
    var preds = np.array(new float[,] { { 0.1f, 0.2f, 0.7f }, { 0.8f, 0.15f, 0.05f } });
    var labels = np.array(new byte[] { 2, 0 });
    var scce = new SparseCategoricalCrossentropy();
    CheckClose("scce loss", (float)scce.Forward(preds, labels), 0.28990925f, 1e-5f);
    CheckArr("scce grad", scce.Backward(preds, labels), new[] { 0f, 0f, -0.71428571f, -0.625f, 0f, 0f }, 1e-4f);
    LossFD("scce", scce, preds, labels);
}
{
    var preds = np.array(new float[,] { { 0.5f, 2.0f }, { -1.5f, 0.1f } });
    var labels = np.array(new float[,] { { 0f, 0f }, { 0.5f, 0.1f } });
    var huber = new Huber(1.0f);
    CheckClose("huber loss", (float)huber.Forward(preds, labels), 0.78125f);
    CheckArr("huber grad", huber.Backward(preds, labels), new[] { 0.125f, 0.25f, -0.25f, 0f }, 1e-5f);
    LossFD("huber", huber, preds, labels);
}
{
    var labels = np.array(new float[,] { { 0.6f, 0.3f, 0.1f }, { 0.0f, 0.5f, 0.5f } });
    var preds = np.array(new float[,] { { 0.5f, 0.4f, 0.1f }, { 0.1f, 0.6f, 0.3f } });
    var kl = new KLDivergence();
    CheckClose("kl loss", (float)kl.Forward(preds, labels), 0.093669482f, 1e-4f);
    CheckArr("kl grad", kl.Backward(preds, labels), new[] { -0.6f, -0.375f, -0.5f, -5e-07f, -0.41666667f, -0.83333333f }, 1e-3f);
    LossFD("kl", kl, preds, labels);
}
{
    var preds = np.array(new float[,] { { 0.8f, -0.4f }, { 0.3f, 1.2f } });
    var labels = np.array(new float[,] { { 1f, 0f }, { 0f, 1f } });
    var hinge = new Hinge();
    CheckClose("hinge loss {0,1}", (float)hinge.Forward(preds, labels), 0.525f);
    CheckArr("hinge grad", hinge.Backward(preds, labels), new[] { -0.25f, 0.25f, 0.25f, 0f }, 1e-5f);
    var pm = np.array(new float[,] { { 1f, -1f }, { -1f, 1f } });
    CheckClose("hinge loss {-1,1}", (float)hinge.Forward(preds, pm), 0.525f);
    LossFD("hinge", hinge, preds, labels);
}
{
    var preds = np.array(new float[,] { { 0.5f, 3.0f }, { -2.0f, 0.0f } });
    var labels = np.array(new float[,] { { 0f, 0f }, { 0.5f, 0f } });
    var logcosh = new LogCosh();
    CheckClose("logcosh loss", (float)logcosh.Forward(preds, labels), 1.0607528f);
    CheckArr("logcosh grad", logcosh.Backward(preds, labels), new[] { 0.11552929f, 0.24876369f, -0.24665357f, 0f }, 1e-4f);
    LossFD("logcosh", logcosh, preds, labels);
}

// ============================ P2: metrics =============================
Console.WriteLine("--- P2 metrics ---");
{
    var bp = np.array(new float[] { 0.9f, 0.6f, 0.4f, 0.2f, 0.7f, 0.51f });
    var bl = np.array(new float[] { 1f, 0f, 1f, 0f, 1f, 1f });
    CheckClose("precision", (float)new Precision().Calculate(bp, bl), 0.75f);
    CheckClose("recall", (float)new Recall().Calculate(bp, bl), 0.75f);
    CheckClose("f1 binary", (float)new F1Score().Calculate(bp, bl), 0.75f);
}
{
    var mp = np.array(new float[,] {
        { .8f, .1f, .1f }, { .1f, .8f, .1f }, { .1f, .1f, .8f },
        { .8f, .1f, .1f }, { .1f, .8f, .1f }, { .8f, .1f, .1f } });
    var idx = new[] { 0, 1, 2, 0, 2, 1 };
    var ml = np.zeros(new Shape(6, 3), NPTypeCode.Single);
    for (int i = 0; i < 6; i++) ml[$"{i}:{i + 1}"][$":, {idx[i]}:{idx[i] + 1}"] = (NDArray)1f;
    CheckClose("f1 macro", (float)new F1Score(F1Average.Macro).Calculate(mp, ml), 0.65555556f, 1e-5f);
}
{
    var tkp = np.array(new float[,] {
        { .4f, .3f, .2f, .1f }, { .1f, .2f, .3f, .4f }, { .25f, .25f, .25f, .25f }, { .5f, .2f, .2f, .1f } });
    var idx = new[] { 1, 0, 3, 0 };
    var tkl = np.zeros(new Shape(4, 4), NPTypeCode.Single);
    for (int i = 0; i < 4; i++) tkl[$"{i}:{i + 1}"][$":, {idx[i]}:{idx[i] + 1}"] = (NDArray)1f;
    CheckClose("top-2 accuracy", (float)new TopKCategoricalAccuracy(2).Calculate(tkp, tkl), 0.75f);
}
{
    var rp = np.array(new float[] { 2.5f, 0.0f, 2.0f, 8.0f });
    var rl = np.array(new float[] { 3.0f, -0.5f, 2.0f, 7.0f });
    CheckClose("rmse", (float)new RootMeanSquaredError().Calculate(rp, rl), 0.61237244f);
    CheckClose("r2", (float)new R2Score().Calculate(rp, rl), 0.94860814f);
    CheckClose("r2 perfect", (float)new R2Score().Calculate(rl, rl), 1f);
}
{
    // typo-shim compat
    #pragma warning disable CS0618
    Check("Accuacy shim", new Accuacy() is Accuracy && new Accuacy().Name == "accuracy");
    Check("BinaryAccuacy shim", new BinaryAccuacy() is BinaryAccuracy);
    #pragma warning restore CS0618
}

// ============================ P2: initializers =============================
Console.WriteLine("--- P2 initializers ---");
{
    np.random.seed(7);
    var w1 = new HeNormal().Initialize(new Shape(256, 128));
    np.random.seed(7);
    var w2 = new HeNormal().Initialize(new Shape(256, 128));
    Check("seeded determinism", np.array_equal(w1, w2));
    Check("float32 output", w1.typecode == NPTypeCode.Single);

    // stats: resulting std ≈ sqrt(2/256); truncation bound at 2σ' = 2·target/0.8796
    double target = Math.Sqrt(2.0 / 256);
    float mean = (float)np.mean(w1);
    float std = (float)np.std(w1);
    float maxAbs = (float)np.max(np.abs(w1));
    Check("he_normal mean≈0", Math.Abs(mean) < 0.01 * target * 10, $"mean={mean}");
    Check("he_normal std", Math.Abs(std - target) < 0.03 * target, $"std={std} target={target}");
    Check("he_normal truncated", maxAbs <= 2.0 * target / 0.87962566103423978 + 1e-6, $"max={maxAbs}");

    var gu = new GlorotUniform().Initialize(new Shape(256, 128));
    double limit = Math.Sqrt(6.0 / (256 + 128));
    Check("glorot_uniform bound", (float)np.max(np.abs(gu)) <= limit + 1e-6);
    double ustd = Math.Sqrt(limit * limit / 3);
    Check("glorot_uniform std", Math.Abs((float)np.std(gu) - ustd) < 0.03 * ustd);

    var z = new Zeros().Initialize(new Shape(4, 4));
    Check("zeros", (float)np.sum(np.abs(z)) == 0f);
    var on = new Ones().Initialize(new Shape(4, 4));
    Check("ones", (float)np.sum(on) == 16f);
    var ct = new Constant(0.5f).Initialize(new Shape(4));
    Check("constant", (float)np.sum(ct) == 2f);
}
{
    // orthogonality: tall (64, 32) → QᵀQ = I; wide (16, 64) → QQᵀ = I
    np.random.seed(11);
    var q = new Orthogonal().Initialize(new Shape(64, 32));
    var qtq = np.dot(q.transpose().copy(), q);
    float dev = MaxDevFromIdentity(qtq, 32);
    Check("orthogonal tall QtQ=I", dev < 1e-4f, $"dev={dev}");

    var qw = new Orthogonal().Initialize(new Shape(16, 64));
    var qqt = np.dot(qw, qw.transpose().copy());
    float dev2 = MaxDevFromIdentity(qqt, 16);
    Check("orthogonal wide QQt=I", dev2 < 1e-4f, $"dev={dev2}");

    var g2 = new Orthogonal(gain: 2f).Initialize(new Shape(32, 32));
    var gg = np.dot(g2.transpose().copy(), g2);
    float diag = gg.GetSingle(0, 0);
    Check("orthogonal gain", Math.Abs(diag - 4f) < 1e-3f, $"diag={diag}");
}
{
    Check("init resolver", BaseInitializer.Get("he_normal") is HeNormal
        && BaseInitializer.Get("xavier_uniform") is GlorotUniform
        && BaseInitializer.Get("orthogonal") is Orthogonal
        && BaseInitializer.Get("") == null);
    bool threw = false;
    try { BaseInitializer.Get("bogus"); } catch (ArgumentException) { threw = true; }
    Check("init resolver unknown throws", threw);
}
{
    // layers actually consume the provided initializers
    var fc = new FullyConnected(8, 4, "relu", kernelInitializer: new Zeros(), biasInitializer: new Constant(0.5f));
    Check("fc kernel init used", (float)np.sum(np.abs(fc.Parameters["w"])) == 0f);
    Check("fc bias init used", Math.Abs((float)np.sum(fc.Parameters["b"]) - 2f) < 1e-6f);
    var ff = new FullyConnectedFused(8, 4, FusedActivation.ReLU, kernelInitializer: new Ones());
    Check("fused kernel init used", (float)np.sum(ff.Parameters["w"]) == 32f);
}

Console.WriteLine();
Console.WriteLine($"RESULT: {pass} passed, {fail} failed");
return fail == 0 ? 0 : 1;

static float MaxDevFromIdentity(NDArray m, int n)
{
    float dev = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            float expect = i == j ? 1f : 0f;
            dev = Math.Max(dev, Math.Abs(m.GetSingle(i, j) - expect));
        }
    return dev;
}

// stub layer with one scalar param "w"
class ParamStub : BaseLayer
{
    public ParamStub(float w) : base("stub")
    {
        Parameters["w"] = np.array(new float[] { w });
    }
}

// layer that echoes its input as its output (predictions == input rows)
class EchoLayer : BaseLayer
{
    public EchoLayer() : base("echo") { }
    public override void Forward(NDArray x)
    {
        base.Forward(x);
        Output = x;
    }
}
