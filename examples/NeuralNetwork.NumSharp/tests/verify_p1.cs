#:project ../NeuralNetwork.NumSharp.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
// P1 verification for NeuralNetwork.NumSharp — training-loop parity.
// Run from THIS directory (file-based apps need a csproj-free CWD):
//   cd examples/NeuralNetwork.NumSharp/tests && dotnet run verify_p1.cs
//
// Sections: weight/architecture serialization, gradient clipping, the four
// callbacks (driven directly against Keras's documented state machine), and
// the trainer itself (shuffle determinism, validation split, partial final
// batch, verbosity, early-stop integration).
//
// Callback semantics are pinned by driving OnTrainBegin/OnEpochEnd with
// synthetic logs rather than by training a model — the point is the state
// machine (wait/best/cooldown/min_delta signs), and a synthetic metric
// sequence exercises branches a real loss curve would reach only by luck.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetwork.NumSharp;
using NeuralNetwork.NumSharp.Callbacks;
using NeuralNetwork.NumSharp.Cost;
using NeuralNetwork.NumSharp.Layers;
using NeuralNetwork.NumSharp.MnistMlp;
using NeuralNetwork.NumSharp.Optimizers;
using NeuralNetwork.NumSharp.Serialization;
using NumSharp;
using NumSharp.Backends;

int pass = 0, fail = 0;
void Check(string name, bool ok, string detail = "")
{
    if (ok) pass++;
    else { fail++; Console.WriteLine($"  FAIL {name} {detail}"); }
}
void CheckClose(string name, float actual, float expected, float tol = 1e-5f)
    => Check(name, Math.Abs(actual - expected) <= tol * Math.Max(1f, Math.Abs(expected)),
             $"actual={actual} expected={expected}");
void CheckEq(string name, long actual, long expected)
    => Check(name, actual == expected, $"actual={actual} expected={expected}");
void CheckStr(string name, string actual, string expected)
    => Check(name, actual == expected, $"actual='{actual}' expected='{expected}'");

string tmpRoot = Path.Combine(Path.GetTempPath(), "nn_p1_" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(tmpRoot);

try
{

// ======================================================================
Console.WriteLine("--- P1 serialization (weights .npz + architecture JSON) ---");
// ======================================================================
{
    np.random.seed(11);
    var model = new List<BaseLayer>
    {
        new FullyConnectedFused(4, 3, FusedActivation.ReLU),
        new FullyConnectedFused(3, 2, FusedActivation.None),
    };

    // A non-trainable buffer must ride along in the archive but never reach an
    // optimizer — this is the slot BatchNorm's running stats use in P4.
    model[0].NonTrainable["running_mean"] = np.array(new float[] { 0.25f, -1.5f, 7f });

    string ckpt = Path.Combine(tmpRoot, "weights.npz");
    ModelWeights.Save(model, ckpt);
    Check("npz written", File.Exists(ckpt));

    // Keys are positional and namespaced by bucket — a real NumPy archive.
    using (var npz = np.load_npz(ckpt))
    {
        var files = npz.Files.ToArray();
        Check("key layer0/param/w", files.Contains("layer0/param/w"), string.Join(",", files));
        Check("key layer1/param/b", files.Contains("layer1/param/b"), string.Join(",", files));
        Check("key layer0/state/running_mean", files.Contains("layer0/state/running_mean"), string.Join(",", files));
        CheckEq("archive entry count", files.Length, 5);   // 2 layers x (w,b) + 1 state
        var w0 = npz["layer0/param/w"];
        CheckStr("archived dtype preserved", w0.dtype.Name, "Single");
    }

    // Round-trip into a DIFFERENTLY-initialized model of the same shape.
    np.random.seed(999);
    var rebuilt = new List<BaseLayer>
    {
        new FullyConnectedFused(4, 3, FusedActivation.ReLU),
        new FullyConnectedFused(3, 2, FusedActivation.None),
    };
    rebuilt[0].NonTrainable["running_mean"] = np.zeros(new Shape(3), NPTypeCode.Single);

    Check("pre-load models differ", MaxAbsDiff(model[0].Parameters["w"], rebuilt[0].Parameters["w"]) > 1e-6);
    ModelWeights.Load(rebuilt, ckpt);
    CheckClose("layer0/w restored", (float)MaxAbsDiff(model[0].Parameters["w"], rebuilt[0].Parameters["w"]), 0f);
    CheckClose("layer1/w restored", (float)MaxAbsDiff(model[1].Parameters["w"], rebuilt[1].Parameters["w"]), 0f);
    CheckClose("layer1/b restored", (float)MaxAbsDiff(model[1].Parameters["b"], rebuilt[1].Parameters["b"]), 0f);
    CheckClose("non-trainable state restored",
               (float)MaxAbsDiff(model[0].NonTrainable["running_mean"], rebuilt[0].NonTrainable["running_mean"]), 0f);

    // Loading is a COPY: mutating the source afterwards must not move the target.
    model[0].Parameters["w"] = model[0].Parameters["w"] + 1f;
    Check("load copied, not aliased", MaxAbsDiff(model[0].Parameters["w"], rebuilt[0].Parameters["w"]) > 0.5);

    // Architecture mismatch is a hard, named error — never a silent partial load.
    var wrongShape = new List<BaseLayer> { new FullyConnectedFused(9, 3, FusedActivation.ReLU) };
    string msg = ExpectThrow(() => ModelWeights.Load(wrongShape, ckpt));
    Check("shape mismatch reports the key", msg.Contains("layer0/param/w"), msg);
    Check("shape mismatch reports both shapes", msg.Contains("4, 3") && msg.Contains("9, 3"), msg);

    var tooManyLayers = new List<BaseLayer>
    {
        new FullyConnectedFused(4, 3, FusedActivation.ReLU),
        new FullyConnectedFused(3, 2, FusedActivation.None),
        new FullyConnectedFused(2, 2, FusedActivation.None),
    };
    msg = ExpectThrow(() => ModelWeights.Load(tooManyLayers, ckpt));
    Check("missing layer reports the key", msg.Contains("layer2/param/w"), msg);

    // A late failure must not leave the model half-overwritten.
    np.random.seed(31337);
    var partial = new List<BaseLayer>
    {
        new FullyConnectedFused(4, 3, FusedActivation.ReLU),
        new FullyConnectedFused(3, 5, FusedActivation.None),   // wrong: layer 1 is (3,2)
    };
    NDArray before = partial[0].Parameters["w"].copy();
    ExpectThrow(() => ModelWeights.Load(partial, ckpt));
    CheckClose("failed load leaves layer 0 untouched", (float)MaxAbsDiff(before, partial[0].Parameters["w"]), 0f);
}

// Capture/Restore snapshots are detached deep copies.
{
    var stub = new ParamStub(2f);
    var snap = ModelWeights.Capture(new List<BaseLayer> { stub });
    stub.Parameters["w"] = np.array(new float[] { 99f });
    CheckClose("snapshot survives a later write", snap["layer0/param/w"].GetSingle(0), 2f);
    ModelWeights.Restore(new List<BaseLayer> { stub }, snap);
    CheckClose("restore writes back", stub.Parameters["w"].GetSingle(0), 2f);
    stub.Parameters["w"] = np.array(new float[] { 7f });
    ModelWeights.Restore(new List<BaseLayer> { stub }, snap);
    CheckClose("snapshot is reusable", stub.Parameters["w"].GetSingle(0), 2f);
}

// Architecture JSON round-trip.
{
    np.random.seed(5);
    var model = new List<BaseLayer>
    {
        new FullyConnectedFused(6, 4, FusedActivation.ReLU),
        new FullyConnected(4, 3, "sigmoid"),
    };
    string json = ModelArchitecture.ToJson(model);
    Check("json names the fused class", json.Contains("FullyConnectedFused"), json);
    Check("json carries units", json.Contains("\"units\""), json);

    var rebuilt = ModelArchitecture.FromJson(json);
    CheckEq("json rebuilt layer count", rebuilt.Count, 2);
    CheckEq("json layer0 in", ((FullyConnectedFused)rebuilt[0]).InputDim, 6);
    CheckEq("json layer0 out", ((FullyConnectedFused)rebuilt[0]).OutputDim, 4);
    Check("json layer0 activation", ((FullyConnectedFused)rebuilt[0]).Activation == FusedActivation.ReLU);
    CheckEq("json layer1 in", ((FullyConnected)rebuilt[1]).InputDim, 4);
    CheckStr("json layer1 activation", ((FullyConnected)rebuilt[1]).ActivationName, "sigmoid");

    // Rebuilt-from-JSON slots line up with a checkpoint of the original.
    string ck = Path.Combine(tmpRoot, "arch.npz");
    ModelWeights.Save(model, ck);
    ModelWeights.Load(rebuilt, ck);
    CheckClose("json+npz full round-trip", (float)MaxAbsDiff(model[0].Parameters["w"], rebuilt[0].Parameters["w"]), 0f);

    // A layer with no registered factory is refused at serialize time, loudly.
    string m = ExpectThrow(() => ModelArchitecture.ToJson(new List<BaseLayer> { new ParamStub(1f) }));
    Check("unregistered layer refused", m.Contains("no registered factory"), m);
}

// ======================================================================
Console.WriteLine("--- P1 gradient clipping (Keras clipnorm / global_clipnorm / clipvalue) ---");
// ======================================================================
{
    // clip_by_norm: g=[3,4] has L2 norm 5. With clipnorm 2.5 the scale is 0.5.
    // SGD at lr=1 from a zero parameter makes the update readable directly.
    var layer = new VecStub(new float[] { 0f, 0f });
    layer.Grads["w"] = np.array(new float[] { 3f, 4f });
    var sgd = new SGD(lr: 1f) { ClipNorm = 2.5f };
    sgd.Update(1, layer);
    CheckClose("clipnorm scales [3,4]->[1.5,2] (0)", layer.Parameters["w"].GetSingle(0), -1.5f);
    CheckClose("clipnorm scales [3,4]->[1.5,2] (1)", layer.Parameters["w"].GetSingle(1), -2f);

    // Under the threshold Keras's v*c/max(norm,c) is v*c/c — the identity.
    var layer2 = new VecStub(new float[] { 0f, 0f });
    layer2.Grads["w"] = np.array(new float[] { 3f, 4f });
    var sgd2 = new SGD(lr: 1f) { ClipNorm = 10f };
    sgd2.Update(1, layer2);
    CheckClose("clipnorm under threshold is exact (0)", layer2.Parameters["w"].GetSingle(0), -3f, 0f);
    CheckClose("clipnorm under threshold is exact (1)", layer2.Parameters["w"].GetSingle(1), -4f, 0f);

    // clipvalue clamps ELEMENTS, leaving in-range ones alone.
    var layer3 = new VecStub(new float[] { 0f, 0f, 0f });
    layer3.Grads["w"] = np.array(new float[] { 3f, -4f, 0.5f });
    var sgd3 = new SGD(lr: 1f) { ClipValue = 1f };
    sgd3.Update(1, layer3);
    CheckClose("clipvalue high", layer3.Parameters["w"].GetSingle(0), -1f);
    CheckClose("clipvalue low", layer3.Parameters["w"].GetSingle(1), 1f);
    CheckClose("clipvalue passthrough", layer3.Parameters["w"].GetSingle(2), -0.5f);

    // global_clipnorm uses ONE norm over the whole model: [3] and [4] across two
    // layers is norm 5, so both scale by 0.5 — per-parameter clipping would
    // instead scale them by 2.5/3 and 2.5/4 respectively.
    var a = new VecStub(new float[] { 0f });
    var b = new VecStub(new float[] { 0f });
    a.Grads["w"] = np.array(new float[] { 3f });
    b.Grads["w"] = np.array(new float[] { 4f });
    var model = new List<BaseLayer> { a, b };
    var sgd4 = new SGD(lr: 1f) { GlobalClipNorm = 2.5f };
    sgd4.ApplyGlobalClipNorm(model);
    foreach (var l in model) sgd4.Update(1, l);
    CheckClose("global_clipnorm layer a", a.Parameters["w"].GetSingle(0), -1.5f);
    CheckClose("global_clipnorm layer b", b.Parameters["w"].GetSingle(0), -2f);

    // Under budget the global scale is exactly 1 (Keras's min(1/n, 1/c) form).
    var c = new VecStub(new float[] { 0f });
    c.Grads["w"] = np.array(new float[] { 3f });
    var sgd5 = new SGD(lr: 1f) { GlobalClipNorm = 100f };
    sgd5.ApplyGlobalClipNorm(new List<BaseLayer> { c });
    sgd5.Update(1, c);
    CheckClose("global_clipnorm under budget is exact", c.Parameters["w"].GetSingle(0), -3f, 0f);

    // Keras rejects the pair; so do we, in either assignment order.
    string m1 = ExpectThrow(() => { var o = new SGD(); o.ClipNorm = 1f; o.GlobalClipNorm = 1f; });
    Check("clipnorm+global rejected (A)", m1.Contains("Only one of"), m1);
    string m2 = ExpectThrow(() => { var o = new SGD(); o.GlobalClipNorm = 1f; o.ClipNorm = 1f; });
    Check("clipnorm+global rejected (B)", m2.Contains("Only one of"), m2);

    // Adam honours the same clip (the gradient is clipped BEFORE the moments).
    var la = new VecStub(new float[] { 0f, 0f });
    la.Grads["w"] = np.array(new float[] { 3f, 4f });
    var adam = new Adam(lr: 1f) { ClipValue = 1f };
    adam.Update(1, la);
    // With m=v=g after bias correction, the step is -lr*g/(sqrt(g^2)+eps) ~ -sign(g).
    CheckClose("adam clipvalue applied", Math.Sign(la.Parameters["w"].GetSingle(0)), -1f);
    Check("adam clipped magnitude ~1", Math.Abs(la.Parameters["w"].GetSingle(0)) <= 1.001f,
          la.Parameters["w"].GetSingle(0).ToString());
}

// ======================================================================
Console.WriteLine("--- P1 EarlyStopping (Keras state machine) ---");
// ======================================================================
{
    // patience=2 on a min metric: improve, improve, then two flat epochs.
    var es = Drive(new EarlyStopping("val_loss", patience: 2),
                   new[] { 1.0f, 0.9f, 0.95f, 0.96f, 0.97f }, "val_loss");
    CheckEq("ES patience=2 stops at epoch 3", es.cb.StoppedEpoch, 3);
    CheckEq("ES best epoch", es.cb.BestEpoch, 1);
    CheckClose("ES best value", es.cb.Best, 0.9f);
    CheckEq("ES epochs consumed", es.epochsSeen, 4);

    // patience=0 must still survive epoch 0 — the `epoch > 0` guard.
    var es0 = Drive(new EarlyStopping("val_loss", patience: 0), new[] { 1.0f, 2.0f, 3.0f }, "val_loss");
    CheckEq("ES patience=0 cannot stop at epoch 0", es0.cb.StoppedEpoch, 1);

    // min_delta is NEGATED in min mode, so the test is `current + min_delta < best`.
    // Driven epoch by epoch: with min_delta=0.1 the 0.05 and 0.08 gains are
    // rejected and `best` stays at 1.0; the 0.15 gain finally clears the bar.
    // The same sequence with min_delta=0 improves on every single epoch — the
    // discriminating comparison, and the one an unsigned min_delta would break.
    {
        var withDelta = new EarlyStopping("val_loss", minDelta: 0.1f, patience: 5);
        var noDelta = new EarlyStopping("val_loss", patience: 5);
        var layers = new List<BaseLayer> { new ParamStub(1f) };
        foreach (var cb in new[] { withDelta, noDelta })
        {
            cb.SetContext(new TrainingContext(layers, new SGD(), 4, 1, 1, true));
            cb.OnTrainBegin();
        }

        float[] seq = { 1.0f, 0.95f, 0.92f, 0.85f };
        for (int e = 0; e < seq.Length; e++)
        {
            var logs = new Dictionary<string, float> { ["val_loss"] = seq[e] };
            withDelta.OnEpochEnd(e, logs);
            noDelta.OnEpochEnd(e, new Dictionary<string, float>(logs));

            if (e == 2)
            {
                CheckClose("ES min_delta holds best at 1.0 after sub-delta gains", withDelta.Best, 1.0f);
                CheckClose("ES min_delta=0 tracks every gain", noDelta.Best, 0.92f);
                CheckEq("ES min_delta best epoch still 0", withDelta.BestEpoch, 0);
            }
        }
        CheckClose("ES min_delta accepts the 0.15 gain", withDelta.Best, 0.85f);
        CheckEq("ES min_delta best epoch", withDelta.BestEpoch, 3);
        CheckEq("ES min_delta never stopped", withDelta.StoppedEpoch, -1);
    }

    // "val_acc" auto-resolves to MAXIMIZE.
    var esa = Drive(new EarlyStopping("val_acc", patience: 1), new[] { 0.5f, 0.6f, 0.55f }, "val_acc");
    CheckEq("ES auto-max on val_acc stops at 2", esa.cb.StoppedEpoch, 2);
    CheckClose("ES auto-max best", esa.cb.Best, 0.6f);

    // An explicit mode overrides the name-based guess: a RISING accuracy now
    // reads as "not improving" and trips the patience on the very next epoch.
    var esm = Drive(new EarlyStopping("val_acc", patience: 1, mode: "min"), new[] { 0.5f, 0.6f, 0.7f }, "val_acc");
    CheckEq("ES explicit mode=min inverts", esm.cb.StoppedEpoch, 1);

    // A missing monitor is a no-op, not a crash (Keras warns and skips).
    var esn = Drive(new EarlyStopping("nope", patience: 0), new[] { 1f, 2f, 3f }, "val_loss");
    CheckEq("ES missing monitor never stops", esn.cb.StoppedEpoch, -1);

    // start_from_epoch delays the monitor entirely.
    var ess = Drive(new EarlyStopping("val_loss", patience: 0, startFromEpoch: 2),
                    new[] { 1f, 2f, 3f, 4f, 5f }, "val_loss");
    CheckEq("ES start_from_epoch=2 stops at 3", ess.cb.StoppedEpoch, 3);

    // restore_best_weights rolls the model back to the best epoch.
    {
        var stub = new ParamStub(1f);
        var layers = new List<BaseLayer> { stub };
        var cb = new EarlyStopping("val_loss", patience: 1, restoreBestWeights: true);
        cb.SetContext(new TrainingContext(layers, new SGD(), 5, 1, 1, true));
        cb.OnTrainBegin();

        stub.Parameters["w"] = np.array(new float[] { 10f });
        cb.OnEpochEnd(0, new Dictionary<string, float> { ["val_loss"] = 0.5f });   // best
        stub.Parameters["w"] = np.array(new float[] { 20f });
        cb.OnEpochEnd(1, new Dictionary<string, float> { ["val_loss"] = 0.9f });   // worse -> stop
        Check("ES restore triggered stop", cb.StoppedEpoch == 1);
        CheckClose("ES restored the best weights", stub.Parameters["w"].GetSingle(0), 10f);
    }

    // Without restore_best_weights the final (worse) weights are kept.
    {
        var stub = new ParamStub(1f);
        var layers = new List<BaseLayer> { stub };
        var cb = new EarlyStopping("val_loss", patience: 1, restoreBestWeights: false);
        cb.SetContext(new TrainingContext(layers, new SGD(), 5, 1, 1, true));
        cb.OnTrainBegin();
        stub.Parameters["w"] = np.array(new float[] { 10f });
        cb.OnEpochEnd(0, new Dictionary<string, float> { ["val_loss"] = 0.5f });
        stub.Parameters["w"] = np.array(new float[] { 20f });
        cb.OnEpochEnd(1, new Dictionary<string, float> { ["val_loss"] = 0.9f });
        CheckClose("ES without restore keeps final weights", stub.Parameters["w"].GetSingle(0), 20f);
    }

    // baseline: an improvement that misses the baseline updates `best` but does
    // NOT reset the patience clock (Keras's documented asymmetry). So `wait`
    // keeps climbing through a run of genuine improvements, and the FIRST
    // non-improving epoch trips an already-exhausted patience. Same sequence
    // with no baseline resets `wait` each time and survives.
    var esb = Drive(new EarlyStopping("val_loss", patience: 2, baseline: 0.2f),
                    new[] { 1.0f, 0.9f, 0.8f, 0.85f }, "val_loss");
    CheckEq("ES baseline blocks the wait reset", esb.cb.StoppedEpoch, 3);
    CheckClose("ES baseline still tracked best", esb.cb.Best, 0.8f);

    var esnb = Drive(new EarlyStopping("val_loss", patience: 2),
                     new[] { 1.0f, 0.9f, 0.8f, 0.85f }, "val_loss");
    CheckEq("ES without baseline survives the same sequence", esnb.cb.StoppedEpoch, -1);
}

// ======================================================================
Console.WriteLine("--- P1 ReduceLROnPlateau ---");
// ======================================================================
{
    var opt = new SGD(lr: 1f);
    var layers = new List<BaseLayer> { new ParamStub(1f) };
    var cb = new ReduceLROnPlateau("val_loss", factor: 0.5f, patience: 1, cooldown: 2, minDelta: 0f);
    cb.SetContext(new TrainingContext(layers, opt, 10, 1, 1, true));
    cb.OnTrainBegin();

    var logs = new Dictionary<string, float>();
    void Epoch(int e, float v) { logs["val_loss"] = v; cb.OnEpochEnd(e, logs); }

    Epoch(0, 1.0f);
    CheckClose("RLROP no cut on first epoch", opt.LearningRate, 1f);
    Epoch(1, 1.0f);                       // plateau -> cut, cooldown 2
    CheckClose("RLROP first cut", opt.LearningRate, 0.5f);
    Epoch(2, 1.0f);                       // cooldown ticks 2->1, blocked
    CheckClose("RLROP cooldown blocks", opt.LearningRate, 0.5f);
    Epoch(3, 1.0f);                       // cooldown ticks 1->0, then cuts
    CheckClose("RLROP second cut after cooldown", opt.LearningRate, 0.25f);
    CheckEq("RLROP reduction count", cb.ReductionCount, 2);

    // The callback publishes the live rate into the logs for CSVLogger et al.
    Check("RLROP publishes learning_rate", logs.ContainsKey("learning_rate"));
    CheckClose("RLROP published value", logs["learning_rate"], 0.25f);

    // An improvement resets the clock.
    var opt2 = new SGD(lr: 1f);
    var cb2 = new ReduceLROnPlateau("val_loss", factor: 0.5f, patience: 2, minDelta: 0f);
    cb2.SetContext(new TrainingContext(layers, opt2, 10, 1, 1, true));
    cb2.OnTrainBegin();
    var l2 = new Dictionary<string, float>();
    foreach (var (e, v) in new[] { (0, 1.0f), (1, 1.0f), (2, 0.9f), (3, 0.9f) })
    { l2["val_loss"] = v; cb2.OnEpochEnd(e, l2); }
    CheckClose("RLROP improvement resets wait", opt2.LearningRate, 1f);

    // min_lr floors the cut and then stops further cuts entirely.
    var opt3 = new SGD(lr: 0.02f);
    var cb3 = new ReduceLROnPlateau("val_loss", factor: 0.5f, patience: 0, minLr: 0.015f, minDelta: 0f);
    cb3.SetContext(new TrainingContext(layers, opt3, 10, 1, 1, true));
    cb3.OnTrainBegin();
    var l3 = new Dictionary<string, float>();
    l3["val_loss"] = 1f; cb3.OnEpochEnd(0, l3);
    l3["val_loss"] = 1f; cb3.OnEpochEnd(1, l3);
    CheckClose("RLROP min_lr floors the cut", opt3.LearningRate, 0.015f);
    l3["val_loss"] = 1f; cb3.OnEpochEnd(2, l3);
    CheckClose("RLROP no cut once at min_lr", opt3.LearningRate, 0.015f);
    CheckEq("RLROP stops counting at min_lr", cb3.ReductionCount, 1);

    // factor must be a genuine reduction.
    string mf = ExpectThrow(() => new ReduceLROnPlateau(factor: 1f));
    Check("RLROP rejects factor>=1", mf.Contains("factor"), mf);
}

// ======================================================================
Console.WriteLine("--- P1 ModelCheckpoint ---");
// ======================================================================
{
    np.random.seed(3);
    var model = new List<BaseLayer> { new FullyConnectedFused(3, 2, FusedActivation.None) };
    string pattern = Path.Combine(tmpRoot, "ck_{epoch:D3}.npz");
    var cb = new ModelCheckpoint(pattern, "val_loss", saveBestOnly: true);
    cb.SetContext(new TrainingContext(model, new SGD(), 3, 1, 1, true));
    cb.OnTrainBegin();

    cb.OnEpochEnd(0, new Dictionary<string, float> { ["val_loss"] = 1.0f });   // save
    cb.OnEpochEnd(1, new Dictionary<string, float> { ["val_loss"] = 0.5f });   // save
    cb.OnEpochEnd(2, new Dictionary<string, float> { ["val_loss"] = 0.7f });   // skip
    CheckEq("checkpoint save_best_only count", cb.SaveCount, 2);
    Check("checkpoint epoch is 1-based in filename", File.Exists(Path.Combine(tmpRoot, "ck_001.npz")));
    Check("checkpoint second save", File.Exists(Path.Combine(tmpRoot, "ck_002.npz")));
    Check("checkpoint skipped epoch not written", !File.Exists(Path.Combine(tmpRoot, "ck_003.npz")));
    CheckClose("checkpoint best tracked", cb.Best, 0.5f);

    // The written archive really reloads.
    np.random.seed(77);
    var fresh = new List<BaseLayer> { new FullyConnectedFused(3, 2, FusedActivation.None) };
    ModelWeights.Load(fresh, cb.LastSavedPath);
    CheckClose("checkpoint reloads equal", (float)MaxAbsDiff(model[0].Parameters["w"], fresh[0].Parameters["w"]), 0f);

    // Save-every-epoch mode, and metric placeholders in the path.
    var every = new ModelCheckpoint(Path.Combine(tmpRoot, "e_{epoch}_{val_loss:F2}.npz"), saveBestOnly: false);
    every.SetContext(new TrainingContext(model, new SGD(), 2, 1, 1, true));
    every.OnTrainBegin();
    every.OnEpochEnd(0, new Dictionary<string, float> { ["val_loss"] = 1.25f });
    every.OnEpochEnd(1, new Dictionary<string, float> { ["val_loss"] = 0.5f });
    CheckEq("checkpoint save-every count", every.SaveCount, 2);
    Check("checkpoint metric placeholder", File.Exists(Path.Combine(tmpRoot, "e_1_1.25.npz")),
          string.Join(",", Directory.GetFiles(tmpRoot).Select(Path.GetFileName)));

    // A missing monitor skips instead of throwing.
    var missing = new ModelCheckpoint(Path.Combine(tmpRoot, "never.npz"), "absent", saveBestOnly: true);
    missing.SetContext(new TrainingContext(model, new SGD(), 1, 1, 1, false));
    missing.OnTrainBegin();
    missing.OnEpochEnd(0, new Dictionary<string, float> { ["loss"] = 1f });
    CheckEq("checkpoint missing monitor skips", missing.SaveCount, 0);
}

// ======================================================================
Console.WriteLine("--- P1 CSVLogger ---");
// ======================================================================
{
    string csv = Path.Combine(tmpRoot, "log.csv");
    var cb = new CSVLogger(csv);
    cb.SetContext(new TrainingContext(new List<BaseLayer>(), new SGD(), 2, 1, 1, true));
    cb.OnTrainBegin();
    cb.OnEpochEnd(0, new Dictionary<string, float> { ["loss"] = 1.5f, ["acc"] = 0.25f, ["val_loss"] = 2f });
    cb.OnEpochEnd(1, new Dictionary<string, float> { ["loss"] = 1.0f, ["acc"] = 0.50f, ["val_loss"] = 1f });
    cb.OnTrainEnd();

    var lines = File.ReadAllLines(csv);
    CheckEq("csv line count", lines.Length, 3);
    CheckStr("csv header is epoch + sorted keys", lines[0], "epoch,acc,loss,val_loss");
    CheckStr("csv row 0 (epoch is 0-based, Keras)", lines[1], "0,0.25,1.5,2");
    CheckStr("csv row 1", lines[2], "1,0.5,1,1");

    // Columns are frozen at the first epoch: a key appearing later gets no
    // column, one that vanishes is written NA.
    string csv2 = Path.Combine(tmpRoot, "log2.csv");
    var cb2 = new CSVLogger(csv2);
    cb2.SetContext(new TrainingContext(new List<BaseLayer>(), new SGD(), 2, 1, 1, false));
    cb2.OnTrainBegin();
    cb2.OnEpochEnd(0, new Dictionary<string, float> { ["loss"] = 1f });
    cb2.OnEpochEnd(1, new Dictionary<string, float> { ["acc"] = 1f });
    cb2.OnTrainEnd();
    var lines2 = File.ReadAllLines(csv2);
    CheckStr("csv frozen header", lines2[0], "epoch,loss");
    CheckStr("csv absent metric is NA", lines2[2], "1,NA");

    // append=true keeps a single header.
    var cb3 = new CSVLogger(csv, append: true);
    cb3.SetContext(new TrainingContext(new List<BaseLayer>(), new SGD(), 1, 1, 1, true));
    cb3.OnTrainBegin();
    cb3.OnEpochEnd(0, new Dictionary<string, float> { ["loss"] = 9f, ["acc"] = 9f, ["val_loss"] = 9f });
    cb3.OnTrainEnd();
    var lines3 = File.ReadAllLines(csv);
    CheckEq("csv append added one row", lines3.Length, 4);
    CheckEq("csv append wrote no second header", lines3.Count(l => l.StartsWith("epoch")), 1);
}

// ======================================================================
Console.WriteLine("--- P1 trainer (shuffle / validation / partial batch / verbose) ---");
// ======================================================================

// Data where row i carries its own index in column 0, so a spy layer can
// report exactly which samples each batch contained.
(NDArray X, NDArray Y) MakeIndexedData(int n)
{
    var xs = np.zeros(new Shape(n, 3), NPTypeCode.Single);
    var lbl = new byte[n];
    for (int i = 0; i < n; i++)
    {
        xs[$"{i}:{i + 1}"][$":, 0:1"] = (NDArray)(float)i;
        lbl[i] = (byte)(i % 3);
    }
    return (xs, np.array(lbl));
}

// validation_split takes the LAST fraction, and the final training batch is
// partial rather than dropped.
{
    var (xs, ys) = MakeIndexedData(10);
    var spy = new SpyLayer();
    MlpTrainer.Train(new List<BaseLayer> { spy }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 3, numClasses: 3,
        shuffle: false, validationSplit: 0.2f, verbose: 0);

    // 10 samples, split 0.2 -> 8 train (rows 0..7) in batches of 3,3,2, then
    // the 2 validation rows (8,9).
    CheckEq("split: batch count", spy.Batches.Count, 4);
    CheckStr("split: train batch 0", Join(spy.Batches[0]), "0,1,2");
    CheckStr("split: train batch 1", Join(spy.Batches[1]), "3,4,5");
    CheckStr("split: partial final train batch", Join(spy.Batches[2]), "6,7");
    CheckStr("split: validation is the TAIL", Join(spy.Batches[3]), "8,9");
}

// No split: every sample is trained on, partial final batch included.
{
    var (xs, ys) = MakeIndexedData(10);
    var spy = new SpyLayer();
    MlpTrainer.Train(new List<BaseLayer> { spy }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    CheckEq("no split: batch count is ceil(10/4)", spy.Batches.Count, 3);
    CheckStr("no split: partial final batch", Join(spy.Batches[2]), "8,9");
    CheckEq("no split: every sample seen once", spy.Batches.SelectMany(b => b).Distinct().Count(), 10);
}

// validation_data overrides the split and is used verbatim.
{
    var (xs, ys) = MakeIndexedData(6);
    var (vx, vy) = MakeIndexedData(2);
    var spy = new SpyLayer();
    var r = MlpTrainer.Train(new List<BaseLayer> { spy }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 3, numClasses: 3, shuffle: false,
        validationSplit: 0.5f,
        validationData: new MlpTrainer.ValidationSet(vx, vy), verbose: 0);
    CheckEq("validation_data: all 6 rows still train", spy.Batches.Take(2).SelectMany(b => b).Count(), 6);
    CheckEq("validation_data reported", r.EpochValLoss.Count, 1);
}

// Shuffling: a permutation (each sample exactly once), reproducible under a
// seed, and genuinely different under another.
{
    var (xs, ys) = MakeIndexedData(9);

    List<int> Order(int seed)
    {
        np.random.seed(seed);
        var spy = new SpyLayer();
        MlpTrainer.Train(new List<BaseLayer> { spy }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
            xs, ys, null, null, epochs: 1, batchSize: 3, numClasses: 3, shuffle: true, verbose: 0);
        return spy.Batches.SelectMany(b => b).ToList();
    }

    var o1 = Order(4242);
    var o2 = Order(4242);
    var o3 = Order(99);
    CheckEq("shuffle covers every sample once", o1.Distinct().Count(), 9);
    CheckStr("shuffle is seed-deterministic", Join(o1), Join(o2));
    Check("shuffle differs across seeds", Join(o1) != Join(o3), Join(o1));
    Check("shuffle actually reorders", Join(o1) != "0,1,2,3,4,5,6,7,8", Join(o1));
}

// Epoch loss/accuracy are SAMPLE-weighted, so a short final batch counts
// proportionally instead of as a whole batch.
{
    // 5 rows, one-hot inputs; rows 0..3 correct, row 4 wrong. batch=4 -> the
    // partial batch of 1 holds the single error, so acc must be 4/5, not the
    // batch-mean of (4/4, 0/1) = 0.5.
    var xs = np.zeros(new Shape(5, 3), NPTypeCode.Single);
    var lbl = new byte[5];
    for (int i = 0; i < 5; i++)
    {
        int cls = i % 3;
        lbl[i] = (byte)cls;
        xs[$"{i}:{i + 1}"][$":, {cls}:{cls + 1}"] = (NDArray)1f;
    }
    lbl[4] = (byte)((4 % 3 + 1) % 3);
    var r = MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, np.array(lbl), null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    CheckClose("epoch acc is sample-weighted", r.EpochTrainAcc[0], 0.8f);
}

// EarlyStopping through the real loop halts it and reports honestly.
{
    var (xs, ys) = MakeIndexedData(12);
    var es = new EarlyStopping("loss", patience: 0);
    var r = MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 20, batchSize: 4, numClasses: 3, shuffle: false,
        callbacks: new List<BaseCallback> { es }, verbose: 0);
    // lr=0 => the loss never moves => the first non-improving epoch stops it.
    CheckEq("early stop halted the loop", r.EpochsRun, 2);
    Check("early stop flagged on the result", r.StoppedEarly);
    CheckEq("early stop recorded epochs", r.EpochLoss.Count, 2);
    CheckEq("requested epochs still reported", r.Epochs, 20);
}

// Callback hooks fire the right number of times, in order.
{
    var (xs, ys) = MakeIndexedData(10);
    var rec = new RecordingCallback();
    MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 2, batchSize: 4, numClasses: 3, shuffle: false,
        callbacks: new List<BaseCallback> { rec }, verbose: 0);
    CheckEq("hook train_begin", rec.TrainBegin, 1);
    CheckEq("hook train_end", rec.TrainEnd, 1);
    CheckEq("hook epoch_begin", rec.EpochBegin, 2);
    CheckEq("hook epoch_end", rec.EpochEnd, 2);
    CheckEq("hook batch_end", rec.BatchEnd, 6);              // ceil(10/4) x 2
    CheckStr("hook order", string.Join(">", rec.Order.Take(5)), "train_begin>epoch_begin>batch>batch>batch");
    CheckEq("epoch indices are 0-based", rec.EpochIndices[0], 0);
    Check("epoch logs carry loss and acc", rec.LastLogs.ContainsKey("loss") && rec.LastLogs.ContainsKey("acc"));
    Check("epoch logs carry learning_rate", rec.LastLogs.ContainsKey("learning_rate"));
    Check("no val keys without validation", !rec.LastLogs.ContainsKey("val_loss"));
}

// Validation metrics reach the logs when a validation set exists.
{
    var (xs, ys) = MakeIndexedData(10);
    var rec = new RecordingCallback();
    MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false,
        validationSplit: 0.2f, callbacks: new List<BaseCallback> { rec }, verbose: 0);
    Check("val_loss in logs", rec.LastLogs.ContainsKey("val_loss"));
    Check("val_acc in logs", rec.LastLogs.ContainsKey("val_acc"));
    Check("context reports validation", rec.SawValidation);
}

// verbose=0 is genuinely silent.
{
    var (xs, ys) = MakeIndexedData(8);
    var sw = new StringWriter();
    var saved = Console.Out;
    Console.SetOut(sw);
    MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 2, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    Console.SetOut(saved);
    CheckEq("verbose=0 prints nothing", sw.ToString().Length, 0);
}

// verbose=2 adds a line per batch on top of the per-epoch line.
{
    var (xs, ys) = MakeIndexedData(8);
    var sw = new StringWriter();
    var saved = Console.Out;
    Console.SetOut(sw);
    MlpTrainer.Train(new List<BaseLayer> { new SpyLayer() }, new SoftmaxCrossEntropy(), new SGD(lr: 0f),
        xs, ys, null, null, epochs: 1, batchSize: 4, numClasses: 3, shuffle: false, verbose: 2);
    Console.SetOut(saved);
    string outp = sw.ToString();
    CheckEq("verbose=2 batch lines", outp.Split('\n').Count(l => l.Contains("batch ")), 2);
    Check("verbose=2 keeps the epoch line", outp.Contains("Epoch   1/1"), outp);
}

// A partial-batch training step still produces correctly-scaled gradients:
// SoftmaxCrossEntropy divides by the ACTUAL batch size, so one epoch over a
// ragged split must not blow the weights up.
{
    var (xs, ys) = MakeIndexedData(7);
    np.random.seed(21);
    var model = new List<BaseLayer> { new FullyConnectedFused(3, 3, FusedActivation.None) };
    var r = MlpTrainer.Train(model, new SoftmaxCrossEntropy(), new SGD(lr: 0.01f),
        xs, ys, null, null, epochs: 3, batchSize: 4, numClasses: 3, shuffle: false, verbose: 0);
    Check("ragged batches keep the loss finite", !float.IsNaN(r.EpochLoss[2]) && !float.IsInfinity(r.EpochLoss[2]),
          r.EpochLoss[2].ToString());
    Check("ragged batches keep weights finite", !float.IsNaN((float)np.sum(model[0].Parameters["w"])));
}

Console.WriteLine();
Console.WriteLine($"RESULT: {pass} passed, {fail} failed");
return fail == 0 ? 0 : 1;

}
finally
{
    try { Directory.Delete(tmpRoot, recursive: true); } catch { /* best effort */ }
}

// ======================================================================
// helpers
// ======================================================================

double MaxAbsDiff(NDArray a, NDArray b)
{
    var fa = np.reshape(a, new Shape((int)a.size)).astype(NPTypeCode.Single);
    var fb = np.reshape(b, new Shape((int)b.size)).astype(NPTypeCode.Single);
    double max = 0;
    for (int i = 0; i < fa.size; i++)
        max = Math.Max(max, Math.Abs(fa.GetSingle(i) - fb.GetSingle(i)));
    return max;
}

string ExpectThrow(Action a)
{
    try { a(); }
    catch (Exception ex) { return ex.Message; }
    return "<no exception thrown>";
}

static string Join(IEnumerable<int> xs) => string.Join(",", xs);

// Drives a callback through a synthetic metric sequence, stopping when it asks.
(T cb, int epochsSeen) Drive<T>(T cb, float[] values, string key) where T : BaseCallback
{
    var layers = new List<BaseLayer> { new ParamStub(1f) };
    var ctx = new TrainingContext(layers, new SGD(), values.Length, 1, 1, true);
    cb.SetContext(ctx);
    cb.OnTrainBegin();
    int seen = 0;
    for (int e = 0; e < values.Length; e++)
    {
        cb.OnEpochEnd(e, new Dictionary<string, float> { [key] = values[e] });
        seen++;
        if (ctx.StopTraining) break;
    }
    cb.OnTrainEnd();
    return (cb, seen);
}

// ---- test doubles -----------------------------------------------------

// Scalar-parameter stub (no forward behavior; exercises optimizer/serialization).
class ParamStub : BaseLayer
{
    public ParamStub(float w) : base("stub") { Parameters["w"] = np.array(new float[] { w }); }
}

// Vector-parameter stub for clipping math.
class VecStub : BaseLayer
{
    public VecStub(float[] w) : base("vec") { Parameters["w"] = np.array(w); }
}

// Echoes its input and records which sample indices each forward pass saw
// (column 0 of every row carries the sample's original index).
class SpyLayer : BaseLayer
{
    public List<int[]> Batches { get; } = new List<int[]>();

    public SpyLayer() : base("spy") { }

    public override void Forward(NDArray x)
    {
        base.Forward(x);
        int n = (int)x.shape[0];
        var ids = new int[n];
        for (int i = 0; i < n; i++)
            ids[i] = (int)Math.Round(x.GetSingle(i, 0));
        Batches.Add(ids);
        Output = x;
    }

    public override void Backward(NDArray grad) => InputGrad = grad;
}

// Records hook invocations for order/count assertions.
class RecordingCallback : BaseCallback
{
    public int TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchEnd;
    public List<string> Order { get; } = new List<string>();
    public List<int> EpochIndices { get; } = new List<int>();
    public Dictionary<string, float> LastLogs { get; private set; } = new Dictionary<string, float>();
    public bool SawValidation;

    public override void OnTrainBegin() { TrainBegin++; Order.Add("train_begin"); SawValidation = Context.HasValidation; }
    public override void OnTrainEnd() { TrainEnd++; Order.Add("train_end"); }
    public override void OnEpochBegin(int epoch) { EpochBegin++; Order.Add("epoch_begin"); EpochIndices.Add(epoch); }
    public override void OnBatchEnd(int batch, IDictionary<string, float> logs) { BatchEnd++; Order.Add("batch"); }
    public override void OnEpochEnd(int epoch, IDictionary<string, float> logs)
    {
        EpochEnd++;
        Order.Add("epoch_end");
        LastLogs = new Dictionary<string, float>(logs);
    }
}
