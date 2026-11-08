using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyPongLiveTests : InteropTestBase
{
    // Independent numerical oracle; never imports Gym/cPickle or runs the source's infinite loop.
    private const string Reference = """
        def forward(w1,w2,x):
            h=np.dot(w1,x)
            h[h<0]=0
            p=1./(1.+np.exp(-np.dot(w2,h)))
            return p,h
        def discount(rewards,gamma=.99):
            result=np.zeros_like(rewards)
            rf=rewards.reshape(-1); out=result.reshape(-1)
            running=0.
            for t in range(len(rf)-1,-1,-1):
                if rf[t]!=0: running=0.
                running=running*gamma+rf[t]
                out[t]=running
            return result
        def standardize(discounted):
            r=discounted.copy()
            r-=np.mean(r)
            r/=np.std(r)
            return r
        def backward(w2,x,h,weighted):
            dw2=np.dot(h.T,weighted.reshape(-1,1)).ravel()
            dh=np.outer(weighted,w2)
            dh[h<=0]=0
            return np.dot(dh.T,x),dw2
        def episode(w1,w2,x,labels,rewards):
            probs=[]; hidden=[]
            for row in x:
                p,h=forward(w1,w2,row)
                probs.append(p); hidden.append(h)
            p=np.array(probs); h=np.vstack(hidden)
            dlog=(labels-p).reshape(-1,1)
            advantages=standardize(discount(rewards)).reshape(-1,1)
            g1,g2=backward(w2,x,h,dlog*advantages)
            return p,h,dlog,advantages,g1,g2
        """;

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void PreprocessingAndDifference_FullRgbBufferAndMutations_Exact()
    {
        using var scope = NDScope.Open();
        var frame = np.full(new Shape(210, 160, 3), (byte)144, dtype: np.uint8);
        frame[35, 0, 0] = (byte)255; frame[35, 2, 0] = (byte)109;
        frame[193, 158, 0] = (byte)30; frame[34, 0, 0] = (byte)255;
        var mirror = frame.copy();
        ExportTo("image", mirror);
        using (Gil())
        {
            using var initial = Scope.Eval("image");
            GistParity.AssertExact(frame, initial, "identical input snapshots before either in-place preprocessing pass");
        }
        var actual = PongPolicyGradient.PreprocessFrame(frame);
        var previous = np.zeros(PongPolicyGradient.InputSize); previous[0] = .25;
        var difference = PongPolicyGradient.FrameDifference(actual, previous);
        var initialDifference = PongPolicyGradient.FrameDifference(actual, null);
        ExportTo("previous", previous);
        PyExec("""
            selected=image[35:195:2,::2,0]
            selected[selected==144]=0
            selected[selected==109]=0
            selected[selected!=0]=1
            current=selected.astype(np.float64).ravel()
            difference=current-previous
            initial_difference=np.zeros(6400)
            """);
        using (Gil())
        {
            using var expected = Scope.Eval("current");
            using var mutated = Scope.Eval("image");
            using var diff = Scope.Eval("difference");
            using var first = Scope.Eval("initial_difference");
            GistParity.AssertExact(actual, expected, "6400 preprocessed float64 pixels");
            GistParity.AssertExact(frame, mutated, "all100800 bytes of source-compatible RGB mutation");
            GistParity.AssertExact(difference, diff, "difference frame");
            GistParity.AssertExact(initialDifference, first, "first observation has zero motion");
        }
    }

    [DataTestMethod]
    [TestCategory("KarpathyByteParity")]
    [DataRow(false)]
    [DataRow(true)]
    public void DiscountAndSequentialStandardization_Exact(bool column)
    {
        using var scope = NDScope.Open();
        var values = np.array(new[] { 0.0, 0, 1, 0, -1, 0, 0, 1 });
        var rewards = column ? values.reshape(-1, 1) : values;
        var discounted = PongPolicyGradient.DiscountRewards(rewards);
        var normalized = PongPolicyGradient.StandardizeReturns(discounted);
        ExportTo("rewards", rewards);
        PyExec(Reference + "\ndiscounted=discount(rewards)\nnormalized=standardize(discounted)");
        using (Gil())
        {
            using var d = Scope.Eval("discounted");
            using var n = Scope.Eval("normalized");
            GistParity.AssertExact(discounted, d, "point-reset discounted return");
            GistParity.AssertExact(normalized, n, "center first, std of centered values second");
        }
    }

    /// <summary>
    /// The full-size (200 hidden x 6400 inputs) seed-3 initialization, the next uniform draw on the same
    /// stream, then one forward/backward pass: all byte-identical to NumPy.
    /// </summary>
    /// <remarks>
    /// The expected weights come from <c>legacy_randn</c> (<see cref="InteropTestBase.DefineLegacyRandn"/>):
    /// on x64 that is NumPy's own <c>rng.randn</c>. On arm64, NumPy's wheel fuses <c>legacy_gauss</c>'s
    /// <c>r2</c>, and W1 failed at element 44 on macos-latest, so there the comparison is against the literal
    /// evaluation of the same live seed-3 stream. <c>legacy_randn</c> advances <c>rng</c> exactly as the literal
    /// sampler does, so <c>expected_draw</c> still comes from NumPy's own generator on every host. The
    /// forward/backward half exports NumSharp's weights, so it compares directly against NumPy everywhere.
    /// </remarks>
    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void Full200By6400_SeededInitializationForwardAndBackward_Exact()
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(3);
        using var model = PongPolicyGradient.Initialize(rng);
        var nextDraw = rng.uniform();
        DefineLegacyRandn();
        PyExec("rng=np.random.RandomState(3)\nexpected_w1=legacy_randn(rng,200,6400)/np.sqrt(6400)\nexpected_w2=legacy_randn(rng,200)/np.sqrt(200)\nexpected_draw=np.array(rng.uniform())");
        using (Gil())
        {
            using var w1 = Scope.Eval("expected_w1"); using var w2 = Scope.Eval("expected_w2"); using var draw = Scope.Eval("expected_draw");
            GistParity.AssertExact(model.W1, w1, "all1280000 W1 coefficients at original dimensions");
            GistParity.AssertExact(model.W2, w2, "200 output weights");
            GistParity.AssertExact(nextDraw, draw, "the next action draw shares the initialization RNG stream");
        }
        var data = new double[4, 6400];
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 20; k++) data[row, (row * 17 + k * 113) % 6400] = k % 2 == 0 ? 1 : -1;
        var inputs = np.array(data);
        int[] actions = { 2, 3, 2, 3 };
        var rollout = Evaluate(model, inputs, actions);
        var rewards = np.array(new[] { 0.0, 1, 0, -1 }).reshape(-1, 1);
        var advantage = PongPolicyGradient.StandardizeReturns(PongPolicyGradient.DiscountRewards(rewards));
        var gradients = PongPolicyGradient.Backward(model, inputs, rollout.Hidden, rollout.LogGradient * advantage);
        ExportTo("w1", model.W1); ExportTo("w2", model.W2); ExportTo("x", inputs); ExportTo("rewards", rewards);
        PyExec(Reference + "\np,h,dlog,advantages,g1,g2=episode(w1,w2,x,np.array([1.,0.,1.,0.]),rewards)");
        using (Gil())
        {
            using var p = Scope.Eval("p"); using var h = Scope.Eval("h"); using var d = Scope.Eval("dlog");
            using var g1 = Scope.Eval("g1"); using var g2 = Scope.Eval("g2");
            GistParity.AssertExact(rollout.Probabilities, p, "per-frame GEMV policy probabilities");
            GistParity.AssertExact(rollout.Hidden, h, "4x200 ReLU hidden states");
            GistParity.AssertExact(rollout.LogGradient, d, "sampled-action log gradients");
            GistParity.AssertExact(gradients.W1, g1, "full200x6400 backward gradient");
            GistParity.AssertExact(gradients.W2, g2, "200 output gradients");
        }
    }

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void RmsProp_TwoStateUpdatesAndGradientClears_Exact()
    {
        using var scope = NDScope.Open();
        var weights = np.array(new double[,] { { .2, -.3 }, { .5, .1 } });
        var cache = np.array(new double[,] { { .01, .02 }, { .03, .04 } });
        ExportTo("initial_w", weights.copy()); ExportTo("initial_cache", cache.copy());
        PyExec("w=initial_w.copy()\ncache=initial_cache.copy()");
        for (int i = 0; i < 2; i++)
        {
            var gradient = np.array(new double[,] { { .1 + i, -.2 }, { -.3, .4 + i } });
            ExportTo("initial_g", gradient.copy());
            PongPolicyGradient.RmsPropUpdate(weights, gradient, cache);
            PyExec("g=initial_g.copy()\ncache=.99*cache+(1-.99)*g**2\nw+=1e-4*g/(np.sqrt(cache)+1e-5)\ng=np.zeros_like(g)");
            using (Gil())
            {
                using var ew = Scope.Eval("w"); using var ec = Scope.Eval("cache"); using var eg = Scope.Eval("g");
                GistParity.AssertExact(weights, ew, $"RMSProp ascent weights step{i}");
                GistParity.AssertExact(cache, ec, $"RMSProp persistent cache step{i}");
                GistParity.AssertExact(gradient, eg, $"batch gradient reset step{i}");
            }
        }
    }

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void BatchedEpisodes_AllPersistentStateAndRewardMean_Exact()
    {
        using var scope = NDScope.Open();
        using var w1 = np.array(new double[,] { { .2, -.3, .4, .1 }, { -.5, .2, .1, .3 }, { .1, .3, -.2, .4 } });
        using var w2 = np.array(new[] { .2, -.4, .3 });
        using var model = new PongPolicyGradient.Model(w1, w2);
        using var trainer = new PongPolicyGradient.Trainer(model, batchSize: 2);
        var inputs = np.array(new double[,] { { 1, -.5, .25, 0 }, { -.3, .5, 1, -.25 }, { .2, .1, -.5, .7 } });
        ExportTo("initial_w1", w1); ExportTo("initial_w2", w2); ExportTo("x", inputs);
        PyExec(Reference + "\nw1=initial_w1.copy(); w2=initial_w2.copy()\nb1=np.zeros_like(w1); b2=np.zeros_like(w2)\nc1=np.zeros_like(w1); c2=np.zeros_like(w2)\nrunning=None");
        for (int episode = 1; episode <= 2; episode++)
        {
            var rollout = Evaluate(model, inputs, new[] { 2, 3, 2 });
            var rewards = np.array(episode == 1 ? new[] { 0.0, 1, -1 } : new[] { 0.0, 0, 1 }).reshape(-1, 1);
            ExportTo("rewards", rewards);
            bool updated = trainer.TrainEpisode(inputs, rollout.Hidden, rollout.LogGradient, rewards);
            PyExec($"p,h,d,a,g1,g2=episode(w1,w2,x,np.array([1.,0.,1.]),rewards)\nb1+=g1; b2+=g2\nupdate={(episode % 2 == 0 ? "True" : "False")}\n" + """
                if update:
                    c1=.99*c1+(1-.99)*b1**2; c2=.99*c2+(1-.99)*b2**2
                    w1+=1e-4*b1/(np.sqrt(c1)+1e-5); w2+=1e-4*b2/(np.sqrt(c2)+1e-5)
                    b1=np.zeros_like(w1); b2=np.zeros_like(w2)
                total=float(np.sum(rewards))
                running=total if running is None else running*.99+total*.01
                """);
            Assert.AreEqual(episode == 2, updated);
            Assert.AreEqual(PyFloat("running"), trainer.RunningReward);
            using (Gil())
            {
                using var ew1 = Scope.Eval("w1"); using var ew2 = Scope.Eval("w2");
                using var eb1 = Scope.Eval("b1"); using var eb2 = Scope.Eval("b2");
                using var ec1 = Scope.Eval("c1"); using var ec2 = Scope.Eval("c2");
                GistParity.AssertExact(model.W1, ew1, $"episode{episode} W1");
                GistParity.AssertExact(model.W2, ew2, $"episode{episode} W2");
                GistParity.AssertExact(trainer.GradientW1, eb1, $"episode{episode} accumulated W1 gradient");
                GistParity.AssertExact(trainer.GradientW2, eb2, $"episode{episode} accumulated W2 gradient");
                GistParity.AssertExact(trainer.CacheW1, ec1, $"episode{episode} W1 cache");
                GistParity.AssertExact(trainer.CacheW2, ec2, $"episode{episode} W2 cache");
            }
        }
    }

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void SigmoidAndActionBoundary_ExactIncludingSaturation()
    {
        using var scope = NDScope.Open();
        double[] values = { double.NegativeInfinity, -1000, -1, 0, 1, 1000, double.PositiveInfinity };
        var logits = np.array(values);
        var probabilities = new double[values.Length];
        for (int i = 0; i < values.Length; i++) probabilities[i] = PongPolicyGradient.Sigmoid(values[i]);
        var actual = np.array(probabilities);
        ExportTo("logits", logits);
        PyExec("with np.errstate(over='ignore'):\n    expected=1./(1.+np.exp(-logits))");
        using (Gil())
        {
            using var expected = Scope.Eval("expected");
            GistParity.AssertExact(actual, expected, "sigmoid limits and finite values");
        }
    }

    [TestMethod]
    [TestCategory("KarpathyShortRun")]
    public void OriginalPong_ShortCompleteRunThroughPythonnet_ReimportsTrainedState()
    {
        using var scope = NDScope.Open();
        LoadOriginalPong();
        PyExec("pg_initialize(23,8)");
        for (int episode = 0; episode < 10; episode++)
        {
            PyExec("pg_begin_episode()");
            for (int tick = 0; tick < 4; tick++)
            {
                using var frame = SyntheticFrame(episode, tick);
                CollectionAssert.AreEqual(new long[] { 210, 160, 3 }, frame.shape);
                Assert.AreEqual(NPTypeCode.Byte, frame.typecode);
                ExportTo("pg_frame_input", frame); // NumSharp.Interop.pythonnet zero-copy lease
                PyExec($"pg_tick(pg_frame_input,{tick})");
                int row = 35 + 2 * ((episode * 5 + tick * 3) % 80);
                int column = 2 * ((episode * 7 + tick * 11 + 5) % 80);
                CollectionAssert.AreEqual(new long[] { 210, 160, 3 }, frame.shape);
                Assert.AreEqual((byte)1, frame.item<byte>(row, column, 0),
                    "Original Python prepro writes through the exported NumSharp frame.");
                Assert.AreEqual((byte)144, frame.item<byte>(row, column, 1), "The green channel is not part of preprocessing.");
                Assert.AreEqual((byte)144, frame.item<byte>(row, column, 2), "The blue channel is not part of preprocessing.");
                Assert.IsTrue(PyBool("np.isfinite(pg_probability) and 0 < pg_probability < 1"));
                Assert.IsTrue(PyBool("pg_current.shape==(6400,) and pg_input.shape==(6400,) and pg_hidden.shape==(8,) and np.isfinite(pg_hidden).all() and pg_action in (2,3)"));
            }
            PyExec("pg_finish_episode()");
            Assert.AreEqual((long)episode + 1, PyLong("pg_episodes"));
            Assert.IsTrue(PyBool("np.isfinite(pg_advantages).all() and all(np.isfinite(g).all() for g in pg_gradients.values())"));
        }
        Assert.AreEqual(1L, PyLong("pg_updates"));
        Assert.IsTrue(PyBool("all(np.count_nonzero(g)==0 for g in pg_gradient_buffer.values())"));
        Assert.IsTrue(PyBool("np.any(pg_rms_cache['W1']>0) and np.any(pg_model['W1']!=pg_initial_model['W1'])"));
        using var trainedWeights = ImportOf("pg_model['W1']"); // owning Python→NumSharp copy
        CollectionAssert.AreEqual(new long[] { 8, 6400 }, trainedWeights.shape);
        Assert.IsTrue(np.all(np.isfinite(trainedWeights)));
        Console.WriteLine($"PONG_SHORT_RUN original_functions=5; episodes=10; frames=40; updates=1; hidden=8; input=6400; changed_weights={PyLong("int(np.count_nonzero(pg_model['W1']!=pg_initial_model['W1']))")}; interop_roundtrip=passed; environment=synthetic_not_Atari");
    }

    [TestMethod]
    [TestCategory("KarpathyByteParity")]
    public void ShortPong_AllStepsAndElevenEpisodes_ByteExactAgainstPinnedOriginal()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(23);
        using var model = PongPolicyGradient.Initialize(random, hidden: 8);
        using var trainer = new PongPolicyGradient.Trainer(model); // original ten-episode batch
        LoadOriginalPong();
        PyExec("pg_initialize(23,8)");
        long checkedArrays = 0, checkedBytes = 0;
        CheckOriginalFields(ref checkedArrays, ref checkedBytes, "initial", new[] {
            (model.W1,"pg_model['W1']"),(model.W2,"pg_model['W2']"),
            (trainer.GradientW1,"pg_gradient_buffer['W1']"),(trainer.GradientW2,"pg_gradient_buffer['W2']"),
            (trainer.CacheW1,"pg_rms_cache['W1']"),(trainer.CacheW2,"pg_rms_cache['W2']") });
        // The extra eleventh episode exercises forward/backward using UPDATED weights and
        // accumulates a fresh post-reset gradient batch. Only episode10 updates the model.
        for (int episode = 0; episode < 11; episode++)
        {
            using var episodeScope = NDScope.Open();
            PyExec("pg_begin_episode()");
            var inputs = new List<NDArray>(); var hidden = new List<NDArray>();
            var logGradients = new double[4]; var rewards = new double[4];
            NDArray? previous = null;
            for (int tick = 0; tick < 4; tick++)
            {
                using var frame = SyntheticFrame(episode, tick);
                using var mirror = frame.copy();
                ExportTo("pg_frame_input", mirror);
                CheckOriginalFields(ref checkedArrays, ref checkedBytes, $"episode{episode+1}/tick{tick}/before", new[] { (frame,"pg_frame_input") });
                var current = PongPolicyGradient.PreprocessFrame(frame);
                var input = PongPolicyGradient.FrameDifference(current, previous);
                previous?.Dispose(); previous = current;
                var prediction = PongPolicyGradient.Forward(model, input);
                var draw = random.uniform();
                int action = PongPolicyGradient.SampleAction(prediction.Probability, draw.item<double>());
                double score = PongPolicyGradient.ActionLogGradient(action, prediction.Probability);
                double reward = PointReward(tick, action);
                inputs.Add(input); hidden.Add(prediction.Hidden); logGradients[tick] = score; rewards[tick] = reward;
                PyExec($"pg_tick(pg_frame_input,{tick})");
                CheckOriginalFields(ref checkedArrays, ref checkedBytes, $"episode{episode+1}/tick{tick}", new[] {
                    (frame,"pg_frame_input"),(current,"pg_current"),(input,"pg_input"),
                    (prediction.Hidden,"pg_hidden"),(np.array(prediction.Probability),"np.asarray(pg_probability)"),
                    (draw,"np.asarray(pg_draw)"),(np.array((long)action),"np.asarray(pg_action,dtype=np.int64)"),
                    (np.array(score),"np.asarray(pg_score)"),(np.array(reward),"np.asarray(pg_point_reward)") });
            }
            var x = np.vstack(inputs.ToArray()); var h = np.vstack(hidden.ToArray());
            var scores = np.array(logGradients).reshape(-1, 1); var returns = np.array(rewards).reshape(-1, 1);
            var discounted = PongPolicyGradient.DiscountRewards(returns);
            var advantages = PongPolicyGradient.StandardizeReturns(discounted);
            var weighted = scores * advantages;
            var gradients = PongPolicyGradient.Backward(model, x, h, weighted);
            var accumulatedW1 = trainer.GradientW1 + gradients.W1;
            var accumulatedW2 = trainer.GradientW2 + gradients.W2;
            bool updated = trainer.TrainEpisode(x, h, scores, returns);
            PyExec("pg_finish_episode()");
            CheckOriginalFields(ref checkedArrays, ref checkedBytes, $"episode{episode+1}/complete", new[] {
                (x,"pg_epx"),(h,"pg_eph"),(scores,"pg_unweighted_scores"),(returns,"pg_epr"),
                (discounted,"pg_discounted"),(advantages,"pg_advantages"),(weighted,"pg_weighted_scores"),
                (gradients.W1,"pg_gradients['W1']"),(gradients.W2,"pg_gradients['W2']"),
                (accumulatedW1,"pg_accumulated_before_update['W1']"),(accumulatedW2,"pg_accumulated_before_update['W2']"),
                (model.W1,"pg_model['W1']"),(model.W2,"pg_model['W2']"),
                (trainer.GradientW1,"pg_gradient_buffer['W1']"),(trainer.GradientW2,"pg_gradient_buffer['W2']"),
                (trainer.CacheW1,"pg_rms_cache['W1']"),(trainer.CacheW2,"pg_rms_cache['W2']"),
                (np.array(trainer.LastEpisodeReward),"np.asarray(pg_episode_reward)"),
                (np.array(trainer.RunningReward!.Value),"np.asarray(pg_running_reward)") });
            Assert.AreEqual(episode == 9, updated);
            Assert.AreEqual((long)trainer.Episodes, PyLong("pg_episodes"));
            Assert.AreEqual((long)trainer.Updates, PyLong("pg_updates"));
        }
        Assert.AreEqual(11, trainer.Episodes); Assert.AreEqual(1, trainer.Updates);
        Assert.IsTrue(PyBool("np.any(pg_model['W1']!=pg_initial_model['W1']) and np.any(pg_gradient_buffer['W1']!=0)"));
        Console.WriteLine($"PONG_BYTE_PARITY original_functions=5; episodes=11; frames=44; updates=1; arrays={checkedArrays}; bytes={checkedBytes}; mismatches=0; max_ulp=0; hidden=8; input=6400; environment=synthetic_not_Atari");
    }

    /// <summary>
    /// Loads the hash-pinned original Pong functions into this test's scope, asserts that nothing beyond
    /// the five numerical functions came along (no Gym, pickle or environment), and defines the bounded
    /// <see cref="OriginalPongDriver"/> around them.
    /// </summary>
    /// <remarks>
    /// <c>legacy_randn</c> is defined first because <c>pg_initialize</c> draws the initial weights through it:
    /// NumPy's own <c>randn</c> on x64, the literal re-evaluation of the same stream on a fusing (arm64)
    /// NumPy. Without that, the short run's very first comparison (W1 element 10 on macos-latest) failed
    /// and nothing after it could be checked. See <see cref="InteropTestBase.DefineLegacyRandn"/>.
    /// </remarks>
    /// <exception cref="AssertFailedException">The pinned source's hash or loaded-name inventory changed.</exception>
    private void LoadOriginalPong()
    {
        KarpathyOriginalSource.Load(Scope, "pong");
        Assert.AreEqual("cf764d11a0ebebb46d02c482c5a9c7c31081960d24b7c332757362735ad71a14", PyStr("original['__source_sha256__']"));
        Assert.IsTrue(PyBool("original['__loaded_names__']==('sigmoid','prepro','discount_rewards','policy_forward','policy_backward')"));
        Assert.IsTrue(PyBool("'gym' not in original and 'pickle' not in original and 'env' not in original"));
        DefineLegacyRandn();
        PyExec(OriginalPongDriver);
    }

    private void CheckOriginalFields(ref long arrays, ref long bytes, string context, (NDArray Value, string PythonExpression)[] fields)
    {
        using (Gil())
        {
            foreach (var field in fields)
            {
                using var expected = Scope.Eval(field.PythonExpression);
                GistParity.AssertExact(field.Value, expected, $"Pong {context}: {field.PythonExpression}");
                arrays++; bytes += field.Value.size * field.Value.dtypesize;
            }
        }
    }

    private static NDArray SyntheticFrame(int episode, int tick)
    {
        using var scope = NDScope.Open();
        var frame = np.full(new Shape(210, 160, 3), (byte)144, dtype: np.uint8);
        frame[35 + 2 * ((episode * 5 + tick * 3) % 80), 2 * ((episode * 7 + tick * 11 + 5) % 80), 0] = (byte)255;
        frame[35, 2, 0] = (byte)109; // second background color is erased
        frame[34, 0, 0] = (byte)255; // out-of-crop pixel stays untouched
        frame[36, 0, 0] = (byte)30;  // skipped row stays untouched
        return scope.Returns(frame);
    }

    private static double PointReward(int tick, int action)
        => tick == 1 ? (action == 2 ? 1.0 : -1.0) : tick == 3 ? (action == 2 ? -1.0 : 1.0) : 0.0;

    // Independent bounded environment/episode driver. The FIVE numerical functions are
    // loaded from the hash-verified original source by KarpathyOriginalSource, not rewritten.
    // Model setup, normalization and optimizer scheduling preserve the top-level numerical
    // equations while replacing only Gym observations/rewards and the infinite loop.
    // The initial weights draw through legacy_randn (InteropTestBase.DefineLegacyRandn, defined by
    // LoadOriginalPong): pg_rng.randn itself on x64. On a NumPy that fuses legacy_gauss (arm64), it
    // is the literal evaluation of pg_rng's own stream, left at the position the literal sampler
    // reaches, so every later pg_rng.uniform() action draw still comes from NumPy's generator.
    private const string OriginalPongDriver = """
        def pg_initialize(seed,hidden):
            global pg_rng,pg_model,pg_initial_model,pg_gradient_buffer,pg_rms_cache
            global pg_episodes,pg_updates,pg_running_reward
            pg_rng=np.random.RandomState(seed)
            pg_model={'W1':legacy_randn(pg_rng,hidden,6400)/np.sqrt(6400),
                      'W2':legacy_randn(pg_rng,hidden)/np.sqrt(hidden)}
            pg_initial_model={name:value.copy() for name,value in pg_model.items()}
            pg_gradient_buffer={name:np.zeros_like(value) for name,value in pg_model.items()}
            pg_rms_cache={name:np.zeros_like(value) for name,value in pg_model.items()}
            pg_episodes=0; pg_updates=0; pg_running_reward=None
            original['model']=pg_model
            original['gamma']=.99
        def pg_begin_episode():
            global pg_previous,pg_inputs,pg_hiddens,pg_scores,pg_rewards,pg_episode_reward
            pg_previous=None
            pg_inputs=[]; pg_hiddens=[]; pg_scores=[]; pg_rewards=[]
            pg_episode_reward=0.
        def pg_tick(frame,tick):
            global pg_previous,pg_current,pg_input,pg_probability,pg_hidden
            global pg_draw,pg_action,pg_score,pg_point_reward,pg_episode_reward
            pg_current=original['prepro'](frame)
            pg_input=pg_current-pg_previous if pg_previous is not None else np.zeros(6400)
            pg_previous=pg_current
            pg_probability,pg_hidden=original['policy_forward'](pg_input)
            pg_draw=pg_rng.uniform()
            pg_action=2 if pg_draw<pg_probability else 3
            pg_score=(1 if pg_action==2 else 0)-pg_probability
            pg_point_reward=(1. if pg_action==2 else -1.) if tick==1 else ((-1. if pg_action==2 else 1.) if tick==3 else 0.)
            pg_episode_reward+=pg_point_reward
            pg_inputs.append(pg_input); pg_hiddens.append(pg_hidden)
            pg_scores.append(pg_score); pg_rewards.append(pg_point_reward)
        def pg_finish_episode():
            global pg_epx,pg_eph,pg_unweighted_scores,pg_epr,pg_discounted,pg_advantages,pg_weighted_scores
            global pg_gradients,pg_accumulated_before_update,pg_episodes,pg_updates,pg_running_reward
            pg_epx=np.vstack(pg_inputs); pg_eph=np.vstack(pg_hiddens)
            pg_unweighted_scores=np.vstack(pg_scores); pg_epr=np.vstack(pg_rewards)
            pg_discounted=original['discount_rewards'](pg_epr)
            pg_advantages=pg_discounted.copy()
            pg_advantages-=np.mean(pg_advantages)
            pg_advantages/=np.std(pg_advantages)
            pg_weighted_scores=pg_unweighted_scores.copy()
            pg_weighted_scores*=pg_advantages
            original['epx']=pg_epx
            pg_gradients=original['policy_backward'](pg_eph,pg_weighted_scores)
            for name in pg_model: pg_gradient_buffer[name]+=pg_gradients[name]
            pg_accumulated_before_update={name:value.copy() for name,value in pg_gradient_buffer.items()}
            pg_episodes+=1
            if pg_episodes%10==0:
                for name,value in pg_model.items():
                    gradient=pg_gradient_buffer[name]
                    pg_rms_cache[name]=.99*pg_rms_cache[name]+(1-.99)*gradient**2
                    pg_model[name]+=1e-4*gradient/(np.sqrt(pg_rms_cache[name])+1e-5)
                    pg_gradient_buffer[name]=np.zeros_like(value)
                pg_updates+=1
            pg_running_reward=pg_episode_reward if pg_running_reward is None else pg_running_reward*.99+pg_episode_reward*.01
        """;

    private static (NDArray Hidden, NDArray LogGradient, NDArray Probabilities) Evaluate(PongPolicyGradient.Model model, NDArray inputs, int[] actions)
    {
        using var scope = NDScope.Open();
        var hidden = new NDArray[actions.Length]; var gradients = np.zeros((actions.Length, 1)); var probabilities = np.zeros(actions.Length);
        for (int t = 0; t < actions.Length; t++)
        {
            var prediction = PongPolicyGradient.Forward(model, inputs[$"{t},:"]);
            hidden[t] = prediction.Hidden;
            probabilities[t] = prediction.Probability;
            gradients[t, 0] = PongPolicyGradient.ActionLogGradient(actions[t], prediction.Probability);
        }
        return scope.Returns((np.vstack(hidden), gradients, probabilities));
    }
}
