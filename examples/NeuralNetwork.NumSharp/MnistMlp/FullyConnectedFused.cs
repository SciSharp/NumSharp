using System;
using NeuralNetwork.NumSharp.Layers;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NeuralNetwork.NumSharp.MnistMlp
{
    public enum FusedActivation
    {
        /// <summary>No activation — forward is y = xW + b, backward passes gradient through unchanged.</summary>
        None,

        /// <summary>Element-wise ReLU — forward is y = max(xW + b, 0), backward is gradOutput * (y &gt; 0).</summary>
        ReLU,
    }

    /// <summary>
    /// Fully-connected (dense) layer with a bias term and an optional fused
    /// activation. Forward and backward each collapse their post-matmul
    /// element-wise chunk into a single NpyIter invocation:
    ///
    ///   Forward  (ReLU): y           = max(xW + b, 0)         — one NpyIter
    ///   Forward  (None): y           = xW + b                  — one NpyIter
    ///   Backward (ReLU): gradPreact  = gradOutput * (y &gt; 0)   — one NpyIter
    ///   Backward (None): gradPreact  = gradOutput              — pass-through
    ///
    /// Parameters follow the existing NeuralNetwork.NumSharp convention:
    /// Parameters["w"] is the weight matrix (InputDim, OutputDim) and
    /// Parameters["b"] is the bias vector (OutputDim,). Both are float32.
    /// Grads["w"] and Grads["b"] are filled in by Backward and consumed by
    /// the attached optimizer (Adam, SGD, etc.).
    ///
    /// The layer fills all the standard BaseLayer slots (Input, Output,
    /// InputGrad), so a vanilla <see cref="NeuralNet"/> pipeline composes
    /// it with existing activations and cost functions.
    /// </summary>
    public class FullyConnectedFused : BaseLayer
    {
        public int InputDim  { get; }
        public int OutputDim { get; }
        public FusedActivation Activation { get; }

        // Stable cache keys — the IL kernel is compiled once per (expr, dtypes)
        // combination and reused on every forward/backward pass for this process.
        private const string KeyBiasRelu    = "fcfused_bias_relu_f32";
        private const string KeyBiasOnly    = "fcfused_bias_only_f32";
        private const string KeyReluBackward = "fcfused_relu_backward_f32";

        public FullyConnectedFused(int inputDim, int outputDim, FusedActivation activation)
            : base("fc_fused")
        {
            if (inputDim  <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
            if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

            InputDim   = inputDim;
            OutputDim  = outputDim;
            Activation = activation;

            // He-normal for ReLU (preserves variance through the non-linearity);
            // Xavier/Glorot for linear output (keeps logits in a reasonable range).
            double stddev = activation == FusedActivation.ReLU
                ? Math.Sqrt(2.0 /  inputDim)
                : Math.Sqrt(2.0 / (inputDim + outputDim));

            Parameters["w"] = np.random.normal(0.0, stddev, new Shape(inputDim, outputDim))
                                       .astype(NPTypeCode.Single);
            Parameters["b"] = np.zeros(new Shape(outputDim), NPTypeCode.Single);
        }

        // =================================================================
        // Forward: y = activation(xW + b)
        // =================================================================

        public override void Forward(NDArray x)
        {
            base.Forward(x);  // stores x into this.Input

            NDArray W = Parameters["w"];
            NDArray b = Parameters["b"];

            NDArray preact = np.dot(x, W);           // (batch, OutputDim) float32
            NDArray output = np.empty_like(preact);  // allocated once, filled by fused kernel

            if (Activation == FusedActivation.ReLU)
                FuseBiasRelu(preact, b, output);
            else
                FuseBiasOnly(preact, b, output);

            Output = output;
        }

        // =================================================================
        // Backward: grad wrt input, weights, bias
        //
        // Given gradOutput (= dL/dy), produces:
        //   gradPreact = dL/d(preact)   (internal, not stored)
        //   Grads["w"] = x.T @ gradPreact
        //   Grads["b"] = sum(gradPreact, axis=0)
        //   InputGrad  = gradPreact @ W.T          (passed to the previous layer)
        // =================================================================

        public override void Backward(NDArray gradOutput)
        {
            NDArray W = Parameters["w"];
            NDArray gradPreact;

            if (Activation == FusedActivation.ReLU)
            {
                // Fused: gradPreact = gradOutput * (Output > 0).
                // Post-ReLU activation is zero wherever the pre-activation was
                // non-positive, so (y > 0) is exactly the ReLU mask.
                gradPreact = np.empty_like(gradOutput);
                FuseReluBackward(gradOutput, Output, gradPreact);
            }
            else
            {
                // No activation — pre-activation gradient equals output gradient.
                gradPreact = gradOutput;
            }

            // Parameter gradients. np.dot now ships a stride-aware GEMM
            // (BLIS-style packing), so transposed views go through the SIMD
            // fast path without materializing contiguous copies.
            Grads["w"] = np.dot(Input.transpose(), gradPreact);  // (InputDim, OutputDim)
            Grads["b"] = np.sum(gradPreact, axis: 0);            // (OutputDim,)

            // Gradient propagated back to the previous layer.
            InputGrad = np.dot(gradPreact, W.transpose());       // (batch, InputDim)
        }

        // =================================================================
        // Fused kernels (NpyIter + NpyExpr)
        // =================================================================

        /// <summary>y = max(preact + bias, 0) — single NpyIter, SIMD-capable.</summary>
        private static void FuseBiasRelu(NDArray preact, NDArray bias, NDArray output)
        {
            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op:  new[] { preact, bias, output },
                flags:   NpyIterGlobalFlags.EXTERNAL_LOOP,
                order:   NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_NO_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Max(
                NpyExpr.Input(0) + NpyExpr.Input(1),
                NpyExpr.Const(0f));

            iter.ExecuteExpression(
                expr,
                inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
                outputType: NPTypeCode.Single,
                cacheKey: KeyBiasRelu);
        }

        /// <summary>y = preact + bias — single NpyIter (final linear layer).</summary>
        private static void FuseBiasOnly(NDArray preact, NDArray bias, NDArray output)
        {
            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op:  new[] { preact, bias, output },
                flags:   NpyIterGlobalFlags.EXTERNAL_LOOP,
                order:   NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_NO_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Input(0) + NpyExpr.Input(1);

            iter.ExecuteExpression(
                expr,
                inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
                outputType: NPTypeCode.Single,
                cacheKey: KeyBiasOnly);
        }

        /// <summary>
        /// gradPreact[i,j] = gradOutput[i,j] * (activations[i,j] &gt; 0).
        ///
        /// Single NpyIter: the multiply and the comparison fuse into one
        /// element-wise sweep. The comparison result is auto-promoted to the
        /// output dtype (float32 here), so (y &gt; 0) evaluates to 1f or 0f and
        /// the multiply gates the gradient in place.
        /// </summary>
        private static void FuseReluBackward(NDArray gradOutput, NDArray activations, NDArray gradPreact)
        {
            using var iter = NpyIterRef.MultiNew(
                nop: 3,
                op:  new[] { gradOutput, activations, gradPreact },
                flags:   NpyIterGlobalFlags.EXTERNAL_LOOP,
                order:   NPY_ORDER.NPY_KEEPORDER,
                casting: NPY_CASTING.NPY_NO_CASTING,
                opFlags: new[]
                {
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.READONLY,
                    NpyIterPerOpFlags.WRITEONLY,
                });

            var expr = NpyExpr.Input(0) * NpyExpr.Greater(NpyExpr.Input(1), NpyExpr.Const(0f));

            iter.ExecuteExpression(
                expr,
                inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
                outputType: NPTypeCode.Single,
                cacheKey: KeyReluBackward);
        }
    }
}
