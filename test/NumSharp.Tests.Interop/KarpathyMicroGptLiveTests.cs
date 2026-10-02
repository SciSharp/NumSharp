using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyMicroGptLiveTests : InteropTestBase
{
    // Independently expressed matrix reverse-mode oracle, not the C# computation delegated to Python.
    private const string Reference = """
        import numpy as np, math
        def rms(x):
            s=np.power(np.mean(x*x,axis=1,keepdims=True)+1e-5,-.5)
            return x*s,s
        def drms(g,x,s):
            return g*s-x*((s*s*s)/x.shape[1])*np.sum(g*x,axis=1,keepdims=True)
        def softmax(x):
            e=np.exp(x-np.max(x,axis=1,keepdims=True))
            return e/np.sum(e,axis=1,keepdims=True)
        def forward(w,tokens,heads,layers):
            raw=w['wte'][tokens]+w['wpe'][:len(tokens)]
            x,scale=rms(raw)
            cache=[]
            for layer in range(layers):
                p='layer%d.'%layer
                incoming=x
                z,s1=rms(x)
                q=z@w[p+'attn_wq'].T; k=z@w[p+'attn_wk'].T; v=z@w[p+'attn_wv'].T
                width=x.shape[1]//heads
                joined=np.empty_like(x); attention=[]
                for head in range(heads):
                    sl=slice(head*width,(head+1)*width)
                    scores=(q[:,sl]@k[:,sl].T)/math.sqrt(width)
                    scores[np.triu_indices(len(tokens),1)]=-np.inf
                    a=softmax(scores); attention.append(a)
                    joined[:,sl]=a@v[:,sl]
                residual=joined@w[p+'attn_wo'].T+incoming
                zn,s2=rms(residual)
                pre=zn@w[p+'mlp_fc1'].T
                hidden=np.maximum(pre,0)
                x=hidden@w[p+'mlp_fc2'].T+residual
                cache.append((incoming,z,s1,q,k,v,attention,joined,residual,zn,s2,pre,hidden))
            return x@w['lm_head'].T,(raw,scale,cache,x)
        def backward(w,tokens,targets,heads,layers):
            logits,(raw,scale,cache,x)=forward(w,tokens,heads,layers)
            probs=softmax(logits)
            loss=-np.mean(np.log(probs)[np.arange(len(tokens)),targets],axis=0)
            g={name:np.zeros_like(value) for name,value in w.items()}
            d=probs.copy(); d[np.arange(len(tokens)),targets]-=1; d=d/len(tokens)
            g['lm_head']=d.T@x
            dx=d@w['lm_head']
            for layer in reversed(range(layers)):
                p='layer%d.'%layer
                incoming,z,s1,q,k,v,attention,joined,residual,zn,s2,pre,hidden=cache[layer]
                g[p+'mlp_fc2']=dx.T@hidden
                dh=(dx@w[p+'mlp_fc2'])*(pre>0)
                g[p+'mlp_fc1']=dh.T@zn
                dr=dx+drms(dh@w[p+'mlp_fc1'],residual,s2)
                g[p+'attn_wo']=dr.T@joined
                da=dr@w[p+'attn_wo']
                dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
                width=x.shape[1]//heads
                for head in range(heads):
                    sl=slice(head*width,(head+1)*width)
                    a=attention[head]; h=da[:,sl]
                    dp=h@v[:,sl].T
                    ds=a*(dp-np.sum(a*dp,axis=1,keepdims=True))
                    dq[:,sl]=(ds@k[:,sl])/math.sqrt(width)
                    dk[:,sl]=(ds.T@q[:,sl])/math.sqrt(width)
                    dv[:,sl]=a.T@h
                g[p+'attn_wq']=dq.T@z; g[p+'attn_wk']=dk.T@z; g[p+'attn_wv']=dv.T@z
                dz=(dq@w[p+'attn_wq']+dk@w[p+'attn_wk'])+dv@w[p+'attn_wv']
                dx=dr+drms(dz,incoming,s1)
            de=drms(dx,raw,scale)
            for row,token in enumerate(tokens):
                g['wte'][token]+=de[row]
                g['wpe'][row]=de[row]
            return loss,logits,g
        def adam(w,g,m,v,step,rate):
            for name in w:
                m[name]=.85*m[name]+(1-.85)*g[name]
                v[name]=.99*v[name]+(1-.99)*np.square(g[name])
                mh=m[name]/(1-math.pow(.85,step))
                vh=v[name]/(1-math.pow(.99,step))
                w[name]-=rate*mh/(np.sqrt(vh)+1e-8)
        # Scalar lists deliberately use sequential multiply/add like the original Value graph.
        # This is a VALUE oracle only; its reduction order is not our vectorized byte contract.
        def scalar_forward(w,tokens,heads,layers):
            def sequential(items):
                total=0
                for item in items: total=total+item
                return total
            def lin(x,w): return [sequential(float(a)*b for a,b in zip(row,x)) for row in w]
            def rn(x):
                scale=(sequential(a*a for a in x)/len(x)+1e-5)**-.5
                return [a*scale for a in x]
            def sm(x):
                mx=max(x); e=[math.exp(a-mx) for a in x]; s=sequential(e)
                return [a/s for a in e]
            keys=[[] for _ in range(layers)]; values=[[] for _ in range(layers)]; out=[]
            for pos,token in enumerate(tokens):
                x=rn([float(a)+float(b) for a,b in zip(w['wte'][token],w['wpe'][pos])])
                for layer in range(layers):
                    p='layer%d.'%layer; residual=x; z=rn(x)
                    q=lin(z,w[p+'attn_wq']); k=lin(z,w[p+'attn_wk']); v=lin(z,w[p+'attn_wv'])
                    keys[layer].append(k); values[layer].append(v)
                    joined=[]; width=len(x)//heads
                    for h in range(heads):
                        start=h*width
                        a=sm([sequential(q[start+j]*kt[start+j] for j in range(width))/math.sqrt(width) for kt in keys[layer]])
                        joined.extend(sequential(a[t]*values[layer][t][start+j] for t in range(len(a))) for j in range(width))
                    x=[a+b for a,b in zip(lin(joined,w[p+'attn_wo']),residual)]
                    residual=x; hidden=[max(0,a) for a in lin(rn(x),w[p+'mlp_fc1'])]
                    x=[a+b for a,b in zip(lin(hidden,w[p+'mlp_fc2']),residual)]
                out.append(lin(x,w['lm_head']))
            return np.array(out)
        """;

    [DataTestMethod, TestCategory("KarpathyByteParity")]
    [DataRow(4, 2, 1)]
    [DataRow(16, 4, 1)]
    [DataRow(4, 2, 2)]
    public void CompleteTransformer_LogitsLossAndEveryGradient_ExactNumpy(int width, int heads, int layers)
    {
        using var model = new MicroGpt.Model(5, width, heads, blockSize: 16, layerCount: layers, seed: 7);
        int[] tokens = { 4, 0, 1, 0 }, targets = { 0, 1, 0, 4 };
        SetReference(model, tokens, targets);
        using var actual = model.LossAndGradients(tokens, targets);
        PyExec($"loss,logits,gradients=backward(w,tokens,targets,{heads},{layers})");
        Exact(actual.Logits, "logits", "microgpt full logits");
        using var loss = np.array(actual.Loss);
        Exact(loss, "np.asarray(loss)", "microgpt loss");
        foreach (var (name, gradient) in actual.Parameters) Exact(gradient, $"gradients['{name}']", "microgpt gradient " + name);
    }

    [TestMethod, TestCategory("KarpathyShortRun")]
    public void OriginalDimensionCausalModel_AgreesWithScalarValueEquations()
    {
        using var model = new MicroGpt.Model(5, seed: 11);
        int[] tokens = { 4, 0, 1, 2, 0, 3 }, targets = { 0, 1, 2, 0, 3, 4 };
        SetReference(model, tokens, targets);
        using var actual = model.Forward(tokens);
        ExportTo("actual", actual);
        PyExec("scalar=scalar_forward(w,tokens,4,1)\nmax_absolute=float(np.max(np.abs(actual-scalar)))");
        double difference = PyFloat("max_absolute");
        Console.WriteLine($"MICROGPT_SCALAR_VALUE max_absolute_logit_difference={difference:R}; this is not a byte-equality claim.");
        Assert.IsTrue(difference <= 2e-14, "The vectorized transformer must agree numerically with the source's scalar causal equations.");
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void AdamFourUpdates_AllWeightsAndBothMoments_ExactNumpy()
    {
        using var model = new MicroGpt.Model(4, embeddingSize: 4, headCount: 2, blockSize: 4, seed: 5);
        int[] tokens = { 3, 0, 1 }, targets = { 0, 1, 3 };
        SetReference(model, tokens, targets);
        PyExec("w={k:v.copy() for k,v in w.items()}\nm={k:np.zeros_like(v) for k,v in w.items()}\nv={k:np.zeros_like(v) for k,v in w.items()}");
        for (int step = 0; step < 4; step++)
        {
            using var batch = model.LossAndGradients(tokens, targets);
            double rate = .01 * (1 - step / 4.0);
            model.ApplyAdam(batch, rate);
            PyExec($"loss,logits,g=backward(w,tokens,targets,2,1)\nadam(w,g,m,v,{step + 1},{rate.ToString("R", System.Globalization.CultureInfo.InvariantCulture)})");
            foreach (var (name, value) in model.Parameters)
            {
                Exact(value, $"w['{name}']", "Adam parameter " + name);
                Exact(model.FirstMoments[name], $"m['{name}']", "Adam first moment " + name);
                Exact(model.SecondMoments[name], $"v['{name}']", "Adam second moment " + name);
            }
        }
        Assert.AreEqual(4, model.OptimizerStep);
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void TwentySamplesAfterTraining_MatchIndependentAutoregressiveInference()
    {
        var tokenizer = new MicroGpt.Tokenizer(MicroGpt.TinyNames);
        using var model = new MicroGpt.Model(tokenizer.Size);
        using var training = model.Train(tokenizer, MicroGpt.TinyNames, steps: 30);
        SetReference(model, new[] { tokenizer.Bos }, new[] { tokenizer.Bos });
        // The same trained weights enter an independent NumPy inference loop; no C# output is reused.
        // Sampling uses the explicitly documented NumPy RNG, not original Python random.choices.
        PyExec($"chars=sorted(set('{string.Concat(MicroGpt.TinyNames)}'))\nbos={tokenizer.Bos}");
        PyExec("""
            rng=np.random.RandomState(42)
            samples=[]
            for sample in range(20):
                tokens=[bos]; result=[]
                for position in range(16):
                    logits,_=forward(w,np.array(tokens),4,1)
                    probabilities=softmax(logits[-1:]/.5)[0]
                    draw=rng.rand(); total=0; token=len(probabilities)-1
                    for i,p in enumerate(probabilities):
                        total+=p
                        if draw<total: token=i; break
                    if token==bos: break
                    result.append(chars[token]); tokens.append(token)
                samples.append(''.join(result))
            """);
        string[] actual = model.Generate(tokenizer, samples: 20, temperature: .5, seed: 42);
        Assert.AreEqual(PyStr("'|'.join(samples)"), string.Join('|', actual));
    }

    [TestMethod, TestCategory("KarpathyShortRun")]
    public void OriginalValueGraph_ThreeTrainingUpdatesAndTwentySamples_NumericalParity()
    {
        var tokenizer = new MicroGpt.Tokenizer(new[] { "ab", "ba" });
        using var model = new MicroGpt.Model(tokenizer.Size, embeddingSize: 4, headCount: 2, blockSize: 4, seed: 5);
        SetReference(model, new[] { 2, 0, 1 }, new[] { 0, 1, 2 });
        KarpathyOriginalSource.Load(Scope, "microgpt");
        PyExec("""
            scalar_weights={name:[[original['Value'](float(z)) for z in row] for row in value] for name,value in w.items()}
            original.update(state_dict=scalar_weights,n_layer=1,n_embd=4,n_head=2,head_dim=2,block_size=4)
            scalar_m={name:np.zeros(value.shape).tolist() for name,value in w.items()}
            scalar_v={name:np.zeros(value.shape).tolist() for name,value in w.items()}
            def source_batch(inputs,targets):
                keys=[[]]; values=[[]]; losses=[]; logits=[]
                for pos,token in enumerate(inputs):
                    row=original['gpt'](int(token),pos,keys,values)
                    logits.append([p.data for p in row])
                    probabilities=original['softmax'](row)
                    losses.append(-probabilities[int(targets[pos])].log())
                loss=(1/len(inputs))*sum(losses)
                loss.backward()
                grads={name:np.array([[p.grad for p in row] for row in value]) for name,value in scalar_weights.items()}
                return loss.data,np.array(logits),grads
            def source_adam(step,rate):
                for name,value in scalar_weights.items():
                    for i,row in enumerate(value):
                        for j,p in enumerate(row):
                            scalar_m[name][i][j]=.85*scalar_m[name][i][j]+(1-.85)*p.grad
                            scalar_v[name][i][j]=.99*scalar_v[name][i][j]+(1-.99)*p.grad**2
                            mh=scalar_m[name][i][j]/(1-.85**step)
                            vh=scalar_v[name][i][j]/(1-.99**step)
                            p.data-=rate*mh/(vh**.5+1e-8)
                            p.grad=0
            source_value_maximum=0.0
            """);
        for (int step = 0; step < 3; step++)
        {
            int[] tokens = step == 1 ? new[] { 2, 1, 0 } : new[] { 2, 0, 1 };
            int[] targets = step == 1 ? new[] { 1, 0, 2 } : new[] { 0, 1, 2 };
            using var batch = model.LossAndGradients(tokens, targets);
            PyExec($"source_loss,source_logits,source_gradients=source_batch([{string.Join(',', tokens)}],[{string.Join(',', targets)}])");
            using var loss = np.array(batch.Loss);
            SourceValues(loss, "np.asarray(source_loss)", $"source loss step {step}");
            SourceValues(batch.Logits, "source_logits", $"source logits step {step}");
            foreach (var (name, gradient) in batch.Parameters)
                SourceValues(gradient, $"source_gradients['{name}']", $"source gradient {name} step {step}");
            double rate = .01 * (1 - step / 3.0);
            model.ApplyAdam(batch, rate);
            PyExec($"source_adam({step + 1},{rate.ToString("R", System.Globalization.CultureInfo.InvariantCulture)})");
            foreach (var (name, parameter) in model.Parameters)
            {
                SourceValues(parameter, $"np.array([[p.data for p in row] for row in scalar_weights['{name}']])", $"source weights {name} step {step}");
                SourceValues(model.FirstMoments[name], $"np.asarray(scalar_m['{name}'])", $"source first moment {name}");
                SourceValues(model.SecondMoments[name], $"np.asarray(scalar_v['{name}'])", $"source second moment {name}");
            }
        }
        PyExec("""
            sample_rng=np.random.RandomState(42); source_samples=[]
            for sample in range(20):
                keys=[[]]; values=[[]]; token=2; chars=[]
                for pos in range(4):
                    logits=original['gpt'](token,pos,keys,values)
                    probabilities=original['softmax']([p/.5 for p in logits])
                    draw=sample_rng.rand(); total=0; token=2
                    for i,p in enumerate(probabilities):
                        total+=p.data
                        if draw<total: token=i; break
                    if token==2: break
                    chars.append('ab'[token])
                source_samples.append(''.join(chars))
            """);
        CollectionAssert.AreEqual(PyStr("'|'.join(source_samples)").Split('|'), model.Generate(tokenizer, samples: 20, seed: 42));
        Console.WriteLine($"KARPATHY_ORIGINAL microGPT: steps=3, samples=20, max_absolute_state_difference={PyFloat("source_value_maximum"):R}; scalar-value contract, not byte identity.");
    }

    [TestMethod, TestCategory("KarpathyByteParity")]
    public void ShortTrainingLedger_EveryLossLogitGradientAndAdamState_ByteExactNumpy()
    {
        using var model = new MicroGpt.Model(4, embeddingSize: 4, headCount: 2, blockSize: 4, seed: 5);
        SetReference(model, new[] { 3, 0, 1 }, new[] { 0, 1, 3 });
        PyExec("w={k:v.copy() for k,v in w.items()}\nm={k:np.zeros_like(v) for k,v in w.items()}\nv={k:np.zeros_like(v) for k,v in w.items()}");
        int checkpoints = 0;
        for (int step = 0; step < 6; step++)
        {
            int[] tokens = step % 2 == 0 ? new[] { 3, 0, 1 } : new[] { 3, 1, 2 };
            int[] targets = step % 2 == 0 ? new[] { 0, 1, 3 } : new[] { 1, 2, 3 };
            PyExec($"tokens=np.array([{string.Join(',', tokens)}])\ntargets=np.array([{string.Join(',', targets)}])\nloss,logits,g=backward(w,tokens,targets,2,1)");
            using var batch = model.LossAndGradients(tokens, targets);
            using var loss = np.array(batch.Loss);
            Exact(loss, "np.asarray(loss)", $"ledger loss {step}"); checkpoints++;
            Exact(batch.Logits, "logits", $"ledger logits {step}"); checkpoints++;
            foreach (var (name, gradient) in batch.Parameters) { Exact(gradient, $"g['{name}']", $"ledger gradient {step}/{name}"); checkpoints++; }
            double rate = .01 * (1 - step / 6.0);
            model.ApplyAdam(batch, rate);
            PyExec($"adam(w,g,m,v,{step + 1},{rate.ToString("R", System.Globalization.CultureInfo.InvariantCulture)})");
            foreach (var (name, parameter) in model.Parameters)
            {
                Exact(parameter, $"w['{name}']", $"ledger weights {step}/{name}");
                Exact(model.FirstMoments[name], $"m['{name}']", $"ledger first moment {step}/{name}");
                Exact(model.SecondMoments[name], $"v['{name}']", $"ledger second moment {step}/{name}");
                checkpoints += 3;
            }
        }
        Assert.AreEqual(228, checkpoints);
        Console.WriteLine($"KARPATHY_BYTES microGPT: updates=6, dtype/shape/byte checkpoints={checkpoints}, relaxed_comparisons=0");
    }

    private void SourceValues(NDArray actual, string expected, string label)
    {
        ExportTo("source_actual", actual);
        PyExec($"source_expected={expected}\nsource_difference=float(np.max(np.abs(source_actual-source_expected)))\nsource_value_maximum=max(source_value_maximum,source_difference)");
        Assert.IsTrue(PyBool("source_actual.shape==source_expected.shape and source_actual.dtype==source_expected.dtype"), label + " dtype/shape");
        // Separate declared numerical gate for vectorized C# versus the ORIGINAL scalar Value graph.
        // This is never used by the byte-level NumPy tier and is not a failed-byte fallback.
        Assert.IsTrue(PyBool("bool(np.all(np.isfinite(source_actual))) and bool(np.all(np.isfinite(source_expected))) and bool(np.allclose(source_actual,source_expected,rtol=1e-9,atol=1e-11))"),
            $"{label}: max absolute difference {PyFloat("source_difference"):R}");
    }

    private void SetReference(MicroGpt.Model model, int[] tokens, int[] targets)
    {
        PyExec(Reference);
        PyExec("w={}");
        foreach (var (name, value) in model.Parameters)
        {
            ExportTo("parameter", value);
            PyExec($"w['{name}']=parameter");
        }
        PyExec($"tokens=np.array([{string.Join(',', tokens)}],dtype=np.int64)\ntargets=np.array([{string.Join(',', targets)}],dtype=np.int64)");
    }
    private void Exact(NDArray actual, string expression, string label)
    {
        using (Gil()) { using var expected = Scope.Eval(expression); GistParity.AssertExact(actual, expected, label); }
    }
}
