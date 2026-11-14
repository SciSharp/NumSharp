---
_layout: landing
title: NumSharp, NumPy for .NET
description: NumSharp brings NumPy's API to .NET - the same functions, dtypes, broadcasting and views, tested against NumPy's own output - and hands its arrays to Python, PyTorch, ONNX Runtime, ML.NET and TorchSharp.
---

<div class="ns-home">
<section class="ns-hero ns-wrap" aria-labelledby="ns-hero-title">
<p class="ns-brand"><img src="images/numsharp.icon.svg" alt="">NumSharp</p>
<h1 id="ns-hero-title"><span>Write NumPy in&nbsp;C#.</span></h1>
<div class="ns-hero-sub">
<p class="ns-lede">NumSharp brings NumPy's API to .NET: the same functions, dtypes, broadcasting and views, tested against NumPy's own output. Its arrays cross into Python, PyTorch, ONNX Runtime and TorchSharp, most of them without a copy.</p>
<div class="ns-actions">
<div class="ns-install"><code>dotnet add package NumSharp</code><button type="button" class="ns-copy" data-ns-copy="dotnet add package NumSharp">Copy</button></div>
<a class="ns-btn ns-btn-primary" href="docs/absolute-basics.md">Start with the basics</a>
<a class="ns-btn" href="docs/quickstart.md">Quickstart</a>
</div>
</div>
<div class="ns-editor" data-ns-editor>
<p class="ns-ed-caption" data-ns-caption aria-live="polite">Same names, same slicing, same output.</p>
<div class="ns-ed-stage">
<button type="button" class="ns-crazier" data-ns-crazier hidden><span class="ns-crazier-label" data-ns-crazier-label>Crazier</span><span class="ns-crazier-meter" data-ns-meter aria-hidden="true"></span></button>
<div class="ns-ed-frame" role="table" aria-label="The same program in Python and in C#">
<div class="ns-ed-tabs" role="row"><div role="columnheader"><span class="ns-ed-dot ns-ed-dot-py"></span>example.py<small>Python with NumPy</small></div><div role="columnheader"><span class="ns-ed-dot ns-ed-dot-cs"></span>Program.cs<small>C# with NumSharp</small></div></div>
<div class="ns-ed-body" data-ns-body>
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">a = np.arange(12).reshape(3, 4)</code><code role="cell">var a = np.arange(12).reshape(3, 4);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">v = a[1:, ::2]  # a view, no copy</code><code role="cell">var v = a["1:, ::2"];  // a view, no copy</code></div>
<div class="ns-ed-row" role="row"><code role="cell">v[0, 0] = 100   # writes through to a</code><code role="cell">v[0, 0] = 100;         // writes through to a</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(a.sum(axis=0))</code><code role="cell">Console.WriteLine(a.sum(axis: 0));</code></div>
</div>
<div class="ns-ed-run" role="row"><div role="cell"><span class="ns-ed-cmd">&gt; python example.py</span><pre data-ns-out>[108  15  18  21]</pre></div><div role="cell"><span class="ns-ed-cmd">&gt; dotnet run</span><pre data-ns-out>[108  15  18  21]</pre></div></div>
</div>
</div>
<template data-ns-example data-title="A multiplication table by broadcasting">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">row = np.arange(1, 10)</code><code role="cell">var row = np.arange(1, 10);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">table = row[:, np.newaxis] * row</code><code role="cell">var table = row[":, np.newaxis"] * row;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(table[6:, 6:])</code><code role="cell">Console.WriteLine(table["6:, 6:"]);</code></div>
<pre data-ns-out>[[49 56 63]
 [56 64 72]
 [63 72 81]]</pre>
</template>
<template data-ns-example data-title="Rank names by score with argsort">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">names = np.array(["Ada", "Bo", "Cy", "Di", "Ed"])</code><code role="cell">var names = np.array(["Ada", "Bo", "Cy", "Di", "Ed"]);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">scores = np.array([88, 92, 79, 95, 60])</code><code role="cell">var scores = np.array([88, 92, 79, 95, 60]);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(names[np.argsort(-scores)][:3])</code><code role="cell">Console.WriteLine(names[np.argsort(-scores)][":3"]);</code></div>
<pre data-ns-out>['Di' 'Bo' 'Ada']</pre>
</template>
<template data-ns-example data-title="FizzBuzz without a loop">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">n = np.arange(1, 16)</code><code role="cell">var n = np.arange(1, 16);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">fb = n.astype(str)</code><code role="cell">var fb = n.astype(np.str_);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">fb = np.where(n % 3 == 0, "Fizz", fb)</code><code role="cell">fb = np.where(n % 3 == 0, "Fizz", fb);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">fb = np.where(n % 5 == 0, "Buzz", fb)</code><code role="cell">fb = np.where(n % 5 == 0, "Buzz", fb);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">fb = np.where(n % 15 == 0, "FizzBuzz", fb)</code><code role="cell">fb = np.where(n % 15 == 0, "FizzBuzz", fb);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(fb)</code><code role="cell">Console.WriteLine(fb);</code></div>
<pre data-ns-out>['1' '2' 'Fizz' '4' 'Buzz' 'Fizz' '7' '8' 'Fizz' 'Buzz' '11' 'Fizz' '13'
 '14' 'FizzBuzz']</pre>
</template>
<template data-ns-example data-title="Every pairwise distance from one einsum">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">pts = np.array([[0, 0], [3, 4], [6, 8]])</code><code role="cell">var pts = np.array(new[,] { { 0, 0 }, { 3, 4 }, { 6, 8 } });</code></div>
<div class="ns-ed-row" role="row"><code role="cell">d = pts[:, np.newaxis] - pts</code><code role="cell">var d = pts[":, np.newaxis"] - pts;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">dist = np.sqrt(np.einsum("ijk,ijk-&gt;ij", d, d))</code><code role="cell">var dist = np.sqrt(np.einsum("ijk,ijk-&gt;ij", d, d));</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(dist)</code><code role="cell">Console.WriteLine(dist);</code></div>
<pre data-ns-out>[[ 0.  5. 10.]
 [ 5.  0.  5.]
 [10.  5.  0.]]</pre>
</template>
<template data-ns-example data-title="Seed 42 draws NumPy's exact numbers">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">np.random.seed(42)</code><code role="cell">np.random.seed(42);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">hands = np.array(["rock", "paper", "scissors"])</code><code role="cell">var hands = np.array(["rock", "paper", "scissors"]);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(np.random.randint(0, 100, size=8))</code><code role="cell">Console.WriteLine(np.random.randint(0, 100, size: 8));</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(np.random.choice(hands, 4))</code><code role="cell">Console.WriteLine(np.random.choice(hands, 4));</code></div>
<pre data-ns-out>[51 92 14 71 60 20 82 86]
['scissors' 'scissors' 'rock' 'scissors']</pre>
</template>
<template data-ns-example data-title="A million random points, the same π">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">np.random.seed(7)</code><code role="cell">np.random.seed(7);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">xy = np.random.rand(2, 1_000_000)</code><code role="cell">var xy = np.random.rand(2, 1_000_000);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">inside = np.square(xy).sum(axis=0) &lt;= 1</code><code role="cell">var inside = np.square(xy).sum(axis: 0) &lt;= 1;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(4 * inside.mean())</code><code role="cell">Console.WriteLine(4 * inside.mean());</code></div>
<pre data-ns-out>3.139796</pre>
</template>
<template data-ns-example data-title="The long-run weather of a Markov chain">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell"># today sunny or rainy -&gt; odds for tomorrow</code><code role="cell">// today sunny or rainy -&gt; odds for tomorrow</code></div>
<div class="ns-ed-row" role="row"><code role="cell">P = np.array([[0.9, 0.1],</code><code role="cell">var P = np.array(new[,] { { 0.9, 0.1 },</code></div>
<div class="ns-ed-row" role="row"><code role="cell">              [0.5, 0.5]])</code><code role="cell">                          { 0.5, 0.5 } });</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(np.linalg.matrix_power(P, 64)[0])</code><code role="cell">Console.WriteLine(np.linalg.matrix_power(P, 64)[0]);</code></div>
<pre data-ns-out>[0.83333333 0.16666667]</pre>
</template>
<template data-ns-example data-title="Find two tones in a signal with an FFT">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">t = np.arange(512) / 512</code><code role="cell">var t = np.arange(512) / 512;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">wave = (np.sin(2 * np.pi * 50 * t)</code><code role="cell">var wave = np.sin(2 * np.pi * 50 * t)</code></div>
<div class="ns-ed-row" role="row"><code role="cell">        + 0.5 * np.sin(2 * np.pi * 120 * t))</code><code role="cell">         + 0.5 * np.sin(2 * np.pi * 120 * t);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">spectrum = np.abs(np.fft.rfft(wave))</code><code role="cell">var spectrum = np.abs(np.fft.rfft(wave));</code></div>
<div class="ns-ed-row" role="row"><code role="cell">top2 = np.argsort(spectrum)[-2:]</code><code role="cell">var top2 = np.argsort(spectrum)["-2:"];</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(np.sort(top2))</code><code role="cell">Console.WriteLine(np.sort(top2));</code></div>
<pre data-ns-out>[ 50 120]</pre>
</template>
<template data-ns-example data-title="A glider crosses the Game of Life">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">b = np.zeros((6, 6), dtype=int)</code><code role="cell">var b = np.zeros((6, 6), np.int64);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">np.put(b, [1, 8, 12, 13, 14], 1)</code><code role="cell">np.put(b, new[] { 1, 8, 12, 13, 14 }, 1);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">for step in range(4):</code><code role="cell">for (int step = 0; step &lt; 4; step++) {</code></div>
<div class="ns-ed-row" role="row"><code role="cell">    n = -b</code><code role="cell">    var n = -b;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">    for s in np.ndindex(3, 3):</code><code role="cell">    foreach (var s in np.ndindex(3, 3))</code></div>
<div class="ns-ed-row" role="row"><code role="cell">        n += np.roll(b, np.array(s) - 1, (0, 1))</code><code role="cell">        n += np.roll(b, np.array(s) - 1, new[] { 0, 1 });</code></div>
<div class="ns-ed-row" role="row"><code role="cell">    b = np.where((n == 3) | (b &amp; (n == 2)), 1, 0)</code><code role="cell">    b = np.where((n == 3) | (b &amp; (n == 2)), 1, 0);</code></div>
<div class="ns-ed-row" role="row"><code role="cell"></code><code role="cell">}</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(b)</code><code role="cell">Console.WriteLine(b);</code></div>
<pre data-ns-out>[[0 0 0 0 0 0]
 [0 0 1 0 0 0]
 [0 0 0 1 0 0]
 [0 1 1 1 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]]</pre>
</template>
<template data-ns-example data-title="The Mandelbrot set in complex128">
<div class="ns-ed-row" role="row"><code role="cell">import numpy as np</code><code role="cell">using NumSharp; using System.Numerics;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">x = np.linspace(-2.0, 0.5, 26)</code><code role="cell">var x = np.linspace(-2.0, 0.5, 26);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">y = np.linspace(-1.0, 1.0, 9)[:, np.newaxis]</code><code role="cell">var y = np.linspace(-1.0, 1.0, 9)[":, np.newaxis"];</code></div>
<div class="ns-ed-row" role="row"><code role="cell">c = x + 1j * y</code><code role="cell">var c = x + Complex.ImaginaryOne * y;</code></div>
<div class="ns-ed-row" role="row"><code role="cell">z, n = 0 * c, np.zeros(c.shape, dtype=int)</code><code role="cell">var (z, n) = (0 * c, np.zeros(c.shape, np.int64));</code></div>
<div class="ns-ed-row" role="row"><code role="cell">for _ in range(9):</code><code role="cell">for (int i = 0; i &lt; 9; i++) {</code></div>
<div class="ns-ed-row" role="row"><code role="cell">    z = np.where(np.abs(z) &lt;= 2, z * z + c, z)</code><code role="cell">    z = np.where(np.abs(z) &lt;= 2, z * z + c, z);</code></div>
<div class="ns-ed-row" role="row"><code role="cell">    n += np.abs(z) &lt;= 2</code><code role="cell">    n += np.abs(z) &lt;= 2;</code></div>
<div class="ns-ed-row" role="row"><code role="cell"></code><code role="cell">}</code></div>
<div class="ns-ed-row" role="row"><code role="cell">print(n)</code><code role="cell">Console.WriteLine(n);</code></div>
<pre data-ns-out>[[0 0 0 1 1 1 2 2 2 2 2 2 2 2 3 3 3 4 6 8 9 3 3 2 2 1]
 [0 0 1 2 2 2 2 2 2 2 2 3 3 3 4 5 6 8 9 9 9 6 5 4 3 2]
 [0 1 2 2 2 2 2 3 4 4 4 4 5 7 9 9 9 9 9 9 9 9 9 9 6 4]
 [0 2 3 3 3 4 4 6 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 4]
 [9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 6 4]
 [0 2 3 3 3 4 4 6 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 4]
 [0 1 2 2 2 2 2 3 4 4 4 4 5 7 9 9 9 9 9 9 9 9 9 9 6 4]
 [0 0 1 2 2 2 2 2 2 2 2 3 3 3 4 5 6 8 9 9 9 6 5 4 3 2]
 [0 0 0 1 1 1 2 2 2 2 2 2 2 2 3 3 3 4 6 8 9 3 3 2 2 1]]</pre>
</template>
</div>
</section>
<section class="ns-band" aria-labelledby="ns-band-title">
<div class="ns-wrap ns-band-head">
<h2 id="ns-band-title"><span>Where an NDArray can go</span></h2>
<p>Each box names a library and how an NDArray crosses into it; its color names the NumSharp package that brings it. Hover a box to open the library's own site or its page in these docs. <a href="docs/interop/index.md">Read the interoperability guide</a></p>
</div>
<div class="ns-picture">
<a class="ns-picture-link" href="docs/interop/index.md"><img src="images/integrations.svg" width="1324" height="944" alt="NumSharp at the center of its integrations: NumPy, PyTorch, pandas, SciPy, scikit-learn, Matplotlib, Pillow, OpenCV, PyArrow and polars through NumSharp.Interop.pythonnet; ONNX Runtime, ML.NET, System.Numerics.Tensors, TorchSharp, System.Drawing and Numpy.NET in .NET; OpenBLAS; .npy and .npz files. Opens the interoperability guide."></a>
<!-- integrations-links:begin -->
<!-- Generated by tools/gen_integrations_diagram.py from its ITEMS and layout; edit the script, then run it. -->
<div class="ns-hot-layer">
<div class="ns-hot ns-hot-core ns-hot-center" style="left:32.2508%;top:35.1695%;width:34.7432%;height:21.1864%"><a class="ns-hot-box" href="docs/interop/index.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 460 200" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 450,0 460,10 460,190 450,200 10,200 0,190"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://github.com/SciSharp/NumSharp"><span>Open Library</span><span class="visually-hidden"> NumSharp</span></a><a class="ns-hot-btn" href="docs/interop/index.md"><span>Open Docs</span><span class="visually-hidden"> NumSharp</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-top" style="left:17.145%;top:3.3898%;width:14.3505%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/np-frombuffer.md#pillow" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 190 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 180,0 190,10 190,50 180,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://python-pillow.org/"><span>Open Library</span><span class="visually-hidden"> Pillow</span></a><a class="ns-hot-btn" href="docs/interop/np-frombuffer.md#pillow"><span>Open Docs</span><span class="visually-hidden"> Pillow</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-top" style="left:43.5801%;top:3.3898%;width:12.8399%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/np-frombuffer.md#opencv" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 170 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 160,0 170,10 170,50 160,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://opencv.org/"><span>Open Library</span><span class="visually-hidden"> OpenCV</span></a><a class="ns-hot-btn" href="docs/interop/np-frombuffer.md#opencv"><span>Open Docs</span><span class="visually-hidden"> OpenCV</span></a></span></div>
<div class="ns-hot ns-hot-native ns-hot-top" style="left:69.2598%;top:3.3898%;width:12.8399%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/openblas.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 170 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 160,0 170,10 170,50 160,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://www.openblas.net/"><span>Open Library</span><span class="visually-hidden"> OpenBLAS</span></a><a class="ns-hot-btn" href="docs/interop/openblas.md"><span>Open Docs</span><span class="visually-hidden"> OpenBLAS</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-bottom" style="left:17.145%;top:83.8983%;width:14.3505%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/np-frombuffer.md#seven-libraries-verbatim" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 190 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 180,0 190,10 190,50 180,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://pola.rs/"><span>Open Library</span><span class="visually-hidden"> polars</span></a><a class="ns-hot-btn" href="docs/interop/np-frombuffer.md#seven-libraries-verbatim"><span>Open Docs</span><span class="visually-hidden"> polars</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-bottom" style="left:42.8248%;top:83.8983%;width:15.1057%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/np-frombuffer.md#pyarrow" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 200 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 190,0 200,10 200,50 190,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://arrow.apache.org/docs/python/"><span>Open Library</span><span class="visually-hidden"> PyArrow</span></a><a class="ns-hot-btn" href="docs/interop/np-frombuffer.md#pyarrow"><span>Open Docs</span><span class="visually-hidden"> PyArrow</span></a></span></div>
<div class="ns-hot ns-hot-file ns-hot-bottom" style="left:67.7492%;top:83.8983%;width:15.861%;height:6.3559%"><a class="ns-hot-box" href="docs/fundamentals/io.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 210 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 200,0 210,10 210,50 200,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html"><span>Open Library</span><span class="visually-hidden"> .npy / .npz</span></a><a class="ns-hot-btn" href="docs/fundamentals/io.md"><span>Open Docs</span><span class="visually-hidden"> .npy / .npz</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:16.1017%;width:17.3716%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pythonnet-numpy.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 230 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 220,0 230,10 230,50 220,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://numpy.org/"><span>Open Library</span><span class="visually-hidden"> NumPy</span></a><a class="ns-hot-btn" href="docs/interop/pythonnet-numpy.md"><span>Open Docs</span><span class="visually-hidden"> NumPy</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:26.6949%;width:18.1269%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pytorch.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 240 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 230,0 240,10 240,50 230,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://pytorch.org/"><span>Open Library</span><span class="visually-hidden"> PyTorch</span></a><a class="ns-hot-btn" href="docs/interop/pytorch.md"><span>Open Docs</span><span class="visually-hidden"> PyTorch</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:37.2881%;width:15.861%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pandas.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 210 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 200,0 210,10 210,50 200,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://pandas.pydata.org/"><span>Open Library</span><span class="visually-hidden"> pandas</span></a><a class="ns-hot-btn" href="docs/interop/pandas.md"><span>Open Docs</span><span class="visually-hidden"> pandas</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:50%;width:15.1057%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pythonnet-numpy.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 200 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 190,0 200,10 200,50 190,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://scipy.org/"><span>Open Library</span><span class="visually-hidden"> SciPy</span></a><a class="ns-hot-btn" href="docs/interop/pythonnet-numpy.md"><span>Open Docs</span><span class="visually-hidden"> SciPy</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:60.5932%;width:16.6163%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pythonnet-numpy.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 220 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 210,0 220,10 220,50 210,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://scikit-learn.org/"><span>Open Library</span><span class="visually-hidden"> scikit-learn</span></a><a class="ns-hot-btn" href="docs/interop/pythonnet-numpy.md"><span>Open Docs</span><span class="visually-hidden"> scikit-learn</span></a></span></div>
<div class="ns-hot ns-hot-py ns-hot-left" style="left:2.0393%;top:71.1864%;width:17.3716%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/pythonnet-numpy.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 230 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 220,0 230,10 230,50 220,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://matplotlib.org/"><span>Open Library</span><span class="visually-hidden"> Matplotlib</span></a><a class="ns-hot-btn" href="docs/interop/pythonnet-numpy.md"><span>Open Docs</span><span class="visually-hidden"> Matplotlib</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:81.3444%;top:16.1017%;width:16.6163%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/onnxruntime.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 220 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 210,0 220,10 220,50 210,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://onnxruntime.ai/"><span>Open Library</span><span class="visually-hidden"> ONNX Runtime</span></a><a class="ns-hot-btn" href="docs/interop/onnxruntime.md"><span>Open Docs</span><span class="visually-hidden"> ONNX Runtime</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:87.3867%;top:26.6949%;width:10.574%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/mlnet.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 140 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 130,0 140,10 140,50 130,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://dotnet.microsoft.com/apps/ai/ml-dotnet"><span>Open Library</span><span class="visually-hidden"> ML.NET</span></a><a class="ns-hot-btn" href="docs/interop/mlnet.md"><span>Open Docs</span><span class="visually-hidden"> ML.NET</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:71.5257%;top:37.2881%;width:26.435%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/system-numerics-tensors.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 350 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 340,0 350,10 350,50 340,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://www.nuget.org/packages/System.Numerics.Tensors"><span>Open Library</span><span class="visually-hidden"> System.Numerics.Tensors</span></a><a class="ns-hot-btn" href="docs/interop/system-numerics-tensors.md"><span>Open Docs</span><span class="visually-hidden"> System.Numerics.Tensors</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:79.0785%;top:50%;width:18.8822%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/torchsharp.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 250 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 240,0 250,10 250,50 240,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://github.com/dotnet/TorchSharp"><span>Open Library</span><span class="visually-hidden"> TorchSharp</span></a><a class="ns-hot-btn" href="docs/interop/torchsharp.md"><span>Open Docs</span><span class="visually-hidden"> TorchSharp</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:79.8338%;top:60.5932%;width:18.1269%;height:6.3559%"><a class="ns-hot-box" href="docs/extensions/bitmap.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 240 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 230,0 240,10 240,50 230,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://learn.microsoft.com/dotnet/api/system.drawing"><span>Open Library</span><span class="visually-hidden"> System.Drawing</span></a><a class="ns-hot-btn" href="docs/extensions/bitmap.md"><span>Open Docs</span><span class="visually-hidden"> System.Drawing</span></a></span></div>
<div class="ns-hot ns-hot-net ns-hot-right" style="left:83.6103%;top:71.1864%;width:14.3505%;height:6.3559%"><a class="ns-hot-box" href="docs/interop/numpy-net.md" tabindex="-1" aria-hidden="true"><svg viewBox="0 0 190 60" preserveAspectRatio="none" focusable="false"><polygon points="0,10 10,0 180,0 190,10 190,50 180,60 10,60 0,50"/></svg></a><span class="ns-hot-menu"><a class="ns-hot-btn" href="https://github.com/SciSharp/Numpy.NET"><span>Open Library</span><span class="visually-hidden"> Numpy.NET</span></a><a class="ns-hot-btn" href="docs/interop/numpy-net.md"><span>Open Docs</span><span class="visually-hidden"> Numpy.NET</span></a></span></div>
</div>
<!-- integrations-links:end -->
</div>
</section>
<section class="ns-docs ns-wrap" aria-labelledby="ns-docs-title">
<h2 id="ns-docs-title"><span>Documentation</span></h2>
<div class="ns-groups">
<div>
<h3><span>Learn NumSharp</span></h3>
<ul>
<li><a href="docs/absolute-basics.md">The absolute basics</a><p>Arrays, shapes, indexing and math, starting from zero.</p></li>
<li><a href="docs/quickstart.md">Quickstart</a><p>The main features on one page, for readers who know NumPy.</p></li>
<li><a href="docs/NDArray.md">NDArray</a><p>Creation, shape, indexing, views and copies, operators and storage.</p></li>
<li><a href="docs/dtypes.md">Data types</a><p>The 15 numeric dtypes and the text dtypes, with NumPy's names, promotion and casting.</p></li>
<li><a href="docs/fundamentals/strings-and-bytes.md">Strings and bytes</a><p>Fixed-width bytes and Unicode, variable-width StringDType, numpy.char.</p></li>
<li><a href="api/index.md">API reference</a><p>Every public type and member, under NumPy's names.</p></li>
</ul>
</div>
<div>
<h3><span>How it works</span></h3>
<ul>
<li><a href="docs/broadcasting.md">Broadcasting</a><p>How operands of different shapes combine without copying.</p></li>
<li><a href="docs/buffering.md">Memory and buffers</a><p>Unmanaged storage, ownership, lifetimes and zero-copy buffers.</p></li>
<li><a href="docs/NDIter.md">NDIter</a><p>The iterator that runs strided, buffered and fused loops.</p></li>
<li><a href="docs/il-generation.md">IL generation</a><p>SIMD kernels compiled at runtime for each dtype and layout.</p></li>
<li><a href="docs/interop/index.md">Interoperability</a><p>One buffer shared with Python, PyTorch, ONNX Runtime, ML.NET and more.</p></li>
</ul>
</div>
<div>
<h3><span>Measured against NumPy</span></h3>
<ul>
<li><a href="docs/coverage-support-dashboard.md">NumPy API coverage</a><p>Every NumPy function: present, partial or missing, and how closely it matches.</p></li>
<li><a href="docs/tests-oracle-dashboard.md">Unit tests and oracle</a><p>The test inventory and the corpus of NumPy results NumSharp replays.</p></li>
<li><a href="docs/benchmarks-dashboard.md">Benchmarks</a><p>Timings per operation, dtype and size, next to NumPy's.</p></li>
<li><a href="docs/compliance.md">NumPy compliance</a><p>What matches NumPy, and the documented differences.</p></li>
</ul>
</div>
</div>
<p class="ns-elsewhere"><a href="https://github.com/SciSharp/NumSharp">Source on GitHub</a><a href="https://www.nuget.org/packages/NumSharp">Package on NuGet</a></p>
</section>
</div>
