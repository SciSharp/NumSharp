<div class="ns-home">
  <section class="ns-home-intro">
    <img class="ns-home-logo" src="images/numsharp.icon.svg" alt="NumSharp logo">
    <p class="ns-home-kicker">NumPy-shaped arrays for .NET</p>
    <h1>NumSharp</h1>
    <p class="ns-home-lede">
      A .NET port of NumPy focused on API parity, unmanaged NDArray storage,
      view semantics, broadcasting, and runtime-generated SIMD kernels.
    </p>
    <div class="ns-home-actions" aria-label="Primary documentation links">
      <a class="ns-home-button ns-home-button-primary" href="docs/intro.md">Start with NDArray</a>
      <a class="ns-home-button" href="docs/benchmarks-dashboard.md">View benchmarks</a>
      <a class="ns-home-button" href="api/index.md">API reference</a>
    </div>
  </section>

  <section class="ns-home-code" aria-label="Quick start example">
    <div class="ns-home-code-head">
      <h2>Install and run</h2>
      <p>Start from familiar NumPy-style calls in ordinary C#.</p>
    </div>
    <div class="ns-code-grid">
      <pre><code class="lang-bash">dotnet add package NumSharp</code></pre>
      <pre><code class="lang-csharp">using NumSharp;
var a = np.array(new[] { 1, 2, 3, 4, 5 });
var b = np.arange(5);
Console.WriteLine(a + b);  // [1 3 5 7 9]</code></pre>
    </div>
  </section>

  <section class="ns-home-stats" aria-label="Project highlights">
    <div>
      <strong>NumPy 2.x</strong>
      <span>Compatibility target</span>
    </div>
    <div>
      <strong>15</strong>
      <span>Core dtypes</span>
    </div>
    <div>
      <strong>V128-V512</strong>
      <span>Runtime SIMD width</span>
    </div>
    <div>
      <strong>Views</strong>
      <span>Slicing shares memory</span>
    </div>
  </section>

  <section class="ns-home-section">
    <div class="ns-home-section-head">
      <h2>Documentation Map</h2>
      <p>Pick the part of the stack you are working on.</p>
    </div>
    <div class="ns-card-grid">
      <a class="ns-doc-card" href="docs/intro.md">
        <span>Arrays</span>
        <strong>NDArray fundamentals</strong>
        <p>Shape, strides, storage, indexing, and the mental model for NumSharp arrays.</p>
      </a>
      <a class="ns-doc-card" href="docs/broadcasting.md">
        <span>Semantics</span>
        <strong>Broadcasting</strong>
        <p>How NumSharp stretches operands without materializing repeated data.</p>
      </a>
      <a class="ns-doc-card" href="docs/NDIter.md">
        <span>Kernels</span>
        <strong>NDIter</strong>
        <p>The iterator layer that schedules strided, buffered, and fused inner loops.</p>
      </a>
      <a class="ns-doc-card" href="docs/il-generation.md">
        <span>Runtime compiler</span>
        <strong>IL Generation</strong>
        <p>DynamicMethod kernels, SIMD paths, cache families, and generator ownership.</p>
      </a>
      <a class="ns-doc-card" href="docs/compliance.md">
        <span>Parity</span>
        <strong>NumPy compliance</strong>
        <p>Compatibility notes, behavioral scope, and areas still being tightened.</p>
      </a>
      <a class="ns-doc-card" href="docs/benchmarks-dashboard.md">
        <span>Performance</span>
        <strong>Benchmarks</strong>
        <p>Current NumSharp-vs-NumPy results with drill-down reports by subsystem.</p>
      </a>
    </div>
  </section>

  <section class="ns-home-section">
    <div class="ns-home-section-head">
      <h2>What NumSharp Optimizes For</h2>
      <p>Behavior first, with fast paths where the layout and dtype make them possible.</p>
    </div>
    <div class="ns-feature-list">
      <div>
        <strong>NumPy-shaped API</strong>
        <p>Creation, indexing, broadcasting, math, reductions, random sampling, and file I/O use familiar NumPy names and behavior.</p>
      </div>
      <div>
        <strong>Unmanaged storage</strong>
        <p>Arrays use raw storage behind shape and stride metadata, so views and kernels can operate close to the metal.</p>
      </div>
      <div>
        <strong>Runtime specialization</strong>
        <p>Elementwise, cast, reduction, scan, and selection paths emit dtype-specific IL and cache the compiled delegates.</p>
      </div>
      <div>
        <strong>Release-tracked performance</strong>
        <p>The benchmark dashboard publishes stable history snapshots, not one-off scratch output.</p>
      </div>
    </div>
  </section>

  <section class="ns-home-links" aria-label="Community links">
    <a href="https://github.com/SciSharp/NumSharp">GitHub Repository</a>
    <a href="https://www.nuget.org/packages/NumSharp">NuGet Package</a>
  </section>
</div>
