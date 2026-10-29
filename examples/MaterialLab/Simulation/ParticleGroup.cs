using System;
using NumSharp;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>The persistent per-particle fields every group stores (index into <see cref="ParticleGroup"/>).</summary>
    public enum PField
    {
        /// <summary>Position x (units).</summary>
        X,
        /// <summary>Position y (units, up).</summary>
        Y,
        /// <summary>Velocity x.</summary>
        VX,
        /// <summary>Velocity y.</summary>
        VY,
        /// <summary>APIC affine velocity matrix C (≈ ∇v), row 0 col 0.</summary>
        C00,
        /// <summary>C, row 0 col 1.</summary>
        C01,
        /// <summary>C, row 1 col 0.</summary>
        C10,
        /// <summary>C, row 1 col 1.</summary>
        C11,
        /// <summary>Elastic deformation gradient F, row 0 col 0 (solids/granular; fluids keep F = I and track <see cref="Jp"/>).</summary>
        F00,
        /// <summary>F, row 0 col 1.</summary>
        F01,
        /// <summary>F, row 1 col 0.</summary>
        F10,
        /// <summary>F, row 1 col 1.</summary>
        F11,
        /// <summary>Scalar plastic state: fluids = volume ratio J, snow = plastic Jp (hardening), others = 1.</summary>
        Jp,
        /// <summary>Uniform random number fixed per particle (render color variation, grain look).</summary>
        Seed,
        /// <summary>Temperature 0..1 (lava: 1 = molten; cools toward rock). Unused (0) elsewhere.</summary>
        Temp,
    }

    /// <summary>
    /// Per-substep scratch fields: written and consumed inside one substep, never preserved across growth,
    /// compaction or sorting (so they cost no copies).
    /// </summary>
    public enum SField
    {
        /// <summary>APIC affine momentum matrix A = stress term + m·C, row 0 col 0 (consumed by P2G).</summary>
        A00,
        /// <summary>A, row 0 col 1.</summary>
        A01,
        /// <summary>A, row 1 col 0.</summary>
        A10,
        /// <summary>A, row 1 col 1.</summary>
        A11,
        /// <summary>Trial deformation gradient (I + dt·C)·F, row 0 col 0.</summary>
        T0,
        /// <summary>Trial F, row 0 col 1.</summary>
        T1,
        /// <summary>Trial F, row 1 col 0.</summary>
        T2,
        /// <summary>Trial F, row 1 col 1.</summary>
        T3,
        /// <summary>σ-space slot 0 (σ1, or the polar cosine for corotated solids).</summary>
        S0,
        /// <summary>σ-space slot 1 (σ2 signed, or the polar sine).</summary>
        S1,
        /// <summary>σ-space slot 2 (cos 2φ of the left singular rotation U).</summary>
        S2,
        /// <summary>σ-space slot 3 (sin 2φ).</summary>
        S3,
        /// <summary>General scratch 0 (eigen gap D, projected log-strain ε1′, …).</summary>
        E0,
        /// <summary>General scratch 1.</summary>
        E1,
        /// <summary>General scratch 2.</summary>
        E2,
        /// <summary>General scratch 3.</summary>
        E3,
    }

    /// <summary>
    /// The particles of ONE material, stored structure-of-arrays as float32 NumSharp arrays with spare
    /// capacity. Every constitutive kernel runs over zero-copy <c>[0:Count]</c> views of these arrays, and
    /// the transfer kernels read/write the same buffers through raw pointers — so NumSharp owns every
    /// byte of particle state and there is no copy between the "array math" and the "loop" halves.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why one group per material.</b> The constitutive models differ per material; keeping each
    /// material contiguous lets each model run as whole-array expressions over exactly its particles
    /// (no masks, no wasted work), and adding/removing particles never shifts other materials.
    /// </para>
    /// <para>
    /// <b>View cache footgun.</b> Views are cached and handed out repeatedly. They are created OUTSIDE any
    /// <see cref="NDScope"/> (a scope would dispose them) and invalidated — disposed and dropped — whenever
    /// <see cref="Count"/> or the backing arrays change. Never hold a view across a call that can change
    /// the group's size.
    /// </para>
    /// </remarks>
    public sealed unsafe class ParticleGroup : IDisposable
    {
        /// <summary>Number of persistent fields.</summary>
        public static readonly int FieldCount = Enum.GetValues<PField>().Length;

        /// <summary>Number of scratch fields.</summary>
        public static readonly int ScratchCount = Enum.GetValues<SField>().Length;

        /// <summary>Capacity-sized buffers of the persistent fields (replaced on growth).</summary>
        private NDArray[] _arrays;
        /// <summary>Cached <c>[0:Count]</c> views of the persistent fields (rebuilt when the count changes).</summary>
        private readonly NDArray[] _views;
        /// <summary>Capacity-sized scratch buffers (constitutive temporaries, the affine matrix).</summary>
        private readonly NDArray[] _scratch;
        /// <summary>Cached <c>[0:Count]</c> views of the scratch buffers.</summary>
        private readonly NDArray[] _sviews;
        /// <summary>The count the cached views were cut for (−1 = none yet).</summary>
        private int _viewCount = -1;
        /// <summary>Per-particle color seeds.</summary>
        private readonly Random _rng;

        /// <summary>The material of every particle in this group.</summary>
        public MaterialDef Material { get; }

        /// <summary>Live particle count.</summary>
        public int Count { get; private set; }

        /// <summary>Allocated capacity (particles).</summary>
        public int Capacity { get; private set; }

        /// <summary>Incremented whenever the backing arrays are replaced (growth) — lets holders of raw pointers notice.</summary>
        public int Generation { get; private set; }

        /// <summary>Creates an empty group.</summary>
        /// <param name="material">The material.</param>
        /// <param name="seed">Seed for the per-particle random seeds (deterministic scenes).</param>
        public ParticleGroup(MaterialDef material, int seed)
        {
            Material = material;
            _arrays = new NDArray[FieldCount];
            _views = new NDArray[FieldCount];
            _scratch = new NDArray[ScratchCount];
            _sviews = new NDArray[ScratchCount];
            _rng = new Random(seed * 7919 + (int)material.Id);
        }

        /// <summary>A zero-copy <c>[0:Count]</c> view of one scratch field (same caching rules as the persistent views).</summary>
        /// <param name="s">The scratch field.</param>
        /// <returns>A float32 view of length <see cref="Count"/>.</returns>
        /// <exception cref="InvalidOperationException">The group is empty.</exception>
        public NDArray this[SField s]
        {
            get
            {
                if (Count == 0) throw new InvalidOperationException($"{Material.Name} group is empty — nothing to view.");
                if (_viewCount != Count) RefreshViews();
                return _sviews[(int)s];
            }
        }

        /// <summary>Raw pointer to element 0 of a scratch field (valid until the next growth).</summary>
        /// <param name="s">The scratch field.</param>
        /// <returns>Pointer to the first float.</returns>
        public float* Ptr(SField s) => (float*)_scratch[(int)s].Unsafe.Pointer<float>();

        /// <summary>
        /// A zero-copy view of the live part (<c>[0:Count]</c>) of one field. Cached; see the class remarks
        /// for its lifetime rule.
        /// </summary>
        /// <param name="f">The field.</param>
        /// <returns>A float32 view of length <see cref="Count"/>.</returns>
        /// <exception cref="InvalidOperationException">The group is empty (there is no live range to view).</exception>
        public NDArray this[PField f]
        {
            get
            {
                if (Count == 0) throw new InvalidOperationException($"{Material.Name} group is empty — nothing to view.");
                if (_viewCount != Count) RefreshViews();
                return _views[(int)f];
            }
        }

        /// <summary>
        /// Raw pointer to element 0 of a field's backing array — valid until the next growth
        /// (<see cref="Generation"/> change). The transfer kernels use this; the constitutive kernels use views.
        /// </summary>
        /// <param name="f">The field.</param>
        /// <returns>Pointer to the first float.</returns>
        public float* Ptr(PField f) => (float*)_arrays[(int)f].Unsafe.Pointer<float>();

        /// <summary>The full-capacity backing array of a field (for the spatial-sort permutation).</summary>
        /// <param name="f">The field.</param>
        /// <returns>The array (length = <see cref="Capacity"/>).</returns>
        internal NDArray Backing(PField f) => _arrays[(int)f];

        /// <summary>Replaces a field's backing array (the spatial sort writes the permuted copy, then swaps it in).</summary>
        /// <param name="f">The field.</param>
        /// <param name="replacement">A float32 array of length <see cref="Capacity"/>.</param>
        /// <returns>The array that was replaced (the caller reuses it as the next scratch).</returns>
        internal NDArray SwapBacking(PField f, NDArray replacement)
        {
            var old = _arrays[(int)f];
            _arrays[(int)f] = replacement;
            InvalidateViews();
            Generation++;
            return old;
        }

        /// <summary>Makes room for at least <paramref name="needed"/> particles (grows by 1.5× to amortize), preserving live data.</summary>
        /// <param name="needed">Required capacity.</param>
        public void EnsureCapacity(int needed)
        {
            if (needed <= Capacity) return;
            int cap = Math.Max(needed, Math.Max(1024, (int)(Capacity * 1.5)));
            InvalidateViews();
            for (int f = 0; f < FieldCount; f++)
            {
                // Fresh C-contiguous arrays sized from dimensions — never from another array's Shape (a view's
                // strides would be kept while only `size` elements are allocated).
                var fresh = np.zeros(new Shape(cap), NPTypeCode.Single);
                var old = _arrays[f];
                if (old is not null)
                {
                    if (Count > 0)
                        Buffer.MemoryCopy(old.Unsafe.Pointer<float>(), fresh.Unsafe.Pointer<float>(), (long)cap * 4, (long)Count * 4);
                    old.Dispose();
                }
                _arrays[f] = fresh;
            }
            // Scratch holds nothing across substeps, so it is reallocated without copying.
            for (int s = 0; s < ScratchCount; s++)
            {
                _scratch[s]?.Dispose();
                _scratch[s] = np.zeros(new Shape(cap), NPTypeCode.Single);
            }
            Capacity = cap;
            Generation++;
        }

        /// <summary>
        /// Appends particles at the given positions with a common initial velocity; the continuum state starts
        /// undeformed (F = I, C = 0, J = 1) and lava starts molten (Temp = 1).
        /// </summary>
        /// <param name="xs">X positions.</param>
        /// <param name="ys">Y positions (same length as <paramref name="xs"/>).</param>
        /// <param name="vx">Initial velocity x.</param>
        /// <param name="vy">Initial velocity y.</param>
        /// <returns>The number of particles appended.</returns>
        /// <exception cref="ArgumentException">The position spans differ in length.</exception>
        public int Append(ReadOnlySpan<float> xs, ReadOnlySpan<float> ys, float vx, float vy)
        {
            if (xs.Length != ys.Length) throw new ArgumentException("xs and ys must have the same length");
            int k = xs.Length;
            if (k == 0) return 0;
            EnsureCapacity(Count + k);
            int start = Count;
            float temp = Material.Id == MaterialId.Lava ? 1f : 0f;
            float* X = Ptr(PField.X), Y = Ptr(PField.Y), VX = Ptr(PField.VX), VY = Ptr(PField.VY);
            float* C00 = Ptr(PField.C00), C01 = Ptr(PField.C01), C10 = Ptr(PField.C10), C11 = Ptr(PField.C11);
            float* F00 = Ptr(PField.F00), F01 = Ptr(PField.F01), F10 = Ptr(PField.F10), F11 = Ptr(PField.F11);
            float* Jp = Ptr(PField.Jp), Seed = Ptr(PField.Seed), Temp = Ptr(PField.Temp);
            for (int i = 0; i < k; i++)
            {
                int p = start + i;
                X[p] = xs[i]; Y[p] = ys[i]; VX[p] = vx; VY[p] = vy;
                C00[p] = C01[p] = C10[p] = C11[p] = 0f;
                F00[p] = 1f; F01[p] = 0f; F10[p] = 0f; F11[p] = 1f;
                Jp[p] = 1f;
                Seed[p] = (float)_rng.NextDouble();
                Temp[p] = temp;
            }
            Count += k;
            InvalidateViews();
            return k;
        }

        /// <summary>
        /// Appends particles copied from another group's particle <paramref name="src"/> (used for lava → rock:
        /// position, velocity and C carry over; the new material starts undeformed).
        /// </summary>
        /// <param name="from">Source group.</param>
        /// <param name="src">Source particle index.</param>
        internal void AppendFrom(ParticleGroup from, int src)
        {
            EnsureCapacity(Count + 1);
            int p = Count;
            foreach (var f in new[] { PField.X, PField.Y, PField.VX, PField.VY, PField.C00, PField.C01, PField.C10, PField.C11, PField.Seed })
                Ptr(f)[p] = from.Ptr(f)[src];
            Ptr(PField.F00)[p] = 1f; Ptr(PField.F01)[p] = 0f; Ptr(PField.F10)[p] = 0f; Ptr(PField.F11)[p] = 1f;
            Ptr(PField.Jp)[p] = 1f;
            Ptr(PField.Temp)[p] = 0f;
            Count++;
            InvalidateViews();
        }

        /// <summary>
        /// Removes every particle for which <paramref name="remove"/> is true, compacting all fields in place
        /// (stable order, no allocation).
        /// </summary>
        /// <param name="remove">Predicate on (x, y, particle index).</param>
        /// <returns>The number of particles removed.</returns>
        public int RemoveWhere(Func<float, float, int, bool> remove)
        {
            if (Count == 0) return 0;
            float* X = Ptr(PField.X), Y = Ptr(PField.Y);
            var fields = new float*[FieldCount];
            for (int f = 0; f < FieldCount; f++) fields[f] = Ptr((PField)f);
            int w = 0;
            for (int r = 0; r < Count; r++)
            {
                if (remove(X[r], Y[r], r)) continue;
                if (w != r)
                    for (int f = 0; f < FieldCount; f++) fields[f][w] = fields[f][r];
                w++;
            }
            int removed = Count - w;
            if (removed > 0) { Count = w; InvalidateViews(); }
            return removed;
        }

        /// <summary>Removes every particle (capacity is kept).</summary>
        public void Clear()
        {
            Count = 0;
            InvalidateViews();
        }

        /// <summary>Rebuilds the cached <c>[0:Count]</c> views.</summary>
        private void RefreshViews()
        {
            InvalidateViews();
            string range = $"0:{Count}";
            for (int f = 0; f < FieldCount; f++)
                _views[f] = _arrays[f][range];
            for (int s = 0; s < ScratchCount; s++)
                _sviews[s] = _scratch[s][range];
            _viewCount = Count;
        }

        /// <summary>
        /// Drops the cached views. They are deliberately NOT disposed: a view only holds a reference to the
        /// backing buffer, and disposing one is only safe if slicing never hands back the base array itself —
        /// an invariant this class has no reason to bet particle state on. Dropped views are reclaimed by the
        /// finalizer, which abandons (never frees) a shared buffer.
        /// </summary>
        private void InvalidateViews()
        {
            Array.Clear(_views);
            Array.Clear(_sviews);
            _viewCount = -1;
        }

        /// <summary>
        /// Builds the cached views now (outside any <see cref="NDScope"/>), so the substep — which runs inside
        /// a scope — only ever reads views that the scope did not create and will not dispose.
        /// </summary>
        public void PrepareViews()
        {
            if (Count > 0 && _viewCount != Count) RefreshViews();
        }

        /// <summary>Releases every array.</summary>
        public void Dispose()
        {
            InvalidateViews();
            for (int f = 0; f < FieldCount; f++) { _arrays[f]?.Dispose(); _arrays[f] = null; }
            for (int s = 0; s < ScratchCount; s++) { _scratch[s]?.Dispose(); _scratch[s] = null; }
            Count = Capacity = 0;
        }
    }
}
