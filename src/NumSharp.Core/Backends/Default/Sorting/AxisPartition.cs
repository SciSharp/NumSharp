using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends.Sorting
{
    /// <summary>
    /// Along-axis partition / argpartition, structured exactly like NumPy's
    /// <c>PyArray_Partition</c> / <c>PyArray_ArgPartition</c> (item_selection.c): the SAME
    /// <see cref="AxisSort.DriveAllButAxis"/> NDIter drive hands one 1-D line per call to an
    /// introselect kernel (<see cref="QuickSelect"/> — the primitive already backing
    /// median/percentile/quantile), around the caller's pre-validated, pre-sorted kth list.
    ///
    /// Validation is the port of <c>partition_prep_kth_array</c> + <c>PyArray_Partition</c>'s own
    /// checks, in NumPy's probed order: kind → order → axis → writeable → kth (kth bounds are
    /// SKIPPED for a size-0 array, negative kth wraps once and the error reports the post-wrap
    /// value, and the kth list is sorted so successive partitions cannot trample each other).
    ///
    /// Float lines NaN-compact first (NumPy's LT sorts any NaN last; original NaN bit patterns are
    /// preserved in the tail, in encounter order — the same policy as <see cref="AxisSort"/>'s
    /// argsort NaN tail); Complex lines use the exact NumPy CDOUBLE_LT comparator
    /// (<see cref="AxisSort.ComplexCmp"/>), whose NaN ordering is NOT a compactable
    /// "any-NaN-part last" — (1, NaN) orders before (NaN, 1).
    /// </summary>
    internal static unsafe class AxisPartition
    {
        // ============================ public entry points ============================

        /// <summary>np.partition: fresh C-contiguous partitioned copy (axis=null flattens first).</summary>
        public static NDArray Partition(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                PartitionInPlaceCore(flat, 0, WidenKth(kth));
                return flat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            // NumPy partition = copy(order='K') + in-place (fromnumeric.partition verbatim):
            // an F input keeps F layout, strided/negative come back C (probed 2.4.2).
            var res = a.copy('K');
            PartitionInPlaceCore(res, ax, WidenKth(kth));
            return res;
        }

        /// <summary>np.partition with an NDArray kth — NumPy's array form, which is what makes its
        /// kth-dtype rejections REACHABLE (a typed C# int[] cannot be bool/float/2-D). Validation is
        /// NumPy's probed STAGING: kind → order → too-deep (before axis!) → axis → bool/integer dtype
        /// → wrap/bounds — a bool kth with a bad axis reports the AXIS, a 2-D kth reports too-deep.</summary>
        public static NDArray Partition(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            if (axis == null)
            {
                var flat = a.ravel().copy('C');
                PartitionInPlaceCore(flat, 0, ExtractKthValues(kth));
                return flat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            // NumPy partition = copy(order='K') + in-place — same rule as the scalar-kth overload.
            var res = a.copy('K');
            PartitionInPlaceCore(res, ax, ExtractKthValues(kth));
            return res;
        }

        /// <summary>ndarray.partition: partitions <paramref name="a"/> in place along the axis
        /// (null flattens in place — the NumSharp extension ndarray.sort already carries).</summary>
        public static void PartitionInPlace(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            if (axis == null)
            {
                // In-place flatten-partition only well-defined for contiguous (ndarray.sort house rule).
                if (!a.Shape.IsWriteable)
                    throw new ValueError("partition array is read-only");
                PartitionInPlaceCore(a.reshape(a.size), 0, WidenKth(kth));
                return;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            if (!a.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            PartitionInPlaceCore(a, ax, WidenKth(kth));
        }

        /// <summary>ndarray.partition with an NDArray kth (staged validation — see the np.partition form).</summary>
        public static void PartitionInPlace(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            if (axis == null)
            {
                if (!a.Shape.IsWriteable)
                    throw new ValueError("partition array is read-only");
                PartitionInPlaceCore(a.reshape(a.size), 0, ExtractKthValues(kth));
                return;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            if (!a.Shape.IsWriteable)
                throw new ValueError("partition array is read-only");
            PartitionInPlaceCore(a, ax, ExtractKthValues(kth));
        }

        /// <summary>np.argpartition: int64 indices that would partition <paramref name="a"/>
        /// (axis=null flattens; the input is only read). A 0-d input ravels to (1,) first —
        /// NumPy's arg-side PyArray_CheckAxis quirk, so kth/axis errors report dimension 1.</summary>
        public static NDArray ArgPartition(NDArray a, int[] kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            if (a.ndim == 0)
                a = a.reshape(1);
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, new Shape((int)a.size), false);
                ArgPartitionInto(flat, outFlat, 0, WidenKth(kth));
                return outFlat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, new Shape((long[])a.Shape.dimensions.Clone()), false);
            ArgPartitionInto(src, ret, ax, WidenKth(kth));
            return ret;
        }

        /// <summary>np.argpartition with an NDArray kth (staged validation — see the np.partition form).</summary>
        public static NDArray ArgPartition(NDArray a, NDArray kth, int? axis, string kind, string order)
        {
            ValidateKindOrder(kind, order);
            KthTooDeepCheck(kth);
            if (a.ndim == 0)
                a = a.reshape(1);
            if (axis == null)
            {
                var flat = a.Shape.IsContiguous ? a.reshape(a.size) : a.ravel().copy('C');
                var outFlat = new NDArray(NPTypeCode.Int64, new Shape((int)a.size), false);
                ArgPartitionInto(flat, outFlat, 0, ExtractKthValues(kth));
                return outFlat;
            }
            int ax = AxisSort.NormalizeAxis(axis.Value, a.ndim);
            var src = a.Shape.IsContiguous ? a : a.copy('C');
            var ret = new NDArray(NPTypeCode.Int64, new Shape((long[])a.Shape.dimensions.Clone()), false);
            ArgPartitionInto(src, ret, ax, ExtractKthValues(kth));
            return ret;
        }

        // ============================ validation (NumPy's order) ============================

        private static void ValidateKindOrder(string kind, string order)
        {
            // NumPy converts `kind` while parsing arguments, so it fires before order/axis/kth.
            if (kind is not null && kind != "introselect")
                throw new ValueError($"select kind must be 'introselect' (got '{kind}')");
            // `order` exists for structured dtypes only, which NumSharp does not have.
            if (order is not null)
                throw new ValueError("Cannot specify order when the array has no fields.");
        }

        /// <summary>Widen an int[] kth to the long[] the cores take (NumPy's kth is intp).
        /// Null rides through so the ArgumentNullException still fires at the kth-validation stage.</summary>
        private static long[] WidenKth(int[] kth)
        {
            if (kth is null)
                return null;
            var k64 = new long[kth.Length];
            for (int i = 0; i < kth.Length; i++)
                k64[i] = kth[i];
            return k64;
        }

        /// <summary>The pre-axis stage of NumPy's array-kth conversion: a &gt;1-D kth is rejected
        /// with the verbatim conversion error BEFORE the axis is even looked at (probed:
        /// <c>np.partition(a, [[1]], axis=9)</c> reports too-deep, not the axis).</summary>
        private static void KthTooDeepCheck(NDArray kth)
        {
            if (kth is null)
                throw new ArgumentNullException(nameof(kth));
            if (kth.ndim > 1)
                throw new ValueError("object too deep for desired array");
        }

        /// <summary>The kth-stage of NumPy's <c>partition_prep_kth_array</c> for an array kth:
        /// bool rejects first ("Booleans unacceptable as partition index"), any non-integer dtype
        /// next ("Partition index must be integer" — TypeError), then the values cast to intp with
        /// NumPy's MODULAR wrap (a uint64 2^63 becomes a negative kth and its bounds error reports
        /// the wrapped value — probed: <c>kth(=-9223372036854775804) out of bounds (4)</c>, while
        /// uint64 2^64-1 wraps to -1 and is a LEGAL last-element kth). Char (no NumPy dtype) is
        /// integer-family, so it rides the same route (house call). Any layout/0-d is normalized
        /// by the astype+ravel.</summary>
        private static long[] ExtractKthValues(NDArray kth)
        {
            if (kth.typecode == NPTypeCode.Boolean)
                throw new ValueError("Booleans unacceptable as partition index");
            switch (kth.typecode)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    break;
                default:
                    throw new TypeError("Partition index must be integer");
            }

            var k64 = (kth.typecode == NPTypeCode.Int64 ? kth : kth.astype(NPTypeCode.Int64)).ravel();
            var result = new long[k64.size];
            for (int i = 0; i < result.Length; i++)
                result[i] = k64.GetValue<long>(i);
            return result;
        }

        /// <summary>
        ///     Port of <c>partition_prep_kth_array</c>: wrap negatives once, bounds-check against
        ///     the axis length UNLESS the array is empty (NumPy's <c>PyArray_SIZE(op) != 0</c> guard),
        ///     report the post-wrap value verbatim, and sort ascending so the partitions compose.
        ///     An empty kth is NumPy's <c>np.array([], dtype=intp)</c> — a valid no-op (Python's
        ///     bare <c>[]</c> is float64 and raises; a typed C# int[] carries no such ambiguity).
        /// </summary>
        private static int[] PrepKth(long[] kth, long axisLength, long size)
        {
            // NumPy's np.partition(a, None) is a TypeError; a null C# array is the same caller bug
            // (house ArgumentNullException precedent: fill_diagonal/lexsort). Guarded HERE so it
            // fires at NumPy's kth-validation stage — after kind/order/axis, like every kth error.
            if (kth is null)
                throw new ArgumentNullException(nameof(kth));
            var ks = new int[kth.Length];
            for (int i = 0; i < kth.Length; i++)
            {
                long k = kth[i];
                if (k < 0)
                    k += axisLength;
                if (size != 0 && (k < 0 || k >= axisLength))
                    throw new ValueError($"kth(={k}) out of bounds ({axisLength})");
                ks[i] = (int)k;
            }
            if (ks.Length > 1)
                Array.Sort(ks);
            return ks;
        }

        // ============================ drive ============================

        private static void PartitionInPlaceCore(NDArray target, int axis, long[] kth)
        {
            long N = target.shape[axis];
            var ks = PrepKth(kth, N, target.size);
            if (target.size == 0 || N <= 1 || ks.Length == 0)
                return;

            var tc = target.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new PartCtx
            {
                n = (int)N,
                inStride = (long)target.Shape.strides[axis] * elsize,
                outStride = 0,
                nKth = ks.Length,
            };

            // per-call scratch, reused across every line (the AxisSort buffer policy): the value
            // gather buffer always, the NaN stash only for the float dtypes that compact.
            var scratch = new byte[N * elsize];
            var nans = NeedsNanTail(tc) ? new byte[N * elsize] : Array.Empty<byte>();
            fixed (byte* ps = scratch, pn = nans)
            fixed (int* pk = ks)
            {
                ctx.scratch = ps;
                ctx.nanTail = pn;
                ctx.kth = pk;
                NDInnerLoopFunc kern = GetPartitionKernel(tc);
                AxisSort.DriveAllButAxis(new[] { target }, new[] { NDIterPerOpFlags.READWRITE }, axis, kern, &ctx);
            }
        }

        private static void ArgPartitionInto(NDArray src, NDArray dst, int axis, long[] kth)
        {
            long N = src.shape[axis];
            var ks = PrepKth(kth, N, src.size);
            if (src.size == 0)
                return;

            var tc = src.GetTypeCode;
            int elsize = tc.SizeOf();
            var ctx = new PartCtx
            {
                n = (int)N,
                inStride = (long)src.Shape.strides[axis] * elsize,
                outStride = (long)dst.Shape.strides[axis] * sizeof(long),
                nKth = ks.Length,
            };

            // argpartition always gathers (the source is read-only); idx is the answer column.
            var scratch = new byte[N * elsize];
            var idx = new long[N];
            fixed (byte* ps = scratch)
            fixed (long* pi = idx)
            fixed (int* pk = ks)
            {
                ctx.scratch = ps;
                ctx.idx = pi;
                ctx.kth = pk;
                NDInnerLoopFunc kern = GetArgPartitionKernel(tc);
                AxisSort.DriveAllButAxis(new[] { src, dst },
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }, axis, kern, &ctx);
            }
        }

        private static bool NeedsNanTail(NPTypeCode tc)
            => tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;

        // line-partition context carried via NDIter auxdata (no per-call captures/allocations)
        private struct PartCtx
        {
            public byte* scratch;   // N-element value gather buffer
            public byte* nanTail;   // N-element NaN stash (float dtypes only)
            public long* idx;       // N-element original-index column (argpartition only)
            public int* kth;        // wrapped, validated, sorted ascending
            public int nKth;
            public long inStride;   // byte stride along the partition axis
            public long outStride;  // byte stride of the int64 output line (argpartition only)
            public int n;
        }

        // NumPy CDOUBLE_LT as a Comparison<Complex> for the QuickSelect comparator path.
        private static readonly Comparison<Complex> ComplexLess =
            static (a, b) => default(AxisSort.ComplexCmp).Compare(a, b);

        // ============================ partition line kernels ============================

        /// <summary>Raw-compare dtypes (ints/bool/char/decimal — no NaN representation): partition the
        /// line in place when it is contiguous, else gather → introselect → scatter.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PartLine<T>(byte* line, PartCtx* c) where T : unmanaged, IComparable<T>
        {
            int n = c->n;
            long s = c->inStride;
            if (s == sizeof(T))
            {
                QuickSelect.PartitionAtMany((T*)line, n, c->kth, c->nKth);
                return;
            }
            T* buf = (T*)c->scratch;
            for (int i = 0; i < n; i++) buf[i] = *(T*)(line + i * s);
            QuickSelect.PartitionAtMany(buf, n, c->kth, c->nKth);
            for (int i = 0; i < n; i++) *(T*)(line + i * s) = buf[i];
        }

        /// <summary>Float dtypes: stable NaN-compact (non-NaN prefix keeps encounter order, NaN bit
        /// patterns stashed verbatim), introselect the prefix with raw compares, NaN tail written back.
        /// kth entries pointing into the NaN tail are correct by construction (every slot there is NaN,
        /// which is where a full NaN-last sort would put them).</summary>
        private static void PartLineF32(byte* line, PartCtx* c)
        {
            int n = c->n; long s = c->inStride;
            float* buf = (float*)c->scratch; float* nans = (float*)c->nanTail;
            int m = 0, q = 0;
            for (int i = 0; i < n; i++) { float v = *(float*)(line + i * s); if (float.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(float*)(line + i * s) = buf[i];
            for (int i = 0; i < q; i++) *(float*)(line + (m + i) * s) = nans[i];
        }

        private static void PartLineF64(byte* line, PartCtx* c)
        {
            int n = c->n; long s = c->inStride;
            double* buf = (double*)c->scratch; double* nans = (double*)c->nanTail;
            int m = 0, q = 0;
            for (int i = 0; i < n; i++) { double v = *(double*)(line + i * s); if (double.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(double*)(line + i * s) = buf[i];
            for (int i = 0; i < q; i++) *(double*)(line + (m + i) * s) = nans[i];
        }

        private static void PartLineF16(byte* line, PartCtx* c)
        {
            int n = c->n; long s = c->inStride;
            Half* buf = (Half*)c->scratch; Half* nans = (Half*)c->nanTail;
            int m = 0, q = 0;
            for (int i = 0; i < n; i++) { Half v = *(Half*)(line + i * s); if (Half.IsNaN(v)) nans[q++] = v; else buf[m++] = v; }
            QuickSelect.PartitionAtMany(buf, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(Half*)(line + i * s) = buf[i];
            for (int i = 0; i < q; i++) *(Half*)(line + (m + i) * s) = nans[i];
        }

        /// <summary>Complex: NumPy's CDOUBLE_LT comparator drives the introselect directly — its NaN
        /// ordering ((1,NaN) before (NaN,1)) is positional, not a compactable any-NaN-last.</summary>
        private static void PartLineComplex(byte* line, PartCtx* c)
        {
            int n = c->n; long s = c->inStride;
            if (s == sizeof(Complex))
            {
                QuickSelect.PartitionAtMany((Complex*)line, n, c->kth, c->nKth, ComplexLess);
                return;
            }
            Complex* buf = (Complex*)c->scratch;
            for (int i = 0; i < n; i++) buf[i] = *(Complex*)(line + i * s);
            QuickSelect.PartitionAtMany(buf, n, c->kth, c->nKth, ComplexLess);
            for (int i = 0; i < n; i++) *(Complex*)(line + i * s) = buf[i];
        }

        // ============================ argpartition line kernels ============================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ArgPartLine<T>(byte* inLine, byte* outLine, PartCtx* c) where T : unmanaged, IComparable<T>
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            T* buf = (T*)c->scratch; long* idx = c->idx;
            for (int i = 0; i < n; i++) { buf[i] = *(T*)(inLine + i * si); idx[i] = i; }
            QuickSelect.PartitionAtMany(buf, idx, n, c->kth, c->nKth);
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = idx[i];
        }

        /// <summary>Float argpartition: NaN indices go to the tail in encounter order — the exact
        /// policy of <see cref="AxisSort"/>'s ArgLineF32/F64 argsort kernels.</summary>
        private static void ArgPartLineF32(byte* inLine, byte* outLine, PartCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            float* buf = (float*)c->scratch; long* idx = c->idx;
            int m = 0;
            for (int i = 0; i < n; i++) { float v = *(float*)(inLine + i * si); if (!float.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            int q = m;
            for (int i = 0; i < n; i++) if (float.IsNaN(*(float*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineF64(byte* inLine, byte* outLine, PartCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            double* buf = (double*)c->scratch; long* idx = c->idx;
            int m = 0;
            for (int i = 0; i < n; i++) { double v = *(double*)(inLine + i * si); if (!double.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            int q = m;
            for (int i = 0; i < n; i++) if (double.IsNaN(*(double*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineF16(byte* inLine, byte* outLine, PartCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            Half* buf = (Half*)c->scratch; long* idx = c->idx;
            int m = 0;
            for (int i = 0; i < n; i++) { Half v = *(Half*)(inLine + i * si); if (!Half.IsNaN(v)) { buf[m] = v; idx[m] = i; m++; } }
            QuickSelect.PartitionAtMany(buf, idx, m, c->kth, c->nKth);
            for (int i = 0; i < m; i++) *(long*)(outLine + i * so) = idx[i];
            int q = m;
            for (int i = 0; i < n; i++) if (Half.IsNaN(*(Half*)(inLine + i * si))) *(long*)(outLine + (q++) * so) = i;
        }

        private static void ArgPartLineComplex(byte* inLine, byte* outLine, PartCtx* c)
        {
            int n = c->n; long si = c->inStride, so = c->outStride;
            Complex* buf = (Complex*)c->scratch; long* idx = c->idx;
            for (int i = 0; i < n; i++) { buf[i] = *(Complex*)(inLine + i * si); idx[i] = i; }
            QuickSelect.PartitionAtMany(buf, idx, n, c->kth, c->nKth, ComplexLess);
            for (int i = 0; i < n; i++) *(long*)(outLine + i * so) = idx[i];
        }

        // ============================ dtype dispatch (one type-switch each) ============================

        private static NDInnerLoopFunc GetPartitionKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => PartLine<bool>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => PartLine<byte>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => PartLine<sbyte>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => PartLine<short>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => PartLine<ushort>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => PartLine<char>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => PartLine<int>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => PartLine<uint>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => PartLine<long>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => PartLine<ulong>((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => PartLineF32((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => PartLineF64((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => PartLineF16((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => PartLineComplex((byte*)p[0], (PartCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => PartLine<decimal>((byte*)p[0], (PartCtx*)a),
            _ => throw new NotSupportedException($"partition not supported for dtype {tc}"),
        };

        private static NDInnerLoopFunc GetArgPartitionKernel(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => static (p, s, c, a) => ArgPartLine<bool>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Byte => static (p, s, c, a) => ArgPartLine<byte>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.SByte => static (p, s, c, a) => ArgPartLine<sbyte>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int16 => static (p, s, c, a) => ArgPartLine<short>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt16 => static (p, s, c, a) => ArgPartLine<ushort>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Char => static (p, s, c, a) => ArgPartLine<char>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int32 => static (p, s, c, a) => ArgPartLine<int>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt32 => static (p, s, c, a) => ArgPartLine<uint>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Int64 => static (p, s, c, a) => ArgPartLine<long>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.UInt64 => static (p, s, c, a) => ArgPartLine<ulong>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Single => static (p, s, c, a) => ArgPartLineF32((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Double => static (p, s, c, a) => ArgPartLineF64((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Half => static (p, s, c, a) => ArgPartLineF16((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Complex => static (p, s, c, a) => ArgPartLineComplex((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            NPTypeCode.Decimal => static (p, s, c, a) => ArgPartLine<decimal>((byte*)p[0], (byte*)p[1], (PartCtx*)a),
            _ => throw new NotSupportedException($"argpartition not supported for dtype {tc}"),
        };
    }
}
