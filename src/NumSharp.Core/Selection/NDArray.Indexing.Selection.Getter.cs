using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on indices, equivalent to nd[NDArray[]].
        /// </summary>
        /// <param name="@out">Alternative output array in which to place the result. It must have the same shape as the expected output and be of dtype <see cref="Int32"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html</remarks>
        /// <exception cref="IndexOutOfRangeException">When one of the indices exceeds limits.</exception>
        /// <exception cref="ArgumentException">indices must be of Int type (byte, u/short, u/int, u/long).</exception>
        public NDArray GetIndices(NDArray @out, NDArray[] indices)
        {
            return FetchIndices(this, indices, @out, true);
        }

        /// <summary>
        /// Normalizes any raw boolean array-like index to an <see cref="NDArray"/> of dtype
        /// Boolean so it is recognized as a boolean mask — NumPy treats any boolean
        /// <c>array_like</c> index as a mask. Covers, via interfaces rather than per-type
        /// cases: a rectangular boolean <see cref="System.Array"/> of any rank
        /// (<c>bool[]</c>, <c>bool[,]</c>, <c>bool[,,]</c>, …) and any
        /// <see cref="IEnumerable{T}"/> of bool (<c>bool[]</c>, <see cref="List{T}"/>, …).
        /// Integer arrays keep their existing (coordinate) semantics and are left untouched.
        /// (A <em>jagged</em> <c>bool[][]</c> cannot be a single 2-D mask here: C# spreads it
        /// through the <c>params object[]</c> indexer into separate per-row <c>bool[]</c>
        /// arguments via array covariance before this runs — use a rectangular <c>bool[,]</c>
        /// for a 2-D mask.) Shared by the getter and setter dispatch; only the general
        /// <c>object[]</c> indexing path reaches it (the typed indexers bypass it).
        /// </summary>
        private static void NormalizeBooleanMaskIndices(object[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                switch (indices[i])
                {
                    case NDArray _:                                          // already an NDArray (mask handled by typecode)
                        continue;
                    case Array arr when arr.GetType().GetElementType() == typeof(bool):
                        indices[i] = np.array(arr).MakeGeneric<bool>();      // bool[], bool[,], bool[,,] (rectangular, any rank)
                        continue;
                    case System.Collections.Generic.IEnumerable<bool> seq:
                        indices[i] = np.array(System.Linq.Enumerable.ToArray(seq)).MakeGeneric<bool>();  // List<bool>, any IEnumerable<bool>
                        continue;
                }
            }
        }

        private NDArray FetchIndices(object[] indicesObjects)
        {
            var indicesLen = indicesObjects.Length;
            NormalizeBooleanMaskIndices(indicesObjects);    // any boolean array-like -> boolean mask
            if (indicesLen == 1)
            {
                switch (indicesObjects[0])
                {
                    case NDArray nd:
                        // Boolean mask indexing: delegate to the specialized NDArray<bool> indexer
                        if (nd.typecode == NPTypeCode.Boolean)
                            return this[nd.MakeGeneric<bool>()];
                        return FetchIndices(this, new NDArray[] {nd}, null, true);
                    case int i:
                        return new NDArray(Storage.GetData(i));
                    case ulong ui:
                        // ulong is the only integer scalar with no implicit conversion to
                        // int/long, so it can't bind to the Slice indexer (where byte/short/
                        // int/uint/long land) and arrives here instead. NumPy indexes with a
                        // uint64 scalar like any other integer: a[np.uint64(1)] == a[1].
                        return new NDArray(Storage.GetData((int)ui));
                    case bool boolean:
                        if (boolean == false)
                            return new NDArray(dtype); //return empty

                        return np.expand_dims(this, 0); //equivalent to [np.newaxis]

                    case int[] coords:
                        // A raw int[]/long[] as the SOLE index is FANCY indexing — NumPy
                        // parity: nd[new int[]{0,2}] selects rows 0 and 2 (shape (2, …)),
                        // NOT the single element at coordinate (0,2). Coordinate access is
                        // preserved via nd.GetData(coords). (A multi-item tuple already
                        // treats int[]/long[] as fancy via the _NDArrayFound scan below.)
                        return FetchIndices(this, new NDArray[] { np.array(coords, copy: false) }, null, true);
                    case long[] coords:
                        return FetchIndices(this, new NDArray[] { np.array(coords, copy: false) }, null, true);
                    case NDArray[] nds:
                        return this[nds];
                    case object[] objs:
                        return this[objs];
                    case string slicesStr:
                    {
                        return new NDArray(Storage.GetView(Slice.ParseSlices(slicesStr)));
                    }
                    case null: throw new ArgumentNullException($"The 1th dimension in given indices is null.");
                    //no default
                }
            }

            // A LEADING boolean mask (any ndim) followed only by basic indices —
            // e.g. arr[mask2d, 1:3], arr[mask, 2] — reduces to: slice the trailing
            // axes, then apply the (now-leading) partial boolean mask. Reuses the
            // unified BooleanMask, so it also covers multi-dimensional masks.
            if (TryFetchLeadingMaskWithBasic(indicesObjects, out var leadResult))
                return leadResult;

            // Mixed basic (slice) + single advanced (array / boolean mask) indexing.
            // NumPy keeps slices as their own output axes and selects the advanced
            // index along its axis (an outer product), instead of broadcasting slices
            // and advanced indices together. Handle that here before the broadcast
            // path below, which only models all-advanced tuples.
            if (TryFetchSliceWithSingleAdvanced(indicesObjects, out var mixedResult))
                return mixedResult;

            // TWO OR MORE advanced indices (arrays / masks) mixed with an explicit
            // slice / newaxis — the general advanced-index case the broadcast path
            // below cannot model (it would broadcast the slices together with the
            // advanced block instead of treating them as outer-product output axes).
            // Builds one integer index array per source axis with NumPy's axis
            // placement (block in place when consecutive, to the front when a
            // slice/newaxis separates the advanced indices) for a single gather.
            if (TryBuildMultiAdvancedGrid(indicesObjects, out var multiGrid))
                return FetchIndices(this, multiGrid, null, true);

            int ints = 0;
            int bools = 0;
            bool foundSlices = false;
            for (var i = 0; i < indicesObjects.Length; i++)
            {
                switch (indicesObjects[i])
                {
                    case NDArray _:
                    case int[] _:
                    case long[] _:
                        goto _NDArrayFound;
                    case int _:
                        ints++;
                        continue;
                    case bool @bool:
                        bools++;
                        continue;
                    case string _:
                    case Slice _:
                        continue;
                    case null: throw new ArgumentNullException($"The {i}th dimension in given indices is null.");
                    default:   throw new IndexError($"only integers, slices (':'), ellipsis ('...'), numpy.newaxis ('None') and integer or boolean arrays are valid indices (got '{(indicesObjects[i]?.GetType()?.Name ?? "null")}')");
                }
            }

            //handle all ints
            if (ints == indicesLen)
                return new NDArray(Storage.GetData(indicesObjects.Cast<int>().ToArray()));

            //handle all booleans
            if (bools == indicesLen)
                return this[np.array(indicesObjects.Cast<bool>().ToArray(), false).MakeGeneric<bool>()];

            Slice[] slices;
            //handle regular slices
            try
            {
                slices = indicesObjects.Select(x =>
                {
                    switch (x)
                    {
                        case Slice o:        return o;
                        case int o:          return Slice.Index(o);
                        case string o:       return new Slice(o);
                        case bool o:         return o ? Slice.NewAxis : throw new NumSharpException("false bool detected"); //TODO: verify this
                        case Half h:         return Slice.Index(Converts.ToInt64(h));
                        case Complex c:      return Slice.Index(Converts.ToInt64(c));
                        case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
                        default:             throw new ArgumentException($"Unsupported slice type: '{(x?.GetType()?.Name ?? "null")}'");
                    }
                }).ToArray();
            }
            catch (NumSharpException e) when (e.Message.Contains("false bool detected"))
            {
                //handle rare case of false bool
                return new NDArray(dtype);
            }

            {
                return new NDArray(Storage.GetView(slices));
            }

            //handle complex ndarrays indexing
            _NDArrayFound:
            var @this = this;
            var indices = new List<NDArray>();
            bool foundNewAxis = false;
            int countNewAxes = 0;
            //handle ndarray indexing
            bool hasCustomExpandedSlice = false; //use for premature slicing detection
            for (int i = 0; i < indicesLen; i++)
            {
                var idx = indicesObjects[i];
                _recuse:
                switch (idx)
                {
                    case Slice o:

                        if (o.IsEllipsis)
                        {
                            indicesObjects = ExpandEllipsis(indicesObjects, @this.ndim).ToArray();
                            //TODO: i think we need to set here indicesLen = indicesObjects.Length
                            continue;
                        }

                        if (o.IsNewAxis)
                        {
                            countNewAxes++;
                            foundNewAxis = true;
                            continue;
                        }

                        hasCustomExpandedSlice = true;
                        indices.Add(GetIndicesFromSlice(@this.Shape.dimensions, o, i - countNewAxes));
                        continue;
                    case int o:
                        indices.Add(NDArray.Scalar<int>(o));
                        continue;
                    case string o:
                        indicesObjects[i] = idx = new Slice(o);

                        goto _recuse;
                    case bool o:
                        if (o)
                        {
                            indicesObjects[i] = idx = Slice.NewAxis;
                            goto _recuse;
                        }
                        else
                            return new NDArray<int>(); //false bool causes nullification of return.
                    case Half h:
                        indices.Add(NDArray.Scalar<int>(Converts.ToInt32(h)));
                        continue;
                    case Complex c:
                        indices.Add(NDArray.Scalar<int>(Converts.ToInt32(c)));
                        continue;
                    case IConvertible o:
                        indices.Add(NDArray.Scalar<int>(o.ToInt32(CultureInfo.InvariantCulture)));
                        continue;
                    case int[] o:
                        indices.Add(np.array(o, copy: false)); //we dont copy, pinning will be freed automatically after we done indexing.
                        continue;
                    case long[] o:
                        indices.Add(np.array(o, copy: false)); //we dont copy, pinning will be freed automatically after we done indexing.
                        continue;
                    case NDArray nd:
                        if (nd.typecode == NPTypeCode.Boolean)
                        {
                            // Combined indexing: a boolean mask mixed with other indices is
                            // NOT a standalone mask — NumPy replaces it with its nonzero()
                            // integer index arrays (one per mask dimension), which then
                            // participate in advanced indexing (mapping.c prepare_index).
                            // e.g. arr[mask1d, 2] == arr[np.nonzero(mask1d)[0], 2].
                            // (A standalone boolean mask never reaches here — it is handled
                            // by the indicesLen == 1 fast-path above.)
                            foreach (var component in np.nonzero(nd.MakeGeneric<bool>()))
                                indices.Add(component);
                            continue;
                        }

                        indices.Add(nd);
                        continue;
                    default: throw new ArgumentException($"Unsupported slice type: '{(idx?.GetType()?.Name ?? "null")}'");
                }
            }

            var indicesArray = indices.ToArray();

            //handle premature slicing when the shapes cant be broadcasted together
            if (hasCustomExpandedSlice && !np.are_broadcastable(indicesArray))
            {
                var ndim = indicesObjects.Length;
                var prematureSlices = new Slice[ndim];
                var dims = @this.shape;
                for (int i = 0; i < ndim; i++)
                {
                    if (indicesObjects[i] is Slice slice)
                    {
                        prematureSlices[i] = slice;
                        //todo: we might need this in the future indicesObjects[i] = Slice.All;
                    }
                    else
                    {
                        prematureSlices[i] = Slice.All;
                    }
                }

                @this = @this[prematureSlices];

                //updated premature axes
                dims = @this.shape;
                for (int i = 0; i < ndim; i++)
                {
                    if (prematureSlices[i] != Slice.All)
                    {
                        indicesArray[i] = GetIndicesFromSlice(dims, Slice.All, i);
                    }
                }
            }

            //TODO: we can use a slice as null indice instead of expanding it, then we use PrepareIndexGetters to actually simulate that.
            var ret = FetchIndices(@this, indicesArray, null, !(indicesObjects[0] is int));

            if (foundNewAxis)
            {
                var targettedAxis = indices.Count - 1;
                var axisOffset = this.ndim - targettedAxis;
                var retShape = ret.Shape;
                for (int i = 0; i < indicesLen; i++)
                {
                    if (!(indicesObjects[i] is Slice slc) || !slc.IsNewAxis)
                        continue;

                    var axis = Math.Max(0, Math.Min(i - axisOffset, ret.ndim));
                    retShape = retShape.ExpandDimension(axis);
                }

                ret = ret.reshape(retShape);
            }

            return ret;
        }

        /// <summary>
        /// Expands a single <c>...</c> (ellipsis) in a normalized index tuple to the
        /// right number of full slices, counting EVERY axis-consuming item (slices,
        /// integers and advanced indices) — unlike <see cref="ExpandEllipsis"/>, which
        /// tallies only Slices and so over-fills when arrays/ints are present. Newaxis
        /// and the ellipsis itself consume no source axis. Each advanced index here is
        /// assumed to consume one axis (the caller falls back for multi-axis masks).
        /// </summary>
        private static object[] ExpandEllipsisForMixed(object[] items, int ndim)
        {
            int consumed = 0;
            bool hasEllipsis = false;
            foreach (var it in items)
            {
                if (it is Slice s && s.IsEllipsis) { hasEllipsis = true; continue; }
                if (it is Slice s2 && s2.IsNewAxis) continue;
                consumed++;
            }

            if (!hasEllipsis)
                return items;

            var outList = new List<object>(items.Length + Math.Max(0, ndim - consumed));
            foreach (var it in items)
            {
                if (it is Slice s && s.IsEllipsis)
                {
                    for (int j = 0; j < ndim - consumed; j++)
                        outList.Add(Slice.All);
                    continue;
                }
                outList.Add(it);
            }
            return outList.ToArray();
        }

        /// <summary>
        /// Detects a LEADING boolean mask followed only by basic indices and builds the
        /// equivalent basic index that slices the trailing axes while leaving the mask
        /// axes full. Shared by the getter and setter. Returns false (caller falls back)
        /// when items[0] is not a boolean mask, any trailing item is advanced / newaxis /
        /// ellipsis, or the index count would exceed the array rank.
        /// </summary>
        private bool TryBuildLeadingMaskBasicIndex(object[] indicesObjects, out NDArray<bool> mask, out object[] basicIndex)
        {
            mask = null;
            basicIndex = null;

            var normalized = new object[indicesObjects.Length];
            for (int i = 0; i < indicesObjects.Length; i++)
            {
                if (indicesObjects[i] is string str)
                {
                    Slice parsed;
                    try { parsed = new Slice(str); }
                    catch { return false; }
                    normalized[i] = parsed;
                }
                else
                {
                    normalized[i] = indicesObjects[i];
                }
            }
            object[] items = ExpandEllipsisForMixed(normalized, this.ndim);

            if (items.Length < 2 || !(items[0] is NDArray nd0 && nd0.typecode == NPTypeCode.Boolean))
                return false;

            int k = nd0.ndim;
            if (k < 1 || k > this.ndim)
                return false;

            // Everything after the leading mask must be basic (slice / scalar int).
            for (int i = 1; i < items.Length; i++)
            {
                switch (items[i])
                {
                    case Slice sl:
                        if (sl.IsNewAxis || sl.IsEllipsis) return false;
                        break;
                    case int _:
                    case long _:
                        break;
                    default:
                        return false;                                   // advanced index after the mask
                }
            }

            int total = k + (items.Length - 1);
            if (total > this.ndim)
                return false;                                           // too many indices for the rank

            basicIndex = new object[total];
            for (int i = 0; i < k; i++) basicIndex[i] = Slice.All;       // mask axes stay full
            for (int i = 1; i < items.Length; i++) basicIndex[k + i - 1] = items[i];
            mask = nd0.MakeGeneric<bool>();
            return true;
        }

        /// <summary>
        /// A boolean mask at the LEADING position of a multi-index tuple, followed only
        /// by basic indices (e.g. <c>arr[mask2d, 1:3]</c>, <c>arr[mask, 2]</c>). Slices
        /// the trailing axes (mask axes left full), then applies the now-leading partial
        /// boolean mask through the unified <see cref="TensorEngine.BooleanMask"/> — so
        /// multi-dimensional masks combined with basic indexing work too.
        /// </summary>
        private bool TryFetchLeadingMaskWithBasic(object[] indicesObjects, out NDArray result)
        {
            result = null;
            if (!TryBuildLeadingMaskBasicIndex(indicesObjects, out var mask, out var basic))
                return false;
            result = this[basic][mask];
            return true;
        }

        /// <summary>
        /// Mixed basic + advanced indexing for the common case of ONE advanced index
        /// (a 1-D integer array, or a 1-D boolean mask via its <see cref="np.nonzero"/>
        /// indices) combined with basic slices/integers — e.g. <c>arr[:, mask]</c>,
        /// <c>arr[mask, 1:3]</c>, <c>arr[1, :, idxArray]</c>. NumPy keeps the slices as
        /// their own output axes and selects the advanced index along its axis (an outer
        /// product), rather than broadcasting slices and advanced indices together.
        /// Implemented as: apply the basic indexing with the advanced axis left full,
        /// then <see cref="np.take"/> the advanced index along its post-basic axis.
        /// Slices, scalar-int reductions, newaxis and ellipsis all compose. Returns
        /// <c>false</c> (caller falls back to the all-advanced broadcast path) when not
        /// applicable: no slice/newaxis present, not exactly one advanced index, or a
        /// multi-dimensional advanced index / k-D boolean mask (multiple advanced axes).
        /// </summary>
        private bool TryFetchSliceWithSingleAdvanced(object[] indicesObjects, out NDArray result)
        {
            result = null;

            // Normalize string slice notations to Slice so newaxis/ellipsis classify
            // uniformly, then expand ellipsis counting EVERY axis-consuming item (the
            // shared ExpandEllipsis miscounts because it only tallies Slices, not the
            // advanced/int indices a mixed tuple also consumes axes with).
            var normalized = new object[indicesObjects.Length];
            for (int i = 0; i < indicesObjects.Length; i++)
            {
                if (indicesObjects[i] is string str)
                {
                    Slice parsed;
                    try { parsed = new Slice(str); }       // single-axis slice notation
                    catch { return false; }                // multi-axis / unparseable: let the caller handle/raise
                    normalized[i] = parsed;
                }
                else
                {
                    normalized[i] = indicesObjects[i];
                }
            }
            object[] items = ExpandEllipsisForMixed(normalized, this.ndim);

            int advItemIdx = -1, advCount = 0;
            bool sawRealSlice = false, sawNewAxis = false;
            object advObj = null;
            for (int i = 0; i < items.Length; i++)
            {
                switch (items[i])
                {
                    case Slice sl:
                        if (sl.IsEllipsis) return false;                // leftover ellipsis (defensive): let caller handle
                        if (sl.IsNewAxis) sawNewAxis = true;            // newaxis: a basic output axis (handled below)
                        else sawRealSlice = true;
                        break;
                    case int _:
                    case long _:
                        break;                                          // scalar (basic, reduces an axis)
                    case NDArray _:
                    case int[] _:
                    case long[] _:
                        advCount++; advItemIdx = i; advObj = items[i]; break;
                    default:
                        return false;                                   // unknown item: let the caller handle/raise
                }
            }
            // Trigger only when a single advanced index is mixed with basic axis-shaping
            // (a real slice or a newaxis). Pure advanced tuples (mask+int, mask+mask,…)
            // are left to the broadcast path, which already models them.
            if (advCount != 1 || !(sawRealSlice || sawNewAxis))
                return false;

            // A 0-D integer array as the SOLE advanced index behaves EXACTLY like a scalar
            // int: its broadcast shape is () so it reduces its axis and contributes no output
            // dimension. NumPy: a[np.array(0), :] -> (2,), NOT (1, 2). Fold it to a scalar int
            // and re-dispatch as pure basic indexing (the np.take path below would instead
            // reshape it to (1,) and leave a spurious size-1 axis).
            if (advObj is NDArray adv0d && adv0d.typecode != NPTypeCode.Boolean && adv0d.ndim == 0)
            {
                var rewritten = (object[])items.Clone();
                rewritten[advItemIdx] = (int)adv0d;          // 0-D scalar -> int (basic reduction)
                result = this[rewritten];
                return true;
            }

            // Resolve the single advanced operand to a 1-D integer index array.
            NDArray advIdx;
            switch (advObj)
            {
                case NDArray nd when nd.typecode == NPTypeCode.Boolean:
                    var nz = np.nonzero(nd.MakeGeneric<bool>());
                    if (nz.Length != 1) return false;                   // k-D mask -> multiple advanced axes; fall back
                    advIdx = nz[0];
                    break;
                case NDArray nd:
                    if (nd.ndim > 1) return false;                      // multi-dim advanced index; fall back
                    advIdx = nd.ndim == 0 ? nd.reshape(1) : nd;
                    break;
                case int[] ia:  advIdx = np.array(ia, copy: false); break;
                case long[] la: advIdx = np.array(la, copy: false); break;
                default: return false;
            }

            // Build the basic index (advanced axis -> full ':', newaxes kept) and locate
            // the advanced axis in the resulting view. Integer indices BEFORE it drop an
            // axis (shift left); newaxes BEFORE it add an axis (shift right).
            var basic = new object[items.Length];
            int curAxis = 0, advSrcAxis = -1, intsBeforeAdv = 0, newAxesBeforeAdv = 0;
            for (int i = 0; i < items.Length; i++)
            {
                if (i == advItemIdx)
                {
                    advSrcAxis = curAxis; basic[i] = Slice.All; curAxis++;
                    continue;
                }

                switch (items[i])
                {
                    case int iv:  basic[i] = iv; if (advSrcAxis < 0) intsBeforeAdv++; curAxis++; break;
                    case long lv: basic[i] = lv; if (advSrcAxis < 0) intsBeforeAdv++; curAxis++; break;
                    case Slice sl when sl.IsNewAxis:
                        basic[i] = Slice.NewAxis; if (advSrcAxis < 0) newAxesBeforeAdv++; break; // consumes no source axis
                    default:
                        basic[i] = items[i]; curAxis++; break;          // real slice passthrough
                }
            }

            var view = this[basic];                                     // basic indexing (slices, int reductions, newaxes)
            result = np.take(view, advIdx, axis: advSrcAxis - intsBeforeAdv + newAxesBeforeAdv);
            return true;
        }

        /// <summary>Kind of one normalized index-tuple item, per source axis (newaxis consumes none).</summary>
        private enum MixKind : byte { Adv, Int, Slice, NewAxis }

        /// <summary>
        /// Shared builder for TWO-OR-MORE advanced indices (integer arrays, or boolean
        /// masks via their <see cref="np.nonzero"/> components) mixed with basic
        /// slices / scalar ints / newaxis — e.g. <c>b[ia,:,ib]</c>, <c>b[:,ia,ib]</c>,
        /// <c>b[ia,ib,:]</c>, <c>b[mask,:,mask2]</c>. Reproduces NumPy's advanced-index
        /// algorithm (<c>mapping.c</c> <c>PyArray_MapIterNew</c> / <c>_get_transpose</c>):
        /// <list type="bullet">
        /// <item>every advanced index broadcasts TOGETHER into one block of axes;</item>
        /// <item>each slice / newaxis keeps its own output axis (outer product with the block);</item>
        /// <item>a scalar int is a 0-d advanced index (part of the block, no output axis);</item>
        /// <item>the block stays IN PLACE when the advanced indices are consecutive, or moves
        /// to the FRONT when a slice / newaxis separates them.</item>
        /// </list>
        /// Emits one integer index array per source axis, each broadcast to the final
        /// output shape, so a single <c>FetchIndices</c> / <c>SetIndices</c> performs the
        /// gather / scatter. Shared by the getter and setter. Returns <c>false</c>
        /// (caller falls back) when there is no explicit slice/newaxis, fewer than two
        /// advanced axes, the tuple over-indexes the rank, or an item is unrecognized.
        /// </summary>
        private bool TryBuildMultiAdvancedGrid(object[] indicesObjects, out NDArray[] grid)
        {
            grid = null;
            int ndim = this.ndim;
            var srcShape = this.Shape;

            // Normalize string slice notations so newaxis/ellipsis classify uniformly.
            var normalized = new object[indicesObjects.Length];
            for (int i = 0; i < indicesObjects.Length; i++)
            {
                if (indicesObjects[i] is string str)
                {
                    Slice parsed;
                    try { parsed = new Slice(str); }
                    catch { return false; }
                    normalized[i] = parsed;
                }
                else
                {
                    normalized[i] = indicesObjects[i];
                }
            }

            // Expand a single ellipsis, counting EACH item's true axis consumption
            // (a k-D boolean mask consumes k axes; slice/int/array one; newaxis zero).
            int consumed = 0;
            bool hasEllipsis = false;
            foreach (var it in normalized)
            {
                switch (it)
                {
                    case Slice s when s.IsEllipsis: hasEllipsis = true; break;
                    case Slice s when s.IsNewAxis: break;
                    case NDArray nd when nd.typecode == NPTypeCode.Boolean: consumed += nd.ndim; break;
                    default: consumed++; break;
                }
            }
            object[] expanded;
            if (hasEllipsis)
            {
                var lst = new List<object>(normalized.Length + Math.Max(0, ndim - consumed));
                foreach (var it in normalized)
                {
                    if (it is Slice s && s.IsEllipsis)
                    {
                        for (int j = 0; j < ndim - consumed; j++) lst.Add(Slice.All);
                    }
                    else lst.Add(it);
                }
                expanded = lst.ToArray();
            }
            else expanded = normalized;

            // Classify into a FLAT per-axis item list; masks expand to their nonzero
            // components (adjacent -> naturally consecutive among themselves).
            var items = new List<(MixKind kind, NDArray adv, long iv, Slice slc)>(expanded.Length + ndim);
            bool hasExplicitBasic = false;
            int srcAxes = 0;
            foreach (var it in expanded)
            {
                switch (it)
                {
                    case Slice s when s.IsEllipsis:
                        return false;                                  // leftover ellipsis (defensive)
                    case Slice s when s.IsNewAxis:
                        items.Add((MixKind.NewAxis, null, 0, null)); hasExplicitBasic = true; break;
                    case Slice s:
                        items.Add((MixKind.Slice, null, 0, s)); hasExplicitBasic = true; srcAxes++; break;
                    case int iv:
                        items.Add((MixKind.Int, null, iv, null)); srcAxes++; break;
                    case long lv:
                        items.Add((MixKind.Int, null, lv, null)); srcAxes++; break;
                    case NDArray nd when nd.typecode == NPTypeCode.Boolean:
                        foreach (var comp in np.nonzero(nd.MakeGeneric<bool>()))
                        { items.Add((MixKind.Adv, comp, 0, null)); srcAxes++; }
                        break;
                    case NDArray nd:
                        items.Add((MixKind.Adv, nd, 0, null)); srcAxes++; break;
                    case int[] ia:
                        items.Add((MixKind.Adv, np.array(ia, copy: false), 0, null)); srcAxes++; break;
                    case long[] la:
                        items.Add((MixKind.Adv, np.array(la, copy: false), 0, null)); srcAxes++; break;
                    case Half h:
                        items.Add((MixKind.Int, null, Converts.ToInt64(h), null)); srcAxes++; break;
                    case Complex c:
                        items.Add((MixKind.Int, null, Converts.ToInt64(c), null)); srcAxes++; break;
                    case IConvertible co:
                        items.Add((MixKind.Int, null, co.ToInt64(CultureInfo.InvariantCulture), null)); srcAxes++; break;
                    default:
                        return false;                                  // unknown item -> caller raises
                }
            }

            if (srcAxes > ndim) return false;                          // over-indexed -> caller raises
            if (!hasExplicitBasic) return false;                       // pure advanced -> existing broadcast path

            int advAxisCount = 0;
            foreach (var t in items) if (t.kind == MixKind.Adv) advAxisCount++;
            if (advAxisCount < 2) return false;                        // single-advanced fast paths already covered it

            // Trailing axes the tuple did not reach are full ':'.
            for (int a = srcAxes; a < ndim; a++)
                items.Add((MixKind.Slice, null, 0, Slice.All));

            // Source axis consumed by each item (-1 for newaxis).
            var axisOfItem = new int[items.Count];
            for (int i = 0, axc = 0; i < items.Count; i++)
                axisOfItem[i] = items[i].kind == MixKind.NewAxis ? -1 : axc++;

            // Consecutiveness: a slice/newaxis between the first and last advanced item
            // (ints count as advanced 0-d) moves the block to the front.
            int firstAdv = -1, lastAdv = -1;
            for (int i = 0; i < items.Count; i++)
                if (items[i].kind == MixKind.Adv || items[i].kind == MixKind.Int)
                { if (firstAdv < 0) firstAdv = i; lastAdv = i; }
            bool consecutive = true;
            for (int i = firstAdv; i <= lastAdv; i++)
                if (items[i].kind == MixKind.Slice || items[i].kind == MixKind.NewAxis)
                { consecutive = false; break; }

            // Broadcast all advanced indices together -> bshape (the block).
            var advArrays = new List<NDArray>(advAxisCount);
            foreach (var t in items) if (t.kind == MixKind.Adv) advArrays.Add(t.adv.astype(NPTypeCode.Int64));
            NDArray[] advB = np.broadcast_arrays(advArrays.ToArray());
            long[] bshape = advB[0].Shape.dimensions;
            int m = bshape.Length;

            // Pre-resolve slice index arrays once (also gives their output lengths).
            var sliceIndex = new NDArray[items.Count];
            for (int i = 0; i < items.Count; i++)
                if (items[i].kind == MixKind.Slice)
                    sliceIndex[i] = GetIndicesFromSlice(srcShape, items[i].slc, axisOfItem[i]);

            // Compute the FINAL (consec-aware) output layout: the block dims, plus one
            // dim per slice and per newaxis, in the right order. slicePos[i] records the
            // output-dim index a slice item occupies.
            var outShape = new List<long>(items.Count + m);
            int blockStart = -1;
            var slicePos = new int[items.Count];
            for (int i = 0; i < slicePos.Length; i++) slicePos[i] = -1;

            if (!consecutive)
            {
                blockStart = 0;
                for (int d = 0; d < m; d++) outShape.Add(bshape[d]);
                for (int i = 0; i < items.Count; i++)
                {
                    if (items[i].kind == MixKind.Slice) { slicePos[i] = outShape.Count; outShape.Add(sliceIndex[i].size); }
                    else if (items[i].kind == MixKind.NewAxis) { outShape.Add(1); }
                }
            }
            else
            {
                bool blockEmitted = false;
                for (int i = 0; i < items.Count; i++)
                {
                    switch (items[i].kind)
                    {
                        case MixKind.Slice: slicePos[i] = outShape.Count; outShape.Add(sliceIndex[i].size); break;
                        case MixKind.NewAxis: outShape.Add(1); break;
                        case MixKind.Adv:
                        case MixKind.Int:
                            if (!blockEmitted) { blockStart = outShape.Count; for (int d = 0; d < m; d++) outShape.Add(bshape[d]); blockEmitted = true; }
                            break;
                    }
                }
            }

            int outRank = outShape.Count;

            // One integer index array per source axis, reshaped to occupy its output
            // role and broadcast (advanced -> the block dims; slice -> its own dim;
            // int -> all-ones constant). broadcast_arrays then stretches them to outShape.
            var axisIndex = new NDArray[ndim];
            int advCursor = 0;
            for (int i = 0; i < items.Count; i++)
            {
                int a = axisOfItem[i];
                switch (items[i].kind)
                {
                    case MixKind.Adv:
                    {
                        var shp = new long[outRank];
                        for (int d = 0; d < outRank; d++) shp[d] = 1;
                        for (int d = 0; d < m; d++) shp[blockStart + d] = bshape[d];
                        axisIndex[a] = advB[advCursor++].reshape(shp);
                        break;
                    }
                    case MixKind.Int:
                    {
                        var shp = new long[outRank];
                        for (int d = 0; d < outRank; d++) shp[d] = 1;
                        axisIndex[a] = np.array(new long[] { items[i].iv }, copy: false).reshape(shp);
                        break;
                    }
                    case MixKind.Slice:
                    {
                        var shp = new long[outRank];
                        for (int d = 0; d < outRank; d++) shp[d] = 1;
                        shp[slicePos[i]] = sliceIndex[i].size;
                        axisIndex[a] = sliceIndex[i].reshape(shp);
                        break;
                    }
                    // NewAxis consumes no source axis (its size-1 output dim is implicit
                    // in every axisIndex' all-ones layout).
                }
            }

            grid = np.broadcast_arrays(axisIndex);
            return true;
        }

        protected static NDArray FetchIndices(NDArray src, NDArray[] indices, NDArray @out, bool extraDim)
        {
            // #region Compute
		    // switch (src.typecode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1: return FetchIndices<#2>(src.MakeGeneric<#2>(), indices, @out, extraDim);
			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }
            // #endregion

#region Compute

            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return FetchIndices<bool>(src.MakeGeneric<bool>(), indices, @out, extraDim);
                case NPTypeCode.Byte:    return FetchIndices<byte>(src.MakeGeneric<byte>(), indices, @out, extraDim);
                case NPTypeCode.SByte:   return FetchIndices<sbyte>(src.MakeGeneric<sbyte>(), indices, @out, extraDim);
                case NPTypeCode.Int16:   return FetchIndices<short>(src.MakeGeneric<short>(), indices, @out, extraDim);
                case NPTypeCode.UInt16:  return FetchIndices<ushort>(src.MakeGeneric<ushort>(), indices, @out, extraDim);
                case NPTypeCode.Int32:   return FetchIndices<int>(src.MakeGeneric<int>(), indices, @out, extraDim);
                case NPTypeCode.UInt32:  return FetchIndices<uint>(src.MakeGeneric<uint>(), indices, @out, extraDim);
                case NPTypeCode.Int64:   return FetchIndices<long>(src.MakeGeneric<long>(), indices, @out, extraDim);
                case NPTypeCode.UInt64:  return FetchIndices<ulong>(src.MakeGeneric<ulong>(), indices, @out, extraDim);
                case NPTypeCode.Char:    return FetchIndices<char>(src.MakeGeneric<char>(), indices, @out, extraDim);
                case NPTypeCode.Half:    return FetchIndices<Half>(src.MakeGeneric<Half>(), indices, @out, extraDim);
                case NPTypeCode.Double:  return FetchIndices<double>(src.MakeGeneric<double>(), indices, @out, extraDim);
                case NPTypeCode.Single:  return FetchIndices<float>(src.MakeGeneric<float>(), indices, @out, extraDim);
                case NPTypeCode.Decimal: return FetchIndices<decimal>(src.MakeGeneric<decimal>(), indices, @out, extraDim);
                case NPTypeCode.Complex: return FetchIndices<System.Numerics.Complex>(src.MakeGeneric<System.Numerics.Complex>(), indices, @out, extraDim);
                default:
                    throw new NotSupportedException();
            }

#endregion

        }

        protected static unsafe NDArray<T> FetchIndices<T>(NDArray<T> source, NDArray[] indices, NDArray @out, bool extraDim) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            if (indices is null)
                throw new ArgumentNullException(nameof(indices));

            if (indices.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(indices));

            if (source.Shape.IsScalar)
                source = source.reshape(1);

            if (source.Shape.IsBroadcasted)
                source = np.copy(source).MakeGeneric<T>();

            long[] retShape = null, subShape = null;

            long indicesSize = indices.Max(nd => nd.size);
            var srcShape = source.Shape;
            var ndsCount = indices.Length;
            bool isSubshaped = ndsCount != source.ndim;
            NDArray idxs;
            long[] indicesImpliedShape = null;
            //preprocess indices -----------------------------------------------------------------------------------------------
            //handle non-flat indices and detect if broadcasting required
            if (indices.Length == 1)
            {
                //fast-lane for 1-d.
                idxs = indices[0];
                if (idxs.Shape.IsEmpty)
                    return new NDArray<T>();

                //handle non-flat index
                if (idxs.ndim != 1)
                {
                    // A 0-D index contributes NO output axis: its shape () is prepended to the
                    // subshape, so a[np.array(0)] matches a[0] -> (subshape), NOT a length-1
                    // leading axis. Keep the natural (possibly empty) implied shape instead of
                    // forcing (1,), which left a spurious size-1 axis (NumPy parity).
                    indicesImpliedShape = idxs.shape;
                    idxs = idxs.flat;
                }

                //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                idxs = NormalizeIndexArray(idxs);
                indices[0] = idxs;
            }
            else
            {
                idxs = indices[0];
                bool broadcastRequired = false;
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];

                    if (nd.Shape.IsEmpty)
                        return new NDArray<T>();

                    //test for broadcasting requirement
                    if (nd.size != indicesSize)
                        broadcastRequired = true;

                    //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                    indices[i] = NormalizeIndexArray(nd);
                }

                //handle broadcasting
                if (broadcastRequired)
                {
                    indices = np.broadcast_arrays(indices);
                    indicesSize = indices[0].size;
                    idxs = indices[0];
                }

                //handle non-flat shapes post (possibly) broadcasted
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];
                    if (nd.ndim != 1)
                    {
                        // 0-D operand (e.g. broadcast of all-scalar advanced indices) -> empty
                        // implied shape, no spurious leading size-1 axis: a[np.array(0), np.array(1)]
                        // -> scalar, matching NumPy.
                        indicesImpliedShape = nd.shape;
                        indices[i] = nd = np.atleast_1d(nd).flat;
                    }
                }
            }

            //resolve retShape
            if (!isSubshaped)
            {
                retShape = indicesImpliedShape ?? (long[])idxs.shape.Clone();
            }
            else
            {
                if (indicesImpliedShape == null)
                {
                    retShape = new long[idxs.ndim + srcShape.NDim - ndsCount];
                    for (int i = 0; i < idxs.ndim; i++)
                        retShape[i] = idxs.shape[i];

                    subShape = new long[srcShape.NDim - ndsCount];
                    for (int dst_i = idxs.ndim, src_i = ndsCount, i = 0; src_i < srcShape.NDim; dst_i++, src_i++, i++)
                    {
                        retShape[dst_i] = srcShape[src_i];
                        subShape[i] = srcShape[src_i];
                    }
                }
                else
                {
                    retShape = indicesImpliedShape;

                    subShape = new long[srcShape.NDim - ndsCount];
                    for (int src_i = ndsCount, i = 0; src_i < srcShape.NDim; src_i++, i++)
                    {
                        subShape[i] = srcShape[src_i];
                    }

                    if (isSubshaped)
                        retShape = Arrays.Concat(indicesImpliedShape, subShape);
                }
            }

            //when -----------------------------------------
            //indices point to an ndarray
            if (isSubshaped && (!source.Shape.IsContiguous || (!(@out is null) && !@out.Shape.IsContiguous)))
            {
                var ret = FetchIndicesNDNonLinear(source, indices, ndsCount, retShape: retShape, subShape: subShape, @out);
                return ret;
            }

            //by now all indices are flat, relative indices, might be subshaped, might be non-linear ---------------
            //we flatten to linear absolute points -----------------------------------------------------------------
            var computedOffsets = new NDArray<long>(Shape.Vector(indicesSize), false);
            var computedAddr = computedOffsets.Address;

            //prepare indices getters
            var indexGetters = PrepareIndexGetters(srcShape, indices);

            //figure out the largest possible abosulte offset
            long largestOffset;
            if (srcShape.IsContiguous)
                largestOffset = source.size - 1;
            else
            {
                var largestIndices = (long[])source.shape.Clone();
                for (int i = 0; i < largestIndices.Length; i++)
                    largestIndices[i] = largestIndices[i] - 1;

                largestOffset = srcShape.GetOffset(largestIndices);
            }

            //compute coordinates
            if (indices.Length > 1)
            {
                var index = stackalloc long[ndsCount];
                for (long i = 0; i < indicesSize; i++)
                {
                    for (int ndIdx = 0; ndIdx < ndsCount; ndIdx++)
                        index[ndIdx] = indexGetters[ndIdx](i); //replace with memory access or iterators

                    if ((computedAddr[i] = srcShape.GetOffset(index, ndsCount)) > largestOffset)
                        throw new IndexOutOfRangeException($"Index [{string.Join(", ", new Span<long>(index, ndsCount).ToArray())}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }
            else
            {
                var getter = indexGetters[0];
                for (long i = 0; i < indicesSize; i++)
                {
                    if ((computedAddr[i] = srcShape.GetOffset_1D(getter(i))) > largestOffset)
                        throw new IndexOutOfRangeException($"Index [{getter(i)}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }

            //based on recently made `computedOffsets` we retreive data -----------------------------------------

            if (!isSubshaped)
            {
                var idxAddr = computedOffsets.Address;
                var srcAddr = source.Address;
                var dst = new NDArray<T>(Shape.Vector(computedOffsets.size), false);
                T* dstAddr = dst.Address;
                //indices point to a scalar
                var len = dst.size;
                for (long i = 0; i < len; i++)
                    dstAddr[i] = srcAddr[idxAddr[i]];

                if (retShape != null)
                    return dst.reshape(retShape);

                return dst;
            }
            else
            {
                //non linear is handled before calculating computedOffsets
                var ret = FetchIndicesND(source, computedOffsets, indices, ndsCount, retShape: retShape, subShape: subShape, @out);

                return ret;
            }
        }

        /// <summary>
        ///     Accepts collapsed 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="offsets"></param>
        /// <param name="retShape"></param>
        /// <param name="absolute">Is the given <paramref name="offsets"/> already point to the offset of <paramref name="src"/>.</param>
        /// <returns></returns>
        protected static unsafe NDArray<T> FetchIndicesND<T>(NDArray<T> src, NDArray<long> offsets, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray @out) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to
            Debug.Assert(offsets.ndim == 1);
            Debug.Assert(retShape != null);

            //handle pointers pointing to subshape
            long subShapeSize = 1;
            for (int i = 0; i < subShape.Length; i++)
                subShapeSize *= subShape[i];

            long* offsetAddr = offsets.Address;
            var offsetsSize = offsets.size;
            T* srcAddr = src.Address;

            NDArray dst;
            if (@out is null)
                dst = new NDArray<T>(retShape, false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.shape))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.shape)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.Name}'");

                dst = @out;
            }

            T* dstAddr = (T*)dst.Address;
            long copySize = subShapeSize * InfoOf<T>.Size;

            for (long i = 0; i < offsetsSize; i++)
                Buffer.MemoryCopy(srcAddr + *(offsetAddr + i), dstAddr + i * subShapeSize, copySize, copySize);

            return dst.MakeGeneric<T>();
        }

        /// <summary>
        ///     Accepts collapsed 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="offsets"></param>
        /// <param name="retShape"></param>
        /// <param name="absolute">Is the given <paramref name="offsets"/> already point to the offset of <paramref name="source"/>.</param>
        /// <returns></returns>
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        protected static unsafe NDArray<T> FetchIndicesNDNonLinear<T>(NDArray<T> source, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray @out) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to
            //handle pointers pointing to subshape
            var subShapeNDim = subShape.Length;

            long size = indices[0].size; //first is ok because they are broadcasted t oeac
            T* srcAddr = source.Address;

            NDArray ret;
            if (@out is null)
                ret = new NDArray<T>(subShape, false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.shape))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.shape)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.Name}'");

                ret = @out;
            }

            T* dstAddr = (T*)ret.Address;

            var srcDims = indices.Length;
            var indexGetters = PrepareIndexGetters(source.Shape, indices);

            //compute coordinates
            //for (long i = 0; i < size; i++)
            long* index = stackalloc long[srcDims];
            for (long i = 0; i < size; i++)
            {
                //load indices
                //index[0] = i;
                for (int k = 0; k < srcDims; k++)
                    index[k] = indexGetters[k](i); //replace with memory access or iterators
#if DEBUG
                var from = source[index, srcDims];
                var to = ret[i];

                //assign
                ret[i] = from;
#else
                ret[i] = source[index, srcDims];
#endif
            };
            //}

            return ret.flat.reshape(retShape).MakeGeneric<T>();
        }
    }
}
