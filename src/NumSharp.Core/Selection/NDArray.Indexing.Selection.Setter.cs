using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Threading.Tasks;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>a[tuple(indices)] = values</c>: the indexer's store through an index tuple whose entries are these
        ///     arrays.
        /// </summary>
        /// <param name="values">The value, stored by NumPy's assignment rules (broadcast to the selection; null is Python's
        ///     None).</param>
        /// <param name="indices">The index tuple's entries, as for <see cref="GetIndices"/>: integer index arrays, boolean
        ///     masks, 0-d arrays; an empty tuple stores into the whole array (<c>a[()] = v</c>).</param>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html
        ///     <para>Until 2026-10-29 the arrays reached the integer-array scatter alone: a mask raised the IndexError for a
        ///     non-integer index array, an empty tuple raised <c>ArgumentException</c>, and a 0-d array was written as a
        ///     one-element 1-D array where NumPy raises "too many indices".</para></remarks>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> or one of its entries is null.</exception>
        /// <exception cref="ValueError">If this array is not writeable (e.g., broadcast array) — NumPy's "assignment
        ///     destination is read-only", checked first — or the value does not fit the selection.</exception>
        /// <exception cref="IndexError">The index does not fit this array (NumPy's <c>prepare_index</c> and MapIter texts,
        ///     <see cref="GetIndices"/>).</exception>
        /// <exception cref="OverflowException">A C# integer outside this array's integer dtype.</exception>
        public void SetIndices(NDArray values, NDArray[] indices)
        {
            NumSharpException.ThrowIfNotWriteable(Shape);
            if (indices is null)
                throw new ArgumentNullException(nameof(indices));
            this[IndexArraysAsKey(indices)] = values;
        }

        /// <summary>
        /// Assignment counterpart to <see cref="TryFetchSliceWithSingleAdvanced"/>:
        /// a single advanced index (1-D integer array, or a 1-D boolean mask via its
        /// nonzero indices) mixed with basic slices / scalar-int reductions — e.g.
        /// <c>arr[:, mask] = v</c>, <c>arr[mask, 1:3] = v</c>. Builds a per-source-axis
        /// integer index grid (slices and the advanced index form an outer product;
        /// scalar ints broadcast) and scatters through the all-advanced assignment path.
        /// Returns <c>false</c> (caller falls back) when not applicable: no slice
        /// present, not exactly one advanced index, a multi-dim advanced index / k-D
        /// mask, a newaxis mixed in (not modelled for assignment), or integers that a
        /// slice separates from the index array (NumPy puts the block in front then).
        /// </summary>
        /// <param name="indicesObjects">The normalized index tuple (<see cref="NormalizeIndexInputs"/>).</param>
        /// <param name="values">The value to store, broadcast over the selection.</param>
        /// <returns>True when the store was made here; false to let the next route take the key.</returns>
        /// <exception cref="ValueError">The value does not broadcast to the selection (MapIter's text, from the scatter).</exception>
        [NDScopedCovered] // only caller: the [NDScoped] SetIndices(object[], values) dispatch
        private bool TrySetSliceWithSingleAdvanced(object[] indicesObjects, NDArray values)
        {
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

            int advItemIdx = -1, advCount = 0;
            bool sawRealSlice = false;
            object advObj = null;
            for (int i = 0; i < items.Length; i++)
            {
                switch (items[i])
                {
                    case Slice sl:
                        if (sl.IsEllipsis || sl.IsNewAxis) return false; // newaxis assignment not modelled here
                        sawRealSlice = true; break;
                    case int _:
                    case long _:
                        break;
                    case NDArray _:
                    case int[] _:
                    case long[] _:
                        advCount++; advItemIdx = i; advObj = items[i]; break;
                    default:
                        return false;
                }
            }
            if (advCount != 1 || !sawRealSlice)
                return false;
            // The index array and the integers are NumPy's advanced entries: when a slice, a newaxis or the ellipsis stands
            // between them (a[0, :, ia]) the advanced block goes in FRONT of the selection, which the per-axis grid below
            // does not express (it keeps each axis in place) — the multi-advanced grid's consecutiveness rule serves it.
            if (!AdvancedEntriesAreConsecutive(normalized))
                return false;

            // A 0-D BOOLEAN is NumPy's HAS_0D_BOOL: it consumes NO source axis and contributes an
            // axis of its own, length 1 (True) or 0 (False). This helper maps its one advanced item
            // ONTO a source axis, so it cannot express that — and when the other items already
            // consume every axis, the axis-mapping loop below stops at `curAxis < ndim` and the
            // bool is silently DROPPED: `a[-1, 2:, -2, np.array(False)] = v`, a no-op in NumPy,
            // became a real write at `a[-1, 2:, -2]`. (It only looked correct when an axis was left
            // over, which placed the empty nonzero() index and emptied the grid by accident.)
            // TryBuild0dBoolWithBasic (which runs first) serves a 0-d bool among basic entries, its
            // block in place or in front (a NON-CONSECUTIVE block went on to TryBuildMultiAdvancedGrid
            // until 2026-10-29), and TryBuildMultiAdvancedGrid, which carries MixKind.ZeroBool, one
            // beside an index array: a 0-d bool that still reaches this helper is refused for the
            // reason above.
            if (advObj is NDArray adv0dBool && adv0dBool.typecode == NPTypeCode.Boolean && adv0dBool.ndim == 0)
                return false;

            // A 0-D integer array as the SOLE advanced index behaves like a scalar int
            // (broadcast shape () -> reduces its axis, no grid dimension). Mirror the getter:
            // fold it to a scalar int and re-dispatch as pure basic-indexing assignment.
            if (advObj is NDArray adv0d && adv0d.typecode != NPTypeCode.Boolean && adv0d.ndim == 0)
            {
                var rewritten = (object[])items.Clone();
                rewritten[advItemIdx] = (int)adv0d;          // 0-D scalar -> int (basic reduction)
                this[rewritten] = values;
                return true;
            }

            NDArray advIdx;
            switch (advObj)
            {
                case NDArray nd when nd.typecode == NPTypeCode.Boolean:
                    var nz = np.nonzero(nd.MakeGeneric<bool>());
                    if (nz.Length != 1) return false;
                    advIdx = nz[0];
                    break;
                case NDArray nd:
                    if (nd.ndim > 1) return false;
                    advIdx = nd.ndim == 0 ? nd.reshape(1) : nd;
                    break;
                case int[] ia:  advIdx = np.array(ia, copy: false); break;
                case long[] la: advIdx = np.array(la, copy: false); break;
                default: return false;
            }

            // One integer index array per source axis. Slices and the advanced index are
            // "grid" axes (outer product); scalar ints stay length-1 and broadcast.
            int ndim = this.ndim;
            var axisIndex = new NDArray[ndim];
            var isGridAxis = new bool[ndim];
            int curAxis = 0;
            for (int i = 0; i < items.Length && curAxis < ndim; i++)
            {
                if (i == advItemIdx)
                {
                    axisIndex[curAxis] = advIdx.astype(NPTypeCode.Int64); isGridAxis[curAxis] = true; curAxis++;
                    continue;
                }
                switch (items[i])
                {
                    case int iv:  axisIndex[curAxis] = np.array(new long[] { iv }, copy: false); isGridAxis[curAxis] = false; curAxis++; break;
                    case long lv: axisIndex[curAxis] = np.array(new long[] { lv }, copy: false); isGridAxis[curAxis] = false; curAxis++; break;
                    default:      axisIndex[curAxis] = GetIndicesFromSlice(this.Shape.dimensions, (Slice)items[i], curAxis); isGridAxis[curAxis] = true; curAxis++; break;
                }
            }
            // Trailing axes the tuple didn't reach are full ':'.
            for (; curAxis < ndim; curAxis++)
            {
                axisIndex[curAxis] = GetIndicesFromSlice(this.Shape.dimensions, Slice.All, curAxis);
                isGridAxis[curAxis] = true;
            }

            // Reshape each grid axis to occupy its own dimension in the outer-product grid.
            int gridDims = 0;
            for (int a = 0; a < ndim; a++) if (isGridAxis[a]) gridDims++;
            int pos = 0;
            for (int a = 0; a < ndim; a++)
            {
                if (!isGridAxis[a]) continue;                          // scalar int: leave length-1
                long len = axisIndex[a].size;
                var sh = new long[gridDims];
                for (int k = 0; k < gridDims; k++) sh[k] = (k == pos) ? len : 1L;
                axisIndex[a] = axisIndex[a].reshape(Shape.Adopt(sh));
                pos++;
            }

            var mesh = np.broadcast_arrays(axisIndex);

            // The value broadcasts to the advanced-index result (grid) shape, per NumPy: leading axes it has
            // beyond the result's rank are dropped first, and a value that still does not fit raises NumPy's
            // "shape mismatch: value array ..." before anything is written (PrepareAssignedValue).
            var gridShape = mesh[0].Shape;
            NDArray v = PrepareAssignedValue(values, gridShape.dimensions);
            if (!System.Linq.Enumerable.SequenceEqual(v.Shape.dimensions, gridShape.dimensions))
                v = np.broadcast_to(v, gridShape);

            Backends.UnmanagedStorage.MapIterSet(this, mesh, v);
            return true;
        }

        [NDScoped]
        protected void SetIndices(object[] indicesObjects, NDArray values)
        {
            // Boundary scope (void method — writes land in `this`, nothing escapes): reclaims the
            // normalization coercions, the boolean-mask MakeGeneric aliases and their nonzero()
            // components, the slice/scalar index arrays, and the mixed-advanced grid temps built
            // by the Try* helpers below. `this`/`values` were made outside: never touched.

            indicesObjects = NormalizeIndexInputs(indicesObjects);    // tuple spread + mask/sequence coercion
            // A C# bool beside other entries is NumPy's 0-d bool array (prepare_index's PyArray_FROM_O): the routes below
            // are written for that spelling (TryBuild0dBoolWithBasic, the multi-advanced grid). A lone bool keeps its case.
            if (indicesObjects.Length > 1)
                indicesObjects = BoolEntriesAsArrays(indicesObjects);
            var indicesLen = indicesObjects.Length;
            if (indicesLen == 1)
            {
                switch (indicesObjects[0])
                {
                    case NDArray nd:
                        // prepare_index's dtype check, as in the getter: a non-index dtype is the IndexError whatever
                        // the array's size, before the too-many check.
                        ThrowIfNotIndexArrayDtype(nd);
                        // A 0-d (scalar) array has NO axes to consume, so any axis-consuming index
                        // — an integer/boolean ARRAY, even an empty one — is "too many indices"
                        // (NumPy mapping.c prepare_index), matching the getter. A 0-d boolean
                        // (a[np.array(True)] = v / a[np.array(False)] = v) consumes no axis and
                        // passes through below.
                        if (this.ndim == 0 && !(nd.typecode == NPTypeCode.Boolean && nd.ndim == 0))
                            throw new IndexError($"too many indices for array: array is 0-dimensional, but {(nd.typecode == NPTypeCode.Boolean ? nd.ndim : 1)} were indexed");
                        // An empty NON-bool index array (size 0) is an empty integer fancy index. An
                        // empty BOOLEAN array stays a MASK (falls through below): it consumes mask.ndim
                        // axes via its nonzero, so the value must broadcast to (0,)+arr.shape[mask.ndim:]
                        // — treating a multi-dim empty mask as a single empty fancy mis-shapes that check.
                        if (nd.size == 0 && nd.ndim >= 1 && nd.typecode != NPTypeCode.Boolean)
                        {
                            Backends.UnmanagedStorage.MapIterSet(this, new NDArray[] { nd }, values);
                            return;
                        }
                        // Boolean mask indexing: delegate to the specialized NDArray<bool> indexer
                        if (nd.typecode == NPTypeCode.Boolean)
                        {
                            this[nd.MakeGeneric<bool>()] = values;
                            return;
                        }
                        Backends.UnmanagedStorage.MapIterSet(this, new NDArray[] {nd}, values);
                        return;
                    case int i:
                        Storage.SetData(values, i);
                        return;
                    case ulong ui:
                        // uint64 scalar index — see the getter: ulong can't bind to the Slice
                        // indexer, so it lands here. Route through the same Slice.Index view as
                        // the int/long scalar path so the value broadcasts across the reduced
                        // sub-array (NumPy: a[np.uint64(1)] = v sets all of row 1).
                        new NDArray(Storage.GetView(Slice.Index((long)ui))).SetData(values, new int[0]);
                        return;
                    case bool boolean:
                    {
                        // NumPy's prepare_index reads a Python bool as the 0-d bool array PyArray_FROM_O makes of it: the
                        // 0-d mask's store (SetAdvanced's, for a caller that reaches this dispatch directly). It used to
                        // store into the whole array for True — no new axis, so the value was checked against a.shape with
                        // the view's text — and to do nothing for False, without NumPy's conversion or value check.
                        using var zeroDMask = ZeroDBoolMask(boolean);
                        SetMasked(zeroDMask, values);
                        return;
                    }

                    case int[] coords:
                        // A raw int[]/long[] as the SOLE index is FANCY scatter — NumPy
                        // parity: nd[new int[]{0,2}] = v assigns to rows 0 and 2, NOT the
                        // single element at coordinate (0,2). Coordinate assignment is
                        // preserved via nd.SetData(values, coords). (A multi-item tuple
                        // already treats int[]/long[] as fancy via the _NDArrayFound scan.)
                        if (this.ndim == 0)
                            throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
                        Backends.UnmanagedStorage.MapIterSet(this, new NDArray[] { np.array(coords, copy: false) }, values);
                        return;
                    case long[] coords:
                        if (this.ndim == 0)
                            throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
                        Backends.UnmanagedStorage.MapIterSet(this, new NDArray[] { np.array(coords, copy: false) }, values);
                        return;
                    case NDArray[] nds:
                        this[nds] = values;
                        return;
                    case object[] objs:
                        this[objs] = values;
                        return;
                    case string slicesStr:
                        new NDArray(Storage.GetView(Slice.ParseSlices(slicesStr))).SetData(values, new int[0]);
                        return;
                    case null:
                        throw new ArgumentNullException($"The 1th dimension in given indices is null.");
                    //no default
                }
            }

            // Phase C gate (see the getter): NumPy prepare_index classification + validation up front,
            // so a structurally invalid assignment tuple raises the NumPy IndexError instead of letting
            // the Try* stack scatter through a malformed shape. Every key that reaches this point is validated, a
            // lone entry the switch above does not serve too (a float, a float16, a complex: NumPy's invalid-index
            // IndexError, which the value's conversion below used to pre-empt — np.zeros(3, uint8)[1.5] = -1 raised
            // the OverflowError). The index ARRAYS' values are not scanned yet: NumPy checks them in its MapIter,
            // after the value's conversion (ScanIndexArrayBounds below).
            var prepared = PrepareIndex(this.Shape, indicesObjects, scanFancyBounds: false);

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar, packed into this dtype with NumPy's range
            // checks (PyArray_Pack) once, here: after prepare_index (its integer-bounds IndexError comes first, as
            // get_view_from_index's does in NumPy) and before the Try* routes below derive views or broadcasts of the
            // value, which would lose the mark. The packed 0-d belongs to this method's scope. A tuple with an advanced
            // component is NumPy's PyArray_FromAny route (a NumPy scalar is cast); a basic-only tuple is a view
            // assignment, PyArray_Pack's route.
            values = PackForStore(values, typecode, pyArrayPackRoute: !HasAdvancedIndex(indicesObjects));

            // MapIter's check of the index arrays' values, after the conversion as in NumPy (a[[0, 99], 0] = 1000 on int8
            // is the OverflowError above, not this IndexError) and before any Try* route below scatters.
            ScanIndexArrayBounds(this.Shape, prepared);

            // A 0-d boolean (np.array(True)/np.array(False)) mixed with basic indices: True
            // assigns through the size-1 newaxis view (which aliases this, value broadcasting
            // against the (1,...) selection); any-False selects nothing. Mirrors the getter's
            // HAS_0D_BOOL handling.
            if (TryBuild0dBoolWithBasic(indicesObjects, out var boolBasic, out var boolAxis, out var boolVal))
            {
                if (boolVal)
                {
                    // HAS_0D_BOOL is advanced indexing: NumPy's MapIter checks the value against the indexing result —
                    // the shape of the newaxis view built here — with ITS text ("shape mismatch: value array of shape
                    // (5,) could not be broadcast to indexing result of shape (1,4)") before any element is written. The
                    // view assignment raised the view's text instead (AssignArray's "could not broadcast input array from
                    // shape (5,) into shape (1,4)"). A value that passes stores the same elements either way. None (null,
                    // which a text array's store converts itself) is a scalar: nothing to check.
                    var selection = this[boolBasic];
                    this[boolBasic] = values is null ? null : PrepareAssignedValue(values, selection.Shape.dimensions);
                    return;
                }
                // any-False -> the selection is EMPTY (size 0 along boolAxis), but NumPy still
                // requires the value to broadcast to that empty indexing-result shape — a
                // non-broadcastable, non-scalar value raises ValueError; it is NOT silently a
                // no-op (mapping.c array_assign_subscript). A scalar always broadcasts.
                if (values.size != 1)
                {
                    var selShape = boolBasic.Length == 0
                        ? (long[])this.Shape.dimensions.Select(x => (long)x).ToArray()
                        : this[boolBasic].Shape.dimensions.Select(x => (long)x).ToArray();
                    selShape[boolAxis] = 0;                      // False -> empty axis
                    string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
                    try { np.broadcast_to(values, Shape.Adopt(selShape)); }
                    catch (IncorrectShapeException ex)
                    {
                        throw new ValueError($"shape mismatch: value array of shape {Tup(values.Shape.dimensions)} " +
                                             $"could not be broadcast to indexing result of shape {Tup(selShape)}", ex);
                    }
                }
                return;
            }

            // A leading boolean mask (any ndim) followed only by basic indices —
            // e.g. arr[mask2d, 1:3] = v. Slice the trailing axes to a writable view
            // (shares memory), then assign through the now-leading partial mask.
            if (TryBuildLeadingMaskBasicIndex(indicesObjects, out var leadMask, out var leadBasic))
            {
                this[leadBasic][leadMask] = values;
                return;
            }

            // Mixed basic (slice) + single advanced (array / boolean mask) assignment,
            // mirroring the getter: slices stay basic output axes, the advanced index
            // scatters along its axis. Handled before the broadcast path below.
            if (TrySetSliceWithSingleAdvanced(indicesObjects, values))
                return;

            // TWO OR MORE advanced indices mixed with an explicit slice / newaxis —
            // the assignment counterpart of the getter's multi-advanced gather. Builds
            // the same per-source-axis integer index grid (NumPy axis placement) and
            // scatters the value (broadcast to the grid shape) in one pass.
            if (TryBuildMultiAdvancedGrid(indicesObjects, out var multiGrid))
            {
                // NumPy's MapIter value rule (leading axes dropped, its ValueError text): PrepareAssignedValue.
                var gridShape = multiGrid[0].Shape;
                NDArray v = PrepareAssignedValue(values, gridShape.dimensions);
                if (!System.Linq.Enumerable.SequenceEqual(v.Shape.dimensions, gridShape.dimensions))
                    v = np.broadcast_to(v, gridShape);
                Backends.UnmanagedStorage.MapIterSet(this, multiGrid, v);
                return;
            }

            int ints = 0;
            int bools = 0;
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
                    default: throw new IndexError(InvalidIndexMessage);   // NumPy's text, no type name (prepare_index)
                }
            }

            //handle all ints
            if (ints == indicesLen)
            {
                Storage.SetData(values, indicesObjects.Cast<int>().ToArray());
                return;
            }

            //handle all booleans
            if (bools == indicesLen)
            {
                this[np.array(indicesObjects.Cast<bool>().ToArray(), false).MakeGeneric<bool>()] = values;
                return;
            }

            Slice[] slices;
            //handle regular slices
            try
            {
                slices = indicesObjects.Select(x =>
                {
                    switch (x)
                    {
                        case Slice o: return o;
                        case int o: return Slice.Index(o);
                        case string o: return new Slice(o);
                        case bool o: return o ? Slice.NewAxis : throw new NumSharpException("false bool detected"); //TODO: verify this
                        case Half h: return Slice.Index(Converts.ToInt64(h));
                        case Complex c: return Slice.Index(Converts.ToInt64(c));
                        case IConvertible o: return Slice.Index(o.ToInt64(CultureInfo.InvariantCulture));
                        default: throw new ArgumentException($"Unsupported slice type: '{(x?.GetType()?.Name ?? "null")}'");
                    }
                }).ToArray();
            }
            catch (NumSharpException e) when (e.Message.Contains("false bool detected"))
            {
                //handle rare case of false bool
                return;
            }

            new NDArray(Storage.GetView(slices)).SetData(values, new int[0]);
            return;   // pure basic (slices/ellipsis/newaxis): assigned via the view; do NOT fall through
                      // into _NDArrayFound, which would re-interpret the slices as fancy index arrays and
                      // broadcast them (the getter's equivalent path likewise returns its view).

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
                            //TODO: whats the approach to handling a newaxis in setter, findout.
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
                            return; //false bool causes nullification of return.
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
                            // Combined indexing: a boolean mask mixed with other indices
                            // expands to its nonzero() integer index arrays (one per mask
                            // dimension) for advanced-index assignment (mapping.c
                            // prepare_index), e.g. arr[mask1d, 2] = v.
                            // (Slice + mask is intercepted earlier by the mixed handler;
                            //  a standalone mask is handled by the indicesLen == 1 path.)
                            foreach (var component in np.nonzero(nd.MakeGeneric<bool>()))
                                indices.Add(component);
                            continue;
                        }

                        indices.Add(nd);
                        continue;
                    default: throw new ArgumentException($"Unsupported slice type: '{(idx?.GetType()?.Name ?? "null")}'");
                }
            }

            NDArray[] indicesArray = indices.ToArray();

            //handle premature slicing when the shapes cant be broadcasted together
            if (hasCustomExpandedSlice && !np.are_broadcastable(indicesArray))
            {
                var ndim = indicesObjects.Length;
                var prematureSlices = new Slice[ndim];
                var dims = @this.Shape.dimensions;
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
                dims = @this.Shape.dimensions;
                for (int i = 0; i < ndim; i++)
                {
                    if (prematureSlices[i] != Slice.All)
                    {
                        indicesArray[i] = GetIndicesFromSlice(dims, Slice.All, i);
                    }
                }
            }

            //TODO: we can use a slice as null indice instead of expanding it, then we use PrepareIndexGetters to actually simulate that.
            Backends.UnmanagedStorage.MapIterSet(@this, indicesArray, values);

            //TODO: this is valid code for getter, we need to impl a similar technique before passing @this.
            //if (foundNewAxis)
            //{
            //    //TODO: This is not the behavior when setting with new axis, is it even possible?
            //    var targettedAxis = indices.Count - 1;
            //    var axisOffset = this.ndim - targettedAxis;
            //    var retShape = ret.Shape;
            //    for (int i = 0; i < indicesLen; i++)
            //    {
            //        if (!(indicesObjects[i] is Slice slc) || !slc.IsNewAxis)
            //            continue;
            //
            //        var axis = Math.Max(0, Math.Min(i - axisOffset, ret.ndim));
            //        retShape = retShape.ExpandDimension(axis);
            //    }
            //
            //    ret = ret.reshape(retShape);
            //}
            //
            //return ret;
        }

        /// <inheritdoc cref="Backends.UnmanagedStorage.MapIterSet(NDArray, NDArray[], NDArray)"/>
        /// <remarks>Forwards to <see cref="Backends.UnmanagedStorage.MapIterSet(NDArray, NDArray[], NDArray)"/>, the typed
        ///     store's home since 2026-10-29 (the np-layer dtype plan's batch 8 moved mapping.c's typed store to the array
        ///     object's storage). Kept under its old name for types that derive from <see cref="NDArray"/>; the indexers call
        ///     the storage member.</remarks>
        protected static void SetIndices(NDArray src, NDArray[] indices, NDArray values)
            => Backends.UnmanagedStorage.MapIterSet(src, indices, values);

        /// <inheritdoc cref="Backends.UnmanagedStorage.MapIterSet{T}(NDArray{T}, NDArray[], NDArray)"/>
        /// <remarks>Forwards to <see cref="Backends.UnmanagedStorage.MapIterSet{T}(NDArray{T}, NDArray[], NDArray)"/> (moved
        ///     there 2026-10-29); kept under its old name for types that derive from <see cref="NDArray"/>.</remarks>
        protected static void SetIndices<T>(NDArray<T> source, NDArray[] indices, NDArray values) where T : unmanaged
            => Backends.UnmanagedStorage.MapIterSet<T>(source, indices, values);

        /// <inheritdoc cref="Backends.UnmanagedStorage.MapIterSetSubspace{T}(NDArray{T}, NDArray{long}, NDArray[], int, long[], long[], NDArray{T})"/>
        /// <remarks>Forwards to <see cref="Backends.UnmanagedStorage.MapIterSetSubspace{T}(NDArray{T}, NDArray{long}, NDArray[], int, long[], long[], NDArray{T})"/>
        ///     (moved there 2026-10-29); kept under its old name for types that derive from <see cref="NDArray"/>.</remarks>
        protected static void SetIndicesND<T>(NDArray<T> dst, NDArray<long> dstOffsets, NDArray[] dstIndices, int ndsCount, long[] retShape, long[] subShape, NDArray<T> values) where T : unmanaged
            => Backends.UnmanagedStorage.MapIterSetSubspace(dst, dstOffsets, dstIndices, ndsCount, retShape, subShape, values);

        /// <inheritdoc cref="Backends.UnmanagedStorage.MapIterSetSubspaceStrided{T}(NDArray{T}, NDArray[], int, long[], long[], NDArray{T})"/>
        /// <remarks>Forwards to <see cref="Backends.UnmanagedStorage.MapIterSetSubspaceStrided{T}(NDArray{T}, NDArray[], int, long[], long[], NDArray{T})"/>
        ///     (moved there 2026-10-29); kept under its old name for types that derive from <see cref="NDArray"/> (and for
        ///     the AuditV2 test that invokes it by reflection).</remarks>
        protected static void SetIndicesNDNonLinear<T>(NDArray<T> dst, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray<T> values) where T : unmanaged
            => Backends.UnmanagedStorage.MapIterSetSubspaceStrided(dst, indices, ndsCount, retShape, subShape, values);
    }
}
