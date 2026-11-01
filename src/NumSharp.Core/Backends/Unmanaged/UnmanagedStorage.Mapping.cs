using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's mapping.c below the indexers: the typed fetch and store an advanced index ends in once its integer index
    // arrays are known — the MapIter's gather (mapiter_get, mapiter_trivial_get) and scatter (mapiter_set,
    // mapiter_trivial_set), the index arrays' broadcast and normalization to intp, the selected positions' offsets and the
    // subspace copies. They are the array object's own members (the np-layer dtype plan's D1, P3 batch 8): positions
    // select and assign elements, and the one value computed, whether a mask selects anything at all, is the engine's Any
    // over the array the scatter marked (np.any's statement). The indexers
    // (NDArray.Indexing*.cs: prepare_index, the routes that turn masks, slices and integers into index arrays, the string
    // routes) call them; NDArray keeps protected forwarders under the old names (FetchIndices, SetIndices, ...) for types
    // that derive from it.
    //
    // The plan's phase P6 made each np.* call of the statements (np.copy, np.broadcast_arrays, np.broadcast_to,
    // np.atleast_1d, np.zeros, np.ones, np.any) the member its facade dispatches to and moved the file from beside the
    // indexers to Backends/Unmanaged/. The IL kernels stay in Backends/Kernels/.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        // ---- The fetch: mapiter_get ------------------------------------------------------------------------------------

        /// <summary>
        ///     The typed fetch an advanced index ends in (NumPy's <c>mapiter_get</c>, with <c>mapiter_trivial_get</c>'s route
        ///     for one index array): the elements of <paramref name="src"/> at the positions the integer index arrays name —
        ///     one array per leading axis, broadcast together — gathered into a new array of shape
        ///     <c>broadcast(index shapes) + src.shape[len(indices):]</c>, or into <paramref name="out"/>. Picks the element
        ///     type from <paramref name="src"/>'s storage lane and runs <see cref="MapIterGet{T}"/>.
        /// </summary>
        /// <param name="src">The array read, any layout (a 0-d one is read as <c>(1,)</c>, a broadcast one copied first).</param>
        /// <param name="indices">One integer index array per leading axis (any integer dtype; a mask arrives as its
        ///     <c>nonzero</c>). The array is REWRITTEN: its entries are replaced by their normalized forms (int32 / int64,
        ///     broadcast, flattened). The indexers bounds-check the values first (<c>PrepareIndex</c>); a value out of range
        ///     still raises here.</param>
        /// <param name="out">A destination of the result's shape and element type, or null for a new array.</param>
        /// <param name="extraDim">Not read; carried for the signature of <c>NDArray.FetchIndices</c>, which forwards here.</param>
        /// <returns>The gathered array: a new C-contiguous array that owns its data (NumPy's fancy result), or
        ///     <paramref name="out"/> (as a typed view when it was not one).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="indices"/> is empty, or <paramref name="out"/> has another
        ///     shape or element type.</exception>
        /// <exception cref="IndexError">More index arrays than <paramref name="src"/> has axes, or an index value out of
        ///     bounds (NumPy's texts).</exception>
        /// <exception cref="IndexOutOfRangeException">The memory-safety backstop: a computed offset outside a buffer, which
        ///     only a miscomputed shape reaches.</exception>
        /// <exception cref="NotSupportedException">A text lane: a fixed-width or <c>StringDType</c> array has no typed fetch
        ///     here (the indexers run its index on a unit or cell view, whose lanes are numeric).</exception>
        /// <remarks><c>[NDScoped]</c>: the per-dispatch <c>MakeGeneric</c> alias, the offsets buffer and every gather
        ///     temporary are released when the call returns, exceptions included; the result (or the caller's
        ///     <paramref name="out"/>, which the scope does not track) is the only survivor.</remarks>
        [NDScoped]
        internal static NDArray MapIterGet(NDArray src, NDArray[] indices, NDArray @out, bool extraDim)
        {
            // #region Compute
		    // switch (src.typecode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1: return MapIterGet<#2>(src.MakeGeneric<#2>(), indices, @out, extraDim);
			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }
            // #endregion

#region Compute

            // NDScoped: reclaims the per-dispatch MakeGeneric alias, the computed-offsets
            // buffer, and every gather temp inside MapIterGet<T>; the yielded result (or the
            // caller's untracked @out) is the only survivor.

            switch (src.typecode)
            {
                case NPTypeCode.Boolean: return MapIterGet<bool>(src.MakeGeneric<bool>(), indices, @out, extraDim);
                case NPTypeCode.Byte:    return MapIterGet<byte>(src.MakeGeneric<byte>(), indices, @out, extraDim);
                case NPTypeCode.SByte:   return MapIterGet<sbyte>(src.MakeGeneric<sbyte>(), indices, @out, extraDim);
                case NPTypeCode.Int16:   return MapIterGet<short>(src.MakeGeneric<short>(), indices, @out, extraDim);
                case NPTypeCode.UInt16:  return MapIterGet<ushort>(src.MakeGeneric<ushort>(), indices, @out, extraDim);
                case NPTypeCode.Int32:   return MapIterGet<int>(src.MakeGeneric<int>(), indices, @out, extraDim);
                case NPTypeCode.UInt32:  return MapIterGet<uint>(src.MakeGeneric<uint>(), indices, @out, extraDim);
                case NPTypeCode.Int64:   return MapIterGet<long>(src.MakeGeneric<long>(), indices, @out, extraDim);
                case NPTypeCode.UInt64:  return MapIterGet<ulong>(src.MakeGeneric<ulong>(), indices, @out, extraDim);
                case NPTypeCode.Char:    return MapIterGet<char>(src.MakeGeneric<char>(), indices, @out, extraDim);
                case NPTypeCode.Half:    return MapIterGet<Half>(src.MakeGeneric<Half>(), indices, @out, extraDim);
                case NPTypeCode.Double:  return MapIterGet<double>(src.MakeGeneric<double>(), indices, @out, extraDim);
                case NPTypeCode.Single:  return MapIterGet<float>(src.MakeGeneric<float>(), indices, @out, extraDim);
                case NPTypeCode.Decimal: return MapIterGet<decimal>(src.MakeGeneric<decimal>(), indices, @out, extraDim);
                case NPTypeCode.Complex: return MapIterGet<System.Numerics.Complex>(src.MakeGeneric<System.Numerics.Complex>(), indices, @out, extraDim);
                default:
                    throw new NotSupportedException();
            }

#endregion

        }

        /// <summary>
        ///     <see cref="MapIterGet(NDArray, NDArray[], NDArray, bool)"/>'s body for one element type: normalizes and
        ///     broadcasts the index arrays, computes the result's shape, then gathers — through the IL take kernel for one
        ///     index array over a C-contiguous source (NumPy's <c>mapiter_trivial_get</c>: one typed move or one sub-array
        ///     copy per index, read at the index array's own width), else by computing each selected position's element
        ///     offset and copying one element (one index array per axis) or one sub-array per position
        ///     (<see cref="MapIterGetSubspace{T}"/>, or <see cref="MapIterGetSubspaceStrided{T}"/> for a source that is not
        ///     C-contiguous).
        /// </summary>
        /// <typeparam name="T">The element type of <paramref name="source"/>'s lane.</typeparam>
        /// <param name="source">The array read, typed.</param>
        /// <param name="indices">The index arrays, as for <see cref="MapIterGet(NDArray, NDArray[], NDArray, bool)"/>
        ///     (rewritten in place).</param>
        /// <param name="out">A destination of the result's shape and element type, or null.</param>
        /// <param name="extraDim">Not read.</param>
        /// <returns>The gathered array, or <paramref name="out"/>. An EMPTY index array returns <c>new NDArray&lt;T&gt;()</c>
        ///     without the selection's shape: the indexers resolve an empty selection before they get here.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="indices"/> is empty, or <paramref name="out"/> does not match.</exception>
        /// <exception cref="IndexError">Too many index arrays, or an index value out of bounds.</exception>
        /// <exception cref="IndexOutOfRangeException">The memory-safety backstop (a computed offset outside the buffer).</exception>
        internal static unsafe NDArray<T> MapIterGet<T>(NDArray<T> source, NDArray[] indices, NDArray @out, bool extraDim) where T : unmanaged
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
                source = ((NDArray)source).copy('K').MakeGeneric<T>();   // np.copy(source): ndarray.copy's plain copy

            long[] retShape = null, subShape = null;

            long indicesSize = indices.Max(nd => nd.size);
            var srcShape = source.Shape;
            var ndsCount = indices.Length;
            // More advanced indices than the array has axes is NEVER valid: the subshape would
            // be sized `srcShape.NDim - ndsCount` (negative) and the offset/getter loops would
            // walk strides[i] past the end — an OOB read/write (memory corruption). NumPy raises.
            if (ndsCount > source.ndim)
                throw new IndexError($"too many indices for array: array is {source.ndim}-dimensional, but {ndsCount} were indexed");
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
                    indicesImpliedShape = idxs.Shape.dimensions;
                    idxs = idxs.flat;
                }

                //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                idxs = NDArray.NormalizeIndexArray(idxs);
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

                    // Broadcasting is required when the index arrays are not ALL the same
                    // shape — NumPy broadcasts them together (e.g. (2,) with (2,1) -> (2,2),
                    // result (2,2)+trailing). A size-only check missed equal-size-but-different-
                    // shape pairs like (2,) vs (2,1) (both size 2), leaving them un-broadcast
                    // and producing a wrong (2,1,...) result instead of (2,2,...).
                    if (!nd.Shape.dimensions.SequenceEqual(idxs.Shape.dimensions))
                        broadcastRequired = true;

                    //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                    indices[i] = NDArray.NormalizeIndexArray(nd);
                }

                //handle broadcasting
                if (broadcastRequired)
                {
                    indices = NDArray.broadcast_arrays(indices);
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
                        indicesImpliedShape = nd.Shape.dimensions;
                        indices[i] = nd = NDArray.atleast_1d(nd).flat;
                    }
                }
            }

            //resolve retShape
            if (!isSubshaped)
            {
                retShape = indicesImpliedShape ?? (long[])idxs.Shape.dimensions.Clone();
            }
            else
            {
                if (indicesImpliedShape == null)
                {
                    retShape = new long[idxs.ndim + srcShape.NDim - ndsCount];
                    for (int i = 0; i < idxs.ndim; i++)
                        retShape[i] = idxs.Shape.dimensions[i];

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

            // Kernel route — ONE index array over a C-contiguous source (a[idx], m[ridx], a[idx2d]):
            // the take kernels gather straight from the index buffer at its own width (int32 or
            // int64, offset honoured), one typed MOV (or one slab cpblk) per index, no offset
            // delegate, no materialised offset array, no second pass — NumPy's mapiter_trivial_get
            // shape, ~3-4x the route below at 100K (docs/NDITER_PERF_DISCOVERY.md §7 lever 1). The
            // general route stays for everything else: several index arrays, a non-contiguous
            // source, a strided / reversed / narrow-dtype index view, a caller-supplied @out, or a
            // zero-byte slab (an empty trailing axis — nothing to gather, shape only).
            if (ndsCount == 1 && @out is null && source.Shape.IsContiguous && DirectILKernelGenerator.Enabled
                && FancyIndexKernelRoute.TryGetIndexPointer(indices[0], out void* kernelIdxPtr, out bool kernelIdx32))
            {
                long slabElements = 1;
                if (isSubshaped)
                    for (int i = 0; i < subShape.Length; i++)
                        slabElements *= subShape[i];
                long slabBytes = slabElements * InfoOf<T>.Size;
                long indexCount = indices[0].size;
                long axisLength = srcShape.dimensions[0];
                if (slabBytes > 0)
                {
                    // Allocated at the FINAL retShape (fresh, C-contiguous, owned — flags.owndata=True
                    // like NumPy's fancy result); the kernel fills it flat.
                    var gathered = new NDArray<T>(Shape.Adopt(retShape), false);
                    byte* srcBase = (byte*)source.Storage.Address + srcShape.offset * InfoOf<T>.Size;
                    long fail = FancyIndexKernelRoute.Gather(srcBase, kernelIdxPtr, kernelIdx32, indexCount, axisLength, slabBytes, (byte*)gathered.Address);
                    if (fail == -1)
                        return gathered;
                    if (fail >= 0)
                    {
                        // Same NumPy-verbatim text the general route raises (the single index array is
                        // axis 0); the partially filled result is a scope-reclaimed temp.
                        long badIndex = FancyIndexKernelRoute.ReadIndex(kernelIdxPtr, kernelIdx32, fail);
                        throw new IndexError($"index {badIndex} is out of bounds for axis 0 with size {axisLength}");
                    }
                    // -2: the IL kernels are unavailable on this host — fall through to the general route.
                }
            }

            //when -----------------------------------------
            //indices point to an ndarray
            if (isSubshaped && (!source.Shape.IsContiguous || (!(@out is null) && !@out.Shape.IsContiguous)))
            {
                var ret = MapIterGetSubspaceStrided(source, indices, ndsCount, retShape: retShape, subShape: subShape, @out);
                return ret;
            }

            //by now all indices are flat, relative indices, might be subshaped, might be non-linear ---------------
            //we flatten to linear absolute points -----------------------------------------------------------------
            var computedOffsets = new NDArray<long>(Shape.Vector(indicesSize), false);
            var computedAddr = computedOffsets.Address;

            //prepare indices getters
            var indexGetters = NDArray.PrepareIndexGetters(srcShape, indices);

            //figure out the largest possible abosulte offset (disengaged for a zero-sized source,
            //which dereferences nothing — see OffsetBackstopUpperBound)
            long largestOffset = OffsetBackstopUpperBound(srcShape, source.size);

            //compute coordinates
            if (indices.Length > 1)
            {
                var index = stackalloc long[ndsCount];
                for (long i = 0; i < indicesSize; i++)
                {
                    for (int ndIdx = 0; ndIdx < ndsCount; ndIdx++)
                        index[ndIdx] = indexGetters[ndIdx](i); //replace with memory access or iterators

                    // Bound BOTH ends: a negative computed offset (an exotic strided / negative-stride
                    // source view whose GetOffset underflows below the buffer base) would otherwise pass
                    // the upper-bound check and read/write BEFORE the buffer — a wild access that can land
                    // in the GC heap (silent corruption). Fault here, like the block-copy guards do.
                    if ((computedAddr[i] = srcShape.GetOffset(index, ndsCount)) > largestOffset || computedAddr[i] < 0)
                        throw new IndexOutOfRangeException($"Index [{string.Join(", ", new Span<long>(index, ndsCount).ToArray())}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }
            else
            {
                var getter = indexGetters[0];
                for (long i = 0; i < indicesSize; i++)
                {
                    if ((computedAddr[i] = srcShape.GetOffset_1D(getter(i))) > largestOffset || computedAddr[i] < 0)
                        throw new IndexOutOfRangeException($"Index [{getter(i)}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }

            //based on recently made `computedOffsets` we retreive data -----------------------------------------

            if (!isSubshaped)
            {
                var idxAddr = computedOffsets.Address;
                var srcAddr = source.Address;
                // Allocate at the FINAL retShape and fill flat (the fresh buffer is C-contiguous either
                // way) — a terminal dst.reshape(retShape) would hand back a VIEW of the hidden temp and
                // misreport flags.owndata=False, where NumPy's fancy result OWNS its data.
                var dst = new NDArray<T>(Shape.Adopt(retShape ?? new[] { computedOffsets.size }), false);
                T* dstAddr = dst.Address;
                //indices point to a scalar
                var len = dst.size;
                for (long i = 0; i < len; i++)
                    dstAddr[i] = srcAddr[idxAddr[i]];

                return dst;
            }
            else
            {
                //non linear is handled before calculating computedOffsets
                var ret = MapIterGetSubspace(source, computedOffsets, indices, ndsCount, retShape: retShape, subShape: subShape, @out);

                return ret;
            }
        }

        /// <summary>
        ///     The subspace copy of <c>mapiter_get</c> for a C-contiguous source: each selected position's sub-array (the axes
        ///     the index arrays leave, <paramref name="subShape"/>) is ONE contiguous block there, copied whole from the
        ///     precomputed element offset into the C-ordered result. Use only when the source is C-contiguous; a strided
        ///     source's sub-arrays are not blocks (<see cref="MapIterGetSubspaceStrided{T}"/>).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="src">The C-contiguous source; the offsets index its buffer, its view offset included.</param>
        /// <param name="offsets">One element offset per selected position (1-D), from the index arrays.</param>
        /// <param name="indices">Not read (the index arrays the offsets came from).</param>
        /// <param name="ndsCount">Not read (the number of leading axes the index arrays consume).</param>
        /// <param name="retShape">The result's shape: the broadcast index shape followed by <paramref name="subShape"/>.</param>
        /// <param name="subShape">The trailing axes each position copies.</param>
        /// <param name="out">A destination of <paramref name="retShape"/> and <typeparamref name="T"/>, or null.</param>
        /// <returns>The owned result INSTANCE (never a <c>MakeGeneric</c> alias of it, which would misreport
        ///     <c>flags.owndata</c> as False), or <paramref name="out"/> itself (a typed view of it when it is not an
        ///     <see cref="NDArray{T}"/>).</returns>
        /// <exception cref="ArgumentException"><paramref name="out"/> has another shape or element type.</exception>
        /// <exception cref="IndexOutOfRangeException">A block outside the source's or the result's buffer: the
        ///     memory-safety guard (the offsets' check upstream validates each block's start only).</exception>
        internal static unsafe NDArray<T> MapIterGetSubspace<T>(NDArray<T> src, NDArray<long> offsets, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray @out) where T : unmanaged
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
                dst = new NDArray<T>(Shape.Adopt(retShape), false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.Shape.dimensions))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.Shape.dimensions)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.type.Name}'");

                dst = @out;
            }

            T* dstAddr = (T*)dst.Address;
            long copySize = subShapeSize * InfoOf<T>.Size;
            long srcBuf = src.Shape.BufferSize;    // element capacity of the source buffer (offsets index into it)
            long dstBuf = dst.size;                // fresh retShape buffer (or validated contiguous @out)

            for (long i = 0; i < offsetsSize; i++)
            {
                long so = *(offsetAddr + i);
                long doff = i * subShapeSize;
                // Memory-safety guard: each copy spans subShapeSize elements but the upstream bound
                // check validated only the block START. A miscomputed retShape/subShape (an exotic
                // mixed-advanced combo the Try* stack mishandles) would copy past either pinned
                // buffer -> silent GC-heap corruption. Fault here instead (catchable).
                if (so < 0 || so + subShapeSize > srcBuf || doff + subShapeSize > dstBuf)
                    throw new IndexOutOfRangeException(IndexingOobMessage("MapIterGetSubspace", so, doff, subShapeSize, srcBuf, dstBuf, retShape, subShape));
                Buffer.MemoryCopy(srcAddr + so, dstAddr + doff, copySize, copySize);
            }

            // Return the owned result INSTANCE, not a MakeGeneric alias of it — the alias is pure
            // type-plumbing but it chains _baseStorage to the (invisible) fresh buffer, misreporting
            // flags.owndata=False where NumPy's fancy result owns its data. A caller-supplied @out that
            // is not already generic still needs the typed wrapper (a view of the user's array — its
            // owndata=False is then genuinely true).
            return dst as NDArray<T> ?? dst.MakeGeneric<T>();
        }

        /// <summary>
        ///     The subspace copy of <c>mapiter_get</c> for a source that is not C-contiguous (transposed, row- or
        ///     column-strided, reversed): each selected position's sub-array is walked element by element through the
        ///     source's strides (an odometer over <paramref name="subShape"/>, last axis fastest) into the C-ordered result,
        ///     as NumPy gathers a strided source into a fresh contiguous array. One <c>GetOffset</c> per element, so slower
        ///     than <see cref="MapIterGetSubspace{T}"/>'s block copies; use it only where those would be wrong.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The source, any non-contiguous layout.</param>
        /// <param name="indices">One flat integer index array per consumed leading axis, broadcast together.</param>
        /// <param name="ndsCount">The number of leading axes the index arrays consume.</param>
        /// <param name="retShape">The result's shape: the broadcast index shape followed by <paramref name="subShape"/>.</param>
        /// <param name="subShape">The trailing axes each position copies.</param>
        /// <param name="out">A destination of <paramref name="retShape"/> and <typeparamref name="T"/>, or null.</param>
        /// <returns>The result (a new C-contiguous array), or a typed view of <paramref name="out"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="out"/> has another shape or element type.</exception>
        /// <exception cref="IndexError">An index value out of bounds (from <see cref="NDArray.PrepareIndexGetters"/>'s
        ///     getters).</exception>
        /// <exception cref="IndexOutOfRangeException">An element outside the source's or the result's buffer: the
        ///     memory-safety guard.</exception>
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        internal static unsafe NDArray<T> MapIterGetSubspaceStrided<T>(NDArray<T> source, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray @out) where T : unmanaged
        {
            // Subshaped fancy gather (ndsCount < source.ndim) from a NON-contiguous source
            // (transposed / row- or col-strided / negative-stride). The contiguous fast path
            // (MapIterGetSubspace) block-copies one CONTIGUOUS subShape per offset, but a strided
            // source's sub-arrays are not contiguous, so gather element-by-element through the
            // source strides into the C-contiguous result. NumPy likewise gathers a strided
            // source straight into a fresh contiguous array.
            //
            // (Was broken: it sized the result as subShape instead of retShape and did
            //  `ret[i] = source[index, ndsCount]` — assigning a whole sub-array into a scalar
            //  slot — so every single-fancy-on-strided-view threw an IncorrectShapeException.)
            long subShapeSize = 1;
            for (int s = 0; s < subShape.Length; s++)
                subShapeSize *= subShape[s];

            long size = indices[0].size; // number of selected (broadcast) fancy multi-indices

            NDArray<T> ret;
            if (@out is null)
                ret = new NDArray<T>(Shape.Adopt(retShape), false);
            else
            {
                //compare computed retShape vs given @out
                if (!retShape.SequenceEqual(@out.Shape.dimensions))
                    throw new ArgumentException($"Given @out NDArray is expected to be shaped [{string.Join(", ", retShape)}] but is instead [{string.Join(", ", @out.Shape.dimensions)}]");
                if (@out.dtype != typeof(T))
                    throw new ArgumentException($"Given @out NDArray is expected to be dtype '{typeof(T).Name}' but is instead '{@out.dtype.type.Name}'");

                ret = @out.MakeGeneric<T>();
            }

            var srcShape = source.Shape;
            int srcNdim = source.ndim;
            int subNdim = subShape.Length;
            var indexGetters = NDArray.PrepareIndexGetters(srcShape, indices);

            T* srcAddr = source.Address;       // buffer base; GetOffset adds the source offset
            T* dstAddr = ret.Address;          // contiguous result (or @out), written in C-order
            long srcBuf = source.Shape.BufferSize;  // element capacity of the source buffer
            long dstBuf = ret.size;                 // contiguous result element capacity

            long* coord = stackalloc long[srcNdim];
            for (long i = 0; i < size; i++)
            {
                // Leading coordinates from the fancy index arrays (one per consumed axis);
                // PrepareIndexGetters has validated/wrapped each to [0, dim-1].
                for (int k = 0; k < ndsCount; k++)
                    coord[k] = indexGetters[k](i);

                // Walk the trailing subShape axes as an odometer (last axis fastest), copying
                // each element through the source strides into the contiguous result block.
                for (int s = 0; s < subNdim; s++)
                    coord[ndsCount + s] = 0;

                long baseDst = i * subShapeSize;
                for (long j = 0; j < subShapeSize; j++)
                {
                    long so = srcShape.GetOffset(coord, srcNdim);
                    // Memory-safety guard (see MapIterGetSubspace): a miscomputed retShape/subShape
                    // would write past the pinned result buffer or read past the source buffer.
                    if (so < 0 || so >= srcBuf || baseDst + j >= dstBuf)
                        throw new IndexOutOfRangeException(IndexingOobMessage("MapIterGetSubspaceStrided", so, baseDst + j, 1, srcBuf, dstBuf, retShape, subShape));
                    dstAddr[baseDst + j] = srcAddr[so];

                    for (int s = subNdim - 1; s >= 0; s--)
                    {
                        if (++coord[ndsCount + s] < subShape[s])
                            break;
                        coord[ndsCount + s] = 0;
                    }
                }
            }

            return ret;
        }

        /// <summary>
        ///     Upper bound for the gather/scatter offset backstop — <see cref="LargestReachableOffset"/>,
        ///     except DISENGAGED for a source that holds no element.
        /// </summary>
        /// <param name="shape">The shape the offsets address (its view offset and strides).</param>
        /// <param name="size">The number of elements the shape holds.</param>
        /// <returns>The largest valid element offset, or <see cref="long.MaxValue"/> for an empty source.</returns>
        /// <remarks>
        ///     That flat-offset comparison is NOT what makes a fancy index valid: every index value has
        ///     already been checked against ITS OWN axis extent (<c>-dim &lt;= idx &lt; dim</c>) — up front and
        ///     axis-major by <c>ScanFancyBounds</c> (<c>NDArray.Indexing.PrepareIndex.cs</c>, NumPy's
        ///     <c>mapping.c</c> order and verbatim <c>IndexError</c>), and again per value inside
        ///     <see cref="NDArray.PrepareIndexGetters"/>. That layer is what rejects <c>np.zeros((3,0))[[5]]</c> and
        ///     <c>np.zeros((0,3))[[0]]</c>, and it keeps doing so. This bound is only the memory-safety
        ///     backstop for the DEREFERENCE that follows.
        ///
        ///     A zero-sized source has no dereference to guard, so the backstop has nothing to say. The
        ///     empty extent is either an INDEXED axis — where the per-axis layer above already rejected
        ///     every possible index value, leaving only an empty index array whose loop body never runs —
        ///     or a TRAILING one, which makes every gathered/scattered block <c>subShapeSize == 0</c>: zero
        ///     bytes copied, no address read. Meanwhile <see cref="LargestReachableOffset"/> reports
        ///     <c>size - 1 == -1</c> there, below every offset it is compared against, so leaving it engaged
        ///     rejected every VALID empty gather — <c>np.zeros((3,0))[[0,1]]</c> threw
        ///     <c>IndexOutOfRangeException</c> where NumPy returns <c>(2,0)</c>.
        ///
        ///     Only the UPPER half is disengaged. The <c>&lt; 0</c> half stays live unconditionally: a negative
        ///     computed offset (an exotic strided / negative-stride view whose <c>GetOffset</c> underflows
        ///     below the buffer base) would read or write BEFORE the buffer — a wild access that can land in
        ///     the GC heap.
        /// </remarks>
        private static long OffsetBackstopUpperBound(Shape shape, long size)
            => size == 0 ? long.MaxValue : LargestReachableOffset(shape, size);

        /// <summary>
        ///     The largest buffer offset REACHABLE by any coordinate of <paramref name="shape"/>:
        ///     its base offset plus, per axis, only the positive-stride contribution (a negative
        ///     stride reaches its maximum at index 0, contributing nothing). For a contiguous view
        ///     this is <c>size-1</c>. Used as the upper bound when validating gather/scatter offsets;
        ///     the previous <c>GetOffset(size-1 corner)</c> was the MINIMUM corner for a negative-stride
        ///     view, so valid early-row offsets were falsely rejected as out of bounds.
        /// </summary>
        /// <param name="shape">The shape the offsets address.</param>
        /// <param name="size">The number of elements the shape holds (<c>-1</c> is returned for an empty contiguous
        ///     shape, which is why <see cref="OffsetBackstopUpperBound"/> disengages for that case).</param>
        /// <returns>The largest element offset any coordinate of the shape reaches.</returns>
        private static long LargestReachableOffset(Shape shape, long size)
        {
            if (shape.IsContiguous)
                return size - 1;

            long largest = shape.offset;
            var strides = shape.strides;
            var dims = shape.dimensions;
            for (int i = 0; i < dims.Length; i++)
            {
                long stride = strides[i];
                if (stride > 0)
                    largest += (dims[i] - 1) * stride;
            }
            return largest;
        }

        /// <summary>
        /// Diagnostic message for the indexing block-copy bounds guards (shared by the getter
        /// gather and setter scatter). Names the exact copy that would overrun a pinned buffer
        /// together with the computed <c>retShape</c>/<c>subShape</c> so a divergence is traced to
        /// the mishandled index combination instead of corrupting the GC heap silently.
        /// </summary>
        /// <param name="where">The copy's routine, as the guard names it.</param>
        /// <param name="srcOff">The copy's source element offset.</param>
        /// <param name="dstOff">The copy's destination element offset.</param>
        /// <param name="span">The number of elements copied.</param>
        /// <param name="srcCap">The source buffer's element capacity.</param>
        /// <param name="dstCap">The destination buffer's element capacity.</param>
        /// <param name="retShape">The computed result shape, or null.</param>
        /// <param name="subShape">The computed sub-array shape, or null.</param>
        /// <returns>The message of the guard's <see cref="IndexOutOfRangeException"/>.</returns>
        private static string IndexingOobMessage(string where, long srcOff, long dstOff, long span, long srcCap, long dstCap, long[] retShape, long[] subShape)
            => $"[indexing OOB guard] {where}: copy src[{srcOff},+{span}) of {srcCap} -> dst[{dstOff},+{span}) of {dstCap} out of bounds " +
               $"(retShape=[{(retShape == null ? "" : string.Join(",", retShape))}] subShape=[{(subShape == null ? "" : string.Join(",", subShape))}]).";

        // ---- The store: mapiter_set ------------------------------------------------------------------------------------

        /// <summary>
        ///     A fancy assignment of TEXT into a numeric array (NumPy's <c>array_assign_subscript</c> → <c>PyArray_FromAny(value,
        ///     descr, FORCECAST)</c> → map iteration): a scalar text value is converted to the array's dtype BEFORE the index
        ///     is resolved — a C# string literal (<see cref="NDArray.WasPythonStr"/>) packed like NumPy's Python str, a 0-d
        ///     text array cast like an array — so its parse error wins even over a bad index; an array value is converted after
        ///     the index and broadcast checks (run on byte proxies of the same shapes, which raise NumPy's errors), and not at
        ///     all when the selection is empty (<c>a[[]] = np.array(['x'])</c> succeeds). The numeric value is then assigned by
        ///     the numeric fancy route.
        /// </summary>
        /// <param name="src">The numeric destination.</param>
        /// <param name="indices">The index arrays.</param>
        /// <param name="values">The text value.</param>
        /// <exception cref="ValueError">A text that does not parse into the destination's dtype (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts), or the broadcast error NumPy raises first.</exception>
        /// <exception cref="OverflowException">An integer text outside the destination's dtype.</exception>
        /// <exception cref="Exception">The index error NumPy raises first (IndexError, ...).</exception>
        /// <remarks>An array value is converted whole, in NumPy's astype order, so when two of its elements fail the one
        ///     reported can differ from the one NumPy's map iterator meets first; the error kind and text form are the same.</remarks>
        private static void MapIterSetFromText(NDArray src, NDArray[] indices, NDArray values)
        {
            if (values.ndim == 0)
            {
                var packed = NDArray.PackForStore(values, src.typecode, pyArrayPackRoute: false);
                NDArray numeric = ReferenceEquals(packed, values) ? values.astype(src.dtype, true, 'K', "unsafe") : packed;
                try
                {
                    MapIterSet(src, indices, numeric);
                }
                finally
                {
                    numeric.Dispose();
                }
                return;
            }
            using (var proxy = new NDArray(NPTypeCode.Byte, new Shape((long[])src.Shape.dimensions.Clone()), true))
            using (var valueProxy = NDArray.ones(new Shape((long[])values.Shape.dimensions.Clone()), NPTypeCode.Byte))
            {
                MapIterSet(proxy, (NDArray[])indices.Clone(), valueProxy);
                if (!proxy.TensorEngine.Any(proxy))
                    return;
            }
            using var converted = values.astype(src.dtype, true, 'K', "unsafe");
            MapIterSet(src, indices, converted);
        }

        /// <summary>
        ///     The typed store an advanced index ends in (NumPy's <c>mapiter_set</c>, with <c>mapiter_trivial_set</c>'s route
        ///     for one index array): <paramref name="values"/>, broadcast to the selection's shape
        ///     (<c>broadcast(index shapes) + src.shape[len(indices):]</c>), written to the positions of
        ///     <paramref name="src"/> the integer index arrays name; a position named twice keeps the LAST write. Text into
        ///     a numeric array is converted where NumPy converts it (<see cref="MapIterSetFromText"/>); every other value is
        ///     stored by <paramref name="src"/>'s storage lane through <see cref="MapIterSet{T}"/>.
        /// </summary>
        /// <param name="src">The array written (writeability is the caller's check).</param>
        /// <param name="indices">One integer index array per leading axis, as for
        ///     <see cref="MapIterGet(NDArray, NDArray[], NDArray, bool)"/> (REWRITTEN in place with their normalized
        ///     forms).</param>
        /// <param name="values">The value: a C# literal (NumPy's Python scalar, packed with NumPy's range checks) or an
        ///     array broadcast to the selection by NumPy's MapIter rule (leading axes beyond the selection's rank must hold
        ///     one element in total).</param>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="indices"/> is empty.</exception>
        /// <exception cref="IndexError">More index arrays than <paramref name="src"/> has axes, or an index value out of
        ///     bounds (NumPy's texts; the one-index-array route checks every index before its first write).</exception>
        /// <exception cref="ValueError">The value does not broadcast to the selection (NumPy's "shape mismatch: value array
        ///     of shape ... could not be broadcast to indexing result of shape ..." and "could not broadcast input array from
        ///     shape ... into shape ..."), or a text value does not parse into the dtype.</exception>
        /// <exception cref="OverflowException">A C# integer literal, or an integer text, outside the dtype.</exception>
        /// <exception cref="IndexOutOfRangeException">The memory-safety backstop (a computed offset outside a buffer).</exception>
        /// <exception cref="NotSupportedException">A text lane: a fixed-width or <c>StringDType</c> array has no typed
        ///     store here (the indexers run its index on a unit view or on marks).</exception>
        /// <remarks><c>[NDScoped]</c> (nothing escapes a store): the per-dispatch <c>MakeGeneric</c> alias, the offsets
        ///     buffer and the broadcast / materialized value temporaries are released when the call returns, exceptions
        ///     included; <paramref name="src"/> and <paramref name="values"/> were made outside and are never touched.</remarks>
        [NDScoped]
        internal static void MapIterSet(NDArray src, NDArray[] indices, NDArray values)
        {
            // Text into a numeric array (a string array takes the string route): converted where NumPy converts it — see
            // MapIterSetFromText.
            if (values is not null && values.typecode == NPTypeCode.String && src.typecode != NPTypeCode.String)
            {
                MapIterSetFromText(src, indices, values);
                return;
            }


            // #region Compute
		    // switch (src.typecode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1: MapIterSet<#2>(src.MakeGeneric<#2>(), indices, values); break;
			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }
            // #endregion

            #region Compute

            switch (src.typecode)
            {
                case NPTypeCode.Boolean:
                    MapIterSet<bool>(src.MakeGeneric<bool>(), indices, values);
                    break;
                case NPTypeCode.Byte:
                    MapIterSet<byte>(src.MakeGeneric<byte>(), indices, values);
                    break;
                case NPTypeCode.SByte:
                    MapIterSet<sbyte>(src.MakeGeneric<sbyte>(), indices, values);
                    break;
                case NPTypeCode.Int16:
                    MapIterSet<short>(src.MakeGeneric<short>(), indices, values);
                    break;
                case NPTypeCode.UInt16:
                    MapIterSet<ushort>(src.MakeGeneric<ushort>(), indices, values);
                    break;
                case NPTypeCode.Int32:
                    MapIterSet<int>(src.MakeGeneric<int>(), indices, values);
                    break;
                case NPTypeCode.UInt32:
                    MapIterSet<uint>(src.MakeGeneric<uint>(), indices, values);
                    break;
                case NPTypeCode.Int64:
                    MapIterSet<long>(src.MakeGeneric<long>(), indices, values);
                    break;
                case NPTypeCode.UInt64:
                    MapIterSet<ulong>(src.MakeGeneric<ulong>(), indices, values);
                    break;
                case NPTypeCode.Char:
                    MapIterSet<char>(src.MakeGeneric<char>(), indices, values);
                    break;
                case NPTypeCode.Half:
                    MapIterSet<Half>(src.MakeGeneric<Half>(), indices, values);
                    break;
                case NPTypeCode.Double:
                    MapIterSet<double>(src.MakeGeneric<double>(), indices, values);
                    break;
                case NPTypeCode.Single:
                    MapIterSet<float>(src.MakeGeneric<float>(), indices, values);
                    break;
                case NPTypeCode.Decimal:
                    MapIterSet<decimal>(src.MakeGeneric<decimal>(), indices, values);
                    break;
                case NPTypeCode.Complex:
                    MapIterSet<System.Numerics.Complex>(src.MakeGeneric<System.Numerics.Complex>(), indices, values);
                    break;
                default:
                    throw new NotSupportedException();
            }

            #endregion

        }

        /// <summary>
        ///     <see cref="MapIterSet(NDArray, NDArray[], NDArray)"/>'s body for one element type, in NumPy's order: the
        ///     too-many-indices check, the value's packing (a C# literal, <c>PyArray_Pack</c>'s range checks) BEFORE any
        ///     index value is read, the index arrays' normalization and broadcast, the selection's shape, the value prepared
        ///     against it (<see cref="NDArray.PrepareAssignedValue"/>, MapIter's value rule), then the scatter — the IL put
        ///     kernel for one index array into a C-contiguous destination (<see cref="TryScatterKernel{T}"/>: every index
        ///     checked before the first write, as <c>mapiter_trivial_set</c> does), else by each selected position's element
        ///     offset, one element (one index array per axis) or one sub-array per position
        ///     (<see cref="MapIterSetSubspace{T}"/>, or <see cref="MapIterSetSubspaceStrided{T}"/> for a destination that is
        ///     not C-contiguous).
        /// </summary>
        /// <typeparam name="T">The element type of <paramref name="source"/>'s lane.</typeparam>
        /// <param name="source">The array written, typed.</param>
        /// <param name="indices">The index arrays (rewritten in place).</param>
        /// <param name="values">The value, as for <see cref="MapIterSet(NDArray, NDArray[], NDArray)"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="indices"/> is empty.</exception>
        /// <exception cref="IndexError">Too many index arrays, or an index value out of bounds.</exception>
        /// <exception cref="ValueError">The value does not broadcast to the selection.</exception>
        /// <exception cref="OverflowException">A C# integer literal outside <typeparamref name="T"/>.</exception>
        /// <exception cref="IndexOutOfRangeException">The memory-safety backstop.</exception>
        /// <remarks>An EMPTY index array returns before anything is written, but after the value's packing: an empty
        ///     selection stores nothing, and an out-of-range literal still raises, as in NumPy.</remarks>
        internal static unsafe void MapIterSet<T>(NDArray<T> source, NDArray[] indices, NDArray values) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            if (indices is null)
                throw new ArgumentNullException(nameof(indices));

            if (indices.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(indices));

            if (source.Shape.IsScalar)
                source = source.reshape(1);

            long[] retShape = null, subShape = null;

            long indicesSize = indices[0].size;
            var srcShape = source.Shape;
            var ndsCount = indices.Length;
            // See the getter: more advanced indices than axes is never valid and would walk
            // strides past the end (OOB write / heap corruption). NumPy raises; so do we.
            if (ndsCount > source.ndim)
                throw new IndexError($"too many indices for array: array is {source.ndim}-dimensional, but {ndsCount} were indexed");

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar, packed into the destination's dtype with
            // NumPy's range checks (PyArray_Pack). NumPy validates the index's structure first (prepare_index: the
            // too-many-indices error above) and converts the value (PyArray_FromAny) BEFORE it checks the index values'
            // bounds, so a[[5]] = 1000 on a 3-element int8 array is the OverflowError, not the IndexError. The packed 0-d
            // is made under the [NDScoped] dispatcher's scope and released with it.
            values = NDArray.PackForStore(values, InfoOf<T>.NPTypeCode, pyArrayPackRoute: false);

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
                    return;

                //handle non-flat index
                if (idxs.ndim != 1)
                {
                    indicesImpliedShape = idxs.Shape.dimensions;
                    idxs = idxs.flat;
                }

                //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                idxs = NDArray.NormalizeIndexArray(idxs);
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
                        return;

                    // Broadcasting is required when the index arrays are not ALL the same
                    // shape — NumPy broadcasts them together (e.g. (2,) with (2,1) -> (2,2)).
                    // A size-only check missed equal-size-but-different-shape pairs like
                    // (2,) vs (2,1) (both size 2), leaving them un-broadcast (wrong shape).
                    if (!nd.Shape.dimensions.SequenceEqual(idxs.Shape.dimensions))
                        broadcastRequired = true;

                    //normalize index dtype (accepts all integer types, rejects float/decimal/etc.)
                    indices[i] = NDArray.NormalizeIndexArray(nd);
                }

                //handle broadcasting
                if (broadcastRequired)
                {
                    indices = NDArray.broadcast_arrays(indices);
                    indicesSize = indices[0].size;
                    idxs = indices[0];
                }

                //handle non-flat shapes post (possibly) broadcasted
                for (int i = 0; i < indices.Length; i++)
                {
                    var nd = indices[i];
                    if (nd.ndim != 1)
                    {
                        indicesImpliedShape = nd.Shape.dimensions;
                        indices[i] = nd = nd.flat;
                    }
                }
            }

            //resolve retShape
            if (!isSubshaped)
            {
                retShape = indicesImpliedShape ?? (long[])idxs.Shape.dimensions.Clone();
            }
            else
            {
                if (indicesImpliedShape == null)
                {
                    retShape = new long[idxs.ndim + srcShape.NDim - ndsCount];
                    for (int i = 0; i < idxs.ndim; i++)
                        retShape[i] = idxs.Shape.dimensions[i];


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

            // NumPy prepares the value against the indexing-result shape BEFORE its first store (MapIter's
            // extra_op block): a value with more axes than the result loses its leading axes (they must hold
            // one element in total: a (1, N) matrix row fits an (N,) selection), and one that still cannot
            // broadcast raises "shape mismatch: value array of shape ... could not be broadcast to indexing
            // result of shape ...". Every route below receives the prepared value.
            values = NDArray.PrepareAssignedValue(values, retShape);

            // Kernel route — ONE index array into a C-contiguous destination (a[idx] = v, m[ridx] = v,
            // m[ridx] = row): validate every index first (NumPy's mapiter_trivial_set does exactly that
            // before its first store, so a bad index leaves the array untouched), then let the put
            // kernels scatter straight from the index buffer at its own width — one typed MOV (or one
            // slab cpblk) per index, a wrapping values cursor for a scalar / single-sub-array value
            // instead of a materialised broadcast copy (docs/NDITER_PERF_DISCOVERY.md §7 lever 1).
            // Everything else keeps the general route below: several index arrays, a non-contiguous
            // destination, a strided / reversed / narrow-dtype index view, a zero-byte slab.
            if (ndsCount == 1 && source.Shape.IsContiguous && DirectILKernelGenerator.Enabled
                && FancyIndexKernelRoute.TryGetIndexPointer(indices[0], out void* kernelIdxPtr, out bool kernelIdx32))
            {
                long slabElements = 1;
                if (isSubshaped)
                    for (int i = 0; i < subShape.Length; i++)
                        slabElements *= subShape[i];
                long slabBytes = slabElements * InfoOf<T>.Size;
                if (slabBytes > 0 && TryScatterKernel(source, srcShape, kernelIdxPtr, kernelIdx32, indices[0].size,
                        srcShape.dimensions[0], slabBytes, retShape, subShape, isSubshaped, values))
                    return;
                // false: the IL kernels are unavailable on this host — general route.
            }

            //when -----------------------------------------
            //indices point to an ndarray
            //TODO: if (isSubshaped && !source.Shape.IsContiguous)
            //TODO:     return MapIterSetSubspaceStrided(source, indices, ndsCount, retShape: retShape, subShape: subShape, values.AsOrMakeGeneric<T>());

            //by now all indices are flat, relative indices, might be subshaped, might be non-linear ---------------
            //we flatten to linear absolute points -----------------------------------------------------------------
            var computedOffsets = new NDArray<long>(Shape.Vector(indicesSize), false);
            var computedAddr = computedOffsets.Address;

            //prepare indices getters
            var indexGetters = NDArray.PrepareIndexGetters(srcShape, indices);

            //figure out the largest possible abosulte offset (true max reachable, neg-stride safe;
            //disengaged for a zero-sized destination, which dereferences nothing — see
            //OffsetBackstopUpperBound)
            long largestOffset = OffsetBackstopUpperBound(srcShape, source.size);

            //compute coordinates
            if (indices.Length > 1)
            {
                var index = stackalloc long[ndsCount];

                for (long i = 0; i < indicesSize; i++)
                {
                    for (int ndIdx = 0; ndIdx < ndsCount; ndIdx++) //todo optimize this loop with unmanaged address.
                        index[ndIdx] = indexGetters[ndIdx](i); //replace with memory access or iterators

                    // Bound BOTH ends: a negative computed offset (an exotic strided / negative-stride
                    // destination view whose GetOffset underflows below the buffer base) would otherwise
                    // pass the upper-bound check and WRITE before the buffer — a wild store that can land
                    // in the GC heap (silent corruption crashing a later GC). Fault here, like the
                    // block-scatter guard in MapIterSetSubspace does.
                    if ((computedAddr[i] = srcShape.GetOffset(index, ndsCount)) > largestOffset || computedAddr[i] < 0)
                        throw new IndexOutOfRangeException($"Index [{string.Join(", ", new Span<long>(index, ndsCount).ToArray())}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }
            else
            {
                var getter = indexGetters[0];
                for (long i = 0; i < indicesSize; i++)
                {
                    if ((computedAddr[i] = srcShape.GetOffset_1D(getter(i))) > largestOffset || computedAddr[i] < 0)
                        throw new IndexOutOfRangeException($"Index [{getter(i)}] exceeds given NDArray's bounds. NDArray is shaped {srcShape}.");
                }
            }

            //based on recently made `computedOffsets` we set data -----------------------------------------

            if (!isSubshaped)
            {
                var idxAddr = computedOffsets.Address;
                var dstAddr = source.Address;
                var valuesTyped = values.AsOrMakeGeneric<T>();
                long len = computedOffsets.size;

                // Scalar fast path: one value fills every selected slot.
                if (valuesTyped.size == 1)
                {
                    T val = *valuesTyped.Address;
                    for (long i = 0; i < len; i++)
                        dstAddr[idxAddr[i]] = val;
                    return;
                }

                // The value must BROADCAST to the indexing-result (selection) shape = retShape; a value
                // of a different, incompatible shape (e.g. a (5,) into a (1,) selection, or a (len,) flat
                // value into a 2-D retShape of the same size) is a NumPy ValueError, not a partial / flat
                // write. Materialize to a C-contiguous buffer of exactly retShape so the flat write below
                // aligns with the C-order offsets.
                if (!(valuesTyped.Shape.IsContiguous && valuesTyped.Shape.dimensions.SequenceEqual(retShape)))
                {
                    string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
                    try
                    {
                        valuesTyped = NDArray.broadcast_to(valuesTyped, Shape.Adopt(retShape)).copy().MakeGeneric<T>();
                    }
                    catch (IncorrectShapeException ex)
                    {
                        throw new ValueError($"could not broadcast input array from shape {Tup(values.Shape.dimensions)} into shape {Tup(retShape)}", ex);
                    }
                }

                T* valAddr = valuesTyped.Address;   // contiguous, exactly retShape (len elements)
                for (long i = 0; i < len; i++)
                    dstAddr[idxAddr[i]] = valAddr[i];
                return;
            }
            else
            {
                // NumPy broadcasts the assigned value to the indexing-RESULT shape
                // (num_offsets,) + subShape before scattering — a scalar, one sub-row
                // (subShape,), or any lower-rank value stretches to fill every selected
                // sub-array (e.g. a[[0,1]] = -1, = [10,20,30,40], = [[100],[200]], = [[1,2,3,4]]).
                // MapIterSetSubspace block-copies one CONTIGUOUS subShape per offset, so materialize the
                // (possibly broadcast / strided) value into a contiguous buffer of exactly retShape.
                // Previously this branch passed the value straight through, tripping a Debug.Assert
                // in DEBUG and over-reading (valuesAddr + i*subShapeSize past a size-1 buffer) in RELEASE.
                var typedValues = values.AsOrMakeGeneric<T>();
                if (!(typedValues.Shape.IsContiguous && typedValues.Shape.dimensions.SequenceEqual(retShape)))
                {
                    try
                    {
                        typedValues = NDArray.broadcast_to(typedValues, Shape.Adopt(retShape)).copy().MakeGeneric<T>();
                    }
                    catch (IncorrectShapeException ex)
                    {
                        // NumPy: ValueError: shape mismatch: value array of shape (X,) could
                        // not be broadcast to indexing result of shape (Y, Z).
                        // NumPy's mapping.c shape string uses NO space after the comma: (2,4).
                        string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
                        throw new ValueError(
                            $"shape mismatch: value array of shape {Tup(values.Shape.dimensions)} " +
                            $"could not be broadcast to indexing result of shape {Tup(retShape)}", ex);
                    }
                }

                // A non-contiguous destination (transposed / row- or col-strided / negative-stride /
                // F-contig) has non-contiguous sub-arrays, so the block-copy fast path (MapIterSetSubspace)
                // would scatter each subShape as if it were contiguous — landing rows at the wrong
                // physical offsets and corrupting the view (e.g. an F-contig a[[0,2]] = v wrote row 2
                // as three consecutive buffer slots instead of the strided column). Route it through
                // the strided element-scatter, mirroring the getter's MapIterGetSubspaceStrided.
                if (!source.Shape.IsContiguous)
                    MapIterSetSubspaceStrided(source, indices, ndsCount, retShape: retShape, subShape: subShape, typedValues);
                else
                    MapIterSetSubspace(source, computedOffsets, indices, ndsCount, retShape: retShape, subShape: subShape, typedValues);
            }

            //return default;
        }

        /// <summary>
        ///     The kernel scatter behind <see cref="MapIterSet{T}"/>'s single-index-array route: a SIMD
        ///     bounds scan over the whole index array FIRST (so an out-of-range index raises before any
        ///     element is written — NumPy's <c>mapiter_trivial_set</c> "check the indices beforehand"),
        ///     then the put kernel with a wrapping values cursor. The value resolves without a copy
        ///     when it is a scalar (a 1-D destination), exactly the selection shape, or exactly one
        ///     trailing sub-array (<c>m[ridx] = row</c>); any other broadcastable value is materialised
        ///     to the selection shape as the general route does, with the same two NumPy error texts.
        ///     Returns false — having written nothing — only when the IL kernels are unavailable.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The C-contiguous destination.</param>
        /// <param name="srcShape">Its shape (the view offset locates the first element).</param>
        /// <param name="idxPtr">The index array's first element, read in place.</param>
        /// <param name="idx32">Whether the index array holds int32 (else int64).</param>
        /// <param name="indexCount">The number of indices.</param>
        /// <param name="axisLength">The length of axis 0, which the indices address.</param>
        /// <param name="slabBytes">The bytes each index writes (one element, or one trailing sub-array).</param>
        /// <param name="retShape">The selection's shape.</param>
        /// <param name="subShape">The trailing sub-array's shape, or null for a full index.</param>
        /// <param name="isSubshaped">Whether each index writes a sub-array.</param>
        /// <param name="values">The value, already prepared against <paramref name="retShape"/>.</param>
        /// <returns>True when the scatter was made; false when the IL kernels are unavailable on this host.</returns>
        /// <exception cref="IndexError">An index out of bounds (NumPy's text), raised before anything is written.</exception>
        /// <exception cref="ValueError">The value does not broadcast to the selection (NumPy's two texts).</exception>
        private static unsafe bool TryScatterKernel<T>(
            NDArray<T> source, Shape srcShape, void* idxPtr, bool idx32, long indexCount, long axisLength,
            long slabBytes, long[] retShape, long[] subShape, bool isSubshaped, NDArray values) where T : unmanaged
        {
            long bad = FancyIndexKernelRoute.FirstOutOfRange(idxPtr, idx32, indexCount, axisLength);
            if (bad >= 0)
                throw new IndexError($"index {FancyIndexKernelRoute.ReadIndex(idxPtr, idx32, bad)} is out of bounds for axis 0 with size {axisLength}");

            string Tup(long[] s) => s.Length == 1 ? $"({s[0]},)" : "(" + string.Join(",", s) + ")";
            var typed = values.AsOrMakeGeneric<T>();
            long valuesCount;
            if (!isSubshaped && typed.size == 1)
            {
                valuesCount = 1;                                   // scalar fill: the values cursor never moves
            }
            else if (typed.Shape.IsContiguous && typed.Shape.dimensions.SequenceEqual(retShape))
            {
                valuesCount = indexCount;                          // exactly the selection: a straight copy
            }
            else if (isSubshaped && typed.Shape.IsContiguous && typed.Shape.dimensions.SequenceEqual(subShape))
            {
                valuesCount = 1;                                   // one sub-array for every selected row
            }
            else
            {
                // The value must BROADCAST to the indexing-result shape; materialise it C-contiguous at
                // exactly retShape (the same ValueError texts as the general route, per branch).
                try
                {
                    typed = NDArray.broadcast_to(typed, Shape.Adopt(retShape)).copy().MakeGeneric<T>();
                }
                catch (IncorrectShapeException ex)
                {
                    if (isSubshaped)
                        throw new ValueError(
                            $"shape mismatch: value array of shape {Tup(values.Shape.dimensions)} " +
                            $"could not be broadcast to indexing result of shape {Tup(retShape)}", ex);
                    throw new ValueError($"could not broadcast input array from shape {Tup(values.Shape.dimensions)} into shape {Tup(retShape)}", ex);
                }
                valuesCount = indexCount;
            }

            byte* dstBase = (byte*)source.Storage.Address + srcShape.offset * InfoOf<T>.Size;
            byte* valuesPtr = (byte*)typed.Storage.Address + typed.Shape.offset * InfoOf<T>.Size;
            long status = FancyIndexKernelRoute.Scatter(dstBase, idxPtr, idx32, indexCount, valuesPtr, valuesCount, axisLength, slabBytes);
            if (status == -2)
                return false;
            if (status >= 0)   // unreachable after the scan above; a hard stop rather than a silent partial write
                throw new IndexError($"index {FancyIndexKernelRoute.ReadIndex(idxPtr, idx32, status)} is out of bounds for axis 0 with size {axisLength}");
            return true;
        }

        /// <summary>
        ///     The subspace scatter of <c>mapiter_set</c> for a C-contiguous destination: each selected position's sub-array
        ///     (the axes the index arrays leave, <paramref name="subShape"/>) is ONE contiguous block there, written whole at
        ///     the precomputed element offset from the value — in order from a contiguous value, through the value's own
        ///     strides otherwise. Use only when the destination is C-contiguous; a strided destination's sub-arrays are not
        ///     blocks (<see cref="MapIterSetSubspaceStrided{T}"/>).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="dst">The C-contiguous destination; the offsets index its buffer, its view offset included.</param>
        /// <param name="dstOffsets">One element offset per selected position (1-D), from the index arrays.</param>
        /// <param name="dstIndices">Not read (the index arrays the offsets came from).</param>
        /// <param name="ndsCount">Not read (the number of leading axes the index arrays consume).</param>
        /// <param name="retShape">The selection's shape (named by the guard's message only).</param>
        /// <param name="subShape">The trailing axes each position writes.</param>
        /// <param name="values">The value, already broadcast to the selection: one <paramref name="subShape"/> block per
        ///     offset.</param>
        /// <exception cref="IndexOutOfRangeException">A block outside the destination's or the value's buffer: the
        ///     memory-safety guard (a miscomputed shape would otherwise write past a pinned buffer).</exception>
        internal static unsafe void MapIterSetSubspace<T>(NDArray<T> dst, NDArray<long> dstOffsets, NDArray[] dstIndices, int ndsCount, long[] retShape, long[] subShape, NDArray<T> values) where T : unmanaged
        {
            //facts:
            //indices are always offsetted to
            Debug.Assert(dstOffsets.ndim == 1);
            Debug.Assert(retShape != null);

            //handle pointers pointing to subshape
            long subShapeSize = 1;
            for (int i = 0; i < subShape.Length; i++)
                subShapeSize *= subShape[i];

            // Each of the dstOffsets.size selected positions receives one subShapeSize block
            // from values, so the value buffer holds offsets*subShape elements — NOT offsets.
            // (The previous assert `dstOffsets.size == values.size` ignored the sub-shape and
            //  tripped on every subshaped fancy-set, e.g. a[[0,2]] = (2,4); the copy loop below
            //  already walks offsets*subShape correctly.)
            Debug.Assert(dstOffsets.size * subShapeSize == values.size);

            long* offsetAddr = dstOffsets.Address;
            long offsetsSize = dstOffsets.size;
            T* valuesAddr = values.Address;
            T* dstAddr = dst.Address;
            long copySize = subShapeSize * InfoOf<T>.Size;
            long dstBuf = dst.Shape.BufferSize;        // element capacity of the destination buffer
            long valBuf = values.Shape.BufferSize;     // element capacity of the value buffer
            if (values.Shape.IsContiguous)
            {
                //linear
                for (long i = 0; i < offsetsSize; i++)
                {
                    long doff = *(offsetAddr + i);
                    long voff = i * subShapeSize;
                    // Memory-safety guard: the scatter copies subShapeSize elements per offset but
                    // the upstream check validated only the block START. A miscomputed retShape/
                    // subShape would write past the destination's pinned buffer (heap corruption)
                    // or read past the (broadcast) value buffer. Fault here instead.
                    if (doff < 0 || doff + subShapeSize > dstBuf || voff + subShapeSize > valBuf)
                        throw new IndexOutOfRangeException(IndexingOobMessage("MapIterSetSubspace", voff, doff, subShapeSize, valBuf, dstBuf, retShape, subShape));
                    Buffer.MemoryCopy(valuesAddr + voff, dstAddr + doff, copySize, copySize);
                }
            }
            else
            {
                //non-linear
                ref Shape shape = ref values.Storage.ShapeReference;
                for (long i = 0; i < offsetsSize; i++)
                {
                    long doff = *(offsetAddr + i);
                    long voff = shape.TransformOffset(i);
                    if (doff < 0 || doff + subShapeSize > dstBuf || voff < 0 || voff + subShapeSize > valBuf)
                        throw new IndexOutOfRangeException(IndexingOobMessage("MapIterSetSubspace(nonlinear)", voff, doff, subShapeSize, valBuf, dstBuf, retShape, subShape));
                    Buffer.MemoryCopy(valuesAddr + voff, dstAddr + doff, copySize, copySize);
                }

                //Parallel.For(0, offsetsSize, i =>
                //    Buffer.MemoryCopy(valuesAddr + *(offsetAddr + i), dstAddr + i * subShapeSize, copySize, copySize));
            }
        }

        /// <summary>
        ///     Subshaped fancy scatter (<paramref name="ndsCount"/> &lt; <c>dst.ndim</c>) into a NON-contiguous
        ///     destination (transposed / row- or col-strided / negative-stride / F-contig). The contiguous
        ///     fast path (<see cref="MapIterSetSubspace{T}"/>) block-copies one CONTIGUOUS <paramref name="subShape"/>
        ///     per selected offset, but a strided destination's sub-arrays are not contiguous in the buffer,
        ///     so scatter element-by-element through the destination strides from the C-contiguous (already
        ///     broadcast to <paramref name="retShape"/>) value buffer. Exact scatter mirror of the getter's
        ///     <c>MapIterGetSubspaceStrided</c>: same odometer over the trailing <paramref name="subShape"/>
        ///     axes, only the copy direction is reversed. NumPy likewise assigns a scalar / lower-rank /
        ///     broadcast value into a strided destination through the view's own strides.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="dst">Destination array (a non-contiguous view aliasing the base buffer).</param>
        /// <param name="indices">One flat integer index array per consumed leading axis (broadcast together).</param>
        /// <param name="ndsCount">Number of leading axes the indices consume (&lt; <c>dst.ndim</c>).</param>
        /// <param name="retShape">Indexing-result shape <c>(num_offsets,) + subShape</c>; the value's shape.</param>
        /// <param name="subShape">The trailing (untouched) axes each selected offset writes across.</param>
        /// <param name="values">Value buffer, C-contiguous and exactly <paramref name="retShape"/>.</param>
        /// <exception cref="IndexError">An index value out of bounds (from <see cref="NDArray.PrepareIndexGetters"/>'s
        ///     getters).</exception>
        /// <exception cref="IndexOutOfRangeException">An element outside the destination's or the value's buffer: the
        ///     memory-safety guard.</exception>
        /// <remarks>An empty trailing axis or an empty index array writes nothing.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        internal static unsafe void MapIterSetSubspaceStrided<T>(NDArray<T> dst, NDArray[] indices, int ndsCount, long[] retShape, long[] subShape, NDArray<T> values) where T : unmanaged
        {
            long subShapeSize = 1;
            for (int s = 0; s < subShape.Length; s++)
                subShapeSize *= subShape[s];

            long size = indices[0].size; // number of selected (broadcast) fancy multi-indices
            if (subShapeSize == 0 || size == 0)                // empty trailing axis / empty index: nothing to scatter
                return;

            var dstShape = dst.Shape;
            int subNdim = subShape.Length;
            var indexGetters = NDArray.PrepareIndexGetters(dstShape, indices);

            T* dstAddr = dst.Address;          // buffer base; offsets are relative to it (offset folded in)
            T* valuesAddr = values.Address;    // C-contiguous value block, exactly retShape (C-order)
            long dstBuf = dst.Shape.BufferSize;     // element capacity of the destination buffer
            long valBuf = values.Shape.BufferSize;  // element capacity of the value buffer

            // Snapshot the leading (consumed) and trailing (sub-array) strides once. The per-element
            // walk then advances the destination offset INCREMENTALLY (section D "Incremental coord
            // advance") — one add per step in the common single-sub-axis case — instead of paying a
            // full GetOffset (a loop over every axis) for each of the size*subShapeSize elements.
            var strides = dstShape.strides;
            long baseOffset = dstShape.offset;
            long* leadStride = stackalloc long[ndsCount == 0 ? 1 : ndsCount];
            for (int k = 0; k < ndsCount; k++) leadStride[k] = strides[k];
            long* subStride = stackalloc long[subNdim == 0 ? 1 : subNdim];
            long* subCoord = stackalloc long[subNdim == 0 ? 1 : subNdim];
            for (int s = 0; s < subNdim; s++) subStride[s] = strides[ndsCount + s];

            // The scatter is memory-latency bound (a strided destination write is a cache miss per
            // sub-array element — NumPy hits the same wall and is only ~1.3x faster on the dense
            // case), so this stays a single straightforward loop; the incremental offset advance
            // above is what keeps the per-element cost off the critical path.
            long vpos = 0;                     // value cursor: contiguous over retShape (C-order)
            for (long i = 0; i < size; i++)
            {
                // Sub-array base offset from the leading fancy indices (each validated/wrapped to
                // [0, dim-1] by PrepareIndexGetters).
                long baseOff = baseOffset;
                for (int k = 0; k < ndsCount; k++)
                    baseOff += indexGetters[k](i) * leadStride[k];

                long off = baseOff;
                for (int s = 0; s < subNdim; s++)
                    subCoord[s] = 0;

                for (long j = 0; j < subShapeSize; j++)
                {
                    // Memory-safety guard (see MapIterSetSubspace): a miscomputed retShape/subShape would
                    // write past the pinned destination buffer or read past the value buffer.
                    if (off < 0 || off >= dstBuf || vpos >= valBuf)
                        throw new IndexOutOfRangeException(IndexingOobMessage("MapIterSetSubspaceStrided", vpos, off, 1, valBuf, dstBuf, retShape, subShape));
                    dstAddr[off] = valuesAddr[vpos++];

                    // Odometer over the trailing subShape axes (last axis fastest), carrying the
                    // offset by +stride on advance and -stride*extent on each axis rollover.
                    for (int s = subNdim - 1; s >= 0; s--)
                    {
                        off += subStride[s];
                        if (++subCoord[s] < subShape[s])
                            break;
                        subCoord[s] = 0;
                        off -= subStride[s] * subShape[s];
                    }
                }
            }
        }
    }
}
