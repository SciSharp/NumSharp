using System;
using NumSharp.Backends;

namespace NumSharp
{
    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    //  numpy.ma's arraysetops (numpy/ma/extras.py: unique / in1d / isin / intersect1d / setxor1d / union1d / setdiff1d),
    //  ported STATEMENT BY STATEMENT. NumPy builds them from np.unique, ma.concatenate, MaskedArray.sort/argsort and the
    //  MaskedArray `==` / `!=` operators over the MaskedArray subclass, and those statements decide the result's mask: a
    //  masked slot sorts by the fill key (NaN for a floating dtype, the dtype maximum otherwise), and `==` / `!=` treat
    //  two masked slots as equal and a masked slot as unequal to every value, while MASKING every comparison that reads
    //  one. So `ma.in1d([3, 5], ma.array([3, 9], mask=[0, 1]))` is [True, --]: 5 sorts next to the masked 9.
    //
    //  The former implementation ran the plain np.* set operation over the compressed (unmasked) values and added one
    //  trailing masked entry by a presence rule. That never masked an in1d / isin flag, kept NaN and masked slots apart
    //  where the sort mixes them, and missed every result shaped by NumPy's comparisons (the corpus' 8 isin / in1d cases
    //  all failed once their `[known gap]` excuse was removed).
    //
    //  One divergence is inherent: NumPy's 'quicksort' (np.unique's argsort with return_inverse, MaskedArray.sort's
    //  default kind) is NOT stable — 32/64-bit dtypes go through x86-simd-sort's AVX2 / AVX-512 networks, the others
    //  through introsort past 16 elements — so its order among EQUAL keys is host-dependent. NumSharp's sort is stable.
    //  The two agree whenever every group of equal sort keys is interchangeable; they can differ when a masked slot
    //  (keyed by the fill value) ties with an unmasked value equal to that key (NaN, the dtype maximum, True for bool,
    //  inf+infj for complex), or two unmasked values compare equal with different bits (+0.0 / -0.0, NaN payloads).
    // ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    public partial class MaskedArrayModule
    {
        /// <summary>
        ///     The result of <see cref="unique(object,bool,bool)"/> with a return flag: NumPy's
        ///     <c>(MaskedArray, ndarray[, ndarray])</c> tuple. It converts implicitly to <see cref="NDMaskedArray"/> (the
        ///     values), deconstructs into the present outputs, and indexes them in NumPy's order.
        /// </summary>
        /// <remarks>
        ///     NumPy returns a bare masked array without flags and a tuple with them; C# cannot vary a return type on a
        ///     runtime flag, so the flag-taking overload always returns this struct and the flag-less
        ///     <see cref="unique(object)"/> keeps returning the masked array itself.
        /// </remarks>
        public readonly struct UniqueResult : INDArrayCarrier
        {
            /// <summary>The sorted unique values (NumPy's <c>output[0].view(MaskedArray)</c>). Always present.</summary>
            public NDMaskedArray values { get; }

            /// <summary>The first-occurrence indices (int64), or <c>null</c> when <c>return_index</c> was false.</summary>
            public NDArray indices { get; }

            /// <summary>The reconstruction indices (int64, shaped like the input), or <c>null</c> when
            ///     <c>return_inverse</c> was false.</summary>
            public NDArray inverse_indices { get; }

            /// <summary>Wraps the outputs <see cref="unique(object,bool,bool)"/> computed.</summary>
            /// <param name="values">The unique values.</param>
            /// <param name="indices">The first-occurrence indices, or null.</param>
            /// <param name="inverse">The inverse indices, or null.</param>
            internal UniqueResult(NDMaskedArray values, NDArray indices, NDArray inverse)
            {
                this.values = values;
                this.indices = indices;
                inverse_indices = inverse;
            }

            /// <summary>The number of present outputs (1 plus one per requested flag), NumPy's tuple length.</summary>
            public int Length => 1 + (indices is null ? 0 : 1) + (inverse_indices is null ? 0 : 1);

            /// <summary>The <paramref name="index"/>-th present output in NumPy's order: the values, then the indices,
            ///     then the inverse.</summary>
            /// <param name="index">0 for the values, then 1 and 2 for the present index arrays.</param>
            /// <returns>The <see cref="NDMaskedArray"/> values or an <see cref="NDArray"/> of indices.</returns>
            /// <exception cref="IndexOutOfRangeException"><paramref name="index"/> is not below <see cref="Length"/>
            ///     (Python's <c>IndexError: tuple index out of range</c>).</exception>
            public object this[int index]
            {
                get
                {
                    if (index == 0)
                        return values;
                    if (index == 1 && indices is not null)
                        return indices;
                    if (index == 1 && inverse_indices is not null)
                        return inverse_indices;
                    if (index == 2 && indices is not null && inverse_indices is not null)
                        return inverse_indices;
                    throw new IndexOutOfRangeException("tuple index out of range");
                }
            }

            /// <summary>The values: NumPy's first tuple element, and the whole result of a flag-less call.</summary>
            /// <param name="result">The result.</param>
            /// <returns><see cref="values"/>.</returns>
            public static implicit operator NDMaskedArray(UniqueResult result) => result.values;

            /// <summary>Deconstructs a two-output result: <c>var (u, inv) = np.ma.unique(a, return_inverse: true);</c></summary>
            /// <param name="values">The unique values.</param>
            /// <param name="second">The one requested index array.</param>
            /// <exception cref="InvalidOperationException">The result does not hold exactly two outputs (Python's
            ///     unpacking ValueError).</exception>
            public void Deconstruct(out NDMaskedArray values, out NDArray second)
            {
                EnsureArity(2);
                values = this.values;
                second = indices ?? inverse_indices;
            }

            /// <summary>Deconstructs a three-output result: <c>var (u, idx, inv) = np.ma.unique(a, true, true);</c></summary>
            /// <param name="values">The unique values.</param>
            /// <param name="indices">The first-occurrence indices.</param>
            /// <param name="inverse_indices">The inverse indices.</param>
            /// <exception cref="InvalidOperationException">The result does not hold three outputs.</exception>
            public void Deconstruct(out NDMaskedArray values, out NDArray indices, out NDArray inverse_indices)
            {
                EnsureArity(3);
                values = this.values;
                indices = this.indices;
                inverse_indices = this.inverse_indices;
            }

            /// <summary>Refuses a deconstruction into a different number of outputs than the result holds.</summary>
            /// <param name="n">The deconstruction's arity.</param>
            /// <exception cref="InvalidOperationException">The arity differs from <see cref="Length"/>.</exception>
            private void EnsureArity(int n)
            {
                if (Length != n)
                    throw new InvalidOperationException(
                        Length < n
                            ? $"not enough values to unpack (expected {n}, got {Length})"
                            : $"too many values to unpack (expected {n})");
            }

            /// <inheritdoc/>
            void INDArrayCarrier.YieldTo(NDScope scope)
            {
                scope.Returns(values?._data);
                scope.Returns(values?._mask);
                // A stored fill created with the values (FinalizeFrom's converted copy) leaves with them.
                scope.Returns(values?._fill_value);
                scope.Returns(indices);
                scope.Returns(inverse_indices);
            }
        }

        /// <summary>
        ///     Finds the unique elements of an array (NumPy's <c>ma.unique</c> without flags): the sorted unique values of
        ///     the flattened operand, where every masked element sorts and compares as ONE value.
        /// </summary>
        /// <param name="ar1">Operand (<see cref="NDMaskedArray"/>, <see cref="NDArray"/>, scalar or array-like).</param>
        /// <returns>The 1-D masked array of unique values; it inherits a masked operand's hard-mask flag and fill value
        ///     (converted to the dtype).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ar1"/> is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand whose answer sorts or compares the
        ///     strings (NumPy reads them): the next phase of NumSharp's string support.</exception>
        /// <remarks>
        ///     NumPy's <c>np.unique(ar1).view(MaskedArray)</c>, whose <c>_unique1d</c> runs with the subclass:
        ///     <c>ar = ar.flatten()</c>; without a masked element the integer / complex dtypes take the hash path (sorted
        ///     distinct values); otherwise <c>ar.sort()</c> (masked slots last, keyed by the fill value) and
        ///     <c>mask[1:] = aux[1:] != aux[:-1]</c> through the masked <c>!=</c>, so all masked slots collapse into the
        ///     first one in sorted order, and with an unmasked NaN last the NaNs collapse into the first one. The kept
        ///     masked slot holds that element's data. See the file header for the tie-order divergence.
        ///     <para>Scoped by hand: the flattened copy, the sort permutation and the keep mask are reclaimed; only the
        ///     result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray unique(object ar1)
        {
            using var scope = NDScope.Open();
            return Yield(scope, UniqueCore(ar1, false, false, out _, out _));
        }

        /// <summary>
        ///     Finds the unique elements of an array, with NumPy's optional index outputs (<c>ma.unique(ar1,
        ///     return_index=False, return_inverse=False)</c>).
        /// </summary>
        /// <param name="ar1">Operand (<see cref="NDMaskedArray"/>, <see cref="NDArray"/>, scalar or array-like).</param>
        /// <param name="return_index">Also return the index of each unique value's first occurrence in the flattened
        ///     operand (NumPy sorts with the stable 'mergesort' here).</param>
        /// <param name="return_inverse">Also return the indices into the unique values that rebuild the operand, shaped
        ///     like the operand (NumPy 2.x).</param>
        /// <returns>The <see cref="UniqueResult"/>: the values (see <see cref="unique(object)"/>) and the requested int64
        ///     index arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ar1"/> is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand whose answer sorts or compares the
        ///     strings: the next phase of NumSharp's string support.</exception>
        /// <remarks>
        ///     With a flag NumPy argsorts instead of sorting in place (<c>perm = ar.argsort(kind='mergesort' if
        ///     return_index else 'quicksort')</c>, <c>aux = ar[perm]</c>), then <c>perm[mask]</c> and <c>inv_idx[perm] =
        ///     cumsum(mask) - 1</c>. Only <c>return_inverse</c> without <c>return_index</c> uses NumPy's unstable kind.
        ///     <para>Scoped by hand, as <see cref="unique(object)"/>; the index arrays leave with the values.</para>
        /// </remarks>
        public UniqueResult unique(object ar1, bool return_index = false, bool return_inverse = false)
        {
            using var scope = NDScope.Open();
            var values = UniqueCore(ar1, return_index, return_inverse, out var index, out var inverse);
            Yield(scope, values);
            scope.Returns(index);
            scope.Returns(inverse);
            return new UniqueResult(values, index, inverse);
        }

        /// <summary>
        ///     NumPy's <c>ma.unique</c> body: <c>np.unique(ar1, return_index, return_inverse)</c>, then
        ///     <c>output[0].view(MaskedArray)</c>.
        /// </summary>
        /// <param name="ar1">The operand.</param>
        /// <param name="return_index">Compute the first-occurrence indices.</param>
        /// <param name="return_inverse">Compute the inverse indices.</param>
        /// <param name="index">The first-occurrence indices, or null.</param>
        /// <param name="inverse">The inverse indices (shaped like the operand), or null.</param>
        /// <returns>The unique values as a masked array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ar1"/> is null.</exception>
        private NDMaskedArray UniqueCore(object ar1, bool return_index, bool return_inverse, out NDArray index, out NDArray inverse)
        {
            if (ar1 is NDMaskedArray masked)
            {
                // np.unique does not dispatch on a MaskedArray: _unique1d runs with the subclass flowing through, and
                // `.view(MaskedArray)` finalizes the values (the fill converted to their dtype).
                var values = Unique1d(masked, return_index, return_inverse, out index, out inverse);
                return new NDMaskedArray(values._data, values._mask).FinalizeFrom(values);
            }
            // A plain operand: np.unique proper (`np.asanyarray(ar1)` stays an ndarray), then `.view(MaskedArray)` —
            // nomask and no fill value.
            var r = np.unique(AsData(ar1), return_index, return_inverse);
            index = return_index ? r.indices : null;
            inverse = return_inverse ? r.inverse_indices : null;
            return new NDMaskedArray(r.values, null);
        }

        /// <summary>
        ///     <c>numpy.lib._arraysetops_impl._unique1d(ar, return_index, return_inverse)</c> (axis None, equal_nan True,
        ///     sorted True) over a masked operand, statement by statement.
        /// </summary>
        /// <param name="source">The masked operand (any shape).</param>
        /// <param name="return_index">Compute the first-occurrence indices.</param>
        /// <param name="return_inverse">Compute the inverse indices.</param>
        /// <param name="index">The first-occurrence indices, or null.</param>
        /// <param name="inverse">The inverse indices, shaped like <paramref name="source"/>, or null.</param>
        /// <returns><c>aux[mask]</c>: the unique values, attributes from the flattened copy.</returns>
        /// <exception cref="StringComputationNotSupportedException">The answer reads text.</exception>
        private NDMaskedArray Unique1d(NDMaskedArray source, bool return_index, bool return_inverse, out NDArray index, out NDArray inverse)
        {
            index = null;
            inverse = null;
            // `inverse_shape = ar.shape` (np.unique), taken before `ar = np.asanyarray(ar).flatten()`.
            var inverseShape = DimsOf(source._data);
            // `_arraymethod('flatten')`: a C-order copy of data and mask that keeps the fill value and hardness.
            var ar = flatten(source);
            bool optionalIndices = return_index || return_inverse;
            // `if not optional_indices and not return_counts and not np.ma.is_masked(ar)`: NumPy's C++ hash path for the
            // integer, complex and string dtypes, then `hash_unique.sort()`; `conv.wrap` keeps ar's attributes. NumSharp's
            // np.unique takes the same per-dtype choice (and the same first-occurrence representative for complex).
            if (!optionalIndices && !is_masked(ar) && IsHashUniqueDtype(ar.typecode))
                return new NDMaskedArray(np.unique(ar._data).values, null).UpdateFrom(ar);

            NDMaskedArray aux;
            NDArray perm = null;
            if (optionalIndices)
            {
                // `perm = ar.argsort(kind='mergesort' if return_index else 'quicksort'); aux = ar[perm]`.
                perm = SetOpArgsort(ar);
                aux = MaskedTake(ar, perm);
            }
            else
            {
                // `ar.sort(); aux = ar`.
                SortInPlace(ar);
                aux = ar;
            }

            long n = aux.size;
            // `mask = np.empty(aux.shape, dtype=np.bool); mask[:1] = True`.
            var keep = np.empty(new Shape(n), np.@bool);
            if (n > 0)
                keep[":1"] = NDArray.Scalar(true);
            // `if equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1])`: a masked last
            // element is `masked`, whose truth is False, so the NaN collapse runs only behind an UNMASKED NaN.
            if (n > 0 && IsNaNCapable(aux.typecode) && LastIsUnmaskedNaN(aux))
            {
                // complex: `np.searchsorted(np.isnan(aux), True, side='left')`; otherwise `np.searchsorted(aux, aux[-1],
                // side='left')`. Both search the RAW data — masked slots included — because MaskedArray has no searchsorted.
                long firstNan;
                if (aux.typecode == NPTypeCode.Complex)
                    firstNan = np.searchsorted(np.isnan(aux._data), NDArray.Scalar(true), "left").GetInt64();
                else
                    firstNan = np.searchsorted(aux._data, aux._data[$"{n - 1}:{n}"].reshape(Shape.Scalar), "left").GetInt64();
                // `if aux_firstnan > 0: mask[1:aux_firstnan] = (aux[1:aux_firstnan] != aux[:aux_firstnan - 1])`.
                if (firstNan > 0)
                    keep[$"1:{firstNan}"] = MaskedEqualityData(MaskedSlice(aux, $"1:{firstNan}"),
                                                               MaskedSlice(aux, $":{firstNan - 1}"), equal: false);
                // `mask[aux_firstnan] = True; mask[aux_firstnan + 1:] = False`.
                keep[$"{firstNan}:{firstNan + 1}"] = NDArray.Scalar(true);
                keep[$"{firstNan + 1}:"] = NDArray.Scalar(false);
            }
            else
            {
                // `mask[1:] = aux[1:] != aux[:-1]`: the masked `!=` — two masked slots are equal, a masked slot is
                // unequal to every value — whose DATA the plain bool array receives.
                keep["1:"] = MaskedEqualityData(MaskedSlice(aux, "1:"), MaskedSlice(aux, ":-1"), equal: false);
            }

            // `ret = (aux[mask],)`.
            var values = MaskedTake(aux, keep);
            // `if return_index: ret += (perm[mask],)`.
            if (return_index)
                index = perm[keep];
            if (return_inverse)
            {
                // `imask = np.cumsum(mask) - 1; inv_idx = np.empty(mask.shape, dtype=np.intp); inv_idx[perm] = imask`,
                // returned reshaped to the operand's shape.
                var imask = np.subtract(np.cumsum(keep), NDArray.Scalar(1L));
                var inv = np.empty(new Shape(n), np.int64);
                inv[perm] = imask;
                inverse = inv.reshape(inverseShape);
            }
            return values;
        }

        /// <summary>
        ///     Whether NumPy 2.4.2's <c>_unique_hash</c> handles the dtype (its C++ map holds the integer types, the
        ///     complex types and the string types; bool and the floating types fall to the sort path).
        /// </summary>
        /// <param name="tc">The data's storage lane.</param>
        /// <returns>True for the hash-path dtypes (NumSharp's char rides with the 16-bit unsigned integer).</returns>
        private static bool IsHashUniqueDtype(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                or NPTypeCode.Complex or NPTypeCode.String => true,
            _ => false,
        };

        /// <summary>Whether the dtype is NumPy kind <c>'f'</c> or <c>'c'</c> (the NaN-capable kinds of
        ///     <c>_unique1d</c>'s <c>"cfmM"</c> test; NumSharp has no datetime dtypes).</summary>
        /// <param name="tc">The data's storage lane.</param>
        /// <returns>True for float16, float32, float64 and complex128.</returns>
        private static bool IsNaNCapable(NPTypeCode tc) =>
            tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary><c>np.isnan(aux[-1])</c> taken as a truth value: false when the last element is masked (NumPy's
        ///     <c>masked</c> is falsy), else whether its value is NaN (a complex with either part NaN).</summary>
        /// <param name="aux">A non-empty 1-D masked array of a NaN-capable dtype.</param>
        /// <returns>Whether the NaN collapse runs.</returns>
        private static bool LastIsUnmaskedNaN(NDMaskedArray aux)
        {
            long last = aux.size - 1;
            if (aux._mask is not null && aux._mask.GetBoolean(last))
                return false;
            return np.isnan(aux._data[$"{last}:{last + 1}"]).GetBoolean(0);
        }

        /// <summary>
        ///     <c>a.argsort(kind=...)</c> on a masked array, as the set operations call it (NumPy's
        ///     <c>MaskedArray.argsort</c> with the default axis and <c>endwith=True</c>): masked slots keyed by NaN for a
        ///     floating dtype and by the dtype maximum otherwise, then the stable sort that serves both of NumPy's kinds.
        /// </summary>
        /// <param name="a">The masked operand.</param>
        /// <returns>The int64 permutation (of the flattened array from two dimensions on, NumPy's deprecated default).</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with keys to compare.</exception>
        /// <remarks>
        ///     A text operand whose every element is masked needs no string comparison: every key is the array's fill
        ///     value, so the stable order is the identity. NumPy's <c>self.filled(None)</c> still reads the fill value
        ///     there, which materializes the text default on <paramref name="a"/>, and so does this method.
        /// </remarks>
        private NDArray SetOpArgsort(NDMaskedArray a)
        {
            if (a._data.typecode == NPTypeCode.String && a._mask is not null && a.size > 0 && np.all(a._mask))
            {
                // `fill_value = minimum_fill_value(self)` is None for text: `self.filled(None)` reads `self.fill_value`.
                _ = a.fill_value;
                return np.arange(a.size);
            }
            return MaxKeyPartitionArgsort(a) ?? argsort(a, endwith: true);
        }

        /// <summary>
        ///     The minimum size from which <see cref="MaxKeyPartitionArgsort"/> pays for its extra steps (two index scans,
        ///     two gathers, one maximum and one join); below it the full radix over the filled keys is cheaper.
        /// </summary>
        private const long MaxKeyPartitionMinSize = 4096;

        /// <summary>
        ///     The masked argsort (<c>endwith=True</c>) of a 1-D array of an integer key dtype, computed without the fill:
        ///     the unmasked values' own stable argsort, then the masked slots in input order — or null when that shortcut
        ///     does not apply.
        /// </summary>
        /// <param name="a">The masked operand.</param>
        /// <returns>The same permutation <see cref="argsort(object,bool,object)"/> returns, or null.</returns>
        /// <remarks>
        ///     NumPy keys every masked slot of an integer dtype by the dtype MAXIMUM (<c>minimum_fill_value</c>), so a
        ///     stable sort puts the masked slots after every smaller value, in input order. Unless an unmasked value equals
        ///     that maximum (a tie that interleaves by index: then null), the order is exactly the unmasked values' stable
        ///     order followed by the masked indices. Sorting the unmasked values alone keeps the radix to the bytes they
        ///     span: the maximum makes every key byte significant (8 passes for int64 where small values need 2 or 3),
        ///     which cost ~16 % of a 1M masked argsort. Bool (its maximum, True, is an ordinary value), decimal (a
        ///     comparison sort) and the NaN-keyed float types keep the general path, as do arrays below
        ///     <see cref="MaxKeyPartitionMinSize"/> elements and arrays with no masked slot.
        /// </remarks>
        private NDArray MaxKeyPartitionArgsort(NDMaskedArray a)
        {
            if (a.ndim != 1 || a.size < MaxKeyPartitionMinSize || a._mask is null)
                return null;
            if (a.typecode is not (NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Char or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64))
                return null;
            var masked = np.flatnonzero(a._mask);
            if (masked.size == 0)
                return null;
            var unmasked = np.flatnonzero(np.logical_not(a._mask));
            var values = a._data[unmasked];
            // An unmasked value equal to the fill key ties with the masked slots: the general path interleaves them.
            if (values.size > 0 && Equals(np.max(values).GetAtIndex(0), MinFill(a.typecode)))
                return null;
            return np.concatenate(new[] { unmasked[np.argsort(values, -1)], masked });
        }

        /// <summary>
        ///     <c>MaskedArray.sort()</c> (axis -1, the default kind, <c>endwith=True</c>) IN PLACE on a set operation's own
        ///     intermediate: without a mask the data is sorted; with one, <c>sidx = self.argsort(axis=-1)</c> and
        ///     <c>self[...] = np.take_along_axis(self, sidx, axis=-1)</c> through the masked setter.
        /// </summary>
        /// <param name="x">A masked array no caller holds (a flattened or concatenated copy).</param>
        /// <exception cref="StringComputationNotSupportedException">A text operand with keys to compare.</exception>
        /// <remarks>
        ///     In place, not through <see cref="sort(object,int?,bool,object)"/>: that one sorts a COPY, which NumPy's
        ///     <c>np.array(a, copy=True, subok=True)</c> finalizes (converting a stored text fill to the dtype), while
        ///     <c>ar.sort()</c> keeps the stored fill as it is. The masked setter keeps NumPy's hard-mask rule: on a HARD
        ///     mask the old and the sorted masks are OR-ed and masked slots keep their old data.
        /// </remarks>
        private void SortInPlace(NDMaskedArray x)
        {
            // `if self._mask is nomask: ndarray.sort(self, axis=axis); return`.
            if (x._mask is null)
            {
                if (x.size > 1)
                    SortDataInPlace(x._data);
                return;
            }
            NDArray sidx;
            if (x._data.typecode == NPTypeCode.String && x.size > 0 && np.all(x._mask))
            {
                // Every key is the fill value: the stable order is the identity (see SetOpArgsort), after the fill read.
                _ = x.fill_value;
                sidx = np.arange(x.size).reshape(DimsOf(x._data));
            }
            else
                sidx = MaxKeyPartitionArgsort(x) ?? argsort(x, -1, endwith: true);
            var taken = new NDMaskedArray(np.take_along_axis(x._data, sidx, -1), np.take_along_axis(x._mask, sidx, -1));
            x[Slice.Ellipsis] = taken;
        }

        /// <summary>
        ///     <c>ndarray.sort(axis=-1)</c> (NumPy's default kind) of an unmasked set-operation intermediate, in place: the
        ///     stable order, and for float32 / float64 lines longer than one element NumPy's canonical NaN.
        /// </summary>
        /// <param name="data">A writeable array no caller holds.</param>
        /// <exception cref="StringComputationNotSupportedException">Text with two or more elements to order.</exception>
        /// <remarks>
        ///     NumPy's default kind is unstable: equal values with different bits (<c>+0.0</c> / <c>-0.0</c>) land in a
        ///     host-dependent order. NumSharp keeps the STABLE order (<see cref="np.argsort(NDArray,int?)"/> + a gather),
        ///     which is NumPy's whenever its order does not matter — the case set the ma_setops oracle tier holds — and
        ///     unlike <see cref="np.sort"/>'s value radix it does not put every <c>-0.0</c> first. The float32 / float64
        ///     kernels NumPy's default kind runs (x86-simd-sort) rewrite every NaN of a line longer than one element to the
        ///     positive quiet NaN, so the gathered NaNs are rewritten too; float16 and complex keep their bits there.
        /// </remarks>
        private static void SortDataInPlace(NDArray data)
        {
            var sorted = np.take_along_axis(data, np.argsort(data, -1), -1);
            if (data.typecode is NPTypeCode.Single or NPTypeCode.Double && data.Shape.dimensions[data.ndim - 1] > 1)
            {
                var nan = np.isnan(sorted);
                if (np.any(nan))
                {
                    NDArray canonical = data.typecode == NPTypeCode.Single
                        ? NDArray.Scalar(BitConverter.UInt32BitsToSingle(0x7FC00000U))
                        : NDArray.Scalar(BitConverter.UInt64BitsToDouble(0x7FF8000000000000UL));
                    np.copyto(sorted, canonical, where: nan);
                }
            }
            np.copyto(data, sorted);
        }

        /// <summary><c>x[index]</c> for an integer or boolean index ARRAY (NumPy's <c>MaskedArray.__getitem__</c>): the
        ///     data and the mask indexed alike (copies), the attributes copied by <c>_update_from</c>.</summary>
        /// <param name="x">The masked array.</param>
        /// <param name="index">An int64 or bool index array.</param>
        /// <returns>The selection as a masked array (0-d for a 0-d integer index).</returns>
        private static NDMaskedArray MaskedTake(NDMaskedArray x, NDArray index)
            => new NDMaskedArray(x._data[index], x._mask is null ? null : x._mask[index]).UpdateFrom(x);

        /// <summary><c>x[slice]</c> for a basic slice (NumPy's <c>MaskedArray.__getitem__</c>): views of the data and the
        ///     mask, the attributes copied by <c>_update_from</c>.</summary>
        /// <param name="x">The masked array.</param>
        /// <param name="slice">The slice expression (<c>"1:"</c>, <c>":-1"</c>, …).</param>
        /// <returns>The sliced masked array (a view).</returns>
        private static NDMaskedArray MaskedSlice(NDMaskedArray x, string slice)
            => new NDMaskedArray(x._data[slice], x._mask is null ? null : x._mask[slice]).UpdateFrom(x);

        /// <summary>
        ///     The masked <c>self == other</c> / <c>self != other</c> of two masked arrays, NumPy's
        ///     <c>MaskedArray._comparison(other, operator.eq / operator.ne)</c>: the mask is <c>mask_or(self.mask,
        ///     other.mask)</c> (a copy, shrunk to nomask), and wherever it is set the data is the comparison of the two
        ///     MASKS — two masked slots are equal, a masked slot is unequal to a value — instead of the values'.
        /// </summary>
        /// <param name="self">The left operand.</param>
        /// <param name="other">The right operand, the same shape.</param>
        /// <param name="equal">True for <c>==</c>, false for <c>!=</c>.</param>
        /// <returns>The masked boolean comparison: its data and its mask only (a soft array with no fill value).</returns>
        /// <exception cref="StringComputationNotSupportedException">Text values to compare (an unmasked slot).</exception>
        /// <remarks>
        ///     NumSharp does not overload <c>==</c> / <c>!=</c> on masked arrays (they would shadow reference equality), so
        ///     the set operations call this directly. Where every slot is masked the values never decide the answer, so
        ///     they are not compared — the only case in which text needs no string comparison.
        ///     <para>NumPy's <c>_comparison</c> also gives the result self's attributes, the fill value cast to bool
        ///     (<c>_check_fill_value(fill, np.bool)</c>, which falls back to the default instead of raising). No set
        ///     operation reads them — in1d concatenates the comparison into a new array, intersect1d indexes with it,
        ///     unique keeps its data — so they are not computed (for a text fill the cast to bool is the next phase).</para>
        /// </remarks>
        private NDMaskedArray MaskedEquality(NDMaskedArray self, NDMaskedArray other, bool equal)
        {
            var smask = self._mask;
            var omask = other._mask;
            // `mask = mask_or(smask, omask, copy=True)`, shrunk to nomask.
            NDArray mask = smask is null && omask is null ? null : ShrinkMask(Or(smask, omask));
            Func<NDArray, NDArray, NDArray> compare = equal
                ? (x, y) => np.equal(x, y)
                : (x, y) => np.not_equal(x, y);
            NDArray check;
            if (mask is not null)
            {
                // `check = np.where(mask, compare(smask, omask), check)`: nomask is NumPy's np.False_.
                var maskCompare = compare(smask ?? NDArray.Scalar(false), omask ?? NDArray.Scalar(false));
                check = np.all(mask)
                    ? np.broadcast_to(maskCompare, Shape.Adopt(mask.Shape.dimensions)).copy()
                    : np.where(mask, maskCompare, compare(self._data, other._data));
            }
            else
                check = compare(self._data, other._data);
            return new NDMaskedArray(check, mask);
        }

        /// <summary>The DATA of <see cref="MaskedEquality"/>: what NumPy's <c>mask[1:] = aux[1:] != aux[:-1]</c> stores into
        ///     a plain bool array.</summary>
        /// <param name="self">The left operand.</param>
        /// <param name="other">The right operand, the same shape.</param>
        /// <param name="equal">True for <c>==</c>, false for <c>!=</c>.</param>
        /// <returns>The plain boolean comparison data.</returns>
        /// <exception cref="StringComputationNotSupportedException">Text values to compare.</exception>
        private NDArray MaskedEqualityData(NDMaskedArray self, NDMaskedArray other, bool equal)
            => MaskedEquality(self, other, equal)._data;

        /// <summary>
        ///     Element-wise membership of a 1-D (or flattened) array in another (NumPy's <c>ma.in1d</c>): True where an
        ///     element of <paramref name="ar1"/> is also in <paramref name="ar2"/>, MASKED wherever NumPy's sorted
        ///     comparison reads a masked slot.
        /// </summary>
        /// <param name="ar1">The values to test (flattened; the result takes its shape).</param>
        /// <param name="ar2">The values to test against.</param>
        /// <param name="assume_unique">Treat both inputs as already unique (NumPy skips the two <c>unique</c> calls; the
        ///     answer for non-unique inputs is then undefined, as in NumPy). <paramref name="ar1"/> is not flattened then:
        ///     it must be 1-D for the concatenation and the result is 1-D.</param>
        /// <param name="invert">Return the complement (True where the element is NOT in <paramref name="ar2"/>).</param>
        /// <returns>The masked boolean flags, shaped like <paramref name="ar1"/> (1-D with
        ///     <paramref name="assume_unique"/>); a 0-d operand gives a 0-d result, or <see cref="NDMasked"/> when masked.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError"><paramref name="assume_unique"/> with a 0-d <paramref name="ar1"/> (<c>len() of
        ///     unsized object</c>).</exception>
        /// <exception cref="ValueError"><paramref name="assume_unique"/> with operands np.concatenate cannot join.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text whose answer sorts or compares the strings, or
        ///     formats numbers into text.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>ar1, rev_idx = unique(ar1, return_inverse=True); ar2 = unique(ar2)</c>; <c>ar =
        ///     ma.concatenate((ar1, ar2))</c>; <c>order = ar.argsort(kind='mergesort')</c>; <c>sar = ar[order]</c>;
        ///     <c>bool_ar = (sar[1:] == sar[:-1])</c> (<c>!=</c> for invert) through the masked comparison; <c>flag =
        ///     ma.concatenate((bool_ar, [invert]))</c>; <c>indx = order.argsort(kind='mergesort')[:len(ar1)]</c>; and
        ///     <c>flag[indx][rev_idx]</c>. A value whose sorted neighbour is masked is therefore masked: <c>ma.in1d([3,
        ///     5], ma.array([3, 9], mask=[0, 1]))</c> is [True, --]. See the file header for the tie-order divergence.
        ///     An EMPTY <paramref name="ar1"/> with a text operand is answered without the text: NumPy formats the numbers
        ///     into the concatenated text and selects nothing, so the result is the empty, unmasked flag array.
        ///     <para>Scoped by hand: the unique values, the concatenation, the permutations and the flags are reclaimed;
        ///     only the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray in1d(object ar1, object ar2, bool assume_unique = false, bool invert = false)
        {
            using var scope = NDScope.Open();
            return Yield(scope, In1dCore(ar1, ar2, assume_unique, invert));
        }

        /// <summary>The body of <see cref="in1d"/> (and of <see cref="isin"/> and <see cref="setdiff1d"/>), without a scope
        ///     of its own.</summary>
        /// <param name="ar1">The values to test.</param>
        /// <param name="ar2">The values to test against.</param>
        /// <param name="assume_unique">Skip the two <c>unique</c> calls.</param>
        /// <param name="invert">Return the complement.</param>
        /// <returns>The masked flags.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">A 0-d <paramref name="ar1"/> with <paramref name="assume_unique"/>.</exception>
        private NDMaskedArray In1dCore(object ar1, object ar2, bool assume_unique, bool invert)
        {
            if (ar1 is null)
                throw new ArgumentNullException(nameof(ar1));
            if (ar2 is null)
                throw new ArgumentNullException(nameof(ar2));
            // An empty ar1 beside a text operand: NumPy reaches `flag[indx][rev_idx]` with an empty `indx` after
            // formatting numbers into the joined text — the empty, unmasked flags, shaped like ar1.
            var d1 = AsData(ar1);
            if (!assume_unique && d1.size == 0 && (d1.typecode == NPTypeCode.String || AsData(ar2).typecode == NPTypeCode.String))
                return new NDMaskedArray(np.zeros(DimsOf(d1), np.@bool), null);

            object a1 = ar1, a2 = ar2;
            NDArray revIdx = null;
            if (!assume_unique)
            {
                // `ar1, rev_idx = unique(ar1, return_inverse=True); ar2 = unique(ar2)`.
                a1 = UniqueCore(ar1, false, true, out _, out revIdx);
                a2 = UniqueCore(ar2, false, false, out _, out _);
            }
            // `ar = ma.concatenate((ar1, ar2))`: a masked array (nomask when neither side has a mask).
            var ar = concatenate(new[] { a1, a2 });
            // `order = ar.argsort(kind='mergesort')`: values from ar1 come before equal values from ar2.
            var order = SetOpArgsort(ar);
            // `sar = ar[order]`.
            var sar = MaskedTake(ar, order);
            // `bool_ar = (sar[1:] != sar[:-1])` for invert, else `==`.
            var boolAr = MaskedEquality(MaskedSlice(sar, "1:"), MaskedSlice(sar, ":-1"), equal: !invert);
            // `flag = ma.concatenate((bool_ar, [invert]))`.
            var flag = concatenate(new object[] { boolAr, np.array(new[] { invert }) });
            // `indx = order.argsort(kind='mergesort')[:len(ar1)]`: len() of a 0-d ar1 is Python's TypeError. `order` is a
            // permutation of 0..N-1, so no two keys are equal and its stable argsort IS its inverse permutation: one
            // scatter, `inv[order] = arange(N)`, gives the same indices as the sort (6.7 ms against 39 ms at 1M).
            var first = AsData(a1);
            if (first.ndim == 0)
                throw new TypeError("len() of unsized object");
            var inverseOrder = np.empty(new Shape(order.size), np.int64);
            inverseOrder[order] = np.arange(order.size);
            var indx = inverseOrder[$":{first.Shape.dimensions[0]}"];
            var taken = MaskedTake(flag, indx);
            if (assume_unique)
                return taken;
            // `flag[indx][rev_idx]`: rev_idx has ar1's original shape, so the flags do too; a 0-d rev_idx selects a
            // scalar, which NumPy returns as `masked` or as a plain np.bool_ (a 0-d array without a mask here).
            var result = MaskedTake(taken, revIdx);
            if (result.ndim == 0)
            {
                if (result._mask is not null && result._mask.GetBoolean())
                    return NDMasked;
                return new NDMaskedArray(result._data, null);
            }
            return result;
        }

        /// <summary>
        ///     Element-wise membership test (NumPy's <c>ma.isin</c>): <see cref="in1d"/> of <paramref name="element"/>,
        ///     reshaped to <paramref name="element"/>'s shape.
        /// </summary>
        /// <param name="element">Values to test.</param>
        /// <param name="test_elements">The values to test against.</param>
        /// <param name="assume_unique">Treat both inputs as already unique (see <see cref="in1d"/>).</param>
        /// <param name="invert">Return the complement.</param>
        /// <returns>The masked boolean flags, shaped like <paramref name="element"/>.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="TypeError">See <see cref="in1d"/>.</exception>
        /// <exception cref="ValueError">See <see cref="in1d"/>; also <paramref name="assume_unique"/> with an N-D
        ///     <paramref name="element"/>, whose flags cannot take its shape.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="in1d"/>.</exception>
        /// <remarks>
        ///     NumPy's <c>element = ma.asarray(element)</c> (which keeps a masked operand's mask, fill value and hardness),
        ///     then <c>in1d(element, test_elements, ...).reshape(element.shape)</c>. The flags are masked wherever NumPy's
        ///     sorted comparison reads a masked slot (see <see cref="in1d"/>).
        ///     <para>Scoped by hand, as <see cref="in1d"/>.</para>
        /// </remarks>
        public NDMaskedArray isin(object element, object test_elements, bool assume_unique = false, bool invert = false)
        {
            using var scope = NDScope.Open();
            var e = asarray(element);
            var flags = In1dCore(e, test_elements, assume_unique, invert);
            // `.reshape(element.shape)` of the `masked` constant (a 0-d element whose flag is masked) is a NEW 0-d masked
            // array, downcast from the constant: float64 data 0.0, mask True, no fill value, soft.
            if (ReferenceEquals(flags, NDMasked))
                return Yield(scope, new NDMaskedArray(np.zeros(DimsOf(e._data), np.float64), np.ones(DimsOf(e._data), np.@bool)));
            return Yield(scope, reshape(flags, DimsOf(e._data)));
        }

        /// <summary>
        ///     The sorted values common to both inputs (NumPy's <c>ma.intersect1d</c>). A masked value is a value: it is
        ///     common when both inputs hold one, and the result keeps one masked slot then.
        /// </summary>
        /// <param name="ar1">First operand.</param>
        /// <param name="ar2">Second operand.</param>
        /// <param name="assume_unique">Treat both inputs as already unique (NumPy skips the two <c>unique</c> calls and
        ///     concatenates them as given; the answer for non-unique inputs is then undefined).</param>
        /// <returns>The masked array of common values, in the promoted dtype.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text whose answer sorts or compares the strings.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>aux = ma.concatenate((unique(ar1), unique(ar2)))</c> (the operands themselves with
        ///     <paramref name="assume_unique"/>); <c>aux.sort()</c>; <c>aux[:-1][aux[1:] == aux[:-1]]</c>, the masked
        ///     <c>==</c> selecting by its data. The result's fill value is the concatenation's (none, or the text default
        ///     the sort materialized).
        ///     <para>Scoped by hand: the unique values, the joined copy and the comparison are reclaimed; only the result's
        ///     data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray intersect1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            var aux = assume_unique
                ? concatenate(new[] { ar1, ar2 })
                : concatenate(new object[] { UniqueCore(ar1, false, false, out _, out _), UniqueCore(ar2, false, false, out _, out _) });
            SortInPlace(aux);
            // `aux[:-1][aux[1:] == aux[:-1]]`: indexing with a masked bool array selects by its data.
            var same = MaskedEquality(MaskedSlice(aux, "1:"), MaskedSlice(aux, ":-1"), equal: true);
            return Yield(scope, MaskedTake(MaskedSlice(aux, ":-1"), same._data));
        }

        /// <summary>
        ///     The sorted values in exactly one of the inputs (NumPy's <c>ma.setxor1d</c>), computed over the FILLED values
        ///     of the joined uniques.
        /// </summary>
        /// <param name="ar1">First operand.</param>
        /// <param name="ar2">Second operand.</param>
        /// <param name="assume_unique">Treat both inputs as already unique (NumPy skips the two <c>unique</c> calls).</param>
        /// <returns>The masked array of the symmetric difference, in the promoted dtype.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text whose answer sorts or compares the strings.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>ar1 = unique(ar1); ar2 = unique(ar2)</c>; <c>aux = ma.concatenate((ar1, ar2),
        ///     axis=None)</c>; an empty <c>aux</c> is the result; <c>aux.sort()</c>; <c>auxf = aux.filled()</c>; <c>flag =
        ///     ma.concatenate(([True], auxf[1:] != auxf[:-1], [True]))</c>; <c>aux[flag[1:] == flag[:-1]]</c>. The
        ///     neighbours are compared FILLED with the default fill value, so a masked slot equals an unmasked value equal
        ///     to that fill (999999 for an integer dtype) — NumPy's answer, reproduced.
        ///     <para>Scoped by hand: the unique values, the joined copy, the filled copy and the flags are reclaimed; only
        ///     the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray setxor1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            object a1 = ar1, a2 = ar2;
            if (!assume_unique)
            {
                a1 = UniqueCore(ar1, false, false, out _, out _);
                a2 = UniqueCore(ar2, false, false, out _, out _);
            }
            var aux = concatenate(new[] { a1, a2 }, axis: null);
            if (aux.size == 0)
                return Yield(scope, aux);
            SortInPlace(aux);
            // `auxf = aux.filled()`: the default fill (for text, the materialized 'N/A').
            var auxf = aux.filled();
            long n = auxf.size;
            var interior = np.not_equal(auxf["1:"], auxf[":-1"]);
            // `flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))`: all three unmasked, so plain data.
            var flag = np.concatenate(new[] { np.array(new[] { true }), interior, np.array(new[] { true }) });
            // `flag2 = (flag[1:] == flag[:-1])` over two nomask masked arrays: the plain comparison.
            var flag2 = np.equal(flag[$"1:{n + 1}"], flag[$":{n}"]);
            return Yield(scope, MaskedTake(aux, flag2));
        }

        /// <summary>
        ///     The sorted unique values of both inputs together (NumPy's <c>ma.union1d</c>): <see cref="unique(object)"/>
        ///     of their flattened concatenation.
        /// </summary>
        /// <param name="ar1">First operand.</param>
        /// <param name="ar2">Second operand.</param>
        /// <returns>The masked array of the union, in the promoted dtype.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text whose answer sorts or compares the strings.</exception>
        /// <remarks>NumPy's <c>unique(ma.concatenate((ar1, ar2), axis=None))</c>.
        ///     <para>Scoped by hand: the joined copy is reclaimed; only the result's data and mask leave.</para></remarks>
        public NDMaskedArray union1d(object ar1, object ar2)
        {
            using var scope = NDScope.Open();
            return Yield(scope, UniqueCore(concatenate(new[] { ar1, ar2 }, axis: null), false, false, out _, out _));
        }

        /// <summary>
        ///     The sorted unique values of <paramref name="ar1"/> that are not in <paramref name="ar2"/> (NumPy's
        ///     <c>ma.setdiff1d</c>), in <paramref name="ar1"/>'s dtype.
        /// </summary>
        /// <param name="ar1">The operand to keep values from.</param>
        /// <param name="ar2">The operand whose values are removed.</param>
        /// <param name="assume_unique">Treat both inputs as already unique: <paramref name="ar1"/> is only flattened
        ///     (NumPy's <c>ma.asarray(ar1).ravel()</c>) and <paramref name="ar2"/> is used as given.</param>
        /// <returns>The masked array of <paramref name="ar1"/>'s remaining values.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">Text whose answer sorts or compares the strings.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>ar1 = unique(ar1); ar2 = unique(ar2)</c>, then <c>ar1[in1d(ar1, ar2,
        ///     assume_unique=True, invert=True)]</c>, the masked flags selecting by their data. So a value of
        ///     <paramref name="ar1"/> is kept wherever the inverted flag's data is True — including a flag that is masked.
        ///     An empty <paramref name="ar1"/> beside text selects nothing (see <see cref="in1d"/>).
        ///     <para>Scoped by hand: the unique values and the flags are reclaimed; only the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray setdiff1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            NDMaskedArray a1;
            object a2 = ar2;
            if (assume_unique)
                a1 = ravel(asarray(ar1));
            else
            {
                a1 = UniqueCore(ar1, false, false, out _, out _);
                a2 = UniqueCore(ar2, false, false, out _, out _);
            }
            // An empty ar1 selects nothing whatever in1d answers (its text formatting is the next phase).
            if (a1.size == 0 && (a1.typecode == NPTypeCode.String || AsData(a2).typecode == NPTypeCode.String))
                return Yield(scope, a1);
            var keep = In1dCore(a1, a2, assume_unique: true, invert: true);
            return Yield(scope, MaskedTake(a1, keep._data));
        }
    }
}
