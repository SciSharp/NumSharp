using System;

// =============================================================================
// NDListArray.Manipulation.cs — the ndarray members of NumPy's object array of lists that only move the lists:
// transposition, reshaping, copies and views, selection, and the attributes that describe the outer array
// =============================================================================
//
// NumPy's object array holds one pointer per element (8 bytes on a 64-bit host), and every member below works on those
// pointers alone: a transpose, a reshape, a take or a repeat moves the POINTERS — the Python lists are shared, never
// copied (a copy is shallow). NDListArray holds the same structure: the pieces once (Units / Offsets) and, per element,
// the range of its pieces (Starts / Ends, int64 arrays of the outer shape, and SelfPiece). So each member is the same
// ndarray operation applied to the three range arrays, with the pieces shared:
//
//   * the range arrays are int64 (bool for SelfPiece): 8 bytes per element like NumPy's pointers, so the operation's
//     view / copy decision, the result's strides IN BYTES, its flags (C / F contiguous, OWNDATA, WRITEABLE: a diagonal is
//     read-only) and every error (texts that never name a dtype) are NumPy's for the object array;
//   * the result holds new references on the shared pieces (a view or a copy may outlive this array), and its own range
//     arrays.
//
// NumPy's `base`: a view's base is the array owning the pointers it views (a view of a view collapses to that owner, as
// NumPy's does; a view of an intermediate copy — reshape in the other order — has that temporary as its base); a copy has
// none. Here the owner is the memory the range arrays view (their storage's base), and `base` is a NEW list array over it,
// as NDArray.base is a new array over its owner's storage: the same pointers and lists, a new object to dispose (NumPy
// returns the owner object itself).
//
// NumPy returns the element itself where its result is one element (`a.take(1)` is the Python list): NumSharp has no
// object scalar, so the result is a 0-d NDListArray, whose item() is that list — the house rule of NDArray.take, whose
// 0-d result stands for NumPy's scalar.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        // =============================================================================================================
        // The attributes of the outer array
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>strides</c> in bytes: an object array steps 8 bytes per element (one pointer), and the range arrays
        ///     are int64, so a view's strides are NumPy's (a transpose of a <c>(2, 3)</c> array is <c>(8, 24)</c>).
        /// </summary>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public long[] strides
        {
            get
            {
                ThrowIfDisposed();
                return Starts.strides;
            }
        }

        /// <summary>NumPy's <c>itemsize</c> of an object array: 8, the size of one element's pointer.</summary>
        public int itemsize => 8;

        /// <summary>NumPy's <c>nbytes</c> of an object array: <c>size * 8</c> (the pointers, not the lists they point to).</summary>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public long nbytes => size * 8;

        /// <summary>NumPy's <c>device</c> (the Array API's): always <c>"cpu"</c>.</summary>
        public string device => "cpu";

        /// <summary>
        ///     NumPy's <c>flags</c> of the outer array: C / F contiguity, OWNDATA (a copy owns its pointers, a view does
        ///     not), WRITEABLE (false for a diagonal view, as in NumPy) and ALIGNED — the flags of the range arrays, which
        ///     have the object array's layout.
        /// </summary>
        /// <remarks>The object is the flags of an internal range array: reading it is NumPy's answer; setting a flag through
        ///     it changes that array only (an <see cref="NDListArray"/> has no element assignment for WRITEABLE to guard).</remarks>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArrayFlags flags
        {
            get
            {
                ThrowIfDisposed();
                return Starts.flags;
            }
        }

        /// <summary>
        ///     NumPy's <c>base</c>: the list array owning the pointers this one views — the array a view was taken from (a view
        ///     of a view gives that owner, as in NumPy), or NumPy's intermediate copy for a view of one (a reshape in the other
        ///     order) — or null for an array that owns its pointers (a copy, a take, a repeat).
        /// </summary>
        /// <remarks>Each read is a NEW list array over the owner's pointers and the shared lists (new references, released by
        ///     its <see cref="Dispose"/>), as <see cref="NDArray.@base"/> is a new array over its owner's storage; NumPy returns
        ///     the owner object itself.</remarks>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray @base
        {
            get
            {
                ThrowIfDisposed();
                // The memory the range arrays view: NumSharp collapses a view of a view to the owning storage, as NumPy
                // collapses base, and the three ranges were made by the same operations, so they view owners of one layout.
                var owner = Starts.Storage._baseStorage;
                if (owner is null)
                    return null;
                return new NDListArray(Alias(Units), Alias(Offsets), new NDArray(owner),
                    new NDArray(Ends.Storage._baseStorage ?? Ends.Storage),
                    SelfPiece is null ? null : new NDArray(SelfPiece.Storage._baseStorage ?? SelfPiece.Storage), Kind);
            }
        }

        /// <summary>
        ///     NumPy's <c>to_device(device, stream=None)</c> (the Array API's): this array itself for <c>"cpu"</c>, the only
        ///     device.
        /// </summary>
        /// <param name="device">The device: <c>"cpu"</c>.</param>
        /// <param name="stream">The Array API's stream argument: must be null.</param>
        /// <returns>This array.</returns>
        /// <exception cref="ArgumentNullException">A null device (NumPy's TypeError <c>to_device() argument 1 must be str,
        ///     not None</c>).</exception>
        /// <exception cref="ArgumentException">Another device (<c>Unsupported device: gpu. Only 'cpu' is accepted.</c>) or a
        ///     stream (<c>The stream argument in to_device() is not supported</c>): <see cref="NDArray.to_device"/>'s
        ///     types and texts.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray to_device(string device, object stream = null)
        {
            ThrowIfDisposed();
            // NDArray's own check (the device text, then the stream) answers for the object array too; it returns the
            // range array itself, which is not this method's result.
            Starts.to_device(device, stream);
            return this;
        }

        // =============================================================================================================
        // Transposition
        // =============================================================================================================

        /// <summary>NumPy's <c>T</c>: the transposed view (the axes reversed), sharing the lists.</summary>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray T => Derive(static a => a.T);

        /// <summary>NumPy's <c>mT</c>: the view with the last two axes swapped.</summary>
        /// <exception cref="ArgumentException">Fewer than two axes (<c>matrix transpose with ndim &lt; 2 is undefined</c>,
        ///     NumPy's ValueError text under <see cref="NDArray.mT"/>'s type).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray mT => Derive(static a => a.mT);

        /// <summary>NumPy's <c>transpose(*axes)</c>: the view with the axes permuted (reversed when none are given).</summary>
        /// <param name="axes">The permutation, or none to reverse the axes.</param>
        /// <returns>The view.</returns>
        /// <exception cref="ValueError">The axes do not match the array (<c>axes don't match array</c>) or repeat
        ///     (<c>repeated axis in transpose</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array (<c>axis 2 is out of bounds for array of
        ///     dimension 2</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray transpose(params int[] axes)
            => Derive(a => a.transpose(axes is { Length: > 0 } ? axes : null));

        /// <summary>NumPy's <c>swapaxes(axis1, axis2)</c>: the view with two axes interchanged.</summary>
        /// <param name="axis1">The first axis (negative counts from the end).</param>
        /// <param name="axis2">The second axis.</param>
        /// <returns>The view.</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray swapaxes(int axis1, int axis2) => Derive(a => a.swapaxes(axis1, axis2));

        // =============================================================================================================
        // Reshaping
        // =============================================================================================================

        /// <summary>NumPy's <c>reshape(*shape)</c> in C order: a view when the pointers allow one, else a copy.</summary>
        /// <param name="shape">The new shape (one entry may be -1, inferred).</param>
        /// <returns>The reshaped list array.</returns>
        /// <exception cref="IncorrectShapeException">The size differs (<c>cannot reshape array of size 6 into shape
        ///     (4,)</c>).</exception>
        /// <exception cref="ValueError">More than one -1 (<c>can only specify one unknown dimension</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray reshape(params long[] shape) => Derive(a => a.reshape(shape));

        /// <summary>NumPy's <c>reshape(shape)</c> in C order, the shape given as a <see cref="NumSharp.Shape"/>.</summary>
        /// <param name="newShape">The new shape.</param>
        /// <returns>The reshaped list array.</returns>
        /// <exception cref="IncorrectShapeException">The size differs.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray reshape(Shape newShape) => Derive(a => a.reshape(newShape));

        /// <summary>NumPy's <c>reshape(shape, order=...)</c>: the elements read and placed in <paramref name="order"/>.</summary>
        /// <param name="newShape">The new shape.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c> or <c>'A'</c> (F when the array is F-contiguous and not C).</param>
        /// <returns>The reshaped list array (a view of an intermediate copy when the order needs one, as in NumPy).</returns>
        /// <exception cref="IncorrectShapeException">The size differs.</exception>
        /// <exception cref="ValueError">An order NumPy does not take.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray reshape(Shape newShape, char order) => Derive(a => a.reshape(newShape, order));

        /// <summary>NumPy's <c>ravel(order='C')</c>: the 1-D list array, a view whenever the pointers are contiguous in that
        ///     order (<c>'K'</c>: their memory order).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>The flattened list array.</returns>
        /// <exception cref="ValueError">An order NumPy does not take.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray ravel(char order = 'C') => Derive(a => a.ravel(order));

        /// <summary>NumPy's <c>flatten(order='C')</c>: a 1-D COPY (the lists shared, the pointers copied).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>The flattened copy.</returns>
        /// <exception cref="ValueError">An order NumPy does not take.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray flatten(char order = 'C') => Derive(a => a.flatten(order));

        /// <summary>
        ///     NumPy's <c>squeeze(axis=None)</c>: the view without the length-1 axes (or without the one given); with none to
        ///     remove this array ITSELF, as NumPy returns its receiver.
        /// </summary>
        /// <param name="axis">The axis to remove, or null for every length-1 axis.</param>
        /// <returns>The view, or this array.</returns>
        /// <exception cref="ValueError">An axis of another length (<c>cannot select an axis to squeeze out which has size not
        ///     equal to one</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray squeeze(int? axis = null)
        {
            ThrowIfDisposed();
            // PyArray_Squeeze returns its receiver when there is no length-1 axis to remove (a.squeeze() is a); an axis
            // given explicitly always has length 1 or raises, so only the all-axes form can find none.
            if (axis is null && Array.IndexOf(Starts.Shape.dimensions, 1L) < 0)
                return this;
            return Derive(a => a.squeeze(axis));
        }

        // =============================================================================================================
        // Copies and views
        // =============================================================================================================

        /// <summary>NumPy's <c>copy(order='C')</c>: a SHALLOW copy — new pointers in <paramref name="order"/>, the same lists
        ///     (NumSharp's lists cannot be changed in place, so sharing them is invisible but for the memory saved).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>The copy (it owns its pointers: no base).</returns>
        /// <exception cref="ValueError">An order NumPy does not take.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray copy(char order = 'C') => Derive(a => a.copy(order));

        /// <summary>NumPy's <c>view()</c>: a new list array over the same pointers (its base is this array's owner).</summary>
        /// <returns>The view.</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray view() => Derive(static a => a.view());

        // =============================================================================================================
        // Selection
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>take(indices, axis=None, out=None, mode='raise')</c>: the lists at <paramref name="indices"/> (of the
        ///     flattened array, or along <paramref name="axis"/>), a new array of pointers to the same lists.
        /// </summary>
        /// <param name="indices">The integer indices (any shape; NumPy's conversion and its TypeError for a float array).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">NumPy's <c>out=</c>: must be null (an <see cref="NDListArray"/> has no element assignment).</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> (null is <c>"raise"</c>).</param>
        /// <returns>The taken lists.</returns>
        /// <exception cref="IndexError">An index out of bounds under <c>"raise"</c> (<c>index 7 is out of bounds for axis 0
        ///     with size 6</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">Indices that are not integers.</exception>
        /// <exception cref="ValueError">A mode NumPy does not take.</exception>
        /// <exception cref="NotSupportedException">An <paramref name="out"/> array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray take(NDArray indices, int? axis = null, NDListArray @out = null, string mode = "raise")
        {
            RefuseOut(@out, "take");
            return Derive(a => a.take(indices, axis, null, mode));
        }

        /// <summary>NumPy's <c>take</c> with one index: a 0-d list array holding the list (NumPy returns the list itself,
        ///     which this array's <see cref="item()"/> gives).</summary>
        /// <param name="index">The index (of the flattened array, or along <paramref name="axis"/>).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">Must be null.</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>The taken list (a 0-d list array without an axis; along an axis, the sub-array).</returns>
        /// <exception cref="IndexError">An index out of bounds under <c>"raise"</c>.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError">A mode NumPy does not take.</exception>
        /// <exception cref="NotSupportedException">An <paramref name="out"/> array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray take(long index, int? axis = null, NDListArray @out = null, string mode = "raise")
        {
            RefuseOut(@out, "take");
            return Derive(a => a.take(index, axis, null, mode));
        }

        /// <summary>NumPy's <c>compress(condition, axis=None, out=None)</c>: the lists where <paramref name="condition"/> is
        ///     true (of the flattened array, or the slices along <paramref name="axis"/>).</summary>
        /// <param name="condition">A 1-D bool array (a shorter one selects its prefix; a longer one with a true entry past the
        ///     end is NumPy's IndexError).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">Must be null.</param>
        /// <returns>The selected lists (a copy of the pointers).</returns>
        /// <exception cref="IndexError">A true entry past the axis (<c>index 6 is out of bounds for axis 0 with size
        ///     6</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">An <paramref name="out"/> array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray compress(NDArray condition, int? axis = null, NDListArray @out = null)
        {
            RefuseOut(@out, "compress");
            return Derive(a => a.compress(condition, axis));
        }

        /// <summary>NumPy's <c>repeat(repeats, axis=None)</c>: every list repeated (of the flattened array, or along
        ///     <paramref name="axis"/>), the pointers copied, the lists shared.</summary>
        /// <param name="repeats">The count for every element.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>The repeated lists.</returns>
        /// <exception cref="ArgumentException">A negative count: <see cref="NDArray.repeat(long, int?)"/>'s text, <c>repeats
        ///     may not contain negative values</c> (NumPy's ValueError reads <c>negative dimensions are not allowed</c>: the
        ///     wording divergence NDArray.repeat has for every dtype).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray repeat(long repeats, int? axis = null) => Derive(a => a.repeat(repeats, axis));

        /// <summary>NumPy's <c>repeat(repeats, axis=None)</c> with a count per element (broadcast against the axis).</summary>
        /// <param name="repeats">The counts.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>The repeated lists.</returns>
        /// <exception cref="ArgumentException">A negative count (NDArray.repeat's text, see the single-count form), or counts
        ///     that do not broadcast (<c>operands could not be broadcast together with shape (6,) (2,)</c>, NumPy's
        ///     ValueError text).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray repeat(NDArray repeats, int? axis = null) => Derive(a => a.repeat(repeats, axis));

        /// <summary>NumPy's <c>diagonal(offset=0, axis1=0, axis2=1)</c>: the READ-ONLY view of the lists on a
        ///     diagonal.</summary>
        /// <param name="offset">The diagonal's offset from the main one (positive above).</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays.</param>
        /// <param name="axis2">The second axis.</param>
        /// <returns>The view (WRITEABLE false, as NumPy's).</returns>
        /// <exception cref="ValueError">Fewer than two axes (<c>diag requires an array of at least two
        ///     dimensions</c>), or equal axes.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) => Derive(a => a.diagonal(offset, axis1, axis2));

        // =============================================================================================================
        // Internals
        // =============================================================================================================

        /// <summary>
        ///     One structural operation applied to the range arrays (<see cref="Starts"/>, <see cref="Ends"/> and, when
        ///     present, <see cref="SelfPiece"/>), giving a list array over the results that shares the pieces.
        /// </summary>
        /// <param name="op">The operation: an NDArray member that depends only on the layout (never on the values).</param>
        /// <returns>The new list array, owning the three results and new references on the pieces.</returns>
        /// <exception cref="Exception">The operation's own error, raised by its first call (on <see cref="Starts"/>), before
        ///     anything is allocated for the result.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        private NDListArray Derive(Func<NDArray, NDArray> op)
        {
            ThrowIfDisposed();
            NDArray starts = null, ends = null, self = null;
            try
            {
                // The ranges share one layout, so the operation fails (with NumPy's error) on the first array or on none.
                starts = op(Starts);
                ends = op(Ends);
                self = SelfPiece is null ? null : op(SelfPiece);
            }
            catch
            {
                starts?.Dispose();
                ends?.Dispose();
                self?.Dispose();
                throw;
            }
            // An operation can hand back its receiver itself (NDArray.view of a view is new, but a no-op squeeze or reshape
            // could be the same object): the list array must own arrays of its own, so such a result is aliased.
            starts = Own(starts, Starts);
            ends = Own(ends, Ends);
            self = SelfPiece is null ? null : Own(self, SelfPiece);
            return new NDListArray(Alias(Units), Alias(Offsets), starts, ends, self, Kind);
        }

        /// <summary>An operation's result as an array the new list array may own: a new alias when the operation returned
        ///     its receiver.</summary>
        /// <param name="result">The operation's result.</param>
        /// <param name="receiver">The range array the operation ran on.</param>
        /// <returns>The result, or a new alias of the receiver's storage.</returns>
        private static NDArray Own(NDArray result, NDArray receiver)
            => ReferenceEquals(result, receiver) ? Alias(receiver) : result;

        /// <summary>A new array over the same storage (a counted reference: the alias keeps the memory alive).</summary>
        /// <param name="array">The array.</param>
        /// <returns>The alias.</returns>
        private static NDArray Alias(NDArray array) => new NDArray(array.Storage.Alias()) { TensorEngine = array.TensorEngine };

        /// <summary>Refuses NumPy's <c>out=</c> of a selection member: the result would be written into another list
        ///     array's pointers, and an <see cref="NDListArray"/> has no element assignment.</summary>
        /// <param name="out">The out array, or null.</param>
        /// <param name="member">The member's name.</param>
        /// <exception cref="NotSupportedException"><paramref name="out"/> is not null.</exception>
        private static void RefuseOut(NDListArray @out, string member)
        {
            if (@out is not null)
                throw new NotSupportedException($"NDListArray.{member}(out=...): an NDListArray has no element assignment to write into.");
        }
    }
}
