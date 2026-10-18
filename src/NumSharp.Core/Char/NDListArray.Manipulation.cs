using System;

// =============================================================================
// NDListArray.Manipulation.cs — the ndarray members of NumPy's object array of lists that only move the lists:
// transposition, reshaping, copies and views, selection, and the attributes that describe the outer array
// =============================================================================
//
// NumPy's object array holds one pointer per element (8 bytes on a 64-bit host), and every member below works on those
// pointers alone: a transpose, a reshape, a take or a repeat moves the POINTERS — the Python lists are shared, never
// copied (a copy is shallow). NDListArray holds the same structure: one list id per element (Ids, an int64 array of the
// outer shape) pointing into the shared lists (NDListHeap). So each member is the same ndarray operation applied to the
// id array, with the lists shared:
//
//   * the ids are int64: 8 bytes per element like NumPy's pointers, so the operation's view / copy decision, the result's
//     strides IN BYTES, its flags (C / F contiguous, OWNDATA, WRITEABLE: a diagonal is read-only) and every error (texts
//     that never name a dtype) are NumPy's for the object array;
//   * the result holds a new reference on the shared lists (a view or a copy may outlive this array) and its own ids.
//
// NumPy's `base`: a view's base is the array owning the pointers it views (a view of a view collapses to that owner, as
// NumPy's does; a view of an intermediate copy — reshape in the other order — has that temporary as its base); a copy has
// none. Here the owner is the memory the ids view (their storage's base), and `base` is a NEW list array over it, as
// NDArray.base is a new array over its owner's storage: the same pointers and lists, a new object to dispose (NumPy
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
                return Ids.strides;
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
        ///     not), WRITEABLE (false for a diagonal view, as in NumPy) and ALIGNED — the flags of the id array, which has
        ///     the object array's layout.
        /// </summary>
        /// <remarks>The object is the flags of the id array: reading it is NumPy's answer, and setting WRITEABLE through it
        ///     (<c>flags.writeable = false</c>, or <see cref="setflags"/>) guards the element assignments as NumPy's does.</remarks>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArrayFlags flags
        {
            get
            {
                ThrowIfDisposed();
                return Ids.flags;
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
                // The memory the ids view: NumSharp collapses a view of a view to the owning storage, as NumPy collapses
                // base.
                var owner = Ids.Storage._baseStorage;
                if (owner is null)
                    return null;
                return new NDListArray(new NDArray(owner), Heap.Share());
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
            // id array itself, which is not this method's result.
#pragma warning disable NDW012 // not a leak: NDArray.to_device("cpu") returns the id array itself, which this array owns
            Ids.to_device(device, stream);
#pragma warning restore NDW012
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

        /// <summary>NumPy's <c>reshape(*shape)</c> with separate integers, NumPy's defaults <c>order='C'</c> and
        ///     <c>copy=None</c>: a view when the pointers allow one, else a copy (see <see cref="reshape(Shape, char, bool?)"/>,
        ///     the spelling that takes <c>order</c> and <c>copy</c>).</summary>
        /// <param name="shape">The new shape (one entry may be -1, inferred).</param>
        /// <returns>The reshaped list array.</returns>
        /// <exception cref="IncorrectShapeException">The size differs (<c>cannot reshape array of size 6 into shape
        ///     (4,)</c>).</exception>
        /// <exception cref="ValueError">More than one -1 (<c>can only specify one unknown dimension</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray reshape(params long[] shape) => Derive(a => a.reshape(shape));

        /// <summary>
        ///     NumPy's <c>reshape(shape, /, *, order='C', copy=None)</c>: the elements read in <paramref name="order"/> and
        ///     placed in the new shape in the same order — a view when the pointers allow one (else a view of a copy, as in
        ///     NumPy), always a copy for <c>copy=true</c>, and NumPy's ValueError for <c>copy=false</c> when no view can be
        ///     made.
        /// </summary>
        /// <param name="shape">The new shape: a <see cref="NumSharp.Shape"/>, or a C# <c>long[]</c> / <c>int[]</c> / tuple /
        ///     integer, which convert to one (one entry may be -1, inferred).</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c> or <c>'A'</c> (F when the array is F-contiguous and not C); NumPy refuses
        ///     <c>'K'</c>.</param>
        /// <param name="copy">NumPy's <c>copy</c>: null (a copy only when needed), true (always a copy, in
        ///     <paramref name="order"/>), false (never: a reshape that needs a copy raises).</param>
        /// <returns>The reshaped list array (it shares the lists in every case: a copy copies the pointers).</returns>
        /// <exception cref="IncorrectShapeException">The size differs.</exception>
        /// <exception cref="ValueError">An order NumPy does not take (<c>order 'K' is not permitted for reshaping</c>); more
        ///     than one -1; <c>Unable to avoid creating a copy while reshaping.</c> for <c>copy=false</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     NumPy takes the shape as one tuple OR as separate integers, then the keyword-only <c>order</c> and
        ///     <c>copy</c>; C# has no keyword-only parameter after a <c>params</c> array, so the two spellings are two members:
        ///     <see cref="reshape(long[])"/> for separate integers and this one for everything else.
        /// </remarks>
        public NDListArray reshape(Shape shape, char order = 'C', bool? copy = null)
        {
            ThrowIfDisposed();
            if (copy == true)
            {
                // _reshape_with_copy_arg's NPY_COPY_ALWAYS: PyArray_NewCopy(array, order) — 'A' meaning F for an
                // F-contiguous, non-C array, as np.copy reads it — then the reshape of that copy, a view of it (OWNDATA false,
                // as in NumPy). The reshape of the copy raises NumPy's order and size errors, the same texts in the same
                // order ('K' first, then the size), so they are not checked twice.
                using var pointers = Ids.copy(order);
                var viewOfCopy = pointers.reshape(shape, order);
                return Over(viewOfCopy);
            }
            var ids = Ids.reshape(shape, order);
            if (copy == false && !IsReshapeView(ids, order))
            {
                // NPY_COPY_NEVER: _attempt_nocopy_reshape found no view.
                ids.Dispose();
                throw new ValueError("Unable to avoid creating a copy while reshaping.");
            }
            return Over(ids);
        }

        /// <summary>
        ///     Whether a reshape of this array's pointers is the VIEW <c>_reshape_with_copy_arg</c> gives without copying: an
        ///     array contiguous in the requested order is reshaped as a view without trying (<c>'A'</c> reads F only for an
        ///     F-contiguous, non-C array; an empty array is contiguous in both orders), and any other needs
        ///     <c>_attempt_nocopy_reshape</c> to have found one.
        /// </summary>
        /// <param name="reshaped">NumSharp's reshape of <see cref="Ids"/> (a view, or a copy where no view exists).</param>
        /// <param name="order">The requested order (<c>'C'</c>, <c>'F'</c> or <c>'A'</c>; <c>'K'</c> already raised).</param>
        /// <returns>True when NumPy's <c>copy=False</c> reshape succeeds.</returns>
        /// <remarks>
        ///     NumSharp's reshape takes NumPy's two routes, but its view is not always the same buffer OBJECT (a reshape to the
        ///     same shape, or of an empty array, re-wraps the slice), so the copy is told apart by memory: a copy is a fresh
        ///     allocation, whose first element is never this array's first element.
        /// </remarks>
        private unsafe bool IsReshapeView(NDArray reshaped, char order)
        {
            var s = Ids.Shape;
            bool fOrder = order is 'F' or 'f' || (order is 'A' or 'a' && s.IsFContiguous && !s.IsContiguous);
            if (fOrder ? s.IsFContiguous : s.IsContiguous)
                return true;
            // The pointers are int64 ids: an element is 8 bytes.
            byte* first = Ids.Storage.Address + Ids.Shape.offset * sizeof(long);
            byte* reshapedFirst = reshaped.Storage.Address + reshaped.Shape.offset * sizeof(long);
            return first == reshapedFirst;
        }

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
            if (axis is null && Array.IndexOf(Ids.Shape.dimensions, 1L) < 0)
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

        /// <summary>
        ///     NumPy's <c>view([dtype][, type])</c> of an object array: a new list array over the same pointers (its base is
        ///     this array's owner). Only the object dtype keeps an array of references viewable, and only <c>ndarray</c> is a
        ///     class a list array can be viewed as here.
        /// </summary>
        /// <param name="dtype">NumPy's <c>dtype</c>, omitted for none. The object dtype — <c>typeof(object)</c>,
        ///     <c>np.object_</c>, or a NumPy spelling of it (<c>"O"</c>, <c>"|O"</c>, <c>"=O"</c>, <c>"&lt;O"</c>,
        ///     <c>"&gt;O"</c>, <c>"O8"</c>, <c>"O4"</c>, <c>"object"</c>, <c>"object_"</c>) — is the same view; any other dtype
        ///     (a <see cref="DType"/>, a CLR element type, an <see cref="NPTypeCode"/>, a dtype string, and null, which is
        ///     NumPy's <c>None</c> and converts to float64) is NumPy's TypeError. An ARRAY CLASS here is the
        ///     <paramref name="type"/> argument, as in NumPy.</param>
        /// <param name="type">NumPy's <c>type</c>, omitted for none: the class of the result — <c>typeof(NDArray)</c> or
        ///     <c>typeof(NDListArray)</c> (NumPy's <c>ndarray</c>) give the view; <c>typeof(NDCharArray)</c> is its finalize
        ///     error; <c>typeof(NDMatrix)</c> and <c>typeof(NDMaskedArray)</c> would be an object matrix / masked array, which
        ///     NumSharp has no type for. Anything else — null (NumPy's <c>None</c>) included — is not an array class.</param>
        /// <returns>The view.</returns>
        /// <exception cref="ValueError"><c>Cannot specify output type twice.</c> (an array class as <paramref name="dtype"/>
        ///     beside a <paramref name="type"/>); <c>Type must be a sub-type of ndarray type</c>; the class's finalize error
        ///     (<c>Can only create a chararray from string data.</c>, <c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError"><c>Cannot change data-type for array of references.</c> for a dtype that is not the
        ///     object dtype; <c>Cannot interpret '1' as a data type</c> for a value that names no dtype.</exception>
        /// <exception cref="NotSupportedException">A dtype string or CLR type NumSharp cannot parse (NumPy's <c>data type
        ///     'x' not understood</c>); a view as an object matrix, masked array or typed <c>NDArray&lt;T&gt;</c>.</exception>
        /// <exception cref="IncorrectShapeException">A matrix view of more than two axes where one has extent 0.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     One member, NumPy's signature. Both parameters are <c>[Optional]</c> objects without a default constant, so an
        ///     omitted argument arrives as <see cref="System.Reflection.Missing"/> while an explicit null stays NumPy's
        ///     <c>None</c> — <c>a.view(None)</c> and <c>a.view(type=None)</c> raise in NumPy, where <c>a.view()</c> does not.
        ///     The checks run in <c>array_view</c>'s order: the swap of an array class given as the dtype, the type, the
        ///     dtype's conversion, the class's finalize (PyArray_View builds the subclass object before it assigns the dtype),
        ///     then the dtype's refusal.
        /// </remarks>
        public NDListArray view([System.Runtime.InteropServices.Optional] object dtype, [System.Runtime.InteropServices.Optional] object type)
        {
            ThrowIfDisposed();
            bool hasDtype = dtype is not System.Reflection.Missing;
            bool hasType = type is not System.Reflection.Missing;
            // array_view: a type given as the dtype is the type when it is an ndarray subclass.
            if (hasDtype && IsArrayClass(dtype))
            {
                if (hasType)
                    throw new ValueError("Cannot specify output type twice.");
                type = dtype;
                hasType = true;
                hasDtype = false;
            }
            if (hasType && !IsArrayClass(type))
                throw new ValueError("Type must be a sub-type of ndarray type");
            // PyArray_DescrConverter runs before the view is made: an unknown name raises here.
            bool refused = hasDtype && !IsObjectDescr(dtype);
            var cls = hasType ? (Type)type : typeof(NDListArray);
            // PyArray_View makes the subclass object first: its __array_finalize__ sees the object dtype.
            if (cls == typeof(NDCharArray))
                throw NDCharArrayBoundary.FinalizeError();
            if (cls == typeof(NDMatrix) && Ids.ndim != 2)
                _ = Ids.Shape.AsMatrix(); // matrix's finalize rule: "shape too large to be a matrix." (no view is kept)
            // Then the dtype is assigned: _view_is_safe refuses every dtype but object for an array of references.
            if (refused)
                throw new TypeError(NDArray.ViewOfReferencesText);
            if (cls != typeof(NDListArray) && cls != typeof(NDArray))
                throw new NotSupportedException(
                    $"NumPy's view of an object array as {cls.Name} is an object array of that class, which NumSharp has no " +
                    "type for; view() gives the list array itself");
            return Derive(static a => a.view());
        }

        /// <summary>
        ///     Whether a view argument is an ARRAY CLASS (NumPy's <c>PyType_IsSubtype(type, &amp;PyArray_Type)</c>): an
        ///     <see cref="NDArray"/> class, or one of NumSharp's array classes that stand for an ndarray subclass without
        ///     deriving from <see cref="NDArray"/> (<see cref="NDListArray"/>, <see cref="NDMaskedArray"/>).
        /// </summary>
        /// <param name="value">The argument.</param>
        /// <returns>True for an array class.</returns>
        private static bool IsArrayClass(object value)
            => value is Type t && (typeof(NDArray).IsAssignableFrom(t) || t == typeof(NDListArray) || t == typeof(NDMaskedArray));

        /// <summary>
        ///     NumPy's <c>PyArray_DescrConverter</c> of a view's dtype argument, reduced to the question a list array asks:
        ///     whether it names the object dtype. Every other dtype is converted for its error only (an unknown name raises
        ///     before the view is refused, as in NumPy).
        /// </summary>
        /// <param name="dtype">The argument (null is NumPy's <c>None</c>, which converts to float64).</param>
        /// <returns>True for the object dtype.</returns>
        /// <exception cref="NotSupportedException">A string or CLR type NumSharp cannot parse as a dtype.</exception>
        /// <exception cref="TypeError">A value that names no dtype (<c>Cannot interpret '1' as a data type</c>).</exception>
        private static bool IsObjectDescr(object dtype)
        {
            switch (dtype)
            {
                case null:
                case DType:
                    return false;
                case Type t when t == typeof(object):
                    return true;
                case Type t:
                    _ = DType.From(t);
                    return false;
                case string s when s is "O" or "|O" or "=O" or "<O" or ">O" or "O8" or "O4" or "object" or "object_":
                    // np.dtype() reads these as dtype('O') (probed against NumPy 2.4.2; "O0" and "Object" are errors).
                    return true;
                case string s:
                    _ = np.dtype(s);
                    return false;
                case NPTypeCode c:
                    _ = DType.From(c);
                    return false;
                case NDScalarType st:
                    // np.object_ is the object dtype's scalar class; any other class names a dtype NumSharp has or refuses.
                    if (ReferenceEquals(st, NDScalarType.Object_))
                        return true;
                    _ = np.dtype(st);
                    return false;
                default:
                    throw new TypeError($"Cannot interpret '{dtype}' as a data type");
            }
        }

        // =============================================================================================================
        // Selection
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>take(indices, axis=None, out=None, mode='raise')</c>: the lists at <paramref name="indices"/> (of the
        ///     flattened array, or along <paramref name="axis"/>), a new array of pointers to the same lists.
        /// </summary>
        /// <param name="indices">The integer indices: an integer array of any shape (NumPy's conversion and its TypeError for
        ///     a float array), or one C# integer (<c>lists.take(1)</c>), which converts to the 0-d array NumPy's
        ///     <c>np.asarray(1)</c> is — the take of one index without an axis is then the element itself, a 0-d list array
        ///     standing for NumPy's returned list (its <see cref="item(long[])"/> is the list).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">NumPy's <c>out=</c>: a list array of the result's shape whose elements are set to the taken
        ///     lists (returned), or null for a new array.</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> (null is <c>"raise"</c>).</param>
        /// <returns>The taken lists (<paramref name="out"/> when given).</returns>
        /// <exception cref="IndexError">An index out of bounds under <c>"raise"</c> (<c>index 7 is out of bounds for axis 0
        ///     with size 6</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">Indices that are not integers.</exception>
        /// <exception cref="ValueError">A mode NumPy does not take; an <paramref name="out"/> of another shape (<c>output
        ///     array does not match result of ndarray.take</c>) or a read-only one (<c>WRITEBACKIFCOPY base is
        ///     read-only</c>), both checked before the indices' bounds.</exception>
        /// <exception cref="NotSupportedException">An <paramref name="out"/> of the other piece class (str and bytes) that
        ///     would receive a non-empty list.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature: NumPy's <c>indices</c> is an int or an int array_like, which this
        ///     <see cref="NDArray"/> parameter receives through NumSharp's implicit conversions.</remarks>
        public NDListArray take(NDArray indices, int? axis = null, NDListArray @out = null, string mode = "raise")
        {
            if (@out is not null)
                return TakeInto(@out, destination => Ids.take(indices, axis, destination, mode));
            // array_take returns PyArray_Return's result: a 0-d take is the element itself, the list.
            return Derive(a => a.take(indices, axis, null, mode)).MarkStandIn();
        }

        /// <summary>NumPy's <c>compress(condition, axis=None, out=None)</c>: the lists where <paramref name="condition"/> is
        ///     true (of the flattened array, or the slices along <paramref name="axis"/>).</summary>
        /// <param name="condition">A 1-D bool array (a shorter one selects its prefix; a longer one with a true entry past the
        ///     end is NumPy's IndexError).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">A list array of the result's shape to store into (NumPy's compress is a take of the condition's
        ///     nonzero indices, so its checks and texts are take's), or null.</param>
        /// <returns>The selected lists (a copy of the pointers), or <paramref name="out"/>.</returns>
        /// <exception cref="IndexError">A true entry past the axis (<c>index 6 is out of bounds for axis 0 with size
        ///     6</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError">An <paramref name="out"/> of another shape (<c>output array does not match result of
        ///     ndarray.take</c>) or a read-only one.</exception>
        /// <exception cref="NotSupportedException">An <paramref name="out"/> of the other piece class.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray compress(NDArray condition, int? axis = null, NDListArray @out = null)
        {
            if (@out is not null)
                return TakeInto(@out, destination => Ids.compress(condition, axis, destination));
            return Derive(a => a.compress(condition, axis));
        }

        /// <summary>NumPy's <c>repeat(repeats, axis=None)</c>: every list repeated (of the flattened array, or along
        ///     <paramref name="axis"/>), the pointers copied, the lists shared.</summary>
        /// <param name="repeats">The counts: one C# integer for every element (<c>lists.repeat(2)</c>, the 0-d array NumPy's
        ///     <c>np.asarray(2)</c> is), or an integer array of one count per element along the axis (broadcast against
        ///     it).</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <returns>The repeated lists.</returns>
        /// <exception cref="ArgumentException">A negative count: <see cref="NDArray.repeat(NDArray, int?)"/>'s text,
        ///     <c>repeats may not contain negative values</c> (NumPy's ValueError reads <c>negative dimensions are not
        ///     allowed</c>: the wording divergence NDArray.repeat has for every dtype); counts that do not broadcast
        ///     (<c>operands could not be broadcast together with shape (6,) (2,)</c>, NumPy's ValueError text).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature: NumPy's <c>repeats</c> is an int or an int array_like, which this
        ///     <see cref="NDArray"/> parameter receives through NumSharp's implicit conversions.</remarks>
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
        ///     One structural operation applied to the id array, giving a list array over the result that shares the lists.
        /// </summary>
        /// <param name="op">The operation: an NDArray member that depends only on the layout (never on the values).</param>
        /// <returns>The new list array, owning the result and a new reference on the lists.</returns>
        /// <exception cref="Exception">The operation's own error (NumPy's for the object array), before anything is
        ///     allocated for the result.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        private NDListArray Derive(Func<NDArray, NDArray> op)
        {
            ThrowIfDisposed();
            var ids = op(Ids);
            // An operation can hand back its receiver itself (a no-op reshape could be the same object): the list array
            // must own an array of its own, so such a result is aliased.
            if (ReferenceEquals(ids, Ids))
                ids = new NDArray(Ids.Storage.Alias()) { TensorEngine = Ids.TensorEngine };
            return Over(ids);
        }

    }
}
