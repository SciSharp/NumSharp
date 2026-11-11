namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's scalar C-API (numpy/_core/src/multiarray/scalarapi.c) over NumSharp's scalar model.
    //
    // NumSharp has no scalar classes: a NumPy scalar (np.float64(1.5), a.sum(), np.dot(v, v)) is a 0-d NDArray, and a 0-d
    // NDArray IS a scalar unless the caller built it as an array with np.array(x) / np.asarray(x) (IsExplicitZeroDimArray).
    // There is no scalar flag and no extra state: IsNumPyScalar computes the answer, and a scalar is immutable through
    // NumPy's own WRITEABLE flag, cleared where NumPy hands a scalar out.
    //
    // Every scalar passes one of NumPy's own seams, so a later phase that gives the scalars classes of their own
    // (NDFloat64 : NDFloating : ... : NDGeneric : NDArray) changes only what the seams build:
    //   - PyArray_Return: a fresh 0-d result becomes the scalar (the reductions, the products, NDArray.Scalar(...));
    //   - IsNumPyScalar: every rule that tells a scalar from a 0-d array reads only this (np.isscalar, the scalar-class
    //     isinstance, numpy.polynomial's operand protocol, numpy.matrix's scaling rule).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     True when this array stands for a NumPy SCALAR (<c>numpy.generic</c>): a 0-d array that was not built as an
        ///     explicit 0-d ARRAY by <c>np.array(x)</c> / <c>np.asarray(x)</c> (<see cref="IsExplicitZeroDimArray"/>). It is
        ///     computed, not stored: a 0-d array is a scalar by default.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Read this, never <c>ndim == 0</c> or the explicit mark directly, wherever NumPy's answer depends on scalar
        ///     against 0-d ndarray (<see cref="np.isscalar"/>, <see cref="NDScalarType.__instancecheck__"/>, numpy.matrix's
        ///     <c>m * x</c>, numpy.polynomial's operand protocol): it is the one place the later scalar-subclass phase turns
        ///     into a class test (<c>this is NDGeneric</c>).
        ///     </para>
        ///     <para>
        ///     <b>Accepted difference from NumPy:</b> a 0-d array built any other way than the two explicit spellings
        ///     (<c>np.zeros(())</c>, <c>x.reshape(())</c>, <c>np.squeeze(x)</c>, a copy of an explicit 0-d array) answers
        ///     true, where NumPy's object is an ndarray. Only <c>np.array</c> / <c>np.asarray</c> of a scalar argument mark
        ///     their result (see <see cref="IsExplicitZeroDimArray"/> for why the kernels' own 0-d allocations cannot).
        ///     </para>
        ///     <para>
        ///     A scalar is not necessarily read-only: NumPy's return points (<see cref="PyArray_Return"/>) clear WRITEABLE,
        ///     while a 0-d array allocated another way (<c>np.zeros(())</c>) keeps it, so <c>fill</c> still works there.
        ///     </para>
        /// </remarks>
        internal bool IsNumPyScalar => Shape.IsScalar && !IsExplicitZeroDimArray;

        /// <summary>
        ///     NumPy's <c>PyArray_Return</c> (C-API slot 76): the step that turns a FRESH result into what NumPy hands the
        ///     caller - the array itself when it has dimensions, NumPy's scalar when it is 0-d. NumSharp's scalar is the 0-d
        ///     array itself, made what NumPy's scalar is: read-only (WRITEABLE cleared) and never an explicit 0-d array.
        /// </summary>
        /// <returns>
        ///     This array. The scalar-subclass phase may return a NEW object here (the scalar of the array's class), so a
        ///     caller should return or store the RESULT of this call, not the array it called it on.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     <b>Where NumPy calls it, NumSharp calls this:</b> every exit of a reduction that hands back a 0-d result
        ///     (<c>axis=None</c>, an axis that reduces the last dimension away, <c>keepdims</c> over a 0-d input; probed 2.4.2:
        ///     <c>np.sum(a).flags.writeable is False</c>, and even <c>np.sum(np.array(5), keepdims=True)</c> comes back a
        ///     read-only scalar), the product gufuncs' 0-d answer without <c>out=</c> (<c>GufuncGuard.Deliver</c>: NumPy's
        ///     <c>replace_with_wrapped_result_and_return</c> → <c>npy_apply_wrap</c> with <c>return_scalar</c>),
        ///     <c>np.dot</c> / <c>linalg.det</c> / the index routines' 0-d answers, and the scalar constructors
        ///     (<see cref="Scalar{T}(T)"/> and its overloads: NumPy's scalar <c>tp_new</c> ends in <c>PyArray_ToScalar</c>,
        ///     the same conversion).
        ///     </para>
        ///     <para>
        ///     <b>Footgun - the array must be FRESH.</b> NumPy's function copies the element into a new scalar object and
        ///     releases the array; this one changes the array in place, so it must be an array nobody else holds: never an
        ///     <c>out=</c> operand (NumPy returns the out array itself, writeable; probed:
        ///     <c>np.sum(a, out=np.array(0)).flags.writeable is True</c>), never a caller's operand returned unchanged, never a
        ///     view of another array's memory (a scalar owns its value: a view would change when its base changes). A caller
        ///     with such an array copies it first (<c>r.copy().PyArray_Return()</c>).
        ///     </para>
        ///     <para>
        ///     A no-op for an array with dimensions, so every exit of a function can route through here: the
        ///     <c>keepdims</c> path reshapes to <c>(1,)*ndim</c> first and passes untouched.
        ///     </para>
        ///     <para>
        ///     The typed wrappers (<c>NDArray&lt;bool&gt;</c> of <c>np.all</c> / <c>np.any</c>, the <c>NDArray&lt;long&gt;</c>
        ///     index routines, <see cref="NDCharArray"/>'s scalar stand-ins) call it as a statement and keep their own
        ///     reference, which is exact while this returns the same object; the scalar-subclass phase revisits them.
        ///     </para>
        /// </remarks>
        internal NDArray PyArray_Return()
        {
            if (Shape.IsScalar)
            {
                // A scalar is immutable: NumPy's scalar object has no writable buffer (flags.writeable is False).
                if (Shape.IsWriteable)
                    Storage.SetShapeUnsafe(Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
                // A fresh result never carries the explicit-array mark; clearing it keeps the contract "what comes out of
                // here is a scalar" (IsNumPyScalar) even for a caller that broke the fresh-array rule above.
                IsExplicitZeroDimArray = false;
            }
            return this;
        }
    }
}
