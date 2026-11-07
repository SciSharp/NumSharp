// =============================================================================
// NDArray.SubclassHooks.cs - NumPy's array-subclass class members, as C# inheritance
// =============================================================================
//
// NumPy puts its subclass behaviour ON the classes: a class attribute (__array_priority__) and methods that ndarray
// defines and a subclass overrides (__array_wrap__, __array_finalize__), plus matrix's own _collapse. NumSharp states
// them the same way, as virtual members of NDArray under NumPy's names, which NDMatrix and NDCharArray override; the
// woven subclass paths (NDSubclass) choose the operand NumPy's machinery chooses and call its member, so no mechanism
// switches on the class.
//
// C# cannot change an object's class or shape after construction, so the two NumPy steps of creating a result as a
// subclass - allocated as type(self), then its __array_finalize__(obj) run on it - are one call here: the plain result
// goes in, the result as the class comes out (__array_finalize__(created, ...)), the class's finalize rule applied.
//
// Only __array_priority__ is public (NumPy's ndarray exposes it, and numpy.matrix's was already public here); the others
// are the subclass machinery's internals, called by the woven code.
// =============================================================================

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.__array_priority__</c>: 0.0 for an ndarray (and a chararray, which declares none). The class
        ///     wrap of a ufunc result (<c>npy_find_array_wrap</c>) and the subtype of a sequence function
        ///     (<c>PyArray_GetSubType</c>) go to the subclass operand of the highest priority, so a matrix (10.0) wins over
        ///     both.
        /// </summary>
        /// <remarks>Read only for a subclass operand: a plain array never takes part in the choice.</remarks>
        public virtual double __array_priority__ => 0.0;

        /// <summary>
        ///     NumPy's <c>ndarray.__array_wrap__(array)</c>: <paramref name="array"/> as this array's class (an ndarray's
        ///     wrap of a plain array is the array itself). The woven subclass path calls it on the operand
        ///     <c>npy_find_array_wrap</c> chose, with the result computed on the plain views.
        /// </summary>
        /// <param name="array">The result to wrap (a plain array, or the caller's operand handed back).</param>
        /// <returns><paramref name="array"/> itself for an ndarray; the class's view of it in an override.</returns>
        /// <exception cref="ValueError">An override's finalize rule refuses the result (a matrix of more than two axes
        ///     longer than 1).</exception>
        internal virtual NDArray __array_wrap__(NDArray array) => array;

        /// <summary>
        ///     NumPy's subtype creation: the result allocated as this array's class (<c>Py_TYPE(self)</c>), then the class's
        ///     <c>__array_finalize__</c> run on it. C# cannot retype an object, so the plain result goes in and the result as
        ///     the class comes out; an ndarray's is the result itself.
        /// </summary>
        /// <param name="created">The result NumPy creates as this class, computed plain.</param>
        /// <param name="keepRank">NumPy indexed a matrix with a new axis (<c>matrix.__getitem__</c> switches its finalize
        ///     rule off), so the result keeps its rank.</param>
        /// <param name="zeroDim">How a 0-d result reaches the caller (a chararray's element stand-in).</param>
        /// <returns>The result as this class; <paramref name="created"/> itself for an ndarray.</returns>
        /// <exception cref="ValueError">An override's finalize rule refuses the result (a matrix of more than two axes
        ///     longer than 1; a chararray of a dtype that is not string data).</exception>
        internal virtual NDArray __array_finalize__(NDArray created, bool keepRank, NewSubtypeZeroDim zeroDim) => created;

        /// <summary>
        ///     The dtype check of this class's <c>__array_finalize__</c>, run where NumPy creates a result of the class
        ///     BEFORE computing it (<c>np.zeros_like(c, dtype)</c>) or for a result that is a C# number (<c>np.argmax(c)</c>'s
        ///     index, created as the class before <c>PyArray_Return</c>). An ndarray checks nothing.
        /// </summary>
        /// <param name="dtype">The dtype of the result about to be created.</param>
        /// <exception cref="ValueError">An override refuses the dtype (a chararray of a dtype that is not string data:
        ///     <c>Can only create a chararray from string data.</c>).</exception>
        internal virtual void __array_finalize__(DType dtype)
        {
        }

        /// <summary>
        ///     numpy.matrix's <c>_collapse(axis)</c> after a <c>keepdims=True</c> reduction, applied to the plain reduction
        ///     NumPy's function computed (<c>np.nansum(m)</c> is <c>np.sum</c> of the nan-replaced matrix). An ndarray has no
        ///     <c>_collapse</c>.
        /// </summary>
        /// <param name="result">The plain reduction (keepdims=False), unaliased.</param>
        /// <param name="axis">The caller's axis: it decides row or column for a 1-d plain result.</param>
        /// <returns><c>null</c>: this class has no <c>_collapse</c> (the reduction keeps NumPy's other answer).</returns>
        internal virtual NDArray _collapse(NDArray result, int? axis) => null;
    }
}
