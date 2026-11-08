namespace NumSharp
{
    /// <summary>
    ///     Which of NumPy's array subclasses a plain alias stands for (<see cref="NDArray.SubclassOrigin"/>): the class of the
    ///     user's operand that the woven subclass path (<see cref="NDSubclass"/>) replaced by a plain view before the
    ///     computation ran.
    /// </summary>
    internal enum NDSubclassKind : byte
    {
        /// <summary>An ordinary array: not an alias of a subclass operand.</summary>
        None = 0,

        /// <summary>The alias stands for a <c>numpy.matrix</c> operand (<see cref="NDMatrix"/>).</summary>
        Matrix = 1,

        /// <summary>The alias stands for a <c>numpy.char.chararray</c> operand (<see cref="NDCharArray"/>).</summary>
        CharArray = 2,
    }

    public partial class NDArray
    {
        /// <summary>
        ///     The subclass this array stands for when it is the PLAIN ALIAS the woven subclass path made of a matrix or
        ///     chararray operand (<see cref="NDSubclass.Plain(NDArray)"/>); <see cref="NDSubclassKind.None"/> for every other
        ///     array.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Computations receive plain views of subclass operands, so their own code and the public functions they call
        ///     compute plain arrays (no subclass rule applies below the user's call). A few NumPy answers still depend on
        ///     the operand's class INSIDE the computation, because NumPy's own Python or C code reacts to the subclass on
        ///     the way: <c>np.block</c> keeps a matrix leaf at two axes, the unique family finds its runs with the
        ///     chararray's stripping comparison. Those computations ask <see cref="NDSubclass.StandsForMatrix(NDArray)"/> /
        ///     <see cref="NDSubclass.StandsForCharArray(NDArray)"/>, which read this mark: the operand's class travels with
        ///     the object, never through per-thread state.
        ///     </para>
        ///     <para>
        ///     The mark belongs to this one alias object: a view, a copy or a result computed from it is a new array without
        ///     it, exactly as NumPy's answers depend on the operand object itself and not on arrays derived from it.
        ///     </para>
        /// </remarks>
        internal NDSubclassKind SubclassOrigin { get; set; }

        /// <summary>
        ///     True while this alias is a chararray INPUT of the ufunc call the user made, for NumPy's trivial-loop copy
        ///     (<see cref="NDCharArray.TrivialLoopCopyError(NDArray, DType, NDArray, DType, NDArray)"/>): the guard
        ///     of a string ufunc copies a short input that needs a cast AS ITS CLASS, and a chararray copy into a dtype that
        ///     is not string data raises the finalize error.
        /// </summary>
        /// <remarks>
        ///     Set only by the ufunc form of the woven subclass path (<see cref="NDSubclass.PlainUfuncInput(NDArray)"/>),
        ///     never by the other rules: a public function that converts its operand with <c>np.asarray</c> must not raise
        ///     NumPy's copy error from a ufunc it calls internally. The first guard that reads the mark clears it, so a ufunc
        ///     the computation calls later on the same alias computes as for a plain array (NumPy copies each input of ONE
        ///     ufunc call).
        /// </remarks>
        internal bool IsTextUfuncInput { get; set; }
    }
}
