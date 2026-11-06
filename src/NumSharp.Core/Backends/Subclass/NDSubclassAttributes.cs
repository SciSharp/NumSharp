using System;

// =============================================================================
// The array-subclass declarations - NumPy's mechanisms, stated on the COMPUTATIONS
// =============================================================================
//
// NumPy decides the class of a function's result by a handful of mechanisms, and each function's implementation picks
// one: the ufunc machinery wraps every ufunc result (npy_find_array_wrap + npy_apply_wrap), an ndarray method creates its
// result as the operand's class (Py_TYPE(self) + __array_finalize__), a C function over a sequence takes the subtype of
// the highest priority (PyArray_GetSubType), fromnumeric hands the call to the operand's own method (_wrapreduction /
// _wrapfunc), and an asarray-based function returns a plain ndarray.
//
// NumSharp states that mechanism ONCE, on the computation a public np.* function calls: a TensorEngine member, a
// sub-engine member (Linalg, Fft), an array-object home (NDArray, UnmanagedStorage, Shape), or whatever other member the
// function computes with. A TensorEngine member SHAPED like a ufunc call (only NDArray inputs, then an optional dtype, out
// and where) needs no declaration at all. NumSharp.Build reads the declaration of the computation each public function
// returns and weaves NumPy's subclass handling into that function (see NDSubclass.cs and tools/NumSharp.Build's
// SubclassWeaver): a subclass operand takes the woven path, which hands the computation plain views and applies the
// declared mechanism to the result.
//
// The subclasses state their own NumPy facts: [ArraySubclass] (the Python name and __array_priority__) and [NumPyMethod]
// on each method NumPy's subclass defines (matrix.sum, chararray.partition), which the generated method dispatch calls.
//
// Every attribute here is INERT at run time: it is a build input of NumSharp.Build's self-weave, and nothing reads it by
// reflection.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's ufunc machinery for this computation: every result goes through <c>npy_find_array_wrap</c> over the
    ///     inputs and <c>npy_apply_wrap</c> (<see cref="NDSubclass.ArrayWrap(NDArray, NDArray, NDArray, NDArray, NDArray)"/>),
    ///     and a chararray input is marked for NumPy's trivial-loop copy.
    /// </summary>
    /// <remarks>
    ///     A member shaped like a ufunc call - only <see cref="NDArray"/> inputs, then an optional <see cref="DType"/>
    ///     <c>dtype</c>, then <see cref="NDArray"/> <c>out</c> and <see cref="NDArray"/> <c>where</c> - is a ufunc without
    ///     this attribute. Declare it on a ufunc whose signature has other parameters (the generalized ufuncs' <c>axes</c>,
    ///     <c>axis</c>, <c>keepdims</c>, <c>casting</c>, <c>order</c>, or a computation that takes a selector first). The
    ///     inputs are the member's <see cref="NDArray"/> parameters other than <c>out</c> and <c>where</c>, in order.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class UfuncAttribute : Attribute
    {
        /// <summary>
        ///     For a computation with several results (an array of them, or a tuple), the zero-based indices of the results
        ///     the wrap applies to, comma separated (<c>"0"</c>, <c>"0,2"</c>); <c>null</c> wraps every result.
        /// </summary>
        public string Outputs { get; set; }

        /// <summary>
        ///     The inputs whose class may wrap the result, comma separated (<c>nameof</c>), when NumPy's function creates the
        ///     ufunc's <c>out=</c> from ONE operand (<c>np.strings.center</c>: <c>out = np.empty_like(a, ...)</c>, so a
        ///     chararray fill beside a plain <c>a</c> gives a plain result); every input is still marked for the
        ///     trivial-loop copy. <c>null</c>: every input may wrap, by priority.
        /// </summary>
        public string Wrap { get; set; }
    }

    /// <summary>
    ///     NumPy's <c>__array_wrap__</c> for this computation's result (a function that ends with a ufunc, with
    ///     <c>_array_converter.wrap</c>, with numpy.linalg's <c>wrap</c> from <c>_makearray</c>, or that indexes its operand
    ///     with an array): the operand chosen by priority wraps the result
    ///     (<see cref="NDSubclass.ArrayWrap(NDArray, NDArray, NDArray, NDArray, NDArray)"/>).
    /// </summary>
    /// <remarks>Unlike <see cref="UfuncAttribute"/>, no input is marked for the trivial-loop copy: NumPy's function is
    ///     not a single ufunc call.</remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class ArrayWrapAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="inputs">The computation's parameters (<c>nameof</c>) whose class may wrap, in NumPy's order: the
        ///     first subclass operand wraps, a later one only with a strictly higher <c>__array_priority__</c>.</param>
        internal ArrayWrapAttribute(params string[] inputs) => Inputs = inputs;

        /// <summary>The wrapping parameters' names.</summary>
        internal string[] Inputs { get; }

        /// <summary>
        ///     For a computation with several results, the zero-based indices of the results NumPy wraps, comma separated
        ///     (<c>np.linalg.eig</c>'s eigenvectors only: <c>"1"</c>; <c>svd</c>'s U and Vh: <c>"0,2"</c>); <c>null</c>
        ///     wraps every result.
        /// </summary>
        public string Outputs { get; set; }
    }

    /// <summary>
    ///     How an ndarray method's 0-d result reaches the caller (<see cref="NewSubtypeAttribute.ZeroDim"/>).
    /// </summary>
    internal enum NewSubtypeZeroDim
    {
        /// <summary>A 0-d result is a 0-d array of the class, like any other rank.</summary>
        Array = 0,

        /// <summary>
        ///     <c>PyArray_Return</c>: a 0-d result is the NumPy scalar of the element, created as the subclass first (so the
        ///     finalize rule runs on its dtype) and then converted - <c>ndarray.take</c>'s unstripped <c>np.str_</c>.
        /// </summary>
        Scalar = 1,

        /// <summary>
        ///     The function ends by INDEXING the operand (<c>np.flip</c>'s <c>m[indexer]</c>): a 0-d result is what the
        ///     subclass's <c>__getitem__</c> returns for an element (a chararray's stripped Python <c>str</c>).
        /// </summary>
        Element = 2,
    }

    /// <summary>
    ///     NumPy's subtype creation for this computation's result: an ndarray method (or a C allocator handed
    ///     <c>Py_TYPE(self)</c>, or <c>np.empty_like</c>'s <c>subok=True</c>) creates the result as the operand's class and
    ///     runs its <c>__array_finalize__</c> (<see cref="NDSubclass.NewSubtype(NDArray, NDArray, NDArray, bool, NewSubtypeZeroDim)"/>).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class NewSubtypeAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="self">The computation's parameter (<c>nameof</c>) whose class the result is created as.</param>
        internal NewSubtypeAttribute(string self) => Self = self;

        /// <summary>The <c>self</c> parameter's name.</summary>
        internal string Self { get; }

        /// <summary>
        ///     True for a function NumPy implements by INDEXING a matrix with a new axis (<c>np.atleast_3d(m)</c> is the 3-d
        ///     matrix object <c>m[:, :, newaxis]</c>): <c>matrix.__getitem__</c> switches the finalize rule off, so the result
        ///     keeps its rank under the class.
        /// </summary>
        public bool KeepRank { get; set; }

        /// <summary>How a 0-d result reaches the caller; <see cref="NewSubtypeZeroDim.Array"/> by default.</summary>
        public NewSubtypeZeroDim ZeroDim { get; set; }

        /// <summary>
        ///     True when NumPy creates the result as the subclass BEFORE it computes anything (<c>np.zeros_like(c, dtype)</c>:
        ///     <c>PyArray_NewLikeArray</c> allocates, the finalize rule refuses a numeric dtype, nothing is filled): the
        ///     subclass's finalize check runs on the result dtype at the function's entry. The result dtype is the function's
        ///     <c>dtype</c> argument, or the operand's own when it is <c>null</c>.
        /// </summary>
        public bool FinalizeFirst { get; set; }
    }

    /// <summary>
    ///     NumPy's <c>PyArray_GetSubType</c> for this computation's result (a C function over a sequence:
    ///     <c>np.concatenate</c>): the subtype of STRICTLY the highest <c>__array_priority__</c> above ndarray's 0.0 -
    ///     a matrix (10.0) wins, a chararray (0.0) never does (<see cref="NDSubclass.GetSubType(NDArray, NDArray, NDArray[])"/>).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class GetSubTypeAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="arrays">The computation's sequence parameter (<c>nameof</c>).</param>
        internal GetSubTypeAttribute(string arrays) => Arrays = arrays;

        /// <summary>The sequence parameter's name.</summary>
        internal string Arrays { get; }
    }

    /// <summary>
    ///     NumPy's <c>asarray</c> conversion for this computation: the result is a plain array, and an operand the
    ///     computation hands back becomes a NEW plain view of it (<see cref="NDSubclass.AsArray(NDArray)"/>), as
    ///     <c>np.asarray(m)</c> is never <c>m</c>.
    /// </summary>
    /// <remarks>A computation without any declaration is also plain, but an operand it hands back stays the caller's object
    ///     (a function NumSharp has not given NumPy's subclass answer, or one whose C# return type cannot carry the class).</remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class AsArrayAttribute : Attribute
    {
    }

    /// <summary>
    ///     The computation is the subclass's OWN construction or conversion code, which reads the operand's class the way
    ///     NumPy's does (<c>matrix.__new__</c>'s <c>isinstance(data, matrix)</c>, <c>np.asanyarray</c>'s pass-through, the
    ///     chararray constructors): the public functions that return it are not woven, so the computation receives the
    ///     caller's objects, never plain views.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    internal sealed class SubclassAwareAttribute : Attribute
    {
    }

    /// <summary>
    ///     A per-class exception to the computation's mechanism: NumPy's function tests <c>isinstance(a, subclass)</c> and
    ///     converts that class with <c>asarray</c> first (<c>np.ravel</c>, <c>np.diagonal</c> and <c>np.trace</c> drop a
    ///     matrix and keep a chararray). The listed subclass gets <see cref="AsArrayAttribute"/>'s answer.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = true, Inherited = false)]
    internal sealed class AsArrayForAttribute : Attribute
    {
        /// <summary>Declares the exception.</summary>
        /// <param name="subclass">The array subclass NumPy converts with <c>asarray</c> (<c>typeof(NDMatrix)</c>).</param>
        internal AsArrayForAttribute(Type subclass) => Subclass = subclass;

        /// <summary>The converted subclass.</summary>
        internal Type Subclass { get; }
    }

    /// <summary>
    ///     NumPy's <c>_wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)</c>: an array subclass with its own
    ///     method of this name gets the call (<c>np.sum(m)</c> is <c>m.sum()</c>). NumSharp.Build GENERATES the dispatch
    ///     from each subclass's <see cref="NumPyMethodAttribute"/> member: the public function's arguments mapped by NAME
    ///     (NumPy passes keywords), <c>axis</c> / <c>out</c> always, <c>dtype</c> when not <c>null</c>, every other one only
    ///     when it differs from its default; one the method lacks is NumPy's
    ///     <c>TypeError: {class}.{method}() got an unexpected keyword argument '{name}'</c>.
    /// </summary>
    /// <remarks>A subclass without a member of that name inherits ndarray's method: the computation runs on the plain view
    ///     and the computation's other declaration (or the plain answer) shapes the result.</remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class WrapReductionAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="method">The NumPy method's name (<c>sum</c>, <c>max</c> for <c>np.amax</c>).</param>
        internal WrapReductionAttribute(string method) => Method = method;

        /// <summary>The NumPy method's name.</summary>
        internal string Method { get; }
    }

    /// <summary>
    ///     NumPy's <c>_wrapfunc(obj, method, *args, **kwds)</c>: the operand's bound method gets the call; when it raises
    ///     <c>TypeError</c> (a keyword its signature lacks), <c>_wrapit</c> computes on the plain array and wraps the result
    ///     with the operand's <c>__array_wrap__</c>. The dispatch is generated as for
    ///     <see cref="WrapReductionAttribute"/>; a subclass without a member of that name inherits ndarray's method (the
    ///     computation's other declaration shapes the result).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class WrapFuncAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="method">The NumPy method's name.</param>
        internal WrapFuncAttribute(string method) => Method = method;

        /// <summary>The NumPy method's name.</summary>
        internal string Method { get; }

        /// <summary>
        ///     The public function's parameters NumPy passes to the method only when the caller gave them (its <c>**kwds</c>,
        ///     built from keyword-only arguments that default to <c>np._NoValue</c>: <c>np.argmax</c>'s <c>keepdims</c>),
        ///     comma separated. Every other parameter is always passed.
        /// </summary>
        public string WhenGiven { get; set; }
    }

    /// <summary>
    ///     An argument NumPy's function COMPUTES before it hands the call to the subclass's method (the generated dispatch of
    ///     <see cref="WrapReductionAttribute"/> / <see cref="WrapFuncAttribute"/>): <c>np.std</c> folds <c>correction=</c>
    ///     into <c>ddof</c> first, so <c>matrix.std</c> receives <c>ddof</c> and never sees <c>correction</c>.
    /// </summary>
    /// <remarks>The method's parameter <see cref="Parameter"/> takes <c>Type.Method(...)</c>, its parameters bound by name to
    ///     the public function's; the function parameters that method reads are consumed (no unexpected-keyword error for
    ///     them).</remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = true, Inherited = false)]
    internal sealed class WrapArgumentAttribute : Attribute
    {
        /// <summary>Declares the computed argument.</summary>
        /// <param name="parameter">The subclass method's parameter it fills (<c>ddof</c>).</param>
        /// <param name="type">The class that holds the computing method.</param>
        /// <param name="method">The static method that computes it (<c>nameof</c>).</param>
        internal WrapArgumentAttribute(string parameter, Type type, string method)
        {
            Parameter = parameter;
            Type = type;
            Method = method;
        }

        /// <summary>The subclass method's parameter.</summary>
        internal string Parameter { get; }

        /// <summary>The class that holds the computing method.</summary>
        internal Type Type { get; }

        /// <summary>The computing method's name.</summary>
        internal string Method { get; }
    }

    /// <summary>
    ///     NumPy's function ends by handing its (plain-computed) answer through <c>np.{method}</c> of an intermediate of the
    ///     operand's class - <c>np.nansum(m)</c> is <c>np.sum</c> of the nan-replaced matrix, <c>np.median(m)</c> ends in
    ///     <c>np.mean</c> of a partitioned matrix: that method's keyword errors apply at the entry and its <c>_collapse</c>
    ///     shapes the result (<see cref="NDSubclass.Collapse(NDArray, NDArray, int?)"/>).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class ReducesThroughAttribute : Attribute
    {
        /// <summary>Declares the mechanism.</summary>
        /// <param name="method">The NumPy method the intermediate reaches, for the keyword errors (<c>sum</c>); <c>null</c>
        ///     when no keyword reaches it (<c>np.median</c>: <c>_ureduce</c> handles <c>keepdims</c> itself).</param>
        internal ReducesThroughAttribute(string method) => Method = method;

        /// <summary>The NumPy method, or <c>null</c>.</summary>
        internal string Method { get; }

        /// <summary>
        ///     The method NumPy reaches FIRST when the operand's dtype is inexact (float or complex), or <c>null</c> when it is
        ///     always <see cref="Method"/>: the nan-functions' <c>_replace_nan</c> returns a mask for an inexact dtype, and the
        ///     masked path calls <c>np.sum(~mask, keepdims=...)</c> before the mean (<c>np.nanmean</c>'s keyword error names
        ///     <c>sum</c> for a float matrix and <c>mean</c> for an integer one).
        /// </summary>
        public string InexactMethod { get; set; }
    }

    /// <summary>
    ///     A NumPy answer that no shared mechanism expresses (a function whose result class depends on the route it took, or
    ///     a composite whose Python body reacts to the subclass in its own way): the rule is a static method of
    ///     <see cref="Type"/>, applied to the computed result.
    /// </summary>
    /// <remarks>
    ///     The rule method's first parameter receives the public function's result (computed on plain views); every other
    ///     parameter is bound by NAME to the public function's parameter of that name, which holds the CALLER's object (the
    ///     matrix, the chararray). It returns the function's answer. An optional <see cref="Check"/> method is called at the
    ///     function's entry with the arguments bound the same way, before anything is computed (NumPy's errors that come
    ///     before the computation).
    ///     <para>A rule may sit on the CONSTRUCTOR of the result struct a function returns (<c>np.unique</c>'s
    ///     <c>UniqueResult</c>, <c>np.meshgrid</c>'s <c>MeshgridResult</c>): the struct is what the function computes, and
    ///     the rule takes and returns it.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    internal sealed class SubclassRuleAttribute : Attribute
    {
        /// <summary>Declares the rule.</summary>
        /// <param name="type">The class that holds the rule method.</param>
        /// <param name="rule">The rule method's name (<c>nameof</c>), or <c>null</c> for an entry check only.</param>
        internal SubclassRuleAttribute(Type type, string rule)
        {
            Type = type;
            Rule = rule;
        }

        /// <summary>The class that holds the rule.</summary>
        internal Type Type { get; }

        /// <summary>The rule method's name, or <c>null</c>.</summary>
        internal string Rule { get; }

        /// <summary>The entry check's name in <see cref="Type"/>, or <c>null</c>.</summary>
        public string Check { get; set; }
    }

    /// <summary>
    ///     Declares a class as one of NumPy's array subclasses: its Python class name (for NumPy's error texts) and its
    ///     <c>__array_priority__</c> (the order the wrap and the subtype selection follow).
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    internal sealed class ArraySubclassAttribute : Attribute
    {
        /// <summary>Declares the class.</summary>
        /// <param name="name">NumPy's class name (<c>matrix</c>, <c>chararray</c>).</param>
        internal ArraySubclassAttribute(string name) => Name = name;

        /// <summary>NumPy's class name.</summary>
        internal string Name { get; }

        /// <summary>The class's <c>__array_priority__</c> (0.0, ndarray's, when NumPy's class declares none).</summary>
        public double Priority { get; set; }
    }

    /// <summary>
    ///     Marks an array subclass's member as its NumPy METHOD of the same name (<c>matrix.sum</c>): the target of the
    ///     generated <c>_wrapreduction</c> / <c>_wrapfunc</c> dispatch. Its parameter names are NumPy's: the dispatch maps
    ///     the public function's arguments to them by name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class NumPyMethodAttribute : Attribute
    {
    }

    /// <summary>
    ///     NumSharp's static-typing rule for an <see cref="NDArray"/> member: called on a subclass object through an
    ///     <see cref="NDArray"/> reference, the member computes exactly as for the object's plain view (<c>a.sum(0)</c> of a
    ///     matrix has N elements). NumSharp.Build weaves the receiver's plain view into the member: the member body, the
    ///     engine and every public function it calls never meet the subclass object, and a receiver the member hands back
    ///     is the caller's object again.
    /// </summary>
    /// <remarks>Declare it on a member that computes with the engine or with public functions; a member that only reads
    ///     the receiver's fields needs nothing.</remarks>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    internal sealed class PlainReceiverAttribute : Attribute
    {
    }
}
