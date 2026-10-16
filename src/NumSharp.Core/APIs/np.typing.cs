using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     <c>numpy.typing</c> — NumPy's static-typing aliases (<c>NDArray</c>, <c>ArrayLike</c>, <c>DTypeLike</c>,
        ///     <c>NBitBase</c>), each as the C# type that plays its role in NumSharp's own signatures.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>Why the members are <see cref="Type"/> objects.</b> NumPy's names are type ALIASES for annotations
        ///     (<c>def f(x: npt.ArrayLike) -&gt; npt.NDArray[np.float64]</c>). C# has no exported type aliases — a property
        ///     cannot stand in a signature — so the type checker's role is the C# compiler's, and each alias already has a
        ///     C# type that NumSharp uses in its place everywhere. The members name those types, the way
        ///     <see cref="dtypes"/> names the dtype classes, so a program can look them up and a port knows what to write:
        ///     </para>
        ///     <list type="table">
        ///       <listheader><term>NumPy annotation</term><description>C# type</description></listheader>
        ///       <item><term><c>npt.NDArray[np.float64]</c></term><description><c>NDArray&lt;double&gt;</c>
        ///         (<see cref="NDArray"/> is its generic definition, so <c>np.typing.NDArray.MakeGenericType(typeof(double))</c>
        ///         is <c>typeof(NDArray&lt;double&gt;)</c>); with an abstract or unknown scalar type
        ///         (<c>npt.NDArray[np.floating]</c>, <c>npt.NDArray[Any]</c>) the untyped <c>NDArray</c>.</description></item>
        ///       <item><term><c>npt.ArrayLike</c></term><description><c>NDArray</c> (<see cref="ArrayLike"/>): every NumSharp
        ///         function takes its array_like arguments as <c>NDArray</c>, whose implicit conversions accept a scalar of
        ///         any dtype, a C# array of any rank (jagged too) and a string; <c>np.asanyarray(object)</c> converts the rest
        ///         of NumPy's array_like set (lists of objects, tuples, enumerables, <c>Memory&lt;T&gt;</c>).</description></item>
        ///       <item><term><c>npt.DTypeLike</c></term><description><c>DType</c> (<see cref="DTypeLike"/>): every
        ///         <c>dtype=</c> parameter, which accepts a <c>DType</c>, a C# <see cref="Type"/>, an <c>NPTypeCode</c> and a
        ///         dtype string implicitly; a scalar class (<c>NDScalarType</c>) converts explicitly.</description></item>
        ///       <item><term><c>npt.NBitBase</c></term><description><see cref="NBitBase"/>: deprecated in NumPy 2.3, and
        ///         without a use in C#, which has no precision-generic scalar classes.</description></item>
        ///     </list>
        /// </remarks>
        /// <example>
        /// <code>
        /// // def mean_of(x: npt.ArrayLike, dtype: npt.DTypeLike = None) -> npt.NDArray[np.float64]: ...
        /// static NDArray&lt;double&gt; mean_of(NDArray x, DType dtype = null) =&gt; ...;
        ///
        /// Type t = np.typing.NDArray.MakeGenericType(typeof(double));   // typeof(NDArray&lt;double&gt;)
        /// </code>
        /// </example>
        [ModuleName("np.typing")]
        public static class typing
        {
            /// <summary>
            ///     <c>numpy.typing.NDArray</c> (<c>ndarray[tuple[Any, ...], dtype[ScalarT]]</c>): the generic definition
            ///     <c>NumSharp.Generic.NDArray&lt;T&gt;</c>, NumSharp's array typed by its element type (it derives from the
            ///     untyped <c>NumSharp.NDArray</c>). <c>npt.NDArray[np.float64]</c> is <c>NDArray&lt;double&gt;</c>, i.e. this
            ///     type closed over the C# element type (<c>NDArray.MakeGenericType(typeof(double))</c>).
            /// </summary>
            /// <remarks>
            ///     Inside this class the name <c>NDArray</c> means this property, so the code spells NumSharp's array types
            ///     with <c>global::</c>.
            /// </remarks>
            public static Type NDArray => typeof(global::NumSharp.Generic.NDArray<>);

            /// <summary>
            ///     <c>numpy.typing.ArrayLike</c> (objects that can be converted to arrays): <c>NDArray</c>, the type NumSharp
            ///     takes every array_like argument as. Its implicit conversions accept a scalar of any of the 15 dtypes, a C#
            ///     array of any rank or a jagged one, and a string; for the rest of NumPy's set (a list of objects, a tuple,
            ///     an enumerable) <c>np.asanyarray(object)</c> builds the array.
            /// </summary>
            public static Type ArrayLike => typeof(global::NumSharp.NDArray);

            /// <summary>
            ///     <c>numpy.typing.DTypeLike</c> (objects that can be converted to dtypes): <c>DType</c>, the type of every
            ///     <c>dtype=</c> parameter. A <c>DType</c>, a C# <see cref="Type"/> (<c>typeof(float)</c>), an
            ///     <c>NPTypeCode</c> and a dtype string (<c>"f4"</c>, <c>"float32"</c>) all convert to it implicitly; a scalar
            ///     class converts explicitly (<c>(DType)t</c>), because an abstract one has no dtype.
            /// </summary>
            public static Type DTypeLike => typeof(global::NumSharp.DType);

            /// <summary>
            ///     <c>numpy.typing.NBitBase</c>: the base of the precision markers NumPy's stubs parametrize its scalar
            ///     classes with (<c>np.floating[npt.NBitBase]</c>), used only by static type checkers. Deprecated in NumPy 2.3
            ///     (accessing it warns, as using this class does) and without a use in C#, which has no precision-generic
            ///     scalar classes; it exists so a ported reference compiles and says why it should go.
            /// </summary>
            /// <remarks>
            ///     NumPy forbids subclassing it except for its own private markers (<c>_8Bit</c> … <c>_128Bit</c>); the
            ///     <see langword="private protected"/> constructor forbids it outside NumSharp.
            /// </remarks>
            [Obsolete("`NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` "
                      + "or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)")]
            public abstract class NBitBase
            {
                /// <summary>Only NumSharp could derive a marker (NumPy's <c>__init_subclass__</c> allows only its own names).</summary>
                private protected NBitBase()
                {
                }
            }
        }
    }
}
