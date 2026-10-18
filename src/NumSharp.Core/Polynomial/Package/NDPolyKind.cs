using System;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyKind.cs — what distinguishes the six numpy.polynomial series classes from one another
// =============================================================================
//
// NumPy's ABCPolyBase holds every behavior; a concrete class (Polynomial, Chebyshev, Legendre, Laguerre, Hermite,
// HermiteE) only binds twelve module functions (_add = chebadd, _val = chebval, ...), its class attributes domain /
// window / basis_name, and — for Polynomial — its own term printers. In NumSharp every module function of a basis is one
// engine entry point keyed by PolyBasis, so a class is fully described by this record: the basis, the names NumPy prints
// (repr, errors), the class defaults, and a factory for a blank instance the base class fills in (NumPy's
// `self.__class__(...)`). The base class reads everything it needs from here; a class-object argument (convert's kind,
// cast's class) is resolved through Of(Type).
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The description of one numpy.polynomial series class: its basis, its NumPy names, its class defaults and a factory
    ///     for its instances. One shared instance per class (<see cref="NDPolynomial"/> and its five siblings each hold one).
    /// </summary>
    /// <remarks>
    ///     NumPy's classes differ only in the module functions they bind and their class attributes; NumSharp's module
    ///     functions take the basis as a parameter, so the base class (<see cref="NDPolyBase"/>) implements every member once
    ///     and asks this record for the parts that differ. A class object that is not one of the six (an abstract base, any
    ///     other type) has no record: <see cref="Of"/> refuses it as Python's attribute lookup on it would.
    /// </remarks>
    internal sealed class NDPolyKind
    {
        /// <summary>The basis whose module functions the class binds (NumPy's <c>_add</c>, <c>_val</c>, ...).</summary>
        public readonly PolyBasis Basis;

        /// <summary>NumPy's class name (<c>Polynomial</c>, <c>Chebyshev</c>, ...): what <c>repr</c> prints and what Python's
        ///     operator errors name (CPython's <c>tp_name</c> of a Python class is its bare name).</summary>
        public readonly string Name;

        /// <summary>The module NumPy defines the class in (<c>numpy.polynomial.polynomial</c>, ...): with
        ///     <see cref="Name"/> it is <c>str(type(p))</c>, <c>&lt;class 'numpy.polynomial.polynomial.Polynomial'&gt;</c>.</summary>
        public readonly string Module;

        /// <summary>The class attribute <c>basis_name</c>: null for <c>Polynomial</c> (NumPy's <c>None</c>), else <c>T</c>,
        ///     <c>P</c>, <c>L</c>, <c>H</c> or <c>He</c> — the letter the default term printers write.</summary>
        public readonly string BasisName;

        /// <summary>The concrete C# type the record describes.</summary>
        public readonly Type ClrType;

        /// <summary>Reads the class default <c>domain</c> (the class attribute: the same object on every read until the
        ///     class's <c>default_domain</c> is reassigned).</summary>
        public readonly Func<NDArray> DefaultDomain;

        /// <summary>Reads the class default <c>window</c> (the class attribute, like <see cref="DefaultDomain"/>).</summary>
        public readonly Func<NDArray> DefaultWindow;

        /// <summary>Reads the class attribute <c>maxpower</c>: a boxed Python int (100, NumPy's <c>ABCPolyBase.maxpower</c>)
        ///     unless the class's <c>default_maxpower</c> was reassigned; null is NumPy's <c>None</c>, no limit.</summary>
        public readonly Func<object> DefaultMaxPower;

        /// <summary>Creates an instance with no state yet: <see cref="NDPolyBase"/> fills it in, exactly once, through its
        ///     constructor logic (NumPy's <c>self.__class__(coef, domain, window, symbol)</c>).</summary>
        public readonly Func<NDPolyBase> Blank;

        /// <summary>Creates the record of one class.</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="name">NumPy's class name.</param>
        /// <param name="module">NumPy's defining module.</param>
        /// <param name="basisName">The class attribute <c>basis_name</c> (null for the power series).</param>
        /// <param name="clrType">The C# type.</param>
        /// <param name="defaultDomain">Reads the class default domain.</param>
        /// <param name="defaultWindow">Reads the class default window.</param>
        /// <param name="defaultMaxPower">Reads the class default maxpower.</param>
        /// <param name="blank">Creates a blank instance.</param>
        public NDPolyKind(PolyBasis basis, string name, string module, string basisName, Type clrType, Func<NDArray> defaultDomain,
            Func<NDArray> defaultWindow, Func<object> defaultMaxPower, Func<NDPolyBase> blank)
        {
            Basis = basis;
            Name = name;
            Module = module;
            BasisName = basisName;
            ClrType = clrType;
            DefaultDomain = defaultDomain;
            DefaultWindow = defaultWindow;
            DefaultMaxPower = defaultMaxPower;
            Blank = blank;
        }

        /// <summary><c>str(cls)</c>: <c>&lt;class 'numpy.polynomial.chebyshev.Chebyshev'&gt;</c> — what NumPy's
        ///     <c>__truediv__</c> and <c>__format__</c> errors print for the class.</summary>
        public string ClassRepr => "<class '" + Module + "." + Name + "'>";

        /// <summary>
        ///     The record of a class given as a C# type (NumPy's class object, e.g. <c>convert(kind=Chebyshev)</c> →
        ///     <c>kind: typeof(NDChebyshev)</c>).
        /// </summary>
        /// <param name="type">The type: <see cref="NDPolynomial"/>, <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>,
        ///     <see cref="NDLaguerre"/>, <see cref="NDHermite"/> or <see cref="NDHermiteE"/>.</param>
        /// <returns>The record.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="type"/> is null.</exception>
        /// <exception cref="AttributeError">Any other type: Python reads the class attributes NumPy's code asks for
        ///     (<c>kind.domain</c>, <c>kind.identity</c>), and a class without them raises
        ///     <c>type object '{name}' has no attribute 'domain'</c> — NumSharp's message names the C# type.</exception>
        public static NDPolyKind Of(Type type)
        {
            if (type is null) throw new ArgumentNullException(nameof(type));
            if (type == typeof(NDPolynomial)) return NDPolynomial.Kind;
            if (type == typeof(NDChebyshev)) return NDChebyshev.Kind;
            if (type == typeof(NDLegendre)) return NDLegendre.Kind;
            if (type == typeof(NDLaguerre)) return NDLaguerre.Kind;
            if (type == typeof(NDHermite)) return NDHermite.Kind;
            if (type == typeof(NDHermiteE)) return NDHermiteE.Kind;
            throw new AttributeError($"type object '{type.Name}' has no attribute 'domain'");
        }
    }
}
