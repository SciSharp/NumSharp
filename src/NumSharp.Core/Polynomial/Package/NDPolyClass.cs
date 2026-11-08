using System;
using System.Threading;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolyClass.cs — NumPy's class object of a numpy.polynomial series class
// =============================================================================
//
// In Python a class is an object with attributes of its own. NumPy's series classes keep four there: Chebyshev.domain,
// Chebyshev.window, Chebyshev.maxpower and Chebyshev.basis_name, which a series reads when it has none of its own (a
// series created without a domain answers p.domain with Chebyshev.domain itself). C# gives a name in a type to ONE
// member, static or instance: a static NDChebyshev.domain would hide the instance p.domain, and p.domain is the one
// NumPy code reads all the time. So the class attributes are members of this class OBJECT, under NumPy's names, and
// the object is reachable from the class (NDChebyshev.Class, NumPy's Chebyshev) and from any of its series
// (p.__class__, Python's p.__class__). One object per class: NumPy's `p.__class__ is Chebyshev` holds as
// ReferenceEquals(p.__class__, NDChebyshev.Class).
//
// The object also carries what NDPolyBase needs to implement every member once for the six classes: the basis whose
// module functions the class binds (NumPy's _add = chebadd, _val = chebval, ...), the names NumPy prints (repr,
// errors) and a factory for a blank instance (NumPy's self.__class__(...)). Those stay internal; the class methods
// (fit, fromroots, identity, basis, cast, interpolate) stay the static members of each C# class.
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's class object of one numpy.polynomial series class (<c>Chebyshev</c>, <c>Polynomial</c>, ...): the class
    ///     attributes <see cref="domain"/>, <see cref="window"/>, <see cref="maxpower"/> and <see cref="basis_name"/> under
    ///     NumPy's names, and the class's <see cref="__name__"/> / <see cref="__module__"/>. Reached as
    ///     <c>NDChebyshev.Class</c> (NumPy's <c>Chebyshev</c>) and as <see cref="NDPolyBase.__class__"/> of a series (Python's
    ///     <c>p.__class__</c>): NumPy's <c>Chebyshev.domain</c> is <c>NDChebyshev.Class.domain</c>.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why an object and not statics.</b> C# cannot declare a static <c>NDChebyshev.domain</c> beside the
    ///     instance <c>p.domain</c>: the static would hide the instance member, and <c>p.domain</c> would stop compiling.
    ///     The instance attributes keep NumPy's names on the series; the class attributes keep them here.</para>
    ///     <para><b>Shared, process-wide state.</b> There is one object per class. Assigning <see cref="domain"/>,
    ///     <see cref="window"/> or <see cref="maxpower"/> changes every series of the class that has none of its own,
    ///     on every thread, from then on (NumPy's <c>Chebyshev.domain = ...</c> is a global change too). The class
    ///     methods, a <c>convert</c> to the class and every constructor without a domain read the attribute at the moment
    ///     they run.</para>
    ///     <para><b>The class itself.</b> Wherever NumSharp takes a class as a <see cref="Type"/>
    ///     (<c>convert(kind: ...)</c>), the object converts implicitly: <c>p.convert(kind: q.__class__)</c> is NumPy's
    ///     <c>p.convert(kind=q.__class__)</c>. The class methods are the static members of the C# class
    ///     (<c>NDChebyshev.fit(...)</c>), not members of this object.</para>
    /// </remarks>
    // [NDBorrowed]: the two attribute arrays are process-lifetime shared values (each holds one extra buffer reference
    // that is never released, as the module constants do), so nothing ever disposes them and this object is not their
    // disposing owner; without the mark NDW016 would ask for an IDisposable no caller could ever call, and NDW017 would
    // ask every series (which references this object) to dispose its class.
    [NDBorrowed]
    public sealed class NDPolyClass
    {
        /// <summary>The basis whose module functions the class binds (NumPy's <c>_add</c>, <c>_val</c>, ...).</summary>
        internal readonly PolyBasis Basis;

        /// <summary>The concrete C# type the object describes (<see cref="NDChebyshev"/>, ...).</summary>
        internal readonly Type ClrType;

        /// <summary>Creates an instance with no state yet: <see cref="NDPolyBase"/> fills it in, exactly once, through its
        ///     constructor logic (NumPy's <c>self.__class__(coef, domain, window, symbol)</c>).</summary>
        internal readonly Func<NDPolyBase> Blank;

        /// <summary>NumPy's class name (<c>Polynomial</c>, <c>Chebyshev</c>, ...).</summary>
        private readonly string _name;

        /// <summary>The module NumPy defines the class in (<c>numpy.polynomial.chebyshev</c>, ...).</summary>
        private readonly string _module;

        /// <summary>The class attribute <c>basis_name</c> (null for the power series).</summary>
        private readonly string _basisName;

        /// <summary>The start of the interval the default domain and window hold (-1, or 0 for Laguerre).</summary>
        private readonly double _lo;

        /// <summary>The end of that interval (1).</summary>
        private readonly double _hi;

        /// <summary>The class attribute <c>domain</c>, created on first use (null until then).</summary>
        private NDArray _domain;

        /// <summary>The class attribute <c>window</c>, created on first use (null until then).</summary>
        private NDArray _window;

        /// <summary>The class attribute <c>maxpower</c>: NumPy's <c>ABCPolyBase.maxpower = 100</c>, a Python int (a boxed
        ///     <see cref="long"/>); null is None.</summary>
        private object _maxpower = 100L;

        /// <summary>Creates the class object of one series class (one per class, held by its static <c>Class</c>).</summary>
        /// <param name="basis">The basis.</param>
        /// <param name="name">NumPy's class name.</param>
        /// <param name="module">NumPy's defining module.</param>
        /// <param name="basisName">The class attribute <c>basis_name</c> (null for the power series).</param>
        /// <param name="clrType">The C# type.</param>
        /// <param name="lo">The start of the default domain and window.</param>
        /// <param name="hi">The end of the default domain and window.</param>
        /// <param name="blank">Creates a blank instance.</param>
        internal NDPolyClass(PolyBasis basis, string name, string module, string basisName, Type clrType, double lo, double hi,
            Func<NDPolyBase> blank)
        {
            Basis = basis;
            _name = name;
            _module = module;
            _basisName = basisName;
            ClrType = clrType;
            _lo = lo;
            _hi = hi;
            Blank = blank;
        }

        /// <summary>NumPy's <c>Chebyshev.__name__</c>: the class name NumPy prints in <c>repr(p)</c> and in Python's operator
        ///     errors (<c>'Chebyshev'</c>, never the C# name <c>NDChebyshev</c>).</summary>
        public string __name__ => _name;

        /// <summary>NumPy's <c>Chebyshev.__qualname__</c>: the class name (the classes are not nested, so it equals
        ///     <see cref="__name__"/>).</summary>
        public string __qualname__ => _name;

        /// <summary>NumPy's <c>Chebyshev.__module__</c>: the module that defines the class
        ///     (<c>'numpy.polynomial.chebyshev'</c>; the power series' is <c>'numpy.polynomial.polynomial'</c>).</summary>
        public string __module__ => _module;

        /// <summary>
        ///     NumPy's class attribute <c>Chebyshev.domain</c>: <c>[-1, 1]</c> (<c>[0, 1]</c> for Laguerre), a float64 array
        ///     created on first read. Every series of the class created without a domain reads THIS object
        ///     (<c>p.domain</c> returns it itself, NumPy's <c>p.domain is Chebyshev.domain</c>), so writing into its elements
        ///     changes all of them. Assigning stores a view of the given array, which those series read from then on.
        /// </summary>
        /// <remarks>
        ///     The array lives as long as the process and holds one extra buffer reference, so a caller's <c>Dispose()</c> of
        ///     it (<c>using var d = p.domain;</c>) cannot free memory every later series reads. An assignment keeps the
        ///     previous array alive for the same reason (series or callers may still hold it); assigning the current array
        ///     back changes nothing, so the attribute keeps its identity, as in NumPy.
        /// </remarks>
        /// <exception cref="ArgumentNullException">Assigning null (NumPy would store None, and every series reading the
        ///     attribute would then fail).</exception>
        public NDArray domain
        {
            get => Attribute(ref _domain);
            set => Assign(ref _domain, value);
        }

        /// <summary>NumPy's class attribute <c>Chebyshev.window</c>: <c>[-1, 1]</c> (<c>[0, 1]</c> for Laguerre), shared and
        ///     assigned like <see cref="domain"/>.</summary>
        /// <exception cref="ArgumentNullException">Assigning null.</exception>
        public NDArray window
        {
            get => Attribute(ref _window);
            set => Assign(ref _window, value);
        }

        /// <summary>
        ///     NumPy's class attribute <c>Chebyshev.maxpower</c>: the largest power <c>__pow__</c> allows on a series of the
        ///     class that has no <see cref="NDPolyBase.maxpower"/> of its own — a boxed Python int, 100 until assigned; null
        ///     is None (no limit). Any value Python can compare with an int works (NumPy's <c>power &gt; maxpower</c> decides).
        /// </summary>
        public object maxpower
        {
            get => Volatile.Read(ref _maxpower);
            set => Volatile.Write(ref _maxpower, value);
        }

        /// <summary>
        ///     NumPy's class attribute <c>Chebyshev.basis_name</c>: the letter the default term printers write (<c>T</c>,
        ///     <c>P</c>, <c>L</c>, <c>H</c>, <c>He</c>); null for the power series, which prints powers of the symbol instead.
        ///     Read-only: NumPy's attribute is writable so that a subclass can name its basis, and NumSharp's classes are
        ///     sealed.
        /// </summary>
        public string basis_name => _basisName;

        /// <summary>Python's <c>str(Chebyshev)</c> / <c>repr(Chebyshev)</c>:
        ///     <c>&lt;class 'numpy.polynomial.chebyshev.Chebyshev'&gt;</c> — also what NumPy's <c>__truediv__</c> and
        ///     <c>__format__</c> errors print for the class.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => "<class '" + _module + "." + _name + "'>";

        /// <summary>
        ///     The class as the C# <see cref="Type"/> NumSharp takes where NumPy takes a class: <c>p.convert(kind:
        ///     NDLegendre.Class)</c> and <c>p.convert(kind: q.__class__)</c> are NumPy's <c>p.convert(kind=Legendre)</c> and
        ///     <c>p.convert(kind=q.__class__)</c>.
        /// </summary>
        /// <param name="cls">The class object, or null.</param>
        /// <returns>The C# type (<c>typeof(NDLegendre)</c>, ...), or null for null (so a null class keeps meaning "this
        ///     series' class" in <c>convert</c>).</returns>
        public static implicit operator Type(NDPolyClass cls) => cls?.ClrType;

        /// <summary>
        ///     Reads a class attribute <c>domain</c> / <c>window</c> (NumPy's <c>np.array(chebdomain)</c>, the class's own
        ///     float64 copy of its module's constant), creating it on first use. Like the module constants
        ///     (<see cref="NDPolySeries.Constant"/>) it is detached from every scope and holds one extra buffer reference.
        /// </summary>
        /// <param name="slot">The attribute's field.</param>
        /// <returns>The class attribute.</returns>
        private NDArray Attribute(ref NDArray slot) => Volatile.Read(ref slot) ?? CreateAttribute(ref slot);

        /// <summary>
        ///     The first-use path of <see cref="Attribute"/>: builds and publishes the attribute (a lost first-use race drops
        ///     its copy). A method of its own because its lambda captures <c>this</c>: C# builds a captured closure at the entry
        ///     of the method that declares it, which in <see cref="Attribute"/> would allocate on every read.
        /// </summary>
        /// <param name="slot">The attribute's field.</param>
        /// <returns>The published attribute (this call's, or the race winner's).</returns>
        private NDArray CreateAttribute(ref NDArray slot) => NDPolySeries.Constant(ref slot, () => np.array(new[] { _lo, _hi }));

        /// <summary>
        ///     Assigns a class attribute <c>domain</c> / <c>window</c> (NumPy's <c>Chebyshev.domain = np.array([0, 2])</c>): a
        ///     view of the given array that no scope tracks, with one extra buffer reference that is never released (series
        ///     created before or after may read the attribute for the process's life, and the array it views may be disposed
        ///     by its owner). Assigning the current attribute itself changes nothing.
        /// </summary>
        /// <param name="slot">The attribute's field.</param>
        /// <param name="value">The new attribute.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        private static void Assign(ref NDArray slot, NDArray value)
        {
            if (value is null) throw new ArgumentNullException(nameof(value));
            if (ReferenceEquals(value, Volatile.Read(ref slot)))
                return;
            var shared = value.view();
            NDScope.Detach(shared);
            shared.Storage.InternalArray.TryAddRef();   // never released: the attribute may be read for the process's life
            Volatile.Write(ref slot, shared);
        }

        /// <summary>
        ///     The class object of a class given as a C# type (NumPy's class object, e.g. <c>convert(kind=Chebyshev)</c> →
        ///     <c>kind: typeof(NDChebyshev)</c>).
        /// </summary>
        /// <param name="type">The type: <see cref="NDPolynomial"/>, <see cref="NDChebyshev"/>, <see cref="NDLegendre"/>,
        ///     <see cref="NDLaguerre"/>, <see cref="NDHermite"/> or <see cref="NDHermiteE"/>.</param>
        /// <returns>The class object.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="type"/> is null.</exception>
        /// <exception cref="AttributeError">Any other type: Python reads the class attributes NumPy's code asks for
        ///     (<c>kind.domain</c>, <c>kind.identity</c>), and a class without them raises
        ///     <c>type object '{name}' has no attribute 'domain'</c> — NumSharp's message names the C# type.</exception>
        internal static NDPolyClass Of(Type type)
        {
            if (type is null) throw new ArgumentNullException(nameof(type));
            if (type == typeof(NDPolynomial)) return NDPolynomial.Class;
            if (type == typeof(NDChebyshev)) return NDChebyshev.Class;
            if (type == typeof(NDLegendre)) return NDLegendre.Class;
            if (type == typeof(NDLaguerre)) return NDLaguerre.Class;
            if (type == typeof(NDHermite)) return NDHermite.Class;
            if (type == typeof(NDHermiteE)) return NDHermiteE.Class;
            throw new AttributeError($"type object '{type.Name}' has no attribute 'domain'");
        }
    }
}
