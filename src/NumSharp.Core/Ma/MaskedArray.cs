using System;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Masked-array namespace — the port of Python's <c>numpy.ma</c> module. Accessed exactly like
        ///     Python: <c>np.ma.add(a, b)</c>, <c>np.ma.abs(a)</c>, <c>np.ma.sqrt(a)</c>, etc. Mirrors the
        ///     <see cref="fft"/>/<see cref="random"/> facade shape (a lowercase property returning a module
        ///     object whose methods are the functions).
        ///     <para>
        ///     A <see cref="MaskedArray"/> pairs an ordinary <see cref="NDArray"/> of DATA with a boolean
        ///     <see cref="NDArray"/> MASK (True = "this element is invalid / ignore it"). The masked ufuncs
        ///     here do NOT re-implement arithmetic: exactly like NumPy they call the underlying
        ///     <c>np.*</c> ufunc on the raw data and only propagate the mask around it, so every layout,
        ///     dtype, promotion rule and SIMD/IL kernel of the base op is inherited for free.
        ///     </para>
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/maskedarray.generic.html</remarks>
        public static MaskedArrayModule ma { get; } = new MaskedArrayModule();
    }

    /// <summary>
    ///     A NumPy <c>numpy.ma.MaskedArray</c>: an <see cref="NDArray"/> of data paired with a boolean mask
    ///     marking elements to ignore. Distinct from a plain <see cref="NDArray"/> precisely because
    ///     operations SKIP the masked elements' influence and preserve their mask into the result — reach for
    ///     it only when some entries are invalid/missing and must not participate in arithmetic while keeping
    ///     the array's shape.
    ///     <para>
    ///     The mask is stored as <c>null</c> when there is no mask (NumPy's <c>nomask</c> fast path), which is
    ///     what keeps an all-valid masked array as cheap as the underlying <see cref="NDArray"/>. The data is
    ///     never null. This type deliberately does NOT own/dispose its arrays — it holds references the way
    ///     NumPy's MaskedArray holds views over its base — so the caller keeps the source arrays alive.
    ///     </para>
    /// </summary>
    public class MaskedArray
    {
        /// <summary>The underlying data buffer. Never null. Masked positions still hold a data value
        /// (whatever the producing op left there); that value is semantically hidden but observable via
        /// <see cref="data"/> for parity with NumPy.</summary>
        internal readonly NDArray _data;

        /// <summary>The boolean mask (True = masked/ignored), or <c>null</c> for NumPy's <c>nomask</c>
        /// (no element masked). When non-null its shape equals <see cref="_data"/>'s shape.</summary>
        internal readonly NDArray _mask;

        /// <summary>The value substituted for masked elements by <see cref="filled(object)"/> when the caller
        /// gives none; null means "use the dtype default" (<see cref="MaskedArrayModule.default_fill_value"/>).</summary>
        internal readonly object _fill_value;

        /// <summary>
        ///     Wraps a data array and (optional) mask into a masked array WITHOUT copying either — the arrays
        ///     are aliased, so later writes to them are visible here and vice-versa. Pass <paramref name="mask"/>
        ///     null for the no-mask (<c>nomask</c>) fast path.
        /// </summary>
        /// <param name="data">The data buffer (aliased, not copied). Must not be null.</param>
        /// <param name="mask">The boolean mask (True = masked), or null for no mask. When non-null it should
        /// already match <paramref name="data"/>'s shape; callers building it via broadcasting materialize it first.</param>
        /// <param name="fill_value">Optional default fill value for <see cref="filled(object)"/>; null = dtype default.</param>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is null.</exception>
        internal MaskedArray(NDArray data, NDArray mask, object fill_value = null)
        {
            // Data is the one invariant that must always hold — every property and the printer dereference it.
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _mask = mask;
            _fill_value = fill_value;
        }

        /// <summary>The underlying data array (NumPy's <c>MaskedArray.data</c>). Masked positions are included
        /// with whatever value the producing op left; use <see cref="filled(object)"/> to substitute them.</summary>
        public NDArray data => _data;

        /// <summary>The boolean mask. Returns the shared <see cref="MaskedArrayModule.nomask"/> sentinel (a 0-d
        /// <c>False</c>) when no element is masked, matching NumPy's <c>MaskedArray.mask</c> which yields
        /// <c>nomask</c> rather than a full <c>False</c> array in that case.</summary>
        public NDArray mask => _mask ?? np.ma.nomask;

        /// <summary>The data shape as a <c>long[]</c> (NumPy's <c>.shape</c>; NumSharp dimensions are 64-bit).
        /// Data and mask share this shape.</summary>
        public long[] shape => _data.shape;

        /// <summary>The data <see cref="Shape"/> struct (dimensions/strides/flags) for internal shape math.</summary>
        internal Shape Shape => _data.Shape;

        /// <summary>The data dtype (NumPy's <c>.dtype</c>); the mask is always boolean regardless.</summary>
        public DType dtype => _data.dtype;

        /// <summary>Element type code of the data, for dtype dispatch.</summary>
        public NPTypeCode typecode => _data.typecode;

        /// <summary>Number of dimensions of the data (NumPy's <c>.ndim</c>).</summary>
        public int ndim => _data.ndim;

        /// <summary>Total element count (NumPy's <c>.size</c>).</summary>
        public long size => _data.size;

        /// <summary>
        ///     Returns a fresh data array with every masked element replaced by <paramref name="fill_value"/>
        ///     (NumPy's <c>MaskedArray.filled</c>). This is the standard way to hand masked data to code that
        ///     cannot see the mask — the mask is "baked in" as a sentinel value while the dtype is preserved.
        /// </summary>
        /// <param name="fill_value">The value written at masked positions; null uses this array's own
        /// <see cref="_fill_value"/> if set, else the dtype default (see
        /// <see cref="MaskedArrayModule.default_fill_value"/>). It is cast to the data dtype (unsafe cast, like NumPy).</param>
        /// <returns>A new C-contiguous <see cref="NDArray"/> of the data dtype; when there is no mask it is a
        /// copy of the data unchanged.</returns>
        public NDArray filled(object fill_value = null)
        {
            var result = _data.copy();
            // No mask ⇒ nothing to substitute; NumPy returns the data unchanged (we return a copy for safety).
            // NOTE: `is null` (reference check) — `_mask == null` would invoke NDArray's ELEMENTWISE `==`.
            if (_mask is null)
                return result;

            var fv = fill_value ?? _fill_value ?? MaskedArrayModule.default_fill_value(_data.dtype);
            // copyto broadcasts the scalar fill into every masked slot; unsafe cast because a generic fill
            // value (e.g. 1e20 into an int array default) must land in the data dtype exactly as NumPy coerces it.
            np.copyto(result, NDArray.Scalar(fv), casting: "unsafe", where: _mask);
            return result;
        }

        /// <summary>
        ///     Renders the array in a <c>masked_array(data=…, mask=…, fill_value=…)</c> shape with masked
        ///     elements shown as <c>--</c> for 0-D/1-D (the common case), falling back to separate data/mask
        ///     blocks for higher rank. NOTE: this is a readable approximation, NOT byte-identical to NumPy's
        ///     aligned masked repr — value/mask parity is the contract here, not print layout.
        /// </summary>
        /// <returns>A human-readable multi-line string.</returns>
        public override string ToString()
        {
            var fv = _fill_value ?? MaskedArrayModule.default_fill_value(_data.dtype);
            // Show data and mask as their own array reprs. Rendering "--" inline (NumPy's aligned masked repr)
            // is deliberately not attempted here — value/mask parity is the contract, not print layout.
            string maskStr = _mask is null ? "False" : _mask.ToString(false);
            return $"masked_array(data={_data.ToString(false)},\n             mask={maskStr},\n       fill_value={fv})";
        }

        /// <summary>
        ///     Adopts a plain <see cref="NDArray"/> as an unmasked <see cref="MaskedArray"/> (no element
        ///     masked), so every <c>np.ma.*</c> function accepts a bare array argument exactly as NumPy does.
        /// </summary>
        /// <param name="a">The array to wrap; null yields null.</param>
        public static implicit operator MaskedArray(NDArray a) => a is null ? null : new MaskedArray(a, null);
    }

    /// <summary>
    ///     NumPy's <c>numpy.ma.masked</c> singleton: the scalar constant that represents "a single masked
    ///     value". A masked ufunc that reduces to one fully-masked 0-D element returns this instance, matching
    ///     NumPy where <c>np.ma.add(masked, x)</c> and a masked 0-D result are <c>masked</c>.
    /// </summary>
    public sealed class MaskedConstant : MaskedArray
    {
        /// <summary>Builds the constant as a 0-D <c>0.0</c> datum under a 0-D <c>True</c> mask, i.e. one masked
        /// scalar — the same internal shape NumPy's <c>masked_singleton</c> carries.</summary>
        internal MaskedConstant() : base(NDArray.Scalar(0.0d), NDArray.Scalar(true)) { }

        /// <summary>Renders as the bare token <c>masked</c>, as NumPy prints its singleton.</summary>
        /// <returns>The string <c>"masked"</c>.</returns>
        public override string ToString() => "masked";
    }

    /// <summary>
    ///     The <c>numpy.ma</c> module surface, reachable as <see cref="np.ma"/>. Holds the masked-array
    ///     substrate (<see cref="getdata"/>/<see cref="getmask"/>/<see cref="getmaskarray"/>/<see cref="filled"/>/
    ///     <see cref="masked"/>/<see cref="nomask"/>) and the masked <b>ufunc family</b> — every unary/binary
    ///     arithmetic, comparison, logical and bitwise op NumPy exposes on <c>numpy.ma</c> via its
    ///     <c>_MaskedUnaryOperation</c>/<c>_MaskedBinaryOperation</c>/<c>_DomainedBinaryOperation</c> wrappers.
    ///     <para>
    ///     Each ufunc is a thin wrapper that (1) pulls the raw data out of its operand(s), (2) calls the
    ///     already-optimized <c>np.*</c> kernel on that data, and (3) propagates the mask — pass-through for a
    ///     domain-free unary, logical-OR of the input masks for a binary, and OR'd with the domain's
    ///     invalid-input predicate for a domained op (sqrt/log/divide/…). No new compute loop is emitted; the
    ///     work is mask bookkeeping over existing IL/NDIter-backed ops.
    ///     </para>
    ///     <para>Reductions (<c>sum</c>/<c>mean</c>/…), creation (<c>zeros</c>/<c>arange</c>/…) and the
    ///     <c>extras</c> functions are separate mechanisms and are NOT part of this ufunc-family module yet.</para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.ma.html</remarks>
    [ModuleName("np.ma")]
    public partial class MaskedArrayModule
    {
        /// <summary>NumPy uses <c>np.finfo(float).tiny</c> (smallest positive normal float64) as the
        /// safe-division threshold; masking where <c>|a|·tiny ≥ |b|</c> catches division by (near-)zero.</summary>
        private const double SafeDivideTiny = 2.2250738585072014e-308;

        /// <summary>Constructs the module; there is one shared instance behind <see cref="np.ma"/>.</summary>
        internal MaskedArrayModule() { }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Substrate: sentinels, extraction, mask construction, filling
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>NumPy's <c>nomask</c>: the shared 0-D <c>False</c> that <see cref="MaskedArray.mask"/>
        /// returns when nothing is masked. Internally "no mask" is stored as a null mask; this is only the
        /// public face of that state.</summary>
        public NDArray nomask { get; } = NDArray.Scalar(false);

        /// <summary>NumPy's <c>masked</c> / <c>masked_singleton</c>: the constant standing for one masked
        /// value; returned by ufuncs whose result is a single fully-masked 0-D element.</summary>
        public MaskedConstant masked { get; } = new MaskedConstant();

        /// <summary>NumPy's <c>MaskType</c>: the dtype of a mask, i.e. boolean.</summary>
        public DType MaskType => np.@bool;

        /// <summary>
        ///     Returns the data of <paramref name="a"/> as a plain <see cref="NDArray"/> — the underlying data
        ///     for a <see cref="MaskedArray"/>, the array itself for an <see cref="NDArray"/>, or a fresh array
        ///     for any other array-like/scalar (NumPy's <c>getdata</c>). The mask, if any, is discarded.
        /// </summary>
        /// <param name="a">A <see cref="MaskedArray"/>, <see cref="NDArray"/>, C# scalar, or array-like.</param>
        /// <returns>The data as an <see cref="NDArray"/> (aliased when possible, not copied).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public NDArray getdata(object a) => AsData(a);

        /// <summary>
        ///     Returns the boolean mask of <paramref name="a"/>, or <see cref="nomask"/> when it has none
        ///     (NumPy's <c>getmask</c>). A plain <see cref="NDArray"/> or scalar has no mask.
        /// </summary>
        /// <param name="a">A <see cref="MaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <returns>The mask array, or the <see cref="nomask"/> sentinel.</returns>
        public NDArray getmask(object a) => (a as MaskedArray)?._mask ?? nomask;

        /// <summary>
        ///     Returns the mask of <paramref name="a"/> as a FULL boolean array of its data's shape, allocating
        ///     an all-<c>False</c> array when there is no mask (NumPy's <c>getmaskarray</c>). Unlike
        ///     <see cref="getmask"/> this never returns the 0-D <c>nomask</c> sentinel, so callers can index it
        ///     positionally.
        /// </summary>
        /// <param name="a">A <see cref="MaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <returns>A boolean <see cref="NDArray"/> matching the data shape.</returns>
        public NDArray getmaskarray(object a)
        {
            var m = (a as MaskedArray)?._mask;
            if (m is not null)
                return m;
            return np.zeros(AsData(a).Shape, np.@bool);
        }

        /// <summary>
        ///     True iff <paramref name="a"/> is a boolean <see cref="NDArray"/> usable as a mask (NumPy's
        ///     <c>is_mask</c> — a plain check on type/dtype, NOT whether any element is set).
        /// </summary>
        /// <param name="a">Any object.</param>
        /// <returns>True if <paramref name="a"/> is an <see cref="NDArray"/> of boolean dtype.</returns>
        public bool is_mask(object a) => a is NDArray nd && nd.typecode == NPTypeCode.Boolean;

        /// <summary>
        ///     True iff <paramref name="x"/> is a <see cref="MaskedArray"/> (NumPy's <c>isMaskedArray</c>/<c>is_masked</c>
        ///     type check). Does not inspect whether any element is actually masked.
        /// </summary>
        /// <param name="x">Any object.</param>
        /// <returns>True for a <see cref="MaskedArray"/> (including the <see cref="masked"/> constant).</returns>
        public bool isMaskedArray(object x) => x is MaskedArray;

        /// <summary>
        ///     Substitutes <paramref name="fill_value"/> for masked elements of <paramref name="a"/> and returns
        ///     a plain array (NumPy's module-level <c>ma.filled</c>). A non-masked input is returned as an array
        ///     unchanged.
        /// </summary>
        /// <param name="a">A <see cref="MaskedArray"/> or array-like.</param>
        /// <param name="fill_value">Fill for masked slots; null uses the dtype default.</param>
        /// <returns>A plain <see cref="NDArray"/> with masked slots filled.</returns>
        public NDArray filled(object a, object fill_value = null)
            => a is MaskedArray m ? m.filled(fill_value) : AsData(a);

        /// <summary>
        ///     Builds a masked array from data and an optional mask (NumPy's <c>ma.array</c>/<c>masked_array</c>).
        ///     The data is taken via <see cref="getdata"/>; the mask, if given, is coerced to boolean.
        /// </summary>
        /// <param name="data">Data array-like (a <see cref="MaskedArray"/>'s own mask is honored when no explicit mask is given).</param>
        /// <param name="mask">Optional boolean mask array-like; null/omitted ⇒ inherit <paramref name="data"/>'s mask or none.</param>
        /// <param name="fill_value">Optional default fill value for <see cref="MaskedArray.filled(object)"/>.</param>
        /// <param name="copy">When true, copies the data (NumPy's <c>copy=</c>); default false aliases it.</param>
        /// <returns>A new <see cref="MaskedArray"/>.</returns>
        public MaskedArray array(object data, object mask = null, object fill_value = null, bool copy = false)
        {
            var d = AsData(data);
            if (copy)
                d = d.copy();
            // Explicit mask wins; otherwise inherit an incoming MaskedArray's mask (NumPy semantics).
            NDArray m = mask != null ? AsData(mask).astype(np.@bool) : (data as MaskedArray)?._mask;
            return new MaskedArray(d, m, fill_value);
        }

        /// <summary>Alias of <see cref="array"/> matching NumPy's <c>ma.masked_array</c> name.</summary>
        /// <param name="data">Data array-like.</param>
        /// <param name="mask">Optional boolean mask.</param>
        /// <param name="fill_value">Optional default fill value.</param>
        /// <param name="copy">Copy the data when true.</param>
        /// <returns>A new <see cref="MaskedArray"/>.</returns>
        public MaskedArray masked_array(object data, object mask = null, object fill_value = null, bool copy = false)
            => array(data, mask, fill_value, copy);

        /// <summary>
        ///     The default fill value NumPy assigns per dtype kind (<c>default_fill_value</c>): <c>1e20</c> for
        ///     float/complex, <c>999999</c> for integers, <c>True</c> for boolean, else <c>0</c>. Used by
        ///     <see cref="MaskedArray.filled(object)"/> when the caller supplies none.
        /// </summary>
        /// <param name="dtype">The data dtype to pick a fill for.</param>
        /// <returns>A boxed scalar of a type castable into <paramref name="dtype"/>.</returns>
        public static object default_fill_value(DType dtype)
        {
            switch (dtype.typecode)
            {
                case NPTypeCode.Boolean: return true;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                case NPTypeCode.Complex: return 1e20d;
                default: return 999999L; // every integer kind (signed/unsigned/char)
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  The three ufunc-wrapper mechanisms (ports of NumPy's operation classes)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Port of NumPy's <c>_MaskedUnaryOperation.__call__</c>: applies unary op <paramref name="f"/> to
        ///     the raw data and propagates the mask. For a DOMAIN-FREE op the input mask passes straight
        ///     through; for a DOMAINED op (sqrt/log/…) the result mask additionally covers non-finite results
        ///     and inputs the domain rejects. Masked positions get the original input value written back so the
        ///     data stays meaningful (NumPy's <c>copyto(result, d, where=m)</c>).
        /// </summary>
        /// <param name="f">The underlying <c>np.*</c> unary op (e.g. <c>np.abs</c>).</param>
        /// <param name="a">The operand (<see cref="MaskedArray"/>/<see cref="NDArray"/>/scalar).</param>
        /// <param name="domain">Optional invalid-input predicate: given the data, returns a boolean mask of
        /// elements to mask; null for a domain-free op.</param>
        /// <returns>The masked result, or the <see cref="masked"/> constant for a fully-masked 0-D result.</returns>
        private MaskedArray Unary(Func<NDArray, NDArray> f, object a, Func<NDArray, NDArray> domain = null)
        {
            var d = AsData(a);
            var result = f(d);
            NDArray m;
            if (domain != null)
            {
                // Domained: mask non-finite outputs, the domain's rejected inputs, and any incoming mask.
                m = Or(NonFiniteMask(result), domain(d));
                m = Or(m, (a as MaskedArray)?._mask);
            }
            else
            {
                // Domain-free (abs/negative/sin/…): the mask is simply carried through unchanged.
                m = (a as MaskedArray)?._mask;
            }

            // NumPy's unary fill-back is copyto(result, d, where=m) with the DEFAULT 'same_kind' casting,
            // caught if it fails — so a dtype-changing unary (e.g. angle: complex→float) keeps its computed
            // value at masked slots rather than force-casting the input in.
            return Wrap(result, m, d, casting: "same_kind");
        }

        /// <summary>
        ///     Port of NumPy's <c>_MaskedBinaryOperation.__call__</c>: applies binary op <paramref name="f"/> to
        ///     the two operands' data and sets the result mask to the logical-OR of the input masks — a position
        ///     is masked iff EITHER operand was masked there. Masked positions get the left operand's data
        ///     written back (NumPy's <c>copyto(result, da, where=m)</c>).
        /// </summary>
        /// <param name="f">The underlying <c>np.*</c> binary op (e.g. <c>np.add</c>).</param>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>The masked result, or the <see cref="masked"/> constant for a fully-masked 0-D result.</returns>
        private MaskedArray Binary(Func<NDArray, NDArray, NDArray> f, object a, object b)
        {
            var da = AsData(a);
            var db = AsData(b);
            var result = f(da, db);
            // OR the two operand masks (nomask fast path: both absent ⇒ no result mask at all).
            var m = Or((a as MaskedArray)?._mask, (b as MaskedArray)?._mask);
            return Wrap(result, m, da);
        }

        /// <summary>
        ///     Port of NumPy's <c>_DomainedBinaryOperation.__call__</c> (divide/floor_divide/remainder/fmod):
        ///     like <see cref="Binary"/> but the result mask ALSO covers non-finite outputs and inputs the
        ///     domain rejects (e.g. division by ~zero via <see cref="SafeDivideDomain"/>). Masked positions are
        ///     zeroed then the left operand's data is added back where a safe cast allows, reproducing NumPy's
        ///     <c>copyto(result, 0, …); result += m*da</c>.
        /// </summary>
        /// <param name="f">The underlying <c>np.*</c> binary op (e.g. <c>np.divide</c>).</param>
        /// <param name="domain">Invalid-input predicate over the two data arrays returning a boolean mask.</param>
        /// <param name="a">Left operand (numerator).</param>
        /// <param name="b">Right operand (denominator).</param>
        /// <returns>The masked result, or the <see cref="masked"/> constant for a fully-masked 0-D result.</returns>
        private MaskedArray DomainedBinary(Func<NDArray, NDArray, NDArray> f, Func<NDArray, NDArray, NDArray> domain, object a, object b)
        {
            var da = AsData(a);
            var db = AsData(b);
            var result = f(da, db);
            var m = Or(NonFiniteMask(result), domain(da, db));
            m = Or(m, (a as MaskedArray)?._mask);
            m = Or(m, (b as MaskedArray)?._mask);
            // Domained fill-back differs from the plain binary: NumPy zeroes masked slots then re-adds da,
            // which we reproduce as "put da back where it can safely cast, else leave zero".
            return Wrap(result, m, da, domainedZeroFill: true);
        }

        /// <summary>
        ///     Shared tail of all three wrappers: materializes the mask to the result shape, handles the 0-D
        ///     scalar case, writes the fill-back value into masked slots, and boxes everything into a
        ///     <see cref="MaskedArray"/>.
        /// </summary>
        /// <param name="result">The freshly computed (owned, writeable) data result.</param>
        /// <param name="m">The propagated mask, or null for no mask.</param>
        /// <param name="fillbackSource">The array whose values are restored at masked positions (the input for
        /// unary, the left operand for binary).</param>
        /// <param name="domainedZeroFill">When true, use the domained fill-back (zero then re-add) instead of
        /// the plain copy-back.</param>
        /// <returns>The masked result, or the <see cref="masked"/> constant when a 0-D result is fully masked.</returns>
        private MaskedArray Wrap(NDArray result, NDArray m, NDArray fillbackSource, bool domainedZeroFill = false, string casting = "unsafe")
        {
            // Keep the stored mask shape-consistent with the data (broadcast predicates/operand masks up).
            // `is not null` throughout — `m != null` would invoke NDArray's ELEMENTWISE `!=`.
            if (m is not null && !m.Shape.Equals(result.Shape))
                m = np.broadcast_to(m, result.Shape).copy();

            // 0-D result: NumPy returns the `masked` singleton if masked, else the bare scalar. We return the
            // singleton when masked and a 0-D masked array otherwise (a deliberate C# type-consistency choice).
            if (result.ndim == 0)
                return (m is not null && np.any(m)) ? masked : new MaskedArray(result, null);

            // Restore data at masked positions so `.data` is meaningful there (masked values are still hidden).
            // No `np.any(m)` gate: copyto with an all-false mask is already a no-op, and the scan would cost a
            // full extra O(n) pass — NumPy's unary path likewise fills back unconditionally.
            if (m is not null)
                FillBack(result, fillbackSource, m, domainedZeroFill, casting);

            return new MaskedArray(result, m);
        }

        /// <summary>
        ///     Writes the pre-op data back into masked slots of <paramref name="result"/>. Plain path copies
        ///     <paramref name="source"/> straight in; the domained path zeroes first and only re-adds
        ///     <paramref name="source"/> when it casts safely to the result dtype (NumPy's exact behavior).
        ///     Both are best-effort: an incompatible cast is swallowed, leaving the computed/zeroed value, as
        ///     NumPy's <c>try/except</c> does.
        /// </summary>
        /// <param name="result">The result array to patch in place.</param>
        /// <param name="source">The data to restore at masked positions.</param>
        /// <param name="m">The boolean mask (result-shaped).</param>
        /// <param name="domainedZeroFill">Selects the domained zero-then-readd path.</param>
        private static void FillBack(NDArray result, NDArray source, NDArray m, bool domainedZeroFill, string casting)
        {
            try
            {
                if (domainedZeroFill)
                {
                    // NumPy: copyto(result, 0, where=m); then result += m*da only if da casts safely.
                    np.copyto(result, NDArray.Scalar(0), casting: "unsafe", where: m);
                    if (np.can_cast(source.dtype, result.dtype, "safe"))
                        np.copyto(result, source, casting: "unsafe", where: m);
                }
                else
                {
                    // Plain path: unary passes "same_kind" (NumPy's default, caught on failure); binary "unsafe".
                    np.copyto(result, source, casting: casting, where: m);
                }
            }
            catch
            {
                // Parity with NumPy's bare except: if the data cannot be placed back, keep what is there.
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Small internal helpers
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Normalizes any accepted operand to its data <see cref="NDArray"/>: unwraps a
        ///     <see cref="MaskedArray"/>, passes an <see cref="NDArray"/> through, and converts a C# scalar or
        ///     array-like via <see cref="np.asarray(object,DType,char)"/>.
        /// </summary>
        /// <param name="a">The operand.</param>
        /// <returns>The operand's data as an <see cref="NDArray"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        private static NDArray AsData(object a)
        {
            switch (a)
            {
                case null: throw new ArgumentNullException(nameof(a));
                case MaskedArray m: return m._data;
                case NDArray nd: return nd;
                case Array arr: return np.array(arr); // C# array-like (double[], int[,], …)
                default: return NDArray.Scalar(a);    // boxed C# scalar (int/double/bool/…)
            }
        }

        /// <summary>
        ///     Combines two optional masks where null means "no contribution" (NumPy's <c>nomask</c>): returns
        ///     null iff both are null, otherwise their broadcasting logical-OR. This is what preserves the
        ///     no-mask fast path through the wrappers.
        /// </summary>
        /// <param name="x">First mask or null.</param>
        /// <param name="y">Second mask or null.</param>
        /// <returns>The OR of the present masks, or null when neither is present.</returns>
        private static NDArray Or(NDArray x, NDArray y)
        {
            // `is null` (reference check): `x == null` would run NDArray's ELEMENTWISE `==` and never null-test.
            if (x is null) return y;
            if (y is null) return x;
            return x | y;
        }

        /// <summary>
        ///     The <c>~isfinite(result)</c> term of a domained op's mask, but only for inexact (float/complex)
        ///     results — for an integer result every element is finite, so this contributes nothing and is
        ///     returned as null to skip an <c>isfinite</c> call the base op does not support meaningfully.
        /// </summary>
        /// <param name="result">The computed result.</param>
        /// <returns>A boolean "non-finite" mask, or null for an exact (integer/bool) result.</returns>
        private static NDArray NonFiniteMask(NDArray result)
        {
            switch (result.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    return !np.isfinite(result); // NaN/±inf ⇒ True
                default:
                    return null; // integer/bool: always finite
            }
        }

        // ── Domain predicates (ports of NumPy's _Domain* classes) ──────────────────────
        // Built from greater/greater_equal/less_equal/logical_not (np.less has a 0-D-scalar
        // broadcasting quirk we route around). NaN differences between `less(x,v)` and
        // `!(x>=v)` are absorbed by the NonFiniteMask term that every domained op also ORs in.

        /// <summary>DomainGreaterEqual(v): masks where <c>x &lt; v</c> (used by sqrt at 0, arccosh at 1).</summary>
        /// <param name="x">Data.</param><param name="v">Critical value.</param>
        /// <returns>Boolean mask, True where <c>x &lt; v</c>.</returns>
        private static NDArray DomainGreaterEqual(NDArray x, double v) => np.logical_not(np.greater_equal(x, NDArray.Scalar(v)));

        /// <summary>DomainGreater(v): masks where <c>x &lt;= v</c> (used by log/log2/log10 at 0).</summary>
        /// <param name="x">Data.</param><param name="v">Critical value.</param>
        /// <returns>Boolean mask, True where <c>x &lt;= v</c>.</returns>
        private static NDArray DomainGreater(NDArray x, double v) => np.less_equal(x, NDArray.Scalar(v));

        /// <summary>DomainCheckInterval(a,b): masks where <c>x &lt; a</c> or <c>x &gt; b</c> (arcsin/arccos/arctanh).</summary>
        /// <param name="x">Data.</param><param name="a">Lower bound.</param><param name="b">Upper bound.</param>
        /// <returns>Boolean mask, True outside <c>[a, b]</c>.</returns>
        private static NDArray DomainCheckInterval(NDArray x, double a, double b)
            => np.logical_not(np.greater_equal(x, NDArray.Scalar(a))) | np.greater(x, NDArray.Scalar(b));

        /// <summary>DomainTan(eps): masks where <c>|cos(x)| &lt; eps</c> (tan's poles).</summary>
        /// <param name="x">Data.</param><param name="eps">Pole tolerance.</param>
        /// <returns>Boolean mask, True near the poles.</returns>
        private static NDArray DomainTan(NDArray x, double eps)
            => np.logical_not(np.greater_equal(np.abs(np.cos(x)), NDArray.Scalar(eps)));

        /// <summary>DomainSafeDivide: masks where <c>|a|·tiny ≥ |b|</c>, i.e. the denominator is (near) zero.</summary>
        /// <param name="a">Numerator data.</param><param name="b">Denominator data.</param>
        /// <returns>Boolean mask, True where division is unsafe.</returns>
        private static NDArray SafeDivideDomain(NDArray a, NDArray b)
            => np.greater_equal(np.multiply(np.abs(a), NDArray.Scalar(SafeDivideTiny)), np.abs(b));

        // ─────────────────────────────────────────────────────────────────────────────
        //  Unary ufuncs — domain-free (mask passes through)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise absolute value over the data; the mask is unchanged. Alias of <see cref="absolute"/>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of |data|.</returns>
        public MaskedArray abs(object a) => Unary(x => np.abs(x), a);

        /// <summary>Element-wise absolute value over the data; the mask is unchanged (same op as <see cref="abs"/>).</summary>
        /// <param name="a">Operand.</param><returns>A masked array of |data|.</returns>
        public MaskedArray absolute(object a) => Unary(x => np.abs(x), a);

        /// <summary>Element-wise negation; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of -data.</returns>
        public MaskedArray negative(object a) => Unary(x => np.negative(x), a);

        /// <summary>Float-only absolute value (promotes ints to float); mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of the float |data|.</returns>
        public MaskedArray fabs(object a) => Unary(x => np.fabs(x), a);

        /// <summary>Complex conjugate (real data passes through); mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of conj(data).</returns>
        public MaskedArray conjugate(object a) => Unary(x => np.conjugate(x), a);

        /// <summary>Angle (argument) of the elements; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of angle(data).</returns>
        public MaskedArray angle(object a) => Unary(x => np.angle(x), a);

        /// <summary>Round to nearest integer, half-to-even; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of round(data).</returns>
        public MaskedArray around(object a) => Unary(x => np.around(x), a);

        /// <summary>Floor; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of floor(data).</returns>
        public MaskedArray floor(object a) => Unary(x => np.floor(x), a);

        /// <summary>Ceiling; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of ceil(data).</returns>
        public MaskedArray ceil(object a) => Unary(x => np.ceil(x), a);

        /// <summary>e^x; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of exp(data).</returns>
        public MaskedArray exp(object a) => Unary(x => np.exp(x), a);

        /// <summary>Sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sin(data).</returns>
        public MaskedArray sin(object a) => Unary(x => np.sin(x), a);

        /// <summary>Cosine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of cos(data).</returns>
        public MaskedArray cos(object a) => Unary(x => np.cos(x), a);

        /// <summary>Hyperbolic sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sinh(data).</returns>
        public MaskedArray sinh(object a) => Unary(x => np.sinh(x), a);

        /// <summary>Hyperbolic cosine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of cosh(data).</returns>
        public MaskedArray cosh(object a) => Unary(x => np.cosh(x), a);

        /// <summary>Hyperbolic tangent; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of tanh(data).</returns>
        public MaskedArray tanh(object a) => Unary(x => np.tanh(x), a);

        /// <summary>Arctangent; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arctan(data).</returns>
        public MaskedArray arctan(object a) => Unary(x => np.arctan(x), a);

        /// <summary>Inverse hyperbolic sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arcsinh(data).</returns>
        public MaskedArray arcsinh(object a) => Unary(x => np.arcsinh(x), a);

        /// <summary>Logical NOT of the data; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked boolean array of !data.</returns>
        public MaskedArray logical_not(object a) => Unary(x => np.logical_not(x), a);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Unary ufuncs — domained (invalid inputs are additionally masked)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Square root; additionally MASKS inputs <c>&lt; 0</c> (and non-finite results) rather than
        /// letting them produce NaN, which is the whole point of the masked variant.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sqrt(data) with negatives masked.</returns>
        public MaskedArray sqrt(object a) => Unary(x => np.sqrt(x), a, d => DomainGreaterEqual(d, 0.0));

        /// <summary>Natural log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log(data) with non-positives masked.</returns>
        public MaskedArray log(object a) => Unary(x => np.log(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Base-2 log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log2(data) with non-positives masked.</returns>
        public MaskedArray log2(object a) => Unary(x => np.log2(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Base-10 log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log10(data) with non-positives masked.</returns>
        public MaskedArray log10(object a) => Unary(x => np.log10(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Tangent; additionally masks inputs near its poles (<c>|cos(x)| &lt; 1e-35</c>).</summary>
        /// <param name="a">Operand.</param><returns>A masked array of tan(data) with pole-neighbors masked.</returns>
        public MaskedArray tan(object a) => Unary(x => np.tan(x), a, d => DomainTan(d, 1e-35));

        /// <summary>Arcsine; additionally masks inputs outside <c>[-1, 1]</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arcsin(data) with out-of-domain inputs masked.</returns>
        public MaskedArray arcsin(object a) => Unary(x => np.arcsin(x), a, d => DomainCheckInterval(d, -1.0, 1.0));

        /// <summary>Arccosine; additionally masks inputs outside <c>[-1, 1]</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arccos(data) with out-of-domain inputs masked.</returns>
        public MaskedArray arccos(object a) => Unary(x => np.arccos(x), a, d => DomainCheckInterval(d, -1.0, 1.0));

        /// <summary>Inverse hyperbolic cosine; additionally masks inputs <c>&lt; 1</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arccosh(data) with sub-1 inputs masked.</returns>
        public MaskedArray arccosh(object a) => Unary(x => np.arccosh(x), a, d => DomainGreaterEqual(d, 1.0));

        /// <summary>Inverse hyperbolic tangent; additionally masks inputs outside <c>(-1, 1)</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arctanh(data) with out-of-domain inputs masked.</returns>
        public MaskedArray arctanh(object a) => Unary(x => np.arctanh(x), a, d => DomainCheckInterval(d, -1.0 + 1e-15, 1.0 - 1e-15));

        // ─────────────────────────────────────────────────────────────────────────────
        //  Binary ufuncs — mask = OR of the two input masks
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise sum; result is masked wherever EITHER operand was masked.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a+b.</returns>
        public MaskedArray add(object a, object b) => Binary((x, y) => np.add(x, y), a, b);

        /// <summary>Element-wise difference; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a-b.</returns>
        public MaskedArray subtract(object a, object b) => Binary((x, y) => np.subtract(x, y), a, b);

        /// <summary>Element-wise product; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a*b.</returns>
        public MaskedArray multiply(object a, object b) => Binary((x, y) => np.multiply(x, y), a, b);

        /// <summary>Two-argument arctangent; masked where either operand was.</summary>
        /// <param name="a">y operand.</param><param name="b">x operand.</param>
        /// <returns>A masked array of arctan2(a, b).</returns>
        public MaskedArray arctan2(object a, object b) => Binary((x, y) => np.arctan2(x, y), a, b);

        /// <summary>Euclidean hypotenuse; masked where either operand was.</summary>
        /// <param name="a">Left leg.</param><param name="b">Right leg.</param>
        /// <returns>A masked array of hypot(a, b).</returns>
        public MaskedArray hypot(object a, object b) => Binary((x, y) => np.hypot(x, y), a, b);

        /// <summary>Element-wise equality; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a==b.</returns>
        public MaskedArray equal(object a, object b) => Binary((x, y) => np.equal(x, y), a, b);

        /// <summary>Element-wise inequality; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a!=b.</returns>
        public MaskedArray not_equal(object a, object b) => Binary((x, y) => np.not_equal(x, y), a, b);

        /// <summary>Element-wise less-than; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&lt;b.</returns>
        public MaskedArray less(object a, object b) => Binary((x, y) => np.less(x, y), a, b);

        /// <summary>Element-wise less-or-equal; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&lt;=b.</returns>
        public MaskedArray less_equal(object a, object b) => Binary((x, y) => np.less_equal(x, y), a, b);

        /// <summary>Element-wise greater-than; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&gt;b.</returns>
        public MaskedArray greater(object a, object b) => Binary((x, y) => np.greater(x, y), a, b);

        /// <summary>Element-wise greater-or-equal; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&gt;=b.</returns>
        public MaskedArray greater_equal(object a, object b) => Binary((x, y) => np.greater_equal(x, y), a, b);

        /// <summary>Logical AND; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&amp;&amp;b.</returns>
        public MaskedArray logical_and(object a, object b) => Binary((x, y) => np.logical_and(x, y), a, b);

        /// <summary>Logical OR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a||b.</returns>
        public MaskedArray logical_or(object a, object b) => Binary((x, y) => np.logical_or(x, y), a, b);

        /// <summary>Logical XOR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a^b.</returns>
        public MaskedArray logical_xor(object a, object b) => Binary((x, y) => np.logical_xor(x, y), a, b);

        /// <summary>Bitwise AND; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a&amp;b.</returns>
        public MaskedArray bitwise_and(object a, object b) => Binary((x, y) => np.bitwise_and(x, y), a, b);

        /// <summary>Bitwise OR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a|b.</returns>
        public MaskedArray bitwise_or(object a, object b) => Binary((x, y) => np.bitwise_or(x, y), a, b);

        /// <summary>Bitwise XOR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a^b.</returns>
        public MaskedArray bitwise_xor(object a, object b) => Binary((x, y) => np.bitwise_xor(x, y), a, b);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Domained binary ufuncs — division family (unsafe divides are masked)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>True division; additionally MASKS positions where the denominator is (near) zero, so a
        /// masked <c>divide</c> never surfaces an inf/NaN from division by zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a/b with unsafe divides masked.</returns>
        public MaskedArray divide(object a, object b) => DomainedBinary((x, y) => np.divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Alias of <see cref="divide"/> (NumPy's <c>true_divide</c>).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a/b with unsafe divides masked.</returns>
        public MaskedArray true_divide(object a, object b) => DomainedBinary((x, y) => np.true_divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Floor division; masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a//b with unsafe divides masked.</returns>
        public MaskedArray floor_divide(object a, object b) => DomainedBinary((x, y) => np.floor_divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Remainder (floored, sign follows divisor); masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a%b with unsafe divides masked.</returns>
        public MaskedArray remainder(object a, object b) => DomainedBinary((x, y) => np.remainder(x, y), SafeDivideDomain, a, b);

        /// <summary>Alias of <see cref="remainder"/> (NumPy's <c>mod</c>).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a%b with unsafe divides masked.</returns>
        public MaskedArray mod(object a, object b) => DomainedBinary((x, y) => np.remainder(x, y), SafeDivideDomain, a, b);

        /// <summary>C-style remainder (sign follows dividend); masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of fmod(a, b) with unsafe divides masked.</returns>
        public MaskedArray fmod(object a, object b) => DomainedBinary((x, y) => np.fmod(x, y), SafeDivideDomain, a, b);
    }
}
