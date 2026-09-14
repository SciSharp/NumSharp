using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Generic;

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
        /// gives none; null means "use the dtype default" (<see cref="MaskedArrayModule.default_fill_value"/>).
        /// NOT readonly: NumPy's <c>MaskedArray.fill_value</c> is settable, so <see cref="fill_value"/> and
        /// <see cref="MaskedArrayModule.set_fill_value"/> mutate it in place on the SAME instance.</summary>
        internal object _fill_value;

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

            var fv = fill_value ?? _fill_value ?? np.ma.default_fill_value(_data.dtype);
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
            var fv = _fill_value ?? np.ma.default_fill_value(_data.dtype);
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

        // ── Mixed MaskedArray/NDArray operators. These EXPLICIT (MaskedArray, NDArray) / (NDArray, MaskedArray)
        //    overloads are REQUIRED: without them `ma + nd` and `nd + ma` are CS0034-ambiguous, because the
        //    implicit NDArray→MaskedArray conversion makes both the (…, MaskedArray) and (…, object) overloads
        //    applicable with no better candidate. An exact NDArray parameter beats both, so these win cleanly
        //    and return a mask-aware MaskedArray — matching NumPy, where `ma + nd`/`nd + ma` are MaskedArrays.
        /// <summary>Masked sum with an NDArray operand (mask = the masked operand's).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a+b, masked.</returns>
        public static MaskedArray operator +(MaskedArray a, NDArray b) => np.ma.add(a, b);
        /// <summary>Masked sum with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a+b, masked.</returns>
        public static MaskedArray operator +(NDArray a, MaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked difference with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a-b, masked.</returns>
        public static MaskedArray operator -(MaskedArray a, NDArray b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a-b, masked.</returns>
        public static MaskedArray operator -(NDArray a, MaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked product with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a*b, masked.</returns>
        public static MaskedArray operator *(MaskedArray a, NDArray b) => np.ma.multiply(a, b);
        /// <summary>Masked product with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a*b, masked.</returns>
        public static MaskedArray operator *(NDArray a, MaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked division with an NDArray denominator.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator NDArray.</param><returns>a/b, masked.</returns>
        public static MaskedArray operator /(MaskedArray a, NDArray b) => np.ma.divide(a, b);
        /// <summary>Masked division with an NDArray numerator.</summary>
        /// <param name="a">Numerator NDArray.</param><param name="b">Denominator masked array.</param><returns>a/b, masked.</returns>
        public static MaskedArray operator /(NDArray a, MaskedArray b) => np.ma.divide(a, b);
        /// <summary>Masked less-than with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;b, masked bool.</returns>
        public static MaskedArray operator <(MaskedArray a, NDArray b) => np.ma.less(a, b);
        /// <summary>Masked less-than with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&lt;b, masked bool.</returns>
        public static MaskedArray operator <(NDArray a, MaskedArray b) => np.ma.less(a, b);
        /// <summary>Masked greater-than with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;b, masked bool.</returns>
        public static MaskedArray operator >(MaskedArray a, NDArray b) => np.ma.greater(a, b);
        /// <summary>Masked greater-than with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&gt;b, masked bool.</returns>
        public static MaskedArray operator >(NDArray a, MaskedArray b) => np.ma.greater(a, b);
        /// <summary>Masked less-or-equal with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;=b, masked bool.</returns>
        public static MaskedArray operator <=(MaskedArray a, NDArray b) => np.ma.less_equal(a, b);
        /// <summary>Masked less-or-equal with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&lt;=b, masked bool.</returns>
        public static MaskedArray operator <=(NDArray a, MaskedArray b) => np.ma.less_equal(a, b);
        /// <summary>Masked greater-or-equal with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;=b, masked bool.</returns>
        public static MaskedArray operator >=(MaskedArray a, NDArray b) => np.ma.greater_equal(a, b);
        /// <summary>Masked greater-or-equal with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&gt;=b, masked bool.</returns>
        public static MaskedArray operator >=(NDArray a, MaskedArray b) => np.ma.greater_equal(a, b);

        /// <summary>
        ///     Casts the DATA to <paramref name="dtype"/>, PRESERVING the mask (NumPy's <c>MaskedArray.astype</c>
        ///     — the mask is boolean and dtype-independent, so it rides through unchanged). Returns a
        ///     <see cref="MaskedArray"/>, unlike the underlying <see cref="NDArray.astype(DType,bool)"/>.
        /// </summary>
        /// <param name="dtype">Target element dtype.</param>
        /// <param name="copy">Copy the data even when the dtype already matches (NumPy default true).</param>
        /// <returns>A masked array of the new dtype with the same mask.</returns>
        public MaskedArray astype(DType dtype, bool copy = true) => new MaskedArray(_data.astype(dtype, copy), _mask, _fill_value);

        // ── Arithmetic operators — compose the np.ma.* binary ufuncs; masks propagate (OR). ──
        /// <summary>Masked element-wise sum (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a+b, masked.</returns>
        public static MaskedArray operator +(MaskedArray a, MaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked sum with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a+b, masked.</returns>
        public static MaskedArray operator +(MaskedArray a, object b) => np.ma.add(a, b);
        /// <summary>Masked sum with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a+b, masked.</returns>
        public static MaskedArray operator +(object a, MaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked element-wise difference.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a-b, masked.</returns>
        public static MaskedArray operator -(MaskedArray a, MaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a-b, masked.</returns>
        public static MaskedArray operator -(MaskedArray a, object b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a-b, masked.</returns>
        public static MaskedArray operator -(object a, MaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked element-wise product.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a*b, masked.</returns>
        public static MaskedArray operator *(MaskedArray a, MaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked product with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a*b, masked.</returns>
        public static MaskedArray operator *(MaskedArray a, object b) => np.ma.multiply(a, b);
        /// <summary>Masked product with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a*b, masked.</returns>
        public static MaskedArray operator *(object a, MaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked element-wise true division (unsafe divides are additionally masked).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param><returns>a/b, masked.</returns>
        public static MaskedArray operator /(MaskedArray a, MaskedArray b) => np.ma.divide(a, b);
        /// <summary>Masked division with a scalar/array-like denominator.</summary>
        /// <param name="a">Numerator masked array.</param><param name="b">Denominator scalar/array-like.</param><returns>a/b, masked.</returns>
        public static MaskedArray operator /(MaskedArray a, object b) => np.ma.divide(a, b);
        /// <summary>Masked division with a scalar/array-like numerator.</summary>
        /// <param name="a">Numerator scalar/array-like.</param><param name="b">Denominator masked array.</param><returns>a/b, masked.</returns>
        public static MaskedArray operator /(object a, MaskedArray b) => np.ma.divide(a, b);
        /// <summary>Masked element-wise negation.</summary>
        /// <param name="a">Operand.</param><returns>-a, masked.</returns>
        public static MaskedArray operator -(MaskedArray a) => np.ma.negative(a);

        // ── Ordering operators — return a masked boolean array (mask = OR of operands'). ==/!= are
        //    deliberately NOT overloaded (they would shadow reference equality and trip the `== null`
        //    elementwise trap); use np.ma.equal / np.ma.not_equal for element-wise (in)equality. ──
        /// <summary>Masked element-wise less-than.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;b, masked bool.</returns>
        public static MaskedArray operator <(MaskedArray a, MaskedArray b) => np.ma.less(a, b);
        /// <summary>Masked less-than against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&lt;b, masked bool.</returns>
        public static MaskedArray operator <(MaskedArray a, object b) => np.ma.less(a, b);
        /// <summary>Masked less-than with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&lt;b, masked bool.</returns>
        public static MaskedArray operator <(object a, MaskedArray b) => np.ma.less(a, b);
        /// <summary>Masked element-wise greater-than.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;b, masked bool.</returns>
        public static MaskedArray operator >(MaskedArray a, MaskedArray b) => np.ma.greater(a, b);
        /// <summary>Masked greater-than against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&gt;b, masked bool.</returns>
        public static MaskedArray operator >(MaskedArray a, object b) => np.ma.greater(a, b);
        /// <summary>Masked greater-than with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&gt;b, masked bool.</returns>
        public static MaskedArray operator >(object a, MaskedArray b) => np.ma.greater(a, b);
        /// <summary>Masked element-wise less-or-equal.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;=b, masked bool.</returns>
        public static MaskedArray operator <=(MaskedArray a, MaskedArray b) => np.ma.less_equal(a, b);
        /// <summary>Masked less-or-equal against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&lt;=b, masked bool.</returns>
        public static MaskedArray operator <=(MaskedArray a, object b) => np.ma.less_equal(a, b);
        /// <summary>Masked less-or-equal with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&lt;=b, masked bool.</returns>
        public static MaskedArray operator <=(object a, MaskedArray b) => np.ma.less_equal(a, b);
        /// <summary>Masked element-wise greater-or-equal.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;=b, masked bool.</returns>
        public static MaskedArray operator >=(MaskedArray a, MaskedArray b) => np.ma.greater_equal(a, b);
        /// <summary>Masked greater-or-equal against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&gt;=b, masked bool.</returns>
        public static MaskedArray operator >=(MaskedArray a, object b) => np.ma.greater_equal(a, b);
        /// <summary>Masked greater-or-equal with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&gt;=b, masked bool.</returns>
        public static MaskedArray operator >=(object a, MaskedArray b) => np.ma.greater_equal(a, b);

        // ── Instance reduction methods (NumPy's a.sum()/a.mean()/… — delegate to the np.ma.* funcs). ──
        /// <summary>Sum over unmasked elements. See <see cref="MaskedArrayModule.sum"/>.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked sum.</returns>
        public MaskedArray sum(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.sum(this, axis, dtype, keepdims);
        /// <summary>Product over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked product.</returns>
        public MaskedArray prod(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.prod(this, axis, dtype, keepdims);
        /// <summary>Mean over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked mean.</returns>
        public MaskedArray mean(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.mean(this, axis, dtype, keepdims);
        /// <summary>Minimum over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked minimum.</returns>
        public MaskedArray min(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.min(this, axis, fill_value, keepdims);
        /// <summary>Maximum over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked maximum.</returns>
        public MaskedArray max(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.max(this, axis, fill_value, keepdims);
        /// <summary>Peak-to-peak (max−min) over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked range.</returns>
        public MaskedArray ptp(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.ptp(this, axis, fill_value, keepdims);
        /// <summary>Standard deviation over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="ddof">Delta DOF.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked std.</returns>
        public MaskedArray std(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false) => np.ma.std(this, axis, dtype, ddof, keepdims);
        /// <summary>Variance over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="ddof">Delta DOF.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked variance.</returns>
        public MaskedArray var(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false) => np.ma.var(this, axis, dtype, ddof, keepdims);
        /// <summary>Count of unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>int64 count array.</returns>
        public NDArray count(int? axis = null, bool keepdims = false) => np.ma.count(this, axis, keepdims);
        /// <summary>Cumulative sum (masked positions preserved).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><returns>Masked cumulative sum.</returns>
        public MaskedArray cumsum(int? axis = null, DType dtype = null) => np.ma.cumsum(this, axis, dtype);
        /// <summary>Cumulative product (masked positions preserved).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><returns>Masked cumulative product.</returns>
        public MaskedArray cumprod(int? axis = null, DType dtype = null) => np.ma.cumprod(this, axis, dtype);
        /// <summary>Indices of the minimum (masked slots ignored).</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><returns>int64 index array.</returns>
        public NDArray argmin(int? axis = null, object fill_value = null) => np.ma.argmin(this, axis, fill_value);
        /// <summary>Indices of the maximum (masked slots ignored).</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><returns>int64 index array.</returns>
        public NDArray argmax(int? axis = null, object fill_value = null) => np.ma.argmax(this, axis, fill_value);
        /// <summary>Logical-AND reduction (masked slots treated as True).</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked bool.</returns>
        public MaskedArray all(int? axis = null, bool keepdims = false) => np.ma.all(this, axis, keepdims);
        /// <summary>Logical-OR reduction (masked slots treated as False).</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked bool.</returns>
        public MaskedArray any(int? axis = null, bool keepdims = false) => np.ma.any(this, axis, keepdims);
        /// <summary>Anomalies (deviations from the mean along the axis).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Mean accumulator dtype.</param><returns>Masked anomalies.</returns>
        public MaskedArray anom(int? axis = null, DType dtype = null) => np.ma.anom(this, axis, dtype);

        // ── fill_value — the sentinel filled() substitutes for masked slots. Settable (NumPy parity):
        //    the getter resolves the stored override or the dtype default; the setter mutates THIS instance
        //    (so `set_fill_value`/`common_fill_value` observe the change), it does NOT return a new array. ──
        /// <summary>
        ///     The value <see cref="filled()"/> substitutes for masked elements (NumPy's settable
        ///     <c>MaskedArray.fill_value</c>). Reading it resolves the stored override, else the dtype default
        ///     (see <see cref="MaskedArrayModule.default_fill_value"/>) — so it is never null. Writing it stores
        ///     the override on THIS instance in place (as NumPy mutates <c>a.fill_value = v</c>); a null clears
        ///     the override back to the dtype default.
        /// </summary>
        /// <remarks>NumPy's method forms <c>get_fill_value()</c>/<c>set_fill_value()</c> are NOT offered as
        /// instance methods: a C# property named <c>fill_value</c> already reserves those exact accessor names,
        /// so use the property (<c>a.fill_value</c> / <c>a.fill_value = v</c>) or the module-level
        /// <see cref="MaskedArrayModule.set_fill_value(object,object)"/>.</remarks>
        public object fill_value
        {
            get => _fill_value ?? np.ma.default_fill_value(_data.dtype);
            set => _fill_value = value;
        }

        // ── Instance shape/manipulation/selection — wire through to the np.ma.* module funcs (which apply
        //    the SAME transform to data and mask), so `a.reshape(...)`/`a.sort()` port from NumPy verbatim. ──
        /// <summary>Transpose (reverse all axes); NumPy's <c>MaskedArray.T</c>. Data and mask are transposed
        /// alike, so the mask stays aligned. A view — no data is moved.</summary>
        public MaskedArray T => np.ma.transpose(this);

        /// <summary>Matrix transpose — swaps the last two axes (NumPy's <c>MaskedArray.mT</c>); requires rank ≥ 2.</summary>
        /// <exception cref="ArgumentException">The array is rank &lt; 2.</exception>
        public MaskedArray mT
        {
            get
            {
                // NumPy raises for < 2-D; swapaxes(-2,-1) itself would too, but the message would name axes
                // rather than mT, so guard explicitly with NumPy's verbatim text.
                if (ndim < 2)
                    throw new ArgumentException("matrix transpose with ndim < 2 is undefined");
                return np.ma.swapaxes(this, -2, -1);
            }
        }

        /// <summary>The real part as a masked array (NumPy's <c>MaskedArray.real</c>); the mask rides through.</summary>
        public MaskedArray real => new MaskedArray(_data.real, _mask, _fill_value);

        /// <summary>The imaginary part as a masked array (NumPy's <c>MaskedArray.imag</c>); the mask rides through.
        /// For a non-complex dtype every element is 0.</summary>
        public MaskedArray imag => new MaskedArray(_data.imag, _mask, _fill_value);

        /// <summary>Flattened VIEW in row-major order (NumPy's <c>ravel</c>); mask flattened alike.</summary>
        /// <param name="order">'C' (row-major, default) or 'F' (column-major).</param><returns>The flattened masked array.</returns>
        public MaskedArray ravel(char order = 'C') => np.ma.ravel(this, order);

        /// <summary>Flattened COPY (NumPy's <c>flatten</c> — always copies, unlike <see cref="ravel"/>).</summary>
        /// <param name="order">'C' (default) or 'F'.</param><returns>The flattened masked copy.</returns>
        public MaskedArray flatten(char order = 'C') => np.ma.flatten(this, order);

        /// <summary>Reshape to <paramref name="shape"/> (mask reshaped alike).</summary>
        /// <param name="shape">New dimensions (one may be -1).</param><returns>The reshaped masked array.</returns>
        public MaskedArray reshape(params int[] shape) => np.ma.reshape(this, shape);

        /// <summary>Permute axes (mask transposed alike); reverses all axes when <paramref name="axes"/> is null.</summary>
        /// <param name="axes">Permutation, or null to reverse.</param><returns>The transposed masked array.</returns>
        public MaskedArray transpose(int[] axes = null) => np.ma.transpose(this, axes);

        /// <summary>Swap two axes (mask swapped alike).</summary>
        /// <param name="axis1">First axis.</param><param name="axis2">Second axis.</param><returns>The masked array with swapped axes.</returns>
        public MaskedArray swapaxes(int axis1, int axis2) => np.ma.swapaxes(this, axis1, axis2);

        /// <summary>Remove size-1 axes (mask squeezed alike).</summary>
        /// <returns>The squeezed masked array.</returns>
        public MaskedArray squeeze() => np.ma.squeeze(this);

        /// <summary>Repeat elements (mask repeated alike, so repeated masked entries stay masked).</summary>
        /// <param name="repeats">Repeat count.</param><param name="axis">Axis or null (flatten).</param><returns>The masked array.</returns>
        public MaskedArray repeat(int repeats, int? axis = null) => np.ma.repeat(this, repeats, axis);

        /// <summary>Gather elements by index (mask gathered alike, so a taken element keeps its masked-ness).</summary>
        /// <param name="indices">Integer index array.</param><param name="axis">Axis or null.</param><returns>The masked array.</returns>
        public MaskedArray take(NDArray indices, int? axis = null) => np.ma.take(this, indices, axis);

        /// <summary>The 1-D array of the UNMASKED values in C-order (NumPy's <c>MaskedArray.compressed</c>) —
        /// a plain <see cref="NDArray"/>, since a compressed result has no mask.</summary>
        /// <returns>A 1-D <see cref="NDArray"/> of the unmasked data.</returns>
        public NDArray compressed() => np.ma.compressed(this);

        /// <summary>Keep only elements where the flattened <paramref name="condition"/> is True (NumPy's
        /// <c>MaskedArray.compress</c>); mask carried alike.</summary>
        /// <param name="condition">Boolean array-like selector over the (flattened) elements.</param>
        /// <param name="axis">Axis to compress along, or null to flatten first.</param>
        /// <returns>The compressed masked array.</returns>
        public MaskedArray compress(object condition, int? axis = null) => np.ma.compress(condition, this, axis);

        /// <summary>Sorts along an axis with masked entries pushed to the end (or front) and re-masked there.</summary>
        /// <param name="axis">Sort axis (default last).</param>
        /// <param name="endwith">True (default) pushes masked to the END; false pushes them to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>The sorted masked array.</returns>
        public MaskedArray sort(int axis = -1, bool endwith = true, object fill_value = null) => np.ma.sort(this, axis, endwith, fill_value);

        /// <summary>Indices that would sort the array with masked entries at the end (or front).</summary>
        /// <param name="axis">Sort axis (default last).</param>
        /// <param name="endwith">True (default) sorts masked LAST; false sorts them FIRST.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices.</returns>
        public NDArray argsort(int axis = -1, bool endwith = true, object fill_value = null) => np.ma.argsort(this, axis, endwith, fill_value);

        /// <summary>Clamps each element into <c>[a_min, a_max]</c>, preserving the mask (NumPy's <c>clip</c>).</summary>
        /// <param name="a_min">Lower bound (array-like/scalar), or null for no lower clip.</param>
        /// <param name="a_max">Upper bound, or null for no upper clip.</param>
        /// <returns>The clamped masked array.</returns>
        public MaskedArray clip(object a_min, object a_max) => np.ma.clip(this, a_min, a_max);

        /// <summary>Rounds each element to <paramref name="decimals"/> places (half-to-even), preserving the mask.</summary>
        /// <param name="decimals">Decimal places.</param><returns>The masked, rounded array.</returns>
        public MaskedArray round(int decimals = 0) => np.ma.round(this, decimals);

        /// <summary>Complex conjugate (real data passes through); mask preserved (NumPy's <c>conj</c>).</summary>
        /// <returns>A masked array of conj(data).</returns>
        public MaskedArray conj() => np.ma.conjugate(this);

        /// <summary>Alias of <see cref="conj"/> (NumPy's <c>conjugate</c>).</summary>
        /// <returns>A masked array of conj(data).</returns>
        public MaskedArray conjugate() => np.ma.conjugate(this);

        /// <summary>Dot product with masked slots treated as 0 (NumPy's <c>MaskedArray.dot</c>).</summary>
        /// <param name="b">Right operand.</param><returns>The masked dot product.</returns>
        public MaskedArray dot(object b) => np.ma.dot(this, b);

        /// <summary>Extracts the <paramref name="offset"/>-th diagonal (mask alike); NumPy's <c>MaskedArray.diagonal</c>.</summary>
        /// <param name="offset">Diagonal offset (0 = main).</param><param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <returns>The masked diagonal.</returns>
        public MaskedArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) => np.ma.diagonal(this, offset, axis1, axis2);

        /// <summary>Sum of the diagonal, treating masked slots as 0 (NumPy's <c>MaskedArray.trace</c>) — a PLAIN
        /// array; the result is float64 unless <paramref name="dtype"/> is given (NumPy's <c>astype(None)</c> quirk).</summary>
        /// <param name="offset">Diagonal offset.</param><param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <param name="dtype">Result dtype; null ⇒ float64.</param>
        /// <returns>A plain <see cref="NDArray"/> trace.</returns>
        public NDArray trace(int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null) => np.ma.trace(this, offset, axis1, axis2, dtype);

        /// <summary>Indices of the non-zero UNMASKED elements (NumPy's <c>MaskedArray.nonzero</c>) — one plain
        /// int64 index array per dimension; masked slots are treated as zero and excluded.</summary>
        /// <returns>An array of int64 index arrays (the house tuple convention).</returns>
        public NDArray<long>[] nonzero() => np.ma.nonzero(this);

        /// <summary>The single element as a boxed scalar (NumPy's <c>MaskedArray.item</c>); a size-1 masked array
        /// whose element is masked returns the <see cref="MaskedArrayModule.masked"/> constant.</summary>
        /// <returns>The scalar value, or <see cref="MaskedArrayModule.masked"/> when the element is masked.</returns>
        /// <exception cref="InvalidOperationException">The array does not hold exactly one element.</exception>
        public object item()
        {
            if (_data.size != 1)
                throw new InvalidOperationException("can only convert an array of size 1 to a Python scalar");
            // A masked single element reads as the `masked` singleton, matching NumPy's `x.item()`.
            if (_mask is not null && np.any(_mask))
                return np.ma.masked;
            return _data.GetAtIndex(0);
        }

        /// <summary>Accepted for NumPy parity; NumSharp masks carry no hard/soft state, so this returns THIS
        /// array unchanged (NumPy's <c>harden_mask</c> would forbid unmasking).</summary>
        /// <returns>This masked array.</returns>
        public MaskedArray harden_mask() => this;

        /// <summary>Accepted for NumPy parity; a no-op in NumSharp (no hard/soft mask state).</summary>
        /// <returns>This masked array.</returns>
        public MaskedArray soften_mask() => this;

        /// <summary>Drops an all-False mask back to nomask (NumPy's <c>shrink_mask</c>); otherwise unchanged.</summary>
        /// <returns>A masked array with a redundant all-False mask removed.</returns>
        public MaskedArray shrink_mask() => np.ma.shrink_mask(this);

        // ── Bitwise / modulo operators — wire to the np.ma.* funcs (mask = OR of operands', division-domain
        //    for %). NumPy defines all of these on MaskedArray; without them `ma % b`, `ma & b`, `~ma` throw. ──
        /// <summary>Masked element-wise modulo (division-by-zero additionally masked).</summary>
        /// <param name="a">Numerator.</param><param name="b">Divisor.</param><returns>a%b, masked.</returns>
        public static MaskedArray operator %(MaskedArray a, MaskedArray b) => np.ma.remainder(a, b);
        /// <summary>Masked modulo with a scalar/array-like divisor.</summary>
        /// <param name="a">Numerator masked array.</param><param name="b">Divisor scalar/array-like.</param><returns>a%b, masked.</returns>
        public static MaskedArray operator %(MaskedArray a, object b) => np.ma.remainder(a, b);
        /// <summary>Masked modulo with a scalar/array-like numerator.</summary>
        /// <param name="a">Numerator scalar/array-like.</param><param name="b">Divisor masked array.</param><returns>a%b, masked.</returns>
        public static MaskedArray operator %(object a, MaskedArray b) => np.ma.remainder(a, b);
        /// <summary>Masked modulo with an NDArray divisor.</summary>
        /// <param name="a">Numerator.</param><param name="b">Divisor NDArray.</param><returns>a%b, masked.</returns>
        public static MaskedArray operator %(MaskedArray a, NDArray b) => np.ma.remainder(a, b);
        /// <summary>Masked modulo with an NDArray numerator.</summary>
        /// <param name="a">Numerator NDArray.</param><param name="b">Divisor masked array.</param><returns>a%b, masked.</returns>
        public static MaskedArray operator %(NDArray a, MaskedArray b) => np.ma.remainder(a, b);

        /// <summary>Masked bitwise AND (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&amp;b, masked.</returns>
        public static MaskedArray operator &(MaskedArray a, MaskedArray b) => np.ma.bitwise_and(a, b);
        /// <summary>Masked bitwise AND with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&amp;b, masked.</returns>
        public static MaskedArray operator &(MaskedArray a, object b) => np.ma.bitwise_and(a, b);
        /// <summary>Masked bitwise AND with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&amp;b, masked.</returns>
        public static MaskedArray operator &(object a, MaskedArray b) => np.ma.bitwise_and(a, b);
        /// <summary>Masked bitwise AND with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a&amp;b, masked.</returns>
        public static MaskedArray operator &(MaskedArray a, NDArray b) => np.ma.bitwise_and(a, b);
        /// <summary>Masked bitwise AND with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&amp;b, masked.</returns>
        public static MaskedArray operator &(NDArray a, MaskedArray b) => np.ma.bitwise_and(a, b);

        /// <summary>Masked bitwise OR (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a|b, masked.</returns>
        public static MaskedArray operator |(MaskedArray a, MaskedArray b) => np.ma.bitwise_or(a, b);
        /// <summary>Masked bitwise OR with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a|b, masked.</returns>
        public static MaskedArray operator |(MaskedArray a, object b) => np.ma.bitwise_or(a, b);
        /// <summary>Masked bitwise OR with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a|b, masked.</returns>
        public static MaskedArray operator |(object a, MaskedArray b) => np.ma.bitwise_or(a, b);
        /// <summary>Masked bitwise OR with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a|b, masked.</returns>
        public static MaskedArray operator |(MaskedArray a, NDArray b) => np.ma.bitwise_or(a, b);
        /// <summary>Masked bitwise OR with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a|b, masked.</returns>
        public static MaskedArray operator |(NDArray a, MaskedArray b) => np.ma.bitwise_or(a, b);

        /// <summary>Masked bitwise XOR (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a^b, masked.</returns>
        public static MaskedArray operator ^(MaskedArray a, MaskedArray b) => np.ma.bitwise_xor(a, b);
        /// <summary>Masked bitwise XOR with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a^b, masked.</returns>
        public static MaskedArray operator ^(MaskedArray a, object b) => np.ma.bitwise_xor(a, b);
        /// <summary>Masked bitwise XOR with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a^b, masked.</returns>
        public static MaskedArray operator ^(object a, MaskedArray b) => np.ma.bitwise_xor(a, b);
        /// <summary>Masked bitwise XOR with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a^b, masked.</returns>
        public static MaskedArray operator ^(MaskedArray a, NDArray b) => np.ma.bitwise_xor(a, b);
        /// <summary>Masked bitwise XOR with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a^b, masked.</returns>
        public static MaskedArray operator ^(NDArray a, MaskedArray b) => np.ma.bitwise_xor(a, b);

        /// <summary>Masked bitwise NOT / invert (mask unchanged). NumPy has no <c>np.ma.invert</c> module function,
        /// so this operator is the only spelling; it wires to the base <c>np.bitwise_not</c> on the data.</summary>
        /// <param name="a">Operand.</param><returns>~a, masked.</returns>
        public static MaskedArray operator ~(MaskedArray a) => np.ma.Invert(a);
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

        /// <summary>Alias of <see cref="masked"/> (NumPy's <c>masked_singleton</c> — the SAME object as
        /// <c>masked</c>, exported under both names).</summary>
        public MaskedConstant masked_singleton => masked;

        /// <summary>NumPy's <c>MaskType</c>: the dtype of a mask, i.e. boolean.</summary>
        public DType MaskType => np.@bool;

        /// <summary>NumPy's <c>bool_</c>: the boolean dtype (same as <see cref="MaskType"/>), exported by
        /// <c>numpy.ma</c> for building masks explicitly.</summary>
        public DType bool_ => np.@bool;

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
        public MaskedArray array(object data, object mask = null, object fill_value = null, bool copy = false, DType dtype = null)
        {
            var d = AsData(data);
            // dtype cast happens first (and subsumes the copy — astype makes a new buffer on a real cast, and
            // copy=true forces one even when the dtype already matches); mask stays boolean regardless of dtype.
            if (dtype != null)
                d = d.astype(dtype, copy);
            else if (copy)
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
        /// <param name="dtype">Optional dtype to cast the data to.</param>
        /// <returns>A new <see cref="MaskedArray"/>.</returns>
        public MaskedArray masked_array(object data, object mask = null, object fill_value = null, bool copy = false, DType dtype = null)
            => array(data, mask, fill_value, copy, dtype);

        /// <summary>
        ///     Converts an array-like to a <see cref="MaskedArray"/> WITHOUT copying (NumPy's <c>ma.asarray</c>):
        ///     an incoming masked array keeps its mask, a plain array/scalar becomes unmasked. This is the
        ///     explicit ndarray→MaskedArray entry point — NumSharp's implicit <c>NDArray→MaskedArray</c> does the
        ///     same for the unmasked case, but this also honors an existing mask and an optional dtype cast.
        /// </summary>
        /// <param name="a">A <see cref="MaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <param name="dtype">Optional dtype to cast the data to (no copy when it already matches).</param>
        /// <returns>A masked array over <paramref name="a"/>'s data (aliased when possible).</returns>
        public MaskedArray asarray(object a, DType dtype = null)
        {
            var d = AsData(a);
            if (dtype != null)
                d = d.astype(dtype, copy: false);
            return new MaskedArray(d, (a as MaskedArray)?._mask);
        }

        /// <summary>Alias of <see cref="asarray"/> (NumPy's <c>ma.asanyarray</c> — NumSharp has no MaskedArray
        /// subclasses to conserve, so it behaves identically).</summary>
        /// <param name="a">Array-like.</param><param name="dtype">Optional dtype cast.</param>
        /// <returns>A masked array over <paramref name="a"/>'s data.</returns>
        public MaskedArray asanyarray(object a, DType dtype = null) => asarray(a, dtype);

        /// <summary>
        ///     The default fill value NumPy assigns per dtype kind (<c>default_fill_value</c>): <c>1e20</c> for
        ///     float/complex, <c>999999</c> for integers, <c>True</c> for boolean, else <c>0</c>. Used by
        ///     <see cref="MaskedArray.filled(object)"/> when the caller supplies none.
        /// </summary>
        /// <param name="dtype">The data dtype to pick a fill for.</param>
        /// <returns>A boxed scalar of a type castable into <paramref name="dtype"/>.</returns>
        /// <remarks>An INSTANCE method (like <see cref="minimum_fill_value"/>/<see cref="maximum_fill_value"/>),
        /// so <c>np.ma.default_fill_value(dtype)</c> ports from NumPy verbatim.</remarks>
        public object default_fill_value(DType dtype)
        {
            switch (dtype.typecode)
            {
                case NPTypeCode.Boolean: return true;
                // Complex fills as (1e20+0j): NumPy's default_fill_value(complex) is a COMPLEX scalar, not a
                // bare double — so a call site reading it (e.g. via `fill_value`) sees the complex value NumPy
                // does, and the imaginary part is an honest 0 rather than absent.
                case NPTypeCode.Complex: return new Complex(1e20d, 0d);
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal: return 1e20d;
                default: return 999999L; // every integer kind (signed/unsigned/char)
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Predicates, shape queries, copy, and the deprecated/aliased names
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     True iff <paramref name="x"/> has AT LEAST ONE masked element (NumPy's <c>is_masked</c>) — the
        ///     VALUE predicate, DISTINCT from the <see cref="isMaskedArray"/> TYPE check. A plain array, a
        ///     <c>nomask</c> operand, an all-False mask, and every non-masked scalar all return false; only a
        ///     masked array with a set mask bit (including the <see cref="masked"/> singleton) returns true.
        /// </summary>
        /// <param name="x">Any object.</param>
        /// <returns>True when some element of <paramref name="x"/> is actually masked.</returns>
        public bool is_masked(object x)
        {
            // `is not null` (reference) then `np.any` — a `!= null` would run NDArray's ELEMENTWISE `!=`.
            var m = (x as MaskedArray)?._mask;
            return m is not null && np.any(m);
        }

        /// <summary>Number of dimensions of the data (NumPy's module-level <c>ma.ndim</c>).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The rank of the data (0 for a scalar).</returns>
        public int ndim(object a) => AsData(a).ndim;

        /// <summary>Data shape as a <c>long[]</c> (NumPy's module-level <c>ma.shape</c>; NumSharp dims are 64-bit).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The shape (empty for a scalar).</returns>
        public long[] shape(object a) => AsData(a).shape;

        /// <summary>Total element count (NumPy's module-level <c>ma.size</c>).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The element count (1 for a scalar).</returns>
        public long size(object a) => AsData(a).size;

        /// <summary>Deep copy of <paramref name="a"/> as a masked array (NumPy's module-level <c>ma.copy</c>):
        /// the data AND the mask are copied, so mutating either the source or the copy never affects the other.</summary>
        /// <param name="a">Array-like/masked-array to copy.</param>
        /// <returns>A masked array over a fresh copy of the data (mask copied when present).</returns>
        public MaskedArray copy(object a)
        {
            var m = (a as MaskedArray)?._mask;
            return new MaskedArray(AsData(a).copy(), m?.copy(), (a as MaskedArray)?._fill_value);
        }

        /// <summary>Maximum over unmasked elements — alias of <see cref="max"/> (NumPy's deprecated-but-exported
        /// <c>amax</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked maximum.</returns>
        public MaskedArray amax(object a, int? axis = null, object fill_value = null, bool keepdims = false) => max(a, axis, fill_value, keepdims);

        /// <summary>Minimum over unmasked elements — alias of <see cref="min"/> (NumPy's <c>amin</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked minimum.</returns>
        public MaskedArray amin(object a, int? axis = null, object fill_value = null, bool keepdims = false) => min(a, axis, fill_value, keepdims);

        /// <summary>Logical-AND reduction — alias of <see cref="all"/> (NumPy's DEPRECATED <c>alltrue</c>, still exported).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked all-reduction.</returns>
        public MaskedArray alltrue(object a, int? axis = null, bool keepdims = false) => all(a, axis, keepdims);

        /// <summary>Logical-OR reduction — alias of <see cref="any"/> (NumPy's DEPRECATED <c>sometrue</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked any-reduction.</returns>
        public MaskedArray sometrue(object a, int? axis = null, bool keepdims = false) => any(a, axis, keepdims);

        /// <summary>Round to <paramref name="decimals"/> places — alias of <see cref="round"/> (NumPy's <c>round_</c>).</summary>
        /// <param name="a">Operand.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array.</returns>
        public MaskedArray round_(object a, int decimals = 0) => round(a, decimals);

        /// <summary>Stack along the first axis — alias of <see cref="vstack"/> (NumPy's <c>row_stack</c>).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public MaskedArray row_stack(params object[] tup) => vstack(tup);

        /// <summary>Inner product — alias of <see cref="inner"/> (NumPy's <c>innerproduct</c>).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The masked inner product.</returns>
        public MaskedArray innerproduct(object a, object b) => inner(a, b);

        /// <summary>Outer product — alias of <see cref="outer"/> (NumPy's <c>outerproduct</c>).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The masked outer product.</returns>
        public MaskedArray outerproduct(object a, object b) => outer(a, b);

        /// <summary>Alias of <see cref="isMaskedArray"/> (NumPy's <c>isMA</c>).</summary>
        /// <param name="x">Any object.</param><returns>True for a masked array.</returns>
        public bool isMA(object x) => isMaskedArray(x);

        /// <summary>Alias of <see cref="isMaskedArray"/> (NumPy's <c>isarray</c>).</summary>
        /// <param name="x">Any object.</param><returns>True for a masked array.</returns>
        public bool isarray(object x) => isMaskedArray(x);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Mask construction / combination (NumPy's make_mask family + mask_or)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Coerces an array-like to a boolean mask (NumPy's <c>make_mask</c>): any non-zero element becomes
        ///     True. With <paramref name="shrink"/> (default) an all-False result collapses to <see cref="nomask"/>
        ///     — the standard "no mask needed" reduction — otherwise a full all-False array is returned.
        /// </summary>
        /// <param name="m">Array-like to coerce (numeric/boolean; non-zero ⇒ masked).</param>
        /// <param name="shrink">Collapse an all-False mask to <see cref="nomask"/> (default true).</param>
        /// <returns>A boolean mask array, or <see cref="nomask"/> when it shrinks to empty.</returns>
        public NDArray make_mask(object m, bool shrink = true)
        {
            // NumPy coerces via `!= 0` (so floats/ints become bool by non-zero-ness), not a plain astype(bool).
            var b = AsData(m);
            var mask = b.typecode == NPTypeCode.Boolean ? b.copy() : np.not_equal(b, NDArray.Scalar(0));
            return (shrink && !np.any(mask)) ? nomask : mask;
        }

        /// <summary>Builds an all-False boolean mask of the given shape (NumPy's <c>make_mask_none</c>) — the
        /// "nothing masked yet, full array" starting mask.</summary>
        /// <param name="newshape">The mask shape.</param>
        /// <returns>An all-False boolean <see cref="NDArray"/>.</returns>
        public NDArray make_mask_none(Shape newshape) => np.zeros(newshape, np.@bool);

        /// <summary>
        ///     Combines two masks with logical-OR, honoring the <see cref="nomask"/> fast path (NumPy's
        ///     <c>mask_or</c>): a null/nomask operand contributes nothing, and — with <paramref name="shrink"/>
        ///     (default) — an all-False OR collapses back to <see cref="nomask"/>.
        /// </summary>
        /// <param name="m1">First mask, or null/nomask.</param>
        /// <param name="m2">Second mask, or null/nomask.</param>
        /// <param name="shrink">Collapse an all-False result to <see cref="nomask"/> (default true).</param>
        /// <returns>The OR of the present masks, or <see cref="nomask"/> when neither contributes (or it shrinks).</returns>
        public NDArray mask_or(object m1, object m2, bool shrink = true)
        {
            // Treat a null OR the shared nomask sentinel (a 0-D False) as "no contribution".
            var a = MaskOperand(m1);
            var b = MaskOperand(m2);
            var r = Or(a, b);
            if (r is null)
                return nomask;
            if (shrink && !np.any(r))
                return nomask;
            return r;
        }

        /// <summary>Normalizes a <c>mask_or</c> operand to a real mask or null: unwraps a masked array's mask,
        /// treats the <see cref="nomask"/> sentinel and any all-False 0-D as "no contribution", else coerces to bool.</summary>
        private NDArray MaskOperand(object m)
        {
            switch (m)
            {
                case null: return null;
                case MaskedArray ma_: return ma_._mask;
                case NDArray nd:
                    // The 0-D nomask sentinel (or any 0-D False) means "no mask".
                    if (ReferenceEquals(nd, nomask)) return null;
                    if (nd.ndim == 0 && !np.any(nd)) return null;
                    return nd.typecode == NPTypeCode.Boolean ? nd : np.not_equal(nd, NDArray.Scalar(0));
                default:
                    var b = AsData(m);
                    return b.typecode == NPTypeCode.Boolean ? b : np.not_equal(b, NDArray.Scalar(0));
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Fill-value helpers (common_fill_value / set_fill_value / fix_invalid)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     The single fill value shared by both inputs, or null when they differ (NumPy's
        ///     <c>common_fill_value</c>). Each operand's fill is its stored override, else its dtype default.
        /// </summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The common fill value (boxed), or null when the two differ.</returns>
        public object common_fill_value(object a, object b)
        {
            var fa = FillOf(a);
            var fb = FillOf(b);
            return Equals(fa, fb) ? fa : null;
        }

        /// <summary>The effective fill value of an operand — its stored override, else its dtype default.</summary>
        private object FillOf(object a)
            => (a as MaskedArray)?._fill_value ?? default_fill_value(AsData(a).dtype);

        /// <summary>
        ///     Sets the fill value of <paramref name="a"/> in place (NumPy's module-level <c>set_fill_value</c>).
        ///     Has effect ONLY on a <see cref="MaskedArray"/> (a plain array has no fill slot); a non-masked
        ///     operand is silently ignored, matching NumPy.
        /// </summary>
        /// <param name="a">The masked array to mutate.</param>
        /// <param name="fill_value">The new fill value (null restores the dtype default).</param>
        public void set_fill_value(object a, object fill_value)
        {
            if (a is MaskedArray m)
                m._fill_value = fill_value;
        }

        /// <summary>
        ///     Masks the non-finite (NaN/±inf) elements of <paramref name="a"/> AND writes the fill value into
        ///     their data slots (NumPy's <c>fix_invalid</c>) — like <see cref="masked_invalid"/> but the hidden
        ///     data becomes the fill rather than the original non-finite value. The result array's
        ///     <c>fill_value</c> stays the dtype default; <paramref name="fill_value"/> only chooses what is
        ///     WRITTEN into the masked slots (matching NumPy, whose param sets the data, not the attribute).
        /// </summary>
        /// <param name="a">Data array-like (an existing mask is preserved and OR'd with the invalid mask).</param>
        /// <param name="mask">Optional extra boolean mask to OR in (NumPy's <c>mask=</c>); null = none.</param>
        /// <param name="copy">Copy the data (default true) vs alias it.</param>
        /// <param name="fill_value">Value written at the newly-masked (invalid) slots; null uses <paramref name="a"/>'s
        /// own fill (dtype default).</param>
        /// <returns>A masked array with non-finite slots masked and their data replaced by the fill.</returns>
        public MaskedArray fix_invalid(object a, object mask = null, bool copy = true, object fill_value = null)
        {
            var d = AsData(a);
            // The invalid mask is ~isfinite for a float/complex dtype, else nothing (integers are all finite).
            var invalid = NonFiniteMask(d) ?? np.zeros(d.Shape, np.@bool);
            var existing = (a as MaskedArray)?._mask;
            var extra = mask is null ? null : AsData(mask).astype(np.@bool);
            // Full mask = existing | extra | invalid (nomask-aware).
            var full = Or(Or(existing, extra), invalid);
            var data = copy ? d.copy() : d;
            var fv = fill_value ?? (a as MaskedArray)?._fill_value ?? default_fill_value(d.dtype);
            // Write the fill into every INVALID slot (not the whole mask — NumPy only overwrites the newly
            // caught non-finite data, leaving pre-existing masked-but-finite data alone).
            if (np.any(invalid))
                np.copyto(data, NDArray.Scalar(fv), casting: "unsafe", where: invalid);
            return new MaskedArray(data, (full is not null && np.any(full)) ? full : null);
        }

        /// <summary>The <c>~</c> operator's backing (NumPy has no <c>ma.invert</c> module function): bitwise-NOT
        /// of the data, mask carried through unchanged. Internal — reached via <c>~maskedArray</c>.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>A masked array of ~data.</returns>
        internal MaskedArray Invert(object a) => Unary(x => np.bitwise_not(x), a);

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

        /// <summary>Bit shift left (<c>a &lt;&lt; b</c>); masked where either operand was (NumPy's <c>left_shift</c>).</summary>
        /// <param name="a">Value to shift.</param><param name="b">Shift amount.</param>
        /// <returns>A masked array of a&lt;&lt;b.</returns>
        public MaskedArray left_shift(object a, object b) => Binary((x, y) => np.left_shift(x, y), a, b);

        /// <summary>Bit shift right (<c>a &gt;&gt; b</c>); masked where either operand was (NumPy's <c>right_shift</c>).</summary>
        /// <param name="a">Value to shift.</param><param name="b">Shift amount.</param>
        /// <returns>A masked array of a&gt;&gt;b.</returns>
        public MaskedArray right_shift(object a, object b) => Binary((x, y) => np.right_shift(x, y), a, b);

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

        // ─────────────────────────────────────────────────────────────────────────────
        //  Reductions — masked entries are excluded by filling with the op's identity, and
        //  a result cell is masked iff EVERY element reduced into it was masked (NumPy's
        //  `_check_mask_axis` = mask.all(axis)). Scans keep the original per-position mask.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     The value NumPy fills masked elements with for a MINIMUM/argmin (the dtype's LARGEST value, so
        ///     a masked slot can never win a min): +inf for float/complex, MaxValue for integers/decimal/char,
        ///     True for bool. NumPy's <c>minimum_fill_value</c>.
        /// </summary>
        /// <param name="a">Array-like whose dtype selects the value.</param>
        /// <returns>A boxed scalar castable into the dtype.</returns>
        public object minimum_fill_value(object a) => MinFill(AsData(a).typecode);

        /// <summary>
        ///     The value NumPy fills masked elements with for a MAXIMUM/argmax (the dtype's SMALLEST value):
        ///     -inf for float/complex, MinValue for integers/decimal, False for bool. NumPy's
        ///     <c>maximum_fill_value</c>.
        /// </summary>
        /// <param name="a">Array-like whose dtype selects the value.</param>
        /// <returns>A boxed scalar castable into the dtype.</returns>
        public object maximum_fill_value(object a) => MaxFill(AsData(a).typecode);

        /// <summary>Per-dtype "largest" fill (for min/argmin) — see <see cref="minimum_fill_value"/>.</summary>
        private static object MinFill(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => (object)true,
            NPTypeCode.Byte => byte.MaxValue,
            NPTypeCode.SByte => sbyte.MaxValue,
            NPTypeCode.Int16 => short.MaxValue,
            NPTypeCode.UInt16 => ushort.MaxValue,
            NPTypeCode.Int32 => int.MaxValue,
            NPTypeCode.UInt32 => uint.MaxValue,
            NPTypeCode.Int64 => long.MaxValue,
            NPTypeCode.UInt64 => ulong.MaxValue,
            NPTypeCode.Char => char.MaxValue,
            NPTypeCode.Decimal => decimal.MaxValue,
            // Complex fills as (inf+infj): both components are +inf so a masked slot sorts strictly last in the
            // real-then-imag lexicographic order NumPy uses — a bare (inf+0j) would lose to a real (inf+kj) value
            // on the imaginary tie-break and be wrongly selected as the minimum.
            NPTypeCode.Complex => new Complex(double.PositiveInfinity, double.PositiveInfinity),
            _ => double.PositiveInfinity, // Half/Single/Double
        };

        /// <summary>Per-dtype "smallest" fill (for max/argmax) — see <see cref="maximum_fill_value"/>.</summary>
        private static object MaxFill(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => (object)false,
            NPTypeCode.Byte => byte.MinValue,
            NPTypeCode.SByte => sbyte.MinValue,
            NPTypeCode.Int16 => short.MinValue,
            NPTypeCode.UInt16 => ushort.MinValue,
            NPTypeCode.Int32 => int.MinValue,
            NPTypeCode.UInt32 => uint.MinValue,
            NPTypeCode.Int64 => long.MinValue,
            NPTypeCode.UInt64 => ulong.MinValue,
            NPTypeCode.Char => (char)0,
            NPTypeCode.Decimal => decimal.MinValue,
            // Complex fills as (-inf-infj): the mirror of MinFill, so a masked slot sorts strictly first for a
            // maximum and is never selected.
            NPTypeCode.Complex => new Complex(double.NegativeInfinity, double.NegativeInfinity),
            _ => double.NegativeInfinity, // Half/Single/Double
        };

        /// <summary>The "all reduced elements were masked" predicate = <c>mask.all(axis)</c> (NumPy's
        /// <c>_check_mask_axis</c>): a reduced cell is masked exactly there.</summary>
        private static NDArray AllAlongAxis(NDArray mask, int? axis, bool keepdims)
            => np.all(mask, axis, null, keepdims);

        /// <summary>Count of UNMASKED elements along the axis, as int64 (NumPy's <c>count</c> uses
        /// <c>(~mask).sum(axis)</c>; a null mask counts every element).</summary>
        private static NDArray CountUnmasked(NDArray mask, Shape shape, int? axis, bool keepdims)
        {
            var unmasked = mask is null ? np.ones(shape, np.@bool) : np.logical_not(mask);
            return np.sum(unmasked, axis, keepdims, np.int64);
        }

        /// <summary>Boxes a computed reduction result + its (possibly null) mask, collapsing a fully-masked
        /// 0-D result to the <see cref="masked"/> singleton (NumPy's scalar-reduction behavior).</summary>
        private MaskedArray Finalize(NDArray result, NDArray newmask)
        {
            if (result.ndim == 0)
                return (newmask is not null && np.any(newmask)) ? masked : new MaskedArray(result, null);
            return new MaskedArray(result, newmask);
        }

        /// <summary>Shared identity-fill reduction: fill masked with <paramref name="identity"/>, reduce via
        /// <paramref name="fn"/>, and mask result cells whose whole reduced slice was masked.</summary>
        /// <param name="a">Operand.</param>
        /// <param name="identity">The op identity written at masked slots (0 sum, 1 prod, ±inf min/max, …).</param>
        /// <param name="fn">The underlying <c>np.*</c> reduction (data, axis, keepdims) → result.</param>
        /// <param name="axis">Reduction axis, or null to reduce the flattened array.</param>
        /// <param name="keepdims">Keep reduced axes as size-1 (NumPy keepdims).</param>
        /// <returns>The masked reduction result (or <see cref="masked"/> for a fully-masked scalar).</returns>
        private MaskedArray ReduceIdentity(object a, object identity, Func<NDArray, int?, bool, NDArray> fn, int? axis, bool keepdims)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = mask is null ? AsData(a) : ((MaskedArray)a).filled(identity);
            var result = fn(d, axis, keepdims);
            if (mask is null)
                return new MaskedArray(result, null);
            return Finalize(result, AllAlongAxis(mask, axis, keepdims));
        }

        /// <summary>Sum over the axis; masked slots contribute 0. A cell is masked iff its whole slice was.
        /// dtype widens like <c>np.sum</c> (NEP50: int→int64).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten).</param>
        /// <param name="dtype">Accumulator dtype (NumPy sum dtype=).</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked sum.</returns>
        public MaskedArray sum(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => ReduceIdentity(a, 0, (x, ax, kd) => np.sum(x, ax, kd, dtype), axis, keepdims);

        /// <summary>Product over the axis; masked slots contribute 1.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked product.</returns>
        public MaskedArray prod(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => ReduceIdentity(a, 1, (x, ax, kd) => np.prod(x, ax, dtype, kd), axis, keepdims);

        /// <summary>Alias of <see cref="prod"/> (NumPy's <c>product</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked product.</returns>
        public MaskedArray product(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => prod(a, axis, dtype, keepdims);

        /// <summary>Minimum over the axis; masked slots are filled with the dtype's largest value so they are
        /// never selected. A cell is masked iff its whole slice was masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Override the masked fill (NumPy fill_value=); null uses <see cref="minimum_fill_value"/>.</param>
        /// <param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked minimum.</returns>
        public MaskedArray min(object a, int? axis = null, object fill_value = null, bool keepdims = false)
            => ReduceIdentity(a, fill_value ?? minimum_fill_value(a), (x, ax, kd) => np.min(x, ax, kd), axis, keepdims);

        /// <summary>Maximum over the axis; masked slots are filled with the dtype's smallest value.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="maximum_fill_value"/>.</param>
        /// <param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked maximum.</returns>
        public MaskedArray max(object a, int? axis = null, object fill_value = null, bool keepdims = false)
            => ReduceIdentity(a, fill_value ?? maximum_fill_value(a), (x, ax, kd) => np.max(x, ax, kd), axis, keepdims);

        /// <summary>Peak-to-peak (max − min) over the axis, over unmasked elements.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Masked fill passed to both min and max.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked range.</returns>
        public MaskedArray ptp(object a, int? axis = null, object fill_value = null, bool keepdims = false)
            => subtract(max(a, axis, fill_value, keepdims), min(a, axis, fill_value, keepdims));

        /// <summary>Number of UNMASKED elements along the axis (NumPy's <c>count</c>) — a plain int64 array,
        /// never masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>An int64 <see cref="NDArray"/> (0-D for a flat count).</returns>
        public NDArray count(object a, int? axis = null, bool keepdims = false)
            => CountUnmasked((a as MaskedArray)?._mask, AsData(a).Shape, axis, keepdims);

        /// <summary>Mean over unmasked elements = sum/count (masked slots excluded from BOTH). A cell whose
        /// whole slice was masked is masked. Result is float (int/bool promote to float64).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="dtype">Sum accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked mean.</returns>
        public MaskedArray mean(object a, int? axis = null, DType dtype = null, bool keepdims = false)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = AsData(a);
            var tc = d.typecode;
            // NumPy mean dtype (default): int/bool → float64; float16 is COMPUTED in float32 then cast BACK to
            // float16; float32/float64/complex128 are preserved. A masked complex mean must stay complex.
            DType compute = tc == NPTypeCode.Complex ? np.complex128 : tc == NPTypeCode.Half ? np.float32 : np.float64;
            // NumPy's `dsum * 1.` promotes float32→float64, so ONLY float16 (cast back) and complex128 keep a
            // non-f8 result; float32/float64/int/bool all yield float64.
            DType outdt = dtype ?? (tc switch
            {
                NPTypeCode.Half => np.float16,
                NPTypeCode.Complex => np.complex128,
                _ => np.float64, // int/bool/float32/float64
            });
            var filled0 = mask is null ? d : ((MaskedArray)a).filled(0);
            var dsum = np.sum(filled0, axis, keepdims, dtype ?? compute).astype(compute);
            var cnt = CountUnmasked(mask, d.Shape, axis, keepdims).astype(compute);
            var result = np.divide(dsum, cnt).astype(outdt);     // count==0 slots → nan, masked below
            if (mask is null)
                return new MaskedArray(result, null);
            var newmask = AllAlongAxis(mask, axis, keepdims);
            // NumPy computes the mean through MASKED arithmetic, so a fully-masked slice's .data is 0 (the
            // sum's identity), NOT the raw 0/0=NaN. Match it at the masked (count==0) positions.
            if (result.ndim != 0)
                result = np.where(newmask, NDArray.Scalar(0.0), result).astype(outdt);
            return Finalize(result, newmask);
        }

        /// <summary>Variance over unmasked elements (masked contribute nothing); a cell is masked where its
        /// whole slice was masked OR the unmasked count minus <paramref name="ddof"/> is ≤ 0.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param>
        /// <param name="ddof">Delta degrees of freedom (divisor is count − ddof).</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked variance (real-valued, even for complex input).</returns>
        public MaskedArray var(object a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = AsData(a);
            bool cplx = d.typecode == NPTypeCode.Complex;
            var computeType = cplx ? np.complex128 : np.float64;

            if (mask is null)
            {
                var rn = axis is null ? np.var(d, keepdims, ddof, dtype) : np.var(d, axis.Value, keepdims, ddof, dtype);
                return new MaskedArray(rn, null);
            }

            var filled0 = ((MaskedArray)a).filled(0);
            // keepdims mean for centering, then squared deviations with masked slots contributing 0.
            var cntK = CountUnmasked(mask, d.Shape, axis, true).astype(computeType);
            var meanK = np.divide(np.sum(filled0, axis, true, dtype).astype(computeType), cntK);
            var dev = np.subtract(d.astype(computeType), meanK);
            var devsq = cplx ? np.multiply(np.abs(dev), np.abs(dev)) : np.multiply(dev, dev);
            devsq = np.where(mask, NDArray.Scalar(0.0), devsq.astype(np.float64)); // masked → 0 contribution
            var cnt = np.subtract(CountUnmasked(mask, d.Shape, axis, keepdims).astype(np.float64), NDArray.Scalar((double)ddof));
            var dvar = np.divide(np.sum(devsq, axis, keepdims, np.float64), cnt);
            var newmask = np.logical_or(AllAlongAxis(mask, axis, keepdims), np.less_equal(cnt, NDArray.Scalar(0.0)));
            // As with mean, masked (all-masked / cnt<=0) slices carry .data 0, not the raw NaN.
            if (dvar.ndim != 0)
                dvar = np.where(newmask, NDArray.Scalar(0.0), dvar);
            return Finalize(dvar, newmask);
        }

        /// <summary>Standard deviation = sqrt(<see cref="var"/>), preserving the variance's mask.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param>
        /// <param name="ddof">Delta degrees of freedom.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked standard deviation.</returns>
        public MaskedArray std(object a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false)
        {
            var v = var(a, axis, dtype, ddof, keepdims);
            return ReferenceEquals(v, masked) ? masked : sqrt(v);
        }

        /// <summary>Anomalies (deviations from the mean along the axis) = <c>a − mean(a, axis, keepdims)</c>,
        /// mask preserved from the input.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">Mean accumulator dtype.</param>
        /// <returns>The masked anomalies, same shape as the input.</returns>
        public MaskedArray anom(object a, int? axis = null, DType dtype = null)
            => subtract(a, mean(a, axis, dtype, keepdims: true));

        /// <summary>Alias of <see cref="anom"/> (NumPy's <c>anomalies</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">Mean accumulator dtype.</param>
        /// <returns>The masked anomalies.</returns>
        public MaskedArray anomalies(object a, int? axis = null, DType dtype = null) => anom(a, axis, dtype);

        /// <summary>Cumulative sum along the axis; masked slots contribute 0 to the running total but their
        /// POSITIONS stay masked in the result (NumPy semantics).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten, C-order).</param><param name="dtype">Accumulator dtype.</param>
        /// <returns>The masked cumulative sum.</returns>
        public MaskedArray cumsum(object a, int? axis = null, DType dtype = null)
            => Scan(a, 0, (x, ax) => np.cumsum(x, ax, dtype), axis);

        /// <summary>Cumulative product along the axis; masked slots contribute 1 but stay masked in the result.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten, C-order).</param><param name="dtype">Accumulator dtype.</param>
        /// <returns>The masked cumulative product.</returns>
        public MaskedArray cumprod(object a, int? axis = null, DType dtype = null)
            => Scan(a, 1, (x, ax) => np.cumprod(x, ax, dtype), axis);

        /// <summary>Shared scan (cumsum/cumprod): fill masked with the identity, scan, and carry the original
        /// per-position mask (raveled to match when axis is null).</summary>
        private MaskedArray Scan(object a, object identity, Func<NDArray, int?, NDArray> fn, int? axis)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = mask is null ? AsData(a) : ((MaskedArray)a).filled(identity);
            var result = fn(d, axis);
            if (mask is null)
                return new MaskedArray(result, null);
            // axis=null flattens the result in C-order — ravel the mask to match.
            var m = axis is null ? np.ravel(mask) : mask;
            return new MaskedArray(result, m);
        }

        /// <summary>Indices of the minimum along the axis, treating masked slots as the dtype's largest value
        /// (NumPy's <c>argmin</c>) — a plain int64 array, never masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flattened index).</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="minimum_fill_value"/>.</param>
        /// <returns>An int64 <see cref="NDArray"/> of indices.</returns>
        public NDArray argmin(object a, int? axis = null, object fill_value = null)
        {
            var d = ArgFilled(a, fill_value ?? minimum_fill_value(a));
            return axis is null ? NDArray.Scalar(np.argmin(d)) : np.argmin(d, axis.Value);
        }

        /// <summary>Indices of the maximum along the axis, treating masked slots as the dtype's smallest value.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flattened index).</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="maximum_fill_value"/>.</param>
        /// <returns>An int64 <see cref="NDArray"/> of indices.</returns>
        public NDArray argmax(object a, int? axis = null, object fill_value = null)
        {
            var d = ArgFilled(a, fill_value ?? maximum_fill_value(a));
            return axis is null ? NDArray.Scalar(np.argmax(d)) : np.argmax(d, axis.Value);
        }

        /// <summary>Data with masked slots filled for an arg-reduction (null mask ⇒ raw data).</summary>
        private static NDArray ArgFilled(object a, object fill)
            => (a as MaskedArray)?._mask is null ? AsData(a) : ((MaskedArray)a).filled(fill);

        /// <summary>Logical-AND reduction: masked slots are treated as True (excluded); a cell whose whole
        /// slice was masked is itself masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked all-reduction (bool data).</returns>
        public MaskedArray all(object a, int? axis = null, bool keepdims = false)
            => ReduceIdentity(a, true, (x, ax, kd) => ax is null ? NDArray.Scalar(np.all(x)) : np.all(x, ax, null, kd), axis, keepdims);

        /// <summary>Logical-OR reduction: masked slots are treated as False (excluded).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked any-reduction (bool data).</returns>
        public MaskedArray any(object a, int? axis = null, bool keepdims = false)
            => ReduceIdentity(a, false, (x, ax, kd) => ax is null ? NDArray.Scalar(np.any(x)) : np.any(x, ax, null, kd), axis, keepdims);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Creation — fresh, UNMASKED arrays (NumPy's `_convert2ma`: a masked wrapper over
        //  the corresponding np.* creation routine, with no element masked).
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Masked array of zeros (no element masked). float64 by default.</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of zeros.</returns>
        public MaskedArray zeros(Shape shape, DType dtype = null) => new MaskedArray(np.zeros(shape, dtype ?? np.float64), null);

        /// <summary>Masked array of ones (no element masked).</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of ones.</returns>
        public MaskedArray ones(Shape shape, DType dtype = null) => new MaskedArray(np.ones(shape, dtype ?? np.float64), null);

        /// <summary>Masked array of uninitialized values (no element masked).</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of arbitrary contents.</returns>
        public MaskedArray empty(Shape shape, DType dtype = null) => new MaskedArray(np.empty(shape, dtype ?? np.float64), null);

        /// <summary>Masked zeros shaped like <paramref name="a"/> (no element masked; the input mask is NOT carried).</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype (null ⇒ prototype's).</param>
        /// <returns>An unmasked masked array of zeros.</returns>
        public MaskedArray zeros_like(object a, DType dtype = null) => new MaskedArray(np.zeros_like(AsData(a), dtype), null);

        /// <summary>Masked ones shaped like <paramref name="a"/> (no element masked).</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype.</param>
        /// <returns>An unmasked masked array of ones.</returns>
        public MaskedArray ones_like(object a, DType dtype = null) => new MaskedArray(np.ones_like(AsData(a), dtype), null);

        /// <summary>Masked uninitialized array shaped like <paramref name="a"/> (no element masked).</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype.</param>
        /// <returns>An unmasked masked array.</returns>
        public MaskedArray empty_like(object a, DType dtype = null) => new MaskedArray(np.empty_like(AsData(a), dtype), null);

        /// <summary>Masked evenly-spaced values in [0, stop) (no element masked).</summary>
        /// <param name="stop">Exclusive upper bound.</param><returns>An unmasked masked range.</returns>
        public MaskedArray arange(int stop) => new MaskedArray(np.arange(stop), null);

        /// <summary>Masked evenly-spaced values in [start, stop) stepping by <paramref name="step"/>.</summary>
        /// <param name="start">Inclusive start.</param><param name="stop">Exclusive stop.</param><param name="step">Step.</param>
        /// <returns>An unmasked masked range.</returns>
        public MaskedArray arange(int start, int stop, int step = 1) => new MaskedArray(np.arange(start, stop, step), null);

        /// <summary>Masked evenly-spaced floating values in [start, stop) stepping by <paramref name="step"/>.</summary>
        /// <param name="start">Inclusive start.</param><param name="stop">Exclusive stop.</param><param name="step">Step.</param>
        /// <returns>An unmasked masked range.</returns>
        public MaskedArray arange(double start, double stop, double step = 1) => new MaskedArray(np.arange(start, stop, step), null);

        /// <summary>Masked identity matrix (no element masked).</summary>
        /// <param name="n">Order.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked identity.</returns>
        public MaskedArray identity(int n, DType dtype = null) => new MaskedArray(np.identity(n, dtype), null);

        /// <summary>Masked grid-index array for <paramref name="dimensions"/> (no element masked).</summary>
        /// <param name="dimensions">Grid shape.</param><param name="dtype">Index dtype.</param>
        /// <returns>An unmasked masked index grid.</returns>
        public MaskedArray indices(int[] dimensions, DType dtype = null) => new MaskedArray(np.indices(dimensions, dtype), null);

        // ─────────────────────────────────────────────────────────────────────────────
        //  masked_* constructors — build a mask from a condition/value and OR it onto any
        //  existing mask (NumPy's masked_where family). copy=true (default) copies the data.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Masks <paramref name="a"/> wherever <paramref name="condition"/> is True, OR'ing onto any
        /// existing mask (NumPy's <c>masked_where</c> — the primitive the rest of this family builds on).</summary>
        /// <param name="condition">Boolean array-like; True ⇒ mask that element.</param>
        /// <param name="a">Data array-like.</param>
        /// <param name="copy">Copy the data (default true) vs alias it.</param>
        /// <returns>A masked array of <paramref name="a"/>'s data with the new mask.</returns>
        public MaskedArray masked_where(object condition, object a, bool copy = true)
        {
            var cm = AsData(condition).astype(np.@bool);
            var d = AsData(a);
            var existing = (a as MaskedArray)?._mask;
            var m = existing is null ? cm : (existing | cm);
            return new MaskedArray(copy ? d.copy() : d, m);
        }

        /// <summary>Masks where <c>a == value</c> (NumPy's <c>masked_equal</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Value to mask.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_equal(object a, object value, bool copy = true)
            => masked_where(np.equal(AsData(a), NDArray.Scalar(value)), a, copy);

        /// <summary>Masks where <c>a != value</c>.</summary>
        /// <param name="a">Data.</param><param name="value">Value kept unmasked.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_not_equal(object a, object value, bool copy = true)
            => masked_where(np.not_equal(AsData(a), NDArray.Scalar(value)), a, copy);

        /// <summary>Masks where <c>a &gt; value</c>.</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_greater(object a, object value, bool copy = true)
            => masked_where(np.greater(AsData(a), NDArray.Scalar(value)), a, copy);

        /// <summary>Masks where <c>a &gt;= value</c>.</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_greater_equal(object a, object value, bool copy = true)
            => masked_where(np.greater_equal(AsData(a), NDArray.Scalar(value)), a, copy);

        /// <summary>Masks where <c>a &lt; value</c>. Computed as <c>value &gt; a</c> (scalar left) to route around
        /// the <c>np.less</c> 0-D-scalar-RHS quirk while staying NaN-exact (both give False for NaN).</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_less(object a, object value, bool copy = true)
            => masked_where(np.greater(NDArray.Scalar(value), AsData(a)), a, copy);

        /// <summary>Masks where <c>a &lt;= value</c>.</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_less_equal(object a, object value, bool copy = true)
            => masked_where(np.less_equal(AsData(a), NDArray.Scalar(value)), a, copy);

        /// <summary>Masks the CLOSED interval where <c>v1 &lt;= a &lt;= v2</c> (bounds swapped if v1&gt;v2), NumPy's
        /// <c>masked_inside</c>.</summary>
        /// <param name="a">Data.</param><param name="v1">One bound.</param><param name="v2">Other bound.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_inside(object a, double v1, double v2, bool copy = true)
        {
            var lo = Math.Min(v1, v2); var hi = Math.Max(v1, v2);
            var d = AsData(a);
            var cond = np.greater_equal(d, NDArray.Scalar(lo)) & np.less_equal(d, NDArray.Scalar(hi));
            return masked_where(cond, a, copy);
        }

        /// <summary>Masks OUTSIDE the closed interval where <c>a &lt; v1 || a &gt; v2</c> (bounds swapped if v1&gt;v2),
        /// NumPy's <c>masked_outside</c>.</summary>
        /// <param name="a">Data.</param><param name="v1">One bound.</param><param name="v2">Other bound.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_outside(object a, double v1, double v2, bool copy = true)
        {
            var lo = Math.Min(v1, v2); var hi = Math.Max(v1, v2);
            var d = AsData(a);
            var cond = np.greater(NDArray.Scalar(lo), d) | np.greater(d, NDArray.Scalar(hi));
            return masked_where(cond, a, copy);
        }

        /// <summary>Masks non-finite elements (NaN/±inf) — NumPy's <c>masked_invalid</c>. For non-float dtypes
        /// nothing is masked (every value is finite).</summary>
        /// <param name="a">Data.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_invalid(object a, bool copy = true)
        {
            var d = AsData(a);
            var cond = NonFiniteMask(d) ?? np.zeros(d.Shape, np.@bool);
            return masked_where(cond, a, copy);
        }

        /// <summary>Masks elements approximately equal to <paramref name="value"/> (NumPy's <c>masked_values</c>,
        /// <c>isclose</c> with the given tolerances); the masked array's default fill becomes <paramref name="value"/>.</summary>
        /// <param name="a">Data.</param><param name="value">Value to mask around.</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array (fill_value = <paramref name="value"/>).</returns>
        public MaskedArray masked_values(object a, double value, double rtol = 1e-5, double atol = 1e-8, bool copy = true)
        {
            // NumPy fills masked positions with `value` FIRST (xnew = filled(x, value)), THEN masks by
            // approximate equality — so a pre-masked slot's data becomes `value` and is re-masked.
            var xnew = filled(a, value);
            bool isFloat = xnew.typecode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;
            var mask = isFloat
                ? np.isclose(xnew, NDArray.Scalar(value), rtol, atol)
                : np.equal(xnew, NDArray.Scalar(value));
            // shrink an all-False mask to nomask (NumPy's shrink_mask default).
            return new MaskedArray(copy ? xnew.copy() : xnew, np.any(mask) ? mask : null, value);
        }

        /// <summary>Masks elements exactly equal to <paramref name="value"/> (NumPy's <c>masked_object</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Value to mask.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        public MaskedArray masked_object(object a, object value, bool copy = true) => masked_equal(a, value, copy);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Extrema (element-wise), power, where, round
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise maximum of two operands (mask = OR); with <paramref name="b"/> omitted it is the
        /// maximum REDUCTION of <paramref name="a"/> — NumPy's dual-purpose <c>maximum</c>.</summary>
        /// <param name="a">First operand (or the array to reduce).</param>
        /// <param name="b">Second operand, or null to reduce <paramref name="a"/>.</param>
        /// <returns>The element-wise or reduced masked maximum.</returns>
        public MaskedArray maximum(object a, object b = null)
            => b is null ? max(a) : where(greater(a, b), a, b);

        /// <summary>Element-wise minimum of two operands (mask = OR); with <paramref name="b"/> omitted it is the
        /// minimum REDUCTION of <paramref name="a"/>.</summary>
        /// <param name="a">First operand (or the array to reduce).</param>
        /// <param name="b">Second operand, or null to reduce <paramref name="a"/>.</param>
        /// <returns>The element-wise or reduced masked minimum.</returns>
        public MaskedArray minimum(object a, object b = null)
            => b is null ? min(a) : where(greater(b, a), a, b); // a<b ≡ b>a (routes around the np.less scalar quirk)

        /// <summary>Masked power (NumPy's <c>ma.power</c>): masked positions take the base's data; a non-finite
        /// result is additionally masked and its data set to the fill value.</summary>
        /// <param name="a">Base.</param><param name="b">Exponent.</param>
        /// <returns>The masked power (or the <see cref="masked"/> scalar for a fully-masked 0-D result).</returns>
        public MaskedArray power(object a, object b)
        {
            var m = Or((a as MaskedArray)?._mask, (b as MaskedArray)?._mask);
            var fa = AsData(a);
            var fb = AsData(b);
            var pw = np.power(fa, fb);
            // At operand-masked positions NumPy keeps the base's data (where(m, fa, power)).
            var result = m is null ? pw : np.where(m, fa, pw);
            var invalid = NonFiniteMask(result); // NaN/±inf ⇒ masked + data set to fill
            var finalMask = Or(m, invalid);
            if (result.ndim == 0)
                return (finalMask is not null && np.any(finalMask)) ? masked : new MaskedArray(result, null);
            if (invalid is not null && np.any(invalid))
            {
                result = result.copy();
                np.copyto(result, NDArray.Scalar(default_fill_value(result.dtype)), casting: "unsafe", where: invalid);
            }
            if (finalMask is not null && !finalMask.Shape.Equals(result.Shape))
                finalMask = np.broadcast_to(finalMask, result.Shape).copy();
            return new MaskedArray(result, finalMask);
        }

        /// <summary>Masked <c>where</c> (NumPy's <c>ma.where</c>): picks <paramref name="x"/> where the condition
        /// is True, else <paramref name="y"/>; a masked or False condition selects <paramref name="y"/>, and a
        /// masked condition (or a masked chosen value) masks the result.</summary>
        /// <param name="condition">Boolean array-like selector.</param>
        /// <param name="x">Values chosen where the condition is True.</param>
        /// <param name="y">Values chosen where the condition is False/masked.</param>
        /// <returns>The masked selection.</returns>
        public MaskedArray where(object condition, object x, object y)
        {
            var cond = condition as MaskedArray;
            var cf = (cond?._mask is null ? AsData(condition) : cond.filled(false)).astype(np.@bool);
            var xd = AsData(x);
            var yd = AsData(y);
            var cm = getmaskarray(condition);
            var xm = getmaskarray(x);
            var ym = getmaskarray(y);
            var data = np.where(cf, xd, yd);
            var mask = np.where(cf, xm, ym);
            mask = np.where(cm, NDArray.Scalar(true), mask);
            // Shrink an all-False mask back to nomask (NumPy's _shrink_mask).
            return new MaskedArray(data, np.any(mask) ? mask : null);
        }

        /// <summary>Rounds each element to <paramref name="decimals"/> places (half-to-even), preserving the mask
        /// (NumPy's <c>ma.round</c>). Masked positions carry their rounded data but stay masked.</summary>
        /// <param name="a">Data.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array.</returns>
        public MaskedArray round(object a, int decimals = 0)
            => new MaskedArray(np.around(AsData(a), decimals), (a as MaskedArray)?._mask);

        /// <summary>Alias of <see cref="round"/> (NumPy's <c>around</c>).</summary>
        /// <param name="a">Data.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array.</returns>
        public MaskedArray around(object a, int decimals = 0) => round(a, decimals);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Shape / manipulation — the SAME transform is applied to the data AND the mask,
        //  so the mask always lines up with its data (NumPy's `_fromnxfunction` pattern).
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Applies a shape transform to both the data and (if present) the mask, keeping them aligned.</summary>
        /// <param name="a">Operand.</param><param name="f">The <c>np.*</c> transform to run on data and mask.</param>
        /// <returns>The transformed masked array.</returns>
        private MaskedArray Map1(object a, Func<NDArray, NDArray> f)
        {
            var m = (a as MaskedArray)?._mask;
            return new MaskedArray(f(AsData(a)), m is null ? null : f(m));
        }

        /// <summary>Applies a joining transform to a sequence's data arrays and, when ANY input is masked, to
        /// their full mask arrays (an all-unmasked sequence stays nomask).</summary>
        /// <param name="arrays">The operands.</param><param name="f">The <c>np.*</c> join over the arrays.</param>
        /// <returns>The joined masked array.</returns>
        private MaskedArray MapSeq(object[] arrays, Func<NDArray[], NDArray> f)
        {
            var rd = f(arrays.Select(AsData).ToArray());
            if (!arrays.Any(x => (x as MaskedArray)?._mask is not null))
                return new MaskedArray(rd, null);
            return new MaskedArray(rd, f(arrays.Select(getmaskarray).ToArray()));
        }

        /// <summary>Flattened VIEW in the given order (mask flattened alike). NumPy's <c>ravel</c>.</summary>
        /// <param name="a">Operand.</param><param name="order">'C' or 'F'.</param><returns>The flattened masked array.</returns>
        public MaskedArray ravel(object a, char order = 'C') => Map1(a, d => np.ravel(d, order));

        /// <summary>Flattened COPY (NumPy's <c>flatten</c> — always copies, unlike <see cref="ravel"/>).</summary>
        /// <param name="a">Operand.</param><param name="order">'C' or 'F'.</param><returns>The flattened masked copy.</returns>
        public MaskedArray flatten(object a, char order = 'C') => Map1(a, d => np.ravel(d, order).copy());

        /// <summary>Reshape to <paramref name="new_shape"/> (mask reshaped alike). The parameter is spelled
        /// <c>new_shape</c> so a NumPy call <c>ma.reshape(a, new_shape=…)</c> ports verbatim.</summary>
        /// <param name="a">Operand.</param><param name="new_shape">New dimensions (one may be -1).</param><returns>The reshaped masked array.</returns>
        public MaskedArray reshape(object a, params int[] new_shape) => Map1(a, d => np.reshape(d, new_shape));

        /// <summary>Reshape to a <see cref="Shape"/> (mask reshaped alike).</summary>
        /// <param name="a">Operand.</param><param name="new_shape">New shape.</param><returns>The reshaped masked array.</returns>
        public MaskedArray reshape(object a, Shape new_shape) => Map1(a, d => np.reshape(d, new_shape));

        /// <summary>Permute axes (mask transposed alike); reverses all axes when <paramref name="axes"/> is null.</summary>
        /// <param name="a">Operand.</param><param name="axes">Permutation, or null to reverse.</param><returns>The transposed masked array.</returns>
        public MaskedArray transpose(object a, int[] axes = null) => Map1(a, d => np.transpose(d, axes));

        /// <summary>Swap two axes (mask swapped alike).</summary>
        /// <param name="a">Operand.</param><param name="axis1">First axis.</param><param name="axis2">Second axis.</param><returns>The masked array with swapped axes.</returns>
        public MaskedArray swapaxes(object a, int axis1, int axis2) => Map1(a, d => np.swapaxes(d, axis1, axis2));

        /// <summary>Move an axis (mask moved alike).</summary>
        /// <param name="a">Operand.</param><param name="source">Source axis.</param><param name="destination">Destination axis.</param><returns>The masked array.</returns>
        public MaskedArray moveaxis(object a, int source, int destination) => Map1(a, d => np.moveaxis(d, source, destination));

        /// <summary>Remove size-1 axes (mask squeezed alike).</summary>
        /// <param name="a">Operand.</param><returns>The squeezed masked array.</returns>
        public MaskedArray squeeze(object a) => Map1(a, d => np.squeeze(d));

        /// <summary>Insert a size-1 axis at <paramref name="axis"/> (mask expanded alike).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis position.</param><returns>The masked array.</returns>
        public MaskedArray expand_dims(object a, int axis) => Map1(a, d => np.expand_dims(d, axis));

        /// <summary>Repeat elements (mask repeated alike, so repeated masked entries stay masked).</summary>
        /// <param name="a">Operand.</param><param name="repeats">Repeat count.</param><param name="axis">Axis or null (flatten).</param><returns>The masked array.</returns>
        public MaskedArray repeat(object a, int repeats, int? axis = null) => Map1(a, d => np.repeat(d, repeats, axis));

        /// <summary>Gather elements by index (mask gathered alike, so a taken element keeps its masked-ness).</summary>
        /// <param name="a">Operand.</param><param name="indices">Integer index array.</param><param name="axis">Axis or null.</param><returns>The masked array.</returns>
        public MaskedArray take(object a, NDArray indices, int? axis = null) => Map1(a, d => np.take(d, indices, axis));

        /// <summary>Diagonal: 1-D input CONSTRUCTS a matrix with the values (and mask) on the k-th diagonal;
        /// 2-D input EXTRACTS the k-th diagonal (and its mask). NumPy's <c>ma.diag</c>.</summary>
        /// <param name="v">Operand.</param><param name="k">Diagonal offset.</param><returns>The masked array.</returns>
        public MaskedArray diag(object v, int k = 0) => Map1(v, d => np.diag(d, k));

        /// <summary>Flatten the input and build a diagonal matrix from it (mask alike).</summary>
        /// <param name="v">Operand.</param><param name="k">Diagonal offset.</param><returns>The masked diagonal matrix.</returns>
        public MaskedArray diagflat(object v, int k = 0) => Map1(v, d => np.diagflat(d, k));

        /// <summary>At-least-1-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 1.</returns>
        public MaskedArray atleast_1d(object a) => Map1(a, d => np.atleast_1d(d));

        /// <summary>At-least-2-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 2.</returns>
        public MaskedArray atleast_2d(object a) => Map1(a, d => np.atleast_2d(d));

        /// <summary>At-least-3-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 3.</returns>
        public MaskedArray atleast_3d(object a) => Map1(a, d => np.atleast_3d(d));

        /// <summary>Join arrays along an existing axis (masks joined alike; all-unmasked ⇒ nomask).</summary>
        /// <param name="arrays">Operands.</param><param name="axis">Join axis.</param><returns>The joined masked array.</returns>
        public MaskedArray concatenate(object[] arrays, int axis = 0) => MapSeq(arrays, ds => np.concatenate(ds, axis));

        /// <summary>Stack arrays along a NEW axis (masks stacked alike).</summary>
        /// <param name="arrays">Operands.</param><param name="axis">New-axis position.</param><returns>The stacked masked array.</returns>
        public MaskedArray stack(object[] arrays, int axis = 0) => MapSeq(arrays, ds => np.stack(ds, axis));

        /// <summary>Stack row-wise / along the first axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public MaskedArray vstack(params object[] tup) => MapSeq(tup, np.vstack);

        /// <summary>Stack column-wise / along the second axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public MaskedArray hstack(params object[] tup) => MapSeq(tup, np.hstack);

        /// <summary>Stack along the third axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public MaskedArray dstack(params object[] tup) => MapSeq(tup, np.dstack);

        /// <summary>Stack 1-D arrays as columns of a 2-D result (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public MaskedArray column_stack(params object[] tup) => MapSeq(tup, np.column_stack);

        /// <summary>The 1-D array of the UNMASKED values in C-order (NumPy's <c>compressed</c>) — a plain
        /// <see cref="NDArray"/>, since a compressed result has, by definition, no mask.</summary>
        /// <param name="a">Operand.</param><returns>A 1-D <see cref="NDArray"/> of the unmasked data.</returns>
        public NDArray compressed(object a)
        {
            var flat = np.ravel(AsData(a));
            var m = (a as MaskedArray)?._mask;
            return m is null ? flat.copy() : flat[np.logical_not(np.ravel(m))];
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Element-wise selection / reduction over masked data (clip / choose / compress /
        //  diagonal / trace / nonzero / diff / append)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Clamps each element into <c>[a_min, a_max]</c>, preserving the mask (NumPy's <c>ma.clip</c>).
        /// A null bound disables that side of the clip.</summary>
        /// <param name="a">Data.</param><param name="a_min">Lower bound (array-like/scalar), or null.</param>
        /// <param name="a_max">Upper bound (array-like/scalar), or null.</param>
        /// <returns>The clamped masked array (mask carried through unchanged).</returns>
        public MaskedArray clip(object a, object a_min, object a_max)
        {
            var m = (a as MaskedArray)?._mask;
            // The base np.clip does the clamp on the raw data; the mask rides through (masked data is hidden
            // anyway, so clamping it too is harmless and matches NumPy's ma.clip).
            var r = np.clip(AsData(a), a_min is null ? null : AsData(a_min), a_max is null ? null : AsData(a_max));
            return new MaskedArray(r, m);
        }

        /// <summary>
        ///     Merges choice arrays by an index array (NumPy's <c>ma.choose</c>): output[i] = choices[indices[i]][i].
        ///     The result element is masked iff the CHOSEN element was masked (or the index itself was masked) —
        ///     the mask is chosen by the SAME index array, mirroring the data.
        /// </summary>
        /// <param name="indices">Integer index array (each in <c>[0, n-1]</c>); a masked index masks its output.</param>
        /// <param name="choices">The <c>n</c> choice arrays (<see cref="MaskedArray"/>/<see cref="NDArray"/>/array-like).</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default), "wrap", or "clip".</param>
        /// <returns>The merged masked array.</returns>
        public MaskedArray choose(object indices, object[] choices, string mode = "raise")
        {
            // Index data with masked slots filled by 0 (NumPy's filled(indices, 0)).
            var im = (indices as MaskedArray)?._mask;
            var c = im is null ? AsData(indices) : ((MaskedArray)indices).filled(0);
            // Pick the data by index; masked choice slots are hidden by the mask set below.
            var data = np.choose(c, choices.Select(AsData).ToArray(), mode: mode);
            bool anyMasked = im is not null || choices.Any(ch => (ch as MaskedArray)?._mask is not null);
            if (!anyMasked)
                return new MaskedArray(data, null);
            // Pick each choice's full mask by the SAME index, then OR in the index's own mask.
            var outmask = np.choose(c, choices.Select(getmaskarray).ToArray(), mode: mode);
            outmask = Or(outmask, im);
            return new MaskedArray(data, (outmask is not null && np.any(outmask)) ? outmask : null);
        }

        /// <summary>Keeps only the elements where the boolean <paramref name="condition"/> is True (NumPy's
        /// <c>ma.compress</c>); data AND mask are compressed alike so the survivors keep their masked-ness.</summary>
        /// <param name="condition">Boolean array-like selector (over the flattened elements, or along <paramref name="axis"/>).</param>
        /// <param name="a">Data.</param><param name="axis">Axis to compress along, or null to flatten first.</param>
        /// <returns>The compressed masked array.</returns>
        public MaskedArray compress(object condition, object a, int? axis = null)
        {
            var cond = AsData(condition).astype(np.@bool);
            var m = (a as MaskedArray)?._mask;
            var data = np.compress(cond, AsData(a), axis);
            return new MaskedArray(data, m is null ? null : np.compress(cond, m, axis));
        }

        /// <summary>Extracts the <paramref name="offset"/>-th diagonal of a 2-D (or stacked) array (NumPy's
        /// <c>ma.diagonal</c>); the mask is extracted alike so each diagonal element keeps its masked-ness.</summary>
        /// <param name="a">Data (rank ≥ 2).</param><param name="offset">Diagonal offset (0 = main).</param>
        /// <param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <returns>The masked diagonal.</returns>
        public MaskedArray diagonal(object a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            var m = (a as MaskedArray)?._mask;
            return new MaskedArray(np.diagonal(AsData(a), offset, axis1, axis2),
                                   m is null ? null : np.diagonal(m, offset, axis1, axis2));
        }

        /// <summary>
        ///     Sum of the diagonal, treating masked slots as 0 (NumPy's <c>ma.trace</c>). Returns a PLAIN array,
        ///     and — matching NumPy's <c>astype(None)</c> quirk — the result is float64 unless
        ///     <paramref name="dtype"/> is given (so <c>ma.trace</c> of an int matrix is float64, not int).
        /// </summary>
        /// <param name="a">Data (rank ≥ 2).</param><param name="offset">Diagonal offset.</param>
        /// <param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <param name="dtype">Result dtype; null ⇒ float64 (NumPy's default).</param>
        /// <returns>A plain <see cref="NDArray"/> trace.</returns>
        public NDArray trace(object a, int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null)
        {
            // Filled0 zeroes masked diagonal contributions; np.trace sums the diagonal; astype(float64) reproduces
            // NumPy's astype(None)→float64 default (which is why an integer trace comes back as a float).
            return np.trace(Filled0(a), offset, axis1, axis2).astype(dtype ?? np.float64);
        }

        /// <summary>Indices of the non-zero UNMASKED elements (NumPy's <c>ma.nonzero</c>): masked slots are filled
        /// with 0 so they never count as non-zero. One plain int64 index array per dimension.</summary>
        /// <param name="a">Data.</param>
        /// <returns>An array of int64 index arrays (the house tuple convention).</returns>
        public NDArray<long>[] nonzero(object a) => np.nonzero(Filled0(a));

        /// <summary>
        ///     The n-th discrete difference along an axis, preserving the mask (NumPy's <c>ma.diff</c>): an output
        ///     element is masked iff ANY of the <c>n+1</c> input elements folded into it was masked.
        /// </summary>
        /// <param name="a">Data (rank ≥ 1).</param>
        /// <param name="n">Number of times to difference (0 returns the input; must be ≥ 0).</param>
        /// <param name="axis">Axis along which to difference (default last).</param>
        /// <param name="prepend">Values prepended along <paramref name="axis"/> before differencing, or null.</param>
        /// <param name="append">Values appended along <paramref name="axis"/> before differencing, or null.</param>
        /// <returns>The masked differences (axis length reduced by <paramref name="n"/>).</returns>
        /// <exception cref="ArgumentException"><paramref name="n"/> is negative or <paramref name="a"/> is 0-D.</exception>
        public MaskedArray diff(object a, int n = 1, int axis = -1, object prepend = null, object append = null)
        {
            if (n == 0) return array(a);
            if (n < 0) throw new ArgumentException("order must be non-negative but got " + n);
            var d = AsData(a);
            if (d.ndim == 0) throw new ArgumentException("diff requires input that is at least one dimensional");

            // Build the (optionally prepend/append-extended) data + full mask, then difference each independently:
            // the DATA via np.diff (which handles bool/unsigned exactly like NumPy), the MASK via an n-fold
            // adjacent-OR along the axis (a position is masked iff any element in its window was).
            var parts = new List<object>();
            if (prepend is not null) parts.Add(prepend);
            parts.Add(a);
            if (append is not null) parts.Add(append);
            NDArray data, mask;
            if (parts.Count > 1)
            {
                var combined = concatenate(parts.ToArray(), axis);
                data = combined._data;
                mask = combined._mask;
            }
            else
            {
                data = d;
                mask = (a as MaskedArray)?._mask;
            }
            int ax = axis < 0 ? axis + data.ndim : axis;
            var resultData = np.diff(data, n, axis);
            if (mask is null)
                return new MaskedArray(resultData, null);
            for (int k = 0; k < n; k++)
                mask = np.logical_or(DropAlong(mask, ax, dropFirst: true), DropAlong(mask, ax, dropFirst: false));
            return new MaskedArray(resultData, np.any(mask) ? mask : null);
        }

        /// <summary>Drops one element from the front (<paramref name="dropFirst"/>) or back of <paramref name="a"/>
        /// along <paramref name="axis"/> — the two operands of an adjacent-difference, built with <see cref="np.take"/>
        /// so no slice-string parsing is needed for an arbitrary axis.</summary>
        private static NDArray DropAlong(NDArray a, int axis, bool dropFirst)
        {
            int len = (int)a.shape[axis];
            var idx = dropFirst ? np.arange(1, len) : np.arange(0, len - 1);
            return np.take(a, idx, axis);
        }

        /// <summary>Appends <paramref name="b"/> to <paramref name="a"/> (NumPy's <c>ma.append</c>) = masked
        /// <see cref="concatenate"/>; with <paramref name="axis"/> null BOTH operands are flattened first.</summary>
        /// <param name="a">First operand.</param><param name="b">Value(s) appended.</param>
        /// <param name="axis">Join axis, or null to flatten both operands and append 1-D.</param>
        /// <returns>The appended masked array (a fresh allocation, never in place).</returns>
        public MaskedArray append(object a, object b, int? axis = null)
        {
            if (axis is null)
                return concatenate(new object[] { ravel(a), ravel(b) }, 0);
            return concatenate(new object[] { a, b }, axis.Value);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  extras.py — masked-aware statistics, products, set/diff helpers, clump/edges
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Data of <paramref name="a"/> with masked slots filled by 0 (for a product/reduction that
        /// must treat masked as absent); a null-mask operand is returned unchanged.</summary>
        private static NDArray Filled0(object a) => (a as MaskedArray)?._mask is null ? AsData(a) : ((MaskedArray)a).filled(0);

        /// <summary>Number of MASKED elements along the axis (NumPy's <c>count_masked</c>) — a plain int64 array.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <returns>An int64 count of masked elements.</returns>
        public NDArray count_masked(object a, int? axis = null)
        {
            var m = (a as MaskedArray)?._mask ?? np.zeros(AsData(a).Shape, np.@bool);
            return np.sum(m, axis, false, np.int64);
        }

        /// <summary>An empty masked array of the given shape with EVERY element masked (NumPy's <c>masked_all</c>) —
        /// the standard "fill me in" accumulator.</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>A fully-masked array.</returns>
        public MaskedArray masked_all(Shape shape, DType dtype = null)
            => new MaskedArray(np.empty(shape, dtype ?? np.float64), np.ones(shape, np.@bool));

        /// <summary>A fully-masked array shaped and typed like <paramref name="a"/> (NumPy's <c>masked_all_like</c>).</summary>
        /// <param name="a">Prototype array-like.</param>
        /// <returns>A fully-masked array.</returns>
        public MaskedArray masked_all_like(object a)
        {
            var d = AsData(a);
            return new MaskedArray(np.empty_like(d), np.ones(d.Shape, np.@bool));
        }

        /// <summary>Weighted average over unmasked elements (NumPy's <c>average</c>): with no weights this is the
        /// mean; otherwise <c>sum(a·w)/sum(w)</c> with masked slots dropped from BOTH sums.</summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null.</param>
        /// <param name="weights">Weights (same shape as <paramref name="a"/>, or 1-D along <paramref name="axis"/>); null ⇒ uniform.</param>
        /// <param name="keepdims">Keep the reduced axes as size-1 (NumPy 2.0 <c>keepdims</c>).</param>
        /// <returns>The masked (weighted) average.</returns>
        public MaskedArray average(object a, int? axis = null, object weights = null, bool keepdims = false)
            => AverageCore(a, axis, weights, keepdims).avg;

        /// <summary>
        ///     Weighted average AND its sum-of-weights (NumPy's <c>average(..., returned=True)</c>, which returns
        ///     the pair). The C# analog of NumPy's tuple return — <c>var (avg, sws) = np.ma.average_returned(a)</c>
        ///     — mirroring the base library's <see cref="np.average_returned(NDArray,int?,NDArray,bool)"/>. With no
        ///     weights the sum-of-weights is the COUNT of unmasked elements (as float64), matching NumPy.
        /// </summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null.</param>
        /// <param name="weights">Weights, or null for uniform.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>A tuple of the masked average and the (plain float64) sum of weights.</returns>
        public (MaskedArray avg, NDArray sumOfWeights) average_returned(object a, int? axis = null, object weights = null, bool keepdims = false)
            => AverageCore(a, axis, weights, keepdims);

        /// <summary>Shared core of <see cref="average"/>/<see cref="average_returned"/>: computes the masked
        /// average and the sum-of-weights so the two public entry points never drift apart.</summary>
        private (MaskedArray avg, NDArray sumOfWeights) AverageCore(object a, int? axis, object weights, bool keepdims)
        {
            var mask = (a as MaskedArray)?._mask;
            if (weights is null)
            {
                // Uniform: the average is the plain masked mean, and the sum-of-weights is the unmasked COUNT.
                var avgU = mean(a, axis, null, keepdims);
                var scl = CountUnmasked(mask, AsData(a).Shape, axis, keepdims).astype(np.float64);
                return (avgU, scl);
            }
            var d = AsData(a).astype(np.float64);
            var wgt = AsData(weights).astype(np.float64);
            // 1-D weights along an axis broadcast to (1,…,len,…,1).
            if (axis is not null && !wgt.Shape.Equals(d.Shape))
            {
                var shp = Enumerable.Repeat(1L, d.ndim).ToArray();
                shp[axis.Value] = d.shape[axis.Value];
                wgt = np.reshape(wgt, new Shape(shp));
            }
            var w = np.broadcast_to(wgt, d.Shape).copy();
            if (mask is not null)
                w = np.where(mask, NDArray.Scalar(0.0), w); // masked ⇒ zero weight
            var sclW = np.sum(w, axis, keepdims, np.float64);
            var num = np.sum(np.multiply(d, w), axis, keepdims, np.float64);
            var avg = np.divide(num, sclW);
            var newmask = mask is null ? null : AllAlongAxis(mask, axis, keepdims);
            return (Finalize(avg, newmask), sclW);
        }

        /// <summary>Median over unmasked elements (NumPy's <c>median</c>). Supports the flat case and the
        /// unmasked-axis case; a masked array with an explicit axis needs the masked sort core and is not yet
        /// implemented.</summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null (flatten).</param>
        /// <param name="keepdims">Keep the reduced axes as size-1 (NumPy 2.0 <c>keepdims</c>).</param>
        /// <returns>The masked median.</returns>
        /// <exception cref="NotSupportedException">A MASKED array with an explicit <paramref name="axis"/>.</exception>
        public MaskedArray median(object a, int? axis = null, bool keepdims = false)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = AsData(a);
            if (mask is null)
                return new MaskedArray(np.median(d, axis, null, false, keepdims), null);
            if (axis is null)
            {
                var comp = compressed(a);
                if (comp.size == 0)
                    return masked;
                var med = np.median(comp);
                // keepdims on a flattened reduction ⇒ an all-ones shape of the input's rank (NumPy's (1,…,1)).
                if (keepdims && d.ndim > 0)
                    med = np.reshape(med, new Shape(Enumerable.Repeat(1L, d.ndim).ToArray()));
                return new MaskedArray(med, null);
            }
            throw new NotSupportedException(
                "np.ma.median with an explicit axis on a MASKED array is not yet implemented (requires the masked sort core); flat and unmasked-axis medians are supported.");
        }

        /// <summary>Differences between consecutive UNMASKED-aware elements of the flattened input (NumPy's
        /// <c>ediff1d</c>), optionally bracketed by <paramref name="to_begin"/>/<paramref name="to_end"/>.</summary>
        /// <param name="arr">Data.</param>
        /// <param name="to_end">Values appended to the end (array-like), or null.</param>
        /// <param name="to_begin">Values prepended (array-like), or null.</param>
        /// <returns>The masked consecutive differences.</returns>
        public MaskedArray ediff1d(object arr, object to_end = null, object to_begin = null)
        {
            var flat = ravel(arr);
            var ed = subtract(Map1(flat, x => x["1:"]), Map1(flat, x => x[":-1"]));
            if (to_begin is null && to_end is null)
                return ed;
            var parts = new List<object>();
            if (to_begin is not null) parts.Add(ravel(to_begin));
            parts.Add(ed);
            if (to_end is not null) parts.Add(ravel(to_end));
            return concatenate(parts.ToArray());
        }

        /// <summary>True iff every corresponding pair of elements is equal, with masked positions treated as
        /// equal when <paramref name="fill_value"/> is true (NumPy's <c>allequal</c>).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <param name="fill_value">Treat masked positions as equal (true) or unequal (false).</param>
        /// <returns>A single bool.</returns>
        public bool allequal(object a, object b, bool fill_value = true)
        {
            NDArray eq = np.equal(AsData(a), AsData(b));
            var mu = Or((a as MaskedArray)?._mask, (b as MaskedArray)?._mask);
            if (mu is not null)
                eq = fill_value ? (eq | mu) : (eq & np.logical_not(mu));
            return np.all(eq);
        }

        /// <summary>True iff all corresponding elements are within tolerance, with masked positions treated as
        /// equal when <paramref name="masked_equal"/> is true (NumPy's <c>allclose</c>).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <param name="masked_equal">Treat masked positions as equal.</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param>
        /// <returns>A single bool.</returns>
        public bool allclose(object a, object b, bool masked_equal = true, double rtol = 1e-5, double atol = 1e-8)
        {
            NDArray close = np.isclose(AsData(a), AsData(b), rtol, atol);
            var mu = Or((a as MaskedArray)?._mask, (b as MaskedArray)?._mask);
            if (mu is not null)
                close = masked_equal ? (close | mu) : (close & np.logical_not(mu));
            return np.all(close);
        }

        /// <summary>Dot product with masked slots treated as 0; a result element is masked only if NO valid
        /// (both-unmasked) term contributed to it (NumPy's <c>ma.dot</c>, strict=False).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The masked dot product.</returns>
        public MaskedArray dot(object a, object b)
        {
            var product = np.dot(Filled0(a), Filled0(b));
            var ma_ = (a as MaskedArray)?._mask;
            var mb_ = (b as MaskedArray)?._mask;
            if (ma_ is null && mb_ is null)
                return new MaskedArray(product, null);
            var va = (ma_ is null ? np.ones(AsData(a).Shape, np.@bool) : np.logical_not(ma_)).astype(np.float64);
            var vb = (mb_ is null ? np.ones(AsData(b).Shape, np.@bool) : np.logical_not(mb_)).astype(np.float64);
            var valid = np.dot(va, vb); // count of valid contributing pairs
            var m = np.equal(valid, NDArray.Scalar(0.0));
            return new MaskedArray(product, np.any(m) ? m : null);
        }

        /// <summary>Inner product with masked slots treated as 0 (NumPy's <c>ma.inner</c>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The masked inner product (mask dropped — result treats masked as 0).</returns>
        public MaskedArray inner(object a, object b) => new MaskedArray(np.inner(Filled0(a), Filled0(b)), null);

        /// <summary>Outer product with masked slots treated as 0; result[i,j] is masked iff a[i] or b[j] was
        /// masked (NumPy's <c>ma.outer</c>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The masked outer product.</returns>
        public MaskedArray outer(object a, object b)
        {
            var da = AsData(a); var db = AsData(b);
            var product = np.outer(Filled0(a), Filled0(b));
            var ma_ = (a as MaskedArray)?._mask;
            var mb_ = (b as MaskedArray)?._mask;
            if (ma_ is null && mb_ is null)
                return new MaskedArray(product, null);
            var maf = ma_ is null ? np.zeros(new Shape(da.size), np.@bool) : np.ravel(ma_);
            var mbf = mb_ is null ? np.zeros(new Shape(db.size), np.@bool) : np.ravel(mb_);
            var mask = np.reshape(maf, new Shape(da.size, 1)) | np.reshape(mbf, new Shape(1, db.size));
            return new MaskedArray(product, mask);
        }

        /// <summary>Vandermonde matrix of a 1-D input; a masked input element ZEROES its whole row and the
        /// result is a PLAIN array (NumPy's <c>ma.vander</c>: "masked values result in rows of zeros").</summary>
        /// <param name="x">1-D data.</param><param name="n">Number of columns (null ⇒ len(x)). Lowercase to match
        /// NumPy's <c>vander(x, n=…)</c> keyword — a call spelled <c>n:</c> ports verbatim.</param>
        /// <returns>A plain <see cref="NDArray"/> Vandermonde matrix with masked rows zeroed.</returns>
        public NDArray vander(object x, int? n = null)
        {
            var d = AsData(x);
            var v = np.vander(d, n);
            var m = (x as MaskedArray)?._mask;
            if (m is not null)
            {
                var col = np.reshape(m, new Shape(d.size, 1));
                np.copyto(v, NDArray.Scalar(0), casting: "unsafe", where: np.broadcast_to(col, v.Shape));
            }
            return v;
        }

        /// <summary>Element-wise membership test (NumPy's <c>ma.isin</c>): True where <paramref name="element"/>'s
        /// value is in <paramref name="test_elements"/>, carrying <paramref name="element"/>'s mask.</summary>
        /// <param name="element">Values to test.</param><param name="test_elements">The set to test against.</param>
        /// <param name="assume_unique">Speed hint that both inputs already hold unique elements (result unchanged).</param>
        /// <param name="invert">Invert the membership test.</param>
        /// <returns>A masked boolean array shaped like <paramref name="element"/>.</returns>
        public MaskedArray isin(object element, object test_elements, bool assume_unique = false, bool invert = false)
        {
            NDArray data = np.isin(AsData(element), AsData(test_elements), assume_unique, invert);
            var em = (element as MaskedArray)?._mask;
            // A masked element is not a valid value, so its membership answer is definite: `invert`
            // (False normally, True for invert). NumPy's unique-based isin yields the same, unmasked.
            if (em is not null)
                data = np.where(em, NDArray.Scalar(invert), data);
            return new MaskedArray(data, null);
        }

        /// <summary>1-D membership test (NumPy's <c>ma.in1d</c>): the flattened <see cref="isin"/>.</summary>
        /// <param name="element">Values to test.</param><param name="test_elements">The set to test against.</param>
        /// <param name="assume_unique">Speed hint (result unchanged).</param>
        /// <param name="invert">Invert the membership test.</param>
        /// <returns>A 1-D masked boolean array.</returns>
        public MaskedArray in1d(object element, object test_elements, bool assume_unique = false, bool invert = false)
            => isin(ravel(element), test_elements, assume_unique, invert);

        /// <summary>Contiguous UNMASKED runs of a 1-D array as slices (NumPy's <c>clump_unmasked</c>).</summary>
        /// <param name="a">1-D operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous unmasked run.</returns>
        public Slice[] clump_unmasked(object a)
        {
            var m = (a as MaskedArray)?._mask;
            if (m is null)
                return new[] { new Slice(0, (int)AsData(a).size) };
            return EzClump(np.logical_not(np.ravel(m)));
        }

        /// <summary>Contiguous MASKED runs of a 1-D array as slices (NumPy's <c>clump_masked</c>).</summary>
        /// <param name="a">1-D operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous masked run (empty when unmasked).</returns>
        public Slice[] clump_masked(object a)
        {
            var m = (a as MaskedArray)?._mask;
            return m is null ? Array.Empty<Slice>() : EzClump(np.ravel(m));
        }

        /// <summary>Finds the contiguous True-runs of a 1-D boolean array as half-open slices.</summary>
        private static Slice[] EzClump(NDArray boolMask)
        {
            var m = boolMask.ToArray<bool>();
            var res = new List<Slice>();
            int i = 0, n = m.Length;
            while (i < n)
            {
                if (m[i]) { int s = i; while (i < n && m[i]) i++; res.Add(new Slice(s, i)); }
                else i++;
            }
            return res.ToArray();
        }

        /// <summary>First and last UNMASKED flat indices (NumPy's <c>flatnotmasked_edges</c>), or null when every
        /// element is masked.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>A 2-element <c>[first, last]</c> int64 array, or null.</returns>
        public long[] flatnotmasked_edges(object a)
        {
            var d = AsData(a);
            var m = (a as MaskedArray)?._mask;
            if (m is null)
                return d.size == 0 ? null : new long[] { 0, d.size - 1 };
            var um = np.logical_not(np.ravel(m)).ToArray<bool>();
            int first = -1, last = -1;
            for (int i = 0; i < um.Length; i++)
                if (um[i]) { if (first < 0) first = i; last = i; }
            return first < 0 ? null : new long[] { first, last };
        }

        /// <summary>Contiguous UNMASKED runs of the flattened array (NumPy's <c>flatnotmasked_contiguous</c>);
        /// same as <see cref="clump_unmasked"/> for a 1-D input.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous unmasked run.</returns>
        public Slice[] flatnotmasked_contiguous(object a) => clump_unmasked(ravel(a));

        /// <summary>Drops the ROWS of a 2-D array that contain ANY masked element (NumPy's <c>compress_rows</c>) —
        /// a plain <see cref="NDArray"/>.</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The surviving rows.</returns>
        public NDArray compress_rows(object a)
        {
            var keep = np.logical_not(np.any(getmaskarray(a), 1, null, false));
            return np.compress(keep, AsData(a), 0);
        }

        /// <summary>Drops the COLUMNS of a 2-D array that contain ANY masked element (NumPy's <c>compress_cols</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The surviving columns.</returns>
        public NDArray compress_cols(object a)
        {
            var keep = np.logical_not(np.any(getmaskarray(a), 0, null, false));
            return np.compress(keep, AsData(a), 1);
        }

        /// <summary>Masks the ENTIRE rows of a 2-D array that contain any masked element (NumPy's <c>mask_rows</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The row-masked array.</returns>
        public MaskedArray mask_rows(object a)
        {
            var rows = np.any(getmaskarray(a), 1, null, true);
            return masked_where(np.broadcast_to(rows, AsData(a).Shape), a);
        }

        /// <summary>Masks the ENTIRE columns of a 2-D array that contain any masked element (NumPy's <c>mask_cols</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The column-masked array.</returns>
        public MaskedArray mask_cols(object a)
        {
            var cols = np.any(getmaskarray(a), 0, null, true);
            return masked_where(np.broadcast_to(cols, AsData(a).Shape), a);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Sorting / unique — masked entries sort to the END (filled with the dtype's largest
        //  value) and the trailing k slots (k = masked count along the axis) are re-masked.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Sorts along an axis with masked entries pushed to the END (or, with <paramref name="endwith"/>
        ///     false, the FRONT) and re-masked there (NumPy's <c>ma.sort</c>). The masked slots' key is the
        ///     dtype's LARGEST value when <paramref name="endwith"/> (so they sort last, ascending) or its
        ///     SMALLEST value otherwise; their ORIGINAL data rides along to the end/front rather than the fill
        ///     value a plain sort-of-filled would leave behind.
        /// </summary>
        /// <param name="a">Operand.</param><param name="axis">Sort axis (default last).</param>
        /// <param name="endwith">True (default) pushes masked entries to the END; false to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default
        /// (<see cref="minimum_fill_value"/> for endwith, <see cref="maximum_fill_value"/> otherwise).</param>
        /// <returns>The sorted masked array.</returns>
        public MaskedArray sort(object a, int axis = -1, bool endwith = true, object fill_value = null)
        {
            var mask = (a as MaskedArray)?._mask;
            var d = AsData(a);
            if (mask is null)
                return new MaskedArray(np.sort(d, axis), null);
            // Sort by the argsort of the FILLED keys, then reorder BOTH data and mask by that permutation.
            var order = np.argsort(((MaskedArray)a).filled(fill_value ?? SortFill(a, endwith)), axis);
            return new MaskedArray(np.take_along_axis(d, order, axis), np.take_along_axis(mask, order, axis));
        }

        /// <summary>
        ///     Indices that would sort the array with masked entries treated as the dtype's largest (or, with
        ///     <paramref name="endwith"/> false, smallest) value — NumPy's <c>ma.argsort</c>. A PLAIN int64
        ///     index array (masked entries land at the end/front).
        /// </summary>
        /// <param name="a">Operand.</param><param name="axis">Sort axis (default last).</param>
        /// <param name="endwith">True (default) sends masked entries to the END; false to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices.</returns>
        public NDArray argsort(object a, int axis = -1, bool endwith = true, object fill_value = null)
        {
            var d = (a as MaskedArray)?._mask is null ? AsData(a) : ((MaskedArray)a).filled(fill_value ?? SortFill(a, endwith));
            return np.argsort(d, axis);
        }

        /// <summary>The masked-slot sort key for <see cref="sort"/>/<see cref="argsort"/>: the dtype's largest
        /// value for endwith (masked sorts last) or its smallest value otherwise (masked sorts first).</summary>
        private object SortFill(object a, bool endwith) => endwith ? minimum_fill_value(a) : maximum_fill_value(a);

        /// <summary>Sorted unique values over the UNMASKED elements, plus ONE trailing masked entry if any
        /// element was masked (NumPy's <c>ma.unique</c>, values only).</summary>
        /// <param name="ar">Operand.</param>
        /// <returns>The masked array of unique values.</returns>
        public MaskedArray unique(object ar)
        {
            NDArray uvals = np.unique(compressed(ar));
            var mask = (ar as MaskedArray)?._mask;
            if (mask is null || !np.any(mask))
                return new MaskedArray(uvals, null);
            // Append a single masked slot (NumPy keeps one masked value in the unique set).
            var data = np.concatenate(new[] { uvals, np.zeros(new Shape(1), uvals.dtype) }, 0);
            var m = np.concatenate(new[] { np.zeros(uvals.Shape, np.@bool), np.ones(new Shape(1), np.@bool) }, 0);
            return new MaskedArray(data, m);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Set operations (extras.py arraysetops) — masked values are all considered EQUAL
        //  to one another (and unequal to every real value), so every masked element in the
        //  inputs collapses to at most ONE masked entry that sorts to the very end of the
        //  result. NumPy builds these by unique/concatenate/sort over the MaskedArray subclass;
        //  NumSharp has no subclass dispatch, so each is reformulated as the identical-output
        //  composition: run the plain np.* set op over the UNMASKED values, then decide from a
        //  boolean masked-presence rule whether the single collapsed masked entry belongs in
        //  the result. Both formulations are verified bit-identical to NumPy 2.4.2 (values,
        //  mask, dtype) across masked/unmasked/all-masked/empty/promotion cases.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     True iff <paramref name="a"/> is a masked array with AT LEAST ONE element actually masked
        ///     (a plain array, a <c>nomask</c> operand, or an all-False mask all count as "not masked").
        ///     This is the predicate that decides whether a set operation carries the single collapsed
        ///     masked value into its result — the whole masked/unmasked branch of the set ops turns on it.
        /// </summary>
        /// <param name="a">Any accepted operand.</param>
        /// <returns>True when the operand contributes a masked value to the set.</returns>
        private static bool HasMasked(object a)
        {
            // `is not null` — a `!= null` here would run NDArray's ELEMENTWISE `!=` (see the file-wide trap).
            var m = (a as MaskedArray)?._mask;
            return m is not null && np.any(m);
        }

        /// <summary>
        ///     Assembles a set-operation result: the sorted unmasked-unique <paramref name="values"/> already
        ///     in the promoted dtype, plus — when <paramref name="includeMasked"/> — ONE trailing masked slot
        ///     (NumPy collapses every masked element to a single masked value that sorts last). Mirrors the
        ///     append pattern <see cref="unique"/> uses and is the shared tail of all four set ops.
        /// </summary>
        /// <param name="values">The 1-D sorted unmasked-unique values, already cast to the result dtype.</param>
        /// <param name="includeMasked">Append the single trailing masked entry when true.</param>
        /// <returns>An unmasked (<c>nomask</c>) masked array when <paramref name="includeMasked"/> is false;
        /// otherwise the values followed by one masked slot.</returns>
        private static MaskedArray WithTrailingMasked(NDArray values, bool includeMasked)
        {
            if (!includeMasked)
                return new MaskedArray(values, null);
            // The datum at the masked slot is arbitrary — the mask hides it and NumPy's own raw value there is
            // whatever its sort/concatenate left (not the fill). A zero of the result dtype keeps `.filled()`
            // (the observable value) consistent with the dtype default, matching NumPy's contract.
            var data = np.concatenate(new[] { values, np.zeros(new Shape(1), values.dtype) }, 0);
            var m = np.concatenate(new[] { np.zeros(values.Shape, np.@bool), np.ones(new Shape(1), np.@bool) }, 0);
            return new MaskedArray(data, m);
        }

        /// <summary>
        ///     The promoted result dtype of a two-operand set op (NumPy's implicit
        ///     <c>result_type</c> from its <c>concatenate</c>). Computed from the DATA dtypes so it is correct
        ///     even when both operands compress to nothing — the plain <c>np.*</c> set op would otherwise fall
        ///     to float64 for two empty inputs and lose the integer dtype NumPy keeps.
        /// </summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The dtype the result values must carry.</returns>
        private static DType SetOpDtype(object a, object b) => np.result_type(AsData(a), AsData(b));

        /// <summary>
        ///     Sorted values common to BOTH inputs among their UNMASKED elements, plus a single masked entry
        ///     iff BOTH inputs contained a masked element (NumPy's <c>ma.intersect1d</c>: masked values are
        ///     equal only to one another, so a masked value is "shared" precisely when both sides carry one).
        ///     The result is ALWAYS a masked array, even when neither input is masked.
        /// </summary>
        /// <param name="ar1">First operand (<see cref="MaskedArray"/>/<see cref="NDArray"/>/scalar/array-like).</param>
        /// <param name="ar2">Second operand.</param>
        /// <param name="assume_unique">Speed hint passed to the underlying set op that the inputs already hold
        /// unique elements; when it is false but the inputs are not unique, the result is undefined (matching NumPy).</param>
        /// <returns>The masked array of shared unique values (sorted; the promoted dtype is preserved even for an
        /// empty/all-masked result).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        public MaskedArray intersect1d(object ar1, object ar2, bool assume_unique = false)
        {
            var dt = SetOpDtype(ar1, ar2);
            // Compress to the unmasked values, run the plain set op, then FORCE the promoted dtype (astype is a
            // no-op copy on the non-empty path since the set op already promoted; it only bites when both inputs
            // compress to empty, where np.intersect1d would otherwise return float64).
            var values = np.intersect1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            return WithTrailingMasked(values, HasMasked(ar1) && HasMasked(ar2));
        }

        /// <summary>
        ///     Sorted union of the UNMASKED values of both inputs, plus a single masked entry iff EITHER input
        ///     contained a masked element (NumPy's <c>ma.union1d</c>). Always returns a masked array.
        /// </summary>
        /// <param name="ar1">First operand.</param><param name="ar2">Second operand.</param>
        /// <returns>The masked array of unique values from either input (sorted; promoted dtype preserved).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        public MaskedArray union1d(object ar1, object ar2)
        {
            var dt = SetOpDtype(ar1, ar2);
            var values = np.union1d(compressed(ar1), compressed(ar2)).astype(dt, copy: false);
            return WithTrailingMasked(values, HasMasked(ar1) || HasMasked(ar2));
        }

        /// <summary>
        ///     Sorted values present in EXACTLY ONE of the inputs among their UNMASKED elements (the symmetric
        ///     difference), plus a single masked entry iff exactly one input contained a masked element (NumPy's
        ///     <c>ma.setxor1d</c> — a masked value is "in" an input iff that input has one, so it survives the
        ///     xor precisely when the two sides disagree on having a masked element). Always returns a masked array.
        /// </summary>
        /// <param name="ar1">First operand.</param><param name="ar2">Second operand.</param>
        /// <param name="assume_unique">Speed hint passed through; results are undefined (as in NumPy) if false but
        /// the inputs are not unique.</param>
        /// <returns>The masked array of the symmetric difference (sorted; promoted dtype preserved).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        public MaskedArray setxor1d(object ar1, object ar2, bool assume_unique = false)
        {
            var dt = SetOpDtype(ar1, ar2);
            var values = np.setxor1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            // XOR of masked presence: the collapsed masked value is in exactly one side ⇒ survives the xor.
            return WithTrailingMasked(values, HasMasked(ar1) ^ HasMasked(ar2));
        }

        /// <summary>
        ///     Sorted UNMASKED values of <paramref name="ar1"/> that are NOT in <paramref name="ar2"/>, plus a
        ///     single masked entry iff <paramref name="ar1"/> had a masked element and <paramref name="ar2"/> did
        ///     NOT (NumPy's <c>ma.setdiff1d</c> — the collapsed masked value is removed by the diff exactly when
        ///     the right side also carries one). Always returns a masked array.
        /// </summary>
        /// <param name="ar1">The operand to keep values from.</param>
        /// <param name="ar2">The operand whose values are removed.</param>
        /// <param name="assume_unique">Speed hint passed through; results are undefined (as in NumPy) if false but
        /// the inputs are not unique.</param>
        /// <returns>The masked array of <paramref name="ar1"/>-only unique values (sorted; <paramref name="ar1"/>'s
        /// promotion preserved).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        public MaskedArray setdiff1d(object ar1, object ar2, bool assume_unique = false)
        {
            var dt = SetOpDtype(ar1, ar2);
            var values = np.setdiff1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            // ar1's masked value survives the diff iff ar2 does not also carry one (masked == masked removes it).
            return WithTrailingMasked(values, HasMasked(ar1) && !HasMasked(ar2));
        }

        // ── Mask hardness — NumSharp has no hard/soft mask distinction (masks are plain boolean
        //    arrays), so these are accepted for API parity and are effectively no-ops. ──

        /// <summary>Accepted for NumPy parity; NumSharp masks have no hard/soft state, so this returns the
        /// array unchanged.</summary>
        /// <param name="a">Operand.</param><returns>The same masked array.</returns>
        public MaskedArray harden_mask(object a) => array(a);

        /// <summary>Accepted for NumPy parity; a no-op in NumSharp (no hard/soft mask state).</summary>
        /// <param name="a">Operand.</param><returns>The same masked array.</returns>
        public MaskedArray soften_mask(object a) => array(a);

        /// <summary>Drops an all-False mask back to nomask (NumPy's <c>shrink_mask</c>); otherwise unchanged.</summary>
        /// <param name="a">Operand.</param><returns>The masked array, with a redundant all-False mask removed.</returns>
        public MaskedArray shrink_mask(object a)
        {
            var m = (a as MaskedArray)?._mask;
            return (m is not null && !np.any(m)) ? new MaskedArray(AsData(a), null) : array(a);
        }
    }
}
