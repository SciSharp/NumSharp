using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;
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
        ///     A <see cref="NDMaskedArray"/> pairs an ordinary <see cref="NDArray"/> of DATA with a boolean
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
    public class NDMaskedArray
    {
        /// <summary>The underlying data buffer. Never null. Masked positions still hold a data value
        /// (whatever the producing op left there); that value is semantically hidden but observable via
        /// <see cref="data"/> for parity with NumPy.</summary>
        internal readonly NDArray _data;

        /// <summary>The boolean mask (True = masked/ignored), or <c>null</c> for NumPy's <c>nomask</c>
        /// (no element masked). When non-null its shape equals <see cref="_data"/>'s shape. NOT readonly:
        /// the indexer's <c>x[i] = masked</c> path, <see cref="MaskedArrayModule.put"/> and
        /// <see cref="MaskedArrayModule.putmask"/> CREATE or mutate the mask in place (NumPy's <c>__setitem__</c>
        /// promotes a <c>nomask</c> array to a real mask on the first masking write).</summary>
        internal NDArray _mask;

        /// <summary>The value substituted for masked elements by <see cref="filled(object)"/> when the caller
        /// gives none; null means "use the dtype default" (<see cref="MaskedArrayModule.default_fill_value"/>).
        /// NOT readonly: NumPy's <c>MaskedArray.fill_value</c> is settable, so <see cref="fill_value"/> and
        /// <see cref="MaskedArrayModule.set_fill_value"/> mutate it in place on the SAME instance.</summary>
        /// <remarks>
        ///     <b>What is stored.</b> A TEXT array stores NumPy's own representation, a 0-d <c>U</c>/<c>S</c> array whose
        ///     WIDTH is part of the state: a fill set by the caller is converted to the data's dtype (<c>'hello'</c> on
        ///     <c>&lt;U2</c> data is stored as <c>'he'</c>), while the default the getter MATERIALIZES on first read is
        ///     stored at its own width (<c>np.array('N/A')</c>, a <c>&lt;U3</c>), and later writes go INTO the stored array
        ///     (<c>_fill_value[()] = target</c>), truncating to that width. A numeric fill is the caller's value, stored
        ///     unconverted (the module's documented simplification). Derived arrays share this object
        ///     (<see cref="UpdateFrom"/>, NumPy's <c>_update_from</c>) or convert it to their dtype
        ///     (<see cref="FinalizeFrom"/>, NumPy's <c>__array_finalize__</c>).
        /// </remarks>
        internal object _fill_value;

        /// <summary>Whether the mask is HARD (NumPy's <c>hardmask</c>): when true, a plain-value assignment
        /// (the indexer, <see cref="MaskedArrayModule.put"/>, <see cref="MaskedArrayModule.putmask"/>) does NOT
        /// unmask a masked slot nor overwrite its data — only currently-UNmasked positions change (masking a slot
        /// via the <c>masked</c> singleton is still allowed). Set by <see cref="harden_mask()"/>/
        /// <see cref="soften_mask()"/>; default false (soft, so assignment unmasks — NumPy's default).</summary>
        /// <remarks>Scope: the flag lives on the instance and gates the direct assignment paths. A derived array takes it
        /// wherever NumPy runs <c>_update_from</c> / <c>__array_finalize__</c> (<see cref="UpdateFrom"/> /
        /// <see cref="FinalizeFrom"/>), probed member by member against NumPy 2.4.2:
        /// <list type="bullet">
        ///     <item>every indexer result (a basic-slice VIEW and a fancy/boolean COPY alike) — what stops a write through
        ///     <c>x["0:3"]</c> from unmasking the parent's protected slots;</item>
        ///     <item>the <c>_arraymethod</c> family: reshape / ravel / flatten / transpose / <see cref="T"/> / swapaxes /
        ///     squeeze / expand_dims / repeat / compress / diagonal / copy / clip / round, and <see cref="astype"/> /
        ///     <see cref="view(DType,object)"/>;</item>
        ///     <item>the masked ufuncs (<c>ma.add</c>, <c>ma.sqrt</c>, the domained divide family, ...) from their first
        ///     masked operand, <c>ma.power</c> from its base only, and the operators: arithmetic through those ufuncs,
        ///     <c>-x</c> / <c>~x</c> / <c>%</c> / <c>&amp;</c> / <c>|</c> / <c>^</c> through ndarray's ufunc and
        ///     <c>__array_wrap__</c> (<see cref="MaskedArrayModule.UfuncOperator(NDMaskedArray,Func{NDArray,NDArray})"/>),
        ///     the ordering operators through <c>_comparison</c>
        ///     (<see cref="MaskedArrayModule.ComparisonOperator"/>);</item>
        ///     <item>the <c>*_like</c> functions, masked_where and the masked_* family built on it (masked_values, which
        ///     runs masked_array over the FILLED data, comes back soft), masked_all_like, var / std / anom, diff, sort and
        ///     unique.</item>
        /// </list>
        /// NumPy returns <see cref="real"/> / <see cref="imag"/> / <see cref="mT"/> / take / the reductions / the
        /// <c>_fromnxfunction</c> results / concatenate / where / dot / the shifts SOFT with the default fill value, and so
        /// does NumSharp.</remarks>
        internal bool _hardmask;

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
        internal NDMaskedArray(NDArray data, NDArray mask, object fill_value = null)
        {
            // Data is the one invariant that must always hold — every property and the printer dereference it.
            if (data is null)
                throw new ArgumentNullException(nameof(data));
            // A numpy.matrix is stored as a plain view of its buffer. numpy.ma keeps the matrix class
            // (_baseclass, a "masked_matrix" whose reductions stay two-dimensional); that has no model here, and
            // every member computes with np.* calls written for plain arrays, long after this constructor
            // returned. The view belongs to this masked array, so no scope may dispose it (Own).
            _data = NDMatrixBoundary.IsMatrix(data) ? MaskedArrayModule.Own(NDMatrixBoundary.PlainOf(data)) : data;
            _mask = NDMatrixBoundary.IsMatrix(mask) ? MaskedArrayModule.Own(NDMatrixBoundary.PlainOf(mask)) : mask;
            _fill_value = fill_value;
        }

        /// <summary>
        ///     Copies onto this array the attributes a masked array DERIVED from <paramref name="source"/> inherits:
        ///     the default fill value and the hard-mask flag. This is NumPy's <c>MaskedArray._update_from(obj)</c>,
        ///     limited to the two attributes NumSharp models. Call it wherever NumPy runs <c>_update_from(self)</c>.
        ///     Without it the result comes back SOFT with the dtype-default fill value, and a write through a view of
        ///     a hard-masked array then unmasks the parent's protected slots.
        /// </summary>
        /// <param name="source">The masked array this one is derived from (for example, the array an indexer read
        /// came from).</param>
        /// <returns>This array, so the call chains onto its <c>new</c> expression.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <remarks>
        ///     The attributes are COPIED, not linked. NumPy keeps them in each instance's own <c>__dict__</c>, so a
        ///     later <see cref="harden_mask()"/>/<see cref="soften_mask()"/> or fill-value change on either array does
        ///     not reach the other. Probed on NumPy 2.4.2: softening a slice leaves its parent hard, and hardening a
        ///     parent leaves an earlier slice soft. NumPy's other <c>_update_from</c> attributes (<c>_sharedmask</c>,
        ///     <c>_isfield</c>, <c>_baseclass</c>, <c>_optinfo</c>) have no NumSharp model and are not copied.
        ///     Decide per construction site: NumPy does NOT run it for every derived array (its <see cref="real"/>/
        ///     <see cref="imag"/>/<see cref="mT"/>/take/reductions come back soft with the default fill value).
        /// </remarks>
        internal NDMaskedArray UpdateFrom(NDMaskedArray source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            _fill_value = source._fill_value;
            _hardmask = source._hardmask;
            return this;
        }

        /// <summary>
        ///     NumPy's <c>MaskedArray.__array_finalize__(obj)</c> for a masked <paramref name="source"/>: the
        ///     <see cref="UpdateFrom"/> copy followed by the fill check <c>_check_fill_value(self._fill_value,
        ///     self.dtype)</c>, which converts an inherited TEXT fill to THIS array's dtype. Use it where NumPy builds the
        ///     result as a new array or view OF a masked array (<c>np.array(a, copy=..., subok=True)</c> in
        ///     <c>masked_where</c>, <c>a.view(MaskedArray)</c> at the end of <c>ma.unique</c>); use <see cref="UpdateFrom"/>
        ///     where NumPy calls <c>_update_from</c> on a view of the plain data (the <c>_arraymethod</c> family: ravel,
        ///     reshape, copy, ...), which shares the fill object unconverted.
        /// </summary>
        /// <param name="source">The masked array this one derives from.</param>
        /// <returns>This array, so the call chains onto its <c>new</c> expression.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="TypeError">The inherited text fill cannot be encoded / decoded into this array's string class
        ///     (see <see cref="MaskedArrayModule.CheckFillValue"/>).</exception>
        /// <remarks>
        ///     The two only differ for a fill whose width is not this array's: chiefly the default NumPy materializes at
        ///     its own width (<c>'N/A'</c>), which this conversion truncates (<c>'N'</c> for <c>&lt;U1</c>). So
        ///     <c>ma.masked_inside</c> of a <c>&lt;U1</c> array (whose <c>filled()</c> read the source's fill and so
        ///     materialized it) comes back with the fill <c>'N'</c>, as NumPy's does. A numeric fill passes unconverted.
        /// </remarks>
        internal NDMaskedArray FinalizeFrom(NDMaskedArray source)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            // `_update_from(obj)`, then `if self._fill_value is not None: self._fill_value = _check_fill_value(...)`.
            _fill_value = source._fill_value is null ? null : MaskedArrayModule.StoreFill(source._fill_value, _data.dtype);
            _hardmask = source._hardmask;
            return this;
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

        /// <summary>Bytes per data element (NumPy's <c>.itemsize</c>). Describes the DATA buffer; the mask is a
        /// separate boolean array and does not contribute.</summary>
        public int itemsize => _data.itemsize;

        /// <summary>Total bytes consumed by the DATA (NumPy's <c>.nbytes</c>): <c>size × itemsize</c>. Matches
        /// NumPy in EXCLUDING the mask's bytes — the mask is tracked separately, so a masked array reports the
        /// same <c>nbytes</c> as its unmasked data.</summary>
        public long nbytes => _data.nbytes;

        /// <summary>Per-axis strides of the DATA, in BYTES (NumPy's <c>.strides</c>; NumSharp reports byte strides
        /// to match NumPy). The mask, when present, shares the data's shape but has its own boolean strides.</summary>
        public long[] strides => _data.strides;

        /// <summary>
        ///     The data with every masked element replaced by <paramref name="fill_value"/> (NumPy's
        ///     <c>MaskedArray.filled</c>): the standard way to hand masked data to code that cannot see the mask — the mask
        ///     is "baked in" as a sentinel value while the dtype is preserved. When there is nothing to substitute (no mask,
        ///     or no slot masked) the result is a VIEW of the data, as NumPy's <c>return self._data</c> is.
        /// </summary>
        /// <param name="fill_value">The value written at masked positions; null uses this array's
        /// <see cref="fill_value"/> (whose getter materializes a text default, as NumPy's does). An explicit value first
        /// goes through NumPy's <c>_check_fill_value</c> (<see cref="MaskedArrayModule.CheckFillValue"/>), then is cast to
        /// the data dtype (unsafe cast, like NumPy). A text fill is truncated to the data's width (<c>'N/A'</c> into
        /// <c>&lt;U2</c> is <c>'N/'</c>).</param>
        /// <returns>A K-order copy of the data with the masked slots filled; or, when no slot is masked, a new VIEW of the
        ///     data (writes through it reach this array; disposing it releases only the view's own reference).</returns>
        /// <exception cref="TypeError">A <see cref="string"/> <paramref name="fill_value"/> for a numeric array, or a text
        ///     fill the data's class cannot encode or decode (see <see cref="MaskedArrayModule.CheckFillValue"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A masked slot of a text array must receive a number
        ///     formatted as text, or a text fill must be parsed as a number for a numeric array: the next phase of the
        ///     string support.</exception>
        /// <remarks>
        ///     NumPy's order: <c>if m is nomask: return self._data</c>; resolve the fill (<c>self.fill_value</c> or
        ///     <c>_check_fill_value(fill_value, self.dtype)</c>); <c>elif not m.any(): return self._data</c>; else
        ///     <c>result = self._data.copy('K'); np.copyto(result, fill_value, where=m)</c>. So an unmasked array never
        ///     refuses (or converts) the argument, and a fill that NumPy refuses is refused even when no slot is masked —
        ///     except a NUMBER for TEXT data: NumPy formats it (which cannot fail) and then returns the data untouched when
        ///     nothing is masked, so that case answers here without the formatting the next phase owns.
        ///     <para>Scoped (<c>[NDScoped]</c>): the 0-d fill scalar handed to <see cref="np.copyto(NDArray, NDArray, string, NDArray)"/> (and a converted array
        ///     fill) is reclaimed on return; only the result leaves.</para>
        /// </remarks>
        [NDScoped]
        public NDArray filled(object fill_value = null)
        {
            // `if m is nomask: return self._data` — NumPy's `_data` property is a new VIEW each time (OWNDATA false, shares
            // the memory). NOTE: `is null` (reference check) — `_mask == null` would invoke NDArray's ELEMENTWISE `==`.
            if (_mask is null)
                return _data.view();

            // A number given for TEXT data is formatted by `_check_fill_value` before the mask is read; the formatting
            // cannot fail, so with nothing masked the data is the answer and the next phase's formatting is never needed.
            if (fill_value is not null && _data.typecode == NPTypeCode.String
                                       && MaskedArrayModule.FormatsAsText(fill_value) && !np.any(_mask))
                return _data.view();

            // `fill_value = self.fill_value` when omitted, else `_check_fill_value(fill_value, self.dtype)`.
            var fv = fill_value is null ? this.fill_value : MaskedArrayModule.CheckFillValue(fill_value, _data.dtype);

            // `elif not m.any(): return self._data`.
            if (!np.any(_mask))
                return _data.view();

            // `result = self._data.copy('K'); np.copyto(result, fill_value, where=m)`: copyto broadcasts the fill into every
            // masked slot; unsafe cast because a generic fill value (e.g. 1e20 into an int array default) must land in the
            // data dtype exactly as NumPy coerces it. An array fill (NumPy's deprecated non-scalar fill) is broadcast as is.
            var result = _data.copy('K');
            np.copyto(result, fv as NDArray ?? NDArray.Scalar(fv), casting: "unsafe", where: _mask);
            return result;
        }

        /// <summary>
        ///     The NumPy <c>str()</c> form of the masked array — the data with masked slots shown inline as the
        ///     <see cref="MaskedArrayModule.masked_print_option"/> token (default <c>--</c>), e.g.
        ///     <c>[1 -- 3 --]</c>. This is a BYTE-EXACT port of NumPy 2.4.2's <c>MaskedArray.__str__</c> (see
        ///     <see cref="ToString(bool)"/> for the full algorithm/caveats). Mirrors <see cref="NDArray.ToString()"/>'s
        ///     convention: <c>ToString()</c> is the <c>str</c> form, <c>ToString(true)</c> the <c>repr</c> form.
        /// </summary>
        /// <returns>The masked <c>str()</c> rendering.</returns>
        public override string ToString() => ToString(flat: false);

        /// <summary>
        ///     Renders the masked array as NumPy 2.4.2 does — a BYTE-EXACT port of <c>MaskedArray.__str__</c>
        ///     (<paramref name="flat"/> false) and <c>MaskedArray.__repr__</c> (<paramref name="flat"/> true).
        ///     <para>
        ///     <b>Why masked prints look UNALIGNED.</b> When a mask is present and
        ///     <see cref="MaskedArrayModule.masked_print_option"/> is enabled, NumPy converts the data to an OBJECT
        ///     array, substitutes the display token at masked slots, and prints via object-array formatting — each
        ///     element its own Python-scalar repr with NO numeric column alignment (so <c>[1 -- 3 --]</c>, not the
        ///     padded numeric form). A <c>nomask</c> array (or a disabled print option, which fills the masked slots
        ///     with the fill value instead) prints the numeric array with normal alignment.
        ///     </para>
        ///     <para>
        ///     <b>repr envelope.</b> The <c>masked_array(data=…, mask=…, fill_value=…[, dtype=…])</c> template,
        ///     indents, one-row-vs-multi-row layout, and the <c>dtype=</c>-shown rule (non-implied dtype, or an
        ///     all-masked/empty array) all match NumPy exactly. The <c>fill_value=</c> line is byte-exact for the
        ///     DEFAULT fill value across every dtype (including the <c>np.&lt;type&gt;(value)</c> wrapper NumPy 2.x
        ///     shows when the fill's promoted dtype differs from the data dtype); a caller-SET custom fill of a
        ///     mismatched dtype is rendered against the data dtype and may differ in that one line only.
        ///     </para>
        /// </summary>
        /// <param name="flat">False ⇒ the NumPy <c>str()</c> form (data only); true ⇒ the <c>repr()</c> form.</param>
        /// <returns>The rendered string.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): printing a masked array builds throwaway arrays (the filled copy a
        /// disabled print option renders, the <c>np.all(mask)</c> probe of the <c>dtype=</c> rule); all of them are
        /// reclaimed before the string is returned.</remarks>
        [NDScoped]
        public string ToString(bool flat)
        {
            var opts = PrintOptions.Current;
            // str() is just the data body (object layout with `--`, or the numeric/filled layout); repr() wraps it
            // in the masked_array(...) template with the mask, fill value and (when needed) dtype lines.
            return flat ? ReprString(opts) : InsertMaskedPrintBody(" ", "", "", opts);
        }

        /// <summary>
        ///     Produces the <c>data=</c> body exactly as NumPy's <c>str(self._insert_masked_print())</c>: the
        ///     object-array layout with the display token at masked slots when a mask is present and the print
        ///     option is enabled, otherwise the plain numeric layout of the data (nomask) or of
        ///     <see cref="filled(object)"/> (disabled). Shared by both <c>str</c> (separator <c>" "</c>, no
        ///     prefix/suffix) and <c>repr</c> (separator <c>", "</c>, with the alignment prefix and <c>","</c> suffix).
        /// </summary>
        /// <param name="separator">Element separator.</param>
        /// <param name="prefix">Alignment prefix (length only) for line wrapping.</param>
        /// <param name="suffix">Trailing text (length only) reserved on the last line.</param>
        /// <param name="opts">The active print options.</param>
        /// <param name="reprForm">True for the <c>repr</c> body (<c>np.array2string</c>), false for the <c>str</c> body
        ///     (<c>str(array)</c>). It decides only a 0-d TEXT element: array2string prints its repr (<c>'ab'</c>,
        ///     <c>b'ab'</c>) where str prints the text itself (<c>ab</c>) and the bytes scalar's repr
        ///     (<c>np.bytes_(b'ab')</c>).</param>
        /// <returns>The rendered data body (no prefix/suffix text, just the bracketed array).</returns>
        private string InsertMaskedPrintBody(string separator, string prefix, string suffix, PrintOptions opts, bool reprForm = false)
        {
            var mpo = np.ma.masked_print_option;
            // nomask OR the print option disabled ⇒ a real numeric array (data, or the fill-substituted data)
            // printed with normal aligned formatting — NumPy's `_insert_masked_print` returns `self._data` /
            // `self.filled(fill_value)` there.
            if (_mask is null || !mpo.enabled())
            {
                var numeric = _mask is null ? _data : filled();
                return numeric.ndim == 0
                    ? ZeroDimElement(numeric.GetAtIndex(0), numeric.typecode, reprForm)
                    : ArrayFormatter.Array2String(numeric, opts, separator, prefix, suffix);
            }
            // masked + enabled ⇒ object-array layout with the display token at masked slots.
            string display = mpo.display();
            if (_data.ndim == 0)
                return Convert.ToBoolean(_mask.GetAtIndex(0))
                    ? display
                    : ZeroDimElement(_data.GetAtIndex(0), _data.typecode, reprForm);
            return ArrayFormatter.MaskedObjectArray2String(_data, _mask, display, opts, separator, prefix, suffix);
        }

        /// <summary>
        ///     The element of a 0-d masked array as NumPy prints it: <c>np.array2string</c> (the repr body) formats a text
        ///     element by its repr, <c>str()</c> by the scalar's str; a numeric element prints the same in both.
        /// </summary>
        /// <param name="value">The boxed element.</param>
        /// <param name="tc">The data's storage lane.</param>
        /// <param name="reprForm">True for the repr body.</param>
        /// <returns>The element's text.</returns>
        private static string ZeroDimElement(object value, NPTypeCode tc, bool reprForm)
        {
            if (reprForm && tc == NPTypeCode.String)
                return value is byte[] bytes ? PyRepr.Bytes(bytes) : PyRepr.Str((string)value);
            return ArrayFormatter.ScalarStr(value, tc);
        }

        /// <summary>Builds the <c>masked_array(...)</c> repr — a line-for-line port of NumPy 2.4.2's
        /// <c>MaskedArray.__repr__</c> (modern print mode): the one-row-vs-multi-row indent scheme, the
        /// <c>data=</c>/<c>mask=</c>/<c>fill_value=</c> lines, and the conditional <c>dtype=</c> line.</summary>
        /// <param name="opts">The active print options.</param>
        /// <returns>The multi-line repr string.</returns>
        private string ReprString(PrintOptions opts)
        {
            const string prefix = "masked_array(";
            // dtype is shown when it is not implied by the repr, OR the whole array is masked, OR it is empty —
            // NumPy's exact condition (an all-masked array shows dtype because its data values never appear).
            bool dtypeNeeded = !DtypeIsImplied(_data.typecode)
                               || (_mask is not null && np.all(_mask))
                               || _data.size == 0;

            // is_one_row: every axis but the last is length 1 (so 0-D/1-D/(1,…,N) put the first key on the type
            // line and align the rest by '='; otherwise each key goes on its own 2-space-indented line).
            bool isOneRow = true;
            for (int i = 0; i < _data.ndim - 1; i++)
                if (_data.Shape.dimensions[i] != 1) { isOneRow = false; break; }

            string dataIndent, maskIndent, fillIndent, dtypeIndent, lead;
            if (isOneRow)
            {
                // First key ('data') sits right after "masked_array("; the rest align so their '=' lines up under
                // "data"'s (n = max(2, len("masked_array(data") - len(key))).
                dataIndent = prefix;
                int anchor = prefix.Length + "data".Length; // len("masked_array(data") == 17
                maskIndent = new string(' ', Math.Max(2, anchor - "mask".Length));
                fillIndent = new string(' ', Math.Max(2, anchor - "fill_value".Length));
                dtypeIndent = new string(' ', Math.Max(2, anchor - "dtype".Length));
                lead = "";
            }
            else
            {
                dataIndent = maskIndent = fillIndent = dtypeIndent = "  ";
                lead = prefix + "\n";
            }

            // data= : the object/numeric body, wrap-aligned under its own "…data=" prefix (np.array2string semantics).
            string dataRepr = InsertMaskedPrintBody(", ", dataIndent + "data=", ",", opts, reprForm: true);
            // mask= : the boolean mask printed normally; a nomask array shows the bare scalar "False".
            string maskRepr = _mask is null
                ? "False"
                : ArrayFormatter.Array2String(_mask, opts, ", ", maskIndent + "mask=", ",");
            string fillRepr = FillValueRepr();

            var sb = new StringBuilder();
            sb.Append(lead);
            sb.Append(dataIndent).Append("data=").Append(dataRepr).Append(",\n");
            sb.Append(maskIndent).Append("mask=").Append(maskRepr).Append(",\n");
            sb.Append(fillIndent).Append("fill_value=").Append(fillRepr);
            // dtype_short_repr: a builtin numeric dtype prints its name, a string dtype its quoted descriptor ('<U2').
            if (dtypeNeeded)
                sb.Append(",\n").Append(dtypeIndent).Append("dtype=").Append(ArrayFormatter.DtypeShortRepr(_data));
            sb.Append(')');
            return sb.ToString();
        }

        /// <summary>NumPy's <c>_typelessdata</c> membership: dtypes whose repr does not need an explicit
        /// <c>dtype=</c> suffix (the four defaults int64/float64/complex128/bool). Mirrors
        /// <c>ArrayFormatter</c>'s private check so the masked repr agrees with the plain-array repr.</summary>
        /// <param name="tc">The data typecode.</param>
        /// <returns>True when the dtype is implied (no <c>dtype=</c> line needed on that ground alone).</returns>
        private static bool DtypeIsImplied(NPTypeCode tc) =>
            tc == NPTypeCode.Int64 || tc == NPTypeCode.Double ||
            tc == NPTypeCode.Complex || tc == NPTypeCode.Boolean;

        /// <summary>
        ///     Renders the <c>fill_value=</c> value exactly as NumPy 2.4.2's repr: the plain scalar string when the
        ///     fill's effective dtype matches the data dtype, else the NumPy-2.x <c>np.&lt;dtype&gt;(value)</c>
        ///     wrapper. For the DEFAULT fill the effective dtype follows NumPy's <c>_check_fill_value</c> promotion
        ///     (signed→int64, unsigned→uint64, float→float64, complex→complex128, bool→bool), which only differs
        ///     from the data dtype for a non-implied data dtype (int8/16/32, uint8/16/32, float16/32) — precisely
        ///     the arrays that also show a <c>dtype=</c> line. A caller-SET custom fill is rendered against the data
        ///     dtype (the plain form), matching NumPy whenever the value fits that dtype. A TEXT array prints Python's repr
        ///     of its fill (<c>fill_value='N/A'</c>, <c>fill_value=b'N/A'</c>), NumPy's string branch.
        /// </summary>
        /// <returns>The fill-value display string.</returns>
        /// <exception cref="ValueError">A text array whose fill value is an array of more or fewer than one element
        ///     (NumPy's deprecated non-scalar fill): NumPy's repr calls <c>.item()</c> on it and raises <c>can only convert
        ///     an array of size 1 to a Python scalar</c>.</exception>
        private string FillValueRepr()
        {
            NPTypeCode dtc = _data.typecode;
            // `if self._fill_value is None: self.fill_value  # initialize fill_value` — NumPy's repr MATERIALIZES a text
            // default through the getter (a numeric default is not materialized; see the fill_value getter).
            object fv = dtc == NPTypeCode.String ? fill_value : _fill_value ?? np.ma.default_fill_value(_data.dtype);
            if (dtc == NPTypeCode.String)
            {
                // `repr(self.fill_value.item())` when the fill's kind is the data's (S with S, U with U), which every text
                // fill is after _check_fill_value: Python's repr of the str ('N/A', "q'") or of the bytes (b'N/A').
                if (fv is NDArray arrayFill)
                {
                    if (arrayFill.size != 1)
                        throw new ValueError("can only convert an array of size 1 to a Python scalar");
                    fv = arrayFill.item();
                }
                return fv is byte[] bytes ? PyRepr.Bytes(bytes) : PyRepr.Str((string)fv);
            }
            // Effective fill dtype: a default fill promotes by kind; a user-set fill is shown against the data dtype.
            NPTypeCode ftc = _fill_value is not null ? dtc : CanonicalFillTypecode(dtc);
            string valStr = ArrayFormatter.ScalarStr(fv, ftc);
            if (ftc == dtc)
                return valStr;
            // Non-matching promoted dtype ⇒ NumPy shows np.<name>(value); a complex value's str carries outer
            // parens that the wrapped form drops (np.complex128(1e+20+0j), not np.complex128((1e+20+0j))).
            string inner = ftc == NPTypeCode.Complex && valStr.Length >= 2 && valStr[0] == '(' && valStr[^1] == ')'
                ? valStr.Substring(1, valStr.Length - 2)
                : valStr;
            return "np." + ftc.AsNumpyDtypeName() + "(" + inner + ")";
        }

        /// <summary>The dtype NumPy's <c>default_fill_value</c> scalar carries per data kind (its repr dtype):
        /// signed integers → int64, unsigned integers → uint64, any float → float64, complex → complex128, bool →
        /// bool. NumSharp-only Char/Decimal (no NumPy analog) map to themselves so their fill shows the plain form.</summary>
        /// <param name="tc">The data typecode.</param>
        /// <returns>The fill scalar's effective typecode for repr.</returns>
        private static NPTypeCode CanonicalFillTypecode(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return NPTypeCode.Boolean;
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.Int32:
                case NPTypeCode.Int64: return NPTypeCode.Int64;
                case NPTypeCode.Byte:
                case NPTypeCode.UInt16:
                case NPTypeCode.UInt32:
                case NPTypeCode.UInt64: return NPTypeCode.UInt64;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double: return NPTypeCode.Double;
                case NPTypeCode.Complex: return NPTypeCode.Complex;
                default: return tc; // Char/Decimal — NumSharp extensions with no NumPy fill-dtype promotion
            }
        }

        /// <summary>
        ///     Adopts a plain <see cref="NDArray"/> as an unmasked <see cref="NDMaskedArray"/> (no element
        ///     masked), so every <c>np.ma.*</c> function accepts a bare array argument exactly as NumPy does.
        /// </summary>
        /// <param name="a">The array to wrap; null yields null.</param>
        public static implicit operator NDMaskedArray(NDArray a) => a is null ? null : new NDMaskedArray(a, null);

        // ── Mixed NDMaskedArray/NDArray operators. These EXPLICIT (NDMaskedArray, NDArray) / (NDArray, NDMaskedArray)
        //    overloads are REQUIRED: without them `ma + nd` and `nd + ma` are CS0034-ambiguous, because the
        //    implicit NDArray→NDMaskedArray conversion makes both the (…, NDMaskedArray) and (…, object) overloads
        //    applicable with no better candidate. An exact NDArray parameter beats both, so these win cleanly
        //    and return a mask-aware NDMaskedArray — matching NumPy, where `ma + nd`/`nd + ma` are MaskedArrays.
        /// <summary>Masked sum with an NDArray operand (mask = the masked operand's).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a+b, masked.</returns>
        public static NDMaskedArray operator +(NDMaskedArray a, NDArray b) => np.ma.add(a, b);
        /// <summary>Masked sum with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a+b, masked.</returns>
        public static NDMaskedArray operator +(NDArray a, NDMaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked difference with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a-b, masked.</returns>
        public static NDMaskedArray operator -(NDMaskedArray a, NDArray b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a-b, masked.</returns>
        public static NDMaskedArray operator -(NDArray a, NDMaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked product with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a*b, masked.</returns>
        public static NDMaskedArray operator *(NDMaskedArray a, NDArray b) => np.ma.multiply(a, b);
        /// <summary>Masked product with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a*b, masked.</returns>
        public static NDMaskedArray operator *(NDArray a, NDMaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked division with an NDArray denominator.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator NDArray.</param><returns>a/b, masked.</returns>
        public static NDMaskedArray operator /(NDMaskedArray a, NDArray b) => np.ma.divide(a, b);
        /// <summary>Masked division with an NDArray numerator.</summary>
        /// <param name="a">Numerator NDArray.</param><param name="b">Denominator masked array.</param><returns>a/b, masked.</returns>
        public static NDMaskedArray operator /(NDArray a, NDMaskedArray b) => np.ma.divide(a, b);
        /// <summary>Masked less-than with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;b, masked bool.</returns>
        public static NDMaskedArray operator <(NDMaskedArray a, NDArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less(x, y));
        /// <summary>Masked less-than with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&lt;b, masked bool.</returns>
        public static NDMaskedArray operator <(NDArray a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.greater(x, y));
        /// <summary>Masked greater-than with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;b, masked bool.</returns>
        public static NDMaskedArray operator >(NDMaskedArray a, NDArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater(x, y));
        /// <summary>Masked greater-than with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&gt;b, masked bool.</returns>
        public static NDMaskedArray operator >(NDArray a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.less(x, y));
        /// <summary>Masked less-or-equal with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;=b, masked bool.</returns>
        public static NDMaskedArray operator <=(NDMaskedArray a, NDArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less_equal(x, y));
        /// <summary>Masked less-or-equal with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&lt;=b, masked bool.</returns>
        public static NDMaskedArray operator <=(NDArray a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.greater_equal(x, y));
        /// <summary>Masked greater-or-equal with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;=b, masked bool.</returns>
        public static NDMaskedArray operator >=(NDMaskedArray a, NDArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater_equal(x, y));
        /// <summary>Masked greater-or-equal with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&gt;=b, masked bool.</returns>
        public static NDMaskedArray operator >=(NDArray a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.less_equal(x, y));

        /// <summary>
        ///     Casts the DATA to <paramref name="dtype"/>, PRESERVING the mask (NumPy's <c>MaskedArray.astype</c>
        ///     — the mask is boolean and dtype-independent, so it rides through unchanged). Returns a
        ///     <see cref="NDMaskedArray"/>, unlike the underlying <see cref="NDArray.astype(DType,bool)"/>.
        /// </summary>
        /// <param name="dtype">Target element dtype.</param>
        /// <param name="copy">Copy the data even when the dtype already matches (NumPy default true).</param>
        /// <returns>A masked array of the new dtype with the same mask.</returns>
        /// <exception cref="StringComputationNotSupportedException">The cast converts between text and numbers (the data or
        ///     a text fill value): the next phase of the string support.</exception>
        /// <remarks>
        ///     NumPy's <c>__array_finalize__</c> re-checks a stored fill against the new dtype
        ///     (<c>_check_fill_value(self._fill_value, self.dtype)</c>), so a text fill follows a width change: a
        ///     <c>'qrs'</c> fill becomes <c>'q'</c> after <c>astype('U1')</c> and <c>b'qr'</c> after <c>astype('S2')</c>.
        ///     The data is cast first, so its own refusal wins. A numeric fill of a numeric cast is kept as given (the
        ///     module's unconverted numeric fills). The result's mask is a COPY of this array's and its hard-mask flag is
        ///     this array's; with <paramref name="copy"/> false and the same dtype the array itself is returned.
        /// </remarks>
        public NDMaskedArray astype(DType dtype, bool copy = true)
        {
            var data = _data.astype(dtype, copy);
            // ndarray.astype(copy=False) of the same dtype returns the array ITSELF.
            if (ReferenceEquals(data, _data))
                return this;
            var fill = _fill_value is not null && (data.typecode == NPTypeCode.String || _data.typecode == NPTypeCode.String)
                ? MaskedArrayModule.StoreFill(_fill_value, data.dtype)
                : _fill_value;
            // `__array_finalize__` over a NEW buffer: the mask is copied (`_mask.astype(...)`, so a masked write through
            // the result never reaches this array) and the hard-mask flag carries over (`_update_from(obj)`).
            return new NDMaskedArray(data, _mask?.copy(), fill) { _hardmask = _hardmask };
        }

        // ── Arithmetic operators — compose the np.ma.* binary ufuncs; masks propagate (OR). ──
        /// <summary>Masked element-wise sum (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a+b, masked.</returns>
        public static NDMaskedArray operator +(NDMaskedArray a, NDMaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked sum with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a+b, masked.</returns>
        public static NDMaskedArray operator +(NDMaskedArray a, object b) => np.ma.add(a, b);
        /// <summary>Masked sum with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a+b, masked.</returns>
        public static NDMaskedArray operator +(object a, NDMaskedArray b) => np.ma.add(a, b);
        /// <summary>Masked element-wise difference.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a-b, masked.</returns>
        public static NDMaskedArray operator -(NDMaskedArray a, NDMaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a-b, masked.</returns>
        public static NDMaskedArray operator -(NDMaskedArray a, object b) => np.ma.subtract(a, b);
        /// <summary>Masked difference with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a-b, masked.</returns>
        public static NDMaskedArray operator -(object a, NDMaskedArray b) => np.ma.subtract(a, b);
        /// <summary>Masked element-wise product.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a*b, masked.</returns>
        public static NDMaskedArray operator *(NDMaskedArray a, NDMaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked product with a scalar/array-like right operand.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a*b, masked.</returns>
        public static NDMaskedArray operator *(NDMaskedArray a, object b) => np.ma.multiply(a, b);
        /// <summary>Masked product with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a*b, masked.</returns>
        public static NDMaskedArray operator *(object a, NDMaskedArray b) => np.ma.multiply(a, b);
        /// <summary>Masked element-wise true division (unsafe divides are additionally masked).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param><returns>a/b, masked.</returns>
        public static NDMaskedArray operator /(NDMaskedArray a, NDMaskedArray b) => np.ma.divide(a, b);
        /// <summary>Masked division with a scalar/array-like denominator.</summary>
        /// <param name="a">Numerator masked array.</param><param name="b">Denominator scalar/array-like.</param><returns>a/b, masked.</returns>
        public static NDMaskedArray operator /(NDMaskedArray a, object b) => np.ma.divide(a, b);
        /// <summary>Masked division with a scalar/array-like numerator.</summary>
        /// <param name="a">Numerator scalar/array-like.</param><param name="b">Denominator masked array.</param><returns>a/b, masked.</returns>
        public static NDMaskedArray operator /(object a, NDMaskedArray b) => np.ma.divide(a, b);
        /// <summary>Element-wise negation, NumPy's <c>-x</c>: MaskedArray defines no <c>__neg__</c>, so ndarray's runs the
        /// plain <c>np.negative</c> and <c>__array_wrap__</c> wraps it (see <see cref="MaskedArrayModule.UfuncOperator(NDMaskedArray,Func{NDArray,NDArray})"/>):
        /// a masked slot holds the negated data (where <see cref="MaskedArrayModule.negative"/> writes the operand's data
        /// back), the mask is the operand's own mask object, and the fill value and hard mask carry over.</summary>
        /// <param name="a">Operand.</param><returns>-a, masked.</returns>
        /// <exception cref="TypeError">A bool or text operand (np.negative's refusal).</exception>
        public static NDMaskedArray operator -(NDMaskedArray a) => np.ma.UfuncOperator(a, x => np.negative(x));

        // ── Ordering operators — NumPy's MaskedArray._comparison (MaskedArrayModule.ComparisonOperator): the data is
        //    the plain comparison (no fill-back), the mask a COPY of the OR shrunk to nomask, the fill value cast to
        //    bool; `scalar < x` is x's reflected `__gt__`. ==/!= are deliberately NOT overloaded (they would shadow
        //    reference equality and trip the `== null` elementwise trap); use np.ma.equal / np.ma.not_equal for
        //    element-wise (in)equality. ──
        /// <summary>Masked element-wise less-than.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;b, masked bool.</returns>
        public static NDMaskedArray operator <(NDMaskedArray a, NDMaskedArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less(x, y));
        /// <summary>Masked less-than against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&lt;b, masked bool.</returns>
        public static NDMaskedArray operator <(NDMaskedArray a, object b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less(x, y));
        /// <summary>Masked less-than with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&lt;b, masked bool.</returns>
        public static NDMaskedArray operator <(object a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.greater(x, y));
        /// <summary>Masked element-wise greater-than.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;b, masked bool.</returns>
        public static NDMaskedArray operator >(NDMaskedArray a, NDMaskedArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater(x, y));
        /// <summary>Masked greater-than against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&gt;b, masked bool.</returns>
        public static NDMaskedArray operator >(NDMaskedArray a, object b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater(x, y));
        /// <summary>Masked greater-than with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&gt;b, masked bool.</returns>
        public static NDMaskedArray operator >(object a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.less(x, y));
        /// <summary>Masked element-wise less-or-equal.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&lt;=b, masked bool.</returns>
        public static NDMaskedArray operator <=(NDMaskedArray a, NDMaskedArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less_equal(x, y));
        /// <summary>Masked less-or-equal against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&lt;=b, masked bool.</returns>
        public static NDMaskedArray operator <=(NDMaskedArray a, object b) => np.ma.ComparisonOperator(a, b, (x, y) => np.less_equal(x, y));
        /// <summary>Masked less-or-equal with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&lt;=b, masked bool.</returns>
        public static NDMaskedArray operator <=(object a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.greater_equal(x, y));
        /// <summary>Masked element-wise greater-or-equal.</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&gt;=b, masked bool.</returns>
        public static NDMaskedArray operator >=(NDMaskedArray a, NDMaskedArray b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater_equal(x, y));
        /// <summary>Masked greater-or-equal against a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&gt;=b, masked bool.</returns>
        public static NDMaskedArray operator >=(NDMaskedArray a, object b) => np.ma.ComparisonOperator(a, b, (x, y) => np.greater_equal(x, y));
        /// <summary>Masked greater-or-equal with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&gt;=b, masked bool.</returns>
        public static NDMaskedArray operator >=(object a, NDMaskedArray b) => np.ma.ComparisonOperator(b, a, (x, y) => np.less_equal(x, y));

        // ── Instance reduction methods (NumPy's a.sum()/a.mean()/… — delegate to the np.ma.* funcs). ──
        /// <summary>Sum over unmasked elements. See <see cref="MaskedArrayModule.sum"/>.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked sum.</returns>
        public NDMaskedArray sum(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.sum(this, axis, dtype, keepdims);
        /// <summary>Product over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked product.</returns>
        public NDMaskedArray prod(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.prod(this, axis, dtype, keepdims);

        /// <summary>Alias of <see cref="prod(int?,DType,bool)"/> (NumPy's <c>MaskedArray.product</c>) — the product
        /// of the UNMASKED elements; a fully-masked slice yields <c>masked</c>.</summary>
        /// <param name="axis">Reduction axis, or null to reduce the whole array.</param>
        /// <param name="dtype">Accumulator/result dtype; null follows NumPy's product-widening rule.</param>
        /// <param name="keepdims">Keep the reduced axis as size 1.</param>
        /// <returns>The masked product.</returns>
        public NDMaskedArray product(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.product(this, axis, dtype, keepdims);
        /// <summary>Mean over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked mean.</returns>
        public NDMaskedArray mean(int? axis = null, DType dtype = null, bool keepdims = false) => np.ma.mean(this, axis, dtype, keepdims);
        /// <summary>Minimum over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked minimum.</returns>
        public NDMaskedArray min(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.min(this, axis, fill_value, keepdims);
        /// <summary>Maximum over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked maximum.</returns>
        public NDMaskedArray max(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.max(this, axis, fill_value, keepdims);
        /// <summary>Peak-to-peak (max−min) over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked range.</returns>
        public NDMaskedArray ptp(int? axis = null, object fill_value = null, bool keepdims = false) => np.ma.ptp(this, axis, fill_value, keepdims);
        /// <summary>Standard deviation over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="ddof">Delta DOF.</param><param name="keepdims">Keep reduced axes.</param><param name="mean">Precomputed centering mean (NumPy 2.0), or null.</param><returns>Masked std.</returns>
        public NDMaskedArray std(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, object mean = null) => np.ma.std(this, axis, dtype, ddof, keepdims, mean);
        /// <summary>Variance over unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><param name="ddof">Delta DOF.</param><param name="keepdims">Keep reduced axes.</param><param name="mean">Precomputed centering mean (NumPy 2.0), or null.</param><returns>Masked variance.</returns>
        public NDMaskedArray var(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, object mean = null) => np.ma.var(this, axis, dtype, ddof, keepdims, mean);
        /// <summary>Count of unmasked elements.</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>int64 count array.</returns>
        public NDArray count(int? axis = null, bool keepdims = false) => np.ma.count(this, axis, keepdims);
        /// <summary>Cumulative sum (masked positions preserved).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><returns>Masked cumulative sum.</returns>
        public NDMaskedArray cumsum(int? axis = null, DType dtype = null) => np.ma.cumsum(this, axis, dtype);
        /// <summary>Cumulative product (masked positions preserved).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Accumulator dtype.</param><returns>Masked cumulative product.</returns>
        public NDMaskedArray cumprod(int? axis = null, DType dtype = null) => np.ma.cumprod(this, axis, dtype);
        /// <summary>Indices of the minimum (masked slots ignored).</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><returns>int64 index array.</returns>
        public NDArray argmin(int? axis = null, object fill_value = null) => np.ma.argmin(this, axis, fill_value);
        /// <summary>Indices of the maximum (masked slots ignored).</summary>
        /// <param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><returns>int64 index array.</returns>
        public NDArray argmax(int? axis = null, object fill_value = null) => np.ma.argmax(this, axis, fill_value);
        /// <summary>Logical-AND reduction (masked slots treated as True).</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked bool.</returns>
        public NDMaskedArray all(int? axis = null, bool keepdims = false) => np.ma.all(this, axis, keepdims);
        /// <summary>Logical-OR reduction (masked slots treated as False).</summary>
        /// <param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param><returns>Masked bool.</returns>
        public NDMaskedArray any(int? axis = null, bool keepdims = false) => np.ma.any(this, axis, keepdims);
        /// <summary>Anomalies (deviations from the mean along the axis).</summary>
        /// <param name="axis">Axis or null.</param><param name="dtype">Mean accumulator dtype.</param><returns>Masked anomalies.</returns>
        public NDMaskedArray anom(int? axis = null, DType dtype = null) => np.ma.anom(this, axis, dtype);

        // ── fill_value — the sentinel filled() substitutes for masked slots. Settable (NumPy parity):
        //    the getter resolves the stored override or the dtype default; the setter mutates THIS instance
        //    (so `set_fill_value`/`common_fill_value` observe the change), it does NOT return a new array. ──
        /// <summary>
        ///     The value <see cref="filled()"/> substitutes for masked elements (NumPy's settable
        ///     <c>MaskedArray.fill_value</c>). Reading it resolves the stored override, else the dtype default
        ///     (see <see cref="MaskedArrayModule.default_fill_value"/>) — so it is never null. Writing it stores
        ///     the override on THIS instance in place (as NumPy mutates <c>a.fill_value = v</c>), after NumPy's
        ///     <c>_check_fill_value</c> (<see cref="MaskedArrayModule.CheckFillValue"/>); a null restores the dtype default.
        /// </summary>
        /// <exception cref="TypeError">Set: a <see cref="string"/> for a numeric array (<c>Cannot set fill value of string
        ///     with array of dtype float64</c>), or text the array's class cannot encode or decode.</exception>
        /// <exception cref="StringComputationNotSupportedException">Set: a number for a text array, or text for a numeric
        ///     array that NumPy would parse (the next phase of the string support).</exception>
        /// <remarks>
        ///     NumPy's method forms <c>get_fill_value()</c>/<c>set_fill_value()</c> are NOT offered as
        ///     instance methods: a C# property named <c>fill_value</c> already reserves those exact accessor names,
        ///     so use the property (<c>a.fill_value</c> / <c>a.fill_value = v</c>) or the module-level
        ///     <see cref="MaskedArrayModule.set_fill_value(object,object)"/>.
        ///     <para>
        ///     <b>Text fills keep NumPy's state.</b> The getter of a text array with no fill yet MATERIALIZES the default, as
        ///     NumPy's <c>self._fill_value = _check_fill_value(None, self.dtype)</c> does: the stored default is
        ///     <c>np.array('N/A')</c> at its own width (<c>&lt;U3</c>), so the getter still reads <c>'N/A'</c> for
        ///     <c>&lt;U2</c> data, while an array later DERIVED by conversion reads it truncated (<c>'N/'</c>; see
        ///     <see cref="FinalizeFrom"/>). A set fill is converted to the data dtype (<c>'he'</c> after setting
        ///     <c>'hello'</c> on <c>&lt;U2</c> data) and written INTO a stored fill (<c>_fill_value[()] = target</c>), so it
        ///     takes the stored width: setting <c>'hello'</c> on <c>&lt;U5</c> data whose default was materialized reads back
        ///     <c>'hel'</c>. Setting null writes the default <c>'N/A'</c> into a stored fill at that fill's width
        ///     (<c>'N/'</c> after a fill set on <c>&lt;U2</c> data, <c>'N/A'</c> over the materialized default).
        ///     </para>
        /// </remarks>
        public object fill_value
        {
            get
            {
                // `if self._fill_value is None: self._fill_value = _check_fill_value(None, self.dtype)` — for a text dtype
                // that is `np.array('N/A')` / `np.array(b'N/A')` at the DEFAULT's own width, kept as the stored state. A
                // numeric default is not materialized: the module stores numeric fills unconverted, so a stored default
                // would change nothing NumSharp models.
                if (_fill_value is null && _data.typecode == NPTypeCode.String)
                    _fill_value = MaskedArrayModule.Own(NDArray.Scalar(np.ma.default_fill_value(_data.dtype)));
                // `return self._fill_value[()]`: the element of a 0-d fill, an N-d (deprecated) fill as itself.
                return MaskedArrayModule.FillScalar(_fill_value) ?? np.ma.default_fill_value(_data.dtype);
            }
            set => _fill_value = ResolveFillAssignment(value);
        }

        /// <summary>
        ///     The value a <see cref="fill_value"/> assignment stores (NumPy's <c>fill_value.setter</c>):
        ///     <c>target = _check_fill_value(value, self.dtype)</c>, then the target itself when no fill is stored, else the
        ///     target written INTO the stored text fill (<c>_fill_value[()] = target</c>, at the stored width).
        /// </summary>
        /// <param name="value">The assigned fill, or null to restore the default.</param>
        /// <returns>The object to store in <see cref="_fill_value"/>.</returns>
        /// <exception cref="TypeError">See <see cref="MaskedArrayModule.CheckFillValue"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="MaskedArrayModule.CheckFillValue"/>.</exception>
        internal object ResolveFillAssignment(object value)
        {
            bool text = _data.typecode == NPTypeCode.String;
            // A numeric array keeps the module's numeric model: a value is stored unconverted, null clears to the default.
            if (!text)
                return value is null ? null : MaskedArrayModule.StoreFill(value, _data.dtype);
            // `target = _check_fill_value(value, self.dtype)`: a value converted to the data dtype; for null, the default at
            // its OWN width (`np.array('N/A')`, not cut to the data's).
            object target = value is null
                ? MaskedArrayModule.Own(NDArray.Scalar(np.ma.default_fill_value(_data.dtype)))
                : MaskedArrayModule.StoreFill(value, _data.dtype);
            // `if _fill_value is None: self._fill_value = target` — also over a stored value that is not NumPy's array
            // representation (a raw string an internal constructor passed).
            if (_fill_value is not NDArray stored)
                return target;
            // `else: _fill_value[()] = target` — written into the stored array, so its width (not the data's) applies, and
            // every array sharing that fill object sees the change, as NumPy's do. The target is ours (converted or
            // created above); it only feeds the write.
            using (var t = target as NDArray ?? NDArray.Scalar(target))
                np.copyto(stored, t, casting: "unsafe");
            return stored;
        }

        // ── Instance shape/manipulation/selection — wire through to the np.ma.* module funcs (which apply
        //    the SAME transform to data and mask), so `a.reshape(...)`/`a.sort()` port from NumPy verbatim. ──
        /// <summary>Transpose (reverse all axes); NumPy's <c>MaskedArray.T</c>. Data and mask are transposed
        /// alike, so the mask stays aligned. A view — no data is moved.</summary>
        public NDMaskedArray T => np.ma.transpose(this);

        /// <summary>Matrix transpose — swaps the last two axes (NumPy's <c>MaskedArray.mT</c>); requires rank ≥ 2. A NEW
        /// masked array (soft, the default fill value) whose mask is a swapped VIEW of this one's.</summary>
        /// <exception cref="ArgumentException">The array is rank &lt; 2.</exception>
        public NDMaskedArray mT
        {
            get
            {
                // NumPy raises for < 2-D; swapaxes(-2,-1) itself would too, but the message would name axes
                // rather than mT, so guard explicitly with NumPy's verbatim text.
                if (ndim < 2)
                    throw new ArgumentException("matrix transpose with ndim < 2 is undefined");
                // `masked_array(data=self._data.mT, mask=self.mask.mT)`: a NEW masked array over the swapped data with
                // the swapped mask VIEW (shared memory) — soft, with the default fill value, inheriting nothing.
                return new NDMaskedArray(np.swapaxes(_data, -2, -1), _mask is null ? null : np.swapaxes(_mask, -2, -1));
            }
        }

        /// <summary>The real part as a masked array (NumPy's <c>MaskedArray.real</c>): <c>self._data.real.view(type(self))</c>
        /// with <c>__setmask__(self._mask)</c> — the mask is COPIED, and the result is soft with the default fill value (the
        /// view of a plain array inherits nothing).</summary>
        public NDMaskedArray real => new NDMaskedArray(_data.real, _mask?.copy());

        /// <summary>The imaginary part as a masked array (NumPy's <c>MaskedArray.imag</c>), built like <see cref="real"/>: the
        /// mask COPIED, soft, the default fill value. For a non-complex dtype every element is 0.</summary>
        public NDMaskedArray imag => new NDMaskedArray(_data.imag, _mask?.copy());

        /// <summary>Flattened VIEW in row-major order (NumPy's <c>ravel</c>); mask flattened alike.</summary>
        /// <param name="order">'C' (row-major, default) or 'F' (column-major).</param><returns>The flattened masked array.</returns>
        public NDMaskedArray ravel(char order = 'C') => np.ma.ravel(this, order);

        /// <summary>Flattened COPY (NumPy's <c>flatten</c> — always copies, unlike <see cref="ravel"/>).</summary>
        /// <param name="order">'C' (default) or 'F'.</param><returns>The flattened masked copy.</returns>
        public NDMaskedArray flatten(char order = 'C') => np.ma.flatten(this, order);

        /// <summary>Reshape to <paramref name="shape"/> (mask reshaped alike).</summary>
        /// <param name="shape">New dimensions (one may be -1).</param><returns>The reshaped masked array.</returns>
        public NDMaskedArray reshape(params int[] shape) => np.ma.reshape(this, shape);

        /// <summary>Permute axes (mask transposed alike); reverses all axes when <paramref name="axes"/> is null.</summary>
        /// <param name="axes">Permutation, or null to reverse.</param><returns>The transposed masked array.</returns>
        public NDMaskedArray transpose(int[] axes = null) => np.ma.transpose(this, axes);

        /// <summary>Swap two axes (mask swapped alike).</summary>
        /// <param name="axis1">First axis.</param><param name="axis2">Second axis.</param><returns>The masked array with swapped axes.</returns>
        public NDMaskedArray swapaxes(int axis1, int axis2) => np.ma.swapaxes(this, axis1, axis2);

        /// <summary>Remove size-1 axes (mask squeezed alike).</summary>
        /// <returns>The squeezed masked array.</returns>
        public NDMaskedArray squeeze(int? axis = null) => np.ma.squeeze(this, axis);

        /// <summary>Repeat elements (mask repeated alike, so repeated masked entries stay masked).</summary>
        /// <param name="repeats">Repeat count.</param><param name="axis">Axis or null (flatten).</param><returns>The masked array.</returns>
        public NDMaskedArray repeat(int repeats, int? axis = null) => np.ma.repeat(this, repeats, axis);

        /// <summary>Gather elements by index (mask gathered alike, so a taken element keeps its masked-ness).</summary>
        /// <param name="indices">Integer index array.</param><param name="axis">Axis or null.</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default)/"wrap"/"clip".</param>
        /// <returns>The masked array.</returns>
        public NDMaskedArray take(NDArray indices, int? axis = null, string mode = "raise") => np.ma.take(this, indices, axis, mode);

        /// <summary>The 1-D array of the UNMASKED values in C-order (NumPy's <c>MaskedArray.compressed</c>) —
        /// a plain <see cref="NDArray"/>, since a compressed result has no mask.</summary>
        /// <returns>A 1-D <see cref="NDArray"/> of the unmasked data.</returns>
        public NDArray compressed() => np.ma.compressed(this);

        /// <summary>Keep only elements where the flattened <paramref name="condition"/> is True (NumPy's
        /// <c>MaskedArray.compress</c>); mask carried alike.</summary>
        /// <param name="condition">Boolean array-like selector over the (flattened) elements.</param>
        /// <param name="axis">Axis to compress along, or null to flatten first.</param>
        /// <returns>The compressed masked array.</returns>
        public NDMaskedArray compress(object condition, int? axis = null) => np.ma.compress(condition, this, axis);

        /// <summary>A DEEP copy of this masked array — data AND mask duplicated, sharing no memory with the source
        /// (NumPy's <c>MaskedArray.copy</c>). Reach for it before mutating a view/slice you must not disturb.</summary>
        /// <returns>An independent <see cref="NDMaskedArray"/> with the same values, mask and fill value.</returns>
        public NDMaskedArray copy() => np.ma.copy(this);

        /// <summary>Uses THIS array's elements as indices to select from <paramref name="choices"/> (NumPy's
        /// <c>MaskedArray.choose</c>); the result is masked wherever this index was masked (and where a chosen
        /// element is). Reproduces the module <see cref="MaskedArrayModule.choose"/> — a masked INDEX slot's hidden
        /// data may differ from NumPy's instance form (the module fills a masked index with 0 before selecting,
        /// NumPy's instance keeps the raw index), but the slot is masked either way, so the observable
        /// values+mask match.</summary>
        /// <param name="choices">The choice arrays (an <see cref="NDArray"/>[] binds via covariance; scalars need
        /// <c>new object[]{…}</c>).</param>
        /// <param name="mode">Out-of-bounds index policy: "raise" (default)/"wrap"/"clip".</param>
        /// <returns>The selected, masked array.</returns>
        public NDMaskedArray choose(object[] choices, string mode = "raise") => np.ma.choose(this, choices, mode);

        /// <summary>Sorts along an axis with masked entries pushed to the end (or front) and re-masked there. Unlike NumPy's
        /// <c>MaskedArray.sort</c>, which sorts in place and returns None, this returns the sorted copy
        /// <see cref="MaskedArrayModule.sort"/> builds and leaves this array as it is.</summary>
        /// <param name="axis">Sort axis (default last).</param>
        /// <param name="endwith">True (default) pushes masked to the END; false pushes them to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>The sorted masked array.</returns>
        public NDMaskedArray sort(int axis = -1, bool endwith = true, object fill_value = null) => np.ma.sort(this, axis, endwith, fill_value);

        /// <summary>Indices that would sort the array with masked entries at the end (or front), with NumPy's DEFAULT axis
        /// (<c>MaskedArray.argsort(axis=np._NoValue)</c>: the last axis of a 0-d / 1-D array, the FLATTENED array —
        /// NumPy's deprecated default — from two dimensions on).</summary>
        /// <param name="endwith">True (default) sorts masked LAST; false sorts them FIRST.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices.</returns>
        public NDArray argsort(bool endwith = true, object fill_value = null) => np.ma.argsort(this, endwith, fill_value);

        /// <summary>Indices that would sort the array along <paramref name="axis"/> with masked entries at the end (or
        /// front).</summary>
        /// <param name="axis">Sort axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="endwith">True (default) sorts masked LAST; false sorts them FIRST.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices.</returns>
        public NDArray argsort(int? axis, bool endwith = true, object fill_value = null) => np.ma.argsort(this, axis, endwith, fill_value);

        /// <summary>Clamps each element into <c>[a_min, a_max]</c>, preserving the mask (NumPy's <c>clip</c>).</summary>
        /// <param name="a_min">Lower bound (array-like/scalar), or null for no lower clip.</param>
        /// <param name="a_max">Upper bound, or null for no upper clip.</param>
        /// <returns>The clamped masked array.</returns>
        public NDMaskedArray clip(object a_min, object a_max) => np.ma.clip(this, a_min, a_max);

        /// <summary>Rounds each element to <paramref name="decimals"/> places (half-to-even), preserving the mask.</summary>
        /// <param name="decimals">Decimal places.</param><returns>The masked, rounded array.</returns>
        public NDMaskedArray round(int decimals = 0) => np.ma.round(this, decimals);

        /// <summary>Complex conjugate (real data passes through); mask preserved (NumPy's <c>conj</c>).</summary>
        /// <returns>A masked array of conj(data).</returns>
        public NDMaskedArray conj() => np.ma.conjugate(this);

        /// <summary>Alias of <see cref="conj"/> (NumPy's <c>conjugate</c>).</summary>
        /// <returns>A masked array of conj(data).</returns>
        public NDMaskedArray conjugate() => np.ma.conjugate(this);

        /// <summary>Dot product with masked slots treated as 0 (NumPy's <c>MaskedArray.dot</c>); with
        /// <paramref name="strict"/> the mask is propagated along the contracted axes first (see
        /// <see cref="MaskedArrayModule.dot(object,object,bool)"/>).</summary>
        /// <param name="b">Right operand.</param>
        /// <param name="strict">Propagate masks along the contracted axes before the product (default false).</param>
        /// <returns>The masked dot product.</returns>
        public NDMaskedArray dot(object b, bool strict = false) => np.ma.dot(this, b, strict);

        /// <summary>Extracts the <paramref name="offset"/>-th diagonal (mask alike); NumPy's <c>MaskedArray.diagonal</c>.</summary>
        /// <param name="offset">Diagonal offset (0 = main).</param><param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <returns>The masked diagonal.</returns>
        public NDMaskedArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) => np.ma.diagonal(this, offset, axis1, axis2);

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
        /// whose element is masked returns the <see cref="MaskedArrayModule.NDMasked"/> constant.</summary>
        /// <returns>The scalar value, or <see cref="MaskedArrayModule.NDMasked"/> when the element is masked.</returns>
        /// <exception cref="InvalidOperationException">The array does not hold exactly one element.</exception>
        public object item()
        {
            if (_data.size != 1)
                throw new InvalidOperationException("can only convert an array of size 1 to a Python scalar");
            // A masked single element reads as the `masked` singleton, matching NumPy's `x.item()`.
            if (_mask is not null && np.any(_mask))
                return np.ma.NDMasked;
            return _data.GetAtIndex(0);
        }

        // ── Instance members that operate on the DATA only (NumPy inherits these straight from ndarray, so
        //    they see through the mask). Each mirrors NumPy's MaskedArray member exactly, including the two
        //    that NumPy itself warns are mask-oblivious (partition/argpartition). ──

        /// <summary>
        ///     Overwrites EVERY data slot with <paramref name="value"/> in place and leaves the mask untouched
        ///     (NumPy's <c>MaskedArray.fill</c>). A masked slot therefore still reads as <c>masked</c> while its
        ///     hidden data becomes <paramref name="value"/> — <c>fill</c> is a data operation, NOT an unmask, so
        ///     it does not clear the mask (unlike an indexer assignment on a soft mask). The mask's hardness is
        ///     irrelevant here (it never gates the data buffer).
        /// </summary>
        /// <param name="value">The scalar written into every data element; cast to the data dtype like NumPy.</param>
        /// <exception cref="System.InvalidOperationException">The underlying data buffer is not writeable
        /// (e.g. a broadcast view), surfaced by the base <see cref="NDArray.fill(object)"/>.</exception>
        public void fill(object value) => _data.fill(value);

        /// <summary>
        ///     Finds the insertion indices that keep the (assumed-sorted) DATA sorted (NumPy's
        ///     <c>MaskedArray.searchsorted</c>, inherited from ndarray). It reads the raw data and IGNORES the
        ///     mask entirely — a masked entry is treated as its underlying value, so the caller is responsible
        ///     for masked slots not distorting the sorted order. Returns a PLAIN <see cref="NDArray"/> (integer
        ///     indices), never a masked array, matching NumPy.
        /// </summary>
        /// <param name="v">The value(s) to insert.</param>
        /// <param name="side">"left" (first suitable index) or "right" (last), as in NumPy.</param>
        /// <param name="sorter">Optional integer index array giving a sort order for the data (an argsort result).</param>
        /// <returns>An integer <see cref="NDArray"/> of insertion indices.</returns>
        public NDArray searchsorted(NDArray v, string side = "left", NDArray sorter = null)
            => _data.searchsorted(v, side, sorter);

        /// <summary>
        ///     Returns the array with each data element's bytes reversed (NumPy's <c>MaskedArray.byteswap</c>),
        ///     carrying the mask through UNCHANGED (a byteswap is a pure data reinterpretation). With
        ///     <paramref name="inplace"/> the existing data buffer is swapped and THIS instance is returned;
        ///     otherwise a fresh masked array over swapped data is returned.
        /// </summary>
        /// <param name="inplace">Swap the data buffer in place and return this instance (true), or return a new
        /// masked array over a swapped copy (false, the default).</param>
        /// <returns>This instance when <paramref name="inplace"/>, else a new <see cref="NDMaskedArray"/> whose data
        /// is byteswapped and whose mask/fill value are carried over (the result starts with a SOFT mask, per this
        /// type's copy-result policy).</returns>
        public NDMaskedArray byteswap(bool inplace = false)
        {
            // In place: mutate the shared data buffer and hand back the same instance (mask + hardmask intact),
            // matching NumPy's `byteswap(inplace=True)` returning self. Otherwise wrap the swapped copy.
            var swapped = _data.byteswap(inplace);
            return inplace ? this : new NDMaskedArray(swapped, _mask, _fill_value);
        }

        /// <summary>
        ///     Partitions the DATA in place around the <paramref name="kth"/>-th element (NumPy's
        ///     <c>MaskedArray.partition</c>). <b>FOOTGUN — this IGNORES the mask</b>: NumPy itself emits a
        ///     <c>UserWarning</c> here and delegates to <c>ndarray.partition</c>, which reorders only the data
        ///     while the mask array stays in its ORIGINAL positions — so after partitioning the mask no longer
        ///     lines up with the values it marked. NumSharp reproduces that behavior (there is no warning channel).
        ///     Prefer <see cref="sort(int,bool,object)"/>, which moves the mask with the data.
        /// </summary>
        /// <param name="kth">Index (or, in the sibling overload, indices) whose element(s) land in final sorted
        /// position; negatives count from the end.</param>
        /// <param name="axis">Axis to partition along; null flattens first. Default last axis.</param>
        /// <param name="kind">Selection algorithm — only "introselect".</param>
        /// <param name="order">Field order (structured dtypes only; unused here).</param>
        public void partition(int kth, int? axis = -1, string kind = "introselect", string order = null)
            => _data.partition(kth, axis, kind, order);

        /// <summary>Multi-<paramref name="kth"/> overload of <see cref="partition(int,int?,string,string)"/> —
        /// same mask-oblivious footgun; each listed element lands in final sorted position.</summary>
        /// <param name="kth">Indices whose elements land in final sorted position.</param>
        /// <param name="axis">Axis to partition along; null flattens first.</param>
        /// <param name="kind">Selection algorithm — only "introselect".</param>
        /// <param name="order">Field order (unused).</param>
        public void partition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
            => _data.partition(kth, axis, kind, order);

        /// <summary>
        ///     Returns the indices that would partition the DATA around the <paramref name="kth"/>-th element
        ///     (NumPy's <c>MaskedArray.argpartition</c>). <b>FOOTGUN — this IGNORES the mask</b> exactly as
        ///     <see cref="partition(int,int?,string,string)"/> does (NumPy warns and delegates to
        ///     <c>ndarray.argpartition</c>): the index permutation is computed from the raw data, and the result
        ///     carries THIS array's mask UNMOVED (positionally), so a masked input position shows as masked in the
        ///     result even though the result holds indices, not values. Reproduced for parity; prefer
        ///     <see cref="argsort(int,bool,object)"/>.
        /// </summary>
        /// <param name="kth">Index whose element lands in final sorted position; negatives count from the end.</param>
        /// <param name="axis">Axis to partition along; null flattens first. Default last axis.</param>
        /// <param name="kind">Selection algorithm — only "introselect".</param>
        /// <param name="order">Field order (unused).</param>
        /// <returns>A masked array of int64 indices carrying this array's mask in its original positions.</returns>
        public NDMaskedArray argpartition(int kth, int? axis = -1, string kind = "introselect", string order = null)
            => new NDMaskedArray(np.argpartition(_data, kth, axis, kind, order), _mask, _fill_value);

        // ── The indexer (NumPy's MaskedArray.__getitem__/__setitem__) — the keystone that makes element/slice
        //    access idiomatic and unblocks put/putmask/mask_rowcols. GET applies the SAME index to data and mask
        //    (a basic slice is a VIEW sharing both buffers; a fancy/boolean index is a COPY); a scalar-reducing
        //    index returns the bare value, or the `masked` singleton when that element is masked. SET writes the
        //    data and reconciles the mask — `= masked` masks in place (data untouched), a plain value UNMASKS the
        //    assigned slots, an NDMaskedArray value PROPAGATES its mask. ──
        /// <summary>
        ///     Indexes the masked array (NumPy's <c>x[key]</c> / <c>x[key] = value</c>). GET returns the bare
        ///     scalar for an index that reduces to a single UNMASKED element, the <see cref="MaskedArrayModule.NDMasked"/>
        ///     singleton for a single MASKED element, or a sub-<see cref="NDMaskedArray"/> otherwise (a VIEW sharing
        ///     memory for a basic slice, a COPY for a fancy/boolean index — matching NumPy). SET accepts a scalar,
        ///     array-like, <see cref="NDArray"/>, <see cref="NDMaskedArray"/>, or the <c>masked</c> singleton
        ///     (which masks the slots without touching their data).
        /// </summary>
        /// <param name="indices">Mixed index objects — ints (coordinate), slice strings ("1:3"), <see cref="Slice"/>s,
        /// boolean/integer <see cref="NDArray"/> masks/fancy indices — exactly as <see cref="NDArray"/>'s own
        /// <c>this[params object[]]</c> accepts.</param>
        /// <returns>A boxed scalar, the <c>masked</c> singleton, or a <see cref="NDMaskedArray"/>. A sub-array
        /// inherits this array's fill value and hard-mask flag (see <see cref="UpdateFrom"/>), so a slice of a
        /// hard-masked array is itself hard and a write through it cannot unmask the parent.</returns>
        /// <remarks>Returns <see cref="object"/> because NumPy's indexer is polymorphic (scalar vs masked vs
        /// sub-array) and a C# indexer cannot switch its type on the runtime index; a slice read is a
        /// <see cref="NDMaskedArray"/> to cast. When a masking write through a VIEW of a <c>nomask</c> array creates
        /// the view's mask, that mask is local to the view: the parent has no mask buffer for it to alias, so the
        /// parent stays unmasked. NumPy 2.4.2 behaves the same way.</remarks>
        public object this[params object[] indices]
        {
            get
            {
                // Scoped by hand, because [NDScoped] cannot take this accessor: the weaver rejects an `object`
                // return (NDW003) and yields a carrier only when it is a struct, so a NDMaskedArray result has no
                // woven egress. The scalar path reads its value out of two 0-D index views and must reclaim both;
                // the sub-array path hands its views back through Yield so they outlive the scope. Yield re-parents
                // only arrays THIS scope tracked (NDScope.Returns ignores the rest), and the parent's own _data and
                // _mask were built before the scope opened, so they are never handed to the caller's scope.
                using var scope = NDScope.Open();
                // Apply the identical index to data and (if present) mask — the two stay aligned by construction.
                var subData = _data[indices];
                var subMask = _mask is null ? null : _mask[indices];
                // Scalar-reducing index (x[0] on 1-D, x[i,j] on 2-D): return the value or the masked singleton.
                if (subData.ndim == 0)
                {
                    if (subMask is not null && np.any(subMask))
                        return np.ma.NDMasked;
                    return subData.GetAtIndex(0);
                }
                // The sub-array inherits this array's fill value AND hard-mask flag, as NumPy's __getitem__ does
                // (`dout._update_from(self)`), for a view and a fancy/boolean copy alike. A soft VIEW would let
                // `x["0:3"][1] = 99` overwrite and unmask the parent's protected slot through the shared buffers.
                return MaskedArrayModule.Yield(scope, new NDMaskedArray(subData, subMask).UpdateFrom(this));
            }
            set => SetItem(indices, value);
        }

        /// <summary>The setter half of the indexer — reconciles the mask with the assigned value per NumPy's
        /// <c>__setitem__</c>: <c>masked</c> masks (data untouched), a plain value unmasks, a masked value
        /// propagates its mask.</summary>
        /// <param name="indices">The index expression.</param>
        /// <param name="value">Scalar/array/NDArray/NDMaskedArray/the masked singleton.</param>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c> (a <c>void</c> method has no egress, so the weaver needs nothing from the
        ///     body): the 0-D True/False scalars, the value normalized from a scalar/array-like, and the hard-mask
        ///     snapshots are all transient and reclaimed on return (exceptions included). Nothing here is handed
        ///     back to the caller's scope. The arrays this write touches in place (<see cref="_data"/>,
        ///     <see cref="_mask"/>, and the parent buffers a slice shares) were built before the scope opened, so the
        ///     scope never tracks them and cannot dispose them. A mask this write has to CREATE
        ///     (<see cref="EnsureMask"/>) is a field store and is detached instead, so it lives as long as this array
        ///     rather than as long as the scope. Under a hard mask the restored mask is written back INTO the existing
        ///     mask buffer (NumPy's <c>_mask[indx] = mindx</c> is in place too), so the mask keeps its identity — no
        ///     replacement array is created and the old one is never orphaned.
        /// </remarks>
        [NDScoped]
        private void SetItem(object[] indices, object value)
        {
            // `x[i] = masked`: set the mask True at the slots, leave the DATA as-is (NumPy hides but keeps it).
            // Masking a slot is ALWAYS allowed, even under a hard mask (hardening only forbids UNmasking).
            if (value is NDMaskedConstant)
            {
                EnsureMask();
                _mask[indices] = NDArray.Scalar(true);
                return;
            }

            // HARD MASK: a data write must not reach a currently-masked slot nor unmask it. Snapshot data+mask,
            // do the ordinary (soft) write below, then restore the originally-masked positions and re-OR any
            // masking the write introduced (a masked-array value's own mask). Snapshotting the WHOLE arrays makes
            // this correct for every index kind — the fancy/boolean SET scatters through the array (unlike the
            // COPY the getter returns), so the write always lands, and the restore is index-agnostic.
            bool hard = _hardmask && _mask is not null;
            NDArray savedMask = hard ? _mask.copy() : null;
            NDArray savedData = hard ? _data.copy() : null;
            // Whether the write below reached the MASK buffer; without such a write the mask still equals its
            // snapshot, so the hard-mask OR-restore would be an identity and is skipped.
            bool maskWritten = false;

            // `x[i] = maskedArray`: write its data, then propagate its mask (or unmask if it has none — soft only).
            if (value is NDMaskedArray mv)
            {
                _data[indices] = mv._data;
                if (mv._mask is not null)
                {
                    EnsureMask();
                    _mask[indices] = mv._mask;
                    maskWritten = true;
                }
                else if (!hard && _mask is not null)
                {
                    _mask[indices] = NDArray.Scalar(false);
                }
            }
            else
            {
                // `x[i] = value` (scalar/array-like/NDArray): write the data and UNMASK the assigned slots (soft).
                // `np.ma.getdata` is the public face of the module's operand normalizer (scalar→0-d, array-like→array).
                _data[indices] = np.ma.getdata(value);
                if (!hard && _mask is not null)
                    _mask[indices] = NDArray.Scalar(false);
            }

            if (hard)
            {
                // NumPy's hard-mask __setitem__ writes DATA only where neither the existing mask nor the value's
                // own mask is set (`copyto(dindx, dval, where=~mindx)` with `mindx = mask_or(_mask[indx], mval)`),
                // so a slot the value MASKS keeps its old (now hidden) data. The mask is therefore settled FIRST —
                // every originally-masked slot kept masked, plus any the write freshly masked, OR'd back into the
                // mask's own buffer — and the data snapshot is then restored wherever that FINAL mask is set.
                // Without a mask write the mask still equals its snapshot, so restoring under it is the plain
                // "never touch a masked slot" rule. (Restoring under the old mask alone left the value's data in
                // the freshly-masked slots: x[0:3] = ma([7,7,7], mask=[0,0,1]) gave data 7 at slot 2, NumPy 3.)
                if (maskWritten)
                    np.logical_or(savedMask, _mask, @out: _mask);
                np.copyto(_data, savedData, casting: "unsafe", where: _mask);
            }
        }

        /// <summary>Promotes a <c>nomask</c> array to a real all-False boolean mask (NumPy's implicit
        /// <c>_mask = make_mask_none(...)</c> on the first masking write). No-op when a mask already exists.</summary>
        /// <remarks>The new mask is a FIELD store, so it is detached from every ambient scope
        /// (<see cref="MaskedArrayModule.Own"/>): it must live as long as this array, not as long as whichever
        /// scope happened to be open when the first masking write arrived.</remarks>
        internal void EnsureMask()
        {
            if (_mask is null)
                // DIMENSIONS only, never _data.Shape: a strided/offset _data view carries strides + a
                // larger bufferSize, and np.zeros(that Shape) builds a strided zeros over the base buffer
                // whose materialized tail is garbage (see MaskedArrayModule.DimsOf). Size the mask by dims.
                _mask = MaskedArrayModule.Own(np.zeros(Shape.Adopt(_data.Shape.dimensions), np.@bool));
        }

        /// <summary>Sets <c>x.flat[indices] = values</c> (mask-aware) — the instance form of
        /// <see cref="MaskedArrayModule.put"/>. Masked <paramref name="values"/> mask the slots; plain values unmask.</summary>
        /// <param name="indices">Flat integer indices.</param><param name="values">Scalar/array/masked values.</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default)/"wrap"/"clip".</param>
        public void put(NDArray indices, object values, string mode = "raise") => np.ma.put(this, indices, values, mode);

        /// <summary>Makes the mask HARD (NumPy's <c>harden_mask</c>): from now on a plain-value assignment cannot
        /// unmask a slot — only currently-unmasked positions change. Mutates and returns THIS array.</summary>
        /// <returns>This masked array (now hard-masked).</returns>
        /// <remarks>Only THIS instance changes. A slice taken earlier copied the flag when it was created (see
        /// <see cref="UpdateFrom"/>), so it stays soft, and a write through it can still unmask this array's
        /// slots, exactly as in NumPy. Harden first, then slice.</remarks>
        public NDMaskedArray harden_mask() { _hardmask = true; return this; }

        /// <summary>Makes the mask SOFT again (NumPy's <c>soften_mask</c>, the default): a plain-value assignment
        /// unmasks the assigned slots. Mutates and returns THIS array.</summary>
        /// <returns>This masked array (now soft-masked).</returns>
        /// <remarks>Only THIS instance changes: softening a slice leaves its parent hard (and the reverse), as
        /// in NumPy, where each array keeps its own copy of the flag.</remarks>
        public NDMaskedArray soften_mask() { _hardmask = false; return this; }

        /// <summary>Drops an all-False mask back to nomask IN PLACE (NumPy's <c>MaskedArray.shrink_mask</c>:
        /// <c>self._mask = _shrink_mask(self._mask); return self</c>); a mask with a masked slot is kept.</summary>
        /// <returns>This array.</returns>
        public NDMaskedArray shrink_mask()
        {
            // `is not null` (reference check) — `_mask != null` would run NDArray's ELEMENTWISE `!=`. The dropped mask is
            // not disposed: like NumPy's, it may be shared with the array it was adopted from.
            if (_mask is not null && !np.any(_mask))
                _mask = null;
            return this;
        }

        // ── Conversion / export + the hard/shared-mask flag surface (NumPy's tolist/tobytes/tofile/view +
        //    hardmask/sharedmask/recordmask/baseclass/unshare_mask/ids/iscontiguous). tolist and tobytes are
        //    genuine value exports; view reinterprets the data dtype. hardmask reports the REAL hard/soft
        //    state (see _hardmask); sharedmask is not offered (NumSharp does not track mask sharing); every
        //    array is non-structured, so recordmask is the plain mask. ──

        /// <summary>
        ///     Returns the array as a nested list structure (NumPy's <c>MaskedArray.tolist</c>): a boxed scalar
        ///     for a 0-D array, otherwise a jagged <see cref="object"/>[] whose leaves are the boxed CLR element
        ///     values. Masked positions become <paramref name="fill_value"/>, or <c>null</c> when it is null —
        ///     NumPy fills masked entries with <c>None</c> by default — so the result is the "data with holes"
        ///     view a non-NumSharp consumer wants.
        /// </summary>
        /// <param name="fill_value">Value substituted at masked positions; null ⇒ <c>null</c> in the output
        /// (NumPy's <c>None</c>). When non-null the array is <see cref="filled(object)"/> first, so no nulls
        /// appear and the value is cast into the data dtype exactly as NumPy coerces it.</param>
        /// <returns>The bare boxed scalar for a 0-D array, else a nested <see cref="object"/>[] (one level per
        /// dimension); leaves are boxed element values, or <c>null</c>/<paramref name="fill_value"/> where masked.</returns>
        public object tolist(object fill_value = null)
        {
            // No mask ⇒ the plain data's nested list (no holes to punch).
            if (_mask is null)
                return BuildNestedList(_data, null);
            // Explicit fill ⇒ bake it in and list the filled data (still no nulls, matching NumPy). The filled copy
            // only feeds the boxed-leaf walk — nothing in the returned object graph references it — so it is
            // released as soon as the list is built.
            if (fill_value is not null)
            {
                using var filledData = filled(fill_value);
                return BuildNestedList(filledData, null);
            }
            // Default ⇒ null at masked positions (NumPy's None).
            return BuildNestedList(_data, _mask);
        }

        /// <summary>Materializes <paramref name="data"/> into a boxed scalar (0-D) or nested <see cref="object"/>[]
        /// (rank ≥ 1), emitting <c>null</c> wherever <paramref name="mask"/> is True.</summary>
        /// <param name="data">The data array to convert.</param>
        /// <param name="mask">The boolean mask (True ⇒ emit null), or null for no holes.</param>
        /// <returns>The nested representation (bare boxed scalar for a 0-D input).</returns>
        private static object BuildNestedList(NDArray data, NDArray mask)
        {
            long flat = 0;
            return BuildNestedRec(data, mask, data.Shape.dimensions, 0, ref flat);
        }

        /// <summary>Recursion behind <see cref="BuildNestedList"/>: <paramref name="dim"/> walks the axes while
        /// <paramref name="flat"/> is the running C-order element counter, read through
        /// <see cref="NDArray.GetAtIndex(long)"/> (which transforms the logical index through the shape's
        /// strides/offset, so any layout — strided/transposed/reversed — is handled with no prior copy).</summary>
        /// <param name="data">The data array.</param>
        /// <param name="mask">The boolean mask, or null.</param>
        /// <param name="shape">The data shape (axis extents).</param>
        /// <param name="dim">The current axis (0 = outermost); equals <c>shape.Length</c> at a scalar leaf.</param>
        /// <param name="flat">The running logical C-order index, advanced once per leaf.</param>
        /// <returns>The boxed leaf value/null, or an <see cref="object"/>[] for this axis.</returns>
        private static object BuildNestedRec(NDArray data, NDArray mask, long[] shape, int dim, ref long flat)
        {
            if (dim == shape.Length) // scalar leaf: null when masked, else the boxed element
            {
                object v = (mask is not null && mask.GetAtIndex<bool>(flat)) ? null : data.GetAtIndex(flat);
                flat++;
                return v;
            }
            var arr = new object[shape[dim]];
            for (long i = 0; i < shape[dim]; i++)
                arr[i] = BuildNestedRec(data, mask, shape, dim + 1, ref flat);
            return arr;
        }

        /// <summary>
        ///     Returns the raw bytes of the array (NumPy's <c>MaskedArray.tobytes</c>) — the array is
        ///     <see cref="filled(object)"/> FIRST, so masked slots carry the fill sentinel and the mask itself is
        ///     NOT encoded (as NumPy documents, shape/dtype/fill_value are all lost).
        /// </summary>
        /// <param name="fill_value">Sentinel written at masked positions before encoding; null uses this array's
        /// <see cref="fill_value"/> (the dtype default when unset).</param>
        /// <param name="order">Byte order of the copy: 'C' (row-major, default), 'F' (column-major), 'A'/'K'
        /// (resolved via the C-contiguous filled copy, so they equal 'C' here — a masked-array nuance vs NumPy,
        /// which can keep 'F' for an F-contiguous source under 'A').</param>
        /// <returns>A fresh <see cref="byte"/>[] of length <c>size * itemsize</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="order"/> is not one of C/F/A/K.</exception>
        public byte[] tobytes(object fill_value = null, char order = 'C')
        {
            // filled() always returns a FRESH copy this call owns; the bytes are copied out of it, so the copy is
            // released on the way out (on a bad `order` throw too) rather than left to the finalizer.
            using var filledData = filled(fill_value);
            return filledData.tobytes(order);
        }

        /// <summary>
        ///     NOT IMPLEMENTED — raises exactly as NumPy's <c>MaskedArray.tofile</c> does. A masked array cannot
        ///     be written to a binary file without losing the mask, so NumPy declines; the parameters exist only
        ///     to mirror its signature. Use <see cref="tobytes(object,char)"/> (with a fill value) to serialize.
        /// </summary>
        /// <param name="fid">Ignored (NumPy's file id).</param>
        /// <param name="sep">Ignored (NumPy's separator).</param>
        /// <param name="format">Ignored (NumPy's format).</param>
        /// <exception cref="NotImplementedException">Always — matching NumPy's <c>NotImplementedError</c>.</exception>
        public void tofile(string fid, string sep = "", string format = "%s")
            => throw new NotImplementedException("MaskedArray.tofile() not implemented yet.");

        /// <summary>
        ///     Reinterprets the data buffer as another dtype (NumPy's <c>MaskedArray.view</c>) — a VIEW sharing
        ///     memory, writes through. A same-itemsize dtype (e.g. int32↔uint32↔float32) or the no-dtype call
        ///     keeps the shape, so the boolean mask still aligns and rides through unchanged.
        /// </summary>
        /// <param name="dtype">Target dtype; null ⇒ a same-dtype alias. Any dtype — the same one too — resets a stored fill
        ///     value to the dtype default (NumPy's <c>view</c>).</param>
        /// <param name="fill_value">A new fill value for the view (NumPy's <c>fill_value=</c>); it replaces a STORED fill
        ///     only — on an array whose fill was never set or read it is ignored, as in NumPy.</param>
        /// <returns>A masked array over the reinterpreted data, with this array's hard-mask flag; the mask is carried as a
        /// view when the shape is preserved, and is absent (nomask) for an itemsize-changing view of an already-unmasked
        /// array.</returns>
        /// <exception cref="NotSupportedException">The target dtype has a different itemsize AND this array has an
        /// active mask — a boolean (itemsize-1) mask has no sound reinterpretation to the new element count, so
        /// the view is refused rather than silently dropping or misaligning it (fill or drop the mask first).</exception>
        public NDMaskedArray view(DType dtype = null, object fill_value = null)
        {
            // Scoped by hand: the reinterpreted view is built BEFORE the mask check that may refuse it, so the
            // refusal path must reclaim it; the success paths hand it back through Yield.
            using var scope = NDScope.Open();
            var nd = _data.view(dtype);
            NDMaskedArray r;
            // Same element count ⇒ shape preserved (no-op alias or same-itemsize reinterpret): the bool mask
            // still lines up element-for-element, so carry it as a VIEW (`output._mask = output._mask.view()`: the same
            // memory, so a masked write through either array reaches the other, as NumPy's do).
            if (nd.size == _data.size)
                r = new NDMaskedArray(nd, _mask?.view(), _fill_value);
            // Itemsize changed ⇒ the trailing axis was rescaled. With no mask the reinterpreted data stands
            // alone; with a live mask there is no meaningful reinterpretation of the boolean mask, so refuse.
            else if (_mask is null)
                r = new NDMaskedArray(nd, null, _fill_value);
            else
                throw new NotSupportedException(
                    "view() to a dtype of a different itemsize is not supported on a masked array with an active " +
                    "mask: the boolean mask cannot be reinterpreted to the new element count. Fill or drop the mask first.");
            // `__array_finalize__` -> `_update_from(self)`: the hard-mask flag carries over.
            r._hardmask = _hardmask;
            // `if output._fill_value is not None: if fill_value is None: if dtype is not None: output._fill_value = None
            //  else: output.fill_value = fill_value` — a stored fill is RESET by any dtype argument (even the same dtype)
            // and replaced by a given fill_value; an array with no stored fill ignores both.
            if (r._fill_value is not null)
            {
                if (fill_value is not null)
                    r.fill_value = fill_value;
                else if (dtype is not null)
                    r._fill_value = null;
            }
            return MaskedArrayModule.Yield(scope, r);
        }

        /// <summary>The addresses of the data and mask buffers (NumPy's <c>MaskedArray.ids</c>); the mask address
        /// is 0 when there is no mask (NumPy returns <c>id(nomask)</c> there — an arbitrary sentinel either way).</summary>
        /// <returns>A <c>(data, mask)</c> tuple of buffer addresses as <see cref="long"/>s.</returns>
        public (long data, long mask) ids() => np.ma.ids(this);

        /// <summary>Whether the data is C-contiguous (NumPy's <c>MaskedArray.iscontiguous</c> — the CONTIGUOUS
        /// flag), an O(1) flag read.</summary>
        /// <returns>True iff the data laid out row-major-contiguous.</returns>
        public bool iscontiguous() => _data.Shape.IsContiguous;

        /// <summary>IN-PLACE conditional write (the instance form of <see cref="MaskedArrayModule.putmask"/>):
        /// writes <paramref name="values"/> where <paramref name="mask"/> is True and reconciles the mask there
        /// (masked values mask the slots, plain values unmask them).</summary>
        /// <param name="mask">Boolean array-like selecting the write positions.</param>
        /// <param name="values">Scalar/array/<see cref="NDMaskedArray"/> supplying the data (and, if masked, the mask).</param>
        public void putmask(object mask, object values) => np.ma.putmask(this, mask, values);

        /// <summary>
        ///     NOT SUPPORTED in place — raises exactly as NumPy's <c>MaskedArray.resize</c> does: a masked array
        ///     does not own its data, so it cannot be resized in place. Use the module function
        ///     <see cref="MaskedArrayModule.resize(object,Shape)"/> (which returns a NEW tiled masked array).
        /// </summary>
        /// <param name="new_shape">Ignored (NumPy's target shape).</param>
        /// <param name="refcheck">Ignored (NumPy's reference-count check flag).</param>
        /// <exception cref="NotSupportedException">Always — NumPy raises <c>ValueError</c> with this verbatim text
        /// (the .NET exception type differs by the house convention; the message is byte-identical).</exception>
        public void resize(Shape new_shape, bool refcheck = true) => throw new NotSupportedException(
            "A masked array does not own its data and therefore cannot be resized.\n" +
            "Use the numpy.ma.resize function instead.");

        /// <summary>
        ///     A flattened row-major VIEW of the array (mask flattened alike). NOTE: this follows NumSharp's
        ///     <see cref="NDArray.flat"/> house convention — a raveled ARRAY (a view for a contiguous source, a
        ///     copy otherwise) — NOT NumPy's <c>flat</c> iterator object. It writes through only when
        ///     <see cref="ravel(char)"/> returned a view (a C-contiguous source).
        /// </summary>
        public NDMaskedArray flat => np.ma.ravel(this, 'C');

        /// <summary>Whether the mask is HARD (NumPy's read-only <c>hardmask</c> property): true once
        /// <see cref="harden_mask()"/> has been called, so a plain-value assignment can no longer unmask a slot;
        /// false (the default) after <see cref="soften_mask()"/>. See <see cref="_hardmask"/> for the honored
        /// assignment paths and the propagation-scope limits.</summary>
        /// <remarks>NumPy's <c>sharedmask</c> flag is deliberately NOT offered: NumSharp cannot track whether a
        /// mask is shared (a view/<see cref="real"/>/<see cref="imag"/>/<see cref="view(DType)"/> aliases it), and
        /// no fixed value is faithful — NumPy reports True right after <c>array(data, mask=…)</c> yet False after
        /// an op or <see cref="unshare_mask()"/>. Since a bogus False would invite mutating a shared mask, just
        /// call <see cref="unshare_mask()"/> before an in-place mask write on a possibly-aliased array.</remarks>
        public bool hardmask => _hardmask;

        /// <summary>The record mask (NumPy's <c>MaskedArray.recordmask</c>): for a non-structured array — every
        /// array in NumSharp, which has no structured dtypes — this is just the element mask (the
        /// <see cref="MaskedArrayModule.nomask"/> sentinel when nothing is masked).</summary>
        public NDArray recordmask => mask;

        /// <summary>The class of the underlying data (NumPy's read-only <c>MaskedArray.baseclass</c>); always
        /// <see cref="NDArray"/> in NumSharp, whose masked array wraps a plain <see cref="NDArray"/>.</summary>
        public Type baseclass => typeof(NDArray);

        /// <summary>
        ///     Ensures the mask is not shared with any other array (NumPy's <c>unshare_mask</c>) by COPYING it,
        ///     then returns this array. Reach for it before a mutating write when the mask may be aliased — the
        ///     view-returning members (<see cref="real"/>/<see cref="imag"/>/<see cref="view(DType)"/> and a
        ///     basic-slice index) share the mask reference, so an in-place mask write would otherwise reach the
        ///     source. A no-op for a <c>nomask</c> array.
        /// </summary>
        /// <returns>This masked array (with an owned mask).</returns>
        /// <remarks>
        ///     NumSharp cannot track whether a mask is actually shared (there is no <c>sharedmask</c> state),
        ///     so — unlike NumPy, which copies only WHEN shared — this copies unconditionally when a mask is present;
        ///     the end state (an unshared mask) is identical, only a redundant copy of an already-owned mask differs.
        ///     <para>
        ///     <b>Ownership.</b> The new copy is a field store: it is detached from every ambient scope
        ///     (<see cref="MaskedArrayModule.Own"/>) so it lives as long as this array — a scope open around the call
        ///     must not dispose the mask this array now holds. The SUPERSEDED mask is deliberately NOT disposed: the
        ///     whole point of unsharing is that other arrays may still read it, so it stays with whoever created it
        ///     (released when they dispose it, else by the finalizer). A caller that knows it was the only holder can
        ///     read <c>mask</c> first and dispose that buffer itself after the call.
        ///     </para>
        /// </remarks>
        public NDMaskedArray unshare_mask()
        {
            if (_mask is not null)
                _mask = MaskedArrayModule.Own(_mask.copy());
            return this;
        }

        // ── Bitwise / modulo operators. MaskedArray defines none of %, &, |, ^ (only +, -, *, /, //, ** and the
        //    comparisons), so NumPy runs ndarray's operator — the PLAIN ufunc with the operands as given (weak C#
        //    numbers) — and MaskedArray.__array_wrap__ wraps it: no fill-back, the masks mask_or-ed (shrunk to nomask),
        //    remainder's safe-divide domain filled with 1 and masked, the first masked operand's fill value and hard mask
        //    (MaskedArrayModule.UfuncOperator). The np.ma.remainder / bitwise_* FUNCTIONS keep numpy.ma's own ufunc
        //    wrappers (fill-back of the left operand's data). ──
        /// <summary>Masked element-wise modulo (division-by-zero additionally masked).</summary>
        /// <param name="a">Numerator.</param><param name="b">Divisor.</param><returns>a%b, masked.</returns>
        public static NDMaskedArray operator %(NDMaskedArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.remainder(x, y), remainderDomain: true);
        /// <summary>Masked modulo with a scalar/array-like divisor.</summary>
        /// <param name="a">Numerator masked array.</param><param name="b">Divisor scalar/array-like.</param><returns>a%b, masked.</returns>
        public static NDMaskedArray operator %(NDMaskedArray a, object b) => np.ma.UfuncOperator(a, b, (x, y) => np.remainder(x, y), remainderDomain: true);
        /// <summary>Masked modulo with a scalar/array-like numerator.</summary>
        /// <param name="a">Numerator scalar/array-like.</param><param name="b">Divisor masked array.</param><returns>a%b, masked.</returns>
        public static NDMaskedArray operator %(object a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.remainder(x, y), remainderDomain: true);
        /// <summary>Masked modulo with an NDArray divisor.</summary>
        /// <param name="a">Numerator.</param><param name="b">Divisor NDArray.</param><returns>a%b, masked.</returns>
        public static NDMaskedArray operator %(NDMaskedArray a, NDArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.remainder(x, y), remainderDomain: true);
        /// <summary>Masked modulo with an NDArray numerator.</summary>
        /// <param name="a">Numerator NDArray.</param><param name="b">Divisor masked array.</param><returns>a%b, masked.</returns>
        public static NDMaskedArray operator %(NDArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.remainder(x, y), remainderDomain: true);

        /// <summary>Masked bitwise AND (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a&amp;b, masked.</returns>
        public static NDMaskedArray operator &(NDMaskedArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_and(x, y));
        /// <summary>Masked bitwise AND with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a&amp;b, masked.</returns>
        public static NDMaskedArray operator &(NDMaskedArray a, object b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_and(x, y));
        /// <summary>Masked bitwise AND with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a&amp;b, masked.</returns>
        public static NDMaskedArray operator &(object a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_and(x, y));
        /// <summary>Masked bitwise AND with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a&amp;b, masked.</returns>
        public static NDMaskedArray operator &(NDMaskedArray a, NDArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_and(x, y));
        /// <summary>Masked bitwise AND with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a&amp;b, masked.</returns>
        public static NDMaskedArray operator &(NDArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_and(x, y));

        /// <summary>Masked bitwise OR (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a|b, masked.</returns>
        public static NDMaskedArray operator |(NDMaskedArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_or(x, y));
        /// <summary>Masked bitwise OR with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a|b, masked.</returns>
        public static NDMaskedArray operator |(NDMaskedArray a, object b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_or(x, y));
        /// <summary>Masked bitwise OR with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a|b, masked.</returns>
        public static NDMaskedArray operator |(object a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_or(x, y));
        /// <summary>Masked bitwise OR with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a|b, masked.</returns>
        public static NDMaskedArray operator |(NDMaskedArray a, NDArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_or(x, y));
        /// <summary>Masked bitwise OR with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a|b, masked.</returns>
        public static NDMaskedArray operator |(NDArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_or(x, y));

        /// <summary>Masked bitwise XOR (mask = OR of operands').</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>a^b, masked.</returns>
        public static NDMaskedArray operator ^(NDMaskedArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_xor(x, y));
        /// <summary>Masked bitwise XOR with a scalar/array-like.</summary>
        /// <param name="a">Left masked array.</param><param name="b">Right scalar/array-like.</param><returns>a^b, masked.</returns>
        public static NDMaskedArray operator ^(NDMaskedArray a, object b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_xor(x, y));
        /// <summary>Masked bitwise XOR with a scalar/array-like left operand.</summary>
        /// <param name="a">Left scalar/array-like.</param><param name="b">Right masked array.</param><returns>a^b, masked.</returns>
        public static NDMaskedArray operator ^(object a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_xor(x, y));
        /// <summary>Masked bitwise XOR with an NDArray operand.</summary>
        /// <param name="a">Left.</param><param name="b">Right NDArray.</param><returns>a^b, masked.</returns>
        public static NDMaskedArray operator ^(NDMaskedArray a, NDArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_xor(x, y));
        /// <summary>Masked bitwise XOR with an NDArray left operand.</summary>
        /// <param name="a">Left NDArray.</param><param name="b">Right masked array.</param><returns>a^b, masked.</returns>
        public static NDMaskedArray operator ^(NDArray a, NDMaskedArray b) => np.ma.UfuncOperator(a, b, (x, y) => np.bitwise_xor(x, y));

        /// <summary>Bitwise NOT / invert, NumPy's <c>~x</c>: ndarray's <c>__invert__</c> runs <c>np.bitwise_not</c> on the
        /// data and <c>__array_wrap__</c> wraps it (a masked slot holds the inverted data, the mask is the operand's own
        /// mask object, the fill value and hard mask carry over). NumPy has no <c>np.ma.invert</c> module function.</summary>
        /// <param name="a">Operand.</param><returns>~a, masked.</returns>
        /// <exception cref="TypeError">A float / complex / text operand (np.invert's refusal).</exception>
        public static NDMaskedArray operator ~(NDMaskedArray a) => np.ma.Invert(a);
    }

    /// <summary>
    ///     NumPy's <c>numpy.ma.masked</c> singleton: the scalar constant that represents "a single masked
    ///     value". A masked ufunc that reduces to one fully-masked 0-D element returns this instance, matching
    ///     NumPy where <c>np.ma.add(masked, x)</c> and a masked 0-D result are <c>masked</c>.
    /// </summary>
    /// <remarks>
    ///     Exactly ONE instance exists (the constructor is internal), reachable under four names that are all the
    ///     same object: <see cref="MaskedArrayModule.NDMasked"/> / <see cref="MaskedArrayModule.NDMaskedSingleton"/>
    ///     (NumSharp's <c>ND*</c> spelling) and <see cref="MaskedArrayModule.masked"/> /
    ///     <see cref="MaskedArrayModule.masked_singleton"/> (NumPy's names). Because it is a singleton, reference
    ///     identity (<c>ReferenceEquals(r, np.ma.masked)</c>) is a valid "is this the masked scalar?" test — the
    ///     C# counterpart of NumPy's <c>r is np.ma.masked</c>.
    /// </remarks>
    public sealed class NDMaskedConstant : NDMaskedArray
    {
        /// <summary>Builds the constant as a 0-D <c>0.0</c> datum under a 0-D <c>True</c> mask, i.e. one masked
        /// scalar — the same internal shape NumPy's <c>masked_singleton</c> carries.</summary>
        /// <remarks>Both scalars are detached from every scope (<see cref="MaskedArrayModule.Own"/>): the single
        /// instance is built during <c>np</c>'s static initialization, which may run inside a caller's open
        /// <see cref="NDScope"/>; left tracked, that scope would dispose the singleton's buffers on close.</remarks>
        internal NDMaskedConstant() : base(MaskedArrayModule.Own(NDArray.Scalar(0.0d)), MaskedArrayModule.Own(NDArray.Scalar(true))) { }

        /// <summary>Renders as the bare token <c>masked</c>, as NumPy prints its singleton.</summary>
        /// <returns>The string <c>"masked"</c>.</returns>
        public override string ToString() => "masked";
    }

    /// <summary>
    ///     NumPy's <c>numpy.ma.masked_print_option</c>: the mutable, process-global object controlling how masked
    ///     elements are DISPLAYED and whether that display happens at all. Reach for it to change the two-character
    ///     <c>--</c> placeholder (<see cref="set_display"/>) or to turn the placeholder off entirely
    ///     (<see cref="enable"/>) so masked slots print their FILL value instead — the two ways NumPy lets you tune
    ///     a masked repr globally.
    ///     <para>
    ///     It is a single shared instance behind <see cref="MaskedArrayModule.masked_print_option"/> — mutating it
    ///     (like NumPy's module global) changes every subsequent <see cref="NDMaskedArray.ToString()"/>, so it is a
    ///     global side effect: set it back if you only meant a local change.
    ///     </para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/maskedarray.baseclass.html</remarks>
    public sealed class MaskedPrintOption
    {
        /// <summary>The token shown at masked positions; NumPy's default is the two-char <c>--</c>.</summary>
        private string _display;

        /// <summary>Whether masked slots are shown as <see cref="_display"/> (true) or the fill value (false).
        /// NumPy's default is true.</summary>
        private bool _enabled = true;

        /// <summary>Builds the option with an initial display token (the module seeds it with <c>--</c>).</summary>
        /// <param name="display">Initial masked-slot token; null falls back to <c>--</c>.</param>
        internal MaskedPrintOption(string display) => _display = display ?? "--";

        /// <summary>The current masked-slot token (NumPy's <c>display()</c>).</summary>
        /// <returns>The display string, e.g. <c>--</c>.</returns>
        public string display() => _display;

        /// <summary>Replaces the masked-slot token (NumPy's <c>set_display</c>). Affects all later masked prints —
        /// a GLOBAL change, since there is one shared option object.</summary>
        /// <param name="s">The new token (any string; NumPy does not validate it).</param>
        public void set_display(string s) => _display = s;

        /// <summary>Whether the masked-slot substitution is active (NumPy's <c>enabled()</c>). When false, a masked
        /// print substitutes the array's FILL value at masked slots and prints numeric-aligned (the
        /// <c>filled(fill_value)</c> path) instead of showing <see cref="display"/>.</summary>
        /// <returns>True when masked slots render as <see cref="display"/>.</returns>
        public bool enabled() => _enabled;

        /// <summary>Turns the masked-slot substitution on or off (NumPy's <c>enable(shrink=1)</c>; the parameter is
        /// named <c>shrink</c> in NumPy but simply sets the enabled flag).</summary>
        /// <param name="shrink">True enables the <c>--</c> substitution (default); false makes masked slots print
        /// their fill value.</param>
        public void enable(bool shrink = true) => _enabled = shrink;

        /// <summary>Renders as the display token, matching NumPy's <c>str(masked_print_option)</c>.</summary>
        /// <returns>The current display string.</returns>
        public override string ToString() => _display;
    }

    /// <summary>
    ///     The masked counterpart of <see cref="np.r_"/> — NumPy's <c>numpy.ma.mr_</c> (an <c>MAxisConcatenator</c>):
    ///     concatenates masked arrays / slices / scalars along the first axis, propagating the mask, with the SAME
    ///     slice-expression grammar as <see cref="np.r_"/> (a colon-string is a slice, a leading directive string
    ///     sets axis/ndmin/matrix — the library-wide convention). Reach for it to stitch masked pieces together
    ///     the way <c>np.r_</c> stitches plain arrays.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Not structured-blocked.</b> The audit long listed <c>mr_</c> under the structured/record family;
    ///     that was wrong. NumPy's <c>MAxisConcatenator</c> is structured only in its de-activated matrix-string
    ///     path — its real job is masked concatenation, which composes cleanly here.
    ///     </para>
    ///     <para>
    ///     <b>How it works.</b> The result DATA is exactly <c>np.r_[key]</c> over the same entries (a masked entry
    ///     contributes its <c>data</c>); the result MASK is <c>np.r_[maskKey]</c> where each entry is replaced by
    ///     its boolean-mask contribution of the SAME natural shape (a masked entry → its full mask, a plain
    ///     array/scalar → all-False, a colon-slice → all-False of the arange's shape, a directive string passes
    ///     through). Because both streams run the identical <c>np.r_</c> grammar, every directive (axis / ndmin /
    ///     <c>r</c>/<c>c</c> matrix) lays the mask out to match the data. When NO entry carries a mask the result
    ///     is <c>nomask</c> (the fast path), exactly as the sibling stacking wrappers do.
    ///     </para>
    ///     <para>
    ///     <b>Lone string → <see cref="MAError"/>.</b> NumPy's <c>mr_</c> rejects a lone string key (its dangerous
    ///     frame-based matrix builder) with <c>MAError("Unavailable for masked array.")</c>; reproduced verbatim.
    ///     A lone masked array / scalar, or any multi-entry key, is fine.
    ///     </para>
    ///     <para>
    ///     <b>One documented divergence.</b> A colon-slice entry combined with a <c>trans1d</c> directive
    ///     (<c>"axis,ndmin,trans1d"</c> with <c>trans1d != -1</c>) lays the DATA out via <c>np.r_</c>'s slice
    ///     branch but the MASK via its array branch, whose axis permutations differ only in that rare case; every
    ///     realistic usage (concatenation, scalars, axis and matrix directives, ndmin with the default
    ///     <c>trans1d=-1</c>) is bit-exact.
    ///     </para>
    ///     https://numpy.org/doc/stable/reference/generated/numpy.ma.mr_.html
    /// </remarks>
    public sealed class MrClass
    {
        /// <summary>There is one shared instance behind <see cref="MaskedArrayModule.mr_"/>.</summary>
        internal MrClass() { }

        /// <summary>
        ///     Expands and concatenates the masked index expression along the first axis.
        /// </summary>
        /// <param name="key">Masked arrays, plain arrays, scalars, slice-expression strings and leading
        /// directive strings — in any mix, following <see cref="np.r_"/>'s grammar.</param>
        /// <returns>A <see cref="NDMaskedArray"/> whose data is <c>np.r_[key]</c> and whose mask is the matching
        /// concatenation of the entries' masks (or <see cref="MaskedArrayModule.nomask"/> when none is masked).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> (or an entry) is null.</exception>
        /// <exception cref="MAError"><paramref name="key"/> is a lone string (NumPy's rejection).</exception>
        public NDMaskedArray this[params object[] key]
        {
            get
            {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                // NumPy's mr_ refuses a LONE string (its frame-based matrix-builder syntax). NumSharp has no such
                // builder, so a lone string could only be a would-be directive or the colon-slice spelling — reject
                // it exactly as NumPy does. Every OTHER form follows np.r_'s grammar.
                if (key.Length == 1 && key[0] is string)
                    throw new MAError("Unavailable for masked array.");

                // Scoped by hand (a masked-array return the weaver cannot yield): the per-entry mask stand-ins —
                // all-False arrays for plain/slice entries, the throwaway r_ probe that sizes a slice's stand-in —
                // only feed the concatenation and are reclaimed on return.
                using var scope = NDScope.Open();

                // DATA stream: unwrap a masked entry to its data; every other entry (slice/directive strings,
                // scalars, arrays) passes straight to np.r_ so its dtype/layout resolution is untouched.
                var dataKey = new object[key.Length];
                bool anyMask = false;
                for (int i = 0; i < key.Length; i++)
                {
                    if (key[i] is NDMaskedArray m)
                    {
                        dataKey[i] = m._data;
                        // A real (non-nomask) mask on ANY entry forces the mask stream; else stay on nomask.
                        if (m._mask is not null)
                            anyMask = true;
                    }
                    else
                        dataKey[i] = key[i];
                }
                NDArray dataResult = np.r_[dataKey];
                if (!anyMask)
                    return MaskedArrayModule.Yield(scope, new NDMaskedArray(dataResult, null));

                // MASK stream: same np.r_ grammar over each entry's boolean-mask contribution.
                var maskKey = new object[key.Length];
                for (int i = 0; i < key.Length; i++)
                    maskKey[i] = MaskEntry(key[i]);
                NDArray maskResult = np.r_[maskKey];
                return MaskedArrayModule.Yield(scope, new NDMaskedArray(dataResult, maskResult));
            }
        }

        /// <summary>Maps a key entry to its boolean-mask contribution of the SAME natural shape it feeds the data
        /// stream: a directive string passes through unchanged (it lays out both streams identically), a
        /// colon-slice materializes an all-False array of the arange's shape, and everything else
        /// (masked array / plain array / scalar) routes through <see cref="MaskedArrayModule.getmaskarray"/>,
        /// which yields the mask for a masked array and an all-False array for a plain one.</summary>
        /// <param name="item">A key entry.</param>
        /// <returns>The mask-stream stand-in for <paramref name="item"/>.</returns>
        private static object MaskEntry(object item)
        {
            switch (item)
            {
                case null:
                    throw new ArgumentNullException(nameof(item), "index expression entries must not be null.");
                // A directive string (no colon) is a layout control token — it must pass through so the mask
                // stream gets the same axis/ndmin/matrix directive as the data stream.
                case string s when s.IndexOf(':') < 0:
                    return s;
                // A colon-slice → all-False of the arange's own shape (np.r_ re-applies any ndmin directive to it).
                case string:
                case Slice:
                case Slice[]:
                    // DIMENSIONS only (see MaskedArrayModule.DimsOf): a non-contiguous r_ result Shape
                    // would make np.zeros build a garbage strided mask; size by dims instead.
                    return np.zeros(Shape.Adopt(np.r_[item].Shape.dimensions), np.@bool);
                // Masked array → its mask; plain array/scalar → an all-False array of its shape.
                default:
                    return np.ma.getmaskarray(item);
            }
        }
    }

    /// <summary>
    ///     The <c>numpy.ma</c> module surface, reachable as <see cref="np.ma"/>. Holds the masked-array
    ///     substrate (<see cref="getdata"/>/<see cref="getmask"/>/<see cref="getmaskarray"/>/<see cref="filled"/>/
    ///     <see cref="NDMasked"/>/<see cref="nomask"/>) and the masked <b>ufunc family</b> — every unary/binary
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

        /// <summary>NumPy's <c>nomask</c>: the shared 0-D <c>False</c> that <see cref="NDMaskedArray.mask"/>
        /// returns when nothing is masked. Internally "no mask" is stored as a null mask; this is only the
        /// public face of that state.</summary>
        /// <remarks>Detached from every scope at construction (<see cref="Own"/>): this module is built by
        /// <c>np</c>'s static constructor, which runs wherever <c>np</c> is FIRST touched — possibly inside a
        /// caller's open <see cref="NDScope"/>, which would otherwise track this process-wide singleton and dispose
        /// it when that scope closed (probed: every later read was a use-after-dispose).</remarks>
        public NDArray nomask { get; } = Own(NDArray.Scalar(false));

        /// <summary>
        ///     The masked constant in NumSharp's <c>ND*</c> spelling: the ONE <see cref="NDMaskedConstant"/> instance
        ///     standing for a single masked value (a 0-D float64 <c>0.0</c> datum under a 0-D <c>True</c> mask).
        ///     Ufuncs and reductions whose result is a single fully-masked 0-D element return THIS object, so test
        ///     for it by reference (<c>ReferenceEquals(r, np.ma.NDMasked)</c>) or with <see cref="is_masked"/>;
        ///     assigning it through the indexer (<c>x[i] = np.ma.NDMasked</c>) masks the slot and leaves the data
        ///     untouched.
        /// </summary>
        /// <remarks>
        ///     NumPy spells this value <c>np.ma.masked</c> (and <c>np.ma.masked_singleton</c>). Both NumPy names are
        ///     exposed as aliases — <see cref="masked"/> and <see cref="masked_singleton"/> — that return this very
        ///     instance, so ported NumPy code compiles unchanged while NumSharp code may use the <c>ND*</c> name.
        ///     Every spelling is the same object; comparing any two of them by reference is always true.
        /// </remarks>
        public NDMaskedConstant NDMasked { get; } = new NDMaskedConstant();

        /// <summary>Alias of <see cref="NDMasked"/> mirroring NumPy's <c>masked_singleton</c> in the <c>ND*</c>
        /// spelling — the SAME object, not a copy.</summary>
        public NDMaskedConstant NDMaskedSingleton => NDMasked;

        /// <summary>
        ///     NumPy's <c>np.ma.masked</c>: the SAME <see cref="NDMaskedConstant"/> instance as <see cref="NDMasked"/>
        ///     (NumSharp's <c>ND*</c> spelling of it), exposed under NumPy's name so NumPy code ports verbatim —
        ///     <c>x[i] = np.ma.masked</c> masks a slot, and a fully-masked reduction such as
        ///     <c>np.ma.sum(allMasked)</c> returns an object reference-equal to this one.
        /// </summary>
        /// <remarks>An expression-bodied alias rather than a second stored instance on purpose: a separate
        /// <c>new NDMaskedConstant()</c> would break the reference-identity contract the ufuncs rely on (they
        /// return <see cref="NDMasked"/>, which must be <c>is</c>-equal to <c>np.ma.masked</c>).</remarks>
        public NDMaskedConstant masked => NDMasked;

        /// <summary>
        ///     NumPy's <c>np.ma.masked_singleton</c> — in NumPy just another name for <c>masked</c>
        ///     (<c>np.ma.masked is np.ma.masked_singleton</c> is <c>True</c>); here likewise the SAME instance as
        ///     <see cref="NDMasked"/> and <see cref="masked"/>.
        /// </summary>
        public NDMaskedConstant masked_singleton => NDMasked;

        /// <summary>NumPy's <c>MaskType</c>: the dtype of a mask, i.e. boolean.</summary>
        public DType MaskType => np.@bool;

        /// <summary>NumPy's <c>bool_</c>: the boolean dtype (same as <see cref="MaskType"/>), exported by
        /// <c>numpy.ma</c> for building masks explicitly.</summary>
        public DType bool_ => np.@bool;

        /// <summary>NumPy's <c>mr_</c>: the masked concatenation index-expression object (the masked
        /// counterpart of <see cref="np.r_"/>). Use as <c>np.ma.mr_[a, b]</c> / <c>np.ma.mr_["1", A, B]</c>.
        /// See <see cref="MrClass"/>.</summary>
        public MrClass mr_ { get; } = new MrClass();

        /// <summary>NumPy's <c>masked_print_option</c>: the shared, mutable object controlling the masked-slot
        /// display token (default <c>--</c>) and whether masked slots are shown as that token or as the fill value.
        /// Mutating it changes every subsequent <see cref="NDMaskedArray.ToString()"/> — a process-global side effect.
        /// See <see cref="MaskedPrintOption"/>.</summary>
        public MaskedPrintOption masked_print_option { get; } = new MaskedPrintOption("--");

        /// <summary>
        ///     Returns the data of <paramref name="a"/> as a plain <see cref="NDArray"/> — the underlying data
        ///     for a <see cref="NDMaskedArray"/>, the array itself for an <see cref="NDArray"/>, or a fresh array
        ///     for any other array-like/scalar (NumPy's <c>getdata</c>). The mask, if any, is discarded.
        /// </summary>
        /// <param name="a">A <see cref="NDMaskedArray"/>, <see cref="NDArray"/>, C# scalar, or array-like.</param>
        /// <returns>The data as an <see cref="NDArray"/>, never copied: a new VIEW of a masked array's data (NumPy's
        ///     <c>a._data</c> property — OWNDATA false, writes reach the masked array), a plain array itself, or a new array
        ///     for a scalar / array-like.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public NDArray getdata(object a) => a is NDMaskedArray m ? m._data.view() : AsData(a);

        /// <summary>
        ///     Returns the boolean mask of <paramref name="a"/>, or <see cref="nomask"/> when it has none
        ///     (NumPy's <c>getmask</c>). A plain <see cref="NDArray"/> or scalar has no mask.
        /// </summary>
        /// <param name="a">A <see cref="NDMaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <returns>The mask array, or the <see cref="nomask"/> sentinel.</returns>
        public NDArray getmask(object a) => (a as NDMaskedArray)?._mask ?? nomask;

        /// <summary>
        ///     Returns the mask of <paramref name="a"/> as a FULL boolean array of its data's shape, allocating
        ///     an all-<c>False</c> array when there is no mask (NumPy's <c>getmaskarray</c>). Unlike
        ///     <see cref="getmask"/> this never returns the 0-D <c>nomask</c> sentinel, so callers can index it
        ///     positionally.
        /// </summary>
        /// <param name="a">A <see cref="NDMaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <returns>A boolean <see cref="NDArray"/> matching the data shape.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): a scalar/array-like operand is materialized only to read its shape,
        /// and that materialization is reclaimed; the returned mask is either <paramref name="a"/>'s own (owned by
        /// its masked array — the caller must NOT dispose it) or a fresh all-False array the caller owns.</remarks>
        [NDScoped]
        public NDArray getmaskarray(object a)
        {
            var m = (a as NDMaskedArray)?._mask;
            if (m is not null)
                return m;
            return np.zeros(DimsOf(AsData(a)), np.@bool);
        }

        /// <summary>
        ///     True iff <paramref name="a"/> is a boolean <see cref="NDArray"/> usable as a mask (NumPy's
        ///     <c>is_mask</c> — a plain check on type/dtype, NOT whether any element is set).
        /// </summary>
        /// <param name="a">Any object.</param>
        /// <returns>True if <paramref name="a"/> is an <see cref="NDArray"/> of boolean dtype.</returns>
        public bool is_mask(object a) => a is NDArray nd && nd.typecode == NPTypeCode.Boolean;

        /// <summary>
        ///     True iff <paramref name="x"/> is a <see cref="NDMaskedArray"/> (NumPy's <c>isMaskedArray</c>/<c>is_masked</c>
        ///     type check). Does not inspect whether any element is actually masked.
        /// </summary>
        /// <param name="x">Any object.</param>
        /// <returns>True for a <see cref="NDMaskedArray"/> (including the <see cref="NDMasked"/> constant).</returns>
        public bool isMaskedArray(object x) => x is NDMaskedArray;

        /// <summary>
        ///     Substitutes <paramref name="fill_value"/> for masked elements of <paramref name="a"/> and returns
        ///     a plain array (NumPy's module-level <c>ma.filled</c>). A non-masked input is returned as an array
        ///     unchanged.
        /// </summary>
        /// <param name="a">A <see cref="NDMaskedArray"/> or array-like.</param>
        /// <param name="fill_value">Fill for masked slots; null uses the dtype default.</param>
        /// <returns>A plain <see cref="NDArray"/> with masked slots filled.</returns>
        public NDArray filled(object a, object fill_value = null)
            => a is NDMaskedArray m ? m.filled(fill_value) : AsData(a);

        /// <summary>
        ///     Builds a masked array from data and an optional mask (NumPy's <c>ma.array</c>/<c>masked_array</c>, the
        ///     <c>MaskedArray.__new__</c> constructor). The data is taken via <see cref="getdata"/>; a given mask is
        ///     converted to boolean and reconciled with the data's shape; a masked-array <paramref name="data"/> lends its
        ///     mask, fill value and hardness.
        /// </summary>
        /// <param name="data">Data array-like (a <see cref="NDMaskedArray"/>'s own mask, fill value and hard-mask flag ride
        ///     along).</param>
        /// <param name="mask">Optional mask array-like; null/omitted (or <see cref="nomask"/>) ⇒ no explicit mask. A C#
        ///     <c>true</c> / <c>false</c> is NumPy's <c>mask=True</c> / <c>mask=False</c>: a FULL array of that value.</param>
        /// <param name="fill_value">Optional default fill value for <see cref="NDMaskedArray.filled(object)"/>; null
        ///     inherits <paramref name="data"/>'s. Checked against the dtype (<see cref="CheckFillValue"/>).</param>
        /// <param name="copy">When true, copies the data (NumPy's <c>copy=</c>), the inherited mask and a given mask array;
        ///     default false aliases them — a given boolean mask array of the data's shape is ADOPTED (the masked array then
        ///     writes into it, as NumPy's does, and <see cref="getmask"/> returns it), the data's own mask is shared through
        ///     a view.</param>
        /// <param name="dtype">Optional dtype to cast the data to.</param>
        /// <param name="hard_mask">True/false makes the result's mask HARD/soft (NumPy's <c>hard_mask=</c>: a hard mask
        ///     refuses a plain-value assignment's unmasking, see <see cref="NDMaskedArray.harden_mask()"/>); null inherits
        ///     <paramref name="data"/>'s hardness. The default is false, NumPy's <c>ma.array</c> FUNCTION
        ///     (<c>hard_mask=False</c>); <see cref="masked_array"/>, NumPy's class, defaults to null.</param>
        /// <param name="keep_mask">With a given <paramref name="mask"/>, true (default) ORs it with
        ///     <paramref name="data"/>'s own mask and false replaces that mask; without one, false erases the data's mask.</param>
        /// <param name="shrink">With no <paramref name="mask"/> and <paramref name="keep_mask"/> false: true (default) leaves
        ///     no mask (<see cref="nomask"/>), false a full all-False mask.</param>
        /// <returns>A new <see cref="NDMaskedArray"/>.</returns>
        /// <exception cref="MaskError">The mask is neither the data's shape, nor one element, nor the data's size
        ///     (<c>Mask and data not compatible: data size is 3, mask size is 2.</c>).</exception>
        /// <exception cref="TypeError">A <see cref="string"/> <paramref name="fill_value"/> for numeric data (see
        ///     <see cref="CheckFillValue"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text mask (NumPy reads the strings' truth
        ///     values), or a fill value that must be formatted as text or parsed as a number.</exception>
        /// <remarks>
        ///     <para>
        ///     <b>The mask, as NumPy reconciles it.</b> A mask of the data's shape is used as is; a ONE-element mask is
        ///     repeated over the data (<c>np.resize</c>); a mask of the data's SIZE in another shape is reshaped; anything
        ///     else is the <see cref="MaskError"/>. The sizes are tested before the mask's values are read, so a text mask of
        ///     the wrong size reports the MaskError even though its strings could not be converted yet (NumPy converts first,
        ///     but that conversion cannot fail).
        ///     </para>
        ///     <para>Scoped by hand: a scalar/array-like <paramref name="data"/> or <paramref name="mask"/> is first
        ///     materialized, then (on a dtype cast / boolean coercion) replaced by the converted array — the
        ///     superseded materializations are reclaimed on return and only the arrays the result carries leave.</para>
        /// </remarks>
        public NDMaskedArray array(object data, object mask = null, object fill_value = null, bool copy = false, DType dtype = null,
            bool? hard_mask = false, bool keep_mask = true, bool shrink = true)
        {
            using var scope = NDScope.Open();
            var source = data as NDMaskedArray;
            var d = AsData(data);
            // dtype cast happens first (and subsumes the copy — astype makes a new buffer on a real cast, and
            // copy=true forces one even when the dtype already matches); mask stays boolean regardless of dtype.
            if (dtype != null)
                d = d.astype(dtype, copy);
            else if (copy)
                d = d.copy();
            // np.array(masked, subok=True) keeps the data's mask: copied with the data (`_mask.copy()`), else a VIEW of it
            // (`__array_finalize__` over the same data: `_mask = _mask.view()`) — shared memory, not the same object.
            NDArray existing = source?._mask;
            if (existing is not null)
                existing = copy ? existing.copy() : existing.view();
            NDArray m;
            if (mask is null || ReferenceEquals(mask, nomask))
            {
                // Case 1, no mask in input: keep the data's mask, or erase it (nomask with shrink, a full False array without).
                m = keep_mask ? existing : (shrink ? null : np.zeros(DimsOf(d), np.@bool));
            }
            else
            {
                // Case 2, a mask in input: OR it with the data's own mask (keep_mask), or let it replace that mask.
                var given = MaskArgument(mask, d, copy);
                m = existing is null || !keep_mask ? given : np.logical_or(given, existing);
            }
            // `if fill_value is None: fill_value = getattr(data, '_fill_value', None)`, then _check_fill_value.
            var result = new NDMaskedArray(d, m, StoreFill(fill_value ?? source?._fill_value, d.dtype));
            // hard_mask=None inherits the data's hardness (`getattr(data, '_hardmask', False)`); true bakes the hardness in at
            // construction, so the very first assignment already refuses to unmask.
            result._hardmask = hard_mask ?? source?._hardmask ?? false;
            return Yield(scope, result);
        }

        /// <summary>
        ///     The <c>mask=</c> argument of <see cref="array"/> as NumPy's constructor reads it (its "Case 2"): the Python
        ///     literals <c>True</c>/<c>False</c> become full arrays, anything else is converted to boolean
        ///     (<c>np.array(mask, dtype=bool)</c>) and then fitted to the data's shape.
        /// </summary>
        /// <param name="mask">The mask argument (not null, not <see cref="nomask"/>).</param>
        /// <param name="d">The (possibly cast) data the mask is for.</param>
        /// <param name="copy">The constructor's <c>copy=</c>: false lets a boolean <see cref="NDArray"/> mask of the data's
        ///     shape be used as is.</param>
        /// <returns>A boolean mask with <paramref name="d"/>'s shape: the caller's own array when it already is one (and
        ///     <paramref name="copy"/> is false), otherwise a fresh array.</returns>
        /// <exception cref="MaskError">The mask's size is neither one nor the data's size, and its shape is not the data's.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text mask (the strings' truth values).</exception>
        /// <remarks>NumPy's <c>mask = np.array(mask, copy=copy, dtype=bool)</c> returns the caller's boolean array itself
        /// when no conversion is needed, and <c>_data._mask = mask</c> then ADOPTS it: <c>x._mask is m</c>, a masked write
        /// such as <c>x[1] = masked</c> lands in the caller's array, and a write to the caller's array shows through
        /// <c>x</c> (probed on 2.4.2). A converted, resized or reshaped mask is a new array.</remarks>
        private static NDArray MaskArgument(object mask, NDArray d, bool copy)
        {
            // `mask is True` / `mask is False`: a full mask of that value, whatever the data's shape.
            if (mask is bool flag)
                return flag ? np.ones(DimsOf(d), np.@bool) : np.zeros(DimsOf(d), np.@bool);
            var raw = AsData(mask);
            // The size test needs no element, so it runs before a text mask's truth values are read.
            bool sameShape = CheckMaskFitsData(raw, d);
            // `np.array(mask, copy=False, dtype=bool)` of a boolean ndarray is that array: adopted, not copied. (A C# bool[]
            // is a fresh conversion either way; AsData built it, so it is ours.)
            if (!copy && sameShape && mask is NDArray && raw.typecode == NPTypeCode.Boolean)
                return raw;
            // np.array(mask, dtype=bool): non-zero numbers and non-empty text are True (a fresh array).
            var converted = raw.astype(np.@bool);
            if (sameShape)
                return converted;
            // np.resize repeats the one element over the data; np.reshape lays a same-size mask out in the data's shape.
            return raw.size == 1 ? np.resize(converted, DimsOf(d)) : np.reshape(converted, DimsOf(d));
        }

        /// <summary>
        ///     The shape rule of NumPy's constructor for a given mask: the data's shape, one element (repeated), or the data's
        ///     size (reshaped); anything else is the <see cref="MaskError"/>. Decided by shapes alone.
        /// </summary>
        /// <param name="raw">The mask argument as an array (not yet converted to boolean).</param>
        /// <param name="d">The data.</param>
        /// <returns>True when the mask already has the data's shape.</returns>
        /// <exception cref="MaskError">The sizes are incompatible (<c>Mask and data not compatible: data size is 3, mask
        ///     size is 2.</c>).</exception>
        private static bool CheckMaskFitsData(NDArray raw, NDArray d)
        {
            if (raw.Shape.dimensions.AsSpan().SequenceEqual(d.Shape.dimensions))
                return true;
            long nd = d.size, nm = raw.size;
            if (nm != 1 && nm != nd)
                throw new MaskError($"Mask and data not compatible: data size is {nd}, mask size is {nm}.");
            return false;
        }

        /// <summary>Alias of <see cref="array"/> matching NumPy's <c>ma.masked_array</c> name.</summary>
        /// <param name="data">Data array-like.</param>
        /// <param name="mask">Optional mask (see <see cref="array"/>).</param>
        /// <param name="fill_value">Optional default fill value.</param>
        /// <param name="copy">Copy the data when true.</param>
        /// <param name="dtype">Optional dtype to cast the data to.</param>
        /// <param name="hard_mask">Hard/soft mask, or null (the default — NumPy's <c>MaskedArray(hard_mask=None)</c>) to
        ///     inherit <paramref name="data"/>'s hardness; <see cref="array"/>, NumPy's function, defaults to soft.</param>
        /// <param name="keep_mask">OR a given mask with the data's own (see <see cref="array"/>).</param>
        /// <param name="shrink">Shrink an erased mask to <see cref="nomask"/> (see <see cref="array"/>).</param>
        /// <returns>A new <see cref="NDMaskedArray"/>.</returns>
        /// <exception cref="MaskError">See <see cref="array"/>.</exception>
        /// <exception cref="TypeError">See <see cref="array"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="array"/>.</exception>
        public NDMaskedArray masked_array(object data, object mask = null, object fill_value = null, bool copy = false, DType dtype = null,
            bool? hard_mask = null, bool keep_mask = true, bool shrink = true)
            => array(data, mask, fill_value, copy, dtype, hard_mask, keep_mask, shrink);

        /// <summary>
        ///     Converts an array-like to a <see cref="NDMaskedArray"/> WITHOUT copying (NumPy's <c>ma.asarray</c>):
        ///     an incoming masked array keeps its mask, a plain array/scalar becomes unmasked. This is the
        ///     explicit ndarray→NDMaskedArray entry point — NumSharp's implicit <c>NDArray→NDMaskedArray</c> does the
        ///     same for the unmasked case, but this also honors an existing mask and an optional dtype cast.
        /// </summary>
        /// <param name="a">A <see cref="NDMaskedArray"/>, <see cref="NDArray"/>, or array-like/scalar.</param>
        /// <param name="dtype">Optional dtype to cast the data to (no copy when it already matches).</param>
        /// <returns>A masked array over <paramref name="a"/>'s data (aliased when possible).</returns>
        /// <exception cref="TypeError">A masked operand's text fill cannot be converted into a text <paramref name="dtype"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">The cast converts between text and numbers.</exception>
        /// <remarks>NumPy's <c>masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=False)</c>: the class
        /// constructor, so a masked operand also lends its fill value (converted to the dtype) and its hard-mask flag.
        /// Scoped by hand: a scalar/array-like operand that is materialized and then CAST leaves its pre-cast
        /// materialization behind, which is reclaimed on return.</remarks>
        public NDMaskedArray asarray(object a, DType dtype = null) => masked_array(a, dtype: dtype);

        /// <summary>NumPy's <c>ma.asanyarray</c>: a masked array whose dtype already matches is returned ITSELF (NumPy's
        /// identity rule, gh-8666); anything else goes through <see cref="asarray"/>.</summary>
        /// <param name="a">Array-like.</param><param name="dtype">Optional dtype cast.</param>
        /// <returns><paramref name="a"/> itself, or a masked array over its data.</returns>
        /// <exception cref="TypeError">See <see cref="asarray"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="asarray"/>.</exception>
        public NDMaskedArray asanyarray(object a, DType dtype = null)
            => a is NDMaskedArray m && (dtype is null || m.dtype.Equals(dtype)) ? m : asarray(a, dtype);

        /// <summary>
        ///     The default fill value NumPy assigns per dtype kind (<c>default_fill_value</c>, its <c>default_filler</c>
        ///     table): <c>True</c> for boolean, <c>999999</c> for integers, <c>1e20</c> for float, <c>(1e20+0j)</c> for
        ///     complex, the text <c>'N/A'</c> for a <c>U</c> dtype and the bytes <c>b'N/A'</c> for an <c>S</c> dtype. Used by
        ///     <see cref="NDMaskedArray.filled(object)"/> and the <see cref="NDMaskedArray.fill_value"/> getter when the
        ///     array holds no fill of its own.
        /// </summary>
        /// <param name="dtype">The data dtype to pick a fill for.</param>
        /// <returns>A boxed scalar of a type castable into <paramref name="dtype"/>: a <see cref="string"/> for a
        ///     <c>U</c> dtype and a fresh <see cref="byte"/>[] for an <c>S</c> dtype.</returns>
        /// <remarks>
        ///     An INSTANCE method (like <see cref="minimum_fill_value"/>/<see cref="maximum_fill_value"/>), so
        ///     <c>np.ma.default_fill_value(dtype)</c> ports from NumPy verbatim.
        ///     <para>
        ///     The text default is NOT sized to the dtype: NumPy stores <c>np.array('N/A')</c> (a <c>&lt;U3</c>), so the fill
        ///     value of a <c>&lt;U1</c> array reads all three characters, and only <see cref="NDMaskedArray.filled(object)"/>
        ///     truncates it, when it copies the fill into the data's width (<c>'N/'</c> into <c>&lt;U2</c>). The bytes default
        ///     is a NEW array on every call, because a <see cref="byte"/>[] is mutable and the caller owns it.
        ///     </para>
        /// </remarks>
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
                // default_filler['U'] = 'N/A', default_filler['S'] = b'N/A' (the dtype's kind picks the class).
                case NPTypeCode.String: return dtype.kind == 'S' ? DefaultBytesFill() : "N/A";
                default: return 999999L; // every integer kind (signed/unsigned/char)
            }
        }

        /// <summary>
        ///     NumPy's default fill value for an <c>S</c> dtype, the bytes <c>b'N/A'</c>, as a fresh array the caller owns.
        /// </summary>
        /// <returns>A new three-byte array <c>{ 'N', '/', 'A' }</c>.</returns>
        private static byte[] DefaultBytesFill() => new[] { (byte)'N', (byte)'/', (byte)'A' };

        /// <summary>
        ///     NumPy's <c>_check_fill_value(fill_value, ndtype)</c>, the conversion every fill value goes through before a
        ///     masked array stores or uses it (the constructor, the <see cref="NDMaskedArray.fill_value"/> setter,
        ///     <see cref="set_fill_value"/>, an explicit <see cref="NDMaskedArray.filled(object)"/> argument), for the cases
        ///     that involve TEXT. A numeric fill for a numeric dtype is returned as given.
        /// </summary>
        /// <param name="fill_value">The fill value as the caller gave it; null keeps the dtype default.</param>
        /// <param name="ndtype">The data dtype the fill is for.</param>
        /// <returns>
        ///     Null for a null <paramref name="fill_value"/>; for a text dtype, the fill converted to the data dtype as a new
        ///     array the caller owns (0-d for a scalar fill — NumPy's <c>np.array(fill)</c>, read through
        ///     <see cref="FillScalar"/> — and the fill's own shape for NumPy's deprecated non-scalar fill); otherwise
        ///     <paramref name="fill_value"/> itself.
        /// </returns>
        /// <exception cref="TypeError">A C# <see cref="string"/> fill for a dtype that is not text (NumPy's <c>Cannot set
        ///     fill value of string with array of dtype float64</c>), or a text fill whose characters the target class
        ///     cannot encode or decode (<c>Cannot convert fill_value é to dtype |S2</c>, NumPy's TypeError around the
        ///     <c>UnicodeEncodeError</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The conversion formats a number as text (a numeric fill
        ///     for a text dtype, NumPy's <c>np.asarray(5, dtype='&lt;U2')</c> is <c>'5'</c>) or parses text as a number (a
        ///     bytes or text-array fill for a numeric dtype): the computing phase of the string dtypes.</exception>
        /// <remarks>
        ///     <para>
        ///     <b>Order.</b> NumPy tests <c>isinstance(fill_value, str)</c> against the dtype's character before it converts
        ///     anything, so a <see cref="string"/> fill for a numeric dtype is refused by its type alone, whatever its text.
        ///     A bytes fill is not a <c>str</c>, so NumPy parses it (<c>b'1.5'</c> becomes <c>1.5</c> for float64).
        ///     </para>
        ///     <para>
        ///     <b>Scope of the port.</b> NumPy converts EVERY fill to the data dtype (<c>fill_value=1.5</c> of an int64 array
        ///     reads back <c>1</c>); NumSharp keeps a numeric fill of a numeric array as given, the module's documented
        ///     simplification, so the existing numeric behavior (and <see cref="common_fill_value"/>'s comparison of raw
        ///     fills) is unchanged. Only the text cases are converted.
        ///     </para>
        ///     <para>
        ///     <b>Ownership.</b> A converted array fill is a fresh array; a caller that stores it into a masked array's field
        ///     must detach it from the ambient scope (<see cref="Own"/>), as for a mask.
        ///     </para>
        /// </remarks>
        internal static object CheckFillValue(object fill_value, DType ndtype)
        {
            if (fill_value is null)
                return null;
            bool textData = ndtype.typecode == NPTypeCode.String;
            switch (fill_value)
            {
                case string s:
                    // `isinstance(fill_value, str) and (ndtype.char not in 'OSTVU')`: refused by its type, before NumPy reads
                    // a character (a str fill for an S dtype is accepted and encoded).
                    if (!textData)
                        throw new TypeError($"Cannot set fill value of string with array of dtype {ndtype}");
                    using (var src = NDArray.Scalar(s))
                        return TextFillToDType(src, ndtype, s);
                case byte[] b:
                    using (var src = NDArray.Scalar(b))
                    {
                        // np.asarray(b'1.5', dtype=float64) parses the bytes: the next phase of the string support.
                        if (!textData)
                            throw NDStringCasting.TextConversion(src.dtype, ndtype);
                        return TextFillToDType(src, ndtype, PyRepr.Bytes(b));
                    }
                case NDMaskedArray m:
                    // np.asarray(masked_array) reads its data (the mask plays no part in a fill value).
                    return CheckFillValue(m._data, ndtype);
                case NDArray nd:
                    bool textFill = nd.typecode == NPTypeCode.String;
                    if (textFill && textData)
                        return TextFillToDType(nd, ndtype, nd.ToString());
                    // Text into a number parses it, a number into text formats it: both compute on text.
                    if (textFill || textData)
                        throw NDStringCasting.TextConversion(nd.dtype, ndtype);
                    return fill_value;
                default:
                    // A C# number, bool, char, Half, decimal or Complex: into a text dtype NumPy formats it as text. A
                    // BigInteger (a Python int of any size) has no storage lane; the message names int64, NumPy's default
                    // integer.
                    if (textData)
                        throw NDStringCasting.TextConversion(
                            fill_value is BigInteger ? DType.From(NPTypeCode.Int64) : (DType)fill_value.GetType(), ndtype);
                    return fill_value;
            }
        }

        /// <summary>
        ///     <see cref="CheckFillValue"/> for a fill a masked array STORES in its <see cref="NDMaskedArray._fill_value"/>
        ///     field (the constructor, the <see cref="NDMaskedArray.fill_value"/> setter, <see cref="NDMaskedArray.astype"/>).
        ///     A text conversion that produced an array (NumPy's deprecated non-scalar fill) is detached from every ambient
        ///     scope, because the field outlives the method that converted it.
        /// </summary>
        /// <param name="fill_value">The fill as given, or null for the dtype default.</param>
        /// <param name="ndtype">The data dtype of the array that stores the fill.</param>
        /// <returns>The object to store.</returns>
        /// <exception cref="TypeError">See <see cref="CheckFillValue"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="CheckFillValue"/>.</exception>
        internal static object StoreFill(object fill_value, DType ndtype)
        {
            var checkedFill = CheckFillValue(fill_value, ndtype);
            // For a text dtype the result always comes from TextFillToDType's fresh astype (0-d for a scalar fill), so it
            // is ours to detach.
            // A numeric fill is the caller's own object, stored as given and left in the caller's scope (detaching a
            // caller's array would change its lifetime).
            if (checkedFill is NDArray converted && ndtype.typecode == NPTypeCode.String)
                return Own(converted);
            return checkedFill;
        }

        /// <summary>
        ///     The string-to-string half of <see cref="CheckFillValue"/>: <c>np.asarray(fill, dtype=ndtype)</c> for a text
        ///     fill and a text dtype (a width change truncates, <c>U → S</c> encodes ASCII, <c>S → U</c> decodes it).
        /// </summary>
        /// <param name="fill">The text fill as an array (a 0-d array for a scalar fill); never modified.</param>
        /// <param name="ndtype">The text data dtype.</param>
        /// <param name="display">The fill as NumPy's <c>%s</c> formats it, for the error text.</param>
        /// <returns>A new array of <paramref name="ndtype"/> the caller owns: 0-d for a scalar fill (NumPy keeps
        ///     <c>np.array(fill)</c>, whose width is part of the fill's state), the fill's shape otherwise.</returns>
        /// <exception cref="TypeError">The target class cannot encode or decode a character (NumPy catches the
        ///     <c>UnicodeError</c>, a <c>ValueError</c>, and raises <c>Cannot convert fill_value … to dtype …</c>).</exception>
        private static object TextFillToDType(NDArray fill, DType ndtype, string display)
        {
            NDArray converted;
            try
            {
                converted = fill.astype(ndtype);
            }
            catch (UnicodeError e)
            {
                throw new TypeError($"Cannot convert fill_value {display} to dtype {ndtype}", e);
            }
            // `return np.array(fill_value)`: the converted array itself, a 0-d one for a scalar fill. The getter reads its
            // element (FillScalar); keeping the array keeps its width, which a later `_fill_value[()] = v` writes into.
            return converted;
        }

        /// <summary>
        ///     What NumPy's <c>fill_value</c> getter returns for a stored fill, <c>self._fill_value[()]</c>: the element of
        ///     a 0-d array (a <see cref="string"/> for <c>U</c>, a <see cref="byte"/>[] for <c>S</c>, the number for a
        ///     numeric array), an array with dimensions (NumPy's deprecated non-scalar fill) as itself, and any other stored
        ///     value (an unconverted numeric fill) as is.
        /// </summary>
        /// <param name="stored">The <see cref="NDMaskedArray._fill_value"/> field, or null.</param>
        /// <returns>The scalar the getter returns, or null when nothing is stored.</returns>
        internal static object FillScalar(object stored)
            => stored is NDArray nd && nd.ndim == 0 ? nd.item() : stored;

        /// <summary>
        ///     True when a fill given for a TEXT dtype would be FORMATTED as text by NumPy's <c>_check_fill_value</c> (a
        ///     number, a bool, or an array of a non-string dtype): a conversion that cannot fail, but whose result is the
        ///     next phase's business. A <see cref="string"/>, <see cref="byte"/>[] or text array is converted between the
        ///     string classes instead.
        /// </summary>
        /// <param name="fill">The fill as the caller gave it (not null).</param>
        /// <returns>True for a value NumPy formats into text.</returns>
        internal static bool FormatsAsText(object fill) => fill switch
        {
            string or byte[] => false,
            NDMaskedArray m => m._data.typecode != NPTypeCode.String,
            NDArray nd => nd.typecode != NPTypeCode.String,
            _ => true,
        };

        // ─────────────────────────────────────────────────────────────────────────────
        //  Predicates, shape queries, copy, and the deprecated/aliased names
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     True iff <paramref name="x"/> has AT LEAST ONE masked element (NumPy's <c>is_masked</c>) — the
        ///     VALUE predicate, DISTINCT from the <see cref="isMaskedArray"/> TYPE check. A plain array, a
        ///     <c>nomask</c> operand, an all-False mask, and every non-masked scalar all return false; only a
        ///     masked array with a set mask bit (including the <see cref="NDMasked"/> singleton) returns true.
        /// </summary>
        /// <param name="x">Any object.</param>
        /// <returns>True when some element of <paramref name="x"/> is actually masked.</returns>
        public bool is_masked(object x)
        {
            // `is not null` (reference) then `np.any` — a `!= null` would run NDArray's ELEMENTWISE `!=`.
            var m = (x as NDMaskedArray)?._mask;
            return m is not null && np.any(m);
        }

        /// <summary>Number of dimensions of the data (NumPy's module-level <c>ma.ndim</c>).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The rank of the data (0 for a scalar).</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): a scalar/array-like operand is materialized only to be measured.</remarks>
        [NDScoped]
        public int ndim(object a) => AsData(a).ndim;

        /// <summary>Data shape as a <c>long[]</c> (NumPy's module-level <c>ma.shape</c>; NumSharp dims are 64-bit).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The shape (empty for a scalar).</returns>
        public long[] shape(object a)
        {
            // Scoped by hand (a long[] return is not a shape the weaver yields): a scalar/array-like operand is
            // materialized only to read its dimensions, and is reclaimed; an array operand is untouched.
            using var scope = NDScope.Open();
            return AsData(a).shape;
        }

        /// <summary>Total element count (NumPy's module-level <c>ma.size</c>).</summary>
        /// <param name="a">Array-like/masked-array/scalar.</param>
        /// <returns>The element count (1 for a scalar).</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): a scalar/array-like operand is materialized only to be measured.</remarks>
        [NDScoped]
        public long size(object a) => AsData(a).size;

        /// <summary>Deep copy of <paramref name="a"/> as a masked array (NumPy's module-level <c>ma.copy</c>, the
        /// <c>_arraymethod('copy')</c>): the data AND the mask are copied, so mutating either the source or the copy never
        /// affects the other, and the copy takes the source's fill value and hard-mask flag (<c>_update_from(self)</c>).</summary>
        /// <param name="a">Array-like/masked-array to copy.</param>
        /// <returns>A masked array over a fresh copy of the data (mask copied when present).</returns>
        /// <remarks>Scoped by hand: a scalar/array-like operand's materialization (the source of the copy) is
        /// reclaimed; only the copies leave.</remarks>
        public NDMaskedArray copy(object a)
        {
            using var scope = NDScope.Open();
            var source = a as NDMaskedArray;
            var r = new NDMaskedArray(AsData(a).copy(), source?._mask?.copy());
            if (source is not null)
                r.UpdateFrom(source);
            return Yield(scope, r);
        }

        /// <summary>Maximum over unmasked elements — alias of <see cref="max"/> (NumPy's deprecated-but-exported
        /// <c>amax</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked maximum.</returns>
        public NDMaskedArray amax(object a, int? axis = null, object fill_value = null, bool keepdims = false) => max(a, axis, fill_value, keepdims);

        /// <summary>Minimum over unmasked elements — alias of <see cref="min"/> (NumPy's <c>amin</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="fill_value">Masked fill override.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked minimum.</returns>
        public NDMaskedArray amin(object a, int? axis = null, object fill_value = null, bool keepdims = false) => min(a, axis, fill_value, keepdims);

        /// <summary>
        ///     Logical-AND reduction along an axis (NumPy's <c>ma.alltrue</c> — <c>_MaskedBinaryOperation(logical_and, 1,
        ///     1).reduce</c>, NOT an alias of <see cref="all"/>): the masked slots are filled with 1 (True), the result is
        ///     masked where its whole slice was masked, and the default axis is 0.
        /// </summary>
        /// <param name="target">Operand.</param>
        /// <param name="axis">The reduced axis (default 0, as NumPy's), or null for the flattened array.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c> of the reduction (used only when <paramref name="target"/> has a mask;
        ///     only bool has a <c>logical_and</c> loop).</param>
        /// <returns>The <see cref="NDMasked"/> singleton for a masked 0-d result, an unmasked 0-d array for a scalar,
        ///     otherwise a boolean masked array whose mask is an ARRAY when the operand had one (NumPy does not shrink it).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="TypeError">A masked operand with a non-bool <paramref name="dtype"/> (<c>No loop matching the
        ///     specified signature and casting was found for ufunc logical_and</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text operand (its truth values are read).</exception>
        public NDMaskedArray alltrue(object target, int? axis = 0, DType dtype = null) => LogicalReduce(target, axis, dtype, and: true);

        /// <summary>
        ///     Logical-OR reduction along an axis (NumPy's <c>ma.sometrue</c> — <c>logical_or.reduce</c> of the masked
        ///     <c>logical_or</c>, NOT an alias of <see cref="any"/>): the masked slots are filled with 0 (False), the result
        ///     is masked where its whole slice was masked, and the default axis is 0.
        /// </summary>
        /// <param name="target">Operand.</param>
        /// <param name="axis">The reduced axis (default 0, as NumPy's), or null for the flattened array.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c> of the reduction (used only when <paramref name="target"/> has a mask;
        ///     only bool has a <c>logical_or</c> loop).</param>
        /// <returns>See <see cref="alltrue"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="TypeError">A masked operand with a non-bool <paramref name="dtype"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text operand (its truth values are read).</exception>
        public NDMaskedArray sometrue(object target, int? axis = 0, DType dtype = null) => LogicalReduce(target, axis, dtype, and: false);

        /// <summary>
        ///     NumPy's <c>_MaskedBinaryOperation.reduce(target, axis, dtype)</c> for <c>logical_and</c> / <c>logical_or</c>:
        ///     <c>t = filled(target, filly)</c> (1 for and, 0 for or); a 0-d operand and its mask are reshaped to <c>(1,)</c>;
        ///     without a mask <c>tr = f.reduce(t, axis)</c> (the dtype is NOT passed), with one
        ///     <c>tr = f.reduce(t, axis, dtype=dtype)</c> and <c>mr = logical_and.reduce(m, axis)</c>; a 0-d <c>tr</c> is
        ///     <c>masked</c> when <c>mr</c> is set, else the scalar; otherwise <c>tr</c> viewed as a masked array with
        ///     <c>mr</c> as its mask (the default fill — nothing is inherited).
        /// </summary>
        /// <param name="target">Operand.</param>
        /// <param name="axis">The reduced axis, or null for the flattened array.</param>
        /// <param name="dtype">The reduction's <c>dtype=</c>, honoured only with a mask.</param>
        /// <param name="and">True for <c>logical_and</c> (alltrue), false for <c>logical_or</c> (sometrue).</param>
        /// <returns>The masked reduction (see <see cref="alltrue"/>).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="TypeError">A masked operand with a non-bool <paramref name="dtype"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text operand.</exception>
        /// <remarks>Scoped by hand: the filled copy, the reshaped views and the mask reduction are reclaimed; only the
        /// result's data and mask leave.</remarks>
        private NDMaskedArray LogicalReduce(object target, int? axis, DType dtype, bool and)
        {
            using var scope = NDScope.Open();
            var m = (target as NDMaskedArray)?._mask;
            // `t = filled(target, self.filly)` — over text the fill is formatted (see FilledForNumericOp); the reduction
            // then reads the strings' truth values, so it defers unless the operand is empty.
            Exception deferral = null;
            var t = FilledForNumericOp(target, and ? 1 : 0, ref deferral);
            // `if t.shape == (): t = t.reshape(1); m = make_mask(m, copy=True); m.shape = (1,)`.
            if (t.ndim == 0)
            {
                t = np.reshape(t, new Shape(1L));
                if (m is not null)
                    m = np.reshape(m.copy(), new Shape(1L));
            }
            // `tr = self.f.reduce(t, axis, dtype=dtype)` only with a mask: logical_and / logical_or have bool loops only.
            if (m is not null && dtype is not null && dtype.typecode != NPTypeCode.Boolean)
                throw new TypeError($"No loop matching the specified signature and casting was found for ufunc {(and ? "logical_and" : "logical_or")}");
            NDArray tr = axis is null
                ? NDArray.Scalar(and ? np.all(t) : np.any(t))
                : and ? np.all(t, axis.Value, null, false) : np.any(t, axis.Value, null, false);
            if (deferral is not null)
                throw deferral;
            NDArray mr = m is null ? null
                : axis is null ? NDArray.Scalar(np.all(m)) : np.all(m, axis.Value, null, false);
            // `if not tr.shape: return masked if mr else tr`.
            if (tr.ndim == 0)
                return mr is not null && mr.GetBoolean() ? NDMasked : Yield(scope, new NDMaskedArray(tr, null));
            // `masked_tr = tr.view(tclass); masked_tr._mask = mr` — not shrunk: an all-False mask stays an array.
            return Yield(scope, new NDMaskedArray(tr, mr));
        }

        /// <summary>Round to <paramref name="decimals"/> places — alias of <see cref="round"/> (NumPy's <c>round_</c>).</summary>
        /// <param name="a">Operand.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array.</returns>
        public NDMaskedArray round_(object a, int decimals = 0) => round(a, decimals);

        /// <summary>Stack along the first axis — alias of <see cref="vstack"/> (NumPy's <c>row_stack</c>).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray row_stack(params object[] tup) => vstack(tup);

        /// <summary>Inner product — alias of <see cref="inner"/> (NumPy's <c>innerproduct</c>).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The masked inner product.</returns>
        public NDMaskedArray innerproduct(object a, object b) => inner(a, b);

        /// <summary>Outer product — alias of <see cref="outer"/> (NumPy's <c>outerproduct</c>).</summary>
        /// <param name="a">Left.</param><param name="b">Right.</param><returns>The masked outer product.</returns>
        public NDMaskedArray outerproduct(object a, object b) => outer(a, b);

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
        ///     Coerces an array-like to a boolean mask (NumPy's <c>make_mask</c>: <c>np.array(filled(m, True),
        ///     dtype=bool)</c>): any non-zero number and any non-empty string becomes True, and a masked array reads True
        ///     at its masked slots. With <paramref name="shrink"/> (default) an all-False result collapses to
        ///     <see cref="nomask"/> — the standard "no mask needed" reduction — otherwise a full all-False array is returned.
        /// </summary>
        /// <param name="m">Array-like to coerce; null or <see cref="nomask"/> is <see cref="nomask"/>.</param>
        /// <param name="shrink">Collapse an all-False mask to <see cref="nomask"/> (default true).</param>
        /// <returns>A boolean mask array, or <see cref="nomask"/> when it shrinks to empty.</returns>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text operand: NumPy reads the strings'
        ///     truth values (the next phase of the string support). An empty one converts (an empty boolean mask).</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the operand's materialization and a coerced mask that shrinks to
        /// <see cref="nomask"/> are reclaimed. The shared <see cref="nomask"/> sentinel must never be disposed by
        /// the caller; a non-shrunk result is a fresh array the caller owns (NumPy's default <c>copy=False</c> would hand
        /// back a boolean input itself; NumSharp copies it, so the caller may mutate the mask freely).</remarks>
        [NDScoped]
        public NDArray make_mask(object m, bool shrink = true)
        {
            // `if m is nomask: return nomask`; np.array(None, dtype=bool) is False, which shrinks to nomask as well.
            if (m is null || ReferenceEquals(m, nomask))
                return nomask;
            // filled(m, True): a masked array's masked slots read True (for text data that formats True as text: the
            // next phase, raised by the fill check).
            var b = m is NDMaskedArray masked && masked._mask is not null ? masked.filled(true) : AsData(m);
            // NumPy converts by truth value: numbers via `!= 0` (NaN is True), text by non-emptiness through the
            // text-to-bool cast (refused unless empty), booleans copied.
            var mask = b.typecode switch
            {
                NPTypeCode.Boolean => b.copy(),
                NPTypeCode.String => b.astype(np.@bool),
                _ => np.not_equal(b, NDArray.Scalar(0)),
            };
            return (shrink && !np.any(mask)) ? nomask : mask;
        }

        /// <summary>Builds an all-False boolean mask of the given shape (NumPy's <c>make_mask_none</c>) — the
        /// "nothing masked yet, full array" starting mask.</summary>
        /// <param name="newshape">The mask shape.</param>
        /// <returns>An all-False boolean <see cref="NDArray"/>.</returns>
        public NDArray make_mask_none(Shape newshape) => np.zeros(newshape, np.@bool);

        /// <summary>
        ///     Combines two masks with logical-OR (NumPy's <c>mask_or</c>): a <see cref="nomask"/> / <c>false</c> operand
        ///     contributes nothing (the other is <see cref="make_mask"/>'d alone), two operands of DIFFERENT dtypes are
        ///     refused, and — with <paramref name="shrink"/> (default) — an all-False OR collapses back to
        ///     <see cref="nomask"/>.
        /// </summary>
        /// <param name="m1">First mask (any array-like), or null/<see cref="nomask"/>/<c>false</c>.</param>
        /// <param name="m2">Second mask (any array-like), or null/<see cref="nomask"/>/<c>false</c>.</param>
        /// <param name="shrink">Collapse an all-False result to <see cref="nomask"/> (default true).</param>
        /// <returns>The OR of the present masks, or <see cref="nomask"/> when neither contributes (or it shrinks).</returns>
        /// <exception cref="ValueError">The two operands' dtypes differ (<c>Incompatible dtypes '&lt;U2'&lt;&gt;'bool'</c>;
        ///     a C# scalar or an <c>object[]</c> list has no dtype and prints <c>'None'</c>), or they do not
        ///     broadcast.</exception>
        /// <exception cref="AttributeError">Both operands are dtype-less scalars/lists: NumPy then reads
        ///     <c>None.names</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">Two non-empty text operands of one dtype (NumPy reads
        ///     their truth values), or one text operand beside a no-mask one (see <see cref="make_mask"/>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements, in order: <c>m1 is nomask or m1 is False</c> → <c>make_mask(m2)</c> (no dtype check: a
        ///     text <paramref name="m2"/> is converted by its truth values), likewise for <paramref name="m2"/>; one
        ///     boolean mask passed twice → itself; then <c>dtype1 != dtype2</c> → the ValueError; then
        ///     <c>make_mask(logical_or(m1, m2))</c>. Only the Python literal <c>False</c> and the nomask object are "no
        ///     mask": a 0-d False ARRAY is compared like any operand (<c>Incompatible dtypes 'bool'&lt;&gt;'&lt;U2'</c>). C# null
        ///     stands for nomask (NumSharp's masked arrays hold a null mask for it).
        ///     </para>
        ///     <para>
        ///     A <see cref="NDMaskedArray"/> operand is an ARRAY to NumPy, not a mask: its DATA joins the OR, and its masked
        ///     slots read True (the ufunc's result is masked there, and <c>make_mask</c> fills masked slots with True).
        ///     </para>
        ///     <para>Scoped (<c>[NDScoped]</c>): operands coerced to boolean and an OR that shrinks away are reclaimed. When
        ///     one boolean mask is passed twice the result IS that mask (NumPy returns it), so the caller must not dispose a
        ///     result it did not create.</para>
        /// </remarks>
        [NDScoped]
        public NDArray mask_or(object m1, object m2, bool shrink = true)
        {
            // `(m1 is nomask) or (m1 is False)`: the other operand alone, through make_mask (its own copy and shrink).
            if (IsNoMaskOperand(m1))
                return make_mask(m2, shrink);
            if (IsNoMaskOperand(m2))
                return make_mask(m1, shrink);
            // `m1 is m2 and is_mask(m1)`: one boolean mask OR'd with itself is itself.
            if (ReferenceEquals(m1, m2) && m1 is NDArray same && same.typecode == NPTypeCode.Boolean)
                return shrink && !np.any(same) ? nomask : same;
            // `if dtype1 != dtype2: raise ValueError(...)`, decided by the dtypes alone — before a text operand's truth
            // values are read.
            DType d1 = MaskOperandDType(m1), d2 = MaskOperandDType(m2);
            if (!Equals(d1, d2))
                throw new ValueError($"Incompatible dtypes '{(object)d1 ?? "None"}'<>'{(object)d2 ?? "None"}'");
            // `dtype1.names` of two dtype-less operands.
            if (d1 is null)
                throw new AttributeError("'NoneType' object has no attribute 'names'");
            // make_mask(umath.logical_or(m1, m2)): text operands are refused here unless empty; a masked operand's slots
            // are masked in the ufunc's result and make_mask reads them as True.
            var r = np.logical_or(AsData(m1), AsData(m2));
            var operandMasks = Or((m1 as NDMaskedArray)?._mask, (m2 as NDMaskedArray)?._mask);
            if (operandMasks is not null)
                r = np.logical_or(r, operandMasks);
            return shrink && !np.any(r) ? nomask : r;
        }

        /// <summary>
        ///     Whether a <see cref="mask_or"/> operand is NumPy's "no mask" (<c>m is nomask or m is False</c>): C# null, the
        ///     <see cref="nomask"/> sentinel itself, or the C# literal <c>false</c> (Python's <c>False</c>).
        /// </summary>
        /// <param name="m">The operand.</param>
        /// <returns>True when the operand contributes nothing.</returns>
        private bool IsNoMaskOperand(object m) => m is null || ReferenceEquals(m, nomask) || m is bool b && !b;

        /// <summary>
        ///     <c>getattr(m, 'dtype', None)</c> for a <see cref="mask_or"/> operand: an array's (a masked array's DATA)
        ///     dtype; a C# scalar or an <c>object[]</c> (a Python list) has none. A typed C# array is an ndarray (the house
        ///     NEP 50 map) and reports the dtype it converts to.
        /// </summary>
        /// <param name="m">The operand.</param>
        /// <returns>The dtype, or null for a dtype-less operand.</returns>
        private static DType MaskOperandDType(object m) => m switch
        {
            NDMaskedArray masked => masked.dtype,
            NDArray nd => nd.dtype,
            object[] => null,
            Array => AsData(m).dtype,
            _ => null,
        };

        /// <summary>
        ///     Ravels a mask to a flat 1-D boolean array (NumPy's <c>flatten_mask</c>). In NumPy this collapses
        ///     the nested per-field masks of a STRUCTURED dtype into one flat sequence of booleans; NumSharp has
        ///     no structured dtypes, so the recursion never branches and this is exactly "ravel in C-order, then
        ///     coerce to bool" — which is precisely what NumPy's own non-structured path does
        ///     (<c>np.array([...raveled...], dtype=bool)</c>). Non-boolean input is coerced by non-zero-ness
        ///     (<c>astype(bool)</c> ≡ <c>!= 0</c>), so an integer/float mask flattens like NumPy's.
        /// </summary>
        /// <param name="mask">A mask array-like (<see cref="NDArray"/>, C# array, scalar, or the
        /// <see cref="nomask"/> sentinel). A 0-D input flattens to a length-1 array (matching NumPy).</param>
        /// <returns>A fresh 1-D boolean <see cref="NDArray"/> in C-order.</returns>
        /// <remarks>Returns a fresh array (never a view), matching NumPy which builds a new <c>np.array</c>. Scoped
        /// (<c>[NDScoped]</c>): the raveled intermediate (and a materialized array-like) are reclaimed.</remarks>
        [NDScoped]
        public NDArray flatten_mask(object mask) => np.ravel(AsData(mask)).astype(np.@bool);

        /// <summary>
        ///     The dtype of the mask for a data dtype (NumPy's <c>make_mask_descr</c>) — ALWAYS boolean here.
        ///     In NumPy this recurses over a STRUCTURED dtype to build a matching boolean-field mask descriptor;
        ///     NumSharp has no structured dtypes, so every data dtype maps to the single scalar boolean mask
        ///     descriptor NumPy returns for a non-structured dtype (<c>dtype('bool')</c>).
        /// </summary>
        /// <param name="dtype">The data dtype (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or
        /// <see cref="DType"/> all convert in). Any invalid spelling throws at the <see cref="DType"/> boundary,
        /// as NumPy raises for a bad dtype.</param>
        /// <returns>The boolean <see cref="DType"/> (<see cref="np.@bool"/>).</returns>
        public DType make_mask_descr(DType dtype) => np.@bool;

        // ─────────────────────────────────────────────────────────────────────────────
        //  Fill-value helpers (common_fill_value / set_fill_value / fix_invalid)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     The single fill value shared by both inputs, or null when they differ (NumPy's
        ///     <c>common_fill_value</c>: <c>t1 == t2</c> on the two <c>get_fill_value</c> results). Each operand's fill is its
        ///     stored override, else its dtype default.
        /// </summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <returns>The common fill value (boxed; the FIRST operand's when they are equal), or null when the two differ.</returns>
        /// <exception cref="ValueError">A fill value is an array (NumPy's deprecated non-scalar fill) and the elementwise
        ///     comparison has more than one element: Python's truth test of the comparison raises.</exception>
        /// <exception cref="StringComputationNotSupportedException">Two text ARRAY fills: NumPy compares their text (the
        ///     next phase of the string support).</exception>
        /// <remarks>
        ///     The comparison is Python's <c>==</c> between the two scalars: text by content (a <c>str</c> never equals a
        ///     <c>bytes</c>, so a <c>U</c> and an <c>S</c> array share no fill), numbers by value (<c>5.0 == 5</c>), and a
        ///     number never equals text.
        /// </remarks>
        public object common_fill_value(object a, object b)
        {
            // Scoped by hand (an object return): a scalar/array-like operand is materialized only to read its
            // dtype for the default fill, and is reclaimed.
            using var scope = NDScope.Open();
            var fa = FillOf(a);
            var fb = FillOf(b);
            return FillValuesEqual(fa, fb) ? fa : null;
        }

        /// <summary>The effective fill value of an operand (NumPy's <c>get_fill_value</c>): a masked array's
        /// <see cref="NDMaskedArray.fill_value"/> getter — which materializes a text default, as NumPy's does — else the
        /// default of the operand's dtype.</summary>
        /// <param name="a">The operand.</param>
        /// <returns>The fill value NumPy's <c>get_fill_value</c> returns for it.</returns>
        private object FillOf(object a)
            => a is NDMaskedArray m ? m.fill_value : default_fill_value(AsData(a).dtype);

        /// <summary>
        ///     Python's <c>t1 == t2</c> between two fill values, as <see cref="common_fill_value"/> evaluates it: text by
        ///     content and class, numbers by value across types, and an array fill through NumPy's elementwise comparison
        ///     and the truth test of its result.
        /// </summary>
        /// <param name="fa">The first fill.</param>
        /// <param name="fb">The second fill.</param>
        /// <returns>True when Python would treat the two as equal.</returns>
        /// <exception cref="ValueError">An array comparison with more than one element (Python's truth test).</exception>
        private static bool FillValuesEqual(object fa, object fb)
        {
            if (fa is NDArray || fb is NDArray)
            {
                // `array == x` is NumPy's elementwise comparison; `if result:` then needs exactly one element. Only the
                // 0-d wrappers made here are disposed: a stored array fill belongs to its masked array.
                using var leftOwned = fa is NDArray ? null : NDArray.Scalar(fa);
                using var rightOwned = fb is NDArray ? null : NDArray.Scalar(fb);
                using var eq = np.equal(fa as NDArray ?? leftOwned, fb as NDArray ?? rightOwned);
                if (eq.size != 1)
                    throw new ValueError(eq.size == 0
                        ? "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty."
                        : "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                return Convert.ToBoolean(eq.GetAtIndex(0));
            }
            switch (fa)
            {
                case string sa:
                    return fb is string sb && string.Equals(sa, sb, StringComparison.Ordinal);
                case byte[] ba:
                    return fb is byte[] bb && ba.AsSpan().SequenceEqual(bb);
            }
            if (fb is string || fb is byte[])
                return false;
            // Python compares numbers by value across types (np.float64(5.0) == np.int64(5)); a complex compares both parts.
            if (fa is Complex || fb is Complex)
                return ToComplex(fa) == ToComplex(fb);
            if (IsIntegral(fa) && IsIntegral(fb))
                return ToInt128(fa) == ToInt128(fb);
            return ToDouble(fa) == ToDouble(fb);

            static bool IsIntegral(object v) => v is bool or sbyte or byte or short or ushort or int or uint or long or ulong or char;
            static Int128 ToInt128(object v) => v switch
            {
                bool bo => bo ? 1 : 0,
                char c => c,
                ulong u => u,
                _ => Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture),
            };
            static Complex ToComplex(object v) => v is Complex c ? c : new Complex(ToDouble(v), 0);
        }

        /// <summary>A boxed real number (any C# numeric primitive, <see cref="bool"/>, <see cref="char"/> or
        /// <see cref="Half"/>) as a <see cref="double"/>; <see cref="Half"/> is not <see cref="IConvertible"/>.</summary>
        /// <param name="v">The boxed number.</param>
        /// <returns>Its value.</returns>
        private static double ToDouble(object v)
            => v is Half h ? (double)h : Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture);

        /// <summary>
        ///     Sets the fill value of <paramref name="a"/> in place (NumPy's module-level <c>set_fill_value</c>).
        ///     Has effect ONLY on a <see cref="NDMaskedArray"/> (a plain array has no fill slot); a non-masked
        ///     operand is silently ignored, matching NumPy.
        /// </summary>
        /// <param name="a">The masked array to mutate.</param>
        /// <param name="fill_value">The new fill value (null restores the dtype default).</param>
        /// <exception cref="TypeError">See <see cref="NDMaskedArray.fill_value"/>: NumPy's method IS the property setter, so
        ///     the same <c>_check_fill_value</c> runs.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="NDMaskedArray.fill_value"/>.</exception>
        public void set_fill_value(object a, object fill_value)
        {
            if (a is NDMaskedArray m)
                m.fill_value = fill_value;
        }

        /// <summary>
        ///     Masks the non-finite (NaN/±inf) elements of <paramref name="a"/> AND writes the fill value into
        ///     their data slots (NumPy's <c>fix_invalid</c>) — like <see cref="masked_invalid"/> but the hidden
        ///     data becomes the fill rather than the original non-finite value. The result keeps <paramref name="a"/>'s own
        ///     <c>fill_value</c>; <paramref name="fill_value"/> only chooses what is WRITTEN into the masked slots
        ///     (matching NumPy, whose param sets the data, not the attribute).
        /// </summary>
        /// <param name="a">Data array-like (an existing mask is preserved and OR'd with the invalid mask).</param>
        /// <param name="mask">Optional extra mask to OR in (NumPy's <c>mask=</c>, read as <see cref="array"/> reads it);
        ///     null = none.</param>
        /// <param name="copy">Copy the data (default true) vs alias it (then the fill is written into
        ///     <paramref name="a"/>'s data, as NumPy does).</param>
        /// <param name="fill_value">Value written at the newly-masked (invalid) slots; null uses the result's own
        ///     fill value (<paramref name="a"/>'s, else the dtype default).</param>
        /// <returns>A masked array with non-finite slots masked and their data replaced by the fill.</returns>
        /// <exception cref="MaskError">The mask's size is neither one nor the data's (NumPy's constructor check, raised
        ///     before the data is examined).</exception>
        /// <exception cref="TypeError">Text data: NumPy's <c>np.isfinite</c> has no loop for it, whatever the size.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text mask for numeric data (its strings'
        ///     truth values), or a text fill written into numbers.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>a = masked_array(a, copy=copy, mask=mask, subok=True)</c>; <c>invalid =
        ///     np.logical_not(np.isfinite(a._data))</c>; <c>if not invalid.any(): return a</c>; <c>a._mask |= invalid</c>;
        ///     <c>a._data[invalid] = fill_value</c>. So the mask stage runs first (its MaskError wins), the mask is never
        ///     shrunk (an explicit all-False mask stays an array), and only the newly caught non-finite slots are
        ///     overwritten — pre-existing masked-but-finite data is left alone.
        ///     <para>Scoped by hand: the partial masks, the fill scalar and any operand materializations are reclaimed; only
        ///     the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray fix_invalid(object a, object mask = null, bool copy = true, object fill_value = null)
        {
            using var scope = NDScope.Open();
            var source = AsData(a);
            if (source.typecode == NPTypeCode.String)
            {
                // Text data: NumPy's constructor converts a mask (its truth values, which cannot fail) after checking its
                // size, and np.isfinite then refuses the text whatever the mask held. So the size check, then the TypeError.
                if (mask is not null && !ReferenceEquals(mask, nomask) && mask is not bool)
                    CheckMaskFitsData(AsData(mask), source);
                throw NDStringUfuncs.NotSupportedForInputTypes("isfinite");
            }
            // masked_array(a, copy=copy, mask=mask): the given mask ORs with a's own (keep_mask) after NumPy's shape rules,
            // and a's fill value and hardness ride along (the CLASS constructor: hard_mask=None inherits).
            var result = masked_array(a, mask, copy: copy);
            var d = result._data;
            // np.isfinite raises its no-loop TypeError for text, whatever the size; NonFiniteMask skips the call for the
            // exact dtypes, whose elements are all finite.
            var invalid = d.typecode == NPTypeCode.String ? np.logical_not(np.isfinite(d)) : NonFiniteMask(d);
            if (invalid is null || !np.any(invalid))
                return Yield(scope, result);
            // a._mask |= invalid (a new array: an aliased input mask is left untouched).
            result._mask = Or(result._mask, invalid);
            // a._data[invalid] = fill_value: only the newly caught non-finite slots are overwritten.
            var fv = fill_value ?? result.fill_value;
            np.copyto(d, fv as NDArray ?? NDArray.Scalar(fv), casting: "unsafe", where: invalid);
            return Yield(scope, result);
        }

        /// <summary>The <c>~</c> operator's backing (NumPy has no <c>ma.invert</c> module function): bitwise-NOT
        /// of the data through ndarray's <c>__invert__</c>, i.e. the plain ufunc and <see cref="UfuncOperator(NDMaskedArray,Func{NDArray,NDArray})"/>'s
        /// <c>__array_wrap__</c> (no fill-back: a masked slot holds the inverted data). Internal — reached via
        /// <c>~maskedArray</c>.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>A masked array of ~data, sharing <paramref name="a"/>'s mask object.</returns>
        /// <exception cref="TypeError">A float / complex / text operand (np.invert's no-loop text).</exception>
        internal NDMaskedArray Invert(NDMaskedArray a) => UfuncOperator(a, x => np.bitwise_not(x));

        /// <summary>
        ///     A unary operator that numpy.ma does NOT define on <c>MaskedArray</c> (<c>-x</c>, <c>~x</c>): Python falls back
        ///     to ndarray's operator, which calls the plain ufunc on the masked array, and NumPy then wraps the result with
        ///     <c>MaskedArray.__array_wrap__(obj, context)</c>. Its statements: <c>result = obj.view(type(self));
        ///     result._update_from(self)</c> (the fill value and the hard mask); <c>m = reduce(mask_or, [getmaskarray(arg) for
        ///     arg in inputs])</c>, which for ONE input is <c>getmaskarray(x)</c> itself — the operand's own mask OBJECT (a
        ///     write through either array reaches the other, as in NumPy), or a fresh all-False array when it has none; a 0-d
        ///     masked result is <c>masked</c>.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="f">The plain ufunc (<c>np.negative</c>, <c>np.bitwise_not</c>).</param>
        /// <returns>The wrapped result: the ufunc of the DATA at every slot (no fill-back, unlike <c>ma.negative</c>), the
        ///     operand's mask, its fill value and hardness; <see cref="NDMasked"/> for a masked 0-d operand.</returns>
        /// <exception cref="TypeError">The ufunc has no loop for the operand's dtype (NumPy's text, e.g. the bool
        ///     negative's).</exception>
        /// <remarks>Scoped by hand: only the result and an all-False mask built for a nomask operand leave.</remarks>
        internal NDMaskedArray UfuncOperator(NDMaskedArray x, Func<NDArray, NDArray> f)
        {
            using var scope = NDScope.Open();
            var result = f(x._data);
            var m = x._mask ?? np.zeros(DimsOf(x._data), np.@bool);
            // `if result is not self and result.shape == () and m: return masked`.
            if (result.ndim == 0 && m.GetBoolean())
                return Yield(scope, NDMasked);
            return Yield(scope, new NDMaskedArray(result, m).UpdateFrom(x));
        }

        /// <summary>
        ///     A binary operator that numpy.ma does NOT define on <c>MaskedArray</c> (<c>%</c>, <c>&amp;</c>, <c>|</c>,
        ///     <c>^</c>): ndarray's operator calls the plain ufunc with the operands AS GIVEN (a C# number stays NumPy's weak
        ///     Python scalar, so <c>float32 % 3.0</c> stays float32), and <c>MaskedArray.__array_wrap__</c> of the first masked
        ///     operand wraps the result: <c>_update_from</c> that operand; <c>m = mask_or(getmaskarray(a),
        ///     getmaskarray(b))</c>, a NEW array that SHRINKS to nomask when nothing is masked; for a ufunc with a domain
        ///     (remainder's <c>_DomainSafeDivide</c>, <c>|a| * tiny &gt;= |b|</c>) the slots outside it are overwritten with
        ///     <c>ufunc_fills[remainder][-1]</c> = 1 and masked; a 0-d masked result is <c>masked</c>.
        /// </summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="f">The plain ufunc.</param>
        /// <param name="remainderDomain">True for <c>%</c> (np.remainder's safe-divide domain).</param>
        /// <returns>The wrapped result — the raw ufunc value at every slot not outside the domain (no fill-back, unlike the
        ///     <c>ma.*</c> functions) —, or <see cref="NDMasked"/> for a masked 0-d result.</returns>
        /// <exception cref="TypeError">The ufunc has no loop for the operands (NumPy's text).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast together.</exception>
        /// <remarks>Scoped by hand: the operands' full masks, the OR and the domain predicate are reclaimed; only the
        /// result's data and mask leave.</remarks>
        internal NDMaskedArray UfuncOperator(object a, object b, Func<NDArray, NDArray, NDArray> f, bool remainderDomain = false)
        {
            using var scope = NDScope.Open();
            var da = AsData(a);
            var db = AsData(b);
            var result = f(da, db);
            var mo = mask_or(getmaskarray(a), getmaskarray(b));
            NDArray m = ReferenceEquals(mo, nomask) ? null : mo;
            if (remainderDomain)
            {
                // `d = domain(*input_args)` on the raw data; `np.copyto(result, fill_value, where=d)`; `m |= d`.
                var d = SafeDivideDomain(da, db);
                if (np.any(d))
                {
                    if (result.ndim > 0)
                        np.copyto(result, NDArray.Scalar(1L), casting: "unsafe", where: d);
                    m = m is null ? d : np.logical_or(m, d);
                }
            }
            if (m is not null && !m.Shape.Equals(result.Shape))
                m = np.broadcast_to(m, result.Shape).copy();
            // `if result is not self and result.shape == () and m: return masked`.
            if (result.ndim == 0 && m is not null && m.GetBoolean())
                return Yield(scope, NDMasked);
            var r = new NDMaskedArray(result, m);
            // The wrap is the first masked operand's (NumPy's __array_priority__ tie goes to the first).
            if (a is NDMaskedArray sa)
                r.UpdateFrom(sa);
            else if (b is NDMaskedArray sb)
                r.UpdateFrom(sb);
            return Yield(scope, r);
        }

        /// <summary>
        ///     The ordering operators (<c>&lt;</c>, <c>&gt;</c>, <c>&lt;=</c>, <c>&gt;=</c>), NumPy's
        ///     <c>MaskedArray._comparison(self, other, compare)</c>: the mask is <c>mask_or(self.mask, getmask(other),
        ///     copy=True)</c> — a COPY, shrunk to nomask when nothing is masked —; the data is <c>compare(self.data,
        ///     getdata(other))</c> with NO fill-back (a masked slot holds the comparison of its data, where the <c>ma.less</c>
        ///     function writes the minuend's data back) and <c>getdata</c> making a C# number a STRONG default-width array
        ///     (a float32 array compares with <c>0.1</c> in float64); a 0-d result is <c>masked</c> or the plain value; the
        ///     result takes <paramref name="self"/>'s hard mask and its fill value CAST TO BOOL (<c>_check_fill_value(fill,
        ///     np.bool)</c>: <c>0.0</c> → False, NaN or any other number → True, text by non-emptiness).
        /// </summary>
        /// <param name="self">The masked operand whose method runs: the left one, or the right one of a reflected
        ///     comparison (<c>1 &lt; x</c> is <c>x.__gt__(1)</c>).</param>
        /// <param name="other">The other operand.</param>
        /// <param name="compare">The comparison, applied as <c>compare(self.data, other)</c>.</param>
        /// <returns>The masked boolean result, a 0-d unmasked one, or <see cref="NDMasked"/>.</returns>
        /// <exception cref="TypeError">No comparison loop for the pair (NumPy's text).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads text, or a text fill value's truth
        ///     is needed: the next phase.</exception>
        /// <remarks>Scoped by hand: the strong operand conversions and the combined mask's intermediates are reclaimed; only
        /// the result's data and mask leave.</remarks>
        internal NDMaskedArray ComparisonOperator(NDMaskedArray self, object other, Func<NDArray, NDArray, NDArray> compare)
        {
            using var scope = NDScope.Open();
            var mo = mask_or(self._mask is null ? nomask : self._mask, getmask(other));
            NDArray mask = ReferenceEquals(mo, nomask) ? null : mo;
            var sdata = self._data;
            var odata = AsData(other);
            // `getdata(other)`: a C# number becomes NumPy's strong default-width array, promoted with the data.
            PromoteMaScalar(self, other, ref sdata, ref odata);
            var check = compare(sdata, odata);
            // `if isinstance(check, (np.bool, bool)): return masked if mask else check`.
            if (check.ndim == 0)
                return Yield(scope, mask is not null && mask.GetBoolean() ? NDMasked : new NDMaskedArray(check, null));
            if (mask is not null && !mask.Shape.Equals(check.Shape))
                mask = np.broadcast_to(mask, check.Shape).copy();
            var r = new NDMaskedArray(check, mask).UpdateFrom(self);
            // `if check._fill_value is not None: check._fill_value = _check_fill_value(fill, np.bool)` (the default bool
            // fill when that conversion fails).
            if (r._fill_value is not null)
                r._fill_value = FillAsBool(r._fill_value);
            return Yield(scope, r);
        }

        /// <summary>
        ///     <c>_check_fill_value(fill, np.bool)</c> of an inherited fill value: <c>np.asarray(fill, dtype=bool)</c>, the
        ///     truth of the value — a number is True unless it is zero (NaN is True), a complex unless both parts are zero,
        ///     text unless it is empty (read through the text-to-bool cast, the next phase for non-empty text).
        /// </summary>
        /// <param name="fill">The stored fill value (a boxed C# scalar, <c>byte[]</c>, string, or 0-d array).</param>
        /// <returns>The boxed bool fill.</returns>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text fill (its truth reads the text).</exception>
        private static object FillAsBool(object fill)
        {
            switch (fill)
            {
                case bool b:
                    return b;
                case NDArray arr:
                    return arr.astype(np.@bool).GetBoolean();
                case string str:
                    return NDArray.Scalar(str).astype(np.@bool).GetBoolean();
                case byte[] bytes:
                    return np.array(new object[] { bytes }, dtype: "S").astype(np.@bool).GetAtIndex<bool>(0);
                case Complex c:
                    return c != Complex.Zero;
                default:
                    // Any C# number: `bool(x)` is `x != 0`, and NaN != 0.
                    return NDArray.Scalar(fill).astype(np.@bool).GetBoolean();
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
        /// <param name="a">The operand (<see cref="NDMaskedArray"/>/<see cref="NDArray"/>/scalar).</param>
        /// <param name="domain">Optional invalid-input predicate: given the data, returns a boolean mask of
        /// elements to mask; null for a domain-free op.</param>
        /// <returns>The masked result, or the <see cref="NDMasked"/> constant for a fully-masked 0-D result. An array
        /// result takes a masked operand's fill value (raw) and hard-mask flag (<see cref="InheritFromOperands"/>).</returns>
        /// <remarks>Scoped by hand — the one scope every unary masked ufunc rides: the operand materialization,
        /// the domain/non-finite predicate masks and their ORs, the broadcast mask view and the fill-back scalar
        /// are all reclaimed here; only the result's data and mask leave (a fully-masked 0-D result's computed
        /// value is reclaimed too, since the <see cref="NDMasked"/> singleton is returned in its place).</remarks>
        private NDMaskedArray Unary(Func<NDArray, NDArray> f, object a, Func<NDArray, NDArray> domain = null)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            var result = f(d);
            NDArray m;
            if (domain != null)
            {
                // Domained: mask non-finite outputs, the domain's rejected inputs, and any incoming mask.
                m = Or(NonFiniteMask(result), domain(d));
                m = Or(m, (a as NDMaskedArray)?._mask);
            }
            else
            {
                // Domain-free (abs/negative/sin/…): the mask is simply carried through unchanged.
                m = (a as NDMaskedArray)?._mask;
            }

            // NumPy's unary fill-back is copyto(result, d, where=m) with the DEFAULT 'same_kind' casting,
            // caught if it fails — so a dtype-changing unary (e.g. angle: complex→float) keeps its computed
            // value at masked slots rather than force-casting the input in.
            return Yield(scope, InheritFromOperands(Wrap(result, m, d, casting: "same_kind"), a));
        }

        /// <summary>
        ///     The last statement of NumPy's <c>_MaskedUnaryOperation</c> / <c>_MaskedBinaryOperation</c> /
        ///     <c>_DomainedBinaryOperation.__call__</c>: <c>masked_result._update_from(a)</c> when <paramref name="a"/> is a
        ///     masked array, else <c>_update_from(b)</c> when <paramref name="b"/> is one. The result takes that operand's
        ///     fill value RAW (an int64 operand's fill stays an int64 on a float64 result, as NumPy's getter reports it) and
        ///     its hard-mask flag, so <c>hard + 1</c> is hard and <c>soft + hard</c> soft.
        /// </summary>
        /// <param name="result">The wrapped result (<see cref="Wrap"/>'s answer).</param>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand, or null for a unary operation.</param>
        /// <returns><paramref name="result"/>, updated in place.</returns>
        /// <remarks>The <see cref="NDMasked"/> singleton and a 0-d result (NumPy's scalar, which has no fill value) are
        /// returned untouched: the singleton is shared by every caller and must never be mutated.</remarks>
        private NDMaskedArray InheritFromOperands(NDMaskedArray result, object a, object b = null)
        {
            if (ReferenceEquals(result, NDMasked) || result._data.ndim == 0)
                return result;
            if (a is NDMaskedArray sa)
                result.UpdateFrom(sa);
            else if (b is NDMaskedArray sb)
                result.UpdateFrom(sb);
            return result;
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
        /// <param name="promoteScalar">True (the ufunc family) makes a bare C# scalar operand numpy.ma's strong
        ///     default-width array (<see cref="PromoteMaScalar"/>).</param>
        /// <returns>The masked result, or the <see cref="NDMasked"/> constant for a fully-masked 0-D result. An array
        /// result takes the fill value (raw) and hard-mask flag of <paramref name="a"/> when it is a masked array, else of
        /// <paramref name="b"/> (<see cref="InheritFromOperands"/>).</returns>
        /// <remarks>Scoped by hand — the one scope every binary masked ufunc rides: operand materializations, the
        /// numpy.ma scalar-widening casts, the OR'd mask intermediates, the broadcast mask view and the fill-back
        /// scalar are reclaimed; only the result's data and mask leave.</remarks>
        private NDMaskedArray Binary(Func<NDArray, NDArray, NDArray> f, object a, object b, bool promoteScalar = true)
        {
            using var scope = NDScope.Open();
            var da = AsData(a);
            var db = AsData(b);
            // numpy.ma treats a bare scalar as a STRONG default-width array (int→int64, float→float64), so a
            // binary op with a scalar operand widens like NumPy's ma rather than NumSharp's weak-scalar np path.
            // EXCEPTION: left_shift/right_shift are plain functions in numpy.ma (NOT _MaskedBinaryOperation), so
            // they do NOT wrap the shift-count scalar as strong — they keep the array's dtype
            // (ma.left_shift(int16, 1) → int16, not int64). Those callers pass promoteScalar:false.
            if (promoteScalar)
                PromoteMaScalar(a, b, ref da, ref db);
            var result = f(da, db);
            // `m = umath.logical_or(ma, getmaskarray(b))` (or the mirror, or both masks): a NEW array whenever either
            // operand has a mask — never an operand's own mask object, so a masked write through the result cannot reach
            // the operand — and nomask when neither has one.
            var ma0 = (a as NDMaskedArray)?._mask;
            var mb0 = (b as NDMaskedArray)?._mask;
            var m = ma0 is null && mb0 is null ? null : np.logical_or(ma0 ?? NDArray.Scalar(false), mb0 ?? NDArray.Scalar(false));
            return Yield(scope, InheritFromOperands(Wrap(result, m, da), a, b));
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
        /// <returns>The masked result, or the <see cref="NDMasked"/> constant for a fully-masked 0-D result. An array
        /// result inherits like <see cref="Binary"/>'s.</returns>
        /// <remarks>Scoped by hand — the one scope every domained masked ufunc (divide/remainder/…) rides: the
        /// domain and non-finite predicate masks, their ORs, the widening casts, the broadcast mask view and the
        /// zero/fill-back scalars are reclaimed; only the result's data and mask leave.</remarks>
        private NDMaskedArray DomainedBinary(Func<NDArray, NDArray, NDArray> f, Func<NDArray, NDArray, NDArray> domain, object a, object b)
        {
            using var scope = NDScope.Open();
            var da = AsData(a);
            var db = AsData(b);
            // Same scalar-strong promotion as Binary (e.g. ma.floor_divide(int32, 2) → int64).
            PromoteMaScalar(a, b, ref da, ref db);
            var result = f(da, db);
            var m = Or(NonFiniteMask(result), domain(da, db));
            m = Or(m, (a as NDMaskedArray)?._mask);
            m = Or(m, (b as NDMaskedArray)?._mask);
            // Domained fill-back differs from the plain binary: NumPy zeroes masked slots then re-adds da,
            // which we reproduce as "put da back where it can safely cast, else leave zero".
            return Yield(scope, InheritFromOperands(Wrap(result, m, da, domainedZeroFill: true), a, b));
        }

        /// <summary>
        ///     Shared tail of all three wrappers: materializes the mask to the result shape, handles the 0-D
        ///     scalar case, writes the fill-back value into masked slots, and boxes everything into a
        ///     <see cref="NDMaskedArray"/>.
        /// </summary>
        /// <param name="result">The freshly computed (owned, writeable) data result.</param>
        /// <param name="m">The propagated mask, or null for no mask.</param>
        /// <param name="fillbackSource">The array whose values are restored at masked positions (the input for
        /// unary, the left operand for binary).</param>
        /// <param name="domainedZeroFill">When true, use the domained fill-back (zero then re-add) instead of
        /// the plain copy-back.</param>
        /// <returns>The masked result, or the <see cref="NDMasked"/> constant when a 0-D result is fully masked.</returns>
        private NDMaskedArray Wrap(NDArray result, NDArray m, NDArray fillbackSource, bool domainedZeroFill = false, string casting = "unsafe")
        {
            // Keep the stored mask shape-consistent with the data (broadcast predicates/operand masks up).
            // `is not null` throughout — `m != null` would invoke NDArray's ELEMENTWISE `!=`.
            if (m is not null && !m.Shape.Equals(result.Shape))
                m = np.broadcast_to(m, result.Shape).copy();

            // 0-D result: NumPy returns the `masked` singleton if masked, else the bare scalar. We return the
            // singleton when masked and a 0-D masked array otherwise (a deliberate C# type-consistency choice).
            if (result.ndim == 0)
                return (m is not null && np.any(m)) ? NDMasked : new NDMaskedArray(result, null);

            // Restore data at masked positions so `.data` is meaningful there (masked values are still hidden).
            // No `np.any(m)` gate: copyto with an all-false mask is already a no-op, and the scan would cost a
            // full extra O(n) pass — NumPy's unary path likewise fills back unconditionally.
            if (m is not null)
                FillBack(result, fillbackSource, m, domainedZeroFill, casting);

            return new NDMaskedArray(result, m);
        }

        /// <summary>
        ///     Writes the pre-op data back into masked slots of <paramref name="result"/>. The plain path copies
        ///     <paramref name="source"/> straight in (<c>copyto(result, da, where=m)</c>). The domained path is NumPy's
        ///     <c>_DomainedBinaryOperation</c> arithmetic exactly: zero the masked slots, then — when <c>m * da</c> casts
        ///     safely to the result dtype — ADD <c>m * da</c> to the WHOLE result. That add is observable in the bits: a masked
        ///     slot gets <c>0 + da</c> (a <c>-0.0</c> dividend comes back <c>+0.0</c>), and an unmasked slot gets
        ///     <c>result + 0 * da</c> (a <c>-0.0</c> quotient of a positive dividend, such as <c>0.0 / -1</c>, comes back
        ///     <c>+0.0</c>). Both paths are best-effort: an incompatible cast is swallowed, leaving the computed/zeroed value,
        ///     as NumPy's <c>try/except</c> does.
        /// </summary>
        /// <param name="result">The result array to patch in place.</param>
        /// <param name="source">The data to restore at masked positions (the left operand, <c>da</c>).</param>
        /// <param name="m">The boolean mask (result-shaped).</param>
        /// <param name="domainedZeroFill">Selects the domained zero-then-add path.</param>
        /// <param name="casting">The plain path's copy casting: <c>"same_kind"</c> for a unary op (NumPy's default),
        /// <c>"unsafe"</c> for a binary one.</param>
        private static void FillBack(NDArray result, NDArray source, NDArray m, bool domainedZeroFill, string casting)
        {
            try
            {
                if (domainedZeroFill)
                {
                    // NumPy: np.copyto(result, 0, casting='unsafe', where=m); masked_da = umath.multiply(m, da);
                    // if np.can_cast(masked_da.dtype, result.dtype, casting='safe'): result += masked_da.
                    np.copyto(result, NDArray.Scalar(0), casting: "unsafe", where: m);
                    var maskedDa = np.multiply(m, source);
                    if (np.can_cast(maskedDa.dtype, result.dtype, "safe"))
                        np.add(result, maskedDa, @out: result);
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
        ///     The masked-array egress of a hand-written <see cref="NDScope"/> — the <see cref="NDMaskedArray"/>
        ///     counterpart of <see cref="NDScope.Returns{T}(T)"/>, and the reason this module scopes by hand: the
        ///     <c>[NDScoped]</c> weaver only yields NDArray-shaped returns and <see cref="INDArrayCarrier"/>
        ///     STRUCTS, so a method returning this reference type must open its scope itself and route every
        ///     return through here. Every array the method built that the result does NOT carry is then reclaimed
        ///     when the scope closes, instead of waiting for the finalizer.
        /// </summary>
        /// <param name="scope">The method's open scope (the one whose temporaries are being reclaimed).</param>
        /// <param name="result">The masked array being returned; null (no result) passes through.</param>
        /// <returns><paramref name="result"/>, unchanged.</returns>
        /// <remarks>
        ///     A masked array owns nothing itself — it aliases a data array and an optional mask — so its egress
        ///     is the egress of those two buffers: each is handed back (re-tracked into the parent scope, which
        ///     still reclaims them if the caller drops the result). An array this scope never tracked — an
        ///     operand's data or mask passed through untouched, or the process-wide
        ///     <see cref="NDMasked"/>/<see cref="nomask"/> singletons — is a no-op there, which is what makes
        ///     "yield every result" a safe blanket rule, including the singleton returns.
        ///     <para>
        ///     NEVER use this for an array stored into a FIELD of a longer-lived masked array (a mask created by an
        ///     in-place write): re-tracking it into the caller's scope would dispose it while the target still
        ///     holds it. A field store is <see cref="NDScope.Detach(NDArray)"/> (see <see cref="Own"/>).
        ///     </para>
        /// </remarks>
        internal static NDMaskedArray Yield(NDScope scope, NDMaskedArray result)
        {
            if (result is not null)
            {
                scope.Returns(result._data);
                scope.Returns(result._mask);
            }
            return result;
        }

        /// <summary>
        ///     The <see cref="NDMaskedArray"/>[] egress of a hand-written scope (a split/sequence result): every
        ///     element's data and mask are yielded exactly as <see cref="Yield(NDScope, NDMaskedArray)"/> does for
        ///     one. A null array and null elements pass through.
        /// </summary>
        /// <param name="scope">The method's open scope.</param>
        /// <param name="results">The masked arrays being returned.</param>
        /// <returns><paramref name="results"/>, unchanged.</returns>
        internal static NDMaskedArray[] Yield(NDScope scope, NDMaskedArray[] results)
        {
            if (results is not null)
                for (int i = 0; i < results.Length; i++)
                    Yield(scope, results[i]);
            return results;
        }

        /// <summary>
        ///     Takes ownership of an array being STORED into a masked array's field (a mask an in-place write
        ///     creates or rebuilds): removes it from every ambient scope so no scope — the writing method's own or
        ///     its caller's — can dispose it while the target masked array still references it. The array then
        ///     lives exactly as long as the masked array holding it (deterministically released when a caller
        ///     disposes that masked array's parts, otherwise by the finalizer), matching the poly1d coefficient
        ///     rule: a field egress is a detach, never a <see cref="NDScope.Returns{T}(T)"/>.
        /// </summary>
        /// <param name="nd">The array being stored; null passes through.</param>
        /// <returns><paramref name="nd"/>, unchanged.</returns>
        internal static NDArray Own(NDArray nd)
        {
            NDScope.Detach(nd);
            return nd;
        }

        /// <summary>
        ///     Normalizes any accepted operand to its data <see cref="NDArray"/>: unwraps a
        ///     <see cref="NDMaskedArray"/>, passes an <see cref="NDArray"/> through, and converts a C# scalar or
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
                case NDMaskedArray m: return m._data;
                // A numpy.matrix operand is read as its plain view: numpy.ma's matrix base class is not modelled
                // (NDMaskedArray's constructor), so the module computes with plain-array semantics throughout.
                case NDArray nd: return NDMatrixBoundary.PlainOf(nd);
                case Array arr: return np.array(arr); // C# array-like (double[], int[,], …)
                default: return NDArray.Scalar(a);    // boxed C# scalar (int/double/bool/…)
            }
        }

        /// <summary>
        ///     A VALUE operand (a comparison value, a bound of <see cref="masked_inside"/>, a fill) as an array: like
        ///     <see cref="AsData"/>, except that a C# <see cref="byte"/>[] is NumPy's <c>bytes</c> scalar (the house map) — a
        ///     0-d <c>S</c> array — where <see cref="AsData"/> reads it as a <c>uint8</c> array of values.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The operand array (a new 0-d array for a scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        private static NDArray AsValueOperand(object value) => value is byte[] bytes ? NDArray.Scalar(bytes) : AsData(value);

        /// <summary>
        ///     The dtype numpy.ma gives a BARE scalar operand of a binary op — its STRONG default width (python
        ///     int→int64, python float→float64, bool→bool, complex→complex128). numpy.ma wraps a scalar as
        ///     <c>np.asarray(scalar)</c> (a STRONG array) rather than a weak NEP50 scalar, which is why
        ///     <c>ma.multiply(int32, 2)</c> is int64 while the plain-np <c>np.multiply(int32, 2)</c> is int32.
        ///     Returns null for a non-scalar operand (<see cref="NDArray"/>/<see cref="NDMaskedArray"/>/C# array/null),
        ///     whose own dtype already applies.
        /// </summary>
        /// <param name="x">A binary-op operand.</param>
        /// <returns>The scalar's strong dtype, or null when <paramref name="x"/> is not a bare scalar.</returns>
        private static NPTypeCode? BareScalarStrongDtype(object x) => x switch
        {
            NDArray => null,
            NDMaskedArray => null,
            Array => null,
            null => null,
            bool => NPTypeCode.Boolean,
            // Every C# integer type maps to int64 (python int's width) and every C# float type to float64
            // (python float's), the closest analogs of NumPy's scalar-wrapping — the common int/long/double/float
            // literal cases are exact; the odd narrow-scalar cases (byte/short) have no python literal analog.
            sbyte or byte or short or ushort or int or uint or long or ulong or char => NPTypeCode.Int64,
            Half or float or double or decimal => NPTypeCode.Double,
            Complex => NPTypeCode.Complex,
            _ => null
        };

        /// <summary>
        ///     Reproduces numpy.ma's scalar promotion for a binary op: when EITHER operand is a bare scalar, both
        ///     data arrays are widened to <c>promote_types(effective a, effective b)</c>, where a bare scalar
        ///     contributes its <see cref="BareScalarStrongDtype"/> and an array its own dtype — so
        ///     <c>ma.multiply(int32, 2)</c> computes at (and returns) int64, matching NumPy. The INPUTS are widened
        ///     (not just the result) so a value that overflows the narrow dtype is computed wide, as NumPy does
        ///     (<c>ma.multiply(int32 2e9, 2) == 4_000_000_000</c>, not the int32-wrapped value). A no-op for
        ///     array×array (each operand keeps its own dtype, and the underlying op already promotes them).
        /// </summary>
        /// <param name="a">Left operand (the original object, to detect a bare scalar).</param>
        /// <param name="b">Right operand.</param>
        /// <param name="da">Left data array — widened in place when promotion applies.</param>
        /// <param name="db">Right data array — widened in place when promotion applies.</param>
        /// <remarks>With a TEXT operand only the bare scalar is converted (to its strong width): text and numbers share no
        /// numeric dtype, and NumPy's ufunc reports the pair itself (<c>ma.greater(u, 1)</c> names <c>Int64DType</c>).</remarks>
        private static void PromoteMaScalar(object a, object b, ref NDArray da, ref NDArray db)
        {
            var sa = BareScalarStrongDtype(a);
            var sb = BareScalarStrongDtype(b);
            if (sa is null && sb is null)
                return; // array×array: nothing to widen — the underlying op promotes them itself.
            // A text operand has no numeric promotion with the scalar: numpy.ma only makes the scalar its strong
            // default-width array (getdata(1) is int64), and the ufunc then answers for the pair — no loop
            // ((StrDType, Int64DType)) or, for two strings, the comparison of their text.
            if (da.typecode == NPTypeCode.String || db.typecode == NPTypeCode.String)
            {
                if (sa is { } strongA)
                    da = da.astype(strongA, copy: false);
                if (sb is { } strongB)
                    db = db.astype(strongB, copy: false);
                return;
            }
            NPTypeCode ea = sa ?? da.typecode;
            NPTypeCode eb = sb ?? db.typecode;
            var td = np.promote_types(ea, eb);
            // astype(copy:false) is a no-op when the dtype already matches, so no redundant allocation.
            da = da.astype(td, copy: false);
            db = db.astype(td, copy: false);
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
        ///     Masks every 1-D vector of <paramref name="a"/> along <paramref name="axis"/> that contains ANY
        ///     masked value (NumPy's <c>_mask_propagate</c>): the mask is OR'd with its own <c>any(axis)</c>
        ///     reduction broadcast back, so a single masked entry taints the whole line. Used by the strict
        ///     <see cref="dot(object,object,bool)"/> path. An operand with no mask (or none set along the axis)
        ///     is returned unchanged.
        /// </summary>
        /// <param name="a">Operand (masked array or plain array-like).</param>
        /// <param name="axis">The axis whose lines are tainted wholesale by any masked element.</param>
        /// <returns>A masked array whose mask has been propagated along <paramref name="axis"/>.</returns>
        /// <remarks>Scoped by hand: the per-line <c>any</c> reduction and its broadcast view are reclaimed; the
        /// returned array carries the operand's data and the freshly propagated mask.</remarks>
        private NDMaskedArray MaskPropagate(object a, int axis)
        {
            using var scope = NDScope.Open();
            var ma_ = array(a);
            var m = ma_._mask;
            if (m is null || !np.any(m))
                return Yield(scope, ma_);
            // any(m, axis, keepdims) has size 1 on `axis`; broadcasting it back and OR-ing taints the whole line.
            var line = np.any(m, axis, null, true);
            var newmask = np.logical_or(m, np.broadcast_to(line, m.Shape));
            return Yield(scope, new NDMaskedArray(ma_._data, newmask));
        }

        /// <summary>
        ///     A fresh C-contiguous <see cref="Shape"/> of <paramref name="a"/>'s DIMENSIONS only — never its
        ///     strides/offset/bufferSize. Use this for EVERY mask/scratch allocation over an operand's shape.
        /// </summary>
        /// <remarks>
        ///     Allocating a mask with <c>np.zeros(a.Shape)</c> is a silent-corruption FOOTGUN: for a
        ///     NON-CONTIGUOUS <paramref name="a"/> (a slice / transpose / strided or offset view) the full
        ///     <see cref="Shape"/> carries the view's strides AND the larger base bufferSize, so
        ///     <c>np.zeros</c> builds a STRIDED zeros array over that base buffer and materializing it reads the
        ///     uninitialized tail as garbage — e.g. <c>getmaskarray</c> of a nomask <c>[::2]</c> view returned
        ///     <c>01</c> bytes instead of all-False, which every manip/constructor op then inherited. The mask
        ///     must be sized by the DIMENSIONS, exactly as NumPy allocates it (<c>make_mask_none(a.shape)</c>).
        /// </remarks>
        /// <param name="a">The array whose logical dimensions the allocation should match.</param>
        /// <returns>A fresh contiguous <see cref="Shape"/> with <paramref name="a"/>'s dimensions, stride 0 offset.</returns>
        private static Shape DimsOf(NDArray a) => Shape.Adopt(a.Shape.dimensions);

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
        /// <remarks><c>tiny</c> is NumPy's <c>np.finfo(float).tiny</c>, an <c>np.float64</c> scalar — STRONG — so
        /// <c>|a| * tiny</c> is float64 for every <paramref name="a"/> dtype (a float16 or float32 <c>|a|</c> would turn
        /// tiny into 0 and an inf <c>|a|</c> into NaN). <c>|a|</c> is widened first, which is the float64 loop's own cast.</remarks>
        private static NDArray SafeDivideDomain(NDArray a, NDArray b)
            => np.greater_equal(np.multiply(np.abs(a).astype(np.float64, copy: false), SafeDivideTiny), np.abs(b));

        // ─────────────────────────────────────────────────────────────────────────────
        //  Unary ufuncs — domain-free (mask passes through)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise absolute value over the data; the mask is unchanged. Alias of <see cref="absolute"/>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of |data|.</returns>
        public NDMaskedArray abs(object a) => Unary(x => np.abs(x), a);

        /// <summary>Element-wise absolute value over the data; the mask is unchanged (same op as <see cref="abs"/>).</summary>
        /// <param name="a">Operand.</param><returns>A masked array of |data|.</returns>
        public NDMaskedArray absolute(object a) => Unary(x => np.abs(x), a);

        /// <summary>Element-wise negation; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of -data.</returns>
        public NDMaskedArray negative(object a) => Unary(x => np.negative(x), a);

        /// <summary>Float-only absolute value (promotes ints to float); mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of the float |data|.</returns>
        public NDMaskedArray fabs(object a) => Unary(x => np.fabs(x), a);

        /// <summary>Complex conjugate (real data passes through); mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of conj(data).</returns>
        public NDMaskedArray conjugate(object a) => Unary(x => np.conjugate(x), a);

        /// <summary>Angle (argument) of the elements; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of angle(data).</returns>
        public NDMaskedArray angle(object a) => Unary(x => np.angle(x), a);

        /// <summary>Round to nearest integer, half-to-even; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of round(data).</returns>
        public NDMaskedArray around(object a) => Unary(x => np.around(x), a);

        /// <summary>Floor; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of floor(data).</returns>
        public NDMaskedArray floor(object a) => Unary(x => np.floor(x), a);

        /// <summary>Ceiling; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of ceil(data).</returns>
        public NDMaskedArray ceil(object a) => Unary(x => np.ceil(x), a);

        /// <summary>e^x; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of exp(data).</returns>
        public NDMaskedArray exp(object a) => Unary(x => np.exp(x), a);

        /// <summary>Sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sin(data).</returns>
        public NDMaskedArray sin(object a) => Unary(x => np.sin(x), a);

        /// <summary>Cosine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of cos(data).</returns>
        public NDMaskedArray cos(object a) => Unary(x => np.cos(x), a);

        /// <summary>Hyperbolic sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sinh(data).</returns>
        public NDMaskedArray sinh(object a) => Unary(x => np.sinh(x), a);

        /// <summary>Hyperbolic cosine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of cosh(data).</returns>
        public NDMaskedArray cosh(object a) => Unary(x => np.cosh(x), a);

        /// <summary>Hyperbolic tangent; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of tanh(data).</returns>
        public NDMaskedArray tanh(object a) => Unary(x => np.tanh(x), a);

        /// <summary>Arctangent; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arctan(data).</returns>
        public NDMaskedArray arctan(object a) => Unary(x => np.arctan(x), a);

        /// <summary>Inverse hyperbolic sine; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arcsinh(data).</returns>
        public NDMaskedArray arcsinh(object a) => Unary(x => np.arcsinh(x), a);

        /// <summary>Logical NOT of the data; mask unchanged.</summary>
        /// <param name="a">Operand.</param><returns>A masked boolean array of !data.</returns>
        public NDMaskedArray logical_not(object a) => Unary(x => np.logical_not(x), a);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Unary ufuncs — domained (invalid inputs are additionally masked)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Square root; additionally MASKS inputs <c>&lt; 0</c> (and non-finite results) rather than
        /// letting them produce NaN, which is the whole point of the masked variant.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of sqrt(data) with negatives masked.</returns>
        public NDMaskedArray sqrt(object a) => Unary(x => np.sqrt(x), a, d => DomainGreaterEqual(d, 0.0));

        /// <summary>Natural log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log(data) with non-positives masked.</returns>
        public NDMaskedArray log(object a) => Unary(x => np.log(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Base-2 log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log2(data) with non-positives masked.</returns>
        public NDMaskedArray log2(object a) => Unary(x => np.log2(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Base-10 log; additionally masks inputs <c>&lt;= 0</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of log10(data) with non-positives masked.</returns>
        public NDMaskedArray log10(object a) => Unary(x => np.log10(x), a, d => DomainGreater(d, 0.0));

        /// <summary>Tangent; additionally masks inputs near its poles (<c>|cos(x)| &lt; 1e-35</c>).</summary>
        /// <param name="a">Operand.</param><returns>A masked array of tan(data) with pole-neighbors masked.</returns>
        public NDMaskedArray tan(object a) => Unary(x => np.tan(x), a, d => DomainTan(d, 1e-35));

        /// <summary>Arcsine; additionally masks inputs outside <c>[-1, 1]</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arcsin(data) with out-of-domain inputs masked.</returns>
        public NDMaskedArray arcsin(object a) => Unary(x => np.arcsin(x), a, d => DomainCheckInterval(d, -1.0, 1.0));

        /// <summary>Arccosine; additionally masks inputs outside <c>[-1, 1]</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arccos(data) with out-of-domain inputs masked.</returns>
        public NDMaskedArray arccos(object a) => Unary(x => np.arccos(x), a, d => DomainCheckInterval(d, -1.0, 1.0));

        /// <summary>Inverse hyperbolic cosine; additionally masks inputs <c>&lt; 1</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arccosh(data) with sub-1 inputs masked.</returns>
        public NDMaskedArray arccosh(object a) => Unary(x => np.arccosh(x), a, d => DomainGreaterEqual(d, 1.0));

        /// <summary>Inverse hyperbolic tangent; additionally masks inputs outside <c>(-1, 1)</c>.</summary>
        /// <param name="a">Operand.</param><returns>A masked array of arctanh(data) with out-of-domain inputs masked.</returns>
        public NDMaskedArray arctanh(object a) => Unary(x => np.arctanh(x), a, d => DomainCheckInterval(d, -1.0 + 1e-15, 1.0 - 1e-15));

        // ─────────────────────────────────────────────────────────────────────────────
        //  Binary ufuncs — mask = OR of the two input masks
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise sum; result is masked wherever EITHER operand was masked.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a+b.</returns>
        public NDMaskedArray add(object a, object b) => Binary((x, y) => np.add(x, y), a, b);

        /// <summary>Element-wise difference; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a-b.</returns>
        public NDMaskedArray subtract(object a, object b) => Binary((x, y) => np.subtract(x, y), a, b);

        /// <summary>Element-wise product; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a*b.</returns>
        public NDMaskedArray multiply(object a, object b) => Binary((x, y) => np.multiply(x, y), a, b);

        /// <summary>Two-argument arctangent; masked where either operand was.</summary>
        /// <param name="a">y operand.</param><param name="b">x operand.</param>
        /// <returns>A masked array of arctan2(a, b).</returns>
        public NDMaskedArray arctan2(object a, object b) => Binary((x, y) => np.arctan2(x, y), a, b);

        /// <summary>Euclidean hypotenuse; masked where either operand was.</summary>
        /// <param name="a">Left leg.</param><param name="b">Right leg.</param>
        /// <returns>A masked array of hypot(a, b).</returns>
        public NDMaskedArray hypot(object a, object b) => Binary((x, y) => np.hypot(x, y), a, b);

        /// <summary>Element-wise equality; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a==b.</returns>
        public NDMaskedArray equal(object a, object b) => Binary((x, y) => np.equal(x, y), a, b);

        /// <summary>Element-wise inequality; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a!=b.</returns>
        public NDMaskedArray not_equal(object a, object b) => Binary((x, y) => np.not_equal(x, y), a, b);

        /// <summary>Element-wise less-than; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&lt;b.</returns>
        public NDMaskedArray less(object a, object b) => Binary((x, y) => np.less(x, y), a, b);

        /// <summary>Element-wise less-or-equal; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&lt;=b.</returns>
        public NDMaskedArray less_equal(object a, object b) => Binary((x, y) => np.less_equal(x, y), a, b);

        /// <summary>Element-wise greater-than; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&gt;b.</returns>
        public NDMaskedArray greater(object a, object b) => Binary((x, y) => np.greater(x, y), a, b);

        /// <summary>Element-wise greater-or-equal; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&gt;=b.</returns>
        public NDMaskedArray greater_equal(object a, object b) => Binary((x, y) => np.greater_equal(x, y), a, b);

        /// <summary>Logical AND; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a&amp;&amp;b.</returns>
        public NDMaskedArray logical_and(object a, object b) => Binary((x, y) => np.logical_and(x, y), a, b);

        /// <summary>Logical OR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a||b.</returns>
        public NDMaskedArray logical_or(object a, object b) => Binary((x, y) => np.logical_or(x, y), a, b);

        /// <summary>Logical XOR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked boolean array of a^b.</returns>
        public NDMaskedArray logical_xor(object a, object b) => Binary((x, y) => np.logical_xor(x, y), a, b);

        /// <summary>Bitwise AND; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a&amp;b.</returns>
        public NDMaskedArray bitwise_and(object a, object b) => Binary((x, y) => np.bitwise_and(x, y), a, b);

        /// <summary>Bitwise OR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a|b.</returns>
        public NDMaskedArray bitwise_or(object a, object b) => Binary((x, y) => np.bitwise_or(x, y), a, b);

        /// <summary>Bitwise XOR; masked where either operand was.</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>A masked array of a^b.</returns>
        public NDMaskedArray bitwise_xor(object a, object b) => Binary((x, y) => np.bitwise_xor(x, y), a, b);

        /// <summary>Bit shift left (<c>a &lt;&lt; b</c>; NumPy's <c>ma.left_shift</c>). Unlike the arithmetic/bitwise ma ops,
        /// NumPy's <c>left_shift</c> is a plain function (not a <c>_MaskedBinaryOperation</c>), so a bare shift-count
        /// scalar is NOT widened to int64 — the result keeps the value operand's dtype (<c>ma.left_shift(int16, 1)</c> →
        /// int16) — the masked slots are shifted from 0 (<c>filled(a, 0)</c>), and the result is a NEW masked array that
        /// inherits nothing from <paramref name="a"/> (see <see cref="Shift"/>).</summary>
        /// <param name="a">Value to shift.</param><param name="b">Shift amount.</param>
        /// <returns>A masked array of a&lt;&lt;b in <paramref name="a"/>'s dtype.</returns>
        /// <exception cref="MaskError"><paramref name="b"/> broadcasts the result past <paramref name="a"/>'s mask (NumPy's
        ///     <c>masked_array(d, mask=m)</c> size check).</exception>
        /// <exception cref="TypeError">The shift has no loop for the operands (a float, complex or text value: NumPy's
        ///     "not supported for the input types" text).</exception>
        public NDMaskedArray left_shift(object a, object b) => Shift(a, b, left: true);

        /// <summary>
        ///     NumPy's <c>ma.left_shift</c> / <c>ma.right_shift</c>, statement for statement: <c>m = getmask(a)</c>; without a
        ///     mask <c>masked_array(umath.left_shift(filled(a), n))</c>, with one
        ///     <c>masked_array(umath.left_shift(filled(a, 0), n), mask=m)</c>. So a masked slot holds the shift of 0, not the
        ///     operand's data; the result is a NEW masked array (soft, the default fill) whose mask is <paramref name="a"/>'s
        ///     own mask object (<c>masked_array</c> adopts it); and a masked shift count meets the ufunc's
        ///     <c>__array_wrap__</c> first (its mask, shrunk to nomask when nothing is masked, and its fill value and hard
        ///     mask), which <c>masked_array</c> keeps and ORs with <paramref name="a"/>'s mask.
        /// </summary>
        /// <param name="a">Value to shift.</param>
        /// <param name="n">Shift amount: a C# integer (a weak Python int), an array or a masked array.</param>
        /// <param name="left">True for <c>left_shift</c>, false for <c>right_shift</c>.</param>
        /// <returns>The shifted masked array, or <see cref="NDMasked"/> for a 0-d result whose shift count is masked.</returns>
        /// <exception cref="MaskError"><paramref name="n"/> broadcasts the result past <paramref name="a"/>'s mask.</exception>
        /// <exception cref="TypeError">The shift has no loop for the operands (a float, complex or text value: NumPy's
        ///     "not supported for the input types" text).</exception>
        /// <remarks>Scoped by hand: the filled copy and the raw shift of a masked count are reclaimed; only the result's
        /// data and mask leave.</remarks>
        private NDMaskedArray Shift(object a, object n, bool left)
        {
            using var scope = NDScope.Open();
            var m = (a as NDMaskedArray)?._mask;
            // `filled(a)` without a mask is the data itself; `filled(a, 0)` writes 0 into the masked slots of a copy — for
            // text NumPy formats '0' (it cannot fail), and the shift's own no-loop refusal decides the call, so the
            // formatting is deferred behind it (FilledForNumericOp).
            Exception deferral = null;
            var fa = m is null ? AsData(a) : FilledForNumericOp(a, 0, ref deferral);
            var raw = left ? np.left_shift(fa, AsData(n)) : np.right_shift(fa, AsData(n));
            if (deferral is not null)
                throw deferral;
            object d = raw;
            if (n is NDMaskedArray nm)
            {
                // The ufunc over a plain value and a masked count: __array_wrap__ gives the count's mask (mask_or with
                // the value's all-False getmaskarray shrinks), collapses a masked 0-d result, and `_update_from(n)`.
                NDArray wm = nm._mask is null ? null : np.logical_or(np.zeros(DimsOf(raw), np.@bool), nm._mask);
                if (wm is not null && !np.any(wm))
                    wm = null;
                if (raw.ndim == 0 && wm is not null)
                    return Yield(scope, NDMasked);
                d = new NDMaskedArray(raw, wm).UpdateFrom(nm);
            }
            // `masked_array(d)` / `masked_array(d, mask=m)`: the class constructor (hard_mask=None keeps d's hardness).
            return Yield(scope, m is null ? masked_array(d) : masked_array(d, mask: m));
        }

        /// <summary>Bit shift right (<c>a &gt;&gt; b</c>; NumPy's <c>ma.right_shift</c>): the statements of
        /// <see cref="left_shift"/> with <c>umath.right_shift</c> — the value operand's dtype is kept for a bare scalar
        /// shift count, the masked slots are shifted from 0, and the result inherits nothing (see <see cref="Shift"/>).</summary>
        /// <param name="a">Value to shift.</param><param name="b">Shift amount.</param>
        /// <returns>A masked array of a&gt;&gt;b in <paramref name="a"/>'s dtype.</returns>
        /// <exception cref="MaskError"><paramref name="b"/> broadcasts the result past <paramref name="a"/>'s mask.</exception>
        /// <exception cref="TypeError">The shift has no loop for the operands (a float, complex or text value: NumPy's
        ///     "not supported for the input types" text).</exception>
        public NDMaskedArray right_shift(object a, object b) => Shift(a, b, left: false);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Domained binary ufuncs — division family (unsafe divides are masked)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>True division; additionally MASKS positions where the denominator is (near) zero, so a
        /// masked <c>divide</c> never surfaces an inf/NaN from division by zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a/b with unsafe divides masked.</returns>
        public NDMaskedArray divide(object a, object b) => DomainedBinary((x, y) => np.divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Alias of <see cref="divide"/> (NumPy's <c>true_divide</c>).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a/b with unsafe divides masked.</returns>
        public NDMaskedArray true_divide(object a, object b) => DomainedBinary((x, y) => np.true_divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Floor division; masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a//b with unsafe divides masked.</returns>
        public NDMaskedArray floor_divide(object a, object b) => DomainedBinary((x, y) => np.floor_divide(x, y), SafeDivideDomain, a, b);

        /// <summary>Remainder (floored, sign follows divisor); masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a%b with unsafe divides masked.</returns>
        public NDMaskedArray remainder(object a, object b) => DomainedBinary((x, y) => np.remainder(x, y), SafeDivideDomain, a, b);

        /// <summary>Alias of <see cref="remainder"/> (NumPy's <c>mod</c>).</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of a%b with unsafe divides masked.</returns>
        public NDMaskedArray mod(object a, object b) => DomainedBinary((x, y) => np.remainder(x, y), SafeDivideDomain, a, b);

        /// <summary>C-style remainder (sign follows dividend); masks positions where the denominator is (near) zero.</summary>
        /// <param name="a">Numerator.</param><param name="b">Denominator.</param>
        /// <returns>A masked array of fmod(a, b) with unsafe divides masked.</returns>
        public NDMaskedArray fmod(object a, object b) => DomainedBinary((x, y) => np.fmod(x, y), SafeDivideDomain, a, b);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Reductions — masked entries are excluded by filling with the op's identity, and
        //  a result cell is masked iff EVERY element reduced into it was masked (NumPy's
        //  `_check_mask_axis` = mask.all(axis)). Scans keep the original per-position mask.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     The value NumPy fills masked elements with for a MINIMUM/argmin (the dtype's LARGEST value, so
        ///     a masked slot can never win a min): +inf for float/complex, MaxValue for integers/decimal/char,
        ///     the integer <c>1</c> for bool, and null for a text dtype. NumPy's <c>minimum_fill_value</c>.
        /// </summary>
        /// <param name="a">Array-like whose dtype selects the value.</param>
        /// <returns>A boxed scalar castable into the dtype, or null for a text dtype.</returns>
        /// <remarks>NumPy's <c>min_filler</c> table holds <c>None</c> for <c>str_</c>/<c>bytes_</c> (no largest text), and its
        /// reductions then fill with the array's own fill value (<c>self.filled(None)</c>); the bool entry is the Python
        /// int <c>1</c>, which is why the result is a <see cref="long"/> rather than <c>true</c>.</remarks>
        public object minimum_fill_value(object a)
        {
            // Scoped by hand (an object return): a scalar/array-like operand is materialized only to read its
            // dtype, and is reclaimed.
            using var scope = NDScope.Open();
            return MinFill(AsData(a).typecode);
        }

        /// <summary>
        ///     The value NumPy fills masked elements with for a MAXIMUM/argmax (the dtype's SMALLEST value):
        ///     -inf for float/complex, MinValue for integers/decimal, the integer <c>0</c> for bool, and null for a text
        ///     dtype. NumPy's <c>maximum_fill_value</c>.
        /// </summary>
        /// <param name="a">Array-like whose dtype selects the value.</param>
        /// <returns>A boxed scalar castable into the dtype, or null for a text dtype.</returns>
        /// <remarks>See <see cref="minimum_fill_value"/>: NumPy's <c>max_filler</c> holds <c>None</c> for text and the
        /// Python int <c>0</c> for bool.</remarks>
        public object maximum_fill_value(object a)
        {
            // Scoped by hand (an object return): a scalar/array-like operand is materialized only to read its
            // dtype, and is reclaimed.
            using var scope = NDScope.Open();
            return MaxFill(AsData(a).typecode);
        }

        /// <summary>Per-dtype "largest" fill (for min/argmin) — see <see cref="minimum_fill_value"/>.</summary>
        /// <param name="tc">The data's storage lane.</param>
        /// <returns>The boxed fill, or null for the text lane.</returns>
        private static object MinFill(NPTypeCode tc) => tc switch
        {
            // _maxvals[np.bool_] = 1: NumPy's entry is the Python int, cast into the bool data as True.
            NPTypeCode.Boolean => (object)1L,
            // _maxvals[np.str_] / [np.bytes_] = None: no largest text; the reduction uses the array's own fill.
            NPTypeCode.String => null,
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
        /// <param name="tc">The data's storage lane.</param>
        /// <returns>The boxed fill, or null for the text lane.</returns>
        private static object MaxFill(NPTypeCode tc) => tc switch
        {
            // _minvals[np.bool_] = 0: the Python int, cast into the bool data as False.
            NPTypeCode.Boolean => (object)0L,
            // _minvals[np.str_] / [np.bytes_] = None, as for MinFill.
            NPTypeCode.String => null,
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
        /// 0-D result to the <see cref="NDMasked"/> singleton (NumPy's scalar-reduction behavior).</summary>
        private NDMaskedArray Finalize(NDArray result, NDArray newmask)
        {
            if (result.ndim == 0)
                return (newmask is not null && np.any(newmask)) ? NDMasked : new NDMaskedArray(result, null);
            return new NDMaskedArray(result, newmask);
        }

        /// <summary>Shared identity-fill reduction: fill masked with <paramref name="identity"/>, reduce via
        /// <paramref name="fn"/>, and mask result cells whose whole reduced slice was masked.</summary>
        /// <param name="a">Operand.</param>
        /// <param name="identity">The op identity written at masked slots (0 sum, 1 prod, ±inf min/max, …); null for a text
        /// min/max, which fills with the operand's own fill value (NumPy's <c>filled(None)</c>).</param>
        /// <param name="fn">The underlying <c>np.*</c> reduction (data, axis, keepdims) → result.</param>
        /// <param name="axis">Reduction axis, or null to reduce the flattened array.</param>
        /// <param name="keepdims">Keep reduced axes as size-1 (NumPy keepdims).</param>
        /// <returns>The masked reduction result (or <see cref="NDMasked"/> for a fully-masked scalar).</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a masked slot whose reduction would
        ///     compute: NumPy formats the numeric identity into those slots first (the next phase).</exception>
        /// <remarks>Scoped by hand — the one scope sum/prod/min/max ride: the identity-filled copy of the data (and
        /// a materialized scalar operand) is reclaimed, as is the computed value when a fully-masked 0-D result
        /// collapses to the <see cref="NDMasked"/> singleton; only the reduction and its mask leave.</remarks>
        private NDMaskedArray ReduceIdentity(object a, object identity, Func<NDArray, int?, bool, NDArray> fn, int? axis, bool keepdims)
        {
            using var scope = NDScope.Open();
            var mask = (a as NDMaskedArray)?._mask;
            // A null identity is NumPy's `self.filled(None)`: minimum_fill_value / maximum_fill_value are None for text, so
            // min/max fill the masked slots with the array's own fill value ('N/A') before the reduction answers.
            if (identity is null && mask is not null)
                identity = ((NDMaskedArray)a).fill_value;
            // `self.filled(0).sum(...)` of TEXT: NumPy formats the identity into the masked slots and the reduction then
            // refuses the dtype (add.reduce / multiply over strings). FilledForNumericOp hands the reduction the unfilled
            // data (same dtype and shape) and defers only if the reduction would compute.
            Exception deferral = null;
            var d = mask is null ? AsData(a) : FilledForNumericOp(a, identity, ref deferral, reduction: true);
            var result = fn(d, axis, keepdims);
            if (deferral is not null)
                throw deferral;
            if (mask is null)
                return Yield(scope, new NDMaskedArray(result, null));
            return Yield(scope, Finalize(result, AllAlongAxis(mask, axis, keepdims)));
        }

        /// <summary>
        ///     numpy.ma's <c>filled(a, fill)</c> as a NUMERIC operation reads it (<c>self.filled(0).sum()</c>,
        ///     <c>self.filled(1).cumprod()</c>, <c>np.dot(filled(a, 0), filled(b, 0))</c>, <c>np.inner(...)</c>), safe for
        ///     TEXT data. NumPy formats the number into the masked text slots (<c>'0'</c>) — a step that cannot fail and
        ///     keeps the dtype and the shape — so the operation then decides the call: over text it refuses by dtype or
        ///     shape (no string loop, <c>data type must provide an itemsize</c>, a misaligned product, a rank error), or it
        ///     computes on the text. Formatting numbers into text is the next phase, so for a text operand with a masked
        ///     slot this returns the UNFILLED data and records the deferral: the caller runs its operation (whose refusal
        ///     is then NumPy's, the dtype and shape being the same) and throws the deferral if the operation returns.
        /// </summary>
        /// <param name="a">The operand: a masked array, a plain array or an array-like.</param>
        /// <param name="fill">The value numpy.ma fills the masked slots with.</param>
        /// <param name="deferral">Receives the next-phase exception when <paramref name="a"/> is text with a masked slot
        ///     and <paramref name="fill"/> is a number (left unchanged otherwise, so several operands share one).</param>
        /// <param name="reduction">True for a reduction's fill (<see cref="FilledForReduction"/>: a K-order copy, reading
        ///     the data's own memory order), false for the public <see cref="NDMaskedArray.filled(object)"/>.</param>
        /// <returns>The filled data (or the data itself / a view of it when no slot is masked), or the unfilled text data
        ///     when the deferral was recorded.</returns>
        /// <exception cref="TypeError">A fill numpy.ma refuses (see <see cref="CheckFillValue"/>).</exception>
        private static NDArray FilledForNumericOp(object a, object fill, ref Exception deferral, bool reduction = false)
        {
            if (a is not NDMaskedArray m || m._mask is null)
                return AsData(a);
            if (m._data.typecode == NPTypeCode.String && FormatsAsText(fill) && np.any(m._mask))
            {
                deferral ??= FormattingDeferral(fill, m._data.dtype);
                return m._data;
            }
            return reduction ? FilledForReduction(m._data, m._mask, fill) : m.filled(fill);
        }

        /// <summary>
        ///     The exception NumPy's <c>_check_fill_value</c> formatting a number into a text dtype stands for in NumSharp:
        ///     the next phase of the string support (the cast <c>number → U/S</c>).
        /// </summary>
        /// <param name="fill">The number (or numeric array) NumPy would format.</param>
        /// <param name="ndtype">The text dtype it would be formatted into.</param>
        /// <returns>The <see cref="StringComputationNotSupportedException"/> to throw.</returns>
        private static Exception FormattingDeferral(object fill, DType ndtype) => fill switch
        {
            NDMaskedArray m => NDStringCasting.TextConversion(m._data.dtype, ndtype),
            NDArray nd => NDStringCasting.TextConversion(nd.dtype, ndtype),
            // A BigInteger (a Python int of any size) has no storage lane; the message names int64, NumPy's default integer.
            BigInteger => NDStringCasting.TextConversion(DType.From(NPTypeCode.Int64), ndtype),
            _ => NDStringCasting.TextConversion((DType)fill.GetType(), ndtype),
        };

        /// <summary>
        ///     NumPy's <c>MaskedArray.filled(fill)</c> as numpy.ma's reductions (<c>self.filled(0).sum(...)</c>, prod, min,
        ///     max, all, any) call it: the data ITSELF when no slot is masked, otherwise a K-order copy
        ///     (<c>self._data.copy('K')</c>) with <paramref name="fill"/> written at the masked slots. The reduction then walks
        ///     the data's own memory order, as NumPy's does — a float sum or product along an axis of an F-ordered input
        ///     adds in NumPy's order (the public <see cref="NDMaskedArray.filled(object)"/> returns a C-order copy).
        /// </summary>
        /// <param name="data">The masked array's data (any layout); never written.</param>
        /// <param name="mask">Its mask (not null).</param>
        /// <param name="fill">The value for the masked slots, cast unsafely into the data dtype.</param>
        /// <returns><paramref name="data"/> itself, or a fresh K-order filled copy.</returns>
        private static NDArray FilledForReduction(NDArray data, NDArray mask, object fill)
        {
            // `elif not m.any(): return self._data` — nothing to substitute, and no copy.
            if (!np.any(mask))
                return data;
            var r = data.copy('K');
            np.copyto(r, NDArray.Scalar(fill), casting: "unsafe", where: mask);
            return r;
        }

        /// <summary>Sum over the axis; masked slots contribute 0. A cell is masked iff its whole slice was.
        /// dtype widens like <c>np.sum</c> (NEP50: int→int64).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten).</param>
        /// <param name="dtype">Accumulator dtype (NumPy sum dtype=).</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked sum.</returns>
        public NDMaskedArray sum(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => ReduceIdentity(a, 0, (x, ax, kd) => np.sum(x, ax, kd, dtype), axis, keepdims);

        /// <summary>Product over the axis; masked slots contribute 1.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked product.</returns>
        public NDMaskedArray prod(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => ReduceIdentity(a, 1, (x, ax, kd) => np.prod(x, ax, dtype, kd), axis, keepdims);

        /// <summary>Alias of <see cref="prod"/> (NumPy's <c>product</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="dtype">Accumulator dtype.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked product.</returns>
        public NDMaskedArray product(object a, int? axis = null, DType dtype = null, bool keepdims = false)
            => prod(a, axis, dtype, keepdims);

        /// <summary>Minimum over the axis; masked slots are filled with the dtype's largest value so they are
        /// never selected. A cell is masked iff its whole slice was masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Override the masked fill (NumPy fill_value=); null uses <see cref="minimum_fill_value"/>.</param>
        /// <param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked minimum.</returns>
        public NDMaskedArray min(object a, int? axis = null, object fill_value = null, bool keepdims = false)
            => ReduceIdentity(a, fill_value ?? minimum_fill_value(a), (x, ax, kd) => np.min(x, ax, kd), axis, keepdims);

        /// <summary>Maximum over the axis; masked slots are filled with the dtype's smallest value.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="maximum_fill_value"/>.</param>
        /// <param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked maximum.</returns>
        public NDMaskedArray max(object a, int? axis = null, object fill_value = null, bool keepdims = false)
            => ReduceIdentity(a, fill_value ?? maximum_fill_value(a), (x, ax, kd) => np.max(x, ax, kd), axis, keepdims);

        /// <summary>Peak-to-peak (max − min) over the axis, over unmasked elements.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <param name="fill_value">Masked fill passed to both min and max.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked range.</returns>
        public NDMaskedArray ptp(object a, int? axis = null, object fill_value = null, bool keepdims = false)
        {
            // Scoped by hand: the separate max and min reductions are only the subtraction's operands and are
            // reclaimed once the range is computed.
            using var scope = NDScope.Open();
            return Yield(scope, subtract(max(a, axis, fill_value, keepdims), min(a, axis, fill_value, keepdims)));
        }

        /// <summary>Number of UNMASKED elements along the axis (NumPy's <c>count</c>) — a plain int64 array,
        /// never masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>An int64 <see cref="NDArray"/> (0-D for a flat count).</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the negated-mask (or all-True) indicator the count sums is reclaimed.</remarks>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand (a 0-d operand accepts 0 only).</exception>
        [NDScoped]
        public NDArray count(object a, int? axis = null, bool keepdims = false)
        {
            var d = AsData(a);
            var m = (a as NDMaskedArray)?._mask;
            // DimsOf, not d.Shape: for a nomask NON-CONTIGUOUS operand the strided Shape made CountUnmasked size/iterate
            // over the base buffer and undercount (a [::2] view of 8 counted 6). The count is over the logical DIMENSIONS.
            if (m is not null)
                return CountUnmasked(m, DimsOf(d), axis, keepdims);
            // NumPy's nomask branch counts from the SHAPE, without a reduction: a 0-d operand is 1 (axis None or 0), no
            // axis is the size, and an axis is `np.full(out_dims, items, dtype=np.intp)` — a fresh WRITEABLE array (a
            // reduction's 0-d result is NumSharp's read-only stand-in for a scalar).
            if (d.ndim == 0)
            {
                if (axis is not null && axis.Value != 0)
                    throw new AxisError(axis.Value, 0);
                return NDArray.Scalar(1L);
            }
            if (axis is null)
                return keepdims
                    ? np.full(Shape.Adopt(Enumerable.Repeat(1L, d.ndim).ToArray()), d.size, np.int64)
                    : NDArray.Scalar(d.size);
            int ax = axis.Value < 0 ? axis.Value + d.ndim : axis.Value;
            if (ax < 0 || ax >= d.ndim)
                throw new AxisError(axis.Value, d.ndim);
            var dims = d.Shape.dimensions;
            long items = dims[ax];
            long[] outDims = keepdims
                ? dims.Select((n, i) => i == ax ? 1L : n).ToArray()
                : dims.Where((_, i) => i != ax).ToArray();
            return np.full(Shape.Adopt(outDims), items, np.int64);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  mean / var / std — NumPy's MaskedArray.mean / var / std, statement for statement
        // ─────────────────────────────────────────────────────────────────────────────
        //
        // numpy/ma/core.py runs these as Python over the masked ufunc wrappers, and almost every statement decides a
        // bit or a mask of the result (plan docs/plans/api-parameter-parity.md, item A6):
        //   * WITH a mask (an all-False one too): `mean` is `self.sum(dtype=f8 | f4 | None) * 1. / self.count()` in masked
        //     arithmetic (ma.multiply, then ma.true_divide, which MASKS a non-finite slice mean). `var` is
        //     `danom = self - self.mean(keepdims=True)`; `danom *= danom` (complex: `abs(danom) ** 2`, which masks a
        //     non-finite square); `divide(danom.sum(axis), count - ddof)`; then the result's mask is REPLACED by
        //     `mask_or(self._mask.all(axis), cnt <= 0)`. So a slice holding an unmasked inf or NaN has a masked mean, a fully
        //     masked anomaly, a zero sum, and comes back 0.0 UNMASKED.
        //   * WITHOUT a mask: `super().mean / var(...)[()]`. numpy's _mean / _var run on the MaskedArray itself, so each
        //     `true_divide(..., out=...)` goes through MaskedArray.__array_wrap__, which evaluates ma's divide domain
        //     (|quotient| * tiny >= count) on the QUOTIENT, writes 1 there and masks it: an inf mean becomes 1.0 before
        //     the deviations are formed, and an inf variance comes back masked with data 1.0.
        //   * `std` is ma.sqrt(var(...)) — `mean=` is not forwarded — which masks a non-finite or negative variance.
        // These statements were proven against numpy.ma on 548,794 cases (result kind, dtype, shape, the data bits at
        // every slot, the mask, nomask-ness, the error type) by a plain-numpy model before this port was written.
        //
        // NumSharp's np.* ufuncs read a 0-d operand as WEAK (oracle registry K7), while numpy.ma's 0-d operands are
        // STRONG: getdata(1.) is np.array(1.), a flat count is an np.intp, a scalar mean is a numpy scalar. So every binary
        // step below computes NumPy's result dtype first (strong promotion, StrongCast) and hands the ufunc two operands
        // of that dtype; a same-dtype call has no promotion left to get wrong.

        /// <summary>
        ///     Mean over the unmasked elements: NumPy's <c>MaskedArray.mean</c>, statement for statement. With a mask
        ///     (even an all-False one) it is <c>self.sum(axis, dtype) * 1. / self.count(axis)</c> in masked arithmetic, so a
        ///     slice whose mean is not finite (an unmasked inf or NaN in it) is MASKED, and a fully masked slice is masked.
        ///     Without a mask it is numpy's <c>_mean</c> run on a masked array: an inf mean (or a mean of at least
        ///     <c>count / tiny</c>) comes back masked with data 1.0.
        /// </summary>
        /// <param name="a">Operand: a <see cref="NDMaskedArray"/> (its mask decides the path), an <see cref="NDArray"/>, a C#
        /// array or scalar (no mask).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>, the sum's loop. When null an integer, bool or char input sums in
        /// float64 and a float16 input in float32 with the mean cast back to float16; every other input sums in its own
        /// dtype.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <returns>The masked mean: the <see cref="NDMasked"/> singleton for a masked 0-d result, an unmasked 0-d array for
        /// a scalar, otherwise an array with the mask NumPy gives it (null for nomask). With a mask the dtype is float64 for
        /// an integer, bool, char, float32 or float64 input (NumPy's <c>* 1.</c> multiplies by a STRONG float64), float16 for
        /// float16, complex128 for complex128; without one it is numpy's <c>_mean</c> dtype (float32 stays float32).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <remarks>Scoped by hand (a masked-array return): the filled copy, the sums, counts and quotients are reclaimed;
        /// only the result's data and mask leave.</remarks>
        public NDMaskedArray mean(object a, int? axis = null, DType dtype = null, bool keepdims = false)
        {
            using var scope = NDScope.Open();
            var m = a as NDMaskedArray;
            // `if self._mask is nomask: result = super().mean(...)[()]` — a non-masked operand is MaskedArray(a) there.
            if (m?._mask is null)
                return Yield(scope, NomaskMeanLikeNumPy(AsData(a), axis, dtype, keepdims));
            return Yield(scope, MaskedMeanLikeNumPy(m, axis, dtype, keepdims));
        }

        /// <summary>
        ///     The masked branch of NumPy's <c>MaskedArray.mean</c>:
        ///     <c>dsum = self.sum(axis, dtype, keepdims); cnt = self.count(axis, keepdims)</c>; a 0-d zero count is
        ///     <c>masked</c>; otherwise <c>dsum * 1. / cnt</c> — masked arithmetic for an array <c>dsum</c>, plain scalar
        ///     arithmetic for a 0-d one — cast back to float16 for a float16 input without <c>dtype=</c>.
        /// </summary>
        /// <param name="self">The operand; its mask is not null.</param>
        /// <param name="axis">The reduced axis, or null.</param>
        /// <param name="dtype">The sum's loop, or null for NumPy's default.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns><see cref="NDMasked"/>, an unmasked 0-d mean, or a masked array of means.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand (ma.sum's reduction raises it; a
        /// 0-d operand takes axis 0 / -1, as ufunc.reduce does).</exception>
        private NDMaskedArray MaskedMeanLikeNumPy(NDMaskedArray self, int? axis, DType dtype, bool keepdims)
        {
            var d = self._data;
            // `if dtype is None: integer / bool -> f8; float16 -> f4 (is_float16_result)`. NumSharp's char is an integer.
            bool float16Result = false;
            if (dtype is null)
            {
                var tc = d.typecode;
                if (tc.IsInteger() || tc == NPTypeCode.Boolean || tc == NPTypeCode.Char)
                    dtype = np.float64;
                else if (tc == NPTypeCode.Half)
                {
                    dtype = np.float32;
                    float16Result = true;
                }
            }

            var dsum = sum(self, axis, dtype, keepdims);
            var cnt = CountUnmasked(self._mask, DimsOf(d), axis, keepdims);
            // `if cnt.shape == () and (cnt == 0): result = masked` — a 0-d count only; an array of zero counts is the
            // domain's business below.
            if (cnt.ndim == 0 && cnt.GetInt64() == 0)
                return NDMasked;

            NDMaskedArray result;
            if (dsum._data.ndim > 0)
            {
                // `dsum * 1. / cnt` on a MaskedArray: ma.multiply by getdata(1.) — a STRONG float64 0-d, so a float32 sum
                // becomes float64 — then ma.true_divide by the count, which masks a non-finite or domain quotient.
                result = MaDivideLikeNumPy(MaMultiplyByStrongOneLikeNumPy(dsum), cnt);
            }
            else
            {
                // A 0-d sum is a numpy SCALAR: plain scalar arithmetic, nothing masked. `np.float32 * 1.` stays float32 (the
                // Python float is weak); `/ cnt` (an np.intp) runs in result_type(sum, int64).
                var (sv, one) = StrongCast(dsum._data, WeakFloatOne(dsum._data.dtype));
                result = new NDMaskedArray(StrongDivide(np.multiply(sv, one), cnt), null);
            }

            // `self.dtype.type(dsum * 1. / cnt)`: np.float16 of a MaskedArray keeps its mask (data cast at every slot), of a
            // scalar is a float16 scalar.
            if (float16Result)
                result = new NDMaskedArray(result._data.astype(d.dtype), result._mask);
            return result;
        }

        /// <summary>
        ///     The nomask branch of NumPy's <c>MaskedArray.mean</c>, <c>super().mean(axis, dtype, keepdims)[()]</c>: numpy's
        ///     <c>_mean</c> run ON the masked array. <c>umr_sum</c> then <c>true_divide(ret, rcount, out=ret,
        ///     casting='unsafe')</c>, whose <c>__array_wrap__</c> masks the quotients in ma's divide domain and writes 1 there
        ///     (<see cref="ArrayWrapDivideDomainFill"/>); float16 is summed in float32 and cast back — and the cast of a 0-d
        ///     masked array is a plain scalar, so a 0-d float16 mean loses that mask (an inf mean is 1.0, unmasked).
        /// </summary>
        /// <param name="d">The data (any layout).</param>
        /// <param name="axis">The reduced axis, or null.</param>
        /// <param name="dtype">The sum's loop, or null for <c>_mean</c>'s default.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>The mean: <see cref="NDMasked"/> for a masked 0-d result, an unmasked 0-d scalar, or an array with the
        /// domain mask (null when nothing hit).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="d"/> (<c>_count_reduce_items</c>
        /// normalizes it first).</exception>
        private NDMaskedArray NomaskMeanLikeNumPy(NDArray d, int? axis, DType dtype, bool keepdims)
        {
            long rcount = ReduceItemCount(d, axis);
            bool float16Result = false;
            if (dtype is null)
            {
                var tc = d.typecode;
                if (tc.IsInteger() || tc == NPTypeCode.Boolean || tc == NPTypeCode.Char)
                    dtype = np.float64;
                else if (tc == NPTypeCode.Half)
                {
                    dtype = np.float32;
                    float16Result = true;
                }
            }

            // ret = umr_sum(arr, axis, dtype, out=None, keepdims); true_divide(ret, rcount, out=ret, casting='unsafe').
            var ret = WriteableReduction(np.sum(d, axis, keepdims, dtype));
            DefaultEngine.DivideByCountLikeNumPy(ret, rcount);
            var mask = ArrayWrapDivideDomainFill(ret, rcount);

            if (float16Result)
            {
                ret = ret.astype(d.dtype);
                // np.float16(<0-d MaskedArray>) is a numpy scalar: the domain mask does not survive the cast.
                if (ret.ndim == 0)
                    return new NDMaskedArray(ret, null);
            }
            return ItemOfMaskedResult(ret, mask);
        }

        /// <summary>
        ///     Variance over the unmasked elements: NumPy's <c>MaskedArray.var</c>, statement for statement. With a mask
        ///     (even an all-False one) the anomaly is formed against the MASKED mean, squared in place
        ///     (<c>danom *= danom</c>; complex <c>abs(danom) ** 2</c>, masked where not finite), summed over its unmasked
        ///     slots, divided by <c>count - ddof</c> in ma's domained divide, and then the mask is REPLACED by
        ///     <c>mask_or(self._mask.all(axis), cnt &lt;= 0)</c>. A slice with an unmasked inf or NaN therefore gives 0.0,
        ///     unmasked, and a flat result with <c>count - ddof &lt; 0</c> is a plain (negative-zero or negative) scalar.
        ///     Without a mask it is numpy's <c>_var</c> run on a masked array (ma's divide domain on the mean and on the
        ///     result; see the section header).
        /// </summary>
        /// <param name="a">Operand: a <see cref="NDMaskedArray"/> (its mask decides the path), an <see cref="NDArray"/>, a C#
        /// array or scalar (no mask).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>. With a mask it only selects the MEAN's sum loop (the squared
        /// anomalies sum in their own dtype), so a complex <c>dtype</c> makes a real input's variance complex; without a
        /// mask it is <c>_var</c>'s loop of both sums.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>count - ddof</c> (with a mask, possibly ≤ 0) or
        /// <c>max(n - ddof, 0)</c> (without).</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="mean">NumPy 2's <c>mean=</c>, or null to compute it. With a mask it is subtracted through
        /// <c>ma.subtract</c>, which makes a C# number a STRONG array (int64 / float64 / complex128, as numpy.ma's
        /// <c>getdata</c> does); without a mask it reaches <c>_var</c> as given, so a C# number stays WEAK there. An
        /// <see cref="NDMaskedArray"/> mean brings its mask.</param>
        /// <returns>The masked variance: <see cref="NDMasked"/> for a masked 0-d result, an unmasked 0-d scalar, or an
        /// array with NumPy's mask (null for nomask). Its dtype is float64 for every input but float16 without a mask
        /// (float16), and complex128 when <paramref name="dtype"/> (or <paramref name="mean"/>) is complex.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="ValueError">
        ///     Without a mask, NumPy's two failures of <c>_var</c> on a masked array:
        ///     <list type="bullet">
        ///         <item>"output array is read-only" — a 0-d operand whose mean is masked (in ma's divide domain, or a masked
        ///         <paramref name="mean"/>): the 0-d <c>subtract</c> returns the read-only <c>masked</c> singleton, which the
        ///         next in-place ufunc writes into.</item>
        ///         <item>"Changing the dtype to a subarray type is only supported if the total itemsize is unchanged" — a
        ///         complex (or bool with a complex <paramref name="dtype"/>) operand whose deviation carries a mask: numpy's
        ///         <c>x.view(dtype=(f8, (2,)))</c> cannot view that mask.</item>
        ///     </list>
        /// </exception>
        /// <remarks>Scoped by hand (a masked-array return): the mean, the anomaly, its squares, the sums and counts are
        /// reclaimed; only the variance's data and mask leave.</remarks>
        public NDMaskedArray var(object a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, object mean = null)
        {
            using var scope = NDScope.Open();
            var m = a as NDMaskedArray;
            if (m?._mask is null)
                return Yield(scope, NomaskVarLikeNumPy(AsData(a), axis, dtype, ddof, keepdims, mean));
            return Yield(scope, MaskedVarLikeNumPy(m, axis, dtype, ddof, keepdims, mean));
        }

        /// <summary>
        ///     The masked branch of NumPy's <c>MaskedArray.var</c> (see <see cref="var"/>), in its statement order:
        ///     <c>cnt = self.count(axis) - ddof</c>; <c>danom = self - mean</c> (the masked keepdims mean when none is given);
        ///     square in place; <c>dvar = divide(danom.sum(axis), cnt)</c>; replace the mask of an array result, collapse a
        ///     masked 0-d one to <c>masked</c>.
        /// </summary>
        /// <param name="self">The operand; its mask is not null.</param>
        /// <param name="axis">The reduced axis, or null.</param>
        /// <param name="dtype">The mean's sum loop, or null.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="mean">The given mean, or null.</param>
        /// <returns><see cref="NDMasked"/>, an unmasked 0-d variance, or a masked array of variances.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand (the count's reduction raises it).</exception>
        /// <exception cref="IncorrectShapeException">A given <paramref name="mean"/> does not broadcast against the operand:
        /// the subtract ufunc's "operands could not be broadcast together with shapes …" text.</exception>
        private NDMaskedArray MaskedVarLikeNumPy(NDMaskedArray self, int? axis, DType dtype, int ddof, bool keepdims, object mean)
        {
            var d = self._data;
            var mask = self._mask;
            var cnt = np.subtract(CountUnmasked(mask, DimsOf(d), axis, keepdims), (long)ddof);

            // `danom = self - mean` / `self - self.mean(axis, dtype, keepdims=True)`: ma.subtract.
            var danom = MaSubtractLikeNumPy(self, mean ?? MaskedMeanLikeNumPy(self, axis, dtype, keepdims: true));

            if (d.typecode == NPTypeCode.Complex)
                danom = AbsSquaredLikeNumPy(danom, self);   // `umath.absolute(danom) ** 2`
            else
                danom = SquareInPlaceLikeNumPy(danom);      // `danom *= danom`

            // `danom.sum(axis, **kwargs)` — MaskedArray.sum with no dtype; a numpy scalar's or `masked`'s own sum.
            NDMaskedArray ssum;
            if (ReferenceEquals(danom, NDMasked))
                ssum = NDMasked;
            else if (danom._data.ndim == 0 && danom._mask is null)
                ssum = danom;
            else
                ssum = sum(danom, axis, null, keepdims);

            var dvar = MaDivideLikeNumPy(ssum, cnt);
            if (ReferenceEquals(dvar, NDMasked))
                return NDMasked;
            if (dvar._data.ndim == 0)
                // `elif getmask(dvar): dvar = masked` — an unmasked 0-d result stays, even with cnt <= 0.
                return dvar._mask is not null && np.any(dvar._mask) ? NDMasked : new NDMaskedArray(dvar._data, null);

            // `dvar._mask = mask_or(self._mask.all(axis, **kwargs), (cnt <= 0)); dvar._update_from(self)` — the divide's
            // own mask (non-finite and domain slots) is DROPPED, and mask_or shrinks an all-False result to nomask.
            var newmask = ShrinkMask(np.logical_or(AllAlongAxis(mask, axis, keepdims), np.less_equal(cnt, 0L)));
            return new NDMaskedArray(dvar._data, newmask).UpdateFrom(self);
        }

        /// <summary>
        ///     The nomask branch of NumPy's <c>MaskedArray.var</c>, <c>super().var(axis, dtype, ddof=ddof, keepdims [,
        ///     mean])[()]</c>: numpy's <c>_var</c> run ON the masked array, statement for statement. Its two
        ///     <c>true_divide(..., out=...)</c> steps go through <c>MaskedArray.__array_wrap__</c>
        ///     (<see cref="ArrayWrapDivideDomainFill"/>): a mean in ma's divide domain becomes 1.0 before the deviations are
        ///     formed, and a variance in it comes back masked with data 1.0.
        /// </summary>
        /// <param name="d">The data (any layout).</param>
        /// <param name="axis">The reduced axis, or null.</param>
        /// <param name="dtype"><c>_var</c>'s loop of both sums, or null.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>max(n - ddof, 0)</c>.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="mean">The given mean (a C# number stays weak), or null to compute it.</param>
        /// <returns>The variance: <see cref="NDMasked"/>, an unmasked 0-d scalar, or an array with the domain mask.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="d"/>.</exception>
        /// <exception cref="ValueError">The two <c>_var</c>-on-a-masked-array failures documented on <see cref="var"/>.</exception>
        private NDMaskedArray NomaskVarLikeNumPy(NDArray d, int? axis, DType dtype, int ddof, bool keepdims, object mean)
        {
            long rcount = ReduceItemCount(d, axis);
            var inT = d.typecode;
            // `if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)): dtype = f8`.
            if (dtype is null && (inT.IsInteger() || inT == NPTypeCode.Boolean || inT == NPTypeCode.Char))
                dtype = np.float64;

            NDArray x;
            bool meanMasked;
            if (mean is null)
            {
                // arrmean = umr_sum(arr, axis, dtype, keepdims=True); true_divide(arrmean, rcount, out=arrmean) — whose wrap
                // writes 1 into (and masks) the means in ma's divide domain.
                var arrmean = WriteableReduction(np.sum(d, axis, true, dtype));
                DefaultEngine.DivideByCountLikeNumPy(arrmean, rcount);
                meanMasked = ArrayWrapDivideDomainFill(arrmean, rcount) is not null;
                x = StrongSubtract(d, arrmean);
            }
            else
            {
                // `arrmean = mean`, handed to um.subtract as given: a C# number is a WEAK Python scalar there; an array keeps
                // its dtype; a masked array subtracts its raw data and brings its mask into x's.
                var mm = mean as NDMaskedArray;
                meanMasked = mm?._mask is not null && np.any(mm._mask);
                var md = AsData(mean);
                x = mean is NDArray || mean is Array || mm is not null ? StrongSubtract(d, md) : WeakSubtract(d, mean);
            }

            // x's mask is mask_or(getmaskarray(arr), getmaskarray(mean)) over the broadcast: an EMPTY x has no True slot,
            // so it shrinks to nomask whatever the mean's mask holds.
            if (x.size == 0)
                meanMasked = false;

            // A 0-d x with a mask is the `masked` singleton (the subtract's wrap); the next ufunc writes into it.
            if (d.ndim == 0 && meanMasked)
                throw new ValueError("output array is read-only");

            // Every square below is written IN PLACE into x, as NumPy's `out=x` writes it: x is the subtract's fresh
            // result in NumPy's ufunc layout (an input's permuted axis order included), and the sum then walks that
            // memory order. A new array would be C-ordered, and a flat sum of a permuted input would add in another
            // order (2 ULP apart on a transposed 3-D int32 input).
            NDArray x2;
            if (inT.IsFloatingPoint() || inT.IsInteger() || inT == NPTypeCode.Char || inT == NPTypeCode.Decimal)
            {
                // umr_multiply(x, x, out=x): the multiply loop, not square's (complex x from a complex dtype= takes
                // NumPy's complex MULTIPLY here, x * x, never |x|**2).
                np.multiply(x, x, @out: x);
                x2 = x;
            }
            else if (x.typecode == NPTypeCode.Complex)
            {
                // xv = x.view(dtype=(f8, (2,))) fails on a masked x. Otherwise um.square(xv, out=xv) squares both parts IN
                // PLACE and um.add(xv[..., 0], xv[..., 1], out=x.real) leaves re**2 + im**2 in x.real — a stride-2 view in
                // x's own layout, which is what the sum then walks (a fresh array would add in another order).
                if (meanMasked)
                    throw new ValueError("Changing the dtype to a subarray type is only supported if the total itemsize is unchanged");
                var re = np.real(x);
                var im = np.imag(x);
                np.multiply(re, re, @out: re);
                np.multiply(im, im, @out: im);
                np.add(re, im, @out: re);
                x2 = re;
            }
            else
            {
                // um.multiply(x, um.conjugate(x), out=x).real — a bool input's real x (conjugate is the identity there).
                np.multiply(x, np.conjugate(x), @out: x);
                x2 = np.real(x);
            }

            // ret = umr_sum(x, axis, dtype, out, keepdims); true_divide(ret, max(rcount - ddof, 0), out=ret).
            var ret = WriteableReduction(np.sum(x2, axis, keepdims, dtype));
            long rc = Math.Max(rcount - ddof, 0);
            DefaultEngine.DivideByCountLikeNumPy(ret, rc);
            var rmask = ArrayWrapDivideDomainFill(ret, rc);
            return ItemOfMaskedResult(ret, rmask);
        }

        /// <summary>Standard deviation: NumPy's <c>MaskedArray.std</c>, <c>sqrt(self.var(axis, dtype, ddof=ddof,
        /// keepdims))</c> through ma's domained sqrt, which masks a non-finite or negative variance (and keeps the
        /// variance's own mask). A <c>masked</c> variance is returned as is.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">The variance's
        /// <c>dtype=</c> (see <see cref="var"/>).</param>
        /// <param name="ddof">Delta degrees of freedom.</param><param name="keepdims">Keep reduced axes.</param>
        /// <param name="mean">Accepted for signature parity but IGNORED — NumPy's <c>ma.std</c> forwards only
        /// <c>keepdims</c> to <c>var</c>, never <c>mean</c> (a NumPy quirk: <c>ma.std(mean=X)</c> yields the PLAIN
        /// std, unlike <c>ma.var(mean=X)</c>). Reproduced here so a ported call behaves identically.</param>
        /// <returns>The masked standard deviation.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="ValueError">The variance's own failures (see <see cref="var"/>).</exception>
        public NDMaskedArray std(object a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, object mean = null)
        {
            // Scoped by hand: the variance is only sqrt's operand; its data is reclaimed once the root is taken
            // (its mask, which sqrt carries through, is re-parented by the Yield instead).
            using var scope = NDScope.Open();
            // NumPy's ma.std deliberately does NOT pass `mean` to var (only keepdims), so a supplied mean is a
            // no-op here too — matching ma.std(mean=X) == plain std, byte-for-byte.
            var v = var(a, axis, dtype, ddof, keepdims);
            return Yield(scope, ReferenceEquals(v, NDMasked) ? NDMasked : sqrt(v));
        }

        // ── Building blocks of mean / var: numpy.ma's statements on NumSharp arrays ──────────────────

        /// <summary>
        ///     <c>_count_reduce_items(arr, axis)</c>: the number of elements each output reduces, with the axis normalized
        ///     first (so a bad axis raises NumPy's <c>AxisError</c> before any arithmetic).
        /// </summary>
        /// <param name="d">The operand.</param>
        /// <param name="axis">The reduced axis, or null for every element.</param>
        /// <returns>The element count of one reduced slice.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="d"/>.</exception>
        private static long ReduceItemCount(NDArray d, int? axis)
            => axis is null ? d.size : d.Shape.dimensions[DefaultEngine.NormalizeAxis(axis.Value, d.ndim)];

        /// <summary>
        ///     A reduction result that the next in-place divide may overwrite: a 0-d reduction comes back as NumSharp's
        ///     read-only reduction scalar, which is copied; an array result is fresh and returned as is.
        /// </summary>
        /// <param name="r">The reduction result.</param>
        /// <returns><paramref name="r"/> or a writeable copy of it.</returns>
        private static NDArray WriteableReduction(NDArray r) => r.Shape.IsWriteable ? r : r.copy();

        /// <summary>
        ///     The domain step of <c>MaskedArray.__array_wrap__</c> after <c>true_divide(q, count, out=q)</c>: ma's divide
        ///     domain <c>|q| * tiny &gt;= count</c> is evaluated on the QUOTIENT (the output already holds it), and where it
        ///     holds the quotient is overwritten with ufunc_fills[divide][-1] = 1 and masked. <c>tiny</c> is an
        ///     <c>np.float64</c> (strong), so the product is float64 for every quotient dtype: an inf quotient hits, a NaN
        ///     never does, and a zero count masks every non-NaN quotient.
        /// </summary>
        /// <param name="q">The quotient, overwritten with 1 where the domain holds. Writeable.</param>
        /// <param name="count">The divisor the quotient came from.</param>
        /// <returns>The domain mask (q's shape), or null when nothing hit (NumPy's mask_or shrinks it to nomask).</returns>
        private static NDArray ArrayWrapDivideDomainFill(NDArray q, long count)
        {
            var hit = np.greater_equal(np.multiply(np.abs(q).astype(np.float64, copy: false), SafeDivideTiny), (double)Math.Abs(count));
            if (!np.any(hit))
                return null;
            np.copyto(q, NDArray.Scalar(1), casting: "unsafe", where: hit);
            return hit;
        }

        /// <summary>
        ///     <c>result[()]</c> of a masked reduction result: a 0-d result is <c>masked</c> when its mask is set and an
        ///     unmasked scalar otherwise; an array is returned as a masked array over its mask.
        /// </summary>
        /// <param name="r">The result data.</param>
        /// <param name="mask">Its mask, or null for nomask.</param>
        /// <returns><see cref="NDMasked"/>, an unmasked 0-d array, or the masked array.</returns>
        private NDMaskedArray ItemOfMaskedResult(NDArray r, NDArray mask)
        {
            if (r.ndim == 0)
                return mask is not null && np.any(mask) ? NDMasked : new NDMaskedArray(r, null);
            return new NDMaskedArray(r, mask);
        }

        /// <summary>
        ///     numpy.ma's <c>make_mask(m, shrink=True)</c> as <c>mask_or</c> applies it: an all-False mask is nomask.
        /// </summary>
        /// <param name="m">A boolean mask.</param>
        /// <returns><paramref name="m"/>, or null when no element is set.</returns>
        private static NDArray ShrinkMask(NDArray m) => m is not null && np.any(m) ? m : null;

        /// <summary>
        ///     NumPy's strong promotion of two array operands: both are cast to <c>result_type(a.dtype, b.dtype)</c>, so the
        ///     NumSharp ufunc that follows sees two operands of one dtype and cannot read a 0-d one as weak.
        /// </summary>
        /// <param name="a">First operand.</param>
        /// <param name="b">Second operand.</param>
        /// <returns>The two operands in the common dtype (a no-op cast returns the operand itself).</returns>
        private static (NDArray A, NDArray B) StrongCast(NDArray a, NDArray b)
        {
            var rt = np.result_type(a.dtype, b.dtype);
            return (a.astype(rt, copy: false), b.astype(rt, copy: false));
        }

        /// <summary>
        ///     <c>np.subtract(a, b)</c> with both operands STRONG (two arrays, a 0-d one included), written into a result
        ///     laid out as NumPy's ufunc lays it out (<see cref="UfuncResultBuffer"/>). The layout is observable: the
        ///     anomaly's sum along an axis (or flat, through numpy.ma's K-order <c>filled</c>) adds in this memory order.
        /// </summary>
        /// <param name="a">Minuend.</param><param name="b">Subtrahend.</param>
        /// <returns>The difference in <c>result_type(a, b)</c>.</returns>
        private static NDArray StrongSubtract(NDArray a, NDArray b)
        {
            var rt = np.result_type(a.dtype, b.dtype);
            // The layout is voted by the ORIGINAL operands: a cast copy would carry its own (C) strides.
            var r = UfuncResultBuffer(rt, a, b);
            np.subtract(a.astype(rt, copy: false), b.astype(rt, copy: false), @out: r);
            return r;
        }

        /// <summary>
        ///     A fresh, uninitialized result buffer of a ufunc over <paramref name="operands"/>, laid out as NumPy's NpyIter
        ///     allocates it (<see cref="Shape.NpyIterOutputShape"/>: each operand votes its memory order and abstains on a
        ///     stride-0 or extent-1 axis, C order wins a conflict). A permuted input thus gets a result in its own axis
        ///     order and a broadcast view (stride 0) a C-ordered one, where NumSharp's eager ufuncs write C order for any
        ///     input that is neither C- nor F-contiguous. The layout is observable in numpy.ma's mean / var: a later sum
        ///     walks this memory order.
        /// </summary>
        /// <param name="dtype">The result dtype.</param>
        /// <param name="operands">The ufunc's operands as it receives them (their strides vote), at least one.</param>
        /// <returns>An uninitialized array of the broadcast shape in NumPy's layout.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast together: NumPy's ufunc text, "operands
        /// could not be broadcast together with shapes (4,5) (4,) " (each operand's shape and a space), which is what a
        /// <c>mean=</c> of the reduced shape without keepdims meets in <c>self - mean</c>.</exception>
        private static NDArray UfuncResultBuffer(DType dtype, params NDArray[] operands)
        {
            var shapes = new Shape[operands.Length];
            for (int i = 0; i < operands.Length; i++)
                shapes[i] = operands[i].Shape;
            // np.broadcast_shapes, not Shape.ResolveReturnShape: the latter reads a (1,) operand as a scalar, so (1,) with
            // () would come back () where NumPy's broadcast is (1,).
            Shape bs;
            try
            {
                bs = np.broadcast_shapes(shapes);
            }
            catch (IncorrectShapeException ex)
            {
                // The buffer stands for the ufunc's own allocation, so a mismatch is the ufunc's error, not
                // np.broadcast_shapes' ("shape mismatch: objects cannot be broadcast to a single shape...").
                var text = new System.Text.StringBuilder("operands could not be broadcast together with shapes ");
                foreach (var s in shapes)
                    text.Append(s.ToPythonTuple()).Append(' ');
                throw new IncorrectShapeException(text.ToString(), ex);
            }
            // A private copy of the dimensions: broadcast_shapes may hand back an operand's own shape, whose dimension
            // array that operand shares.
            var dims = bs.NDim == 0 ? Array.Empty<long>() : (long[])bs.dimensions.Clone();
            return new NDArray(dtype, Shape.NpyIterOutputShape(dims, shapes), false);
        }

        /// <summary><c>np.true_divide(a, b)</c> with both operands STRONG.</summary>
        /// <param name="a">Dividend.</param><param name="b">Divisor.</param>
        /// <returns>The quotient in <c>result_type(a, b)</c> (float64 for an integer divisor and a float32 or float64
        /// dividend, NumPy's float64 loop).</returns>
        private static NDArray StrongDivide(NDArray a, NDArray b)
        {
            var (ca, cb) = StrongCast(a, b);
            return np.true_divide(ca, cb);
        }

        /// <summary>
        ///     <c>np.subtract(a, number)</c> with the number a WEAK Python scalar, as numpy's <c>_var</c> receives a numeric
        ///     <c>mean=</c>: the loop is NEP 50's <c>result_type(a.dtype, number)</c>, both operands cast to it.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="number">A boxed C# number.</param>
        /// <returns>The difference in the weak-promoted dtype.</returns>
        private static NDArray WeakSubtract(NDArray a, object number)
        {
            var rt = np.result_type(new object[] { a.dtype, number });
            var b = NDArray.Scalar(number).astype(rt);
            // NumPy's `arr - mean` allocates in arr's own axis order (the 0-d number abstains from the vote), and the
            // squares of x are then summed in that memory order.
            var r = UfuncResultBuffer(rt, a, b);
            np.subtract(a.astype(rt, copy: false), b, @out: r);
            return r;
        }

        /// <summary>
        ///     The 0-d <c>1</c> of NumPy's <c>dsum * 1.</c> on a numpy scalar: the weak Python float takes the scalar's dtype
        ///     when it is inexact (float32 stays float32, complex stays complex — the full complex multiply by 1+0j, which
        ///     turns an infinite imaginary part into a NaN real one) and float64 otherwise.
        /// </summary>
        /// <param name="dt">The scalar's dtype.</param>
        /// <returns>A 0-d one of <c>result_type(dt, 1.0)</c>.</returns>
        private static NDArray WeakFloatOne(DType dt)
            => NDArray.Scalar(1.0).astype(np.result_type(new object[] { dt, 1.0 }));

        /// <summary>
        ///     numpy.ma's <c>subtract(a, b)</c> (<c>_MaskedBinaryOperation</c>) for <c>self - mean</c> (var, anom): the data
        ///     difference in <c>result_type(getdata(a), getdata(b))</c> — <c>getdata</c> makes a C# number a STRONG array,
        ///     int64 / float64 / complex128, and a 0-d mean is strong too —, the masks OR-ed (nomask when neither side has
        ///     one, a mask-free side counted as all-False otherwise), the minuend's data put back (unsafe) at masked slots,
        ///     and a 0-d result collapsed to <c>masked</c> or a plain scalar.
        /// </summary>
        /// <param name="a">The minuend: a <see cref="NDMaskedArray"/> (with or without a mask), an <see cref="NDArray"/>,
        /// a C# array or number.</param>
        /// <param name="b">The subtrahend: a <see cref="NDMaskedArray"/> (the masked mean, <c>masked</c>, a scalar), an
        /// <see cref="NDArray"/>, or a C# number.</param>
        /// <returns>The masked difference — taking <paramref name="a"/>'s fill value and hard mask when it is a masked
        /// array, else <paramref name="b"/>'s (<see cref="InheritFromOperands"/>) —, a plain scalar, or
        /// <see cref="NDMasked"/>.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast together.</exception>
        private NDMaskedArray MaSubtractLikeNumPy(object a, object b)
        {
            var da = AsData(a);
            var db = AsData(b);
            // getdata(number) is np.array(number): int64 / float64 / complex128 / bool — strong.
            var strongA = BareScalarStrongDtype(a);
            if (strongA.HasValue)
                da = da.astype(strongA.Value);
            var strongB = BareScalarStrongDtype(b);
            if (strongB.HasValue)
                db = db.astype(strongB.Value);
            var result = StrongSubtract(da, db);
            var amask = (a as NDMaskedArray)?._mask;
            var bmask = (b as NDMaskedArray)?._mask;
            // `m = nomask` when both are nomask, else logical_or(getmaskarray(a) or ma, getmaskarray(b) or mb): a missing
            // side is all-False, which a 0-d False broadcasts to.
            NDArray m = null;
            if (amask is not null || bmask is not null)
                m = np.logical_or(amask ?? NDArray.Scalar(false), bmask ?? NDArray.Scalar(false));
            if (result.ndim == 0)
                return m is not null && np.any(m) ? NDMasked : new NDMaskedArray(result, null);
            // `masked_result._update_from(a)` (else b): the anomaly takes the operand's fill value and hard mask.
            if (m is null)
                return InheritFromOperands(new NDMaskedArray(result, null), a, b);
            if (!m.Shape.Equals(result.Shape))
                m = np.broadcast_to(m, result.Shape).copy();
            if (np.any(m))
                np.copyto(result, da, casting: "unsafe", where: m);
            return InheritFromOperands(new NDMaskedArray(result, m), a, b);
        }

        /// <summary>
        ///     numpy.ma's <c>multiply(dsum, 1.)</c> (<c>_MaskedBinaryOperation</c>), the <c>dsum * 1.</c> of the masked mean:
        ///     the data times <c>getdata(1.)</c> = <c>np.array(1.)</c>, a STRONG float64 0-d (a float16 or float32 sum becomes
        ///     float64, an integer or bool <c>dtype=</c> sum too, complex stays complex), into a fresh result in NumPy's ufunc
        ///     layout (dsum's own axis order: the 0-d abstains). The mask is dsum's (<c>logical_or(ma, False)</c>), and the
        ///     masked slots get dsum's data back.
        /// </summary>
        /// <param name="dsum">The masked sum of an array (ndim &gt; 0) operand.</param>
        /// <returns>The masked product.</returns>
        private static NDMaskedArray MaMultiplyByStrongOneLikeNumPy(NDMaskedArray dsum)
        {
            var da = dsum._data;
            var one = NDArray.Scalar(1.0);
            var (ca, cb) = StrongCast(da, one);
            // The layout is voted by the original operands, as in StrongSubtract.
            var result = UfuncResultBuffer(ca.dtype, da, one);
            np.multiply(ca, cb, @out: result);
            var m = dsum._mask;
            // `if m is not nomask and m.any(): np.copyto(result, da, casting='unsafe', where=m)`.
            if (m is not null && np.any(m))
                np.copyto(result, da, casting: "unsafe", where: m);
            return new NDMaskedArray(result, m);
        }

        /// <summary>
        ///     <c>danom *= danom</c> (<c>MaskedArray.__imul__</c>) in place: the masked slots multiply by 1 (they keep the
        ///     minuend's data), the others are squared in danom's own layout, which the next sum's order follows. A numpy
        ///     scalar is squared plainly; <c>masked</c> stays <c>masked</c>.
        /// </summary>
        /// <param name="danom">The anomaly (its array data is a fresh array this method overwrites).</param>
        /// <returns><paramref name="danom"/> itself with its data squared in place, a new scalar for a numpy scalar, or
        /// <see cref="NDMasked"/>.</returns>
        private static NDMaskedArray SquareInPlaceLikeNumPy(NDMaskedArray danom)
        {
            if (danom is NDMaskedConstant)
                return danom;
            var data = danom._data;
            if (danom._mask is null)
                // A numpy scalar (`x *= x` rebinds the name): a new 0-d value, never a write into the subtract's result.
                return new NDMaskedArray(np.multiply(data, data), null);
            // other_data = np.where(self._mask, other_data.dtype.type(1), other_data); self._data *= other_data — in place,
            // so the squares keep danom's layout (the next sum's order).
            var other = np.where(danom._mask, NDArray.Scalar(1).astype(data.dtype), data);
            np.multiply(data, other, @out: data);
            return danom;
        }

        /// <summary>
        ///     <c>umath.absolute(danom) ** 2</c> for a complex anomaly: the magnitude of every slot (masked ones too) under
        ///     danom's mask, then numpy.ma's <c>power(x, 2)</c> — <c>where(m, x, power(x, 2))</c>, and a non-finite square
        ///     is MASKED and overwritten with the fill value (so the sum skips it). A numpy scalar takes the plain
        ///     <c>abs(x) ** 2</c>; <c>masked</c> stays <c>masked</c>.
        /// </summary>
        /// <param name="danom">The complex anomaly.</param>
        /// <param name="self">The variance's operand, whose fill value numpy.ma writes over a non-finite square.</param>
        /// <returns>The squared magnitudes as a float64 masked array, a plain scalar, or <see cref="NDMasked"/>.</returns>
        private NDMaskedArray AbsSquaredLikeNumPy(NDMaskedArray danom, NDMaskedArray self)
        {
            if (ReferenceEquals(danom, NDMasked))
                return NDMasked;
            var data = danom._data;
            // getdata(2) is np.array(2), a strong int64: power(float64 |danom|, int64 2) runs the float64 loop.
            var two = NDArray.Scalar(2L).astype(np.float64);
            if (data.ndim == 0 && danom._mask is null)
                return new NDMaskedArray(np.power(np.abs(data), two), null);

            // umath.absolute(danom) and umath.power(fa, fb) are fresh ufunc results in NumPy's layout (danom's own axis
            // order), and so is np.where's: the masked sum that follows walks that memory order.
            var fa = UfuncResultBuffer(np.float64, data);
            np.abs(data, @out: fa);
            // power's values only: NumPy's DOUBLE_power takes its scalar-exponent fast path here (a stride-0 2.0 is
            // `x * x`), which the plain np.power call takes too; the out= route does not (it calls pow, 1 ULP off on
            // ~0.05 % of inputs), so the result buffer is filled by copies below rather than written by power.
            var pw = np.power(fa, two);

            // m = mask_or(getmask(a), nomask): danom's mask, shrunk to nomask when nothing is set.
            var m = ShrinkMask(danom._mask);
            NDArray result;
            if (m is null)
            {
                // np.where(nomask, fa, pw): the 0-d False abstains from the layout vote and selects pw everywhere; fa
                // and the power result share danom's layout, which the result takes.
                result = UfuncResultBuffer(np.float64, fa);
                np.copyto(result, pw);
            }
            else
            {
                // np.where(m, fa, pw) — NpyIter's layout over the three operands (pw votes with fa's layout, the one
                // NumPy's power result has), filled in two copies.
                result = UfuncResultBuffer(np.float64, m, fa, fa);
                np.copyto(result, pw);
                np.copyto(result, fa, where: m);
            }
            var invalid = np.logical_not(np.isfinite(result));
            NDArray rmask = m is null ? null : np.logical_or(m, invalid);
            if (np.any(invalid))
            {
                rmask ??= invalid;
                // result._data[invalid] = result.fill_value: the operand's fill value (real part), else float64's 1e20.
                object fv = self._fill_value ?? 1e20;
                if (fv is Complex c)
                    fv = c.Real;
                np.copyto(result, NDArray.Scalar(fv), casting: "unsafe", where: invalid);
            }
            return new NDMaskedArray(result, rmask);
        }

        /// <summary>
        ///     numpy.ma's <c>divide(a, b)</c> (<c>_DomainedBinaryOperation(umath.divide)</c>) with both operands STRONG: the
        ///     quotient in <c>result_type(a, b)</c>; a slot is masked where the quotient is not finite, an operand is masked,
        ///     or ma's divide domain <c>|a| * tiny &gt;= |b|</c> holds; a 0-d result collapses to <c>masked</c> or a plain
        ///     scalar; otherwise the masked slots are zeroed and <c>m * a</c> is ADDED to the whole result (NumPy's
        ///     <c>result += masked_da</c>, which also turns a -0.0 quotient of a positive dividend into +0.0).
        /// </summary>
        /// <param name="a">The dividend: a masked array, a plain scalar, or <see cref="NDMasked"/>.</param>
        /// <param name="b">The divisor (an unmasked count array or 0-d count).</param>
        /// <returns>The masked quotient, a plain scalar, or <see cref="NDMasked"/>.</returns>
        private NDMaskedArray MaDivideLikeNumPy(NDMaskedArray a, NDArray b)
        {
            var da = a._data;
            var (ca, cb) = StrongCast(da, b);
            // `self.f(da, db)`: a fresh quotient in NumPy's ufunc layout, voted by the original operands (a cast copy
            // would carry C strides). The mean it becomes votes again in the anomaly's subtract. true_divide's loop
            // keeps a float or complex dtype and divides integers (an integer `mean=` minus integer data) in float64.
            var tc = ca.typecode;
            var qdt = tc.IsFloatingPoint() || tc == NPTypeCode.Complex ? ca.dtype : np.float64;
            var result = UfuncResultBuffer(qdt, da, b);
            np.true_divide(ca, cb, @out: result);
            var m = np.logical_not(np.isfinite(result));
            if (a._mask is not null)
                m = np.logical_or(m, a._mask);
            m = np.logical_or(m, SafeDivideDomain(da, b));
            if (m.ndim == 0)
                return m.GetBoolean() ? NDMasked : new NDMaskedArray(result, null);
            if (!m.Shape.Equals(result.Shape))
                m = np.broadcast_to(m, result.Shape).copy();
            FillBack(result, da, m, domainedZeroFill: true, casting: "unsafe");
            return new NDMaskedArray(result, m);
        }

        /// <summary>
        ///     Anomalies (deviations from the mean along the axis): NumPy's <c>MaskedArray.anom</c>,
        ///     <c>m = self.mean(axis, dtype)</c>, then <c>self - m</c> when <c>not axis</c> (no axis, AND axis 0) and
        ///     <c>self - expand_dims(m, axis)</c> otherwise, through numpy.ma's subtract. The mean is the plain one (no
        ///     keepdims), which is observable: a flat or 1-D mean is a numpy scalar (an inf one unmasked, a float32 one turned
        ///     float64 by <c>dsum * 1.</c>) or <c>masked</c>, which then masks every anomaly.
        /// </summary>
        /// <param name="a">Operand: a <see cref="NDMaskedArray"/>, an <see cref="NDArray"/>, a C# array or scalar.</param>
        /// <param name="axis">The axis of the mean, or null for the flattened array.</param>
        /// <param name="dtype">The mean's <c>dtype=</c> (see <see cref="mean"/>).</param>
        /// <returns>The masked anomalies, the input's shape, in <c>result_type(a, mean)</c>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <remarks>Scoped by hand: the mean (and its expanded view) is only the subtraction's operand and is
        /// reclaimed.</remarks>
        public NDMaskedArray anom(object a, int? axis = null, DType dtype = null)
        {
            using var scope = NDScope.Open();
            var m = mean(a, axis, dtype);
            // `if not axis: return self - m` — None and 0 are both falsy, so axis 0 subtracts the reduced mean by
            // broadcasting (its shape is the input's without axis 0, which aligns from the right).
            if (axis is null || axis.Value == 0)
                return Yield(scope, MaSubtractLikeNumPy(a, m));
            return Yield(scope, MaSubtractLikeNumPy(a, expand_dims(m, axis.Value)));
        }

        /// <summary>Alias of <see cref="anom"/> (NumPy's <c>anomalies</c>).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="dtype">Mean accumulator dtype.</param>
        /// <returns>The masked anomalies.</returns>
        public NDMaskedArray anomalies(object a, int? axis = null, DType dtype = null) => anom(a, axis, dtype);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Covariance / correlation — PAIRWISE-COMPLETE over the mask (NumPy's ma.cov/corrcoef,
        //  a port of extras._covhelper + cov + corrcoef): each variable is centered by its own
        //  UNMASKED mean, and each covariance entry divides by the count of observations where
        //  BOTH variables are unmasked (so a missing value only drops the pairs it touches).
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Port of NumPy's <c>_covhelper</c>: returns the FILLED (masked→0) mean-CENTERED data, the
        /// float "not-masked" indicator matrix, and the resolved <paramref name="rowvar"/>. Centering uses the
        /// per-variable masked mean; the fill-to-0 is safe because the covariance product multiplies those
        /// positions against a 0 in the not-mask count anyway.</summary>
        private (NDArray filledCentered, NDArray xnotmask, bool rowvar) CovHelper(object x, object y, bool rowvar, bool allow_masked)
        {
            var xd = np.atleast_2d(AsData(x).astype(np.float64)).copy();
            var xmask = np.atleast_2d(getmaskarray(x));
            if (!allow_masked && np.any(xmask))
                throw new ValueError("Cannot process masked data.");
            if (y != null)
            {
                var yd = np.atleast_2d(AsData(y).astype(np.float64));
                var ymask = np.atleast_2d(getmaskarray(y));
                if (!allow_masked && np.any(ymask))
                    throw new ValueError("Cannot process masked data.");
                // Same-shape masked pairs get a COMMON mask (a value masked in one masks it in both).
                if ((np.any(xmask) || np.any(ymask)) && xd.Shape.Equals(yd.Shape))
                {
                    var common = np.logical_or(xmask, ymask);
                    xmask = common; ymask = common;
                }
                int catAxis = rowvar ? 0 : 1;
                xd = np.concatenate(new[] { xd, yd }, catAxis);
                xmask = np.concatenate(new[] { xmask, ymask }, catAxis);
            }
            if (xd.Shape.dimensions[0] == 1)
                rowvar = true;
            int meanAxis = rowvar ? 1 : 0;
            // Per-variable masked mean, broadcast back for the centering subtraction.
            var meanMa = mean(new NDMaskedArray(xd, np.any(xmask) ? xmask : null), meanAxis);
            var meanData = getdata(meanMa).astype(np.float64);
            NDArray meanB = rowvar
                ? np.reshape(meanData, new Shape(meanData.size, 1))
                : np.reshape(meanData, new Shape(1, meanData.size));
            var centered = np.subtract(xd, meanB);
            var filledCentered = np.any(xmask) ? np.where(xmask, NDArray.Scalar(0.0), centered) : centered;
            var xnotmask = np.logical_not(xmask).astype(np.float64);
            return (filledCentered, xnotmask, rowvar);
        }

        /// <summary>
        ///     Estimates the covariance matrix, PAIRWISE-COMPLETE over the mask (NumPy's <c>ma.cov</c>): each
        ///     entry divides by the number of observations where both variables are unmasked, minus
        ///     <c>ddof</c>. Any entry whose pairwise count is ≤ 0 (no complete observation) is MASKED.
        /// </summary>
        /// <param name="x">Observations (rows are variables when <paramref name="rowvar"/>, else columns).</param>
        /// <param name="y">Optional additional variables, stacked onto <paramref name="x"/> (a common mask is
        /// taken when same-shape).</param>
        /// <param name="rowvar">True (default): each ROW is a variable. False: each column is.</param>
        /// <param name="bias">Normalize by N (true) instead of N-1 — overridden by <paramref name="ddof"/>.</param>
        /// <param name="allow_masked">When false, raise if any value is masked (NumPy's flag).</param>
        /// <param name="ddof">Explicit delta DOF (divisor is pairwise-count − ddof); null uses bias.</param>
        /// <returns>The masked covariance matrix (squeezed to a scalar for a single variable).</returns>
        /// <exception cref="ValueError"><paramref name="allow_masked"/> is false and data is masked.</exception>
        /// <remarks>Scoped by hand: the centered/filled data, the not-mask indicator, the pairwise counts, the
        /// transposed products and the unsqueezed result are reclaimed; only the (squeezed) covariance and its
        /// mask leave.</remarks>
        public NDMaskedArray cov(object x, object y = null, bool rowvar = true, bool bias = false, bool allow_masked = true, int? ddof = null)
        {
            using var scope = NDScope.Open();
            int dd = ddof ?? (bias ? 0 : 1);
            var (centered, xnotmask, rv) = CovHelper(x, y, rowvar, allow_masked);
            NDArray fact, data;
            if (rv)
            {
                // fact[i,j] = # observations where variables i AND j are both unmasked, minus ddof.
                fact = np.subtract(np.dot(xnotmask, xnotmask.T), NDArray.Scalar((double)dd));
                data = np.divide(np.dot(centered, centered.T), fact);
            }
            else
            {
                fact = np.subtract(np.dot(xnotmask.T, xnotmask), NDArray.Scalar((double)dd));
                data = np.divide(np.dot(centered.T, centered), fact);
            }
            var mask = np.less_equal(fact, NDArray.Scalar(0.0));
            return Yield(scope, squeeze(new NDMaskedArray(data, np.any(mask) ? mask : null)));
        }

        /// <summary>
        ///     Pearson correlation coefficients from the pairwise-complete covariance (NumPy's <c>ma.corrcoef</c>):
        ///     <c>cov(x)</c> normalized by the outer product of the per-variable standard deviations.
        /// </summary>
        /// <param name="x">Observations (see <see cref="cov"/>).</param>
        /// <param name="y">Optional additional variables.</param>
        /// <param name="rowvar">True (default): each row is a variable.</param>
        /// <param name="allow_masked">When false, raise if any value is masked.</param>
        /// <returns>The masked correlation matrix.</returns>
        /// <exception cref="ValueError"><paramref name="allow_masked"/> is false and data is masked.</exception>
        /// <remarks>Scoped by hand: the covariance, its diagonal, the standard deviations and their outer product
        /// only feed the final division and are reclaimed.</remarks>
        public NDMaskedArray corrcoef(object x, object y = null, bool rowvar = true, bool allow_masked = true)
        {
            using var scope = NDScope.Open();
            var corr = cov(x, y, rowvar, false, allow_masked);
            // std = sqrt(diagonal(cov)); corr /= outer(std, std).
            var std = sqrt(diagonal(corr));
            return Yield(scope, divide(corr, outer(std, std)));
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Sliding products — convolve / correlate with mask propagation (NumPy's
        //  _convolve_or_correlate): the mask is computed by convolving/correlating the
        //  boolean masks against ones, so a result element's masked-ness follows exactly
        //  which input cells contributed to its sum.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Shared core of <see cref="convolve"/>/<see cref="correlate"/> (port of NumPy's
        /// <c>_convolve_or_correlate</c>). With <paramref name="propagate_mask"/> a result element is masked if
        /// ANY masked cell contributed to it (mask = the boolean masks slid against ones); without it, a result
        /// is masked only when NO unmasked cell contributed (and the data is computed from the 0-filled inputs).</summary>
        /// <remarks>Scoped by hand: the 0/1 mask casts, the ones kernels, the slid mask counts, the filled copies
        /// and a mask that turns out all-False are reclaimed; only the product and its mask leave.</remarks>
        private NDMaskedArray ConvolveOrCorrelate(Func<NDArray, NDArray, string, NDArray> f, object a, object v, string mode, bool propagate_mask)
        {
            using var scope = NDScope.Open();
            var da = getdata(a);
            var dv = getdata(v);
            NDArray data, mask;
            if (propagate_mask)
            {
                // Slide each operand's mask (as 0/1) against ones — a nonzero count means a masked cell contributed.
                var m1 = np.not_equal(f(getmaskarray(a).astype(np.int32), np.ones(DimsOf(dv), np.int32), mode), NDArray.Scalar(0));
                var m2 = np.not_equal(f(np.ones(DimsOf(da), np.int32), getmaskarray(v).astype(np.int32), mode), NDArray.Scalar(0));
                mask = np.logical_or(m1, m2);
                data = f(da, dv, mode);
            }
            else
            {
                // Masked iff NO unmasked pair contributed: ~(slide of the not-masks). Data from the 0-filled inputs.
                var contributed = np.not_equal(f(np.logical_not(getmaskarray(a)).astype(np.int32), np.logical_not(getmaskarray(v)).astype(np.int32), mode), NDArray.Scalar(0));
                mask = np.logical_not(contributed);
                // `f(filled(a, 0), filled(v, 0), mode=mode)`: over text the 0 fill is formatted first (the next phase).
                Exception deferral = null;
                data = f(FilledForNumericOp(a, 0, ref deferral), FilledForNumericOp(v, 0, ref deferral), mode);
                if (deferral is not null)
                    throw deferral;
            }
            return Yield(scope, new NDMaskedArray(data, np.any(mask) ? mask : null));
        }

        /// <summary>Discrete linear convolution of two 1-D sequences, propagating the mask (NumPy's
        /// <c>ma.convolve</c>). Default <paramref name="mode"/> is "full".</summary>
        /// <param name="a">First sequence.</param><param name="v">Second sequence.</param>
        /// <param name="mode">"full" (default), "same", or "valid".</param>
        /// <param name="propagate_mask">Mask a result if ANY masked cell contributed (true) vs only if NO
        /// unmasked cell did (false).</param>
        /// <returns>The masked convolution.</returns>
        public NDMaskedArray convolve(object a, object v, string mode = "full", bool propagate_mask = true)
            => ConvolveOrCorrelate((x, y, m) => np.convolve(x, y, m), a, v, mode, propagate_mask);

        /// <summary>Cross-correlation of two 1-D sequences, propagating the mask (NumPy's <c>ma.correlate</c>).
        /// Default <paramref name="mode"/> is "valid" (unlike <see cref="convolve"/>).</summary>
        /// <param name="a">First sequence.</param><param name="v">Second sequence.</param>
        /// <param name="mode">"valid" (default), "same", or "full".</param>
        /// <param name="propagate_mask">Mask a result if ANY masked cell contributed (true) vs only if NO
        /// unmasked cell did (false).</param>
        /// <returns>The masked cross-correlation.</returns>
        public NDMaskedArray correlate(object a, object v, string mode = "valid", bool propagate_mask = true)
            => ConvolveOrCorrelate((x, y, m) => np.correlate(x, y, m), a, v, mode, propagate_mask);

        /// <summary>
        ///     Applies <paramref name="func"/> repeatedly over the given <paramref name="axes"/> (NumPy's
        ///     <c>ma.apply_over_axes</c>): each pass calls <c>func(val, axis)</c> and — when the result dropped
        ///     the reduced axis — re-expands it, so a keepdims-less reduction still composes. <paramref name="func"/>
        ///     receives (and returns) a whole <see cref="NDMaskedArray"/>, so the mask is handled by the reduction
        ///     itself (this is why it composes cleanly, unlike per-1-D-slice <c>apply_along_axis</c>).
        /// </summary>
        /// <param name="func">The reduction, e.g. <c>(m, ax) =&gt; np.ma.sum(m, ax)</c>.</param>
        /// <param name="a">Operand.</param>
        /// <param name="axes">The axes to apply <paramref name="func"/> over, in order.</param>
        /// <returns>The successively-reduced masked array.</returns>
        /// <exception cref="ValueError"><paramref name="func"/> returns an array of the wrong rank.</exception>
        /// <remarks>
        ///     <b>Ownership</b> — the same contract as <see cref="np.apply_over_axes(Func{NDArray, int, NDArray}, NDArray, int[])"/>:
        ///     every masked array <paramref name="func"/> returns stays the CALLER's (this method never disposes
        ///     one — a callback may return an array it keeps using), so an allocating reduction's per-axis
        ///     intermediates are released only by a GC unless the caller runs the call inside an
        ///     <see cref="NDScope"/>. That is also why this method opens NO scope of its own: one would capture —
        ///     and dispose — whatever the callback allocates. The re-expanded views this method creates (and a
        ///     scalar operand it had to materialize) are its own and are released as soon as they are superseded.
        /// </remarks>
        public NDMaskedArray apply_over_axes(Func<NDMaskedArray, int, NDMaskedArray> func, object a, int[] axes)
        {
            NDMaskedArray val = asanyarray(a);
            // Whether the running value's parts are THIS method's own (a scalar/array-like operand asanyarray had
            // to materialize, or a re-expanded view) — only those are released when superseded; an array or
            // masked-array operand wraps the caller's buffers and a callback result is the caller's.
            bool valIsOurs = !(a is NDMaskedArray || a is NDArray);
            foreach (var ax in axes)
            {
                int axis = ax < 0 ? ax + val.ndim : ax;
                var res = func(val, axis);
                if (res.ndim == val.ndim)
                {
                    SupersedeMasked(ref val, ref valIsOurs, res, nextIsOurs: false);
                }
                else
                {
                    // NumPy re-expands a keepdims-less result along the reduced axis so the next pass lines up.
                    var expanded = expand_dims(res, axis);
                    if (expanded.ndim == val.ndim)
                    {
                        SupersedeMasked(ref val, ref valIsOurs, expanded, nextIsOurs: true);
                    }
                    else
                    {
                        // Our own views, never exposed — release them before refusing.
                        ReleaseParts(expanded);
                        throw new ValueError("function is not returning an array of the correct shape");
                    }
                }
            }
            return val;
        }

        /// <summary>
        ///     Advances <see cref="apply_over_axes(Func{NDMaskedArray, int, NDMaskedArray}, object, int[])"/>'s
        ///     running value to <paramref name="next"/>, releasing the superseded value's parts only when they are
        ///     the method's OWN (a re-expanded view or a materialized operand) — never a callback result, never the
        ///     caller's input.
        /// </summary>
        /// <param name="val">The running value (replaced by <paramref name="next"/>).</param>
        /// <param name="valIsOurs">Whether <paramref name="val"/>'s parts are this method's own (updated for <paramref name="next"/>).</param>
        /// <param name="next">The new running value.</param>
        /// <param name="nextIsOurs">Whether <paramref name="next"/>'s parts are this method's own.</param>
        private static void SupersedeMasked(ref NDMaskedArray val, ref bool valIsOurs, NDMaskedArray next, bool nextIsOurs)
        {
            // An identity callback hands the running value straight back: nothing is superseded, and ownership
            // is unchanged (it may still be our view).
            if (ReferenceEquals(val, next))
                return;
            // A callback may also return a NEW wrapper over the very same part objects (np.ma.asarray(x) and
            // kin alias rather than copy) — those parts now belong to `next` and must survive the release.
            if (valIsOurs)
                ReleaseParts(val, keep: next);
            val = next;
            valIsOurs = nextIsOurs;
        }

        /// <summary>
        ///     Disposes the data and (present) mask a masked array carries — for a masked array whose parts this
        ///     code created and is abandoning (a superseded view, a slice, a stand-in). Never call it on an array
        ///     whose parts belong to a caller: a masked array aliases, it does not own.
        /// </summary>
        /// <param name="m">The masked array whose parts to release; null and the <see cref="NDMasked"/> singleton
        /// are ignored (the singleton's scalars are process-wide).</param>
        /// <param name="keep">An optional masked array that must stay intact: any part <paramref name="m"/> shares
        /// with it BY REFERENCE (the same <see cref="NDArray"/> object, not merely the same buffer) is skipped. A
        /// view sharing only the buffer needs no such guard — it holds its own counted reference.</param>
        private static void ReleaseParts(NDMaskedArray m, NDMaskedArray keep = null)
        {
            if (m is null || m is NDMaskedConstant)
                return;
            ReleaseUnlessKept(m._data, keep);
            ReleaseUnlessKept(m._mask, keep);
        }

        /// <summary>Disposes <paramref name="part"/> unless it is null or one of <paramref name="keep"/>'s own parts
        /// (reference identity) — the guard behind <see cref="ReleaseParts"/>.</summary>
        /// <param name="part">A data or mask array being released.</param>
        /// <param name="keep">The masked array whose parts must survive, or null.</param>
        private static void ReleaseUnlessKept(NDArray part, NDMaskedArray keep)
        {
            if (part is null)
                return;
            if (keep is not null && (ReferenceEquals(part, keep._data) || ReferenceEquals(part, keep._mask)))
                return;
            part.Dispose();
        }

        /// <summary>Single-axis convenience for <see cref="apply_over_axes"/>.</summary>
        /// <param name="func">The reduction.</param><param name="a">Operand.</param><param name="axis">The axis.</param>
        /// <returns>The reduced masked array.</returns>
        public NDMaskedArray apply_over_axes(Func<NDMaskedArray, int, NDMaskedArray> func, object a, int axis)
            => apply_over_axes(func, a, new[] { axis });

        /// <summary>
        ///     Applies <paramref name="func1d"/> to each 1-D MASKED slice of <paramref name="arr"/> taken along
        ///     <paramref name="axis"/>, assembling the results (NumPy's <c>ma.apply_along_axis</c>). The slice is
        ///     handed over as a <see cref="NDMaskedArray"/> (via the indexer), so the mask rides through the
        ///     function; the output shape is <paramref name="arr"/>'s shape with the <paramref name="axis"/> entry
        ///     replaced by <paramref name="func1d"/>'s result shape (dropped entirely for a scalar result).
        /// </summary>
        /// <param name="func1d">The per-slice function; may return a scalar (0-d) or a 1-D masked array.</param>
        /// <param name="axis">The axis along which the 1-D slices are taken.</param>
        /// <param name="arr">Operand.</param>
        /// <returns>The assembled masked array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="NotSupportedException"><paramref name="func1d"/> returns a result of rank ≥ 2 (NumSharp
        /// supports the scalar- and 1-D-result cases; a higher-rank per-slice result needs the object-array
        /// assembly NumSharp lacks).</exception>
        /// <remarks>
        ///     <b>Ownership</b> — the same contract as <see cref="np.apply_along_axis"/>: every masked array
        ///     <paramref name="func1d"/> returns stays the CALLER's and is never disposed here (a callback may
        ///     return an array it keeps using), which is why the callbacks run OUTSIDE any scope this method opens.
        ///     The per-slice views handed to <paramref name="func1d"/> (and a scalar operand this method had to
        ///     materialize) are this method's own and are released once the result is assembled — on a throw too;
        ///     a callback must therefore not retain its ARGUMENT past the call. The assembly (stacking, reshaping,
        ///     the stand-in all-False masks) runs inside a scope, so only the result's data and mask leave.
        /// </remarks>
        public NDMaskedArray apply_along_axis(Func<NDMaskedArray, NDMaskedArray> func1d, int axis, object arr)
        {
            var a = asanyarray(arr);
            // A scalar/array-like operand is materialized by asanyarray and is ours; an array or masked-array
            // operand only wraps the caller's buffers.
            bool aIsOurs = !(arr is NDMaskedArray || arr is NDArray);
            // Every slice view built for func1d — ours, released in the finally once nothing reads them.
            var slices = new List<NDMaskedArray>();
            try
            {
                int nd = a.ndim;
                int ax = axis < 0 ? axis + nd : axis;
                if (ax < 0 || ax >= nd)
                    throw new AxisError($"axis {axis} is out of bounds for array of dimension {nd}");
                var shape = a.shape;
                var outerAxes = Enumerable.Range(0, nd).Where(d => d != ax).ToArray();
                var outerShape = outerAxes.Select(d => shape[d]).ToArray();

                // One masked 1-D slice at the given outer coordinate: ints on the outer axes, ":" on `axis`.
                NDMaskedArray SliceAt(long[] outer)
                {
                    var idx = new object[nd];
                    idx[ax] = Slice.All;
                    for (int j = 0; j < outerAxes.Length; j++)
                        idx[outerAxes[j]] = (int)outer[j];
                    var slice = (NDMaskedArray)a[idx];
                    slices.Add(slice);
                    return slice;
                }

                // C-order odometer over the outer index space (last outer axis fastest) so the flat result order
                // matches a reshape to outerShape.
                var results = new List<NDMaskedArray>();
                var coord = new long[outerAxes.Length];
                while (true)
                {
                    results.Add(func1d(SliceAt(coord)));
                    int p = outerAxes.Length - 1;
                    for (; p >= 0; p--)
                    {
                        if (++coord[p] < outerShape[p]) break;
                        coord[p] = 0;
                    }
                    if (p < 0) break;
                }

                // Assembly: every array built from here on is this method's own (no callback runs any more), so a
                // scope reclaims the stacks, reshapes and stand-in masks and the Yield keeps just the result.
                using var scope = NDScope.Open();
                var datas = results.Select(r => getdata(r)).ToArray();
                var masks = results.Select(r => getmaskarray(r)).ToArray();
                int resNd = results[0].ndim;
                if (resNd == 0)
                {
                    // Scalar per slice → output shape is exactly the outer shape.
                    var d = np.reshape(np.stack(datas, 0), Shape.Adopt(outerShape));
                    var m = np.reshape(np.stack(masks, 0), Shape.Adopt(outerShape));
                    return Yield(scope, new NDMaskedArray(d, np.any(m) ? m : null));
                }
                if (resNd == 1)
                {
                    // 1-D result of length L per slice → stack to (outerSize, L), reshape to (outerShape…, L), then
                    // move that trailing L axis into the original `axis` position (a.shape with axis → L).
                    var resLen = results[0].shape[0];
                    var newShape = outerShape.Concat(new[] { resLen }).ToArray();
                    var d = np.moveaxis(np.reshape(np.stack(datas, 0), Shape.Adopt(newShape)), outerAxes.Length, ax);
                    var m = np.moveaxis(np.reshape(np.stack(masks, 0), Shape.Adopt(newShape)), outerAxes.Length, ax);
                    return Yield(scope, new NDMaskedArray(d, np.any(m) ? m : null));
                }
                throw new NotSupportedException(
                    "np.ma.apply_along_axis supports a scalar or 1-D per-slice result; a rank-{resNd} result needs the object-array assembly NumSharp lacks.".Replace("{resNd}", resNd.ToString()));
            }
            finally
            {
                // The slice views were only the callbacks' arguments; the assembled result copied what it needed.
                // `keep: a` guards the degenerate case where indexing handed back the operand's OWN part object
                // (a full-extent index on a 1-D operand) — that object is the caller's, not a view of ours.
                foreach (var slice in slices)
                    ReleaseParts(slice, keep: a);
                if (aIsOurs)
                    ReleaseParts(a);
            }
        }

        /// <summary>Cumulative sum along the axis; masked slots contribute 0 to the running total but their
        /// POSITIONS stay masked in the result (NumPy semantics).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten, C-order).</param><param name="dtype">Accumulator dtype.</param>
        /// <returns>The masked cumulative sum.</returns>
        public NDMaskedArray cumsum(object a, int? axis = null, DType dtype = null)
            => Scan(a, 0, (x, ax) => np.cumsum(x, ax, dtype), axis);

        /// <summary>Cumulative product along the axis; masked slots contribute 1 but stay masked in the result.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flatten, C-order).</param><param name="dtype">Accumulator dtype.</param>
        /// <returns>The masked cumulative product.</returns>
        public NDMaskedArray cumprod(object a, int? axis = null, DType dtype = null)
            => Scan(a, 1, (x, ax) => np.cumprod(x, ax, dtype), axis);

        /// <summary>Shared scan (cumsum/cumprod): fill masked with the identity, scan, and carry the original
        /// per-position mask (raveled to match when axis is null).</summary>
        /// <param name="a">Operand.</param>
        /// <param name="identity">The scan identity written at masked slots (0 cumsum, 1 cumprod).</param>
        /// <param name="fn">The underlying <c>np.*</c> scan (data, axis) → result.</param>
        /// <param name="axis">Scan axis, or null to scan the flattened array.</param>
        /// <returns>The masked scan.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a masked slot whose scan would
        ///     compute (see <see cref="FilledForNumericOp"/>).</exception>
        /// <remarks>Scoped by hand — the one scope cumsum/cumprod ride: the identity-filled copy (and a
        /// materialized scalar operand) is reclaimed; the scan and its (possibly raveled) mask leave. NumPy's
        /// <c>self.filled(0).cumsum(...)</c> over TEXT formats the identity and then refuses the dtype
        /// (<c>add.accumulate</c> / multiply over strings): the scan runs over the unfilled text to raise that refusal.</remarks>
        private NDMaskedArray Scan(object a, object identity, Func<NDArray, int?, NDArray> fn, int? axis)
        {
            using var scope = NDScope.Open();
            var mask = (a as NDMaskedArray)?._mask;
            Exception deferral = null;
            var d = FilledForNumericOp(a, identity, ref deferral);
            var result = fn(d, axis);
            if (deferral is not null)
                throw deferral;
            if (mask is null)
                return Yield(scope, new NDMaskedArray(result, null));
            // axis=null flattens the result in C-order — ravel the mask to match.
            var m = axis is null ? np.ravel(mask) : mask;
            return Yield(scope, new NDMaskedArray(result, m));
        }

        /// <summary>Indices of the minimum along the axis, treating masked slots as the dtype's largest value
        /// (NumPy's <c>argmin</c>) — a plain int64 array, never masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flattened index).</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="minimum_fill_value"/>.</param>
        /// <returns>An int64 <see cref="NDArray"/> of indices.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the fill-substituted copy the index is computed over is reclaimed.</remarks>
        [NDScoped]
        public NDArray argmin(object a, int? axis = null, object fill_value = null)
        {
            var d = ArgFilled(a, fill_value ?? minimum_fill_value(a));
            return axis is null ? NDArray.Scalar(np.argmin(d)) : np.argmin(d, axis.Value);
        }

        /// <summary>Indices of the maximum along the axis, treating masked slots as the dtype's smallest value.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null (flattened index).</param>
        /// <param name="fill_value">Override the masked fill; null uses <see cref="maximum_fill_value"/>.</param>
        /// <returns>An int64 <see cref="NDArray"/> of indices.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the fill-substituted copy the index is computed over is reclaimed.</remarks>
        [NDScoped]
        public NDArray argmax(object a, int? axis = null, object fill_value = null)
        {
            var d = ArgFilled(a, fill_value ?? maximum_fill_value(a));
            return axis is null ? NDArray.Scalar(np.argmax(d)) : np.argmax(d, axis.Value);
        }

        /// <summary>Data with masked slots filled for an arg-reduction (null mask ⇒ raw data).</summary>
        private static NDArray ArgFilled(object a, object fill)
            => (a as NDMaskedArray)?._mask is null ? AsData(a) : ((NDMaskedArray)a).filled(fill);

        /// <summary>Logical-AND reduction: masked slots are treated as True (excluded); a cell whose whole
        /// slice was masked is itself masked.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked all-reduction (bool data).</returns>
        public NDMaskedArray all(object a, int? axis = null, bool keepdims = false)
            => ReduceIdentity(a, true, (x, ax, kd) => ax is null ? NDArray.Scalar(np.all(x)) : np.all(x, ax, null, kd), axis, keepdims);

        /// <summary>Logical-OR reduction: masked slots are treated as False (excluded).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>The masked any-reduction (bool data).</returns>
        public NDMaskedArray any(object a, int? axis = null, bool keepdims = false)
            => ReduceIdentity(a, false, (x, ax, kd) => ax is null ? NDArray.Scalar(np.any(x)) : np.any(x, ax, null, kd), axis, keepdims);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Creation — fresh, UNMASKED arrays (NumPy's `_convert2ma`: a masked wrapper over
        //  the corresponding np.* creation routine, with no element masked).
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Masked array of zeros (no element masked). float64 by default.</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of zeros.</returns>
        public NDMaskedArray zeros(Shape shape, DType dtype = null) => new NDMaskedArray(np.zeros(shape, dtype ?? np.float64), null);

        /// <summary>Masked array of ones (no element masked).</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of ones.</returns>
        public NDMaskedArray ones(Shape shape, DType dtype = null) => new NDMaskedArray(np.ones(shape, dtype ?? np.float64), null);

        /// <summary>Masked array of uninitialized values (no element masked).</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked array of arbitrary contents.</returns>
        public NDMaskedArray empty(Shape shape, DType dtype = null) => new NDMaskedArray(np.empty(shape, dtype ?? np.float64), null);

        /// <summary>Masked zeros shaped like <paramref name="a"/> (NumPy's <c>ma.zeros_like</c>, <c>np.zeros_like</c> of the
        /// masked array): a masked prototype lends a COPY of its mask (its masked slots stay masked over the zeros), its fill
        /// value (converted to the result dtype) and its hard-mask flag; a plain prototype gives an unmasked result.</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype (null ⇒ prototype's).</param>
        /// <returns>A masked array of zeros.</returns>
        /// <exception cref="TypeError">The prototype's text fill cannot be converted into a text <paramref name="dtype"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">A zero must be written into text, or a fill converted
        ///     between text and numbers (the next phase).</exception>
        public NDMaskedArray zeros_like(object a, DType dtype = null)
        {
            // Scoped by hand: a scalar/array-like prototype is materialized only to read its shape/dtype.
            using var scope = NDScope.Open();
            return Yield(scope, LikePrototype(a, np.zeros_like(AsData(a), dtype)));
        }

        /// <summary>Masked ones shaped like <paramref name="a"/> (NumPy's <c>ma.ones_like</c>): the prototype's mask (a
        /// copy), fill value and hardness ride along, as in <see cref="zeros_like"/>.</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype.</param>
        /// <returns>A masked array of ones.</returns>
        /// <exception cref="TypeError">See <see cref="zeros_like"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">See <see cref="zeros_like"/>.</exception>
        public NDMaskedArray ones_like(object a, DType dtype = null)
        {
            // Scoped by hand: a scalar/array-like prototype is materialized only to read its shape/dtype.
            using var scope = NDScope.Open();
            return Yield(scope, LikePrototype(a, np.ones_like(AsData(a), dtype)));
        }

        /// <summary>Masked uninitialized array shaped like <paramref name="a"/> (NumPy's <c>ma.empty_like</c>): the
        /// prototype's mask (a copy), fill value and hardness ride along, as in <see cref="zeros_like"/>.</summary>
        /// <param name="a">Prototype array-like.</param><param name="dtype">Override dtype.</param>
        /// <returns>A masked array of arbitrary contents (a <c>U</c> result is zero-filled, as NumPy's is).</returns>
        /// <exception cref="TypeError">See <see cref="zeros_like"/>.</exception>
        public NDMaskedArray empty_like(object a, DType dtype = null)
        {
            // Scoped by hand: a scalar/array-like prototype is materialized only to read its shape/dtype.
            using var scope = NDScope.Open();
            return Yield(scope, LikePrototype(a, np.empty_like(AsData(a), dtype)));
        }

        /// <summary>
        ///     The masked result of NumPy's <c>np.*_like(masked_prototype)</c>: <c>np.empty_like</c> makes a masked array
        ///     whose <c>__array_finalize__</c> copies the prototype's mask (<c>_mask.astype(..., order)</c> — never the
        ///     prototype's own buffer), takes its hard-mask flag and converts its fill value to the result dtype; the data is
        ///     then filled (<c>np.copyto(res, 0)</c> touches the data only). A plain prototype gives an unmasked result.
        /// </summary>
        /// <param name="a">The prototype.</param>
        /// <param name="data">The result's data (already filled).</param>
        /// <returns>The masked result.</returns>
        /// <exception cref="TypeError">The prototype's text fill cannot be converted into the result's text dtype.</exception>
        private static NDMaskedArray LikePrototype(object a, NDArray data)
        {
            if (a is not NDMaskedArray source)
                return new NDMaskedArray(data, null);
            return new NDMaskedArray(data, source._mask?.copy()).FinalizeFrom(source);
        }

        /// <summary>Masked evenly-spaced values in [0, stop) (no element masked).</summary>
        /// <param name="stop">Exclusive upper bound.</param><returns>An unmasked masked range.</returns>
        public NDMaskedArray arange(int stop) => new NDMaskedArray(np.arange(stop), null);

        /// <summary>Masked evenly-spaced values in [start, stop) stepping by <paramref name="step"/>.</summary>
        /// <param name="start">Inclusive start.</param><param name="stop">Exclusive stop.</param><param name="step">Step.</param>
        /// <returns>An unmasked masked range.</returns>
        public NDMaskedArray arange(int start, int stop, int step = 1) => new NDMaskedArray(np.arange(start, stop, step), null);

        /// <summary>Masked evenly-spaced floating values in [start, stop) stepping by <paramref name="step"/>.</summary>
        /// <param name="start">Inclusive start.</param><param name="stop">Exclusive stop.</param><param name="step">Step.</param>
        /// <returns>An unmasked masked range.</returns>
        public NDMaskedArray arange(double start, double stop, double step = 1) => new NDMaskedArray(np.arange(start, stop, step), null);

        /// <summary>Masked identity matrix (no element masked).</summary>
        /// <param name="n">Order.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>An unmasked masked identity.</returns>
        public NDMaskedArray identity(int n, DType dtype = null) => new NDMaskedArray(np.identity(n, dtype), null);

        /// <summary>Masked grid-index array for <paramref name="dimensions"/> (no element masked).</summary>
        /// <param name="dimensions">Grid shape (32-bit spelling).</param><param name="dtype">Index dtype.</param>
        /// <returns>An unmasked masked index grid.</returns>
        public NDMaskedArray indices(int[] dimensions, DType dtype = null) => new NDMaskedArray(np.indices(dimensions, dtype), null);

        /// <summary>Masked grid-index array for <paramref name="dimensions"/> (no element masked) —
        ///     the house <see cref="long"/> shape spelling, so an <see cref="NDArray"/>'s own <c>shape</c>
        ///     binds without a down-cast.</summary>
        /// <param name="dimensions">Grid shape.</param><param name="dtype">Index dtype.</param>
        /// <returns>An unmasked masked index grid.</returns>
        public NDMaskedArray indices(long[] dimensions, DType dtype = null) => new NDMaskedArray(np.indices(dimensions, dtype), null);

        /// <summary>Masked grid-index array for <paramref name="dimensions"/> (no element masked) —
        ///     <see cref="Shape"/> spelling (accepts an <see cref="NDArray.Shape"/> or a value tuple).</summary>
        /// <param name="dimensions">Grid shape.</param><param name="dtype">Index dtype.</param>
        /// <returns>An unmasked masked index grid.</returns>
        public NDMaskedArray indices(Shape dimensions, DType dtype = null) => new NDMaskedArray(np.indices(dimensions, dtype), null);

        // ─────────────────────────────────────────────────────────────────────────────
        //  masked_* constructors — build a mask from a condition/value and OR it onto any
        //  existing mask (NumPy's masked_where family). copy=true (default) copies the data.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Masks <paramref name="a"/> wherever <paramref name="condition"/> is True, OR'ing onto any
        /// existing mask (NumPy's <c>masked_where</c> — the primitive the rest of this family builds on).</summary>
        /// <param name="condition">Array-like of truth values (<see cref="make_mask"/>: non-zero numbers, non-empty
        ///     strings; a masked condition reads True at its masked slots); True ⇒ mask that element. A 0-d condition
        ///     applies to every element.</param>
        /// <param name="a">Data array-like.</param>
        /// <param name="copy">Copy the data (default true). False makes the result a VIEW of <paramref name="a"/> and masks
        ///     a masked <paramref name="a"/> IN PLACE, as NumPy's does: the combined mask is written into its mask buffer
        ///     (the result shares it), or — when it had no mask — it adopts the result's new mask.</param>
        /// <returns>A masked array of <paramref name="a"/>'s data with the new mask; it keeps a masked
        ///     <paramref name="a"/>'s fill value and hardness.</returns>
        /// <exception cref="IndexError">A condition with dimensions whose shape is not <paramref name="a"/>'s
        ///     (<c>Inconsistent shape between the condition and the input (got (2,) and (3,))</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text condition (its strings' truth values,
        ///     the next phase of the string support).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements: <c>cond = make_mask(condition, shrink=False)</c>, then <c>if cshape and cshape !=
        ///     ashape</c> the IndexError (a condition is never broadcast, only a 0-d one fills the array), then
        ///     <c>mask_or(cond, a._mask)</c> for a masked input and <c>result.mask = _shrink_mask(cond)</c>. The shape test
        ///     runs first here: NumPy's conversion before it cannot fail, so the shapes decide the error even for a text
        ///     condition.
        ///     </para>
        ///     <para>
        ///     The mask's FORM follows NumPy's: an all-False condition over an input without a mask leaves no mask
        ///     (<c>mask=False</c> in the repr), while an input that already had a mask array keeps an array (the setter
        ///     writes the condition into it).
        ///     </para>
        ///     <para>Scoped by hand: the boolean-coerced condition copy (when an existing mask is OR'd onto it) and
        ///     any materialized operand are reclaimed; only the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray masked_where(object condition, object a, bool copy = true)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            // A masked condition keeps its masked-array form for make_mask (its masked slots read True); any other
            // condition is materialized once.
            object condOperand = condition is NDMaskedArray ? condition : AsData(condition);
            var condShape = AsData(condOperand).Shape;
            // `if cshape and cshape != ashape`: a 0-d condition (an empty tuple) is exempt; any other shape must match.
            if (condShape.NDim != 0 && !condShape.dimensions.AsSpan().SequenceEqual(d.Shape.dimensions))
                throw new IndexError("Inconsistent shape between the condition and the input " +
                                     $"(got {ArrayFormatter.ShapeTuple(condShape.dimensions)} and {ArrayFormatter.ShapeTuple(d.Shape.dimensions)})");
            // make_mask(condition, shrink=False): always an array, a fresh one (text: the next phase unless empty).
            var cond = make_mask(condOperand, shrink: false);
            var source = a as NDMaskedArray;
            var existing = source?._mask;
            NDArray m;
            if (existing is not null)
                // mask_or(cond, a._mask), written into the input's (copied) mask array: an array however few slots are set.
                m = np.logical_or(cond, existing);
            else if (!np.any(cond))
                // _shrink_mask(cond) is nomask, and assigning nomask to an unmasked array leaves it unmasked.
                m = null;
            else
                // `current_mask.flat = cond`: a 0-d condition fills every slot.
                m = cond.ndim == 0 ? np.broadcast_to(cond, DimsOf(d)).copy() : cond;
            NDMaskedArray result;
            if (!copy && source is not null)
            {
                // copy=False masks the INPUT in place: `a = np.array(a, copy=False, subok=True)` is the input itself and
                // `result = a.view(cls)` shares its mask, so `result.mask = cond` writes the combined mask INTO the
                // input's mask buffer; an input without a mask adopts the result's new one (`a._mask =
                // result._mask.view()`), detached from this scope so it lives as long as the input.
                if (existing is not null)
                {
                    np.copyto(existing, m);
                    result = new NDMaskedArray(d, existing.view());
                }
                else
                {
                    result = new NDMaskedArray(d, m);
                    if (m is not null)
                        source._mask = Own(m.view());
                }
            }
            else
                result = new NDMaskedArray(copy ? d.copy() : d, m);
            // `a = np.array(a, copy=copy, subok=True)` finalizes FROM the masked input: its hard-mask flag and its fill value,
            // converted to the dtype (a materialized 'N/A' default of a <U1 input becomes 'N').
            if (source is not null)
                result.FinalizeFrom(source);
            return Yield(scope, result);
        }

        /// <summary>Masks where <c>a == value</c> (NumPy's <c>masked_equal</c>), and makes <paramref name="value"/> the
        /// result's fill value.</summary>
        /// <param name="a">Data.</param><param name="value">Value to mask (a scalar or an array-like).</param>
        /// <param name="copy">Copy the data.</param>
        /// <returns>The masked array (fill_value = <paramref name="value"/>, checked against the dtype).</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (text against a number:
        ///     <c>(&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class 'numpy.dtypes.Float64DType'&gt;)</c>), or the fill check
        ///     refuses a <see cref="string"/> value for numeric data.</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text (the next phase of
        ///     the string support).</exception>
        /// <remarks>NumPy's statements: <c>output = masked_where(equal(x, value), x, copy=copy); output.fill_value =
        /// value</c>, with numpy.ma's masked <c>equal</c>: a bare scalar becomes a STRONG default-width array
        /// (<c>getdata(1.5)</c> is float64), so the comparison of float32 data runs in float64. Scoped by hand: the
        /// comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_equal(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            var output = masked_where(equal(a, BytesAsScalar(value)), a, copy);
            // output.fill_value = value: NumPy's setter, so the value is checked against the data's dtype.
            output.fill_value = value;
            return Yield(scope, output);
        }

        /// <summary>Masks where <c>a != value</c> (NumPy's <c>masked_not_equal</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Value kept unmasked.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (see <see cref="masked_equal"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>NumPy's <c>masked_where(not_equal(x, value), x, copy=copy)</c> with numpy.ma's masked comparison.
        /// Scoped by hand: the comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_not_equal(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            return Yield(scope, masked_where(not_equal(a, BytesAsScalar(value)), a, copy));
        }

        /// <summary>Masks where <c>a &gt; value</c> (NumPy's <c>masked_greater</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (see <see cref="masked_equal"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>NumPy's <c>masked_where(greater(x, value), x, copy=copy)</c> with numpy.ma's masked comparison.
        /// Scoped by hand: the comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_greater(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            return Yield(scope, masked_where(greater(a, BytesAsScalar(value)), a, copy));
        }

        /// <summary>Masks where <c>a &gt;= value</c> (NumPy's <c>masked_greater_equal</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (see <see cref="masked_equal"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>NumPy's <c>masked_where(greater_equal(x, value), x, copy=copy)</c> with numpy.ma's masked comparison.
        /// Scoped by hand: the comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_greater_equal(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            return Yield(scope, masked_where(greater_equal(a, BytesAsScalar(value)), a, copy));
        }

        /// <summary>Masks where <c>a &lt; value</c> (NumPy's <c>masked_less</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (see <see cref="masked_equal"/>;
        ///     the text names <c>'less'</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>NumPy's <c>masked_where(less(x, value), x, copy=copy)</c> with numpy.ma's masked comparison.
        /// Scoped by hand: the comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_less(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            return Yield(scope, masked_where(less(a, BytesAsScalar(value)), a, copy));
        }

        /// <summary>Masks where <c>a &lt;= value</c> (NumPy's <c>masked_less_equal</c>).</summary>
        /// <param name="a">Data.</param><param name="value">Threshold.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">NumPy's comparison has no loop for the operands (see <see cref="masked_equal"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>NumPy's <c>masked_where(less_equal(x, value), x, copy=copy)</c> with numpy.ma's masked comparison.
        /// Scoped by hand: the comparison only feeds masked_where's condition.</remarks>
        public NDMaskedArray masked_less_equal(object a, object value, bool copy = true)
        {
            using var scope = NDScope.Open();
            return Yield(scope, masked_where(less_equal(a, BytesAsScalar(value)), a, copy));
        }

        /// <summary>Masks the CLOSED interval where <c>v1 &lt;= a &lt;= v2</c> (bounds swapped if <c>v2 &lt; v1</c>),
        /// NumPy's <c>masked_inside</c>.</summary>
        /// <param name="a">Data.</param>
        /// <param name="v1">One bound: a number, a <see cref="string"/> / <see cref="byte"/>[] for text data, or an
        ///     array-like.</param>
        /// <param name="v2">Other bound.</param>
        /// <param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">Python cannot order the bounds (<c>'&lt;' not supported between instances of 'str' and
        ///     'float'</c>), or NumPy's comparison has no loop for the data and a bound (text data against a C# number names
        ///     the Python scalar's class: <c>(&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class
        ///     'numpy.dtypes._PyFloatDType'&gt;)</c>, whatever the data's size).</exception>
        /// <exception cref="ValueError">Array bounds whose comparison has more than one element (Python's truth test of
        ///     <c>v2 &lt; v1</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>if v2 &lt; v1: (v1, v2) = (v2, v1)</c> (a Python comparison: a NaN bound never swaps),
        ///     <c>xf = filled(x)</c>, <c>condition = (xf &gt;= v1) &amp; (xf &lt;= v2)</c> with the RAW bounds — a C# number is a
        ///     weak Python scalar there, so float16/float32 data compare at their own width — then masked_where.
        ///     <para>Scoped by hand: the two bound comparisons and their conjunction only feed masked_where.</para>
        /// </remarks>
        public NDMaskedArray masked_inside(object a, object v1, object v2, bool copy = true)
        {
            using var scope = NDScope.Open();
            if (PythonLess(v2, v1))
                (v1, v2) = (v2, v1);
            var xf = filled(a);
            // Python evaluates the left comparison first, so its refusal is the one reported.
            var lower = CompareWithPythonValue("greater_equal", xf, v1);
            var upper = CompareWithPythonValue("less_equal", xf, v2);
            return Yield(scope, masked_where(WithBoundMasks(np.logical_and(lower, upper), v1, v2), a, copy));
        }

        /// <summary>Masks OUTSIDE the closed interval, where <c>a &lt; v1 || a &gt; v2</c> (bounds swapped if
        /// <c>v2 &lt; v1</c>), NumPy's <c>masked_outside</c>.</summary>
        /// <param name="a">Data.</param>
        /// <param name="v1">One bound (see <see cref="masked_inside"/>).</param>
        /// <param name="v2">Other bound.</param>
        /// <param name="copy">Copy the data.</param>
        /// <returns>The masked array.</returns>
        /// <exception cref="TypeError">See <see cref="masked_inside"/>; the comparison refusal names <c>'less'</c>, the
        ///     first comparison NumPy evaluates here.</exception>
        /// <exception cref="ValueError">See <see cref="masked_inside"/>.</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>
        ///     NumPy's statements: the same swap, <c>xf = filled(x)</c>, <c>condition = (xf &lt; v1) | (xf &gt; v2)</c>, then
        ///     masked_where. A NaN bound is not swapped away: <c>masked_outside(x, nan, 2.0)</c> masks <c>x &gt; 2</c>.
        ///     <para>Scoped by hand: the two bound comparisons and their disjunction only feed masked_where.</para>
        /// </remarks>
        public NDMaskedArray masked_outside(object a, object v1, object v2, bool copy = true)
        {
            using var scope = NDScope.Open();
            if (PythonLess(v2, v1))
                (v1, v2) = (v2, v1);
            var xf = filled(a);
            var below = CompareWithPythonValue("less", xf, v1);
            var above = CompareWithPythonValue("greater", xf, v2);
            return Yield(scope, masked_where(WithBoundMasks(np.logical_or(below, above), v1, v2), a, copy));
        }

        /// <summary>Masks non-finite elements (NaN/±inf) — NumPy's <c>masked_invalid</c>. For exact dtypes nothing is
        /// masked (every value is finite), but the result always carries a mask ARRAY.</summary>
        /// <param name="a">Data.</param><param name="copy">Copy the data.</param>
        /// <returns>The masked array; its mask is an array even when nothing is masked (NumPy's gh-22842 rule).</returns>
        /// <exception cref="TypeError">Text data: NumPy's <c>np.isfinite</c> has no loop for it (<c>ufunc 'isfinite' not
        ///     supported for the input types, …</c>), whatever the size.</exception>
        /// <remarks>NumPy's statements: <c>res = masked_where(~(np.isfinite(a)), a, copy=copy)</c>, then <c>if res._mask
        /// is nomask: res._mask = make_mask_none(res.shape)</c>. Scoped by hand: the non-finite predicate (or its
        /// all-False stand-in) only feeds masked_where.</remarks>
        public NDMaskedArray masked_invalid(object a, bool copy = true)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            // np.isfinite raises its no-loop TypeError for text before anything is masked; NonFiniteMask skips the call for
            // the exact dtypes, whose elements are all finite.
            var cond = d.typecode == NPTypeCode.String
                ? np.logical_not(np.isfinite(d))
                : NonFiniteMask(d) ?? np.zeros(DimsOf(d), np.@bool);
            var res = masked_where(cond, a, copy);
            // masked_invalid never returns nomask (matplotlib relies on it): an unmasked result gets a full False mask.
            if (res._mask is null)
                res._mask = np.zeros(DimsOf(res._data), np.@bool);
            return Yield(scope, res);
        }

        /// <summary>Masks elements approximately equal to <paramref name="value"/> (NumPy's <c>masked_values</c>:
        /// <c>isclose</c> with the given tolerances for float data, exact equality otherwise); the result's fill value
        /// becomes <paramref name="value"/>.</summary>
        /// <param name="a">Data.</param>
        /// <param name="value">Value to mask around: a number, a <see cref="string"/> / <see cref="byte"/>[] for text data,
        ///     or an array-like.</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param>
        /// <param name="copy">Copy the data.</param>
        /// <param name="shrink">Collapse an all-False mask to no mask (NumPy's <c>shrink=True</c>).</param>
        /// <returns>The masked array (fill_value = <paramref name="value"/>, checked against the dtype).</returns>
        /// <exception cref="TypeError">Text data with a C# number (NumPy's <c>equal</c> has no loop: the text names the
        ///     Python scalar's class, <c>_PyFloatDType</c> / <c>_PyLongDType</c>), or a masked numeric array with a
        ///     <see cref="string"/> value (the fill check).</exception>
        /// <exception cref="DTypePromotionError">Float data with a text value (<c>np.isclose</c> cannot promote it).</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text, or a number must be
        ///     formatted as text.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>xnew = filled(x, value)</c>, the mask from <c>isclose</c> (float) or
        ///     <c>umath.equal(xnew, value)</c>, <c>masked_array(xnew, mask=mask, copy=copy, fill_value=value)</c>, then
        ///     <c>shrink_mask()</c>. For text data and a C# number NumPy formats the number into the masked slots first, which
        ///     cannot fail and keeps the dtype, so the comparison's TypeError is its answer: it is raised up front.
        ///     <para>Scoped by hand: the fill-substituted copy (when a further copy is taken), the comparison scalars and a
        ///     mask that shrinks away are reclaimed; only the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray masked_values(object a, object value, double rtol = 1e-5, double atol = 1e-8, bool copy = true, bool shrink = true)
        {
            using var scope = NDScope.Open();
            var data = AsData(a);
            if (data.typecode == NPTypeCode.String && PythonScalarClass(value) is { } scalarClass)
                throw ComparisonNoLoop("equal", data.dtype.Meta, scalarClass);
            // NumPy fills masked positions with `value` FIRST (xnew = filled(x, value)), THEN masks by
            // approximate equality — so a pre-masked slot's data becomes `value` and is re-masked. A numeric ARRAY value
            // for text data is formatted into the masked slots (the next phase) and then refused by `equal` (no loop for
            // text and numbers): the comparison runs over the unfilled text so that refusal is raised.
            Exception deferral = null;
            var xnew = FilledForNumericOp(a, value, ref deferral);
            bool isFloat = xnew.typecode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;
            var mask = isFloat
                ? np.isclose(xnew, value as NDArray ?? (value as NDMaskedArray)?._data ?? NDArray.Scalar(value), rtol, atol)
                : CompareWithPythonValue("equal", xnew, value);
            if (deferral is not null)
                throw deferral;
            var ret = array(xnew, mask, fill_value: value, copy: copy);
            // ret.shrink_mask(): an all-False mask collapses to nomask.
            if (shrink && ret._mask is not null && !np.any(ret._mask))
                ret._mask = null;
            return Yield(scope, ret);
        }

        /// <summary>Masks elements exactly equal to <paramref name="value"/> (NumPy's <c>masked_object</c>), and makes
        /// <paramref name="value"/> the result's fill value.</summary>
        /// <param name="a">Data.</param><param name="value">Value to mask (compared as a raw Python value).</param>
        /// <param name="copy">Copy the data.</param>
        /// <param name="shrink">Shrink the comparison's mask to no mask when nothing matched (NumPy's
        ///     <c>shrink=True</c>).</param>
        /// <returns>The masked array (fill_value = <paramref name="value"/>, checked against the dtype).</returns>
        /// <exception cref="TypeError">NumPy's <c>equal</c> has no loop for the data and the value (text data with a C#
        ///     number names the Python scalar's class, <c>_PyFloatDType</c> / <c>_PyLongDType</c>; a bool names
        ///     <c>BoolDType</c>), or the fill check refuses a <see cref="string"/> value for numeric data.</exception>
        /// <exception cref="StringComputationNotSupportedException">The comparison reads non-empty text.</exception>
        /// <remarks>
        ///     Unlike <see cref="masked_equal"/>, NumPy compares with <c>umath.equal(x._data, value)</c> — the RAW value, a
        ///     weak Python scalar — so float16/float32 data compare at their own width (<c>masked_object(float32 [1.1],
        ///     1.1)</c> masks the element, <c>masked_equal</c> does not). Then <c>mask_or(x._mask, make_mask(condition,
        ///     shrink=shrink))</c> and <c>masked_array(x, mask=mask, copy=copy, fill_value=value)</c>.
        ///     <para>Scoped by hand: the comparison and the combined mask only feed the constructor.</para>
        /// </remarks>
        public NDMaskedArray masked_object(object a, object value, bool copy = true, bool shrink = true)
        {
            using var scope = NDScope.Open();
            var condition = WithBoundMasks(CompareWithPythonValue("equal", AsData(a), value), value, null);
            var mask = mask_or((a as NDMaskedArray)?._mask, make_mask(condition, shrink));
            // `masked_array(x, mask=mask, copy=copy, fill_value=value)`: the class constructor, so x's hardness rides along.
            return Yield(scope, masked_array(a, mask, fill_value: value, copy: copy));
        }

        /// <summary>
        ///     A value for numpy.ma's masked comparisons: a C# <see cref="byte"/>[] becomes NumPy's <c>bytes</c> scalar (a
        ///     0-d <c>S</c> array, the house map); anything else passes through raw, so a bare C# number still reaches
        ///     numpy.ma's strong-scalar promotion (<see cref="BareScalarStrongDtype"/>).
        /// </summary>
        /// <param name="value">The value as the caller gave it.</param>
        /// <returns>The value to compare against.</returns>
        private static object BytesAsScalar(object value) => value is byte[] bytes ? NDArray.Scalar(bytes) : value;

        /// <summary>
        ///     <c>umath.&lt;ufunc&gt;(x, value)</c> with <paramref name="value"/> as Python passes it: a C# number is a WEAK
        ///     Python scalar (float16/float32 data compare at their own width, as NEP 50 casts the scalar to the array's
        ///     float), text data against a Python number has no loop whatever its size, and any other value (text, bytes, an
        ///     array-like) is an ordinary operand.
        /// </summary>
        /// <param name="ufunc">The comparison: <c>"equal"</c>, <c>"less"</c>, <c>"less_equal"</c>, <c>"greater"</c> or
        ///     <c>"greater_equal"</c>.</param>
        /// <param name="x">The data side.</param>
        /// <param name="value">The value side, raw.</param>
        /// <returns>The boolean comparison of the data (a masked-array <paramref name="value"/> compares by its data; its
        ///     mask joins through <see cref="WithBoundMasks"/>).</returns>
        /// <exception cref="TypeError">No loop for the pair (NumPy's class-naming UFuncTypeError).</exception>
        /// <exception cref="StringComputationNotSupportedException">Two non-empty text operands: the next phase.</exception>
        private static NDArray CompareWithPythonValue(string ufunc, NDArray x, object value)
        {
            if (PythonScalarClass(value) is { } scalarClass && x.typecode == NPTypeCode.String)
                throw ComparisonNoLoop(ufunc, x.dtype.Meta, scalarClass);
            NDArray operand;
            if (IsWeakPythonNumber(value) && x.typecode is NPTypeCode.Half or NPTypeCode.Single)
                // NEP 50: the Python number takes the array's float width, so the comparison runs at float16/float32.
                operand = NDArray.Scalar(value).astype(x.dtype);
            else
                operand = value as NDArray ?? (value as NDMaskedArray)?._data ?? AsValueOperand(value);
            return ufunc switch
            {
                "equal" => np.equal(x, operand),
                "less" => np.less(x, operand),
                "less_equal" => np.less_equal(x, operand),
                "greater" => np.greater(x, operand),
                "greater_equal" => np.greater_equal(x, operand),
                _ => throw new ArgumentOutOfRangeException(nameof(ufunc), ufunc, "not a comparison ufunc"),
            };
        }

        /// <summary>
        ///     A condition computed against raw values, with the masks of masked-array values joined in: the raw ufunc's result
        ///     on a masked operand is masked where that operand is (and the elementwise <c>&amp;</c> / <c>|</c> of two such
        ///     results is masked where either is), and <see cref="make_mask"/> reads a masked slot as True. So every slot a
        ///     value masks becomes True in the condition.
        /// </summary>
        /// <param name="condition">The data condition.</param>
        /// <param name="v1">The first raw value, or null.</param>
        /// <param name="v2">The second raw value, or null.</param>
        /// <returns><paramref name="condition"/> itself when neither value is a masked array with a mask; else a new array.</returns>
        private static NDArray WithBoundMasks(NDArray condition, object v1, object v2)
        {
            var masks = Or((v1 as NDMaskedArray)?._mask, (v2 as NDMaskedArray)?._mask);
            return masks is null ? condition : np.logical_or(condition, masks);
        }

        /// <summary>
        ///     The DType class NumPy names for a raw Python scalar operand in a no-loop comparison: <c>_PyLongDType</c> for an
        ///     int (any C# integer type or <see cref="BigInteger"/>), <c>_PyFloatDType</c> for a float (<see cref="double"/>,
        ///     <see cref="float"/>), <c>_PyComplexDType</c> for a <see cref="Complex"/>, and <c>BoolDType</c> for a bool (a
        ///     Python bool is not weak: it is NumPy's own bool).
        /// </summary>
        /// <param name="value">The raw value.</param>
        /// <returns>The DType class, or null when the value is not a Python number (text, bytes, arrays, NumPy scalars such
        ///     as <see cref="Half"/>).</returns>
        private static DTypeMeta PythonScalarClass(object value) => value switch
        {
            bool => DType.Boolean.Meta,
            sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => DTypeRegistry.PyLong,
            float or double => DTypeRegistry.PyFloat,
            Complex => DTypeRegistry.PyComplex,
            _ => null,
        };

        /// <summary>Whether a raw value is a weak Python int or float (NEP 50): it adopts an array operand's width.</summary>
        /// <param name="value">The raw value.</param>
        /// <returns>True for a C# integer type, <see cref="BigInteger"/>, <see cref="float"/> or <see cref="double"/>.</returns>
        private static bool IsWeakPythonNumber(object value)
            => value is sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger or float or double;

        /// <summary>
        ///     NumPy's UFuncTypeError for a comparison with no loop for the operand classes: <c>ufunc 'equal' did not contain
        ///     a loop with signature matching types (&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class
        ///     'numpy.dtypes._PyFloatDType'&gt;) -&gt; None</c>.
        /// </summary>
        /// <param name="ufunc">The comparison's name.</param>
        /// <param name="first">The first operand's class.</param>
        /// <param name="second">The second operand's class.</param>
        /// <returns>The exception (not thrown).</returns>
        private static TypeError ComparisonNoLoop(string ufunc, DTypeMeta first, DTypeMeta second)
            => new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types ({first}, {second}) -> None");

        /// <summary>
        ///     Python's <c>a &lt; b</c> between two bounds of <see cref="masked_inside"/>/<see cref="masked_outside"/>: numbers
        ///     by value (a NaN never orders), text by code point, bytes by byte value, arrays through NumPy's elementwise
        ///     <c>less</c> and the truth test of its result.
        /// </summary>
        /// <param name="a">The left bound.</param>
        /// <param name="b">The right bound.</param>
        /// <returns>True when Python's <c>a &lt; b</c> is truthy.</returns>
        /// <exception cref="TypeError">Python cannot order the two (<c>'&lt;' not supported between instances of 'str' and
        ///     'float'</c>, and always for a complex).</exception>
        /// <exception cref="ValueError">An array comparison with other than one element (Python's truth test).</exception>
        /// <exception cref="StringComputationNotSupportedException">Two text arrays: NumPy compares their text.</exception>
        private static bool PythonLess(object a, object b)
        {
            if (a is NDArray or NDMaskedArray || b is NDArray or NDMaskedArray)
            {
                using var less = np.less(AsValueOperand(a), AsValueOperand(b));
                if (less.size != 1)
                    throw new ValueError(less.size == 0
                        ? "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty."
                        : "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                return Convert.ToBoolean(less.GetAtIndex(0));
            }
            switch (a, b)
            {
                case (string sa, string sb):
                    return CompareCodePoints(sa, sb) < 0;
                case (byte[] ba, byte[] bb):
                    return ba.AsSpan().SequenceCompareTo(bb) < 0;
            }
            bool aReal = IsRealNumber(a), bReal = IsRealNumber(b);
            if (!aReal || !bReal)
                throw new TypeError($"'<' not supported between instances of '{PythonTypeName(a)}' and '{PythonTypeName(b)}'");
            // Python compares an int and a float by their exact values; two ints exactly.
            if (IsIntegerLike(a) && IsIntegerLike(b))
                return ToBigInteger(a) < ToBigInteger(b);
            return ToDouble(a) < ToDouble(b);

            static bool IsIntegerLike(object v) => v is bool or sbyte or byte or short or ushort or int or uint or long or ulong or char or BigInteger;
            static bool IsRealNumber(object v) => IsIntegerLike(v) || v is float or double or Half or decimal;
            static BigInteger ToBigInteger(object v) => v switch
            {
                BigInteger big => big,
                bool bo => bo ? BigInteger.One : BigInteger.Zero,
                char c => c,
                ulong u => u,
                _ => Convert.ToInt64(v, System.Globalization.CultureInfo.InvariantCulture),
            };
        }

        /// <summary>Python's ordering of two <c>str</c>: by Unicode code point (a C# ordinal comparison orders UTF-16 code
        /// units, which differs once a supplementary character meets a BMP character above the surrogate range).</summary>
        /// <param name="a">The left text.</param>
        /// <param name="b">The right text.</param>
        /// <returns>Negative, zero or positive, as <c>a</c> sorts before, equal to or after <c>b</c>.</returns>
        private static int CompareCodePoints(string a, string b)
        {
            int i = 0, j = 0;
            while (i < a.Length && j < b.Length)
            {
                int ca = CodePointAt(a, i, out int la), cb = CodePointAt(b, j, out int lb);
                if (ca != cb)
                    return ca < cb ? -1 : 1;
                i += la;
                j += lb;
            }
            return (a.Length - i).CompareTo(b.Length - j);

            // A surrogate pair is one code point; a lone surrogate is its own code unit (Python keeps it as a code point).
            static int CodePointAt(string s, int index, out int length)
            {
                if (char.IsHighSurrogate(s[index]) && index + 1 < s.Length && char.IsLowSurrogate(s[index + 1]))
                {
                    length = 2;
                    return char.ConvertToUtf32(s[index], s[index + 1]);
                }
                length = 1;
                return s[index];
            }
        }

        /// <summary>The Python type name of a raw C# value, as Python's comparison error quotes it.</summary>
        /// <param name="v">The value.</param>
        /// <returns><c>str</c>, <c>bytes</c>, <c>bool</c>, <c>int</c>, <c>float</c>, <c>complex</c>, a NumPy scalar's
        ///     qualified name, or the C# type name for anything else.</returns>
        private static string PythonTypeName(object v) => v switch
        {
            string => "str",
            byte[] => "bytes",
            bool => "bool",
            sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => "int",
            float or double => "float",
            Complex => "complex",
            Half => "numpy.float16",
            char => "numpy.uint16",
            decimal => "decimal.Decimal",
            null => "NoneType",
            _ => v.GetType().Name,
        };

        // ─────────────────────────────────────────────────────────────────────────────
        //  Extrema (element-wise), power, where, round
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Element-wise maximum of two operands (mask = OR): NumPy's <c>ma.maximum(a, b)</c>, the
        /// <c>_extrema_operation</c> call <c>where(greater(a, b), a, b)</c>.</summary>
        /// <param name="a">First operand.</param>
        /// <param name="b">Second operand (required, as in NumPy 2.x, whose <c>__call__(self, a, b)</c> has no default:
        ///     the reduction is <see cref="max(object,int?,object,bool)"/>).</param>
        /// <returns>The element-wise masked maximum.</returns>
        /// <exception cref="TypeError">The operands have no comparison loop (text against numbers): NumPy's
        ///     <c>ufunc 'greater' did not contain a loop ...</c>, raised by <c>greater(a, b)</c> in that operand order.</exception>
        /// <exception cref="StringComputationNotSupportedException">Two text operands: the comparison reads the text.</exception>
        public NDMaskedArray maximum(object a, object b)
        {
            // Scoped by hand: the masked comparison is only where's condition operand and is reclaimed.
            using var scope = NDScope.Open();
            return Yield(scope, where(greater(a, b), a, b));
        }

        /// <summary>Element-wise minimum of two operands (mask = OR): NumPy's <c>ma.minimum(a, b)</c>, the
        /// <c>_extrema_operation</c> call <c>where(less(a, b), a, b)</c>.</summary>
        /// <param name="a">First operand.</param>
        /// <param name="b">Second operand (required, as in NumPy 2.x; the reduction is
        ///     <see cref="min(object,int?,object,bool)"/>).</param>
        /// <returns>The element-wise masked minimum.</returns>
        /// <exception cref="TypeError">The operands have no comparison loop: NumPy's <c>ufunc 'less' did not contain a
        ///     loop ...</c>, naming the operands in call order.</exception>
        /// <exception cref="StringComputationNotSupportedException">Two text operands: the comparison reads the text.</exception>
        public NDMaskedArray minimum(object a, object b)
        {
            // Scoped by hand: the masked comparison is only where's condition operand and is reclaimed. `less(a, b)`
            // itself, not `greater(b, a)`: the two agree on every value, but a refusal names the operands in call order.
            using var scope = NDScope.Open();
            return Yield(scope, where(less(a, b), a, b));
        }

        /// <summary>Masked power (NumPy's <c>ma.power</c>, a plain function rather than a masked ufunc): masked positions
        /// take the base's data; a non-finite result is additionally masked and its data set to the RESULT's fill value —
        /// the base's own when the base is a masked array (<c>result._update_from(a)</c>), the dtype default otherwise.
        /// The <paramref name="third"/> modulus slot exists ONLY for signature parity — NumPy rejects a 3-argument masked
        /// power, and so does this.</summary>
        /// <param name="a">Base.</param><param name="b">Exponent.</param>
        /// <param name="third">Must be null — a non-null modulus raises <see cref="MaskError"/>, matching NumPy.</param>
        /// <returns>The masked power: it inherits the fill value and hard mask of <paramref name="a"/> ONLY (a plain base
        /// with a masked exponent gives the defaults); <see cref="NDMasked"/> for a 0-D result with a mask or a
        /// non-finite value.</returns>
        /// <exception cref="MaskError"><paramref name="third"/> is non-null (NumPy's
        /// "3-argument power not supported.").</exception>
        /// <remarks>
        ///     NumPy's statements: <c>m = mask_or(getmask(a), getmask(b))</c> — which SHRINKS an all-False mask to nomask —;
        ///     <c>result = np.where(m, fa, umath.power(fa, fb)).view(basetype)</c>; <c>result._update_from(a)</c>;
        ///     <c>invalid = ~isfinite(result)</c>; with a mask the result's mask is <c>m | invalid</c> (a 0-D result is
        ///     <c>masked</c>); without one it is <c>invalid</c> when any slot is invalid, else nomask; the invalid slots get
        ///     <c>result.fill_value</c>.
        ///     <para>Scoped by hand: the widening casts, the raw power, the base-restoring select, the non-finite predicate,
        ///     the fill scalar and the broadcast mask view are reclaimed; only the result's data and mask leave (the computed
        ///     value too when a 0-D result collapses to <see cref="NDMasked"/>).</para>
        /// </remarks>
        public NDMaskedArray power(object a, object b, object third = null)
        {
            // NumPy's ma.power raises on a modulus argument rather than computing pow(a, b, mod).
            if (third is not null)
                throw new MaskError("3-argument power not supported.");
            using var scope = NDScope.Open();
            // `m = mask_or(ma, mb)`: an all-False combined mask SHRINKS to nomask (so `invalid` alone decides the mask).
            var mo = mask_or(getmask(a), getmask(b));
            NDArray m = ReferenceEquals(mo, nomask) ? null : mo;
            var fa = AsData(a);
            var fb = AsData(b);
            // ma.power(int32, 2) → int64 (scalar-strong promotion, like the other binary ops).
            PromoteMaScalar(a, b, ref fa, ref fb);
            var pw = np.power(fa, fb);
            // At operand-masked positions NumPy keeps the base's data (where(m, fa, power)); np.where(False, fa, pw) is pw.
            var result = m is null ? pw : np.where(m, fa, pw);
            var invalid = NonFiniteMask(result); // NaN/±inf ⇒ masked + data set to fill
            bool anyInvalid = invalid is not null && np.any(invalid);
            NDArray finalMask = null;
            if (m is not null)
            {
                // `if not result.ndim: return masked` — any mask at all collapses a 0-D power.
                if (result.ndim == 0)
                    return Yield(scope, NDMasked);
                finalMask = invalid is null ? m : np.logical_or(m, invalid);
            }
            if (anyInvalid)
            {
                if (result.ndim == 0)
                    return Yield(scope, NDMasked);
                finalMask ??= invalid;
            }
            if (finalMask is not null && !finalMask.Shape.Equals(result.Shape))
                finalMask = np.broadcast_to(finalMask, result.Shape).copy();
            var r = new NDMaskedArray(result, finalMask);
            // `result._update_from(a)`: the BASE's attributes only — `basetype` and the update both read `a`.
            if (a is NDMaskedArray source)
                r.UpdateFrom(source);
            if (anyInvalid)
            {
                // `result._data[invalid] = result.fill_value`: the inherited fill (raw), else the result dtype's default.
                // np.where's result (or pw) is a fresh array this method owns, so it is written in place. A scalar fill
                // becomes a 0-d array the scope reclaims; an array fill is the stored object and is only read.
                var fv = r.fill_value;
                np.copyto(result, fv as NDArray ?? NDArray.Scalar(fv), casting: "unsafe", where: invalid);
            }
            return Yield(scope, r);
        }

        /// <summary>Masked <c>where</c> (NumPy's <c>ma.where</c>): picks <paramref name="x"/> where the condition
        /// is True, else <paramref name="y"/>; a masked or False condition selects <paramref name="y"/>, and a
        /// masked condition (or a masked chosen value) masks the result.</summary>
        /// <param name="condition">Boolean array-like selector.</param>
        /// <param name="x">Values chosen where the condition is True.</param>
        /// <param name="y">Values chosen where the condition is False/masked.</param>
        /// <returns>The masked selection.</returns>
        /// <remarks>Scoped by hand: the boolean condition copy, the widening casts, the three full masks and the
        /// intermediate mask selects are reclaimed; only the selected data and the final mask leave.</remarks>
        public NDMaskedArray where(object condition, object x, object y)
        {
            using var scope = NDScope.Open();
            var cond = condition as NDMaskedArray;
            var cf = (cond?._mask is null ? AsData(condition) : cond.filled(false)).astype(np.@bool);
            var xd = AsData(x);
            var yd = AsData(y);
            // Scalar-strong promotion (like the binary ufuncs): ma.where(cond, int32, 2) → int64, which also
            // widens ma.maximum/ma.minimum (they route through where) to match NumPy.
            PromoteMaScalar(x, y, ref xd, ref yd);
            var cm = getmaskarray(condition);
            var xm = getmaskarray(x);
            var ym = getmaskarray(y);
            var data = np.where(cf, xd, yd);
            var mask = np.where(cf, xm, ym);
            mask = np.where(cm, NDArray.Scalar(true), mask);
            // Shrink an all-False mask back to nomask (NumPy's _shrink_mask).
            return Yield(scope, new NDMaskedArray(data, np.any(mask) ? mask : null));
        }

        /// <summary>Rounds each element to <paramref name="decimals"/> places (half-to-even), preserving the mask
        /// (NumPy's <c>ma.round</c> → <c>MaskedArray.round</c>). Masked positions carry their rounded data but stay
        /// masked.</summary>
        /// <param name="a">Data.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array: it SHARES the operand's mask object and takes its fill value and hard-mask
        /// flag; a 0-d result is <see cref="NDMasked"/> when masked, else an unmasked 0-d array.</returns>
        public NDMaskedArray round(object a, int decimals = 0)
        {
            // Scoped by hand: a scalar/array-like operand's materialization only feeds the rounding.
            using var scope = NDScope.Open();
            var source = a as NDMaskedArray;
            var result = np.around(AsData(a), decimals);
            // MaskedArray.round: `if result.ndim > 0: result._mask = self._mask; result._update_from(self)` — the SAME mask
            // object, the fill value and hard mask — `elif self._mask: result = masked`.
            if (result.ndim == 0)
                return Yield(scope, source?._mask is not null && source._mask.GetBoolean() ? NDMasked : new NDMaskedArray(result, null));
            var r = new NDMaskedArray(result, source?._mask);
            if (source is not null)
                r.UpdateFrom(source);
            return Yield(scope, r);
        }

        /// <summary>Alias of <see cref="round"/> (NumPy's <c>around</c>).</summary>
        /// <param name="a">Data.</param><param name="decimals">Decimal places.</param>
        /// <returns>The masked, rounded array.</returns>
        public NDMaskedArray around(object a, int decimals = 0) => round(a, decimals);

        // ─────────────────────────────────────────────────────────────────────────────
        //  Shape / manipulation — the SAME transform is applied to the data AND the mask,
        //  so the mask always lines up with its data (NumPy's `_fromnxfunction` pattern).
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Applies a shape transform to both the data and (if present) the mask, keeping them aligned.</summary>
        /// <param name="a">Operand.</param><param name="f">The <c>np.*</c> transform to run on data and mask.</param>
        /// <param name="inherit">True for the transforms NumPy runs through <c>_arraymethod</c> or a MaskedArray method
        ///     (ravel, flatten, reshape, transpose, swapaxes, moveaxis, squeeze, expand_dims, repeat, diagonal): the result
        ///     takes the operand's fill value and hard-mask flag (<c>result._update_from(self)</c>, see
        ///     <see cref="NDMaskedArray.UpdateFrom"/>). False where NumPy views a plain result instead (take, a 1-D
        ///     diag), which starts soft with the default fill.</param>
        /// <returns>The transformed masked array.</returns>
        /// <remarks>Scoped by hand — the one scope every shape transform (ravel/reshape/transpose/take/diag/…) rides:
        /// a materialized scalar/array-like operand and anything the transform builds besides its result are
        /// reclaimed. When the transform hands back its OWN operand (a no-op squeeze) the result aliases the caller's
        /// arrays, exactly as NumPy's does.</remarks>
        private NDMaskedArray Map1(object a, Func<NDArray, NDArray> f, bool inherit = true)
        {
            using var scope = NDScope.Open();
            var m = (a as NDMaskedArray)?._mask;
            var result = new NDMaskedArray(f(AsData(a)), m is null ? null : f(m));
            if (inherit && a is NDMaskedArray source)
                result.UpdateFrom(source);
            return Yield(scope, result);
        }

        /// <summary>
        ///     numpy.ma.extras' <c>_fromnxfunction_single</c> / <c>_fromnxfunction_allargs</c> for one operand:
        ///     <c>masked_array(func(np.asarray(x)), mask=func(getmaskarray(x)))</c>. The mask is ALWAYS an array (an
        ///     unmasked operand gives an all-False one, never <c>nomask</c>), and the result is a NEW masked array over plain
        ///     results: it inherits neither the fill value nor the hard-mask flag.
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="f">The <c>np.*</c> function run on the data and on the full mask.</param>
        /// <returns>The masked result.</returns>
        /// <remarks>Scoped by hand: the full mask stand-in of an unmasked operand and a materialized array-like operand
        /// are reclaimed; only the results leave.</remarks>
        private NDMaskedArray MapFromNx(object a, Func<NDArray, NDArray> f)
        {
            using var scope = NDScope.Open();
            return Yield(scope, new NDMaskedArray(f(AsData(a)), f(getmaskarray(a))));
        }

        /// <summary>Applies a joining transform to a sequence's data arrays and to their full mask arrays — numpy.ma's
        /// two joining shapes: <c>ma.concatenate</c> (<paramref name="shrink"/>) keeps <c>nomask</c> when no operand has a
        /// mask and SHRINKS an all-False joined mask to <c>nomask</c> (<c>_shrink_mask(dm)</c>); the
        /// <c>_fromnxfunction_seq</c> joins (stack, vstack, hstack, dstack, column_stack) ALWAYS build the mask array
        /// (<c>masked_array(func(datas), mask=func(getmaskarrays))</c>). Neither inherits a fill value or hardness.</summary>
        /// <param name="arrays">The operands.</param><param name="f">The <c>np.*</c> join over the arrays.</param>
        /// <param name="shrink">True for concatenate's mask rule, false for the <c>_fromnxfunction_seq</c> joins.</param>
        /// <returns>The joined masked array.</returns>
        /// <remarks>Scoped by hand — the one scope every join (concatenate/stack/…) rides: materialized operands and
        /// the all-False stand-in masks of the unmasked operands only feed the joins and are reclaimed.</remarks>
        private NDMaskedArray MapSeq(object[] arrays, Func<NDArray[], NDArray> f, bool shrink = false)
        {
            using var scope = NDScope.Open();
            var rd = f(arrays.Select(AsData).ToArray());
            // concatenate: `for x in arrays: if getmask(x) is not nomask: break; else: return data`.
            if (shrink && !arrays.Any(x => (x as NDMaskedArray)?._mask is not null))
                return Yield(scope, new NDMaskedArray(rd, null));
            var dm = f(arrays.Select(getmaskarray).ToArray());
            // `data._mask = _shrink_mask(dm)` — an all-False joined mask (an empty one included) is nomask again.
            if (shrink && !np.any(dm))
                dm = null;
            return Yield(scope, new NDMaskedArray(rd, dm));
        }

        /// <summary>Flattened VIEW in the given order (mask flattened alike). NumPy's <c>ravel</c>.</summary>
        /// <param name="a">Operand.</param><param name="order">'C' or 'F'.</param><returns>The flattened masked array.</returns>
        public NDMaskedArray ravel(object a, char order = 'C') => Map1(a, d => np.ravel(d, order));

        /// <summary>Flattened COPY (NumPy's <c>flatten</c> — always copies, unlike <see cref="ravel"/>).</summary>
        /// <param name="a">Operand.</param><param name="order">'C' or 'F'.</param><returns>The flattened masked copy.</returns>
        public NDMaskedArray flatten(object a, char order = 'C') => Map1(a, d => np.ravel(d, order).copy());

        /// <summary>Reshape to <paramref name="new_shape"/> (mask reshaped alike). The parameter is spelled
        /// <c>new_shape</c> so a NumPy call <c>ma.reshape(a, new_shape=…)</c> ports verbatim.</summary>
        /// <param name="a">Operand.</param><param name="new_shape">New dimensions (one may be -1).</param><returns>The reshaped masked array.</returns>
        public NDMaskedArray reshape(object a, params int[] new_shape) => Map1(a, d => np.reshape(d, new_shape));

        /// <summary>Reshape to a <see cref="Shape"/> (mask reshaped alike).</summary>
        /// <param name="a">Operand.</param><param name="new_shape">New shape.</param><returns>The reshaped masked array.</returns>
        public NDMaskedArray reshape(object a, Shape new_shape) => Map1(a, d => np.reshape(d, new_shape));

        /// <summary>Permute axes (mask transposed alike); reverses all axes when <paramref name="axes"/> is null.</summary>
        /// <param name="a">Operand.</param><param name="axes">Permutation, or null to reverse.</param><returns>The transposed masked array.</returns>
        public NDMaskedArray transpose(object a, int[] axes = null) => Map1(a, d => np.transpose(d, axes));

        /// <summary>Swap two axes (mask swapped alike).</summary>
        /// <param name="a">Operand.</param><param name="axis1">First axis.</param><param name="axis2">Second axis.</param><returns>The masked array with swapped axes.</returns>
        public NDMaskedArray swapaxes(object a, int axis1, int axis2) => Map1(a, d => np.swapaxes(d, axis1, axis2));

        /// <summary>Move an axis (mask moved alike).</summary>
        /// <param name="a">Operand.</param><param name="source">Source axis.</param><param name="destination">Destination axis.</param><returns>The masked array.</returns>
        public NDMaskedArray moveaxis(object a, int source, int destination) => Map1(a, d => np.moveaxis(d, source, destination));

        /// <summary>Remove size-1 axes (mask squeezed alike). With <paramref name="axis"/> null EVERY size-1 axis
        /// is dropped; an explicit axis drops ONLY that one (and errors if it is not size 1, via
        /// <see cref="np.squeeze(NDArray,int)"/>) — NumPy's <c>squeeze(axis=…)</c>.</summary>
        /// <param name="a">Operand.</param>
        /// <param name="axis">The single size-1 axis to drop, or null to drop all size-1 axes.</param>
        /// <returns>The squeezed masked array.</returns>
        public NDMaskedArray squeeze(object a, int? axis = null)
            => Map1(a, d => axis is null ? np.squeeze(d) : np.squeeze(d, axis.Value));

        /// <summary>Insert a size-1 axis at <paramref name="axis"/> (mask expanded alike).</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis position.</param><returns>The masked array.</returns>
        public NDMaskedArray expand_dims(object a, int axis) => Map1(a, d => np.expand_dims(d, axis));

        /// <summary>Repeat elements (mask repeated alike, so repeated masked entries stay masked).</summary>
        /// <param name="a">Operand.</param><param name="repeats">Repeat count.</param><param name="axis">Axis or null (flatten).</param><returns>The masked array.</returns>
        public NDMaskedArray repeat(object a, int repeats, int? axis = null) => Map1(a, d => np.repeat(d, repeats, axis));

        /// <summary>Gather elements by index (mask gathered alike, so a taken element keeps its masked-ness).
        /// <paramref name="mode"/> selects the out-of-bounds policy ("raise"/"wrap"/"clip"), applied identically
        /// to the data and the mask gather so they stay aligned — NumPy's <c>ma.take(..., mode=…)</c>.</summary>
        /// <param name="a">Operand.</param><param name="indices">Integer index array.</param>
        /// <param name="axis">Axis or null.</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default)/"wrap"/"clip".</param>
        /// <returns>The masked array.</returns>
        public NDMaskedArray take(object a, NDArray indices, int? axis = null, string mode = "raise")
            // NumPy's MaskedArray.take views the plain gather (`_data.take(...)[...].view(cls)`): no _update_from.
            => Map1(a, d => np.take(d, indices, axis, null, mode), inherit: false);

        /// <summary>Diagonal: 1-D input CONSTRUCTS a matrix with the values (and mask) on the k-th diagonal;
        /// 2-D input EXTRACTS the k-th diagonal (and its mask). NumPy's <c>ma.diag</c>.</summary>
        /// <param name="v">Operand.</param><param name="k">Diagonal offset.</param><returns>The masked array.</returns>
        public NDMaskedArray diag(object v, int k = 0)
            // `np.diag(v, k).view(MaskedArray)`: a 1-D v builds a plain matrix (no inheritance), a 2-D v extracts through
            // MaskedArray.diagonal, an _arraymethod that carries the fill value and hardness along.
            => Map1(v, d => np.diag(d, k), inherit: v is NDMaskedArray mv && mv.ndim >= 2);

        /// <summary>Flatten the input and build a diagonal matrix from it (mask alike).</summary>
        /// <param name="v">Operand.</param><param name="k">Diagonal offset.</param><returns>The masked diagonal matrix.</returns>
        public NDMaskedArray diagflat(object v, int k = 0) => MapFromNx(v, d => np.diagflat(d, k));

        /// <summary>At-least-1-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 1.</returns>
        public NDMaskedArray atleast_1d(object a) => MapFromNx(a, d => np.atleast_1d(d));

        /// <summary>At-least-2-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 2.</returns>
        public NDMaskedArray atleast_2d(object a) => MapFromNx(a, d => np.atleast_2d(d));

        /// <summary>At-least-3-D view (mask alike).</summary>
        /// <param name="a">Operand.</param><returns>The masked array, rank ≥ 3.</returns>
        public NDMaskedArray atleast_3d(object a) => MapFromNx(a, d => np.atleast_3d(d));

        /// <summary>Join arrays along an existing axis (masks joined alike; all-unmasked ⇒ nomask).</summary>
        /// <param name="arrays">Operands.</param><param name="axis">Join axis.</param><returns>The joined masked array.</returns>
        public NDMaskedArray concatenate(object[] arrays, int axis = 0) => MapSeq(arrays, ds => np.concatenate(ds, axis), shrink: true);

        /// <summary>Stack arrays along a NEW axis (masks stacked alike).</summary>
        /// <param name="arrays">Operands.</param><param name="axis">New-axis position.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray stack(object[] arrays, int axis = 0) => MapSeq(arrays, ds => np.stack(ds, axis));

        /// <summary>Stack row-wise / along the first axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray vstack(params object[] tup) => MapSeq(tup, np.vstack);

        /// <summary>Stack column-wise / along the second axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray hstack(params object[] tup) => MapSeq(tup, np.hstack);

        /// <summary>Stack along the third axis (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray dstack(params object[] tup) => MapSeq(tup, np.dstack);

        /// <summary>Stack 1-D arrays as columns of a 2-D result (masks alike).</summary>
        /// <param name="tup">Operands.</param><returns>The stacked masked array.</returns>
        public NDMaskedArray column_stack(params object[] tup) => MapSeq(tup, np.column_stack);

        /// <summary>The 1-D array of the UNMASKED values in C-order (NumPy's <c>compressed</c>) — a plain
        /// <see cref="NDArray"/>, since a compressed result has, by definition, no mask.</summary>
        /// <param name="a">Operand.</param><returns>A 1-D <see cref="NDArray"/> of the unmasked data.</returns>
        /// <remarks>NumPy's statements: <c>data = ndarray.ravel(self._data)</c>, then
        /// <c>data.compress(~ravel(mask))</c> when there is a mask. So an UNMASKED operand's result is its raveled data — a
        /// VIEW of contiguous data (writes reach the operand), a copy otherwise — and a masked one's a fresh selection.
        /// Scoped (<c>[NDScoped]</c>): the raveled mask and the negated selector are reclaimed.</remarks>
        [NDScoped]
        public NDArray compressed(object a)
        {
            var flat = np.ravel(AsData(a));
            var m = (a as NDMaskedArray)?._mask;
            return m is null ? flat : flat[np.logical_not(np.ravel(m))];
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Element-wise selection / reduction over masked data (clip / choose / compress /
        //  diagonal / trace / nonzero / diff / append)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Clamps each element into <c>[a_min, a_max]</c>, preserving the mask (NumPy's <c>ma.clip</c>).
        /// A null bound disables that side of the clip.</summary>
        /// <param name="a">Data.</param><param name="a_min">Lower bound (array-like/scalar), or null.</param>
        /// <param name="a_max">Upper bound (array-like/scalar), or null.</param>
        /// <returns>The clamped masked array: a COPY of the operand's mask, its fill value and hard-mask flag.</returns>
        /// <remarks>NumPy's <c>ma.clip</c> calls <c>np.clip</c>, which runs <c>MaskedArray.clip</c> =
        /// <c>_arraymethod('clip', onmask=False)</c>: the clamp of the raw data (masked slots too), <c>_update_from(self)</c>,
        /// and <c>__setmask__(self._mask)</c>, which writes the mask's values into a mask of the result's own.</remarks>
        public NDMaskedArray clip(object a, object a_min, object a_max)
        {
            // Scoped by hand: the bounds (and a scalar operand) are materialized only to feed np.clip.
            using var scope = NDScope.Open();
            var source = a as NDMaskedArray;
            var m = source?._mask;
            // The base np.clip does the clamp on the raw data; masked data is hidden anyway, so clamping it too is
            // harmless and matches NumPy's ma.clip.
            var r = new NDMaskedArray(np.clip(AsData(a), a_min is null ? null : AsData(a_min), a_max is null ? null : AsData(a_max)), m?.copy());
            if (source is not null)
                r.UpdateFrom(source);
            return Yield(scope, r);
        }

        /// <summary>
        ///     Merges choice arrays by an index array (NumPy's <c>ma.choose</c>): output[i] = choices[indices[i]][i].
        ///     The result element is masked iff the CHOSEN element was masked (or the index itself was masked) —
        ///     the mask is chosen by the SAME index array, mirroring the data.
        /// </summary>
        /// <param name="indices">Integer index array (each in <c>[0, n-1]</c>); a masked index masks its output.</param>
        /// <param name="choices">The <c>n</c> choice arrays (<see cref="NDMaskedArray"/>/<see cref="NDArray"/>/array-like).</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default), "wrap", or "clip".</param>
        /// <returns>The merged masked array.</returns>
        /// <remarks>Scoped by hand: the zero-filled index copy, the materialized choices, the stand-in masks of
        /// unmasked choices and the pre-OR chosen mask are reclaimed; only the merged data and mask leave.</remarks>
        public NDMaskedArray choose(object indices, object[] choices, string mode = "raise")
        {
            using var scope = NDScope.Open();
            // Index data with masked slots filled by 0 (NumPy's filled(indices, 0)).
            var im = (indices as NDMaskedArray)?._mask;
            var c = im is null ? AsData(indices) : ((NDMaskedArray)indices).filled(0);
            // Pick the data by index; masked choice slots are hidden by the mask set below.
            var data = np.choose(c, choices.Select(AsData).ToArray(), mode: mode);
            bool anyMasked = im is not null || choices.Any(ch => (ch as NDMaskedArray)?._mask is not null);
            if (!anyMasked)
                return Yield(scope, new NDMaskedArray(data, null));
            // Pick each choice's full mask by the SAME index, then OR in the index's own mask.
            var outmask = np.choose(c, choices.Select(getmaskarray).ToArray(), mode: mode);
            outmask = Or(outmask, im);
            return Yield(scope, new NDMaskedArray(data, (outmask is not null && np.any(outmask)) ? outmask : null));
        }

        /// <summary>Keeps only the elements where <paramref name="condition"/> is truthy (NumPy's <c>ma.compress</c>);
        /// data AND mask are compressed alike so the survivors keep their masked-ness.</summary>
        /// <param name="condition">1-D array-like selector, read by truth value (non-zero numbers, non-empty strings) over
        ///     the flattened elements or along <paramref name="axis"/>.</param>
        /// <param name="a">Data.</param><param name="axis">Axis to compress along, or null to flatten first.</param>
        /// <returns>The compressed masked array; it keeps a masked <paramref name="a"/>'s fill value and hardness.</returns>
        /// <exception cref="ArgumentException">The condition is not 1-D (NumPy's ValueError <c>condition must be a 1-d
        ///     array</c>), checked before its values are read.</exception>
        /// <exception cref="IndexError">A truthy condition entry past the axis length.</exception>
        /// <exception cref="StringComputationNotSupportedException">A non-empty text condition (its strings' truth
        ///     values, the next phase of the string support).</exception>
        /// <remarks>NumPy's <c>MaskedArray.compress</c>: <c>condition = np.asarray(condition)</c>, then
        /// <c>_data.compress(condition, axis)</c> — whose 1-d check, nonzero and take run in NumPy's order inside
        /// <see cref="np.compress"/> — and the same compress of the mask, with <c>_update_from(self)</c>.</remarks>
        public NDMaskedArray compress(object condition, object a, int? axis = null)
        {
            // Scoped by hand: the materialized operands only feed the two compresses.
            using var scope = NDScope.Open();
            // np.asarray(condition): the condition is passed as it is, so np.compress checks its rank before it reads a value.
            var cond = AsData(condition);
            var source = a as NDMaskedArray;
            var m = source?._mask;
            var data = np.compress(cond, AsData(a), axis);
            var result = new NDMaskedArray(data, m is null ? null : np.compress(cond, m, axis));
            if (source is not null)
                result.UpdateFrom(source);
            return Yield(scope, result);
        }

        /// <summary>Extracts the <paramref name="offset"/>-th diagonal of a 2-D (or stacked) array (NumPy's
        /// <c>ma.diagonal</c>, the <c>_arraymethod('diagonal')</c>); the mask is extracted alike so each diagonal element
        /// keeps its masked-ness, and the result takes the operand's fill value and hard-mask flag.</summary>
        /// <param name="a">Data (rank ≥ 2).</param><param name="offset">Diagonal offset (0 = main).</param>
        /// <param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <returns>The masked diagonal: read-only VIEWS of the data and of the mask (np.diagonal's), as in NumPy.</returns>
        public NDMaskedArray diagonal(object a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            // Scoped by hand: an array-like operand's materialization is released (the diagonal VIEW keeps its
            // buffer alive through its own counted reference).
            using var scope = NDScope.Open();
            var source = a as NDMaskedArray;
            var m = source?._mask;
            var r = new NDMaskedArray(np.diagonal(AsData(a), offset, axis1, axis2), m is null ? null : np.diagonal(m, offset, axis1, axis2));
            // `result._update_from(self)`.
            if (source is not null)
                r.UpdateFrom(source);
            return Yield(scope, r);
        }

        /// <summary>
        ///     Sum of the diagonal, treating masked slots as 0 (NumPy's <c>ma.trace</c>, <c>MaskedArray.trace</c>). Returns a
        ///     PLAIN array, and — matching NumPy's <c>astype(None)</c> quirk — the result is float64 unless
        ///     <paramref name="dtype"/> is given (so <c>ma.trace</c> of an int matrix is float64, not int).
        /// </summary>
        /// <param name="a">Data (rank ≥ 2).</param><param name="offset">Diagonal offset.</param>
        /// <param name="axis1">First plane axis.</param><param name="axis2">Second plane axis.</param>
        /// <param name="dtype">Result dtype; null ⇒ float64 (NumPy's default).</param>
        /// <returns>A plain <see cref="NDArray"/> trace.</returns>
        /// <exception cref="ArgumentException">The operand has fewer than two dimensions (NumPy's ValueError <c>diag requires
        ///     an array of at least two dimensions</c>), or the axes are invalid.</exception>
        /// <exception cref="TypeError">A nomask text operand: the diagonal sum has no string loop.</exception>
        /// <exception cref="StringComputationNotSupportedException">A masked text operand: NumPy casts the diagonal to
        ///     float64 first (<c>D.astype(None)</c>), which parses the text.</exception>
        /// <remarks>
        ///     NumPy's statements: with no mask, <c>super().trace(offset, axis1, axis2).astype(dtype)</c>; with a mask,
        ///     <c>D = self.diagonal(offset, axis1, axis2); return D.astype(dtype).filled(0).sum(axis=-1)</c> — the cast comes
        ///     BEFORE the fill, so the diagonal is summed in the result dtype (float64 by default), and a rank error is the
        ///     diagonal's own. Scoped (<c>[NDScoped]</c>): the diagonal, its cast and the zero-filled copy are reclaimed.
        /// </remarks>
        [NDScoped]
        public NDArray trace(object a, int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null)
        {
            var dt = dtype ?? np.float64;
            // `if m is nomask: return super().trace(...).astype(dtype)` — astype(None) is float64.
            if ((a as NDMaskedArray)?._mask is null)
                return np.trace(AsData(a), offset, axis1, axis2).astype(dt);
            // `D = self.diagonal(...)` raises for a 1-D or 0-d operand before anything reads the data.
            var D = diagonal(a, offset, axis1, axis2);
            return np.sum(D.astype(dt).filled(0), -1);
        }

        /// <summary>Indices of the non-zero UNMASKED elements (NumPy's <c>ma.nonzero</c>): masked slots are filled
        /// with 0 so they never count as non-zero. One plain int64 index array per dimension.</summary>
        /// <param name="a">Data.</param>
        /// <returns>An array of int64 index arrays (the house tuple convention).</returns>
        /// <exception cref="ValueError">A 0-d operand (<c>Calling nonzero on 0d arrays is not allowed</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand with elements (its truth values are
        ///     read; a masked slot also receives the formatted <c>'0'</c>).</exception>
        /// <remarks>NumPy's <c>np.asarray(self.filled(0)).nonzero()</c>; over text the fill is formatted first, so a 0-d
        /// operand's rank refusal still wins (see <see cref="FilledForNumericOp"/>). Scoped (<c>[NDScoped]</c>): the
        /// zero-filled copy the indices are read from is reclaimed.</remarks>
        [NDScoped]
        public NDArray<long>[] nonzero(object a)
        {
            Exception deferral = null;
            var result = np.nonzero(FilledForNumericOp(a, 0, ref deferral));
            if (deferral is not null)
                throw deferral;
            return result;
        }

        /// <summary>
        ///     The n-th discrete difference along an axis, preserving the mask (NumPy's <c>ma.diff</c>): an output
        ///     element is masked iff ANY of the <c>n+1</c> input elements folded into it was masked.
        /// </summary>
        /// <param name="a">Data (rank ≥ 1).</param>
        /// <param name="n">Number of times to difference (0 returns the input; must be ≥ 0).</param>
        /// <param name="axis">Axis along which to difference (default last).</param>
        /// <param name="prepend">Values prepended along <paramref name="axis"/> before differencing, or null.</param>
        /// <param name="append">Values appended along <paramref name="axis"/> before differencing, or null.</param>
        /// <returns>The masked differences (axis length reduced by <paramref name="n"/>), with <paramref name="a"/>'s fill
        ///     value and hard mask when nothing is prepended or appended; <paramref name="a"/> ITSELF for <c>n == 0</c>.</returns>
        /// <exception cref="ArgumentException"><paramref name="n"/> is negative or <paramref name="a"/> is 0-D.</exception>
        /// <remarks>Scoped by hand: the prepend/append concatenation, the per-step dropped-edge takes and their
        /// index ranges, and every intermediate OR'd mask are reclaimed; only the differences and the final mask
        /// leave.</remarks>
        public NDMaskedArray diff(object a, int n = 1, int axis = -1, object prepend = null, object append = null)
        {
            // `if n == 0: return a` — the operand itself (a plain array is wrapped, without a copy, to fit the return type).
            if (n == 0) return a as NDMaskedArray ?? array(a);
            if (n < 0) throw new ArgumentException("order must be non-negative but got " + n);
            using var scope = NDScope.Open();
            var d = AsData(a);
            if (d.ndim == 0) throw new ArgumentException("diff requires input that is at least one dimensional");

            // Build the (optionally prepend/append-extended) data + full mask, then difference each independently:
            // the DATA via np.diff (which handles bool/unsigned exactly like NumPy), the MASK via an n-fold
            // adjacent-OR along the axis (a position is masked iff any element in its window was).
            var parts = new List<object>();
            if (prepend is not null) parts.Add(DiffEdge(prepend, d, axis));
            parts.Add(a);
            if (append is not null) parts.Add(DiffEdge(append, d, axis));
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
                mask = (a as NDMaskedArray)?._mask;
            }
            int ax = axis < 0 ? axis + data.ndim : axis;
            var resultData = np.diff(data, n, axis);
            // `np.diff(a, n, axis)` runs the plain subtract over views of the masked array, so __array_wrap__ gives the
            // result the operand's fill value and hard mask — the operand's own when nothing was prepended or appended,
            // the defaults of numpy.ma's concatenate otherwise.
            var source = parts.Count == 1 ? a as NDMaskedArray : null;
            NDMaskedArray r;
            if (mask is null)
                r = new NDMaskedArray(resultData, null);
            else
            {
                for (int k = 0; k < n; k++)
                    mask = np.logical_or(DropAlong(mask, ax, dropFirst: true), DropAlong(mask, ax, dropFirst: false));
                r = new NDMaskedArray(resultData, np.any(mask) ? mask : null);
            }
            if (source is not null)
                r.UpdateFrom(source);
            return Yield(scope, r);
        }

        /// <summary>
        ///     A <c>prepend=</c> / <c>append=</c> value of <see cref="diff"/> as NumPy's <c>ma.diff</c> passes it to the
        ///     concatenation: <c>np.ma.asanyarray(value)</c>, and a 0-d value <c>np.broadcast_to(value, shape)</c> with the
        ///     axis extent 1 — a PLAIN read-only array (broadcast_to's <c>subok=False</c> drops a masked 0-d value's mask).
        /// </summary>
        /// <param name="value">The prepend / append value.</param>
        /// <param name="d">The differenced operand's data.</param>
        /// <param name="axis">The difference axis (negative counts from the end).</param>
        /// <returns>The value to concatenate.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        private static object DiffEdge(object value, NDArray d, int axis)
        {
            var v = AsData(value);
            if (v.ndim != 0)
                return value;
            // `shape = list(a.shape); shape[axis] = 1` — d.shape is already a copy of the dimensions.
            var shape = d.shape;
            shape[DefaultEngine.NormalizeAxis(axis, d.ndim)] = 1;
            return np.broadcast_to(v, new Shape(shape));
        }

        /// <summary>Drops one element from the front (<paramref name="dropFirst"/>) or back of <paramref name="a"/>
        /// along <paramref name="axis"/> — the two operands of an adjacent-difference, built with <see cref="np.take"/>
        /// so no slice-string parsing is needed for an arbitrary axis.</summary>
        private static NDArray DropAlong(NDArray a, int axis, bool dropFirst)
        {
            int len = (int)a.Shape.dimensions[axis];
            var idx = dropFirst ? np.arange(1, len) : np.arange(0, len - 1);
            return np.take(a, idx, axis);
        }

        /// <summary>Appends <paramref name="b"/> to <paramref name="a"/> (NumPy's <c>ma.append</c>) = masked
        /// <see cref="concatenate"/>; with <paramref name="axis"/> null BOTH operands are flattened first.</summary>
        /// <param name="a">First operand.</param><param name="b">Value(s) appended.</param>
        /// <param name="axis">Join axis, or null to flatten both operands and append 1-D.</param>
        /// <returns>The appended masked array (a fresh allocation, never in place).</returns>
        public NDMaskedArray append(object a, object b, int? axis = null)
        {
            // Scoped by hand: the flattened operands of the axis=null form only feed the concatenation.
            using var scope = NDScope.Open();
            if (axis is null)
                return Yield(scope, concatenate(new object[] { ravel(a), ravel(b) }, 0));
            return Yield(scope, concatenate(new object[] { a, b }, axis.Value));
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  extras.py — masked-aware statistics, products, set/diff helpers, clump/edges
        // ─────────────────────────────────────────────────────────────────────────────


        /// <summary>Number of MASKED elements along the axis (NumPy's <c>count_masked</c>) — a plain int64 array.</summary>
        /// <param name="a">Operand.</param><param name="axis">Axis or null.</param>
        /// <returns>An int64 count of masked elements.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the all-False stand-in of an unmasked operand is reclaimed.</remarks>
        [NDScoped]
        public NDArray count_masked(object a, int? axis = null)
        {
            var m = (a as NDMaskedArray)?._mask ?? np.zeros(DimsOf(AsData(a)), np.@bool);
            return np.sum(m, axis, false, np.int64);
        }

        /// <summary>An empty masked array of the given shape with EVERY element masked (NumPy's <c>masked_all</c>) —
        /// the standard "fill me in" accumulator.</summary>
        /// <param name="shape">Result shape.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <returns>A fully-masked array.</returns>
        public NDMaskedArray masked_all(Shape shape, DType dtype = null)
            => new NDMaskedArray(np.empty(shape, dtype ?? np.float64), np.ones(shape, np.@bool));

        /// <summary>A fully-masked array shaped and typed like <paramref name="a"/> (NumPy's <c>masked_all_like</c>,
        /// <c>np.empty_like(arr).view(MaskedArray)</c> with an all-True mask): a masked prototype also lends its fill value
        /// and hard-mask flag.</summary>
        /// <param name="a">Prototype array-like.</param>
        /// <returns>A fully-masked array (uninitialized data; a <c>U</c> one is zero-filled, as NumPy's is).</returns>
        public NDMaskedArray masked_all_like(object a)
        {
            // Scoped by hand: a scalar/array-like prototype is materialized only to read its shape/dtype.
            using var scope = NDScope.Open();
            var d = AsData(a);
            var result = new NDMaskedArray(np.empty_like(d), np.ones(DimsOf(d), np.@bool));
            // np.empty_like of a masked array runs __array_finalize__ from it: the fill value and the hardness ride along.
            if (a is NDMaskedArray source)
                result.FinalizeFrom(source);
            return Yield(scope, result);
        }

        /// <summary>Weighted average over unmasked elements (NumPy's <c>average</c>): with no weights this is the
        /// mean; otherwise <c>sum(a·w)/sum(w)</c> with masked slots dropped from BOTH sums.</summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null.</param>
        /// <param name="weights">Weights (same shape as <paramref name="a"/>, or 1-D along <paramref name="axis"/>); null ⇒ uniform.</param>
        /// <param name="keepdims">Keep the reduced axes as size-1 (NumPy 2.0 <c>keepdims</c>).</param>
        /// <returns>The masked (weighted) average.</returns>
        public NDMaskedArray average(object a, int? axis = null, object weights = null, bool keepdims = false)
        {
            // Scoped by hand: the shared core also computes the sum-of-weights, which this form discards — it is
            // reclaimed with the core's other intermediates; only the average and its mask leave.
            using var scope = NDScope.Open();
            return Yield(scope, AverageCore(a, axis, weights, keepdims).avg);
        }

        /// <summary>
        ///     Weighted average AND its sum-of-weights (NumPy's <c>average(..., returned=True)</c>, which returns
        ///     the pair). The C# analog of NumPy's tuple return — <c>var (avg, sws) = np.ma.average_returned(a)</c>
        ///     — mirroring the base library's <see cref="np.average_returned(NDArray,int?,NDArray,bool)"/>. With no
        ///     weights the sum-of-weights is the COUNT of unmasked elements (as float64), matching NumPy.
        /// </summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null.</param>
        /// <param name="weights">Weights, or null for uniform.</param><param name="keepdims">Keep reduced axes.</param>
        /// <returns>A tuple of the masked average and the (plain float64) sum of weights.</returns>
        public (NDMaskedArray avg, NDArray sumOfWeights) average_returned(object a, int? axis = null, object weights = null, bool keepdims = false)
        {
            // Scoped by hand (a tuple carrying a masked array is not a shape the weaver yields): both halves of the
            // pair are handed back — the average's data/mask and the sum-of-weights — and everything else the core
            // built is reclaimed.
            using var scope = NDScope.Open();
            var result = AverageCore(a, axis, weights, keepdims);
            Yield(scope, result.avg);
            scope.Returns(result.sumOfWeights);
            return result;
        }

        /// <summary>Shared core of <see cref="average"/>/<see cref="average_returned"/>: computes the masked
        /// average and the sum-of-weights so the two public entry points never drift apart.</summary>
        /// <param name="a">Data.</param>
        /// <param name="axis">Axis or null.</param>
        /// <param name="weights">Weights, or null for the uniform (mean) form.</param>
        /// <param name="keepdims">Keep the reduced axis.</param>
        /// <returns>The masked average and the sum of weights (broadcast to the average's shape, NumPy's
        ///     <c>returned=True</c> form).</returns>
        /// <exception cref="AxisError">The axis is out of bounds (<c>axis: axis 5 is out of bounds for array of dimension
        ///     1</c>, NumPy's <c>normalize_axis_tuple(..., argname="axis")</c>).</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>); text weights against a masked <paramref name="a"/> (<c>multiply</c> has no text ×
        ///     bool loop); or a text operand, whose result dtype is a sized string the sum's <c>dtype=</c> refuses (NumPy's
        ///     general-DType TypeError).</exception>
        /// <exception cref="ValueError">Weights that are not 1-D of the axis's length (<c>Shape of weights must be consistent
        ///     with shape of a along specified axis.</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements for weights: <c>result_dtype = np.result_type(a.dtype, wgt.dtype[, 'f8'])</c> (an integer or
        ///     bool <paramref name="a"/> averages in at least float64; float32 stays float32, complex stays complex), the two
        ///     shape checks, <c>wgt = wgt * (~a.mask); wgt.mask |= a.mask</c> when <paramref name="a"/> has a mask,
        ///     <c>scl = wgt.sum(axis, dtype=result_dtype)</c> and <c>avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis)
        ///     / scl</c>. The division is numpy.ma's (a zero sum of weights is masked) except for a full reduction, whose sums
        ///     are NumPy scalars: there <c>0/0</c> is a plain nan.
        ///     </para>
        ///     <para>Unscoped on purpose: both callers open the scope and decide which halves of the pair leave, so
        ///     the core's intermediates (the casts, broadcast weights, weighted products) ride the caller's scope.</para>
        /// </remarks>
        private (NDMaskedArray avg, NDArray sumOfWeights) AverageCore(object a, int? axis, object weights, bool keepdims)
        {
            // a = asarray(a); m = getmask(a)
            var am = asarray(a);
            var mask = am._mask;
            // axis = normalize_axis_tuple(axis, a.ndim, argname="axis"), before either form computes anything.
            int? ax = null;
            if (axis is { } requested)
            {
                int nd = am.ndim;
                if (requested < -nd || requested >= nd)
                    throw new AxisError($"axis: axis {requested} is out of bounds for array of dimension {nd}");
                ax = requested < 0 ? requested + nd : requested;
            }
            if (weights is null)
            {
                // Uniform: the average is the plain masked mean, and the sum-of-weights is the unmasked COUNT in the
                // average's dtype (`avg.dtype.type(a.count(axis))`).
                var avgU = mean(am, ax, null, keepdims);
                var scl = CountUnmasked(mask, am.Shape, ax, keepdims).astype(avgU.dtype);
                return (avgU, scl);
            }
            var wgt = asarray(weights);
            // An integer or bool average is at least float64; anything else averages in the operands' common dtype.
            DType resultDtype = am.dtype.kind is 'i' or 'u' or 'b'
                ? np.result_type(am.dtype, wgt.dtype, np.float64)
                : np.result_type(am.dtype, wgt.dtype);
            if (!wgt.Shape.dimensions.AsSpan().SequenceEqual(am.Shape.dimensions))
            {
                if (ax is null)
                    throw new TypeError("Axis must be specified when shapes of a and weights differ.");
                if (wgt.ndim != 1 || wgt.Shape.dimensions[0] != am.Shape.dimensions[ax.Value])
                    throw new ValueError("Shape of weights must be consistent with shape of a along specified axis.");
                // wgt.transpose(np.argsort(axis)) is the identity for one axis; the reshape lays the weights along it.
                var shp = Enumerable.Repeat(1L, am.ndim).ToArray();
                shp[ax.Value] = am.Shape.dimensions[ax.Value];
                wgt = new NDMaskedArray(np.reshape(wgt._data, Shape.Adopt(shp)),
                    wgt._mask is null ? null : np.reshape(wgt._mask, Shape.Adopt((long[])shp.Clone())));
            }
            if (mask is not null)
            {
                // wgt = wgt * (~a.mask): numpy.ma's multiply (text weights have no text × bool loop); wgt.mask |= a.mask.
                var weighted = multiply(wgt, np.logical_not(mask));
                // A 0-d product is a NumPy SCALAR in numpy.ma (`_MaskedBinaryOperation` returns the bare result), and
                // `wgt.mask |= a.mask` then fails on it: NumPy's own AttributeError, naming the scalar's type.
                if (!ReferenceEquals(weighted, NDMasked) && weighted._data.ndim == 0)
                    throw new AttributeError($"'numpy.{weighted.dtype.name}' object has no attribute 'mask'");
                weighted._mask = Or(weighted._mask, mask);
                wgt = weighted;
            }
            // scl = wgt.sum(axis=axis, dtype=result_dtype): the ufunc's dtype check refuses a sized string dtype (a text
            // operand). NumPy fills the masked slots with 0 first, which cannot fail, so the check is its answer.
            DefaultEngine.UfuncLoopCode(resultDtype);
            var sclMasked = sum(wgt, ax, resultDtype, keepdims);
            // avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis) / scl; the raw ufunc's result is masked where
            // either operand is.
            var product = np.multiply(am._data, wgt._data, dtype: resultDtype);
            var productMask = Or(mask, wgt._mask);
            if (productMask is not null && !productMask.Shape.dimensions.AsSpan().SequenceEqual(product.Shape.dimensions))
                productMask = np.broadcast_to(productMask, product.Shape).copy();
            var num = sum(new NDMaskedArray(product, productMask), ax, null, keepdims);
            NDMaskedArray avg;
            if (ax is null && !keepdims)
                // A full reduction's sums are NumPy scalars (or `masked`): plain scalar division, so 0/0 is nan, unmasked.
                avg = ReferenceEquals(num, NDMasked) || ReferenceEquals(sclMasked, NDMasked)
                    ? NDMasked
                    : new NDMaskedArray(np.true_divide(num._data, sclMasked._data), null);
            else
                avg = true_divide(num, sclMasked);
            // `if scl.shape != avg.shape: scl = np.broadcast_to(scl, avg.shape).copy()`. An all-masked sum of weights is
            // NumPy's `masked`; the plain-array result reports its zero instead (the singleton's data is shared).
            var sumOfWeights = ReferenceEquals(sclMasked, NDMasked) ? NDArray.Scalar(0).astype(resultDtype) : sclMasked._data;
            if (!sumOfWeights.Shape.dimensions.AsSpan().SequenceEqual(avg._data.Shape.dimensions))
                sumOfWeights = np.broadcast_to(sumOfWeights, avg._data.Shape).copy();
            return (avg, sumOfWeights);
        }

        /// <summary>Median over unmasked elements (NumPy's <c>median</c>). Supports the flat case, the
        /// unmasked-axis case, AND a masked array with an explicit axis (the per-slice masked median — the axis is
        /// masked-sorted, then the low/high middle of each slice's unmasked count is averaged).</summary>
        /// <param name="a">Data.</param><param name="axis">Axis or null (flatten).</param>
        /// <param name="keepdims">Keep the reduced axes as size-1 (NumPy 2.0 <c>keepdims</c>).</param>
        /// <returns>The masked median (a slice with no unmasked element is <see cref="NDMasked"/>; an unmasked NaN in
        /// a slice makes that slice's median NaN, matching NumPy).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="TypeError">A text operand: NumPy's <c>_median</c> averages the middles with <c>mean</c>, whose
        ///     <c>add.reduce</c> has no string loop (<c>the resolved dtypes are not compatible with add.reduce</c>).</exception>
        /// <remarks>Scoped by hand: the compressed copy, the masked sort, the middle-index arithmetic, the gathered
        /// middles and the NaN probe (see <see cref="MedianAxisMasked"/>) are all reclaimed here; only the median
        /// and its mask leave.</remarks>
        public NDMaskedArray median(object a, int? axis = null, bool keepdims = false)
        {
            using var scope = NDScope.Open();
            var mask = (a as NDMaskedArray)?._mask;
            var d = AsData(a);
            if (mask is null)
                return Yield(scope, new NDMaskedArray(np.median(d, axis, null, false, keepdims), null));
            // TEXT with a mask: NumPy's _median sorts the strings (a computation that cannot fail; its axis check comes
            // first) and then takes ma.mean of the middle elements — of an empty slice too — whose add.reduce refuses the
            // string dtype. So every text median is that refusal, whatever the strings and the mask: np.mean raises it
            // with NumPy's text for the operand's dtype.
            if (d.typecode == NPTypeCode.String)
            {
                if (axis is { } textAxis && (textAxis < -d.ndim || textAxis >= d.ndim))
                    throw new AxisError(textAxis, d.ndim);
                np.mean(d);
                // np.mean never returns for a string dtype; this keeps the method total if that ever changes.
                throw NDStringUfuncs.NextPhase("median", d.dtype);
            }
            if (axis is null)
            {
                var comp = compressed(a);
                if (comp.size == 0)
                    return NDMasked;
                var med = np.median(comp);
                // keepdims on a flattened reduction ⇒ an all-ones shape of the input's rank (NumPy's (1,…,1)).
                if (keepdims && d.ndim > 0)
                    med = np.reshape(med, Shape.Adopt(Enumerable.Repeat(1L, d.ndim).ToArray()));
                return Yield(scope, new NDMaskedArray(med, null));
            }
            return Yield(scope, MedianAxisMasked((NDMaskedArray)a, axis.Value, keepdims));
        }

        /// <summary>
        ///     Per-slice median of a MASKED array along an explicit axis — a port of NumPy's
        ///     <c>numpy.ma.extras._median</c> (the axis branch). Masked entries are sorted BEHIND the valid ones
        ///     (so they never win the middle), the low/high middle indices are derived from each slice's UNMASKED
        ///     count, and the two middles are averaged; an all-masked slice stays masked, and an unmasked NaN
        ///     (which sorts even past the fill value) forces that slice's median to NaN.
        /// </summary>
        /// <param name="self">The masked operand (guaranteed to carry a mask by the caller).</param>
        /// <param name="axis">The reduction axis (may be negative).</param>
        /// <param name="keepdims">Re-insert the reduced axis as size 1.</param>
        /// <returns>The per-slice masked median.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the operand's rank.</exception>
        private NDMaskedArray MedianAxisMasked(NDMaskedArray self, int axis, bool keepdims)
        {
            NDArray data = self._data;
            int nd = data.ndim;
            int ax = axis < 0 ? axis + nd : axis;
            if (ax < 0 || ax >= nd)
                throw new AxisError(axis, nd); // report the ORIGINAL axis, NumPy-style

            // ANY empty dimension ⇒ every slice is empty. NumPy's _median takes the mean of a 0-length
            // slice (→ NaN where the reduced axis has 0 valid elements, or an empty result when a DIFFERENT
            // axis is 0). Short-circuit to mean BEFORE the sort / take_along_axis / middle-averaging below:
            // those index into 0-row data and write OUT OF BOUNDS on an empty operand (a heap-corrupting
            // crash on empty complex128, found by the ma_reduce edge-layout corpus — the (0,3)/axis=1 case
            // that slips past the reduced-axis-only guard further down). mean is the correct empty-slice
            // value AND is already the empty-reduced-axis handler below.
            if (data.size == 0)
                return mean(self, ax, null, keepdims);

            // Masked entries sort to the END (endwith default → minimum_fill_value key = +inf for float, dtype-max
            // for int), reproducing NumPy's fill_value=inf/None so a masked slot can never be picked as a middle.
            NDMaskedArray asorted = sort(self, ax);
            NDArray sdata = asorted._data;
            NDArray smask = asorted._mask;

            // Empty reduced axis: the median of an empty slice is NaN (NumPy takes the mean of the 0-length slice).
            if (sdata.Shape.dimensions[ax] == 0)
                return mean(asorted, ax, null, keepdims);

            bool inexact = IsInexact(data.typecode);

            // Middle indices from each slice's UNMASKED count (keepdims so they broadcast along `ax`). Integer
            // floor-division/modulo are spelled explicitly — `/` on an int NDArray is TRUE division here.
            NDArray counts = CountUnmasked(smask, sdata.Shape, ax, true);
            NDArray two = NDArray.Scalar(2L);
            NDArray h = np.floor_divide(counts, two);                          // counts // 2
            NDArray odd = np.equal(np.remainder(counts, two), NDArray.Scalar(1L));
            NDArray l = np.where(odd, h, np.subtract(h, NDArray.Scalar(1L)));  // odd → h, even → h-1
            NDArray lh = np.concatenate(new[] { l, h }, ax);                   // the two middles, size 2 along `ax`

            NDArray lhData = np.take_along_axis(sdata, lh, ax);
            NDArray lhMask = smask is null ? null : np.take_along_axis(smask, lh, ax);

            // replace_masked: a gathered middle can still be masked when a real value sorted past the fill value
            // (e.g. [4, --, inf]). Where that happens AND the slice is not all-masked, substitute the fill value
            // and unmask so a valid median is produced; an all-masked slice is left masked.
            if (lhMask is not null && np.any(lhMask))
            {
                NDArray notAll = np.logical_not(np.all(smask, ax, null, true));
                NDArray rep = np.logical_and(np.broadcast_to(notAll, lhMask.Shape), lhMask);
                if (np.any(rep))
                {
                    np.copyto(lhData, NDArray.Scalar(minimum_fill_value(asorted)), casting: "unsafe", where: rep);
                    np.copyto(lhMask, NDArray.Scalar(false), casting: "unsafe", where: rep);
                }
            }

            var lowHigh = new NDMaskedArray(lhData, lhMask);

            // Average the two middles across `ax`. Inexact: sum then /2 (avoids the masked inf/x pitfall) plus the
            // NaN-propagation check; integer/bool: a straight mean (which widens to float64 like NumPy).
            NDMaskedArray s;
            if (inexact)
            {
                s = sum(lowHigh, ax);
                s = new NDMaskedArray(np.true_divide(s._data, NDArray.Scalar(2.0)), s._mask);
                s = MedianNanCheck(sdata, s, ax);
            }
            else
            {
                s = mean(lowHigh, ax);
            }

            // A 1-D input reduces to a 0-D result, and NumSharp treats a 0-D operand as WEAK — so the
            // `true_divide(_, 2.0)` above widens a narrow float (float16/float32) to float64, while a 2-D input
            // (whose result is a 1-D STRONG array) keeps its dtype. NumPy's median PRESERVES the input float
            // dtype in every path; dividing a float by 2 is exact (exponent decrement, no rounding), so this cast
            // back is byte-exact, and it is a no-op whenever the dtype already matches (integers stay float64 via
            // the `mean` branch, guarded by `inexact`).
            if (inexact && s._data.typecode != data.typecode)
                s = new NDMaskedArray(s._data.astype(data.dtype), s._mask);

            if (keepdims)
                s = new NDMaskedArray(np.expand_dims(s._data, ax), s._mask is null ? null : np.expand_dims(s._mask, ax));
            return s;
        }

        /// <summary>
        ///     Forces a slice's median to NaN when the SORTED slice ends in an unmasked NaN (NumPy's
        ///     <c>_median_nancheck</c>): masked entries were sorted behind the fill value, but a real NaN sorts
        ///     even further, so the last element along <paramref name="axis"/> being NaN means an unmasked NaN was
        ///     present and the median is undefined.
        /// </summary>
        /// <param name="sortedData">The masked-sorted data (masked entries and NaNs pushed to the tail).</param>
        /// <param name="result">The computed per-slice median.</param>
        /// <param name="axis">The reduced axis of <paramref name="sortedData"/>.</param>
        /// <returns><paramref name="result"/> with NaN copied into the slices whose sorted tail is NaN.</returns>
        private NDMaskedArray MedianNanCheck(NDArray sortedData, NDMaskedArray result, int axis)
        {
            if (sortedData.size == 0)
                return result;
            NDArray potentialNans = np.take(sortedData, -1L, axis); // last element along axis → axis removed
            NDArray n = np.isnan(potentialNans);
            if (!np.any(n))
                return result;
            NDArray rd = result._data.copy();
            np.copyto(rd, potentialNans, casting: "unsafe", where: n);
            return new NDMaskedArray(rd, result._mask);
        }

        /// <summary>True for the inexact (float/complex) dtypes, where median averages in floating point and the
        /// NaN-propagation check applies; false for the exact integer/bool dtypes.</summary>
        /// <param name="tc">The element type code.</param>
        /// <returns>Whether <paramref name="tc"/> is Half/Single/Double/Complex.</returns>
        private static bool IsInexact(NPTypeCode tc)
            => tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary>Differences between consecutive UNMASKED-aware elements of the flattened input (NumPy's
        /// <c>ediff1d</c>), optionally bracketed by <paramref name="to_begin"/>/<paramref name="to_end"/>.</summary>
        /// <param name="arr">Data.</param>
        /// <param name="to_end">Values appended to the end (array-like), or null.</param>
        /// <param name="to_begin">Values prepended (array-like), or null.</param>
        /// <returns>The masked consecutive differences.</returns>
        public NDMaskedArray ediff1d(object arr, object to_end = null, object to_begin = null)
        {
            // Scoped by hand: the flattened operand, the two shifted views, and (when bracketing) the raveled
            // to_begin/to_end and the unbracketed differences are reclaimed; only the final result leaves.
            using var scope = NDScope.Open();
            var flat = ravel(arr);
            var ed = subtract(Map1(flat, x => x["1:"]), Map1(flat, x => x[":-1"]));
            if (to_begin is null && to_end is null)
                return Yield(scope, ed);
            var parts = new List<object>();
            if (to_begin is not null) parts.Add(ravel(to_begin));
            parts.Add(ed);
            if (to_end is not null) parts.Add(ravel(to_end));
            return Yield(scope, concatenate(parts.ToArray()));
        }

        /// <summary>True iff every corresponding pair of elements is equal, with masked positions treated as
        /// equal when <paramref name="fill_value"/> is true (NumPy's <c>allequal</c>).</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <param name="fill_value">Treat masked positions as equal (true) or unequal (false).</param>
        /// <returns>A single bool.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the element-wise equality and mask combinations are reclaimed.</remarks>
        [NDScoped]
        public bool allequal(object a, object b, bool fill_value = true)
        {
            NDArray eq = np.equal(AsData(a), AsData(b));
            var mu = Or((a as NDMaskedArray)?._mask, (b as NDMaskedArray)?._mask);
            if (mu is not null)
                eq = fill_value ? (eq | mu) : (eq & np.logical_not(mu));
            return np.all(eq);
        }

        /// <summary>True iff all corresponding elements are within tolerance, with masked positions treated as
        /// equal when <paramref name="masked_equal"/> is true (NumPy's <c>ma.allclose</c>, statement for statement):
        /// <c>|x - y| &lt;= atol + rtol * |y|</c> over the masked arithmetic, and infinities must sit at the same places
        /// with the same signs.</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param>
        /// <param name="masked_equal">Treat masked positions as equal.</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param>
        /// <returns>A single bool.</returns>
        /// <exception cref="DTypePromotionError"><paramref name="b"/> is text: <c>np.result_type(y, 1.)</c> finds no common
        ///     DType with a Python float — raised before any element is read.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is text: <c>np.isinf</c> has no string loop.</exception>
        /// <exception cref="MaskError">The combined mask does not fit <paramref name="a"/> (a 0-d <paramref name="a"/>
        ///     against a masked array).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's ValueError text).</exception>
        /// <remarks>
        ///     NumPy's statements: <c>x = masked_array(a); y = masked_array(b)</c>; y is cast to
        ///     <c>np.result_type(y, 1.)</c> (a bool or integer y becomes float64, so <c>abs(MIN_INT)</c> cannot wrap);
        ///     <c>m = mask_or(getmask(x), getmask(y))</c>; <c>xinf = np.isinf(masked_array(x, mask=m)).filled(False)</c>;
        ///     unequal infinity positions are False; with no infinity the closeness is
        ///     <c>filled(less_equal(absolute(x - y), atol + rtol * absolute(y)), masked_equal)</c>; otherwise the infinities
        ///     must be equal (<c>x[xinf] == y[xinf]</c>) and the rest close. A NaN is never close unless masked.
        ///     <para>Scoped (<c>[NDScoped]</c>): every intermediate is reclaimed; only the verdict leaves.</para>
        /// </remarks>
        [NDScoped]
        public bool allclose(object a, object b, bool masked_equal = true, double rtol = 1e-5, double atol = 1e-8)
        {
            var x = masked_array(a);
            var y = masked_array(b);
            // `if y.dtype.kind != "m": dtype = np.result_type(y, 1.)` — NumSharp has no timedelta, so the guard always
            // holds. The Python float is WEAK (NEP 50), hence the dtype form of result_type.
            var ydt = np.result_type(y.dtype, 1.0);
            if (!y.dtype.Equals(ydt))
                y = masked_array(y, dtype: ydt);
            var m = mask_or(getmask(x), getmask(y));
            // `xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)`: the mask must fit x (MaskError), isinf
            // has no string loop (TypeError), and a masked slot is not an infinity.
            var xm = masked_array(x, mask: ReferenceEquals(m, nomask) ? null : m);
            var xinf = IsInfFilledFalse(xm);
            // `if not np.all(xinf == filled(np.isinf(y), False)): return False`.
            if (!np.all(np.equal(xinf, IsInfFilledFalse(y))))
                return false;
            // `if not np.any(xinf)`: no infinity anywhere, one closeness test over everything.
            if (!np.any(xinf))
                return np.all(CloseFilled(x, y, rtol, atol, masked_equal));
            // `if not np.all(filled(x[xinf] == y[xinf], masked_equal)): return False` — equal infinities (sign included).
            var xi = (NDMaskedArray)x[xinf];
            var yi = (NDMaskedArray)y[xinf];
            if (!np.all(filled(equal(xi, yi), masked_equal)))
                return false;
            // `x = x[~xinf]; y = y[~xinf]` and the closeness test over the finite rest.
            var finite = np.logical_not(xinf);
            return np.all(CloseFilled((NDMaskedArray)x[finite], (NDMaskedArray)y[finite], rtol, atol, masked_equal));
        }

        /// <summary><c>np.isinf(x).filled(False)</c> for a masked operand: true at the unmasked infinite elements.</summary>
        /// <param name="x">The masked operand.</param>
        /// <returns>A boolean array of <paramref name="x"/>'s shape.</returns>
        /// <exception cref="TypeError"><paramref name="x"/> is text (isinf has no string loop).</exception>
        private static NDArray IsInfFilledFalse(NDMaskedArray x)
        {
            var inf = np.isinf(x._data);
            return x._mask is null ? inf : np.logical_and(inf, np.logical_not(x._mask));
        }

        /// <summary>
        ///     <see cref="allclose"/>'s closeness test, <c>filled(less_equal(absolute(x - y), atol + rtol * absolute(y)),
        ///     masked_equal)</c>, in the masked arithmetic (the tolerances are STRONG float64 0-d operands there, as
        ///     numpy.ma's <c>getdata(rtol)</c> makes them).
        /// </summary>
        /// <param name="x">First operand.</param><param name="y">Second operand.</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param>
        /// <param name="masked_equal">The verdict at a masked position.</param>
        /// <returns>The boolean closeness array, masked positions filled with <paramref name="masked_equal"/>.</returns>
        private NDArray CloseFilled(NDMaskedArray x, NDMaskedArray y, double rtol, double atol, bool masked_equal)
            => filled(less_equal(absolute(subtract(x, y)), add(atol, multiply(rtol, absolute(y)))), masked_equal);

        /// <summary>Dot product with masked slots treated as 0; a result element is masked only if NO valid
        /// (both-unmasked) term contributed to it (NumPy's <c>ma.dot</c>). With <paramref name="strict"/> the
        /// mask is PROPAGATED first — any masked value in a contracted row/column masks that whole vector — so a
        /// result element is masked whenever a masked value touched it (NumPy's <c>strict=True</c>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <param name="strict">Propagate masks along the contracted axes before the product (default false =
        /// treat masked as 0).</param>
        /// <returns>The masked dot product, soft with the default fill value: an array result always carries a mask ARRAY
        ///     (all-False when every slot has a valid pair); a 0-d result is unmasked or carries a 0-d True mask.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a masked slot whose product would
        ///     compute (NumPy formats the 0 fill into it; see <see cref="FilledForNumericOp"/>).</exception>
        /// <remarks>Scoped by hand: the propagated operands, the zero-filled copies, the validity indicators and
        /// their product are reclaimed; only the product and its mask leave. Over text NumPy's product refuses by dtype
        /// (<c>data type must provide an itemsize</c>, or a scalar operand's multiply loop) after formatting the fill, and
        /// that refusal is raised over the unfilled data.</remarks>
        public NDMaskedArray dot(object a, object b, bool strict = false)
        {
            using var scope = NDScope.Open();
            if (strict)
            {
                // Propagate the mask along the axes being contracted, mirroring NumPy's ma.dot: for a matrix @
                // matrix that is a's last axis and b's second-to-last; for a matrix @ vector, b's last axis;
                // scalars have no axis to propagate along.
                int nda = AsData(a).ndim, ndb = AsData(b).ndim;
                if (nda != 0 && ndb != 0)
                {
                    a = MaskPropagate(a, nda - 1);
                    b = MaskPropagate(b, ndb == 1 ? ndb - 1 : ndb - 2);
                }
            }
            // `d = np.dot(filled(a, 0), filled(b, 0))`.
            Exception deferral = null;
            var product = np.dot(FilledForNumericOp(a, 0, ref deferral), FilledForNumericOp(b, 0, ref deferral));
            if (deferral is not null)
                throw deferral;
            var ma_ = (a as NDMaskedArray)?._mask;
            var mb_ = (b as NDMaskedArray)?._mask;
            // `m = ~np.dot(~getmaskarray(a), ~getmaskarray(b))` — computed even when neither operand has a mask: a slot
            // with NO valid pair is masked, which an empty contraction (shape (2,0) @ (0,3)) makes of every slot.
            var va = (ma_ is null ? np.ones(DimsOf(AsData(a)), np.@bool) : np.logical_not(ma_)).astype(np.float64);
            var vb = (mb_ is null ? np.ones(DimsOf(AsData(b)), np.@bool) : np.logical_not(mb_)).astype(np.float64);
            var valid = np.dot(va, vb); // count of valid contributing pairs
            var m = np.equal(valid, NDArray.Scalar(0.0));
            // `r.__setmask__(m)` on the fresh result: an array result gets the FULL mask (all-False included, never
            // shrunk); a 0-d one stays nomask unless masked (`np.False_ is nomask`), then a 0-d True mask — not the
            // `masked` singleton.
            if (m.ndim == 0)
                return Yield(scope, new NDMaskedArray(product, m.GetBoolean() ? m : null));
            return Yield(scope, new NDMaskedArray(product, m));
        }

        /// <summary>Inner product with masked slots treated as 0 (NumPy's <c>ma.inner</c>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The masked inner product (mask dropped — result treats masked as 0).</returns>
        /// <exception cref="IncorrectShapeException">The last axes differ (<c>shapes (3,) and (1,) not aligned: ...</c>);
        ///     a 0-d operand counts as one element.</exception>
        /// <exception cref="ValueError">A text operand (<c>data type must provide an itemsize</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a masked slot whose product would
        ///     compute (see <see cref="FilledForNumericOp"/>).</exception>
        /// <remarks>NumPy's statements: <c>fa = filled(a, 0); fb = filled(b, 0)</c>; a 0-d one is reshaped to <c>(1,)</c>
        /// (so a scalar operand is a one-element vector, not np.inner's scalar product); then <c>np.inner(fa, fb)</c>
        /// viewed as a masked array with no mask. Scoped by hand: the zero-filled operand copies only feed the
        /// product.</remarks>
        public NDMaskedArray inner(object a, object b)
        {
            using var scope = NDScope.Open();
            Exception deferral = null;
            var fa = FilledForNumericOp(a, 0, ref deferral);
            var fb = FilledForNumericOp(b, 0, ref deferral);
            // `if fa.ndim == 0: fa.shape = (1,)` — an in-place reshape of NumPy's own temporary; a view here.
            if (fa.ndim == 0)
                fa = np.reshape(fa, new Shape(1L));
            if (fb.ndim == 0)
                fb = np.reshape(fb, new Shape(1L));
            var product = np.inner(fa, fb);
            if (deferral is not null)
                throw deferral;
            return Yield(scope, new NDMaskedArray(product, null));
        }

        /// <summary>Outer product with masked slots treated as 0; result[i,j] is masked iff a[i] or b[j] was
        /// masked (NumPy's <c>ma.outer</c>).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param>
        /// <returns>The masked outer product.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a masked slot whose product would
        ///     compute (see <see cref="FilledForNumericOp"/>).</exception>
        /// <remarks>Scoped by hand: the zero-filled operand copies, the stand-in all-False masks and the raveled /
        /// reshaped mask views are reclaimed; only the product and its mask leave. Over text NumPy's
        /// <c>np.outer(filled(a, 0).ravel(), ...)</c> refuses by dtype (the multiply loops) after formatting the
        /// fill.</remarks>
        public NDMaskedArray outer(object a, object b)
        {
            using var scope = NDScope.Open();
            var da = AsData(a); var db = AsData(b);
            // `d = np.outer(filled(a, 0).ravel(), filled(b, 0).ravel())` (np.outer ravels its operands itself).
            Exception deferral = null;
            var product = np.outer(FilledForNumericOp(a, 0, ref deferral), FilledForNumericOp(b, 0, ref deferral));
            if (deferral is not null)
                throw deferral;
            var ma_ = (a as NDMaskedArray)?._mask;
            var mb_ = (b as NDMaskedArray)?._mask;
            if (ma_ is null && mb_ is null)
                return Yield(scope, new NDMaskedArray(product, null));
            var maf = ma_ is null ? np.zeros(new Shape(da.size), np.@bool) : np.ravel(ma_);
            var mbf = mb_ is null ? np.zeros(new Shape(db.size), np.@bool) : np.ravel(mb_);
            var mask = np.reshape(maf, new Shape(da.size, 1)) | np.reshape(mbf, new Shape(1, db.size));
            return Yield(scope, new NDMaskedArray(product, mask));
        }

        /// <summary>Vandermonde matrix of a 1-D input; a masked input element ZEROES its whole row and the
        /// result is a PLAIN array (NumPy's <c>ma.vander</c>: "masked values result in rows of zeros").</summary>
        /// <param name="x">1-D data.</param><param name="n">Number of columns (null ⇒ len(x)). Lowercase to match
        /// NumPy's <c>vander(x, n=…)</c> keyword — a call spelled <c>n:</c> ports verbatim.</param>
        /// <returns>A plain <see cref="NDArray"/> Vandermonde matrix with masked rows zeroed.</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the reshaped mask column, its broadcast and the zero scalar are
        /// reclaimed; only the matrix leaves.</remarks>
        [NDScoped]
        public NDArray vander(object x, int? n = null)
        {
            var d = AsData(x);
            var v = np.vander(d, n);
            var m = (x as NDMaskedArray)?._mask;
            // `if m is not nomask: _vander[m] = 0` — a write that selects nothing (an empty result, or no masked row) has
            // no effect, and for a TEXT result (np.vander returns one only when it is empty) it must not format the 0.
            if (m is not null && v.size != 0 && np.any(m))
            {
                var col = np.reshape(m, new Shape(d.size, 1));
                np.copyto(v, NDArray.Scalar(0), casting: "unsafe", where: np.broadcast_to(col, v.Shape));
            }
            return v;
        }

        /// <summary>
        ///     Least-squares polynomial fit that DROPS masked observations first (NumPy's <c>ma.polyfit</c>):
        ///     a masked <paramref name="x"/> or <paramref name="w"/> entry — or, for 2-D <paramref name="y"/>, a
        ///     masked entry ANYWHERE in a row — removes that whole observation, and the surviving
        ///     <c>(x, y[, w])</c> rows are handed to <see cref="np.polyfit(NDArray,NDArray,int,double?,bool,NDArray,object)"/>.
        ///     Because it composes on <see cref="np.polyfit(NDArray,NDArray,int,double?,bool,NDArray,object)"/> —
        ///     which solves through <c>np.linalg.lstsq</c> — it inherits that method's backend requirement: with
        ///     no LAPACK-capable BLAS installed it throws exactly as <c>np.polyfit</c> does (an
        ///     <c>OpenBlasMissingBackendException</c>), never a silently-wrong fit. The coefficients are PLAIN
        ///     arrays, as in NumPy (the masked rows are gone, so nothing in the result is masked).
        /// </summary>
        /// <param name="x">x-coordinates, shape <c>(M,)</c>; a masked entry drops its observation.</param>
        /// <param name="y">y-coordinates, shape <c>(M,)</c> or <c>(M, K)</c>; for 2-D <paramref name="y"/> a masked
        /// entry anywhere in a row drops that row (NumPy's <c>mask_rows</c> reduction).</param>
        /// <param name="deg">Degree of the fitting polynomial.</param>
        /// <param name="rcond">Relative condition number (null ⇒ <c>len(x)·eps</c>), forwarded to <c>np.polyfit</c>.</param>
        /// <param name="full">When true the <see cref="PolyfitResult"/> also carries the SVD diagnostics.</param>
        /// <param name="w">Optional weights, shape <c>(M,)</c>; a masked entry drops its observation.</param>
        /// <param name="cov">null/false, true, or the string <c>"unscaled"</c> — forwarded to <c>np.polyfit</c>'s
        /// covariance return.</param>
        /// <returns>The <see cref="PolyfitResult"/> of the fit over the UNMASKED observations; it converts
        /// implicitly to the coefficient array and deconstructs to the <c>full</c>/<c>cov</c> tuples.</returns>
        /// <exception cref="TypeError"><paramref name="y"/> is not 1-D or 2-D, or <paramref name="w"/> is not 1-D
        /// or has a length differing from <paramref name="y"/> (NumPy's verbatim texts).</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>; <see cref="PolyfitResult"/> is an <see cref="INDArrayCarrier"/>, so
        /// the weaver yields its parts): the unioned observation mask, the row reduction of a 2-D <paramref name="y"/>'s
        /// mask, the keep-mask and the selected observation rows are reclaimed — on the backend-missing throw too.</remarks>
        [NDScoped]
        public PolyfitResult polyfit(object x, object y, int deg, double? rcond = null,
            bool full = false, object w = null, object cov = null)
        {
            NDArray xd = AsData(x);
            NDArray yd = AsData(y);

            // Union every operand's mask into ONE per-observation mask (null = nothing masked). For 2-D y a masked
            // element masks its whole ROW (NumPy's mask_rows), so reduce y's mask across columns to one bit/row.
            NDArray m = (x as NDMaskedArray)?._mask;
            if (yd.ndim == 1)
                m = Or(m, (y as NDMaskedArray)?._mask);
            else if (yd.ndim == 2)
            {
                NDArray ymask = (y as NDMaskedArray)?._mask;
                if (ymask is not null)
                    m = Or(m, np.any(ymask, 1, null, false)); // (M,) — True where any column of the row is masked
            }
            else
                throw new TypeError("Expected a 1D or 2D array for y!");

            if (w is not null)
            {
                NDArray wcheck = AsData(w);
                if (wcheck.ndim != 1)
                    throw new TypeError("expected a 1-d array for weights");
                if (wcheck.Shape.dimensions[0] != yd.Shape.dimensions[0])
                    throw new TypeError("expected w and y to have the same length");
                m = Or(m, (w as NDMaskedArray)?._mask);
            }

            NDArray wd = w is null ? null : AsData(w);

            // With any observation masked, keep only the unmasked rows (a boolean row-index selects them on
            // every operand) before delegating; otherwise hand the data straight through. The selected rows are
            // all unmasked, so their DATA is exactly the values NumPy's `x[~m]` would carry.
            if (m is not null && np.any(m))
            {
                NDArray keep = np.logical_not(m);
                NDArray xs = xd[keep];
                NDArray ys = yd[keep];
                NDArray ws = wd is null ? null : wd[keep];
                return np.polyfit(xs, ys, deg, rcond, full, ws, cov);
            }
            return np.polyfit(xd, yd, deg, rcond, full, wd, cov);
        }

        /// <summary>Element-wise membership test (NumPy's <c>ma.isin</c>): True where <paramref name="element"/>'s
        /// value is in <paramref name="test_elements"/>, carrying <paramref name="element"/>'s mask.</summary>
        /// <param name="element">Values to test.</param><param name="test_elements">The set to test against.</param>
        /// <param name="assume_unique">Speed hint that both inputs already hold unique elements (result unchanged).</param>
        /// <param name="invert">Invert the membership test.</param>
        /// <returns>A masked boolean array shaped like <paramref name="element"/>.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a non-empty
        ///     <paramref name="element"/>: numpy.ma's in1d sorts and compares the strings.</exception>
        /// <remarks>Scoped by hand: the plain membership array a masked <paramref name="element"/> overwrites, the
        /// fill scalar and any array-like operand this method had to materialize are reclaimed.
        /// <para>numpy.ma's <c>in1d</c> is NOT <c>np.isin</c>: it takes the masked uniques of both operands, concatenates
        /// them, argsorts the concatenation (mergesort) and compares neighbours. With a TEXT operand that sorts and compares
        /// strings — and formats a numeric operand into the concatenated text — so it is the next phase, except for an
        /// empty <paramref name="element"/>, whose answer is the empty membership array (no mask) whatever the test
        /// set.</para></remarks>
        public NDMaskedArray isin(object element, object test_elements, bool assume_unique = false, bool invert = false)
        {
            using var scope = NDScope.Open();
            var ed = AsData(element);
            var td = AsData(test_elements);
            if (ed.typecode == NPTypeCode.String || td.typecode == NPTypeCode.String)
            {
                if (ed.size != 0)
                    throw NDStringUfuncs.NextPhase("isin", ed.typecode == NPTypeCode.String ? ed.dtype : td.dtype);
                // `in1d(...).reshape(element.shape)` of an empty element: an empty bool array, nomask.
                return Yield(scope, new NDMaskedArray(np.zeros(DimsOf(ed), np.@bool), null));
            }
            NDArray data = np.isin(ed, td, assume_unique, invert);
            var em = (element as NDMaskedArray)?._mask;
            // A masked element is not a valid value, so its membership answer is definite: `invert`
            // (False normally, True for invert). NumPy's unique-based isin yields the same, unmasked.
            if (em is not null)
                data = np.where(em, NDArray.Scalar(invert), data);
            return Yield(scope, new NDMaskedArray(data, null));
        }

        /// <summary>1-D membership test (NumPy's <c>ma.in1d</c>): the flattened <see cref="isin"/>.</summary>
        /// <param name="element">Values to test.</param><param name="test_elements">The set to test against.</param>
        /// <param name="assume_unique">Speed hint (result unchanged).</param>
        /// <param name="invert">Invert the membership test.</param>
        /// <returns>A 1-D masked boolean array.</returns>
        /// <remarks>Scoped by hand: the raveled operand (a copy for a non-contiguous input) is reclaimed once
        /// <see cref="isin"/> has read it — the membership result never aliases it.</remarks>
        public NDMaskedArray in1d(object element, object test_elements, bool assume_unique = false, bool invert = false)
        {
            using var scope = NDScope.Open();
            return Yield(scope, isin(ravel(element), test_elements, assume_unique, invert));
        }

        /// <summary>Contiguous UNMASKED runs of a 1-D array as slices (NumPy's <c>clump_unmasked</c>).</summary>
        /// <param name="a">1-D operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous unmasked run.</returns>
        /// <remarks>Scoped by hand (the result holds no array): the inverted mask and any operand this method had
        /// to materialize are reclaimed before the slices are returned.</remarks>
        public Slice[] clump_unmasked(object a)
        {
            using var scope = NDScope.Open();
            var m = (a as NDMaskedArray)?._mask;
            if (m is null)
                return new[] { new Slice(0, (int)AsData(a).size) };
            return EzClump(np.logical_not(np.ravel(m)));
        }

        /// <summary>Contiguous MASKED runs of a 1-D array as slices (NumPy's <c>clump_masked</c>).</summary>
        /// <param name="a">1-D operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous masked run (empty when unmasked).</returns>
        /// <remarks>Scoped by hand (the result holds no array): the raveled mask — a copy when the mask is not
        /// contiguous — is reclaimed before the slices are returned.</remarks>
        public Slice[] clump_masked(object a)
        {
            using var scope = NDScope.Open();
            var m = (a as NDMaskedArray)?._mask;
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
        /// <returns>NumPy's int64 array <c>[first, last]</c> — <c>[0, size - 1]</c> when nothing is masked (an EMPTY
        ///     operand included: <c>[0, -1]</c>) — or null (NumPy's None) when every element is masked.</returns>
        /// <remarks>NumPy's statements: <c>if m is nomask or not np.any(m): return np.array([0, a.size - 1])</c>;
        /// <c>unmasked = np.flatnonzero(~m)</c>; <c>unmasked[[0, -1]]</c> when there is one, else None. Scoped by hand:
        /// the inverted, raveled mask and any operand this method had to materialize are reclaimed; only the index
        /// array leaves.</remarks>
        public NDArray flatnotmasked_edges(object a)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            var m = (a as NDMaskedArray)?._mask;
            if (m is null || !np.any(m))
                return scope.Returns(np.array(new long[] { 0, d.size - 1 }));
            var um = np.logical_not(np.ravel(m)).ToArray<bool>();
            long first = -1, last = -1;
            for (int i = 0; i < um.Length; i++)
                if (um[i]) { if (first < 0) first = i; last = i; }
            return first < 0 ? null : scope.Returns(np.array(new long[] { first, last }));
        }

        /// <summary>Contiguous UNMASKED runs of the flattened array (NumPy's <c>flatnotmasked_contiguous</c>);
        /// same as <see cref="clump_unmasked"/> for a 1-D input.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>One <see cref="Slice"/> per contiguous unmasked run.</returns>
        /// <remarks>Scoped by hand (the result holds no array): the raveled operand — a copy of a non-contiguous
        /// input — is reclaimed once its runs are read.</remarks>
        public Slice[] flatnotmasked_contiguous(object a)
        {
            using var scope = NDScope.Open();
            return clump_unmasked(ravel(a));
        }

        /// <summary>Drops the ROWS of a 2-D array that contain ANY masked element (NumPy's <c>compress_rows</c>) —
        /// a plain <see cref="NDArray"/>.</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The surviving rows: the data ITSELF when nothing is masked, an empty float64 array when everything
        ///     is (NumPy's <c>compress_nd</c>).</returns>
        /// <exception cref="NotImplementedException"><paramref name="a"/> is not 2-D (<c>compress_rows works for 2D arrays
        ///     only.</c>).</exception>
        /// <remarks>NumPy's statements: <c>a = asarray(a); if a.ndim != 2: raise NotImplementedError(...); return
        /// compress_rowcols(a, 0)</c>. Scoped (<c>[NDScoped]</c>): an array-like operand the rank check had to materialize
        /// is reclaimed.</remarks>
        [NDScoped]
        public NDArray compress_rows(object a)
        {
            if (AsData(a).ndim != 2)
                throw new NotImplementedException("compress_rows works for 2D arrays only.");
            return compress_rowcols(a, 0);
        }

        /// <summary>Drops the COLUMNS of a 2-D array that contain ANY masked element (NumPy's <c>compress_cols</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The surviving columns (see <see cref="compress_rows"/> for the unmasked and all-masked forms).</returns>
        /// <exception cref="NotImplementedException"><paramref name="a"/> is not 2-D (<c>compress_cols works for 2D arrays
        ///     only.</c>).</exception>
        /// <remarks>NumPy's <c>compress_rowcols(a, 1)</c> after its own rank check. Scoped (<c>[NDScoped]</c>), as
        /// <see cref="compress_rows"/>.</remarks>
        [NDScoped]
        public NDArray compress_cols(object a)
        {
            if (AsData(a).ndim != 2)
                throw new NotImplementedException("compress_cols works for 2D arrays only.");
            return compress_rowcols(a, 1);
        }

        /// <summary>Masks the ENTIRE rows of a 2-D array that contain any masked element (NumPy's <c>mask_rows</c>, which
        /// is <c>mask_rowcols(a, 0)</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The row-masked array (over the operand's data, with a new mask).</returns>
        /// <exception cref="NotImplementedException"><paramref name="a"/> is not 2-D: NumPy reports it from
        ///     <c>mask_rowcols</c> (<c>mask_rowcols works for 2D arrays only.</c>).</exception>
        public NDMaskedArray mask_rows(object a) => mask_rowcols(a, 0);

        /// <summary>Masks the ENTIRE columns of a 2-D array that contain any masked element (NumPy's <c>mask_cols</c>,
        /// which is <c>mask_rowcols(a, 1)</c>).</summary>
        /// <param name="a">2-D operand.</param>
        /// <returns>The column-masked array (over the operand's data, with a new mask).</returns>
        /// <exception cref="NotImplementedException"><paramref name="a"/> is not 2-D (<c>mask_rowcols works for 2D arrays
        ///     only.</c>).</exception>
        public NDMaskedArray mask_cols(object a) => mask_rowcols(a, 1);

        /// <summary>
        ///     Masks whole rows and/or columns of a 2-D array that contain any masked element (NumPy's
        ///     <c>mask_rowcols</c>): <paramref name="axis"/> null masks BOTH, 0 masks rows, 1/-1 masks columns.
        ///     The row/column sets are computed from the ORIGINAL mask (so masking a row does not spill into
        ///     "every column now has a masked element") — the two are OR'd onto a fresh mask over the shared data.
        /// </summary>
        /// <param name="a">2-D operand.</param>
        /// <param name="axis">null (rows AND cols), 0 (rows), or 1/-1 (cols).</param>
        /// <returns>The row/column-masked array (an all-unmasked input is returned unchanged).</returns>
        /// <exception cref="NotImplementedException"><paramref name="a"/> is not 2-D.</exception>
        /// <remarks>Scoped by hand: the working mask copy, the per-axis reductions, their broadcasts and the
        /// superseded OR steps are reclaimed (on the rank throw too); only the result's data and final mask leave.</remarks>
        public NDMaskedArray mask_rowcols(object a, int? axis = null)
        {
            using var scope = NDScope.Open();
            var ma_ = array(a);
            if (ma_.ndim != 2)
                throw new NotImplementedException("mask_rowcols works for 2D arrays only.");
            var m = ma_._mask;
            if (m is null || !np.any(m))
                return Yield(scope, ma_);
            var newmask = m.copy();
            // `not axis` in NumPy is truthy for None AND 0, so rows are masked for axis ∈ {null, 0}; columns for
            // axis ∈ {null, 1, -1}. Both sets read the ORIGINAL mask m.
            bool doRows = axis is null || axis == 0;
            bool doCols = axis is null || axis == 1 || axis == -1;
            if (doRows)
                newmask = np.logical_or(newmask, np.broadcast_to(np.any(m, 1, null, true), m.Shape));
            if (doCols)
                newmask = np.logical_or(newmask, np.broadcast_to(np.any(m, 0, null, true), m.Shape));
            // NumPy returns `a` itself (`a[rows] = masked` only rewrites its mask), so the fill value and hardness stay.
            return Yield(scope, new NDMaskedArray(ma_._data, newmask).UpdateFrom(ma_));
        }

        /// <summary>
        ///     Suppresses the slices (along each requested axis) that contain any masked value and returns the
        ///     surviving DATA as a PLAIN array (NumPy's <c>compress_nd</c>). <paramref name="axis"/> null
        ///     compresses along EVERY axis; an explicit axis compresses only that one. The keep-mask for each
        ///     axis is computed from the full original mask, so an N-D box is trimmed to the sub-block with no
        ///     masked element on any face.
        /// </summary>
        /// <param name="x">Operand.</param>
        /// <param name="axis">null (all axes) or a single axis (negative allowed).</param>
        /// <returns>A plain <see cref="NDArray"/> of the surviving data (an unmasked input returns its data;
        /// a fully-masked input returns an empty array).</returns>
        /// <remarks>Scoped (<c>[NDScoped]</c>): every per-axis reduction, keep-mask and intermediate compression is
        /// reclaimed; only the final compression (or the operand's own data, which the scope never owned) leaves.</remarks>
        [NDScoped]
        public NDArray compress_nd(object x, int? axis = null)
        {
            var data = AsData(x);
            var m = (x as NDMaskedArray)?._mask;
            // `if m is nomask or not m.any(): return x._data` — a VIEW of the data (OWNDATA false), as NumPy's
            // `asarray(x)._data` is, never the operand's own array object.
            if (m is null || !np.any(m))
                return data.view();
            if (np.all(m))
                return np.array(Array.Empty<double>());
            int nd = data.ndim;
            int[] axes = axis is null
                ? Enumerable.Range(0, nd).ToArray()
                : new[] { axis.Value < 0 ? axis.Value + nd : axis.Value };
            var result = data;
            foreach (var ax in axes)
            {
                // Keep the slices along `ax` that have NO masked element (reduce the mask over every OTHER axis).
                var keep = np.logical_not(AnyOverComplement(m, ax, nd));
                result = np.compress(keep, result, ax);
            }
            return result;
        }

        /// <summary>Reduces a boolean mask over every axis EXCEPT <paramref name="ax"/>, leaving a 1-D bool of
        /// length <c>shape[ax]</c> — True where that slice holds any masked element. Reduces highest-axis-first so
        /// <paramref name="ax"/>'s position stays put.</summary>
        private static NDArray AnyOverComplement(NDArray m, int ax, int nd)
        {
            var acc = m;
            for (int d = nd - 1; d >= 0; d--)
                if (d != ax)
                    acc = np.any(acc, d, null, false);
            return acc;
        }

        /// <summary>Drops the rows AND/OR columns of a 2-D array that contain any masked value (NumPy's
        /// <c>compress_rowcols</c>) — the 2-D specialization of <see cref="compress_nd"/>. Returns a plain array.</summary>
        /// <param name="x">2-D operand.</param>
        /// <param name="axis">null (rows AND cols), 0 (rows), or 1 (cols).</param>
        /// <returns>The surviving data as a plain <see cref="NDArray"/>.</returns>
        /// <exception cref="NotImplementedException"><paramref name="x"/> is not 2-D.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): an array-like operand the rank check had to materialize is
        /// reclaimed; <see cref="compress_nd"/> scopes its own intermediates.</remarks>
        [NDScoped]
        public NDArray compress_rowcols(object x, int? axis = null)
        {
            if (AsData(x).ndim != 2)
                throw new NotImplementedException("compress_rowcols works for 2D arrays only.");
            return compress_nd(x, axis);
        }

        /// <summary>
        ///     First and last UNMASKED indices along an axis (NumPy's <c>notmasked_edges</c>). Its return is
        ///     POLYMORPHIC, exactly like NumPy's (which returns a Python list): for <paramref name="axis"/> null or
        ///     a 1-D input it is the flat <c>long[] {first, last}</c> (or <c>null</c> when every element is masked);
        ///     for an explicit axis on a &gt;1-D array it is <c>NDArray[][]</c> = <c>{ mins, maxs }</c>, where
        ///     <c>mins</c>/<c>maxs</c> each hold ONE <see cref="NDArray"/> per dimension — the compressed first/last
        ///     unmasked coordinate along that dimension — matching NumPy's <c>[tuple(mins), tuple(maxs)]</c>.
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="axis">null (flatten) or the axis along which to find the edges.</param>
        /// <returns>NumPy's int64 <c>[first, last]</c> array (or <c>null</c>) for the flat/1-D case (see
        /// <see cref="flatnotmasked_edges"/>); <c>NDArray[][]</c> <c>{mins, maxs}</c> for a &gt;1-D explicit axis. Returned
        /// as <see cref="object"/> because the shape depends on the runtime axis, as it does in NumPy — cast at the call
        /// site.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the operand's rank.</exception>
        /// <remarks>Scoped by hand (the weaver does not wrap an <see cref="object"/> return): the full mask, the
        /// coordinate grids, their per-axis reductions and the masked wrappers are reclaimed; every compressed
        /// edge array in the returned jagged pair is handed back to the caller's scope.</remarks>
        public object notmasked_edges(object a, int? axis = null)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            // The flat edges are re-tracked into the CALLER's scope: returned from inside this method's scope without it,
            // they would be reclaimed when that scope closes.
            if (axis is null || d.ndim == 1)
                return scope.Returns(flatnotmasked_edges(a));

            int nd = d.ndim;
            int ax = axis.Value < 0 ? axis.Value + nd : axis.Value;
            if (ax < 0 || ax >= nd)
                throw new AxisError(axis.Value, nd);

            // NumPy: idx = array(np.indices(a.shape), mask=[m]*a.ndim); then per dimension i, the compressed
            // min/max of idx[i] along `axis`. idx[i] is the i-th coordinate grid, masked wherever a is masked.
            var mArr = getmaskarray(a);
            var indices = np.indices(d.Shape.dimensions); // (nd, *shape); indices[i] = grid i — d.shape is long[], no down-cast
            var mins = new NDArray[nd];
            var maxs = new NDArray[nd];
            for (int i = 0; i < nd; i++)
            {
                var gi = new NDMaskedArray(indices[i], mArr); // the i-th coordinate grid, masked like a
                mins[i] = compressed(gi.min(ax));
                maxs[i] = compressed(gi.max(ax));
            }
            // The edge arrays are the only survivors: each is re-tracked into the caller's scope (a compressed
            // view keeps its own reference to the reduction buffer it views, so the reductions' wrappers can go).
            for (int i = 0; i < nd; i++)
            {
                scope.Returns(mins[i]);
                scope.Returns(maxs[i]);
            }
            return new NDArray[][] { mins, maxs };
        }

        /// <summary>
        ///     Contiguous UNMASKED runs of the array as slices (NumPy's <c>notmasked_contiguous</c>). Its return is
        ///     POLYMORPHIC, exactly like NumPy's: for <paramref name="axis"/> null or a 1-D input it is a flat
        ///     <c>Slice[]</c> (one <see cref="Slice"/> per run); for a 2-D array with an explicit axis it is a
        ///     <c>Slice[][]</c> — one inner <c>Slice[]</c> per line along the OTHER axis (NumPy's list-of-lists),
        ///     an empty inner array for a fully-masked line. Only ≤ 2-D is supported (NumPy's own limit).
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="axis">null (flatten) or the axis along which each line's runs are collected.</param>
        /// <returns><c>Slice[]</c> for the flat/1-D case; <c>Slice[][]</c> for a 2-D explicit axis. Returned as
        /// <see cref="object"/> because the nesting depends on the runtime axis, as it does in NumPy — cast at the
        /// call site.</returns>
        /// <exception cref="NotSupportedException">The input has more than 2 dimensions (NumPy's
        /// <c>NotImplementedError("Currently limited to at most 2D array.")</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the operand's rank.</exception>
        /// <remarks>Scoped by hand (the result holds no array): the masked operand wrapper, every per-line view and
        /// the raveled/inverted masks behind each line's runs are reclaimed before the slices are returned.</remarks>
        public object notmasked_contiguous(object a, int? axis = null)
        {
            using var scope = NDScope.Open();
            var d = AsData(a);
            int nd = d.ndim;
            if (nd > 2)
                throw new NotSupportedException("Currently limited to at most 2D array."); // NumPy's message
            if (axis is null || nd == 1)
                return flatnotmasked_contiguous(a);

            int ax = axis.Value < 0 ? axis.Value + nd : axis.Value;
            if (ax < 0 || ax >= nd)
                throw new AxisError(axis.Value, nd);

            // Walk each line along the OTHER axis (NumPy fixes `other`=i and slices the full `axis`), collecting
            // that 1-D line's contiguous unmasked runs — one inner list per line.
            int other = (ax + 1) % 2;
            var ma_ = array(a);
            long n = d.Shape.dimensions[other];
            var result = new Slice[n][];
            for (int i = 0; i < n; i++)
            {
                NDMaskedArray line = other == 0
                    ? (NDMaskedArray)ma_[i, Slice.All]  // fix row i, vary the columns (axis 1)
                    : (NDMaskedArray)ma_[Slice.All, i]; // fix column i, vary the rows (axis 0)
                result[i] = flatnotmasked_contiguous(line);
            }
            return result;
        }

        /// <summary>The raw buffer addresses of the data and mask areas (NumPy's <c>ids</c>) — the mask address
        /// is 0 when the array is <c>nomask</c>. Diagnostic only; the values are unmanaged pointers into
        /// NumSharp's own buffers, not comparable across processes.</summary>
        /// <param name="a">Operand.</param>
        /// <returns>A tuple of the data-buffer and mask-buffer addresses (mask 0 for nomask).</returns>
        /// <remarks>Scoped by hand: an array-like operand this method had to materialize is reclaimed once its
        /// address is read — so, exactly as NumPy's <c>ids</c> of a temporary, the reported address names a buffer
        /// that no longer exists by the time the caller sees it. Only an array/masked-array operand yields an
        /// address that stays meaningful.</remarks>
        public unsafe (long data, long mask) ids(object a)
        {
            using var scope = NDScope.Open();
            long dataPtr = (long)AsData(a).Storage.Address;
            var m = (a as NDMaskedArray)?._mask;
            long maskPtr = m is null ? 0L : (long)m.Storage.Address;
            return (dataPtr, maskPtr);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  In-place scatter / resize (NumPy's put / putmask / resize)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     IN-PLACE flat scatter (NumPy's <c>ma.put</c>): sets <c>a.flat[indices] = values</c>, cycling
        ///     <paramref name="values"/> when shorter. A masked <paramref name="values"/> (the <c>masked</c>
        ///     singleton or a masked-array's mask) masks the written slots; a plain value UNMASKS them. The mask
        ///     is created on demand and shrunk back to <c>nomask</c> if it ends up all-False.
        /// </summary>
        /// <param name="a">The <see cref="NDMaskedArray"/> (or plain array) to mutate.</param>
        /// <param name="indices">Flat integer target indices.</param>
        /// <param name="values">Scalar/array/<see cref="NDMaskedArray"/>/the <c>masked</c> singleton.</param>
        /// <param name="mode">Out-of-bounds policy: "raise" (default)/"wrap"/"clip".</param>
        /// <remarks>
        ///     Scoped by hand: the hard-mask snapshots, the fill scalars and any array-like value this method had
        ///     to materialize are reclaimed — on an out-of-bounds throw too. A mask the call CREATES
        ///     (<see cref="NDMaskedArray.EnsureMask"/>) is a field store and so escapes every scope by design; the
        ///     hard-mask restore ORs into the existing mask IN PLACE (NumPy writes <c>self._mask</c> in place too),
        ///     so no second mask array is created and none is orphaned. When the mask shrinks back to
        ///     <c>nomask</c> the dropped mask array is NOT disposed: another masked array may share it (NumPy's
        ///     <c>_sharedmask</c>), so it is left to its remaining owners and, finally, the GC.
        /// </remarks>
        public void put(object a, NDArray indices, object values, string mode = "raise")
        {
            using var scope = NDScope.Open();
            var ma_ = a as NDMaskedArray;
            var data = AsData(a);
            if (values is NDMaskedConstant)
            {
                // Mask the slots; leave the data (hidden). A plain array carries no mask, so nothing to do there.
                if (ma_ is not null)
                {
                    ma_.EnsureMask();
                    np.put(ma_._mask, indices, NDArray.Scalar(true), mode);
                }
                return;
            }
            // HARD MASK: a plain put must not reach masked slots nor unmask them (NumPy leaves a hard-masked
            // target untouched). Snapshot data+mask, do the ordinary put, then restore the masked positions.
            bool hard = ma_ is not null && ma_._hardmask && ma_._mask is not null;
            NDArray savedMask = hard ? ma_._mask.copy() : null;
            NDArray savedData = hard ? data.copy() : null;
            if (values is NDMaskedArray mv)
            {
                np.put(data, indices, mv._data, mode);
                if (mv._mask is not null && ma_ is not null)
                {
                    ma_.EnsureMask();
                    np.put(ma_._mask, indices, mv._mask, mode);
                }
                else if (!hard && ma_?._mask is not null)
                {
                    np.put(ma_._mask, indices, NDArray.Scalar(false), mode); // unmask (soft only)
                }
            }
            else
            {
                np.put(data, indices, AsData(values), mode);
                if (!hard && ma_?._mask is not null)
                    np.put(ma_._mask, indices, NDArray.Scalar(false), mode); // unmask (soft only)
            }
            if (hard)
            {
                // Restore data where originally masked and keep those slots masked (retain any freshly-masked ones).
                // The OR lands IN PLACE in the live mask: replacing the field with a fresh array would orphan the
                // mask the put just wrote into (and break a sharer that sees this mask object).
                np.copyto(data, savedData, casting: "unsafe", where: savedMask);
                np.logical_or(savedMask, ma_._mask, @out: ma_._mask);
            }
            // NumPy shrinks the mask after put: an all-False mask collapses to nomask.
            if (ma_?._mask is not null && !np.any(ma_._mask))
                ma_._mask = null;
        }

        /// <summary>
        ///     IN-PLACE conditional write (NumPy's <c>ma.putmask</c>): writes <paramref name="values"/> into
        ///     <paramref name="a"/>'s data wherever <paramref name="mask"/> is True, and reconciles the mask
        ///     there — masked <paramref name="values"/> mask those slots, plain values UNMASK them. Unlike
        ///     <see cref="put"/>, alignment is by <see cref="np.copyto"/> broadcast (not cycling).
        /// </summary>
        /// <param name="a">The <see cref="NDMaskedArray"/> (or plain array) to mutate.</param>
        /// <param name="mask">Boolean array selecting the write positions (a C# <c>bool[]</c> converts to one; NumPy's
        ///     <c>np.copyto(..., where=mask)</c> takes it, so a numeric or text array is refused).</param>
        /// <param name="values">Scalar/array/<see cref="NDMaskedArray"/> supplying the data (and, if masked, the mask). A
        ///     C# number is <c>getdata(values)</c>: a strong default-width array (an int is int64, a float float64).</param>
        /// <exception cref="ArgumentException">A <paramref name="mask"/> that is not boolean: NumPy's TypeError <c>Cannot
        ///     cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'</c> (the house
        ///     ArgumentException of <see cref="np.copyto(NDArray, NDArray, string, NDArray)"/>), raised whatever the size — an empty text mask included.</exception>
        /// <exception cref="InvalidCastException"><paramref name="values"/> cannot reach the data's dtype by a same_kind cast
        ///     (<c>Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'same_kind'</c>; text
        ///     values for numeric data whatever the mask holds).</exception>
        /// <exception cref="StringComputationNotSupportedException">Numbers written into text data (NumPy formats them
        ///     as text: the next phase of the string support).</exception>
        /// <remarks>
        ///     NumPy's copies all use <c>np.copyto</c>'s default <c>same_kind</c> casting: a float written into int data is
        ///     refused, an int64 written into int8 data wraps.
        ///     <para>
        ///     Scoped by hand: the hard-mask working copy, the stand-in all-False value mask and any array-like operand this
        ///     method had to materialize are reclaimed. A mask the call CREATES on a <c>nomask</c> target is a field store
        ///     and is detached from every scope (it lives as long as the target); the hard-mask branch ORs into the existing
        ///     mask IN PLACE — NumPy's <c>a.mask |= m</c> — rather than replacing the field, so no mask array is orphaned.
        ///     </para>
        /// </remarks>
        public void putmask(object a, object mask, object values)
        {
            using var scope = NDScope.Open();
            var ma_ = a as NDMaskedArray;
            var data = AsData(a);
            // `where=mask`: np.copyto converts the mask itself, refusing a non-bool array by the safe-cast rule.
            var maskArr = AsData(mask);
            // (valdata, valmask) = (getdata(values), getmask(values)).
            var valdata = GetDataStrong(values);
            var valmask = (values as NDMaskedArray)?._mask;
            // NumPy's ma.putmask has three mask branches; the DATA is ALWAYS written at the end (even at masked
            // slots — putmask does NOT protect data, unlike put; the hard mask only freezes the MASK).
            if (ma_ is null || ma_._mask is null)
            {
                // No existing mask: only create one if the VALUES bring a mask (else stay nomask). The new mask is
                // a FIELD of the target: Own detaches it so this method's scope cannot dispose it.
                if (valmask is not null && ma_ is not null)
                {
                    ma_._mask = Own(np.zeros(DimsOf(data), np.@bool));
                    np.copyto(ma_._mask, valmask, where: maskArr);
                }
            }
            else if (ma_._hardmask)
            {
                // HARD: never UNmask — only OR in a masked value's own mask at the written slots; a plain
                // (unmasked) value leaves the mask untouched. (Data is still written below, keeping masked
                // slots masked but with new hidden data — exactly NumPy's `a.mask |= m` branch, in place.)
                if (valmask is not null)
                {
                    var m = ma_._mask.copy();
                    np.copyto(m, valmask, where: maskArr);
                    np.logical_or(ma_._mask, m, @out: ma_._mask);
                }
            }
            else
            {
                // SOFT: unmasked values (nomask) become all-False so the written slots unmask.
                var vm = valmask ?? np.zeros(DimsOf(valdata), np.@bool);
                np.copyto(ma_._mask, vm, where: maskArr);
            }
            // np.copyto(a._data, valdata, where=mask): same_kind, so a float into int data or text into numbers is refused.
            np.copyto(data, valdata, where: maskArr);
        }

        /// <summary>
        ///     numpy.ma's <c>getdata(value)</c> for a value operand: a masked array's data, an array itself, and a bare C#
        ///     scalar as <c>np.array(value)</c> — a STRONG array of its default width (an int is int64, a float float64; see
        ///     <see cref="BareScalarStrongDtype"/>), so a cast check names NumPy's dtype (<c>dtype('int64')</c>).
        /// </summary>
        /// <param name="value">The value operand.</param>
        /// <returns>The data array (the operand's own, or a new 0-d array).</returns>
        private static NDArray GetDataStrong(object value)
        {
            var d = AsData(value);
            return BareScalarStrongDtype(value) is { } strong && d.typecode != strong ? d.astype(strong) : d;
        }

        /// <summary>
        ///     Returns a NEW masked array of <paramref name="new_shape"/> filled with repeated copies of
        ///     <paramref name="x"/>'s data AND mask (NumPy's <c>ma.resize</c>) — always a masked array, never in
        ///     place. Enlarging TILES the input (data and mask alike, so a masked element recurs masked);
        ///     shrinking truncates.
        /// </summary>
        /// <param name="x">Operand.</param><param name="new_shape">Target shape.</param>
        /// <returns>A fresh masked array of <paramref name="new_shape"/>.</returns>
        /// <remarks>Scoped by hand: an array-like operand this method had to materialize is reclaimed once it has
        /// been tiled; only the tiled data and mask leave.</remarks>
        public NDMaskedArray resize(object x, Shape new_shape)
        {
            using var scope = NDScope.Open();
            var data = np.resize(AsData(x), new_shape);
            var m = (x as NDMaskedArray)?._mask;
            return Yield(scope, new NDMaskedArray(data, m is null ? null : np.resize(m, new_shape)));
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Creation / split / iteration wrappers (frombuffer / fromfunction / hsplit / ndenumerate)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>Interprets a byte buffer as a 1-D array (NumPy's <c>ma.frombuffer</c>) — an UNMASKED masked
        /// array wrapping <see cref="np.frombuffer(byte[],DType,long,long)"/>.</summary>
        /// <param name="buffer">The source bytes.</param><param name="dtype">Element dtype (null ⇒ float64).</param>
        /// <param name="count">Elements to read (-1 = all).</param><param name="offset">Start byte offset.</param>
        /// <returns>An unmasked masked array over the buffer's values.</returns>
        public NDMaskedArray frombuffer(byte[] buffer, DType dtype = null, long count = -1, long offset = 0)
            => new NDMaskedArray(np.frombuffer(buffer, dtype, count, offset), null);

        /// <summary>
        ///     Builds an array by applying <paramref name="function"/> to the coordinate grids of
        ///     <paramref name="shape"/> (NumPy's <c>ma.fromfunction</c>) — an UNMASKED result. The delegate
        ///     receives one <see cref="NDArray"/> per axis (the i-th being that axis's index grid), matching
        ///     NumPy's <c>function(*indices)</c>; it must be vectorized (operate on the whole grids at once).
        /// </summary>
        /// <param name="function">Vectorized function of the per-axis index grids.</param>
        /// <param name="shape">Output shape (the grids' shape).</param>
        /// <param name="dtype">Index-grid dtype; null ⇒ <see cref="np.float64"/>, NumPy's <c>dtype=float</c> default
        /// (probed 2.4.2: <c>ma.fromfunction(lambda i, j: i + j, (3, 4)).dtype</c> is float64). Pass
        /// <see cref="np.int64"/> for integer grids.</param>
        /// <returns>An unmasked masked array of <paramref name="function"/>'s result.</returns>
        /// <remarks>
        ///     The null default USED to fall through to <see cref="np.indices(int[], DType)"/>'s own default
        ///     (int64), so every default-dtype call computed on integer grids where NumPy computes on float64 —
        ///     a wrong result dtype (and integer arithmetic, e.g. division) for the most common spelling.
        ///     <para>
        ///     <b>Ownership</b> — the np.ma callback contract (<see cref="apply_along_axis"/>): whatever
        ///     <paramref name="function"/> returns becomes the result's data and stays the CALLER's, so the call
        ///     runs outside any scope of this method's (one would capture — and dispose — the callback's own
        ///     allocations). The index grid and the per-axis grid views handed to <paramref name="function"/> are
        ///     this method's own and are released once it returns (on a throw too) — so a callback must not retain
        ///     its ARGUMENTS past the call. A grid the callback returns as-is is the result's data and is kept; a
        ///     view it returns holds its own reference to the grid buffer, so releasing the rest is safe.
        ///     </para>
        /// </remarks>
        public NDMaskedArray fromfunction(Func<NDArray[], NDArray> function, Shape shape, DType dtype = null)
        {
            var dims = shape.dimensions.Select(x => (int)x).ToArray();
            // shape (ndim, *dims); grid[i] is the i-th coordinate grid, float64 unless the caller asked otherwise.
            var grid = np.indices(dims, dtype ?? np.float64);
            var args = new NDArray[dims.Length];
            NDArray result = null;
            try
            {
                for (int i = 0; i < dims.Length; i++)
                    args[i] = grid[i];
                result = function(args);
                return new NDMaskedArray(result, null);
            }
            finally
            {
                // Release the grid views the callback was handed — except one it returned verbatim, which IS the
                // result's data now — then the grid wrapper itself. A returned view of the grid keeps the buffer
                // alive through its own counted reference, so this never frees memory the result still reads.
                foreach (var arg in args)
                    if (arg is not null && !ReferenceEquals(arg, result))
                        arg.Dispose();
                grid.Dispose();
            }
        }

        /// <summary>Two-axis convenience for <see cref="fromfunction(Func{NDArray[],NDArray},Shape,DType)"/> —
        /// matches NumPy's <c>fromfunction(lambda i, j: …, (m, n))</c>.</summary>
        /// <param name="function">Vectorized function of the row and column index grids.</param>
        /// <param name="shape">Output 2-D shape.</param><param name="dtype">Index-grid dtype (null ⇒ float64, NumPy's
        /// default).</param>
        /// <returns>An unmasked masked array.</returns>
        /// <remarks>Same ownership contract as the array-of-grids overload: the two grids handed to
        /// <paramref name="function"/> are released after the call unless one IS the returned data.</remarks>
        public NDMaskedArray fromfunction(Func<NDArray, NDArray, NDArray> function, Shape shape, DType dtype = null)
            => fromfunction(idx => function(idx[0], idx[1]), shape, dtype);

        /// <summary>Splits <paramref name="a"/> COLUMN-wise into <paramref name="sections"/> equal parts (NumPy's
        /// <c>ma.hsplit</c>); data AND mask are split alike so each part keeps its masked-ness.</summary>
        /// <param name="a">Operand (rank ≥ 1; splits along axis 1, or axis 0 for a 1-D input).</param>
        /// <param name="sections">Number of equal parts.</param>
        /// <returns>The masked-array parts.</returns>
        /// <remarks>Scoped by hand: an array-like operand this method had to materialize is reclaimed (each part is
        /// a view holding its own reference to that buffer); every part's data and mask leave.</remarks>
        public NDMaskedArray[] hsplit(object a, int sections)
        {
            using var scope = NDScope.Open();
            return Yield(scope, ZipSplit(np.hsplit(AsData(a), sections), (a as NDMaskedArray)?._mask is null ? null : np.hsplit(((NDMaskedArray)a)._mask, sections)));
        }

        /// <summary>Splits <paramref name="a"/> COLUMN-wise at the given cut points (NumPy's <c>ma.hsplit</c>
        /// with an index list); data AND mask split alike.</summary>
        /// <param name="a">Operand.</param><param name="indices">Column cut points.</param>
        /// <returns>The masked-array parts.</returns>
        /// <remarks>Scoped by hand, as the <c>sections</c> overload.</remarks>
        public NDMaskedArray[] hsplit(object a, int[] indices)
        {
            using var scope = NDScope.Open();
            return Yield(scope, ZipSplit(np.hsplit(AsData(a), indices), (a as NDMaskedArray)?._mask is null ? null : np.hsplit(((NDMaskedArray)a)._mask, indices)));
        }

        /// <summary>Pairs the split data parts with their (optional) mask parts into masked arrays.</summary>
        private static NDMaskedArray[] ZipSplit(NDArray[] dataParts, NDArray[] maskParts)
        {
            var res = new NDMaskedArray[dataParts.Length];
            for (int i = 0; i < dataParts.Length; i++)
                res[i] = new NDMaskedArray(dataParts[i], maskParts?[i]);
            return res;
        }

        /// <summary>
        ///     Enumerates <c>(index, value)</c> pairs for the UNMASKED elements only, in C-order (NumPy's
        ///     <c>ma.ndenumerate</c> — masked positions are SKIPPED, unlike <see cref="np.ndenumerate(NDArray)"/>
        ///     which yields every element). Each index is a fresh <c>long[]</c> coordinate.
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <returns>The unmasked <c>(index, value)</c> pairs.</returns>
        /// <remarks>
        ///     An ITERATOR, so it cannot hold an <see cref="NDScope"/> (a scope is per-thread and would straddle
        ///     every <c>yield</c> — the consumer's own scopes would interleave with it). It releases its two own
        ///     arrays explicitly instead: the raveled mask (a copy when the mask is not contiguous) as soon as it
        ///     has been read into the managed flag array, and an array-like operand <see cref="AsData"/> had to
        ///     materialize when the walk ends — including when the consumer stops early and disposes the
        ///     enumerator. An array or masked-array operand is the caller's and is never released.
        /// </remarks>
        public IEnumerable<(long[] index, object value)> ndenumerate(object a)
        {
            var d = AsData(a);
            // AsData materializes a scalar/array-like operand into a fresh array; an array or masked-array
            // operand only hands back the caller's own data.
            bool dIsOurs = !(a is NDMaskedArray || a is NDArray);
            try
            {
                var m = (a as NDMaskedArray)?._mask;
                // np.ndenumerate walks C-order, so the k-th pair aligns with the k-th C-order mask element.
                bool[] mflat = null;
                if (m is not null)
                {
                    // np.ravel always returns a DISTINCT array (a view or a copy), never the mask object itself,
                    // so disposing it releases only the raveled wrapper's own reference.
                    using var raveled = np.ravel(m);
                    mflat = raveled.ToArray<bool>();
                }
                long k = 0;
                foreach (var pair in np.ndenumerate(d))
                {
                    if (mflat is null || !mflat[k])
                        yield return pair;
                    k++;
                }
            }
            finally
            {
                if (dIsOurs)
                    d.Dispose();
            }
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Sorting / unique — masked entries sort to the END (filled with the dtype's largest
        //  value) and the trailing k slots (k = masked count along the axis) are re-masked.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Sorts along an axis with masked entries pushed to the END (or, with <paramref name="endwith"/>
        ///     false, the FRONT) and re-masked there (NumPy's <c>ma.sort</c>). The masked slots' key is NaN for a
        ///     floating dtype and the dtype's LARGEST value otherwise when <paramref name="endwith"/> (so they sort last,
        ///     ascending), its SMALLEST value otherwise; their ORIGINAL data rides along to the end/front rather than the
        ///     fill value a plain sort-of-filled would leave behind.
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="axis">Sort axis (default last; negative counts from the end), or null to sort the flattened array.</param>
        /// <param name="endwith">True (default) pushes masked entries to the END; false to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default (see
        /// <see cref="SortFill"/>).</param>
        /// <returns>The sorted masked array — a copy finalized from <paramref name="a"/>: its hard-mask flag and fill value.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the operand.</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand with elements to compare.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>a = np.array(a, copy=True, subok=True)</c> (a copy that keeps the subclass, so it
        ///     finalizes from the operand), <c>a = a.flatten(); axis = 0</c> for <c>axis=None</c>, then
        ///     <c>MaskedArray.sort</c>: without a mask the data is sorted in place; with one <c>sidx = a.argsort(axis,
        ///     endwith=, fill_value=)</c> and <c>a[...] = np.take_along_axis(a, sidx, axis)</c>. That last write goes through
        ///     <c>__setitem__</c>, so on a HARD-masked operand the slots that were masked keep their old data and the masks
        ///     are OR-ed — NumPy's (surprising) answer, reproduced here through this module's indexer.
        ///     <para>Scoped by hand: the filled sort keys, the permutation and the gathered values are reclaimed (as is an
        ///     array-like operand this method had to materialize); only the sorted copy leaves.</para>
        /// </remarks>
        public NDMaskedArray sort(object a, int? axis = -1, bool endwith = true, object fill_value = null)
        {
            using var scope = NDScope.Open();
            var source = a as NDMaskedArray;
            // `a = np.array(a, copy=True, subok=True)`: the data and mask copied, the attributes finalized from the operand.
            var x = new NDMaskedArray(AsData(a).copy(), source?._mask?.copy());
            if (source is not null)
                x.FinalizeFrom(source);
            int ax;
            if (axis is null)
            {
                // `a = a.flatten(); axis = 0` — the _arraymethod flatten keeps the attributes.
                x = flatten(x);
                ax = 0;
            }
            else
                ax = axis.Value;
            // MaskedArray.sort: `if self._mask is nomask: ndarray.sort(self, axis=axis); return`.
            if (x._mask is null)
                return Yield(scope, new NDMaskedArray(np.sort(x._data, ax), null).UpdateFrom(x));
            // `sidx = self.argsort(...)`; `self[...] = np.take_along_axis(self, sidx, axis=axis)` through __setitem__.
            var sidx = argsort(x, ax, endwith, fill_value);
            var taken = new NDMaskedArray(np.take_along_axis(x._data, sidx, ax), np.take_along_axis(x._mask, sidx, ax));
            x[Slice.Ellipsis] = taken;
            return Yield(scope, x);
        }

        /// <summary>
        ///     Indices that would sort the array with masked entries treated as the dtype's largest (or, with
        ///     <paramref name="endwith"/> false, smallest) value — NumPy's <c>ma.argsort</c>. A PLAIN int64
        ///     index array (masked entries land at the end/front).
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="axis">Sort axis (negative counts from the end), or null for the flattened array (NumPy's
        ///     <c>axis=None</c>). Omitting it is NumPy's default (see <see cref="argsort(object,bool,object)"/>).</param>
        /// <param name="endwith">True (default) sends masked entries to the END; false to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with two or more elements to order.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): the filled sort keys (and an array-like operand this method had to
        /// materialize) are reclaimed; only the index array leaves.</remarks>
        [NDScoped]
        public NDArray argsort(object a, int? axis, bool endwith = true, object fill_value = null)
        {
            var d = (a as NDMaskedArray)?._mask is null ? AsData(a) : ((NDMaskedArray)a).filled(fill_value ?? SortFill(a, endwith));
            // `filled.argsort(axis=None)` sorts the flattened array.
            return axis is null ? np.argsort(np.ravel(d), -1) : np.argsort(d, axis.Value);
        }

        /// <summary>
        ///     <see cref="argsort(object,int?,bool,object)"/> with NumPy's DEFAULT axis (<c>ma.argsort(a,
        ///     axis=np._NoValue)</c>): <c>_deprecate_argsort_axis</c> picks the last axis for a 0-d or 1-D operand and None
        ///     (the FLATTENED array, with NumPy's FutureWarning) from two dimensions on.
        /// </summary>
        /// <param name="a">Operand.</param>
        /// <param name="endwith">True (default) sends masked entries to the END; false to the FRONT.</param>
        /// <param name="fill_value">Override the masked sort key; null uses the endwith default.</param>
        /// <returns>An int64 <see cref="NDArray"/> of sort indices (1-D for an operand of two or more dimensions).</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with two or more elements to order.</exception>
        /// <remarks>Scoped (<c>[NDScoped]</c>): an array-like operand materialized to read its rank is reclaimed.</remarks>
        [NDScoped]
        public NDArray argsort(object a, bool endwith = true, object fill_value = null)
            => argsort(a, AsData(a).ndim <= 1 ? -1 : (int?)null, endwith, fill_value);

        /// <summary>The masked-slot sort key for <see cref="sort"/>/<see cref="argsort(object,int?,bool,object)"/>, NumPy's
        /// <c>MaskedArray.argsort</c>: with endwith a FLOATING dtype (float16 / float32 / float64) uses NaN ("nan &gt;
        /// inf": a masked slot sorts after a +inf element) and every other dtype its largest value
        /// (<see cref="minimum_fill_value"/>); without endwith the dtype's smallest value (masked sorts first).</summary>
        /// <param name="a">The operand.</param>
        /// <param name="endwith">True to send masked slots to the end.</param>
        /// <returns>The fill value the masked slots are sorted by.</returns>
        private object SortFill(object a, bool endwith)
        {
            if (!endwith)
                return maximum_fill_value(a);
            // `if np.issubdtype(self.dtype, np.floating): fill_value = np.nan`.
            var tc = (a as NDMaskedArray)?._data.typecode ?? AsData(a).typecode;
            return tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double ? double.NaN : minimum_fill_value(a);
        }

        /// <summary>Sorted unique values over the UNMASKED elements, plus ONE trailing masked entry if any
        /// element was masked (NumPy's <c>ma.unique</c>, values only).</summary>
        /// <param name="ar">Operand.</param>
        /// <returns>The masked array of unique values.</returns>
        /// <exception cref="StringComputationNotSupportedException">A text operand with both masked and unmasked elements
        ///     (NumPy's masked sort compares the strings with the fill value), or with two or more unmasked elements.</exception>
        /// <remarks>
        ///     NumPy's <c>np.unique</c> of the masked array: the operand is flattened; with no masked element the unique
        ///     values carry no mask; otherwise the masked sort keys every masked element by the same fill value, so they
        ///     sort together (last for numbers, whose key is the dtype maximum) and the ONE kept masked slot holds the data of
        ///     the FIRST masked element in C order — what NumPy's sort leaves first among equal keys. For TEXT the key is the
        ///     array's fill value (<c>minimum_fill_value</c> is None), so where the masked slot lands depends on how the
        ///     strings compare with <c>'N/A'</c> — the next phase — unless every element is masked. Reading that fill
        ///     materializes the text default, which the result (a view of the sorted copy) converts to its dtype
        ///     (<c>'N/'</c> for <c>&lt;U2</c>). The result inherits the hard-mask flag and fill value.
        ///     <para>Scoped by hand: the compressed values, the plain unique set a masked input extends, and the raveled
        ///     stand-ins behind the trailing masked slot are reclaimed; only the result's data and mask leave.</para>
        /// </remarks>
        public NDMaskedArray unique(object ar)
        {
            using var scope = NDScope.Open();
            var source = ar as NDMaskedArray;
            var mask = source?._mask;
            var data = AsData(ar);
            bool anyMasked = mask is not null && np.any(mask);
            bool text = data.typecode == NPTypeCode.String;
            // A masked text element sorts by the fill value among the strings: where it lands is a string comparison.
            if (text && anyMasked && !np.all(mask))
                throw NDStringUfuncs.NextPhase("unique", data.dtype);
            NDArray uvals = np.unique(compressed(ar));
            NDMaskedArray result;
            if (!anyMasked)
                result = new NDMaskedArray(uvals, null);
            else
            {
                // The one masked slot: the first masked element's data, C order (NumPy flattens before it sorts).
                long first = np.argmax(np.ravel(mask));
                var masked = np.ravel(data)[$"{first}:{first + 1}"];
                var values = np.concatenate(new[] { uvals, masked }, 0);
                var m = np.concatenate(new[] { np.zeros(DimsOf(uvals), np.@bool), np.ones(new Shape(1), np.@bool) }, 0);
                result = new NDMaskedArray(values, m);
            }
            if (source is not null)
            {
                // `output.view(MaskedArray)` finalizes from the sorted copy: its fill value converted, its hardness.
                result.FinalizeFrom(source);
                // That copy's text default was MATERIALIZED by the masked sort's `filled(None)` read of `fill_value`.
                if (text && anyMasked && source._fill_value is null)
                    result._fill_value = StoreFill(default_fill_value(data.dtype), data.dtype);
            }
            return Yield(scope, result);
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
            var m = (a as NDMaskedArray)?._mask;
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
        private static NDMaskedArray WithTrailingMasked(NDArray values, bool includeMasked)
        {
            if (!includeMasked)
                return new NDMaskedArray(values, null);
            // The datum at the masked slot is arbitrary — the mask hides it and NumPy's own raw value there is
            // whatever its sort/concatenate left (not the fill). A zero of the result dtype keeps `.filled()`
            // (the observable value) consistent with the dtype default, matching NumPy's contract.
            var data = np.concatenate(new[] { values, np.zeros(new Shape(1), values.dtype) }, 0);
            var m = np.concatenate(new[] { np.zeros(DimsOf(values), np.@bool), np.ones(new Shape(1), np.@bool) }, 0);
            return new NDMaskedArray(data, m);
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
        /// <param name="ar1">First operand (<see cref="NDMaskedArray"/>/<see cref="NDArray"/>/scalar/array-like).</param>
        /// <param name="ar2">Second operand.</param>
        /// <param name="assume_unique">Speed hint passed to the underlying set op that the inputs already hold
        /// unique elements; when it is false but the inputs are not unique, the result is undefined (matching NumPy).</param>
        /// <returns>The masked array of shared unique values (sorted; the promoted dtype is preserved even for an
        /// empty/all-masked result).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        /// <remarks>Scoped by hand: the compressed operands, the plain set result and its dtype cast are
        /// reclaimed (as is any operand this method had to materialize); only the result's data and mask leave.</remarks>
        public NDMaskedArray intersect1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            var dt = SetOpDtype(ar1, ar2);
            // Compress to the unmasked values, run the plain set op, then FORCE the promoted dtype (astype is a
            // no-op copy on the non-empty path since the set op already promoted; it only bites when both inputs
            // compress to empty, where np.intersect1d would otherwise return float64).
            var values = np.intersect1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            return Yield(scope, WithTrailingMasked(values, HasMasked(ar1) && HasMasked(ar2)));
        }

        /// <summary>
        ///     Sorted union of the UNMASKED values of both inputs, plus a single masked entry iff EITHER input
        ///     contained a masked element (NumPy's <c>ma.union1d</c>). Always returns a masked array.
        /// </summary>
        /// <param name="ar1">First operand.</param><param name="ar2">Second operand.</param>
        /// <returns>The masked array of unique values from either input (sorted; promoted dtype preserved).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        /// <remarks>Scoped by hand, as <see cref="intersect1d"/>.</remarks>
        public NDMaskedArray union1d(object ar1, object ar2)
        {
            using var scope = NDScope.Open();
            var dt = SetOpDtype(ar1, ar2);
            var values = np.union1d(compressed(ar1), compressed(ar2)).astype(dt, copy: false);
            return Yield(scope, WithTrailingMasked(values, HasMasked(ar1) || HasMasked(ar2)));
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
        /// <remarks>Scoped by hand, as <see cref="intersect1d"/>.</remarks>
        public NDMaskedArray setxor1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            var dt = SetOpDtype(ar1, ar2);
            var values = np.setxor1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            // XOR of masked presence: the collapsed masked value is in exactly one side ⇒ survives the xor.
            return Yield(scope, WithTrailingMasked(values, HasMasked(ar1) ^ HasMasked(ar2)));
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
        /// <returns>The masked array of <paramref name="ar1"/>-only unique values (sorted, in <paramref name="ar1"/>'s
        /// dtype: NumPy's result is a selection of ar1, not of the promoted comparison).</returns>
        /// <exception cref="ArgumentNullException">Either operand is null.</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand with a non-empty
        ///     <paramref name="ar1"/> (the uniques and the membership test compare strings).</exception>
        /// <remarks>Scoped by hand, as <see cref="intersect1d"/>.</remarks>
        public NDMaskedArray setdiff1d(object ar1, object ar2, bool assume_unique = false)
        {
            using var scope = NDScope.Open();
            // numpy.ma's setdiff1d is `ar1[in1d(unique(ar1), unique(ar2), invert=True)]`: with a TEXT operand the uniques
            // sort strings and in1d compares them (formatting a numeric operand into text) — the next phase — unless ar1
            // is empty, whose selection is empty whatever ar2 holds.
            var d1 = AsData(ar1);
            var d2 = AsData(ar2);
            if ((d1.typecode == NPTypeCode.String || d2.typecode == NPTypeCode.String) && d1.size != 0)
                throw NDStringUfuncs.NextPhase("setdiff1d", d1.typecode == NPTypeCode.String ? d1.dtype : d2.dtype);
            // `return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]`: the result is a selection of ar1 (its unique
            // values), so it keeps ar1's dtype — not the promoted dtype the membership test compares in.
            var dt = d1.dtype;
            var values = np.setdiff1d(compressed(ar1), compressed(ar2), assume_unique).astype(dt, copy: false);
            // ar1's masked value survives the diff iff ar2 does not also carry one (masked == masked removes it).
            return Yield(scope, WithTrailingMasked(values, HasMasked(ar1) && !HasMasked(ar2)));
        }

        // ── Mask hardness (NumPy's harden_mask/soften_mask): a HARD mask forbids UNmasking via a plain-value
        //    assignment (indexer/put/putmask), so masked slots survive `x[i] = v`. Masking a slot (`x[i] = masked`)
        //    is always allowed. The module functions delegate to the instance methods, mutating IN PLACE and
        //    returning the SAME masked array (NumPy's `np.ma.harden_mask(a)` is `a.harden_mask()`). ──

        /// <summary>Makes <paramref name="a"/>'s mask HARD in place (NumPy's <c>ma.harden_mask</c>): a later
        /// plain-value assignment can no longer unmask a slot. A plain array is wrapped first (NumPy's
        /// <c>asanyarray(a).harden_mask()</c>), so it comes back HARD too (inert until the array gains a mask).</summary>
        /// <param name="a">Operand (a <see cref="NDMaskedArray"/> is mutated in place).</param>
        /// <returns>The same masked array, now hard-masked.</returns>
        public NDMaskedArray harden_mask(object a) => asanyarray(a).harden_mask();

        /// <summary>Makes <paramref name="a"/>'s mask SOFT again in place (NumPy's <c>ma.soften_mask</c>, the
        /// default): a plain-value assignment unmasks the assigned slots. A plain array is wrapped first
        /// (<c>asanyarray(a).soften_mask()</c>).</summary>
        /// <param name="a">Operand (a <see cref="NDMaskedArray"/> is mutated in place).</param>
        /// <returns>The same masked array, now soft-masked.</returns>
        public NDMaskedArray soften_mask(object a) => asanyarray(a).soften_mask();

        /// <summary>Drops an all-False mask back to nomask IN PLACE (the module form of NumPy's
        /// <c>MaskedArray.shrink_mask</c>, as <c>_frommethod</c> builds them: <c>asanyarray(a).shrink_mask()</c>).</summary>
        /// <param name="a">Operand (a <see cref="NDMaskedArray"/> is changed in place and returned; a plain array is wrapped,
        ///     with no mask).</param>
        /// <returns>The masked array, with a redundant all-False mask removed.</returns>
        public NDMaskedArray shrink_mask(object a) => asanyarray(a).shrink_mask();
    }
}
