"""
ma_setops_reference.py — numpy.ma's arraysetops (numpy/ma/extras.py: unique / in1d / isin / intersect1d / setxor1d /
union1d / setdiff1d) copied statement for statement from NumPy 2.4.2, with every sort made STABLE. Used by
gen_ma_oracle.py's `setops` tier to decide which cases the corpus can hold.

NumPy's 'quicksort' (np.unique's argsort with return_inverse, MaskedArray.sort's default kind, ndarray.sort) is not
stable: 32/64-bit dtypes go through x86-simd-sort's AVX2 / AVX-512 networks, the others through introsort past 16
elements, so its order among EQUAL keys is host-dependent. NumSharp's sort is stable. The two agree on every result
that does not depend on that order; it can depend on it only when an unstable sort meets a group of equal keys that are
not interchangeable for the observable result (filled(0) data + mask):

  * a masked slot (keyed by the fill value: NaN for a floating dtype, the dtype maximum otherwise, True for bool,
    inf+infj for complex) beside an unmasked value equal to that key, or
  * two unmasked values that compare equal with different bits (+0.0 / -0.0, NaN payloads, complex NaNs), or
  * the NaN collapse's searchsorted reading the RAW data of masked slots whose NaN-ness differs.

Every unstable sort here records such a group in SENSITIVE. The generator keeps a case only when NumPy's observable
answer equals this reference's (the answer NumSharp's stable sort gives), and asserts that every case it drops was
SENSITIVE — a dropped case that was not would mean this reference is not NumPy's statements.
"""
import numpy as np
import numpy.ma as ma
from numpy.ma.core import MaskedArray

SENSITIVE = False


def _lt(x, y):
    """NumPy's sort comparator (npysort's Tag::less) elementwise over two equal-length arrays."""
    k = x.dtype.kind
    if k == "f":
        return (x < y) | ((y != y) & (x == x))
    if k == "c":
        xr, xi, yr, yi = x.real, x.imag, y.real, y.imag
        r1 = (xi == xi) | (yi != yi)
        r2 = (yi != yi) & (xi == xi)
        r3 = (xi < yi) | ((yi != yi) & (xi == xi))
        r4 = yr != yr
        same = (xr == yr) | ((xr != xr) & (yr != yr))
        return np.where(xr < yr, r1, np.where(xr > yr, r2, np.where(same, r3, r4)))
    return x < y


def _unstable_argsort(keys, mask):
    """NumPy's quicksort argsort of 1-D KEYS, done stably; marks SENSITIVE when a group of equal keys mixes members
    that are not interchangeable (MASK: True where the slot is masked, or None)."""
    global SENSITIVE
    perm = keys.argsort(kind="stable")
    s = keys[perm]
    if s.shape[0] > 1:
        tied = ~_lt(s[:-1], s[1:])
        m = np.zeros(s.shape, bool) if mask is None else np.asarray(mask)[perm]
        raw = np.ascontiguousarray(s).view(np.uint8).reshape(s.shape[0], -1)
        same_bits = (raw[:-1] == raw[1:]).all(axis=1)
        # Interchangeable adjacent members: both masked, or both unmasked with the same bits. Interchangeability is
        # an equivalence within a tie group, so checking adjacent pairs covers the group.
        bad = tied & ((m[:-1] != m[1:]) | (~m[:-1] & ~m[1:] & ~same_bits))
        if bad.any():
            SENSITIVE = True
    return perm


def _argsort(a, kind):
    """MaskedArray.argsort / ndarray.argsort (endwith=True, default axis); 'mergesort' is stable in NumPy too."""
    if isinstance(a, MaskedArray):
        axis = -1 if a.ndim <= 1 else None
        fill = np.nan if np.issubdtype(a.dtype, np.floating) else ma.minimum_fill_value(a)
        filled = a.filled(fill)
        if kind == "mergesort":
            return filled.argsort(axis=axis, kind="stable")
        return _unstable_argsort(np.asarray(filled).ravel(), ma.getmaskarray(a).ravel())
    if kind == "mergesort":
        return a.argsort(kind="stable")
    return _unstable_argsort(np.asarray(a).ravel(), None)


def _sort_inplace(ar):
    """MaskedArray.sort() / ndarray.sort(): axis -1, the default (unstable) kind, endwith=True."""
    if isinstance(ar, MaskedArray):
        if ar._mask is ma.nomask:
            d = ar.data
            if d.ndim == 1:
                d[...] = d[_unstable_argsort(d, None)]
            else:
                for idx in np.ndindex(d.shape[:-1]):
                    row = d[idx]
                    row[...] = row[_unstable_argsort(row.copy(), None)]
            return
        if ar is ma.masked:
            return
        if ar.ndim == 1:
            sidx = _argsort(ar, "quicksort")
        else:
            fill = np.nan if np.issubdtype(ar.dtype, np.floating) else ma.minimum_fill_value(ar)
            filled = np.asarray(ar.filled(fill))
            mk = ma.getmaskarray(ar)
            sidx = np.empty(ar.shape, dtype=np.intp)
            for idx in np.ndindex(ar.shape[:-1]):
                sidx[idx] = _unstable_argsort(filled[idx].copy(), mk[idx])
        ar[...] = np.take_along_axis(ar, sidx, axis=-1)
    else:
        ar[...] = ar[_unstable_argsort(np.asarray(ar).ravel(), None)]


def _unique1d(ar, return_index=False, return_inverse=False, inverse_shape=None):
    """numpy.lib._arraysetops_impl._unique1d (equal_nan=True, axis=None, sorted=True)."""
    global SENSITIVE
    ar = np.asanyarray(ar).flatten()
    optional_indices = return_index or return_inverse
    if not optional_indices and not np.ma.is_masked(ar):
        # NumPy's hash path (integer / complex / string dtypes): its own C++ hash dedup keeps the FIRST element of each
        # equality class, then sorts the distinct values (no ties left).
        from numpy.lib._arraysetops_impl import _unique_hash
        from numpy._core._multiarray_umath import _array_converter
        conv = _array_converter(ar)
        ar_, = conv
        hash_unique = _unique_hash(ar_, equal_nan=True)
        if hash_unique is not NotImplemented:
            hash_unique.sort()
            return (conv.wrap(hash_unique),)
    if optional_indices:
        perm = _argsort(ar, "mergesort" if return_index else "quicksort")
        aux = ar[perm]
    else:
        _sort_inplace(ar)
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True
    if (aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1])):
        # The searchsorted below reads the RAW data under masked slots, whose order is the unstable sort's.
        if isinstance(aux, MaskedArray) and ma.is_masked(aux):
            rawnan = np.isnan(aux.data)[ma.getmaskarray(aux)]
            if rawnan.any() and not rawnan.all():
                SENSITIVE = True
        if aux.dtype.kind == "c":
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = (aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]
    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx.reshape(inverse_shape),)
    return ret


def _np_unique(ar, return_index=False, return_inverse=False):
    """np.unique(ar, return_index, return_inverse) for axis=None."""
    ar = np.asanyarray(ar)
    ret = _unique1d(ar, return_index, return_inverse, inverse_shape=ar.shape)
    return ret[0] if len(ret) == 1 else ret


def unique(ar1, return_index=False, return_inverse=False):
    output = _np_unique(ar1, return_index=return_index, return_inverse=return_inverse)
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        output = output.view(MaskedArray)
    return output


def in1d(ar1, ar2, assume_unique=False, invert=False):
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)
    ar = ma.concatenate((ar1, ar2))
    order = _argsort(ar, "mergesort")
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = ma.concatenate((bool_ar, [invert]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]
    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):
    element = ma.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)


def intersect1d(ar1, ar2, assume_unique=False):
    if assume_unique:
        aux = ma.concatenate((ar1, ar2))
    else:
        aux = ma.concatenate((unique(ar1), unique(ar2)))
    _sort_inplace(aux)
    return aux[:-1][aux[1:] == aux[:-1]]


def setxor1d(ar1, ar2, assume_unique=False):
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    aux = ma.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux
    _sort_inplace(aux)
    auxf = aux.filled()
    flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))
    flag2 = (flag[1:] == flag[:-1])
    return aux[flag2]


def union1d(ar1, ar2):
    return unique(ma.concatenate((ar1, ar2), axis=None))


def setdiff1d(ar1, ar2, assume_unique=False):
    if assume_unique:
        ar1 = ma.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


def run(fn, *args, **kw):
    """Runs FN on this reference: (result or the exception it raised, whether an unstable sort met a sensitive tie)."""
    global SENSITIVE
    SENSITIVE = False
    try:
        r = fn(*args, **kw)
    except Exception as e:  # noqa: BLE001 — the generator compares NumPy's exceptions too
        r = e
    return r, SENSITIVE
