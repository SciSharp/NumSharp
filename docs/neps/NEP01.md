# NEP 01 - A Simple File Format for NumPy Arrays (.npy)

**Status:** Final
**NumSharp Impact:** HIGH - NumSharp implements `np.save`/`np.load`

## Summary

Defines the `.npy` binary file format for persisting single NumPy arrays to disk, preserving shape and dtype information.

## NumSharp Relevance

NumSharp implements this format in `np.save.cs` and `np.load.cs`. Must match the specification exactly for interoperability.

## File Format Specification

### Magic Number and Version
```
Bytes 0-5:     Magic string "\x93NUMPY" (6 bytes)
Byte 6:        Major version (0x01 or 0x02)
Byte 7:        Minor version (0x00)
```

### Header Length
- **Version 1.0:** Bytes 8-9 = little-endian unsigned short (max 65535)
- **Version 2.0:** Bytes 8-11 = little-endian unsigned int (max 4 GiB)

### Header Format
ASCII string containing a Python dictionary literal, padded with spaces to make total header divisible by 16, terminated with newline.

```python
{
    "descr": "<f8",           # dtype descriptor
    "fortran_order": False,   # True if Fortran-contiguous
    "shape": (100, 200)       # dimensions tuple
}
```

### Data Section
- Non-object arrays: Contiguous bytes (C or Fortran order)
- Size: `product(shape) * dtype.itemsize` bytes

## Version Selection

NumPy automatically uses:
- **1.0** for headers ≤ 65535 bytes (most cases)
- **2.0** when header exceeds 1.0 limits

## NumSharp Implementation Checklist

- [ ] Read both version 1.0 and 2.0 formats
- [ ] Write version 1.0 by default, 2.0 when needed
- [ ] Parse Python dict literal in header
- [ ] Handle endianness correctly (`<` = little, `>` = big)
- [ ] Support all NumSharp dtypes
- [ ] Handle Fortran-order arrays (convert to C-order or preserve)

## Related Files

- `src/NumSharp.Core/APIs/np.save.cs`
- `src/NumSharp.Core/APIs/np.load.cs`

## References

- [NEP 1 Full Text](https://numpy.org/neps/nep-0001-npy-format.html)
- [NumPy format.py](https://github.com/numpy/numpy/blob/main/numpy/lib/format.py)
