using System;
using System.Numerics;
using System.Text;

namespace NumSharp.Tests.Indexing;

/// <summary>
///     Storing a C# literal — NumPy's Python <c>int</c> / <c>float</c> / <c>complex</c> — into an array packs it the way
///     NumPy's <c>PyArray_Pack</c> does (range-checked, never wrapped), at every store entry point: the indexer setters,
///     <c>ndarray.fill</c>, <c>np.full</c> / <c>np.full_like</c>, <c>np.put</c>, <c>np.putmask</c>, <c>np.place</c>,
///     <c>np.put_along_axis</c>, <c>np.insert</c>, <c>np.fill_diagonal</c>, <c>np.pad</c>, the flat iterator and
///     <c>np.asarray</c> with a dtype. A 0-d ARRAY of the same value is cast instead, as in NumPy. Every expected value
///     and text was probed against NumPy 2.4.2 (LP64 texts past the C <c>long</c>, the house pin).
/// </summary>
[TestClass]
public class StorePythonLiteralTests
{
    /// <summary>The little-endian bytes of <paramref name="a"/> in C order, as NumPy's <c>a.tobytes().hex()</c>.</summary>
    /// <param name="a">The array.</param>
    /// <returns>The hex text.</returns>
    private static string Hex(NDArray a)
    {
        var c = np.ascontiguousarray(a);
        var sb = new StringBuilder();
        foreach (var b in c.Unsafe.ReadOnlyBytes())
            sb.Append(b.ToString("x2"));
        return sb.ToString();
    }

    /// <summary>Asserts that <paramref name="act"/> raises exactly <typeparamref name="T"/> with NumPy's text.</summary>
    /// <typeparam name="T">The exception type.</typeparam>
    /// <param name="act">The store.</param>
    /// <param name="message">NumPy's message.</param>
    private static void Raises<T>(Action act, string message) where T : Exception
    {
        var ex = Assert.ThrowsException<T>(act);
        Assert.AreEqual(message, ex.Message);
    }

    private static NDArray Z(NPTypeCode t, int n = 3) => np.zeros(new Shape(n), t);

    private static NDArray Mask() => np.array(new[] { true, false, true });

    /// <summary>
    ///     The implicit conversion of an integer / float / complex literal marks its 0-d array as NumPy's
    ///     <c>NPY_ARRAY_WAS_PYTHON_LITERAL</c> temporary; bool, Half, char, decimal, an explicit
    ///     <see cref="NDArray.Scalar{T}(T)"/> and anything derived from a literal are not marked.
    /// </summary>
    [TestMethod]
    public void ImplicitConversion_MarksPythonLiterals_Only()
    {
        NDArray i = 1000, l = 1000L, u = 9223372036854775808UL, d = 1.5, f = 1.5f, c = new Complex(3, 4);
        foreach (var marked in new[] { i, l, u, d, f, c })
            Assert.IsTrue(marked.WasPythonLiteral);

        NDArray b = true, h = (Half)1, ch = 'x', m = 1m;
        foreach (var unmarked in new[] { b, h, ch, m, NDArray.Scalar(1000), np.array(1000), i.reshape(1), i.copy() })
            Assert.IsFalse(unmarked.WasPythonLiteral);
    }

    /// <summary>
    ///     Every indexer route range-checks a Python int: integer, slice, ellipsis, fancy and boolean-mask assignment, on
    ///     1-D and 2-D arrays. A 0-d ARRAY of 1000 is cast (-24), as NumPy casts <c>np.array(1000)</c>.
    /// </summary>
    [TestMethod]
    public void Setters_PythonInt_OutOfRange_RaisesNumPysOverflowError()
    {
        const string msg = "Python integer 1000 out of bounds for int8";
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[0] = 1000, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[":"] = 1000, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)["..."] = 1000, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[new[] { 0, 2 }] = 1000, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[Mask()] = 1000, msg);
        Raises<OverflowException>(() => np.zeros(new Shape(2, 2), NPTypeCode.SByte)[0] = 1000, msg);
        Raises<OverflowException>(() => np.zeros(new Shape(2, 2), NPTypeCode.SByte)[":, 1"] = 1000, msg);
        Raises<OverflowException>(() => np.zeros(new Shape(2, 2), NPTypeCode.SByte)[np.array(new[] { 0, 1 }), np.array(new[] { 1, 0 })] = 1000, msg);
        // A float literal is truncated first (int(1000.5) is 1000), then range-checked.
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[Mask()] = 1000.5, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.Byte)[0] = -1, "Python integer -1 out of bounds for uint8");
        Raises<OverflowException>(() => Z(NPTypeCode.UInt64)[0] = -1, "Python integer -1 out of bounds for uint64");

        // The strong 0-d array of the same value is cast, as in NumPy.
        var a = Z(NPTypeCode.SByte);
        a[0] = NDArray.Scalar(1000);
        Hex(a).Should().Be("e80000");
    }

    /// <summary>
    ///     NumPy packs at different points per index kind: after an integer index's bounds check
    ///     (<c>a[5] = 1000</c> is the IndexError), but before a fancy index's bounds are read
    ///     (<c>a[[5]] = 1000</c> is the OverflowError) and for an empty selection too (<c>a[5:] = 1000</c> raises).
    /// </summary>
    [TestMethod]
    public void Setters_PackOrder_FollowsNumPysIndexKinds()
    {
        Assert.ThrowsException<IndexError>(() => Z(NPTypeCode.SByte)[5] = 1000);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[new[] { 5 }] = 1000, "Python integer 1000 out of bounds for int8");
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)["5:"] = 1000, "Python integer 1000 out of bounds for int8");
    }

    /// <summary>
    ///     The texts past the C <c>long</c> (LP64, the house pin), NaN / infinity into an integer, a complex into a real
    ///     dtype, and the bool / complex destinations.
    /// </summary>
    [TestMethod]
    public void Setters_PythonScalarKinds_FollowPyArrayPack()
    {
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[0] = 9223372036854775808UL, "Python int too large to convert to C long");
        Raises<OverflowException>(() => Z(NPTypeCode.Int64)[0] = 9223372036854775808UL, "Python int too large to convert to C long");
        Raises<OverflowException>(() => Z(NPTypeCode.UInt32)[0] = 9223372036854775808UL, "Python integer 9223372036854775808 out of bounds for uint32");
        Raises<ValueError>(() => Z(NPTypeCode.Int32)[0] = double.NaN, "cannot convert float NaN to integer");
        Raises<OverflowException>(() => Z(NPTypeCode.Int32)[0] = double.PositiveInfinity, "cannot convert float infinity to integer");
        Raises<TypeError>(() => Z(NPTypeCode.Double)[0] = new Complex(3, 4), "float() argument must be a string or a real number, not 'complex'");
        Raises<TypeError>(() => Z(NPTypeCode.SByte)[0] = new Complex(3, 4), "int() argument must be a string, a bytes-like object or a real number, not 'complex'");

        var b = Z(NPTypeCode.Boolean);
        b[0] = 2.5;
        b[1] = double.NaN;
        Hex(b).Should().Be("010100");
        var b1 = np.ones(new Shape(3), NPTypeCode.Boolean);
        b1[0] = 0.0;
        Hex(b1).Should().Be("000101");

        var c = Z(NPTypeCode.Complex);
        c[0] = 1000;
        Hex(c).Should().Be("0000000000408f40" + "0000000000000000" + new string('0', 64));
    }

    /// <summary>
    ///     A Python int past 2^53 reaches float32 / float16 through the correctly rounded double, so it rounds TWICE as in
    ///     NumPy: 2^60 + 2^36 + 1 is 0x5d800000 in float32 (one rounding would give 0x5d800001); 2049 rounds to 2048 and
    ///     65520 overflows to inf in float16.
    /// </summary>
    [TestMethod]
    public void Setters_PythonIntIntoNarrowFloats_RoundsTwice()
    {
        var f = Z(NPTypeCode.Single);
        f[0] = 1152921573326323713L;
        Hex(f).Should().Be("0000805d" + "00000000" + "00000000");

        var h = Z(NPTypeCode.Half);
        h[0] = 2049;
        h[1] = 65520;
        Hex(h).Should().Be("0068" + "007c" + "0000");
    }

    /// <summary>
    ///     <c>ndarray.fill</c> is one <c>PyArray_Pack</c>: the LP64 texts, two roundings into float32, and the NumPy-scalar
    ///     rule for a <see cref="Half"/> (range-checked into a SIGNED integer, cast into every other: 300 into uint8 is
    ///     44). A strong 0-d array is cast; a C# literal's 0-d array packs as the literal.
    /// </summary>
    [TestMethod]
    public void Fill_IsPyArrayPack()
    {
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill(1000), "Python integer 1000 out of bounds for int8");
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill(9223372036854775808UL), "Python int too large to convert to C long");
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill(1e300), "Python int too large to convert to C long");
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill((Half)300), "Python integer 300 out of bounds for int8");

        var f = Z(NPTypeCode.Single);
        f.fill(1152921573326323713L);
        Hex(f).Should().Be("0000805d0000805d0000805d");

        var u8 = Z(NPTypeCode.Byte);
        u8.fill((Half)300);
        Hex(u8).Should().Be("2c2c2c");

        var strong = Z(NPTypeCode.SByte);
        strong.fill(NDArray.Scalar(1000));
        Hex(strong).Should().Be("e8e8e8");

        NDArray literal = 1000;
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill(literal), "Python integer 1000 out of bounds for int8");
    }

    /// <summary>
    ///     <c>np.full</c> / <c>np.full_like</c> are <c>copyto(a, fill_value, casting='unsafe')</c>: a Python int is still
    ///     range-checked in the result's dtype, while a Python float or complex is stored as float64 / complex128 and then
    ///     cast unsafely (1e300 into int8 is 0, 3+4j is 3, 1.5 is 1).
    /// </summary>
    [TestMethod]
    public void Full_IsCopytoUnsafe()
    {
        Raises<OverflowException>(() => np.full(new Shape(3), 1000, np.int8), "Python integer 1000 out of bounds for int8");
        Raises<OverflowException>(() => np.full(new Shape(3), -1, np.uint8), "Python integer -1 out of bounds for uint8");
        Raises<OverflowException>(() => np.full_like(Z(NPTypeCode.SByte), 1000), "Python integer 1000 out of bounds for int8");
        Hex(np.full(new Shape(3), 1.5, np.int8)).Should().Be("010101");
        Hex(np.full(new Shape(3), 1e300, np.int8)).Should().Be("000000");
        Hex(np.full(new Shape(3), new Complex(3, 4), np.int8)).Should().Be("030303");
        Hex(np.full(new Shape(3), 1152921573326323713L, np.float32)).Should().Be("0000805d0000805d0000805d");
    }

    /// <summary>
    ///     <c>np.put</c> converts its values (<c>PyArray_FromAny</c>, FORCECAST) before it reads an index — so even an
    ///     empty index list raises for an out-of-range literal — and an empty values array is a no-op.
    /// </summary>
    [TestMethod]
    public void Put_PacksTheLiteral_BeforeTheIndices()
    {
        const string msg = "Python integer 1000 out of bounds for int8";
        Raises<OverflowException>(() => np.put(Z(NPTypeCode.SByte), np.array(new long[] { 0 }), 1000), msg);
        Raises<OverflowException>(() => np.put(Z(NPTypeCode.SByte), np.array(new long[0]), 1000), msg);
        Raises<OverflowException>(() => np.put(Z(NPTypeCode.SByte), 0L, (object)1000), msg);

        var a = Z(NPTypeCode.SByte);
        np.put(a, np.array(new long[] { 0 }), NDArray.Scalar(1000));
        Hex(a).Should().Be("e80000");
    }

    /// <summary>
    ///     <c>np.putmask</c>, <c>np.place</c> and <c>np.put_along_axis</c> pack a literal (a float is truncated:
    ///     <c>place(int8, m, 1.5)</c> stores 1).
    /// </summary>
    [TestMethod]
    public void MaskedAndAlongAxisStores_PackTheLiteral()
    {
        const string msg = "Python integer 1000 out of bounds for int8";
        Raises<OverflowException>(() => np.putmask(Z(NPTypeCode.SByte), Mask(), 1000), msg);
        Raises<OverflowException>(() => np.putmask(Z(NPTypeCode.Byte), Mask(), -1), "Python integer -1 out of bounds for uint8");
        Raises<OverflowException>(() => np.place(Z(NPTypeCode.SByte), Mask(), 1000), msg);
        Raises<OverflowException>(() => np.put_along_axis(Z(NPTypeCode.SByte), np.array(new long[] { 0 }), 1000, 0), msg);

        var p = Z(NPTypeCode.SByte);
        np.place(p, Mask(), 1.5);
        Hex(p).Should().Be("010001");
    }

    /// <summary>
    ///     <c>np.insert</c> (<c>array(values, dtype=arr.dtype)</c>) and <c>np.fill_diagonal</c> (<c>a.flat[...] = val</c>)
    ///     pack the literal, through their NDArray and object overloads; fill_diagonal packs even for an empty array.
    /// </summary>
    [TestMethod]
    public void InsertAndFillDiagonal_PackTheLiteral()
    {
        const string msg = "Python integer 1000 out of bounds for int8";
        Raises<OverflowException>(() => np.insert(Z(NPTypeCode.SByte), 1, 1000), msg);
        Raises<OverflowException>(() => np.insert(Z(NPTypeCode.SByte), 1, (object)1000), msg);
        Raises<OverflowException>(() => np.insert(Z(NPTypeCode.SByte), new[] { 1, 2 }, 1000), msg);
        Raises<OverflowException>(() => np.fill_diagonal(np.zeros(new Shape(2, 2), NPTypeCode.SByte), 1000), msg);
        Raises<OverflowException>(() => np.fill_diagonal(np.zeros(new Shape(0, 0), NPTypeCode.SByte), 1000), msg);
        Hex(np.insert(Z(NPTypeCode.SByte), 1, 1.5)).Should().Be("00010000");
    }

    /// <summary>
    ///     <c>np.pad</c> stores each constant as a NumPy SCALAR (<c>_as_pairs</c> runs <c>np.array(constant_values)</c>):
    ///     int()-converted with the range check into a SIGNED integer array, CAST into every other — so int8 refuses 1000
    ///     where uint8 pads with 232, and a complex constant keeps its real part.
    /// </summary>
    [TestMethod]
    public void Pad_StoresNumPyScalars()
    {
        Raises<OverflowException>(() => np.pad(Z(NPTypeCode.SByte), 1, constant_values: 1000), "Python integer 1000 out of bounds for int8");
        Raises<ValueError>(() => np.pad(Z(NPTypeCode.SByte), 1, constant_values: double.NaN), "cannot convert float NaN to integer");
        Hex(np.pad(Z(NPTypeCode.Byte), 1, constant_values: 1000)).Should().Be("e8000000e8");
        Hex(np.pad(Z(NPTypeCode.SByte), 1, constant_values: new Complex(3, 4))).Should().Be("0300000003");
        Hex(np.pad(Z(NPTypeCode.SByte), 1, constant_values: new object[] { 1, 2.5 })).Should().Be("0100000002");
    }

    /// <summary>
    ///     The flat iterator: a slice assignment packs the literal with NumPy's own error (even an empty slice); a
    ///     single-index assignment reports every packing failure as "Error setting single item of array.".
    /// </summary>
    [TestMethod]
    public void FlatIterator_PacksTheLiteral()
    {
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).flatiter["0:2"] = 1000, "Python integer 1000 out of bounds for int8");
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).flatiter["0:0"] = 1000, "Python integer 1000 out of bounds for int8");
        var ex = Assert.ThrowsException<ValueError>(() => Z(NPTypeCode.SByte).flatiter[0] = double.NaN);
        Assert.AreEqual("Error setting single item of array.", ex.Message);
        Assert.IsInstanceOfType(ex.InnerException, typeof(ValueError));
    }

    /// <summary>
    ///     A C# <see cref="Half"/> / <c>char</c> value is a NumPy SCALAR (<c>np.float16</c> / <c>np.uint16</c>), which NumPy
    ///     stores two ways: the <c>PyArray_Pack</c> routes (an integer index, a slice view, <c>fill</c>, the flat iterator's
    ///     single index) int()-convert it with the range check into a SIGNED integer; the <c>PyArray_FromAny</c> routes
    ///     (fancy and mask assignment, <c>np.put</c>, <c>np.insert</c>, <c>np.full</c>, <c>np.asarray</c>, the flat
    ///     iterator's slices) cast it like a 0-d array (300 → 44). An unsigned destination casts on every route. Probed
    ///     against NumPy 2.4.2.
    /// </summary>
    [TestMethod]
    public void NumPyScalar_PacksOnPyArrayPackRoutes_CastsOnFromAnyRoutes()
    {
        const string msg = "Python integer 300 out of bounds for int8";
        var h = (Half)300;
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[0] = h, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[":"] = h, msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte).fill(h), msg);
        Raises<OverflowException>(() => Z(NPTypeCode.SByte)[0] = (char)300, msg);
        Raises<ValueError>(() => Z(NPTypeCode.SByte)[0] = Half.NaN, "cannot convert float NaN to integer");
        Assert.ThrowsException<ValueError>(() => Z(NPTypeCode.SByte).flatiter[0] = h);

        NDArray Set(Action<NDArray> store)
        {
            var a = Z(NPTypeCode.SByte);
            store(a);
            return a;
        }

        Hex(Set(a => a[new[] { 0, 2 }] = h)).Should().Be("2c002c");
        Hex(Set(a => a[Mask()] = h)).Should().Be("2c002c");
        Hex(Set(a => np.put(a, np.array(new long[] { 0 }), h))).Should().Be("2c0000");
        Hex(Set(a => np.putmask(a, Mask(), h))).Should().Be("2c002c");
        Hex(Set(a => a.flatiter["0:2"] = h)).Should().Be("2c2c00");
        Hex(np.insert(Z(NPTypeCode.SByte), 1, h)).Should().Be("002c0000");
        Hex(np.full(new Shape(3), h, np.int8)).Should().Be("2c2c2c");
        Hex(np.asarray(h, np.int8)).Should().Be("2c");

        // An unsigned destination casts on every route.
        var u = Z(NPTypeCode.Byte);
        u[0] = h;
        u[1] = (char)300;
        Hex(u).Should().Be("2c2c00");
    }

    /// <summary>
    ///     <c>np.asarray(literal, dtype)</c> packs the Python scalar into a new array; a 0-d ARRAY is cast.
    /// </summary>
    [TestMethod]
    public void Asarray_LiteralWithDtype_Packs()
    {
        Raises<OverflowException>(() => np.asarray(1000, np.int8), "Python integer 1000 out of bounds for int8");
        Raises<TypeError>(() => np.asarray(new Complex(3, 4), np.float64), "float() argument must be a string or a real number, not 'complex'");
        Hex(np.asarray(1.5, np.int8)).Should().Be("01");
        Hex(np.asarray(1152921573326323713L, np.float32)).Should().Be("0000805d");
        Hex(np.asarray(NDArray.Scalar(1000), np.int8)).Should().Be("e8");
    }
}
