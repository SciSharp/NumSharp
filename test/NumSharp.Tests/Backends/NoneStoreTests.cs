using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     Python's <c>None</c> (a C# <c>null</c>) stored into numeric arrays through every store route (strings phase 1,
    ///     P1.7b), and into text through the routes the indexers do not reach. NumPy's setitem of <c>None</c> is NaN for a
    ///     float dtype — <c>NPY_NAN</c>, the POSITIVE quiet NaN, compared by bits here because .NET's <see cref="double.NaN"/>
    ///     has the sign bit set — <c>nan+nanj</c> for complex, <c>False</c> for bool, and <c>int(None)</c>'s TypeError for every
    ///     integer dtype. Each expectation, its error text and its ORDER against the route's other checks were probed against
    ///     <c>numpy==2.4.2</c>; the <c>storenone</c> string-oracle section replays the same routes over every dtype.
    /// </summary>
    [TestClass]
    public class NoneStoreTests
    {
        /// <summary>Python's <c>int(None)</c> text, which NumPy's integer setitem raises for <c>None</c>.</summary>
        private const string IntNone = "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'";

        /// <summary>NumPy's <c>NPY_NAN</c> as float64 bits.</summary>
        private const long NpyNan64 = 0x7FF8_0000_0000_0000L;

        /// <summary>The float64 bits of element <paramref name="i"/> (C order).</summary>
        /// <param name="a">A float64 array.</param>
        /// <param name="i">The flat index.</param>
        /// <returns>The raw bits.</returns>
        private static long Bits(NDArray a, long i) => BitConverter.DoubleToInt64Bits((double)a.GetAtIndex(i));

        /// <summary>A fresh float64 array 1.0, 2.0, ... of <paramref name="n"/> elements.</summary>
        /// <param name="n">The length.</param>
        /// <returns>The array.</returns>
        private static NDArray F(int n) => np.arange(1, n + 1).astype(np.float64);

        /// <summary>A fresh int64 array 1, 2, ... of <paramref name="n"/> elements.</summary>
        /// <param name="n">The length.</param>
        /// <returns>The array.</returns>
        private static NDArray I(int n) => np.arange(1, n + 1).astype(np.int64);

        /// <summary>
        ///     The element <c>None</c> stores per dtype (<c>fill(None)</c>): float16 <c>0x7e00</c>, float32 <c>0x7fc00000</c>,
        ///     float64 <c>0x7ff8000000000000</c>, complex both parts that NaN, bool <c>False</c>; every integer dtype and
        ///     <c>char</c> (NumPy's uint16) raise <c>int(None)</c>'s TypeError, <c>decimal</c> (no NaN) Python's
        ///     <c>decimal.Decimal(None)</c> TypeError.
        /// </summary>
        [TestMethod]
        public void Fill_StoresNumPysNoneValue_PerDtype()
        {
            var h = np.ones(new Shape(2), np.float16);
            h.fill(null);
            BitConverter.HalfToUInt16Bits((Half)h.GetAtIndex(1)).Should().Be(0x7E00);
            var f = np.ones(new Shape(2), np.float32);
            f.fill(null);
            BitConverter.SingleToInt32Bits((float)f.GetAtIndex(1)).Should().Be(0x7FC00000);
            var d = F(2);
            d.fill(null);
            Bits(d, 0).Should().Be(NpyNan64);
            var c = np.ones(new Shape(2), np.complex128);
            c.fill(null);
            var z = (Complex)c.GetAtIndex(0);
            BitConverter.DoubleToInt64Bits(z.Real).Should().Be(NpyNan64);
            BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(NpyNan64);
            var b = np.ones(new Shape(2), np.@bool);
            b.fill(null);
            b.ToArray<bool>().Should().Equal(false, false);

            foreach (var dt in new DType[] { np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, NPTypeCode.Char })
            {
                var a = np.ones(new Shape(2), dt);
                Action act = () => a.fill(null);
                act.Should().Throw<TypeError>().WithMessage(IntNone);
            }
            Action dec = () => np.ones(new Shape(2), NPTypeCode.Decimal).fill(null);
            dec.Should().Throw<TypeError>().WithMessage("conversion from NoneType to Decimal is not supported");
        }

        /// <summary>
        ///     The indexers convert <c>None</c> where NumPy packs a value: AFTER an integer index's bounds check
        ///     (<c>a[10] = None</c> is the IndexError for every dtype), BEFORE a fancy index's (<c>a[[10]] = None</c> on an
        ///     integer array is the TypeError, on a float one the IndexError), for an empty slice too (<c>a[5:] = None</c> on an
        ///     integer array raises), and after a mask's shape check.
        /// </summary>
        [TestMethod]
        public void Indexers_ConvertNone_InNumPysOrder()
        {
            Action intOob = () => I(4)[10] = null;
            intOob.Should().Throw<IndexError>().WithMessage("index 10 is out of bounds for axis 0 with size 4");
            Action floatOob = () => F(4)[10] = null;
            floatOob.Should().Throw<IndexError>();

            Action intFancyOob = () => I(4)[np.array(new long[] { 10 })] = null;
            intFancyOob.Should().Throw<TypeError>().WithMessage(IntNone);
            Action floatFancyOob = () => F(4)[np.array(new long[] { 10 })] = null;
            floatFancyOob.Should().Throw<IndexError>();

            Action emptySlice = () => I(4)["5:"] = null;
            emptySlice.Should().Throw<TypeError>().WithMessage(IntNone);

            Action badMask = () => I(4)[np.array(new[] { true, false, true }).MakeGeneric<bool>()] = null;
            badMask.Should().Throw<IndexError>();

            var a = F(4);
            a[np.array(new[] { true, false, true, false }).MakeGeneric<bool>()] = null;
            Bits(a, 0).Should().Be(NpyNan64);
            a.GetDouble(1).Should().Be(2.0);
            Bits(a, 2).Should().Be(NpyNan64);

            var ro = np.broadcast_to(F(1), new Shape(3));
            Action readOnly = () => ro[0] = null;
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
        }

        /// <summary>
        ///     <c>np.full</c> / <c>np.full_like</c> are NumPy's <c>empty</c> + <c>copyto(…, None, casting='unsafe')</c>:
        ///     <c>None</c> is converted per element written, so an EMPTY integer result returns while one element raises; no
        ///     dtype is NumPy's object array, refused; the allocation guard runs before any conversion. This used to return
        ///     ZEROS.
        /// </summary>
        [TestMethod]
        public void Full_And_FullLike_StoreNone()
        {
            var c = np.full(new Shape(2, 2), null, np.complex128);
            foreach (Complex z in c.ToArray<Complex>())
            {
                BitConverter.DoubleToInt64Bits(z.Real).Should().Be(NpyNan64);
                BitConverter.DoubleToInt64Bits(z.Imaginary).Should().Be(NpyNan64);
            }
            np.full(new Shape(0), null, np.int64).size.Should().Be(0);
            Action one = () => np.full(new Shape(1), null, np.int64);
            one.Should().Throw<TypeError>().WithMessage(IntNone);
            Action noDtype = () => np.full(new Shape(2), null);
            noDtype.Should().Throw<NotSupportedException>().WithMessage("np.full with None and no dtype*");
            Action tooBig = () => np.full(new Shape(1L << 62, 1L << 62), null, np.int64);
            tooBig.Should().Throw<ValueError>().WithMessage("array is too big*");

            var like = np.full_like(I(3), null, np.float64);
            Bits(like, 2).Should().Be(NpyNan64);
            Action likeInt = () => np.full_like(F(3), null, np.int64);
            likeInt.Should().Throw<TypeError>().WithMessage(IntNone);
            np.full_like(np.zeros(new Shape(0, 2), np.int64), null).size.Should().Be(0);
        }

        /// <summary>
        ///     <c>np.copyto(dst, None)</c>: NumPy's 0-d OBJECT source, castable only under <c>'unsafe'</c> (every other rule is
        ///     the "Cannot cast scalar from dtype('O')" error, after the read-only check); under <c>'unsafe'</c> a destination of
        ///     more than one element converts <c>None</c> once up front — an integer one raises under any mask and before a
        ///     mask that does not broadcast — while a one-element destination converts only a selected element, and an empty
        ///     one never.
        /// </summary>
        [TestMethod]
        public void Copyto_None_IsAnObjectScalar()
        {
            Action sameKind = () => np.copyto(F(3), (NDArray)null);
            sameKind.Should().Throw<InvalidCastException>()
                .WithMessage("Cannot cast scalar from dtype('O') to dtype('float64') according to the rule 'same_kind'");
            var ro = np.broadcast_to(F(1), new Shape(3));
            Action readOnly = () => np.copyto(ro, (NDArray)null);
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");

            var f = F(4);
            np.copyto(f, (NDArray)null, "unsafe", np.array(new[] { true, false, true, false }));
            Bits(f, 0).Should().Be(NpyNan64);
            f.GetDouble(1).Should().Be(2.0);

            Action intUnsafe = () => np.copyto(I(4), (NDArray)null, "unsafe");
            intUnsafe.Should().Throw<TypeError>().WithMessage(IntNone);
            Action intAllFalse = () => np.copyto(I(4), (NDArray)null, "unsafe", np.zeros(new Shape(4), np.@bool));
            intAllFalse.Should().Throw<TypeError>();
            Action intBadMask = () => np.copyto(I(4), (NDArray)null, "unsafe", np.zeros(new Shape(3), np.@bool));
            intBadMask.Should().Throw<TypeError>();

            var one = I(1);
            np.copyto(one, (NDArray)null, "unsafe", np.array(new[] { false }));
            one.GetInt64(0).Should().Be(1);
            Action oneBadMask = () => np.copyto(I(1), (NDArray)null, "unsafe", np.zeros(new Shape(3), np.@bool));
            oneBadMask.Should().Throw<ValueError>().WithMessage("could not broadcast where mask from shape (3,) into shape (1,)");
            np.copyto(np.zeros(new Shape(0), np.int64), (NDArray)null, "unsafe");

            var tnc = np.array(new[] { "a", "b", "c" }, np.dtypes.StringDType(coerce: false));
            np.copyto(tnc, (NDArray)null, "unsafe", np.zeros(new Shape(3), np.@bool));
            tnc.Storage.GetAtIndex(0).Should().Be("a");
            var u = np.array(new[] { "ab", "cd" }, np.dtype("<U2"));
            np.copyto(u, (NDArray)null, "unsafe");
            u.Storage.GetAtIndex(1).Should().Be("No");
        }

        /// <summary>
        ///     <c>np.put</c> / <c>np.putmask</c> / <c>np.place</c> convert <c>None</c> after their target and mask checks and
        ///     before the indices' bounds and the mask's contents: <c>np.put(int, [6], None)</c> and <c>np.put(int, [], None)</c>
        ///     raise the TypeError, <c>np.put(float, [6], None)</c> the IndexError, an empty target with an index its own
        ///     IndexError; a mask of the wrong size is reported first; an all-false mask still converts.
        /// </summary>
        [TestMethod]
        public void Put_Putmask_Place_ConvertNone_InNumPysOrder()
        {
            var f = F(4);
            np.put(f, np.array(new long[] { 0, 2 }), (NDArray)null);
            Bits(f, 0).Should().Be(NpyNan64);
            Bits(f, 2).Should().Be(NpyNan64);
            Action intOob = () => np.put(I(4), np.array(new long[] { 6 }), (NDArray)null);
            intOob.Should().Throw<TypeError>().WithMessage(IntNone);
            Action intEmpty = () => np.put(I(4), np.array(new long[0]), (NDArray)null);
            intEmpty.Should().Throw<TypeError>();
            Action floatOob = () => np.put(F(4), np.array(new long[] { 6 }), (NDArray)null);
            floatOob.Should().Throw<IndexError>();
            Action emptyTarget = () => np.put(np.zeros(new Shape(0), np.int64), np.array(new long[] { 0 }), (NDArray)null);
            emptyTarget.Should().Throw<IndexError>().WithMessage("cannot replace elements of an empty array");
            var ro = np.broadcast_to(F(1), new Shape(3));
            Action putRo = () => np.put(ro, np.array(new long[] { 0 }), (NDArray)null);
            putRo.Should().Throw<ValueError>().WithMessage("put: output array is read-only");
            var s = F(4);
            np.put(s, 1, (object)null);
            Bits(s, 1).Should().Be(NpyNan64);

            Action badSize = () => np.putmask(I(4), np.array(new[] { true, false }), (NDArray)null);
            badSize.Should().Throw<ValueError>().WithMessage("putmask: mask and data must be the same size");
            Action allFalse = () => np.place(I(4), np.zeros(new Shape(4), np.@bool), (NDArray)null);
            allFalse.Should().Throw<TypeError>().WithMessage(IntNone);
            var p = F(4);
            np.place(p, np.zeros(new Shape(4), np.@bool), (NDArray)null);
            p.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);
            var b = np.ones(new Shape(4), np.@bool);
            np.putmask(b, np.array(new[] { true, false, true, false }), null);
            b.ToArray<bool>().Should().Equal(false, true, false, true);
        }

        /// <summary>
        ///     <c>np.insert</c>: a scalar position is bounds-checked before <c>None</c> is converted (the IndexError wins for an
        ///     integer array, as in NumPy); a sequence of positions converts the values even when it is EMPTY (the fancy
        ///     assignment NumPy runs with the empty index), so <c>np.insert(int, [], None)</c> raises while a float array is
        ///     copied; the inserted elements are NaN.
        /// </summary>
        [TestMethod]
        public void Insert_ConvertsNone_AfterTheIndexCheck()
        {
            Action oob = () => np.insert(I(4), 10, (object)null);
            oob.Should().Throw<Exception>().WithMessage("index 10 is out of bounds for axis 0 with size 4");
            Action emptyInt = () => np.insert(I(4), new int[0], (object)null);
            emptyInt.Should().Throw<TypeError>().WithMessage(IntNone);
            np.insert(F(4), new int[0], (object)null).ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);
            var r = np.insert(F(4), new[] { 1, 3 }, (object)null);
            r.size.Should().Be(6);
            Bits(r, 1).Should().Be(NpyNan64);
            Bits(r, 4).Should().Be(NpyNan64);
            var u = np.insert(np.array(new[] { "ab", "cd" }, np.dtype("<U2")), 1, (object)null);
            u.Storage.GetAtIndex(1).Should().Be("No");
        }

        /// <summary>
        ///     The flat iterator, NumPy's <c>iter_ass_subscript</c>: an integer index rewords any pack failure as
        ///     <c>ValueError: Error setting single item of array.</c>; a slice or an index list converts the value BEFORE its
        ///     positions are resolved and keeps the setitem's own error (<c>a.flat[[9]] = None</c> on an integer array is the
        ///     TypeError, on a float one the IndexError). Text elements are stored by their own setitem, a value and None alike.
        /// </summary>
        [TestMethod]
        public void FlatIterator_StoresNone_AndRewordsOnlyTheIntegerIndex()
        {
            Action intIndex = () => I(4).flatiter[1] = null;
            intIndex.Should().Throw<ValueError>().WithMessage("Error setting single item of array.")
                .WithInnerException<TypeError>();
            Action intSlice = () => I(4).flatiter["1:3"] = null;
            intSlice.Should().Throw<TypeError>().WithMessage(IntNone);
            Action intFancyOob = () => I(4).flatiter[new long[] { 9 }] = null;
            intFancyOob.Should().Throw<TypeError>().WithMessage(IntNone);
            Action floatFancyOob = () => F(4).flatiter[new long[] { 9 }] = null;
            floatFancyOob.Should().Throw<IndexError>();

            var f = F(4);
            f.flatiter["::2"] = null;
            Bits(f, 0).Should().Be(NpyNan64);
            f.GetDouble(1).Should().Be(2.0);
            Bits(f, 2).Should().Be(NpyNan64);

            var u = np.array(new[] { "ab", "cd", "ef" }, np.dtype("<U2"));
            u.flatiter[0] = null;
            u.flatiter[1] = "zzz";
            u.flatiter["2:"] = null;
            u.Storage.GetAtIndex(0).Should().Be("No");
            u.Storage.GetAtIndex(1).Should().Be("zz");
            u.Storage.GetAtIndex(2).Should().Be("No");
            var tnc = np.array(new[] { "a", "b" }, np.dtypes.StringDType(coerce: false));
            Action refused = () => tnc.flatiter[0] = null;
            refused.Should().Throw<ValueError>().WithMessage("Error setting single item of array.");
        }

        /// <summary>
        ///     The element setters <c>SetValue</c> / <c>SetAtIndex</c> / <c>SetData(object)</c> and <c>setfield</c> store
        ///     <c>None</c>'s value too (a float field of an integer array included), and <c>np.asarray</c> /
        ///     <c>np.asanyarray</c> / <c>np.array</c> with a dtype build it, a list's elements packed in C order (the first
        ///     failure wins).
        /// </summary>
        [TestMethod]
        public void ElementSetters_And_Creation_StoreNone()
        {
            var f = F(3);
            f.SetValue(null, 0);
            f.SetAtIndex(null, 1);
            f.SetData((object)null, new[] { 2 });
            Bits(f, 0).Should().Be(NpyNan64);
            Bits(f, 1).Should().Be(NpyNan64);
            Bits(f, 2).Should().Be(NpyNan64);
            Action setValueInt = () => I(2).SetValue(null, 0);
            setValueInt.Should().Throw<TypeError>().WithMessage(IntNone);

            var i8 = np.zeros(new Shape(2), np.int64);
            i8.setfield(null, np.float64, 0);
            i8.ToArray<long>().Should().Equal(NpyNan64, NpyNan64);

            var arr = np.array(new object[] { null, 1.0 }, np.float64);
            Bits(arr, 0).Should().Be(NpyNan64);
            arr.GetDouble(1).Should().Be(1.0);
            var grid = np.array(new object[,] { { null, 1 }, { 2, null } }, np.float32);
            grid.shape.Should().Equal(2L, 2L);
            float.IsNaN(grid.GetSingle(0, 0)).Should().BeTrue();
            grid.GetSingle(0, 1).Should().Be(1f);
            Action intList = () => np.array(new object[] { 1, null }, np.int8);
            intList.Should().Throw<TypeError>().WithMessage(IntNone);
            Action firstWins = () => np.array(new object[] { 1e300, null }, np.int8);
            firstWins.Should().Throw<OverflowException>();

            var c = np.asarray((NDArray)null, np.complex128);
            BitConverter.DoubleToInt64Bits(((Complex)c.GetAtIndex(0)).Imaginary).Should().Be(NpyNan64);
            var text = np.asanyarray(null, np.dtype("U"));
            text.dtype.ToString().Should().Be("<U4");
            text.Storage.GetAtIndex(0).Should().Be("None");
        }

        /// <summary>
        ///     A masked array's setitem stores <c>None</c> into its data (NaN) and UNMASKS the slot, as for any plain value
        ///     (NumPy's <c>getmask(None)</c> is nomask); a hard mask keeps the masked slot; an integer masked array raises
        ///     <c>int(None)</c>'s TypeError.
        /// </summary>
        [TestMethod]
        public void MaskedArray_SetItem_StoresNoneAndUnmasks()
        {
            var m = np.ma.array(F(3), mask: np.array(new[] { true, false, false }));
            m[0] = null;
            Bits(m.data, 0).Should().Be(NpyNan64);
            m.mask.ToArray<bool>().Should().Equal(false, false, false);

            var h = np.ma.array(F(3), mask: np.array(new[] { true, false, false })).harden_mask();
            h[0] = null;
            h[1] = null;
            h.data.GetDouble(0).Should().Be(1.0);
            Bits(h.data, 1).Should().Be(NpyNan64);
            h.mask.ToArray<bool>().Should().Equal(true, false, false);

            var mi = np.ma.array(I(3), mask: np.array(new[] { true, false, false }));
            Action intStore = () => mi[0] = null;
            intStore.Should().Throw<TypeError>().WithMessage(IntNone);
        }
    }
}
