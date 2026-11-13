using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 7 (the owner approved all 15 picks): every expected value, dtype and text is NumPy 2.4.2's
    ///     (scratchpad batch7_numpy_evidence*.txt), for the call exactly as the C# fix makes it.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch7Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        // ---- fix 1: np.reshape(a, shape, order='C', copy=None) --------------------------------------------------------

        /// <summary>np.reshape's one signature: NumPy's order (F values, 'A', the 'K' refusal) and copy modes (None /
        ///     True / False over C, F and transposed sources), the size text before the copy check, NumPy's names and
        ///     every shape spelling.</summary>
        [TestMethod]
        public void Reshape_OneSignature_OrderAndCopy()
        {
            // NumPy: np.reshape(arange(6), (2, 3), order='F') is [[0, 2, 4], [1, 3, 5]], F-contiguous.
            var a = np.arange(6L);
            var f = np.reshape(a, (2, 3), 'F');
            f.ToArray<long>().Should().Equal(0L, 2L, 4L, 1L, 3L, 5L);
            f.Shape.IsFContiguous.Should().BeTrue();
            np.reshape(a, (2, 3), order: 'A').Shape.IsContiguous.Should().BeTrue();
            Throws<ValueError>(() => np.reshape(a, (2, 3), order: 'K'), "order 'K' is not permitted for reshaping");

            // copy=None / False on a C-contiguous source: a view; copy=True: a copy sharing nothing.
            np.shares_memory(np.reshape(a, (2, 3)), a).Should().BeTrue();
            np.shares_memory(np.reshape(a, (2, 3), copy: false), a).Should().BeTrue();
            var copied = np.reshape(a, (2, 3), copy: true);
            np.shares_memory(copied, a).Should().BeFalse();
            copied.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            // The same shape still copies under copy=True (NumPy skips its same-dims shortcut).
            np.shares_memory(np.reshape(a, 6, copy: true), a).Should().BeFalse();

            // A transposed source: None copies ([0, 3, 1, 4, 2, 5]); False refuses with NumPy's text.
            var t = np.arange(6L).reshape(2, 3).T;
            np.reshape(t, 6).ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            Throws<ValueError>(() => np.reshape(t, 6, copy: false), "Unable to avoid creating a copy while reshaping.");
            // ...but in F order the transposed source is contiguous: a view.
            np.shares_memory(np.reshape(t, 6, order: 'F', copy: false), t).Should().BeTrue();

            // An F-contiguous source: C order cannot avoid the copy, F order can.
            var fsrc = np.asfortranarray(np.arange(6L).reshape(2, 3));
            Throws<ValueError>(() => np.reshape(fsrc, 6, copy: false), "Unable to avoid creating a copy while reshaping.");
            np.shares_memory(np.reshape(fsrc, 6, order: 'F', copy: false), fsrc).Should().BeTrue();
            // copy=True in F order: an F-contiguous copy, NumPy's [[0, 4], [3, 2], [1, 5]].
            var fcopy = np.reshape(fsrc, (3, 2), order: 'F', copy: true);
            fcopy.Shape.IsFContiguous.Should().BeTrue();
            fcopy.ToArray<long>().Should().Equal(0L, 4L, 3L, 2L, 1L, 5L);

            // Size errors come first, whatever copy says.
            // NumPy's text, under NumSharp's size-mismatch type (NumPy raises it as a ValueError).
            Throws<IncorrectShapeException>(() => np.reshape(a, (4, 2), copy: false), "cannot reshape array of size 6 into shape (4,2)");
            // NumPy's names (a, shape) and every spelling of the shape bind the one signature.
            np.reshape(a: a, shape: (3, 2)).shape.Should().Equal(3L, 2L);
            np.reshape(a, new[] { 2, 3 }).shape.Should().Equal(2L, 3L);
            np.reshape(a, new long[] { 3, -1 }, 'F').ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            typeof(np).GetMethods().Count(m => m.Name == "reshape").Should().Be(1);
        }

        /// <summary>NumPy's np.reshape calls the operand's own reshape, so a numpy.matrix stays a matrix in every
        ///     copy mode (the woven [NewSubtype] method path).</summary>
        [TestMethod]
        public void Reshape_SubclassOperands_KeepTheirClass_InEveryCopyMode()
        {
            // NumPy's np.reshape calls the operand's own reshape: a matrix stays a matrix, copy or not.
            var m = np.asmatrix(np.arange(6.0).reshape(2, 3));
            np.reshape(m, (3, 2)).Should().BeOfType<NDMatrix>();
            np.reshape(m, (3, 2), copy: true).Should().BeOfType<NDMatrix>();
            np.reshape(m, (3, 2), copy: false).Should().BeOfType<NDMatrix>();
            ((NDMatrix)m).reshape(new Shape(3, 2), 'C', copy: true).Should().BeOfType<NDMatrix>();
        }

        // ---- fix 2: np.astype(x, dtype, copy=True, device=None) --------------------------------------------------------

        /// <summary>np.astype is x.astype(dtype, copy) after NumPy's input and device checks: the cast, copy=False's
        ///     identity, None as float64, a dtype string, unsafe truncation, the two error texts and a matrix
        ///     operand's class.</summary>
        [TestMethod]
        public void AsType_FunctionForm_IsTheMethodAfterNumPysChecks()
        {
            var x = np.arange(3L);
            var f32 = np.astype(x, np.float32);
            f32.dtype.Should().Be(np.float32);
            f32.ToArray<float>().Should().Equal(0f, 1f, 2f);
            np.shares_memory(np.astype(x, np.int64), x).Should().BeFalse();        // copy=True copies even the same dtype
            ReferenceEquals(np.astype(x, np.int64, copy: false), x).Should().BeTrue();
            np.astype(x, np.float64, copy: false).dtype.Should().Be(np.float64);   // a cast must copy
            np.astype(x, null).dtype.Should().Be(np.float64);                       // dtype=None is float64
            np.astype(x, "f4").dtype.Should().Be(np.float32);
            np.astype(x, np.float64, device: "cpu").dtype.Should().Be(np.float64);
            np.astype(np.array(new[] { 1.7, -2.7 }), np.int32).ToArray<int>().Should().Equal(1, -2);
            Throws<ValueError>(() => np.astype(x, np.float64, device: "gpu"),
                "Device not understood. Only \"cpu\" is allowed, but received: gpu");
            Throws<TypeError>(() => np.astype(null, np.float64),
                "Input should be a NumPy array or scalar. It is a <class 'NoneType'> instead.");
            np.astype(np.asmatrix(np.ones(new Shape(1, 2))), np.int32).Should().BeOfType<NDMatrix>();
        }

        // ---- fix 3: np.fromiter(iter, dtype, count=-1) ------------------------------------------------------------------

        /// <summary>np.fromiter converts each item by NumPy's setitem rules and honours count (a negative count other
        ///     than -1 reads nothing); strings iterate characters, an NDArray iterates as an ndarray, the result is a
        ///     fresh C array.</summary>
        [TestMethod]
        public void FromIter_ReadsConvertsAndCounts_AsNumPy()
        {
            np.fromiter(Enumerable.Range(0, 5), np.float64).ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0);
            var three = np.fromiter(Enumerable.Range(0, 5), np.int32, count: 3);
            three.dtype.Should().Be(np.int32);
            three.ToArray<int>().Should().Equal(0, 1, 2);
            np.fromiter(Enumerable.Range(0, 5), np.int32, count: -2).shape.Should().Equal(0L);      // NumPy reads nothing
            np.fromiter(Enumerable.Range(0, 5), null).dtype.Should().Be(np.float64);                 // dtype=None
            np.fromiter(new double[0], np.float64).shape.Should().Equal(0L);
            np.fromiter(new[] { 1.7, 2.2 }, np.int32).ToArray<int>().Should().Equal(1, 2);
            np.fromiter(new[] { "1.5", "2" }, np.float64).ToArray<double>().Should().Equal(1.5, 2.0);
            np.fromiter(new object[] { null }, np.float64).GetDouble(0).Should().Be(double.NaN);
            np.fromiter(new[] { 0, 1, 2 }, np.@bool).ToArray<bool>().Should().Equal(false, true, true);
            np.fromiter(new[] { new Complex(1, 2) }, np.complex128).GetComplex(0).Should().Be(new Complex(1, 2));
            var u3 = np.fromiter(new[] { "ab", "cdef" }, "U3");
            u3.dtype.ToString().Should().Be("<U3");
            u3.item(1).Should().Be("cde");
            // A C# string iterates its characters, as a Python str does: '1', '2' parse into int32.
            np.fromiter("12", np.int32).ToArray<int>().Should().Equal(1, 2);
            // An NDArray iterates as NumPy's ndarray: a 1-D one by element.
            np.fromiter(np.array(new[] { 1.7, -2.7 }), np.int32).ToArray<int>().Should().Equal(1, -2);
            // The result owns a fresh C-contiguous buffer.
            var r = np.fromiter(new long[] { 4, 5 }, np.int64);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        /// <summary>np.fromiter's errors carry NumPy's texts in NumPy's order: the iterator first (None, a 0-d
        ///     array), then the unsized dtype, then each item as it is read (a bad item beats a short iterator), then
        ///     the short-iterator check.</summary>
        [TestMethod]
        public void FromIter_Errors_AreNumPys_InNumPysOrder()
        {
            Throws<ValueError>(() => np.fromiter(Enumerable.Range(0, 5), np.int32, count: 7),
                "iterator too short: Expected 7 but iterator had only 5 items.");
            Throws<OverflowError>(() => np.fromiter(new[] { 300 }, np.int8), "Python integer 300 out of bounds for int8");
            Throws<TypeError>(() => np.fromiter(new object[] { null }, np.int64),
                "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            Throws<ValueError>(() => np.fromiter(new[] { "ab", "cdef" }, "U"),
                "Must specify length when using variable-size data-type.");
            Throws<ValueError>(() => np.fromiter("ab", np.int32), "invalid literal for int() with base 10: 'a'");
            Throws<TypeError>(() => np.fromiter((IEnumerable<int>)null, np.float64), "'NoneType' object is not iterable");
            // NumPy gets the iterator before the unsized-dtype check.
            Throws<TypeError>(() => np.fromiter((IEnumerable<int>)null, "U"), "'NoneType' object is not iterable");
            Throws<TypeError>(() => np.fromiter(NDArray.Scalar(5.0), np.float64), "iteration over a 0-d array");
            // A bad item beats a short iterator (each item is packed as it is read).
            Throws<OverflowError>(() => np.fromiter(new[] { 300 }, np.int8, count: 5), "Python integer 300 out of bounds for int8");
            Throws<OverflowError>(() => np.fromiter(new object[] { 300, new[] { 2 } }, np.int8),
                "Python integer 300 out of bounds for int8");
        }

        /// <summary>A sequence item meets each dtype's own setitem answer (float / text / integer / complex refusals
        ///     in their words, bool's truth value), a 0-d array item is its value, and earlier items convert first.</summary>
        [TestMethod]
        public void FromIter_SequenceItems_EachDtypesOwnAnswer()
        {
            // NumPy's per-item store hands a sequence to the dtype's setitem, which answers in its own words.
            Throws<ValueError>(() => np.fromiter(new object[] { new[] { 1, 2 } }, np.float64), "setting an array element with a sequence.");
            Throws<TypeError>(() => np.fromiter(new object[] { new[] { 1, 2 } }, np.int8),
                "int() argument must be a string, a bytes-like object or a real number, not 'list'");
            Throws<TypeError>(() => np.fromiter(new object[] { (1, 2) }, np.int64),
                "int() argument must be a string, a bytes-like object or a real number, not 'tuple'");
            Throws<ValueError>(() => np.fromiter(new object[] { np.array(new[] { 5L }) }, np.int64),
                "setting an array element with a sequence.");
            Throws<TypeError>(() => np.fromiter(new object[] { new[] { 1, 2 } }, np.complex128), "must be real number, not list");
            Throws<TypeError>(() => np.fromiter(new object[] { np.array(new[] { 5L }) }, np.complex128),
                "only 0-dimensional arrays can be converted to Python scalars");
            Throws<ValueError>(() => np.fromiter(new object[] { new[] { 1, 2 } }, "U3"), "setting an array element with a sequence");
            // bool stores the truth value: a non-empty list is True, a one-element array its element's truth.
            np.fromiter(new object[] { new[] { 1, 2 }, new int[0], np.array(new[] { 0L }) }, np.@bool).ToArray<bool>()
                .Should().Equal(true, false, false);
            Throws<ValueError>(() => np.fromiter(new object[] { np.array(new[] { 5L, 6L }) }, np.@bool),
                "setting an array element with a sequence.");
            // A 0-d array item is its value; an N-D array iterates its rows (sequences).
            np.fromiter(new object[] { NDArray.Scalar(5L) }, np.float64).ToArray<double>().Should().Equal(5.0);
            Throws<ValueError>(() => np.fromiter(np.arange(4.0).reshape(2, 2), np.float64), "setting an array element with a sequence.");
            // The items before a refused sequence item convert first (NumPy packs as it reads).
            Throws<TypeError>(() => np.fromiter(new object[] { 1, new[] { 2 }, 300 }, np.int8),
                "int() argument must be a string, a bytes-like object or a real number, not 'list'");
        }

        /// <summary>With a count, np.fromiter never pulls item count + 1 (NumPy's loop): a sequence that fails past
        ///     the count is safe.</summary>
        [TestMethod]
        public void FromIter_Count_StopsReading_BeforeTheNextItem()
        {
            // NumPy's loop never pulls item count + 1: an infinite (or failing) tail is never reached.
            np.fromiter(ThreeThenFail(), np.int64, count: 3).ToArray<long>().Should().Equal(0L, 1L, 2L);
        }

        // ---- fix 4: np.fromfunction(function, shape, dtype=float) -------------------------------------------------------

        /// <summary>np.fromfunction calls the function once with the float64 (or given dtype) index grids and returns
        ///     its result; a grid returned verbatim is kept; NumPy's negative-dimension and not-callable errors.</summary>
        [TestMethod]
        public void FromFunction_CallsOnceWithTheIndexGrids()
        {
            var r = np.fromfunction(ix => ix[0] + ix[1], (2, 3));
            r.dtype.Should().Be(np.float64);
            r.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 1.0, 2.0, 3.0);
            np.fromfunction(ix => ix[0] * 10 + ix[1], (2, 3), dtype: np.int64).ToArray<long>()
                .Should().Equal(0L, 1L, 2L, 10L, 11L, 12L);
            np.fromfunction(ix => ix[0] == ix[1], (2, 2), dtype: np.@bool).ToArray<bool>().Should().Equal(true, false, false, true);
            np.fromfunction(ix => ix[0] * 2, new[] { 4 }).ToArray<double>().Should().Equal(0.0, 2.0, 4.0, 6.0);
            np.fromfunction(ix => ix[0] + ix[1], (0, 3)).shape.Should().Equal(0L, 3L);
            // A grid returned verbatim is the result (kept, not released with the other grids).
            np.fromfunction(ix => ix[0], (2, 3)).ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
            Throws<ValueError>(() => np.fromfunction(ix => ix[0], new[] { -1 }), "negative dimensions are not allowed");
            Throws<TypeError>(() => np.fromfunction(null, (2, 2)), "'NoneType' object is not callable");
        }

        /// <summary>The two documented C# differences of np.fromfunction: an int shape converts to a 1-D Shape
        ///     (NumPy's TypeError), and a number the function returns arrives as a 0-d array (NumPy returns the
        ///     number).</summary>
        [TestMethod, Misaligned]
        public void FromFunction_CSharpConversions_Differ_FromNumPysPython()
        {
            // NumPy: np.fromfunction(f, 3) is TypeError ('int' object is not iterable); C# converts 3 to the Shape (3,).
            np.fromfunction(ix => ix[0], 3).ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
            // NumPy returns the function's own value (the int 1); C# converts it to the NDArray the delegate returns.
            np.fromfunction(ix => 1, (2, 2)).ndim.Should().Be(0);
        }

        // ---- fix 5: np.little_endian ---------------------------------------------------------------------------------

        /// <summary>np.little_endian reports the process's byte order (BitConverter.IsLittleEndian).</summary>
        [TestMethod]
        public void LittleEndian_IsTheProcesssByteOrder()
        {
            np.little_endian.Should().Be(BitConverter.IsLittleEndian);
        }

        // ---- fix 6: finfo's nmant / iexp / nexp / machep / negep, and NumPy's minexp ----------------------------------

        /// <summary>finfo's nmant / iexp / nexp / machep / negep and the corrected minexp are NumPy 2.4.2's for
        ///     float16, float32, float64, complex128 and longdouble (eps, epsneg and smallest_normal are their powers
        ///     of two); decimal keeps its documented base-10 values.</summary>
        [TestMethod]
        public void FInfo_LayoutAttributes_AreNumPys()
        {
            CheckFInfo(np.float16, 10, 5, -10, -11, -14, 16);
            CheckFInfo(np.float32, 23, 8, -23, -24, -126, 128);
            CheckFInfo(np.float64, 52, 11, -52, -53, -1022, 1024);
            CheckFInfo(np.complex128, 52, 11, -52, -53, -1022, 1024);
            CheckFInfo(np.longdouble, 52, 11, -52, -53, -1022, 1024);
            var dec = np.finfo(NPTypeCode.Decimal);   // NumSharp-only: decimal's base-10 layout
            (dec.nmant, dec.iexp, dec.nexp, dec.machep, dec.negep).Should().Be((96, 8, 8, -28, -28));
        }

        // ---- fix 7: tensordot's one axes argument -----------------------------------------------------------------------

        /// <summary>np.tensordot and np.linalg.tensordot take NumPy's one axes argument as a pair of axis lists
        ///     (values from NumPy), np.linalg.tensordot the (int, int) pair too, and no four-parameter axesA / axesB
        ///     form remains.</summary>
        [TestMethod]
        public void Tensordot_AxesPair_IsNumPysOneArgument()
        {
            var A = np.arange(60.0).reshape(3, 4, 5);
            var B = np.arange(24.0).reshape(4, 3, 2);
            var expected = new[] { 4400.0, 4730.0, 4532.0, 4874.0, 4664.0, 5018.0, 4796.0, 5162.0, 4928.0, 5306.0 };
            var r = np.tensordot(A, B, axes: (new[] { 1, 0 }, new[] { 0, 1 }));
            r.shape.Should().Equal(5L, 2L);
            r.ToArray<double>().Should().Equal(expected);
            np.linalg.tensordot(A, B, axes: (new[] { 1, 0 }, new[] { 0, 1 })).ToArray<double>().Should().Equal(expected);
            np.linalg.tensordot(A, B, axes: (1, 0)).shape.Should().Equal(3L, 5L, 3L, 2L);
            np.tensordot(A, B, (new[] { -2 }, new[] { 0 })).shape.Should().Equal(3L, 5L, 3L, 2L);
            np.tensordot(A, B, (new int[0], new int[0])).shape.Should().Equal(3L, 4L, 5L, 4L, 3L, 2L);
            // NumPy has no axesA / axesB: no four-parameter form remains.
            typeof(np).GetMethods().Where(m => m.Name == "tensordot").Should().OnlyContain(m => m.GetParameters().Length == 3);
            typeof(np.linalg).GetMethods().Where(m => m.Name == "tensordot").Should().OnlyContain(m => m.GetParameters().Length == 3);
        }

        // ---- fix 8: multi_dot's one signature ---------------------------------------------------------------------------

        /// <summary>np.linalg.multi_dot takes the arrays as one sequence (NumPy's signature), with no params form.</summary>
        [TestMethod]
        public void MultiDot_TakesOneSequence()
        {
            var r = np.linalg.multi_dot(new[] { np.ones(new Shape(2, 3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4, 2)) });
            r.ToArray<double>().Should().Equal(12.0, 12.0, 12.0, 12.0);
            var forms = typeof(np.linalg).GetMethods().Where(m => m.Name == "multi_dot").ToArray();
            forms.Should().ContainSingle();
            forms[0].GetParameters()[0].IsDefined(typeof(ParamArrayAttribute), false).Should().BeFalse();
        }

        // ---- fix 9: unravel_index / ravel_multi_index keyword names -------------------------------------------------

        /// <summary>unravel_index / ravel_multi_index answer NumPy's keyword names (indices, multi_index, mode) and
        ///     keep none of the old ones.</summary>
        [TestMethod]
        public void IndexConversions_TakeNumPysKeywordNames()
        {
            np.unravel_index(indices: 5L, shape: new[] { 2, 3 }).Should().Equal(1L, 2L);
            np.ravel_multi_index(multi_index: new long[] { 1, 2 }, dims: new[] { 3, 4 }).Should().Be(6L);
            var arrs = new[] { np.array(new long[] { 1, 5 }), np.array(new long[] { 2, 9 }) };
            np.ravel_multi_index(arrs, new[] { 3, 4 }, mode: new[] { "clip", "wrap" }).ToArray<long>().Should().Equal(6L, 9L);
            var names = typeof(np).GetMethods().Where(m => m.Name is "unravel_index" or "ravel_multi_index")
                .SelectMany(m => m.GetParameters()).Select(p => p.Name).ToArray();
            names.Should().NotContain(new[] { "index", "coords", "modes" });
        }

        // ---- fix 10: iinfo / finfo: NumPy's keywords, string reading, texts, no arrays --------------------------------

        /// <summary>iinfo / finfo read a string as NumPy does (a non-dtype string is a str value, None is float64),
        ///     raise NumPy's ValueError texts (the kind character, the scalar class), take NumPy's keyword names and
        ///     have no array overload.</summary>
        [TestMethod]
        public void IInfoFInfo_ReadStringsAndRefuse_AsNumPy()
        {
            np.iinfo(int_type: np.int32).max.Should().Be(int.MaxValue);
            np.iinfo(int_type: "int8").min.Should().Be(-128);
            np.iinfo("i8").max.Should().Be(long.MaxValue);
            np.finfo(dtype: "float32").bits.Should().Be(32);
            np.finfo("f2").bits.Should().Be(16);
            // A string that names no dtype is NumPy's str VALUE (kind 'U'); None is iinfo's float64 kind 'f' and finfo's
            // TypeError since NumPy 2.5 (2.4 read it as float64).
            Throws<ValueError>(() => np.iinfo("foo"), "Invalid integer data type 'U'.");
            Throws<ValueError>(() => np.iinfo(""), "Invalid integer data type 'U'.");
            Throws<ValueError>(() => np.iinfo("float32"), "Invalid integer data type 'f'.");
            Throws<ValueError>(() => np.iinfo((string)null), "Invalid integer data type 'f'.");
            Throws<ValueError>(() => np.iinfo(np.float32), "Invalid integer data type 'f'.");
            Throws<ValueError>(() => np.iinfo(np.complex128), "Invalid integer data type 'c'.");
            // NumPy 2.5.4's texts name the descriptor (2.4 named the scalar class: data type <class 'numpy.int32'> not inexact).
            Throws<ValueError>(() => np.finfo("int32"), "data type dtype('int32') not compatible with finfo");
            Throws<ValueError>(() => np.finfo("foo"), "data type dtype('<U') not compatible with finfo");
            Throws<ValueError>(() => np.finfo(""), "data type dtype('<U') not compatible with finfo");
            Throws<ValueError>(() => np.finfo(np.int64), "data type dtype('int64') not compatible with finfo");
            Throws<TypeError>(() => np.finfo((string)null), "dtype must not be None");   // NumPy 2.5.4 (2.4: float64's)
            // NumPy's keyword names, and no array overload (NumPy refuses an array).
            typeof(np).GetMethods().Where(m => m.Name == "iinfo" && !m.IsGenericMethod)
                .Should().OnlyContain(m => m.GetParameters().Single().Name == "int_type");
            typeof(np).GetMethods().Where(m => m.Name == "finfo" && !m.IsGenericMethod)
                .Should().OnlyContain(m => m.GetParameters().Single().Name == "dtype");
            typeof(np).GetMethods().Where(m => m.Name is "iinfo" or "finfo")
                .SelectMany(m => m.GetParameters()).Should().NotContain(p => p.ParameterType == typeof(NDArray));
        }

        /// <summary>One dtype's NumPy finfo layout: the five attributes, minexp / maxexp, and eps / epsneg /
        ///     smallest_normal as the powers of two machep / negep / minexp name.</summary>
        /// <param name="dt">The dtype.</param>
        /// <param name="nmant">NumPy's nmant.</param>
        /// <param name="iexp">NumPy's iexp (and nexp).</param>
        /// <param name="machep">NumPy's machep.</param>
        /// <param name="negep">NumPy's negep.</param>
        /// <param name="minexp">NumPy's minexp.</param>
        /// <param name="maxexp">NumPy's maxexp.</param>
        /// <exception cref="AssertFailedException">A value differs from NumPy's.</exception>
        private static void CheckFInfo(DType dt, int nmant, int iexp, int machep, int negep, int minexp, int maxexp)
        {
            var fi = np.finfo(dt);
            fi.nmant.Should().Be(nmant, dt.ToString());
            fi.iexp.Should().Be(iexp, dt.ToString());
            fi.nexp.Should().Be(iexp, dt.ToString());
            fi.machep.Should().Be(machep, dt.ToString());
            fi.negep.Should().Be(negep, dt.ToString());
            fi.minexp.Should().Be(minexp, dt.ToString());
            fi.maxexp.Should().Be(maxexp, dt.ToString());
            System.Math.Pow(2, fi.machep).Should().Be(fi.eps, dt.ToString());
            System.Math.Pow(2, fi.negep).Should().Be(fi.epsneg, dt.ToString());
            System.Math.Pow(2, fi.minexp).Should().Be(fi.smallest_normal, dt.ToString());
        }

        /// <summary>0, 1, 2, then a failure: a sequence whose fourth item must never be pulled.</summary>
        /// <returns>The lazy sequence.</returns>
        /// <exception cref="InvalidOperationException">On reaching the fourth item (np.fromiter read past its count).</exception>
        private static IEnumerable<long> ThreeThenFail()
        {
            for (long i = 0; ; i++)
            {
                if (i == 3)
                    throw new InvalidOperationException("read past count");
                yield return i;
            }
        }

        // ---- fix 11: np.ma.apply_over_axes(func, a, NDAxis axes) ---------------------------------------------------------

        /// <summary>The evidence script's masked cube: arange(24.).reshape(2, 3, 4) with every multiple of 5 masked.</summary>
        /// <returns>A fresh masked array.</returns>
        private static NDMaskedArray MaskedCube()
            => np.ma.array(np.arange(24.0).reshape(2, 3, 4), mask: np.arange(24).reshape(2, 3, 4) % 5 == 0);

        /// <summary>np.ma.apply_over_axes takes NumPy's int-or-tuple axes (values and mask from NumPy), the kept int
        ///     / int[] forms, repeats and negative axes, every form's parameter named axes.</summary>
        [TestMethod]
        public void MaApplyOverAxes_TakesNumPysAxes()
        {
            var m = MaskedCube();
            Func<NDMaskedArray, int, NDMaskedArray> sum = (v, ax) => np.ma.sum(v, ax);
            var tuple = np.ma.apply_over_axes(sum, m, (0, 2));
            tuple.shape.Should().Equal(1L, 3L, 1L);
            tuple.data.ToArray<double>().Should().Equal(45.0, 87.0, 94.0);
            np.ma.getmaskarray(tuple).ToArray<bool>().Should().Equal(false, false, false);
            np.ma.apply_over_axes(sum, m, axes: 1).data.ToArray<double>()
                .Should().Equal(12.0, 10.0, 8.0, 21.0, 28.0, 51.0, 54.0, 42.0);
            np.ma.apply_over_axes(sum, m, new[] { 0, 2 }).data.ToArray<double>().Should().Equal(45.0, 87.0, 94.0);
            np.ma.apply_over_axes(sum, m, (0, 0)).shape.Should().Equal(1L, 3L, 4L);
            np.ma.apply_over_axes(sum, m, -1).shape.Should().Equal(2L, 3L, 1L);
            typeof(MaskedArrayModule).GetMethods().Where(x => x.Name == "apply_over_axes")
                .Should().OnlyContain(x => x.GetParameters()[2].Name == "axes");
        }

        // ---- fix 12: np.ma.angle(a, deg) ----------------------------------------------------------------------------------

        /// <summary>np.ma.angle passes deg on to np.angle: degrees, the mask unchanged, the masked data converted
        ///     too, real input.</summary>
        [TestMethod]
        public void MaAngle_TakesDeg()
        {
            var c = np.ma.array(np.array(new[] { new Complex(1, 1), new Complex(-1, 0), new Complex(0, 1) }),
                mask: np.array(new[] { false, true, false }));
            var deg = np.ma.angle(c, deg: true);
            deg.data.ToArray<double>().Should().Equal(45.0, 180.0, 90.0);
            np.ma.getmaskarray(deg).ToArray<bool>().Should().Equal(false, true, false);
            np.ma.angle(c, true).data.ToArray<double>().Should().Equal(45.0, 180.0, 90.0);
            np.ma.angle(c).data.GetDouble(0).Should().Be(System.Math.PI / 4);
            var ints = np.ma.array(np.array(new long[] { 1, -1, 0 }), mask: np.array(new[] { false, false, true }));
            np.ma.angle(ints, deg: true).data.ToArray<double>().Should().Equal(0.0, 180.0, 0.0);
        }

        // ---- fix 13: numpy.ma's creation keywords --------------------------------------------------------------------------

        /// <summary>np.ma's zeros / ones / empty / identity take NumPy's order, device, fill_value (converted, or
        ///     refused with numpy.ma's texts) and hardmask keywords, with NumPy's order and device errors.</summary>
        [TestMethod]
        public void MaCreation_TakesNumPysKeywords()
        {
            np.ma.zeros(3, fill_value: 5).fill_value.Should().Be(5.0);
            np.ma.zeros(2, np.int64, fill_value: 7.9).fill_value.Should().Be(7L);
            Throws<TypeError>(() => np.ma.zeros(2, fill_value: "x"), "Cannot set fill value of string with array of dtype float64");
            Throws<TypeError>(() => np.ma.zeros(2, np.int8, fill_value: 300), "Cannot convert fill_value 300 to dtype int8");
            np.ma.zeros(3, hardmask: true).hardmask.Should().BeTrue();
            np.ma.zeros(3).hardmask.Should().BeFalse();
            np.ma.ones((2, 2), order: 'F').data.Shape.IsFContiguous.Should().BeTrue();
            np.ma.zeros((2, 2), np.float64, 'F').data.Shape.IsFContiguous.Should().BeTrue();
            Throws<ValueError>(() => np.ma.empty(2, device: "gpu"), "Device not understood. Only \"cpu\" is allowed, but received: gpu");
            Throws<ValueError>(() => np.ma.zeros(2, order: 'X'), "order must be one of 'C', 'F', 'A', or 'K' (got 'X')");
            var identity = np.ma.identity(2, fill_value: 1, hardmask: true);
            identity.fill_value.Should().Be(1.0);
            identity.hardmask.Should().BeTrue();
            identity.data.ToArray<double>().Should().Equal(1.0, 0.0, 0.0, 1.0);
        }

        /// <summary>np.ma's *_like functions follow NumPy's __array_finalize__: the prototype mask is reshaped for a
        ///     same-size shape and dropped for another size, copied in the result's order, with the fill and hardness
        ///     carried; copy takes order.</summary>
        [TestMethod]
        public void MaLike_ShapeAndOrder_FollowNumPysArrayFinalize()
        {
            var m = np.ma.array(np.arange(6.0).reshape(2, 3), mask: np.array(new[,] { { true, false, false }, { false, true, false } }),
                fill_value: 7.0, hard_mask: true);
            // Another shape of the same size: the mask is reshaped (C order); fill and hardness ride along.
            var reshaped = np.ma.zeros_like(m, shape: (3, 2));
            reshaped.shape.Should().Equal(3L, 2L);
            np.ma.getmaskarray(reshaped).ToArray<bool>().Should().Equal(true, false, false, false, true, false);
            reshaped.fill_value.Should().Be(7.0);
            reshaped.hardmask.Should().BeTrue();
            // Another size: no mask.
            var cube = MaskedCube();
            np.ma.getmaskarray(np.ma.zeros_like(cube, shape: (2, 2))).ToArray<bool>().Should().Equal(false, false, false, false);
            // order='F': the result and its mask copy are F-ordered.
            var fl = np.ma.zeros_like(m, order: 'F');
            fl.data.Shape.IsFContiguous.Should().BeTrue();
            np.ma.getmaskarray(fl).Shape.IsFContiguous.Should().BeTrue();
            np.ma.ones_like(m, dtype: np.int64).fill_value.Should().Be(7L);
            np.ma.empty_like(m, shape: 6).shape.Should().Equal(6L);
            Throws<ValueError>(() => np.ma.zeros_like(m, device: "gpu"), "Device not understood. Only \"cpu\" is allowed, but received: gpu");
            // copy(order): data and mask alike.
            var fcopy = np.ma.copy(m, order: 'F');
            fcopy.data.Shape.IsFContiguous.Should().BeTrue();
            np.ma.getmaskarray(fcopy).Shape.IsFContiguous.Should().BeTrue();
            fcopy.fill_value.Should().Be(7.0);
            m.copy('F').data.Shape.IsFContiguous.Should().BeTrue();
        }

        // ---- fix 14: np.ma.argmax / argmin's out and keepdims ------------------------------------------------------------

        /// <summary>np.ma.argmax / argmin and the methods take NumPy's out and keepdims (values from NumPy), with
        ///     np.argmax's out checks and the common call unchanged.</summary>
        [TestMethod]
        public void MaArgMaxArgMin_TakeOutAndKeepdims()
        {
            var m = MaskedCube();
            var kd = np.ma.argmax(m, keepdims: true);
            kd.shape.Should().Equal(1L, 1L, 1L);
            kd.GetInt64(0, 0, 0).Should().Be(23L);
            var kd1 = np.ma.argmax(m, 1, keepdims: true);
            kd1.shape.Should().Equal(2L, 1L, 4L);
            kd1.ToArray<long>().Should().Equal(2L, 2L, 1L, 2L, 1L, 2L, 2L, 2L);
            np.ma.argmin(m, 2, keepdims: true).ToArray<long>().Should().Equal(1L, 0L, 0L, 0L, 0L, 1L);
            var o = np.zeros(new Shape(2, 4), np.int64);
            ReferenceEquals(np.ma.argmax(m, 1, null, o), o).Should().BeTrue();
            o.ToArray<long>().Should().Equal(2L, 2L, 1L, 2L, 1L, 2L, 2L, 2L);
            var o2 = np.zeros(new Shape(2, 4), np.int64);
            ReferenceEquals(m.argmin(1, null, o2), o2).Should().BeTrue();
            o2.ToArray<long>().Should().Equal(1L, 0L, 0L, 0L, 0L, 0L, 0L, 1L);
            Throws<TypeError>(() => np.ma.argmax(m, 1, @out: np.zeros(new Shape(2, 4))),
                "Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'");
            Throws<ValueError>(() => np.ma.argmax(m, 1, @out: np.zeros(new Shape(3, 4), np.int64)),
                "output array does not match result of np.argmax.");
            np.ma.argmax(m, 1, @out: np.zeros(new Shape(2, 4), np.int32)).ToArray<int>().Should().Equal(2, 2, 1, 2, 1, 2, 2, 2);
            np.ma.argmax(m).GetInt64().Should().Be(23L);
            np.ma.argmax(m, 2, 100).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 3L, 0L);
        }

        // ---- fix 15: np.lib.stride_tricks.sliding_window_view -----------------------------------------------------------

        /// <summary>sliding_window_view returns NumPy's window view over x's buffer (values for 1-D, 2-D, axis,
        ///     repeated axis, strided and transposed inputs), read-only by default, writing through with writeable:
        ///     true.</summary>
        [TestMethod]
        public void SlidingWindowView_IsNumPysReadOnlyWindowView()
        {
            var x = np.arange(6L);
            var v = np.lib.stride_tricks.sliding_window_view(x, 3);
            v.shape.Should().Equal(4L, 3L);
            v.ToArray<long>().Should().Equal(0L, 1L, 2L, 1L, 2L, 3L, 2L, 3L, 4L, 3L, 4L, 5L);
            np.shares_memory(v, x).Should().BeTrue();
            v.Shape.IsWriteable.Should().BeFalse();
            Throws<ValueError>(() => v[0, 0] = 5, "assignment destination is read-only");

            var x2 = np.arange(12L).reshape(3, 4);
            var w = np.lib.stride_tricks.sliding_window_view(x2, (2, 2));
            w.shape.Should().Equal(2L, 3L, 2L, 2L);
            w[0, 0].ToArray<long>().Should().Equal(0L, 1L, 4L, 5L);
            np.lib.stride_tricks.sliding_window_view(x2, 2, axis: 0).ToArray<long>()
                .Should().Equal(0L, 4L, 1L, 5L, 2L, 6L, 3L, 7L, 4L, 8L, 5L, 9L, 6L, 10L, 7L, 11L);
            np.lib.stride_tricks.sliding_window_view(x2, 2, axis: -1).shape.Should().Equal(3L, 3L, 2L);
            np.lib.stride_tricks.sliding_window_view(x2, (2, 3), axis: (0, 1)).shape.Should().Equal(2L, 2L, 2L, 3L);
            var rep = np.lib.stride_tricks.sliding_window_view(x2, (2, 2), axis: (1, 1));
            rep.shape.Should().Equal(3L, 2L, 2L, 2L);
            rep[0, 0].ToArray<long>().Should().Equal(0L, 1L, 1L, 2L);
            np.lib.stride_tricks.sliding_window_view(x, 6).shape.Should().Equal(1L, 6L);
            np.lib.stride_tricks.sliding_window_view(x, 0).shape.Should().Equal(7L, 0L);
            np.lib.stride_tricks.sliding_window_view(x["::2"], 2).ToArray<long>().Should().Equal(0L, 2L, 2L, 4L);
            var tw = np.lib.stride_tricks.sliding_window_view(x2.T, (2, 2));
            tw.shape.Should().Equal(3L, 2L, 2L, 2L);
            tw[0, 1].ToArray<long>().Should().Equal(4L, 8L, 5L, 9L);

            // writeable: true writes through to x.
            var xx = np.arange(6L);
            var wv = np.lib.stride_tricks.sliding_window_view(xx, 3, writeable: true);
            wv[1, 0] = 50L;
            xx.ToArray<long>().Should().Equal(0L, 50L, 2L, 3L, 4L, 5L);
        }

        /// <summary>sliding_window_view's refusals carry NumPy's texts: a window too large or negative, a
        ///     window_shape that does not match the axes, an axis out of range.</summary>
        [TestMethod]
        public void SlidingWindowView_Refusals_AreNumPys()
        {
            var x = np.arange(6L);
            var x2 = np.arange(12L).reshape(3, 4);
            Throws<ValueError>(() => np.lib.stride_tricks.sliding_window_view(x, 7), "window shape cannot be larger than input array shape");
            Throws<ValueError>(() => np.lib.stride_tricks.sliding_window_view(x, -1), "`window_shape` cannot contain negative values");
            Throws<ValueError>(() => np.lib.stride_tricks.sliding_window_view(x2, 2),
                "Since axis is `None`, must provide window_shape for all dimensions of `x`; got 1 window_shape elements and `x.ndim` is 2.");
            Throws<ValueError>(() => np.lib.stride_tricks.sliding_window_view(x2, (2, 2), axis: 0),
                "Must provide matching length window_shape and axis; got 2 window_shape elements and 1 axes elements.");
            new Action(() => np.lib.stride_tricks.sliding_window_view(x2, 2, axis: 2)).Should().Throw<AxisError>()
                .Which.Message.Should().StartWith("axis 2 is out of bounds for array of dimension 2");
            Throws<ValueError>(() => np.lib.stride_tricks.sliding_window_view(np.arange(0L), 1),
                "window shape cannot be larger than input array shape");
        }
    }
}
