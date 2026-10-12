using System;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     Creating, writing and reading NEP 55 <c>StringDType</c> arrays (strings phase 1, P1.5): <c>stringdtype_setitem</c>'s
    ///     coercion through every C# spelling of a Python object — a literal (str / int / float / bool), null for <c>None</c>, a
    ///     boxed value through the object-typed APIs —, the NA objects, the creation functions, element access and printing.
    ///     Expected values were probed against <c>numpy==2.4.2</c>.
    /// </summary>
    [TestClass]
    public class StringDTypeCreationTests
    {
        /// <summary>A fresh <c>StringDType()</c> descriptor.</summary>
        private static DType T => np.dtypes.StringDType();

        /// <summary>A fresh <c>StringDType(na_object=None)</c> descriptor.</summary>
        private static DType TN => np.dtypes.StringDType(na_object: null);

        /// <summary>A fresh <c>StringDType(coerce=False)</c> descriptor.</summary>
        private static DType TC => np.dtypes.StringDType(coerce: false);

        /// <summary>The text NumPy raises for a non-text value under <c>coerce=False</c>.</summary>
        private const string NoCoercion = "StringDType only allows string data when string coercion is disabled.";

        /// <summary>The elements of an array in C order.</summary>
        /// <param name="a">An array.</param>
        /// <returns>The values.</returns>
        private static object[] V(NDArray a) => Enumerable.Range(0, (int)a.size).Select(i => a.Storage.GetAtIndex(i)).ToArray();

        /// <summary>
        ///     <c>np.array</c> of mixed Python objects: under coercion each is its <c>str()</c> (<c>None</c> too, unless it is the
        ///     NA object); under <c>coerce=False</c> any non-text value is refused; a ragged list is NumPy's inhomogeneous-shape
        ///     error.
        /// </summary>
        [TestMethod]
        public void Array_MixedObjects()
        {
            using (var a = np.array(new object[] { "a", 1L, 2.5, true, null }, T))
                V(a).Should().Equal(new object[] { "a", "1", "2.5", "True", "None" });
            using (var a = np.array(new object[] { "a", null }, TN))
                V(a).Should().Equal(new object[] { "a", null });
            Action noCoerce = () => np.array(new object[] { "a", 1L }, TC);
            noCoerce.Should().Throw<ValueError>().WithMessage(NoCoercion);
            Action ragged = () => np.array(new object[] { new object[] { "a" }, new object[] { "b", "c" } }, T);
            ragged.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.*inhomogeneous*");
        }

        /// <summary>
        ///     <c>NDArray.Scalar(value, StringDType)</c> is <c>np.array(value, dtype='T')</c>: one Python object by setitem.
        /// </summary>
        [TestMethod]
        public void Scalar_IsSetitem()
        {
            using (var s = NDArray.Scalar(5L, T))
            {
                s.ndim.Should().Be(0);
                s.GetValue().Should().Be("5");
            }
            using (var s = NDArray.Scalar(null, TN))
                s.GetValue().Should().BeNull();
            Action noCoerce = () => NDArray.Scalar(5L, TC);
            noCoerce.Should().Throw<ValueError>().WithMessage(NoCoercion);
        }

        /// <summary>
        ///     The creation functions: <c>np.full</c> with <c>None</c> (the NA object, <c>'None'</c>, or the coerce=False
        ///     refusal) and with an array fill value (cast), <c>np.ones</c> (<c>'1'</c>), <c>np.zeros</c> / <c>np.empty</c>
        ///     (<c>''</c>), the <c>_like</c> forms keeping the descriptor.
        /// </summary>
        [TestMethod]
        public void Creation_Functions()
        {
            using (var f = np.full(new Shape(2), null, TN))
                V(f).Should().Equal(new object[] { null, null });
            using (var f = np.full(new Shape(2), null, T))
                V(f).Should().Equal(new object[] { "None", "None" });
            Action noneNoCoerce = () => np.full(new Shape(2), null, TC);
            noneNoCoerce.Should().Throw<ValueError>().WithMessage(NoCoercion);
            using (var f = np.full(new Shape(2, 2), np.array(new[] { "x", "y" }), T))
                V(f).Should().Equal(new object[] { "x", "y", "x", "y" });
            using (var o = np.ones(new Shape(2), T))
                V(o).Should().Equal(new object[] { "1", "1" });
            using (var z = np.zeros(new Shape(2), TN))
            using (var zl = np.zeros_like(z))
            {
                V(z).Should().Equal(new object[] { "", "" });
                zl.dtype.Equals(TN).Should().BeTrue();
            }
            using (var e = np.empty(new Shape(1), T))
                e.GetValue(0).Should().Be("");
        }

        /// <summary>
        ///     Python's <c>None</c> (C# null) through the indexers: the NA element of a <c>None</c>-NA descriptor, <c>'None'</c>
        ///     under coercion, NumPy's ValueError under <c>coerce=False</c>; a row target takes it too; an out-of-range index is
        ///     still NumPy's IndexError first.
        /// </summary>
        [TestMethod]
        public void Setitem_None()
        {
            using (var a = np.array(new[] { "a", "b", "c" }, TN))
            {
                a["::2"] = null;
                V(a).Should().Equal(new object[] { null, "b", null });
            }
            using (var a = np.array(new[] { "a", "b", "c" }, T))
            {
                a[np.array(new[] { 1 })] = null;
                V(a).Should().Equal(new object[] { "a", "None", "c" });
            }
            using (var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, TN))
            {
                m[0] = null;
                V(m).Should().Equal(new object[] { null, null, "c", "d" });
            }
            using (var a = np.array(new[] { "a" }, TC))
            {
                Action none = () => a[":"] = null;
                none.Should().Throw<ValueError>().WithMessage(NoCoercion);
            }
            using (var e = np.empty(new Shape(0, 2), TN))
            {
                Action outOfRange = () => e[0] = null;
                outOfRange.Should().Throw<Exception>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            }
        }

        /// <summary>
        ///     C# literals are Python scalars: a bool is <c>'True'</c> under coercion and refused under <c>coerce=False</c> (the
        ///     cast of a 0-d bool array would store <c>'True'</c> in both); a string keeps its trailing NUL (its 0-d <c>U</c>
        ///     array alone reads without it), and so does every StringDType store of it.
        /// </summary>
        [TestMethod]
        public void Setitem_BoolAndStrLiterals()
        {
            using (var a = np.array(new[] { "a", "b" }, T))
            {
                a[":"] = true;
                V(a).Should().Equal(new object[] { "True", "True" });
                a["0"] = "x\0";
                a.GetValue(0).Should().Be("x\0");
            }
            using (var a = np.array(new[] { "a", "b" }, TC))
            {
                Action boolean = () => a[":"] = true;
                boolean.Should().Throw<ValueError>().WithMessage(NoCoercion);
                a[":"] = "ok";
                V(a).Should().Equal(new object[] { "ok", "ok" });
            }
        }

        /// <summary>
        ///     Element access returns the NA object itself for an NA element (null, a NaN, the NA string) and each text as a C#
        ///     string; the printed forms are NumPy's (<c>None</c> / <c>nan</c> unquoted, texts quoted with Python's repr rules).
        /// </summary>
        [TestMethod]
        public void Access_And_Printing()
        {
            using (var a = np.array(new object[] { "it's", null, "a\nb" }, TN))
            {
                a.item(1).Should().BeNull();
                a.tolist().Should().BeEquivalentTo(new object[] { "it's", null, "a\nb" });
                a.ToString(true).Should().Be("array([\"it's\", None, 'a\\nb'], dtype=StringDType(na_object=None))");
                a.ToString(false).Should().Be("[\"it's\" None 'a\\nb']");
            }
            using (var a = np.array(new object[] { "x", double.NaN }, np.dtypes.StringDType(na_object: double.NaN)))
                a.ToString(true).Should().Be("array(['x', nan], dtype=StringDType(na_object=nan))");
            using (var z = np.array("abc", T))
            {
                z.ToString(true).Should().Be("array('abc', dtype=StringDType())");
                z.ToString(false).Should().Be("abc");
            }
        }

        /// <summary>
        ///     The explicit <c>(Array)</c> conversion of a <c>StringDType</c> array (<c>tolist()</c> with C# arrays): a
        ///     <c>string</c> array of the array's shape, whose NA elements are null for a <c>None</c> NA object; an
        ///     <c>object</c> array when the NA object is not a string or None (its NA elements carry that object); a 0-d array
        ///     becomes one element.
        /// </summary>
        [TestMethod]
        public void ToManagedArray_KeepsTheNAObjects()
        {
            using (var a = np.array(new object[] { "a", null }, TN))
            {
                var m = (Array)a;
                m.Should().BeOfType<string[]>();
                ((string[])m).Should().Equal("a", null);
            }
            using (var a = np.array(new object[] { "a", double.NaN }, np.dtypes.StringDType(na_object: double.NaN)))
            {
                var m = (object[])(Array)a;
                m[0].Should().Be("a");
                m[1].Should().BeOfType<double>().Which.Should().Be(double.NaN);
            }
            using (var z = np.array("abc", T))
                ((string[])(Array)z).Should().Equal("abc");
            using (var m2 = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T))
                ((string[,])(Array)m2)[1, 0].Should().Be("c");
        }

        /// <summary>
        ///     <c>np.insert</c> of a Python object into a <c>StringDType</c> array stores it by setitem (a number is its
        ///     <c>str()</c>): <c>['a','x','b']</c>, <c>['a','5','b']</c> — through the literal's NDArray overload and through
        ///     the object-typed one (a boxed value).
        /// </summary>
        [TestMethod]
        public void Insert_PythonObjects_AreSetitem()
        {
            using var t = np.array(new[] { "a", "b" }, T);
            using (var r = np.insert(t, 1, "x"))
                V(r).Should().Equal(new object[] { "a", "x", "b" });
            using (var r = np.insert(t, 1, 5))
                V(r).Should().Equal(new object[] { "a", "5", "b" });
            using (var r = np.insert(t, 1, (object)"x"))
                V(r).Should().Equal(new object[] { "a", "x", "b" });
            using (var r = np.insert(t, 1L, (object)5))
                V(r).Should().Equal(new object[] { "a", "5", "b" });
        }

        /// <summary>
        ///     <c>np.ones</c> with an explicit memory order lays a <c>StringDType</c> array out in it (<c>'1'</c> everywhere,
        ///     F-contiguous for <c>'F'</c>, as NumPy).
        /// </summary>
        [TestMethod]
        public void Ones_WithOrder()
        {
            using var f = np.ones(new Shape(2, 2), 'F', T);
            f.Shape.IsFContiguous.Should().BeTrue();
            V(f).Should().Equal(new object[] { "1", "1", "1", "1" });
        }

        /// <summary>
        ///     <c>SetData(value, coordinates)</c> naming every coordinate is an element store: a Python object goes in by
        ///     setitem, as an integer index does.
        /// </summary>
        [TestMethod]
        public void SetData_FullCoordinates_IsAnItemStore()
        {
            using var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, TN);
            NDArray text = "q";
            m.SetData(text, 1, 1);
            m.SetData((NDArray)null, 0, 0);
            V(m).Should().Equal(new object[] { null, "b", "c", "q" });
        }

        /// <summary>
        ///     <c>np.nditer</c> allocates a <c>StringDType</c> output operand with the requested descriptor instance (a buffer
        ///     of empty strings with its own allocator, NumPy's <c>['', '']</c>), and its iterator views write through to it.
        /// </summary>
        [TestMethod]
        public void Nditer_AllocatesAStringDTypeOperand()
        {
            using var t = np.array(new[] { "a", "b" }, T);
            var requested = TN;
            using var it = np.nditer(new NDArray[] { t, null }, flags: new[] { "refs_ok" },
                op_flags: new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, op_dtypes: new DType[] { null, requested });
            var output = it.operands[1];
            output.dtype.Equals(requested).Should().BeTrue();
            V(output).Should().Equal(new object[] { "", "" });
            foreach (var _ in it)
                it[1]["..."] = "w"; // the current element's 0-d view
            V(output).Should().Equal(new object[] { "w", "w" });

            // Buffering or casting a StringDType operand is a later package (P1.7): refused loudly.
            Action buffered = () => np.nditer(t, flags: new[] { "refs_ok", "buffered" });
            buffered.Should().Throw<NotSupportedException>().WithMessage("An iterator that buffers or casts a StringDType operand*");
            Action cast = () => np.nditer(new NDArray[] { t }, flags: new[] { "refs_ok" }, op_dtypes: new DType[] { TN });
            cast.Should().Throw<NotSupportedException>().WithMessage("An iterator that buffers or casts a StringDType operand*");
        }

        /// <summary>
        ///     An index of integers alone stores a Python object by setitem straight into the element (NumPy's
        ///     <c>array_assign_item</c>; the C# integer indexer, a string index and a 2-D coordinate alike), while a 0-d view
        ///     selected by an ellipsis takes the view assignment — both give the same element.
        /// </summary>
        [TestMethod]
        public void ItemStores_AndZeroDViewStores()
        {
            using var t = np.array(new[] { "a", "b", "c" }, TN);
            t[2] = "z";
            t["-3"] = null;
            V(t).Should().Equal(new object[] { null, "b", "z" });
            using var m = np.array(new[,] { { "a", "b" }, { "c", "d" } }, T);
            m[1, 0] = 7L;
            m["0, 1"] = "\U0001F600";
            V(m).Should().Equal(new object[] { "a", "\U0001F600", "7", "d" });
            using var z = np.array("abc", T);
            z["..."] = "xyz";
            z.GetValue().Should().Be("xyz");
        }
    }
}
