using System;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the code examples of <c>docs/website-src/docs/getting-and-setting-values.md</c>, except
    ///     its "Reading and writing text" snippet, which <see cref="TextInGuidesDocTests"/> covers. Each test mirrors one
    ///     section and asserts what the page claims; every value was re-run in NumPy 2.4.2. The accessor tests pin the
    ///     conversion rule the page documents: every typed getter and setter applies NumPy's cast when its type is not the
    ///     dtype's own, and only the by-<c>ref</c> <c>np.nditer&lt;T&gt;</c> refuses a mismatched <c>T</c>.
    /// </summary>
    [TestClass]
    public class GettingAndSettingValuesDocTests
    {
        /// <summary>"Views vs copies": a slice is a view whose store reaches the original, while a copy is
        /// independent.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ViewsAndCopies()
        {
            var a = np.arange(10);
            var v = a["2:5"];
            v[0] = 999;
            ((long)a[2]).Should().Be(999L, "the view writes through");
            var c = a["2:5"].copy();
            c[0] = 0;
            ((long)a[2]).Should().Be(999L, "the copy is independent");
        }

        /// <summary>"Basic indexing" and "Boolean and fancy indexing": row and element access, slices with steps and an
        /// ellipsis, and the gathers that copy.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Indexing_BasicMaskAndFancy()
        {
            var a = np.arange(20).reshape(4, 5);
            a[0].shape.Should().Equal(new long[] { 5 });
            a[-1].ToArray<long>().Should().Equal(15L, 16L, 17L, 18L, 19L);
            a[1, 2].ndim.Should().Be(0, "a single element is a 0-d array");
            ((long)a[1, 2]).Should().Be(7L);
            a["1:3"].shape.Should().Equal(new long[] { 2, 5 });
            a["1:3, :2"].ToArray<long>().Should().Equal(5L, 6L, 10L, 11L);
            a["::2"].shape.Should().Equal(new long[] { 2, 5 });
            a["::-1"][0].ToArray<long>().Should().Equal(15L, 16L, 17L, 18L, 19L);
            a["..., -1"].ToArray<long>().Should().Equal(4L, 9L, 14L, 19L);

            var arr = np.array([10, 20, 30, 40, 50]);
            arr[arr > 20].ToArray<int>().Should().Equal(30, 40, 50);
            arr[np.array([0, 2, 4])].ToArray<int>().Should().Equal(10, 30, 50);
            arr[new[] { 0, 2, 4 }].ToArray<int>().Should().Equal(new[] { 10, 30, 50 }, "a raw int[] as the sole index is fancy");
        }

        /// <summary>"Single elements - five accessors": the indexer, <c>item&lt;T&gt;</c>, <c>GetValue&lt;T&gt;</c>,
        /// <c>GetAtIndex&lt;T&gt;</c> and the typed named getters, each converting with NumPy's cast when its type is not
        /// the dtype's own.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void SingleElements_FiveAccessors()
        {
            var a = np.arange(12).reshape(3, 4);
            ((long)a[1, 2]).Should().Be(6L);
            a.item<long>(6).Should().Be(6L);
            a.item<double>(6).Should().Be(6.0);
            a.item(6).Should().BeOfType<long>();
            a.GetValue<long>(1, 2).Should().Be(6L);
            a.GetValue<double>(1, 2).Should().Be(6.0, "another T is NumPy's cast to T");
            a.GetAtIndex<long>(6).Should().Be(6L);
            a.GetAtIndex<double>(6).Should().Be(6.0);
            np.zeros(4).GetDouble(0).Should().Be(0.0);
            a.GetDouble(1, 2).Should().Be(6.0, "the int64 element converted, as astype converts it");
        }

        /// <summary>"Reading flat, in order": the boxed flat walk, the flat iterator's element, and the by-ref typed
        /// iteration that requires the dtype's own type.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ReadingFlat_InOrder()
        {
            var a = np.arange(12).reshape(3, 4);
            int count = 0;
            foreach (var _ in a.flat)
                count++;
            count.Should().Be(12);
            ((long)a.flatiter[5]).Should().Be(5L);

            long total = 0;
            foreach (ref long e in np.nditer<long>(a))
                total += e;
            total.Should().Be(66L);
            Action wrongT = () => { foreach (ref double e in np.nditer<double>(a)) { } };
            wrongT.Should().Throw<ArgumentException>("a ref T cannot convert");
        }

        /// <summary>"Setting values: Elements": the indexer and the typed setters, which cast a value of another type
        /// to the dtype (2.5 into int64 stores 2).</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void SettingElements()
        {
            var a = np.arange(12).reshape(3, 4);
            a[1, 2] = 99;
            ((long)a[1, 2]).Should().Be(99L);
            a.SetValue(98L, 1, 2);
            ((long)a[1, 2]).Should().Be(98L);
            a.SetAtIndex(97L, 6);
            ((long)a[1, 2]).Should().Be(97L);
            a.SetValue(2.5, 1, 3);
            ((long)a[1, 3]).Should().Be(2L, "the value is cast to int64");
            var f = np.zeros((3, 4));
            f.SetDouble(3.5, 1, 2);
            ((double)f[1, 2]).Should().Be(3.5);

            // The Troubleshooting entry "GetInt32 / SetDouble changed my value": both directions cast.
            np.array([2.9]).GetInt32(0).Should().Be(2);
            a.SetDouble(3.5, 0, 0);
            ((long)a[0, 0]).Should().Be(3L);
        }

        /// <summary>"Slices", "Boolean masks" and "Fancy indices" stores: a scalar fills, a row broadcasts, a mask takes
        /// one value per True position, and an index array one value per index.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void SettingSlicesMasksAndFancy()
        {
            var a = np.arange(20).reshape(4, 5);
            a["1:3"] = 0;
            a[0] = 7;
            a[0] = np.zeros(5);
            a["1:3, :"] = np.array([1, 2, 3, 4, 5]);
            a["::2"] = -1;
            a.ToArray<long>().Should().Equal(
                -1L, -1L, -1L, -1L, -1L, 1L, 2L, 3L, 4L, 5L, -1L, -1L, -1L, -1L, -1L, 15L, 16L, 17L, 18L, 19L);

            var arr = np.array([10, 20, 30, 40, 50]);
            arr[arr > 20] = -1;
            arr.ToArray<int>().Should().Equal(10, 20, -1, -1, -1);
            arr[arr < 0] = np.array([1, 2, 3]);
            arr.ToArray<int>().Should().Equal(10, 20, 1, 2, 3);

            var fancy = np.array([10, 20, 30, 40, 50]);
            fancy[np.array([0, 2, 4])] = 0;
            fancy.ToArray<int>().Should().Equal(0, 20, 0, 40, 0);
            fancy[np.array([0, 2, 4])] = np.array([1, 2, 3]);
            fancy.ToArray<int>().Should().Equal(1, 20, 2, 40, 3);
        }

        /// <summary>The whole-array and scatter stores: <c>fill</c>, <c>put</c> (with the clip mode), <c>place</c>'s cyclic
        /// values, <c>copyto</c> with a mask, <c>fill_diagonal</c>, and flat-iterator stores through a transposed
        /// view.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void FillPutPlaceCopytoDiagonalAndFlat()
        {
            var z = np.zeros((3, 4));
            z.fill(7);
            ((double)z.sum()).Should().Be(84.0);

            var p = np.arange(5);
            p.put(np.array([0, 2]), np.array([10, 20]));
            p.ToArray<long>().Should().Equal(10L, 1L, 20L, 3L, 4L);
            np.put(p, np.array([9]), np.array([99]), "clip");
            p.ToArray<long>().Should().Equal(new[] { 10L, 1L, 20L, 3L, 99L }, "clip sends index 9 to the last element");

            var q = np.arange(6);
            np.place(q, q > 2, np.array([100, 200]));
            q.ToArray<long>().Should().Equal(0L, 1L, 2L, 100L, 200L, 100L);

            var dst = np.zeros(3);
            np.copyto(dst, np.array([1.0, 2.0, 3.0]), where: np.array([true, false, true]));
            dst.ToArray<double>().Should().Equal(1.0, 0.0, 3.0);

            var m = np.zeros((4, 4));
            np.fill_diagonal(m, 5);
            np.diag(m).ToArray<double>().Should().Equal(5.0, 5.0, 5.0, 5.0);
            np.fill_diagonal(m, np.array([1, 2, 3, 4]));
            np.diag(m).ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0);

            var t = np.arange(6).reshape(2, 3).T;
            t.flatiter[5] = 99;
            t.ToArray<long>().Should().Equal(0L, 3L, 1L, 4L, 2L, 99L);
            t.flatiter["::2"] = 0;
            t.ToArray<long>().Should().Equal(0L, 3L, 0L, 4L, 0L, 99L);
            t.flatiter[new[] { 0, 2 }] = np.array([7, 8]);
            t.ToArray<long>().Should().Equal(7L, 3L, 8L, 4L, 0L, 99L);
        }

        /// <summary>"Value conversion on assignment": a weak C# int out of int8's range raises NumPy's OverflowError text,
        /// while a 0-d int32 array casts (300 wraps to 44), as NumPy's <c>a.fill(np.array(300, np.int32))</c> does.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Fill_WeakAndStrongScalars()
        {
            var a = np.zeros(3, dtype: np.int8);
            Action weak = () => a.fill(300);
            weak.Should().Throw<OverflowError>().WithMessage("Python integer 300 out of bounds for int8");
            a.fill(NDArray.Scalar<int>(300));
            a.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)44, (sbyte)44);
        }

        /// <summary>"Compound assignment": C#'s <c>+=</c> rebinds the variable to a new array (the documented difference
        /// from NumPy), a slice store rewrites the buffer in place, and <c>+=</c> on a view writes into its
        /// parent.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void CompoundAssignment()
        {
            var x = np.array([1, 2, 3]);
            var alias = x;
            x += 10;
            x.ToArray<int>().Should().Equal(11, 12, 13);
            alias.ToArray<int>().Should().Equal(new[] { 1, 2, 3 }, "unlike NumPy, the alias keeps the old array");
            x[":"] = x + 1;
            x.ToArray<int>().Should().Equal(12, 13, 14);

            var a = np.arange(20).reshape(4, 5);
            a["1:3"] += 5;
            a.ToArray<long>().Should().Equal(
                0L, 1L, 2L, 3L, 4L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L, 18L, 19L, 15L, 16L, 17L, 18L, 19L);
        }
    }
}
