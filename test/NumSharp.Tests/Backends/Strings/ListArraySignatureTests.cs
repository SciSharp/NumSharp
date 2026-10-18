using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     <see cref="NDListArray"/>'s members as NumPy's one signature each (the one-signature review, 2026-10-19): mean's
    ///     <c>where=</c> in a <c>dtype=</c> loop (NumPy computes it), <c>reshape(copy=False)</c>'s view rule, and clip's
    ///     <c>where=</c> with an <c>out=</c> — the cases the review's oracle section found wrong. Every literal was probed
    ///     against NumPy 2.4.2 over <c>np.char.split(np.array(['a b', 'c', '', 'd e f']))</c> (<c>r</c>) and
    ///     <c>np.char.split(np.array([['a b', 'c', 'x y'], ['', 'd e f', 'g']]))</c> (<c>a</c>); the string oracle's section
    ///     <c>strcharlist</c> (its <c>sig:</c> cases) replays the whole parameter matrix.
    /// </summary>
    [TestClass]
    public class ListArraySignatureTests
    {
        /// <summary>The (4,) split result <c>r</c>.</summary>
        /// <returns>A new list array (the caller disposes it).</returns>
        private static NDListArray R() => np.@char.split(np.array(new[] { "a b", "c", "", "d e f" }));

        /// <summary>The (2, 3) split result <c>a</c>.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray A() => np.@char.split(np.array(new[,] { { "a b", "c", "x y" }, { "", "d e f", "g" } }));

        /// <summary>The empty (0,) split result.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray E() => np.@char.split(np.array(new string[0]));

        /// <summary>Python's repr of a nested .NET list of str pieces, as NumPy's <c>tolist()</c> prints.</summary>
        /// <param name="o">A list or a string.</param>
        /// <returns>The repr.</returns>
        private static string Py(object o) => o switch
        {
            string s => "'" + s + "'",
            IList l => "[" + string.Join(", ", l.Cast<object>().Select(Py)) + "]",
            _ => o.ToString(),
        };

        /// <summary>
        ///     <c>_mean</c> with a mask in a <c>dtype=</c> loop: the loop's masked add (the bool loop ORs the selected lists'
        ///     truths) divided by each output's count of selected elements — <c>True / 2</c> is True, <c>False / 1</c> False,
        ///     <c>False / 0</c> (NaN) True; into a float64 out= the quotient itself. Every element is still cast first, so a
        ///     float loop is the conversion error whatever the mask selects. (The members used to refuse every mask with a
        ///     loop.)
        /// </summary>
        [TestMethod]
        public void Mean_WhereInADtypeLoop_DividesTheMaskedSumByEachOutputsCount()
        {
            using var r = R();
            using (var m = np.array(new[] { true, false, true, false }))
            using (var mean = r.mean(dtype: np.@bool, where: m))
                mean.GetBoolean().Should().BeTrue();
            using (var m = np.array(new[] { false, false, true, false }))
            using (var mean = r.mean(dtype: np.@bool, where: m))
                mean.GetBoolean().Should().BeFalse("the one selected list is empty: False / 1");
            using (var m = np.array(new[] { false, false, false, false }))
            using (var mean = r.mean(dtype: np.@bool, where: m))
                mean.GetBoolean().Should().BeTrue("False / 0 is NaN, whose bool is True");
            using (var m = np.array(new[] { true, false, true, false }))
            using (var o = np.zeros(Shape.Scalar, np.float64))
            {
                var mean = r.mean(dtype: np.@bool, @out: o, where: m);
                ReferenceEquals(mean, o).Should().BeTrue();
                o.GetDouble().Should().Be(0.5, "the OR-sum True as 1.0, divided by the 2 selected");
            }
            using var a = A();
            using (var m = np.array(new[,] { { false, false, true }, { true, true, false } }))
            {
                using (var mean = a.mean(0, np.@bool, where: m))
                    mean.ToArray<bool>().Should().Equal(false, true, true);
                using var o = np.zeros(new Shape(3), np.float64);
                var into = a.mean(0, np.@bool, o, where: m);
                ReferenceEquals(into, o).Should().BeTrue();
                o.ToArray<double>().Should().Equal(0.0, 1.0, 1.0);
            }
            using (var m = np.array(new[] { true, false, true }))
            using (var mean = a.mean(1, np.@bool, keepdims: true, where: m))
            {
                mean.shape.Should().Equal(2L, 1L);
                mean.ToArray<bool>().Should().Equal(true, true);
            }
            using (var m = np.array(new[] { true, false, true, false }))
            {
                Action cast = () => r.mean(dtype: np.float64, where: m);
                cast.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence.");
            }
            using var e = E();
            using (var m = np.zeros(new Shape(0), np.@bool))
            {
                using (var mean = e.mean(dtype: np.float64, where: m))
                    double.IsNaN(mean.GetDouble()).Should().BeTrue();
                using (var mean = e.mean(dtype: np.@bool, where: m))
                    mean.GetBoolean().Should().BeTrue();
            }
        }

        /// <summary>
        ///     <c>reshape(copy=False)</c> is <c>_reshape_with_copy_arg</c>'s rule: an array contiguous in the requested order is
        ///     reshaped as a view without trying (a reshape to its own shape, an empty array), any other needs a no-copy
        ///     reshape to exist. (A reshape to the same shape and an empty array's reshape raised, their view not being the
        ///     same buffer object.)
        /// </summary>
        [TestMethod]
        public void Reshape_CopyFalse_IsNumPysViewRule()
        {
            using var r = R();
            using (var same = r.reshape(new Shape(4), 'C', false))
            {
                same.flags.owndata.Should().BeFalse();
                Py(same.tolist()).Should().Be("[['a', 'b'], ['c'], [], ['d', 'e', 'f']]");
            }
            using var e = E();
            using (var empty = e.reshape(new Shape(0), 'C', false))
                empty.flags.owndata.Should().BeFalse();
            using var a = A();
            using var t = a.T;
            Action c = () => t.reshape(new Shape(6), 'C', false);
            c.Should().Throw<ValueError>().WithMessage("Unable to avoid creating a copy while reshaping.");
            using (var f = t.reshape(new Shape(6), 'F', false))
            {
                f.flags.owndata.Should().BeFalse();
                Py(f.tolist()).Should().Be("[['a', 'b'], ['c'], ['x', 'y'], [], ['d', 'e', 'f'], ['g']]");
            }
        }

        /// <summary>
        ///     clip into an out= of another heap: <c>where=True</c> (no mask) stores every clipped list, a mask only the selected
        ///     ones (the others keep the out's lists). (A list out of another heap without a mask raised a
        ///     NullReferenceException.)
        /// </summary>
        [TestMethod]
        public void Clip_WhereWithOut_StoresTheSelectedLists()
        {
            using var r = R();
            using var lo = np.@char.split(np.array(new[] { "c" }));
            using var hi = np.@char.split(np.array(new[] { "c" }));
            using (var o = np.@char.split(np.array(new[] { "q", "r", "s", "t" })))
            {
                var res = r.clip(lo, hi, o, where: true);
                ReferenceEquals(res, o).Should().BeTrue();
                Py(o.tolist()).Should().Be("[['c'], ['c'], ['c'], ['c']]");
            }
            using (var o = np.@char.split(np.array(new[] { "q", "r", "s", "t" })))
            using (var m = np.array(new[] { true, false, false, true }))
            {
                var res = r.clip(lo, hi, o, where: m);
                ReferenceEquals(res, o).Should().BeTrue();
                Py(o.tolist()).Should().Be("[['c'], ['r'], ['s'], ['c']]");
            }
        }
    }
}
