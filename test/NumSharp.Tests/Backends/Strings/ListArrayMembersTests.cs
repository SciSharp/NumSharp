using System;
using System.Collections;
using System.Linq;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The ndarray members of numpy.char's object result (C-D3's <see cref="NDListArray"/>, NumPy's object array of
    ///     lists) that move the lists: transposition, reshaping, copies and views, selection, and the attributes of the outer
    ///     array. NumPy's object array holds 8-byte pointers and the list array int64 ranges of the same layout, so each
    ///     result's values, byte strides, flags and base are NumPy's; every literal was probed against NumPy 2.4.2 over
    ///     <c>np.char.split(np.array([['a b', 'c', 'x y'], ['', 'd e f', 'g']]))</c>, and the string oracle's section
    ///     <c>strcharlist</c> replays the full matrix over every class and core layout.
    /// </summary>
    [TestClass]
    public class ListArrayMembersTests
    {
        /// <summary>The (2, 3) split result of the probes.</summary>
        /// <returns>A new list array (the caller disposes it).</returns>
        private static NDListArray Grid() => np.@char.split(np.array(new[,] { { "a b", "c", "x y" }, { "", "d e f", "g" } }));

        /// <summary>The (4,) split result of the probes.</summary>
        /// <returns>A new list array.</returns>
        private static NDListArray Row() => np.@char.split(np.array(new[] { "a b", "c", "", "d e f" }));

        /// <summary>Python's repr of a nested .NET list of str pieces (the shape of NumPy's <c>tolist()</c>).</summary>
        /// <param name="o">A list, a string, or null.</param>
        /// <returns>The repr.</returns>
        private static string Py(object o) => o switch
        {
            null => "None",
            string s => "'" + s + "'",
            IList l => "[" + string.Join(", ", l.Cast<object>().Select(Py)) + "]",
            _ => o.ToString(),
        };

        /// <summary>The outer layout of a list array: shape, byte strides and the C / F / OWNDATA / WRITEABLE flags.</summary>
        /// <param name="l">The list array.</param>
        /// <returns>For example <c>(3, 2) (8, 24) C=False F=True own=False W=True</c>.</returns>
        private static string Layout(NDListArray l)
        {
            var f = l.flags;
            return $"({string.Join(", ", l.shape)}) ({string.Join(", ", l.strides)}) C={f.c_contiguous} F={f.f_contiguous} " +
                   $"own={f.owndata} W={f.writeable}";
        }

        /// <summary>
        ///     <c>T</c> / <c>mT</c> / <c>transpose</c> / <c>swapaxes</c> are VIEWS of the pointers (NumPy's strides
        ///     <c>(8, 24)</c>, F-contiguous, OWNDATA false, the base the array they view) sharing the lists; the axis errors are
        ///     NumPy's texts.
        /// </summary>
        [TestMethod]
        public void Transposition_ViewsThePointers()
        {
            using var a = Grid();
            const string transposed = "[[['a', 'b'], []], [['c'], ['d', 'e', 'f']], [['x', 'y'], ['g']]]";
            foreach (var t in new[] { a.T, a.mT, a.transpose(), a.transpose(1, 0), a.swapaxes(0, 1) })
            {
                using (t)
                {
                    Py(t.tolist()).Should().Be(transposed);
                    Layout(t).Should().Be("(3, 2) (8, 24) C=False F=True own=False W=True");
                    using var b = t.@base;
                    b.Should().NotBeNull();
                    Py(b.tolist()).Should().Be(Py(a.tolist()), "the base is the array the view was taken from");
                }
            }
            using var once = a.T;
            using var back = once.T;
            Layout(back).Should().Be("(2, 3) (24, 8) C=True F=False own=False W=True");

            Action repeated = () => a.transpose(0, 0);
            repeated.Should().Throw<ValueError>().Where(e => e.Message == "repeated axis in transpose");
            Action outOfRange = () => a.transpose(2, 0);
            outOfRange.Should().Throw<AxisError>().Where(e => e.Message.StartsWith("axis 2 is out of bounds for array of dimension 2"));
            using var row = Row();
            Action matrix = () => _ = row.mT;
            matrix.Should().Throw<ArgumentException>().Where(e => e.Message.StartsWith("matrix transpose with ndim < 2 is undefined"));
        }

        /// <summary>
        ///     <c>reshape</c> / <c>ravel</c> / <c>flatten</c> / <c>squeeze</c> follow NumPy's view-or-copy rules: a C-ordered
        ///     reshape or ravel of contiguous pointers is a view, an F-ordered ravel of a C array or a C ravel of a transpose a
        ///     copy, <c>flatten</c> always a copy, <c>squeeze()</c> with nothing to remove the array ITSELF.
        /// </summary>
        [TestMethod]
        public void Reshaping_FollowsNumPysViewAndCopyRules()
        {
            using var a = Grid();
            using var six = a.reshape(6);
            Py(six.tolist()).Should().Be("[['a', 'b'], ['c'], ['x', 'y'], [], ['d', 'e', 'f'], ['g']]");
            Layout(six).Should().Be("(6) (8) C=True F=True own=False W=True");
            using var three = a.reshape(3, 2);
            Layout(three).Should().Be("(3, 2) (16, 8) C=True F=False own=False W=True");
            using var fortran = a.reshape(new Shape(3, 2), 'F');
            Py(fortran.tolist()).Should().Be("[[['a', 'b'], ['d', 'e', 'f']], [[], ['x', 'y']], [['c'], ['g']]]");
            Action size = () => a.reshape(4);
            size.Should().Throw<IncorrectShapeException>().Where(e => e.Message == "cannot reshape array of size 6 into shape (4,)");

            using var ravelF = a.ravel('F');
            Py(ravelF.tolist()).Should().Be("[['a', 'b'], [], ['c'], ['d', 'e', 'f'], ['x', 'y'], ['g']]");
            ravelF.flags.owndata.Should().BeTrue("an F ravel of a C array copies the pointers");
            using var t = a.T;
            using var memory = t.ravel('K');
            memory.flags.owndata.Should().BeFalse("K order of a transpose walks its memory: a view");
            Py(memory.tolist()).Should().Be(Py(six.tolist()));
            using var flat = a.flatten();
            flat.flags.owndata.Should().BeTrue("flatten always copies");

            a.squeeze().Should().BeSameAs(a, "with no length-1 axis NumPy returns the receiver itself");
            using var lifted = a.reshape(1, 2, 3);
            using var squeezed = lifted.squeeze();
            Layout(squeezed).Should().Be("(2, 3) (24, 8) C=True F=False own=False W=True");
            Action notOne = () => a.squeeze(0);
            notOne.Should().Throw<ValueError>().Where(e => e.Message == "cannot select an axis to squeeze out which has size not equal to one");
        }

        /// <summary>
        ///     <c>copy</c> is a SHALLOW copy in the order asked (F: strides <c>(8, 16)</c>), owning its pointers (no base);
        ///     <c>view</c> is a view of the same pointers.
        /// </summary>
        [TestMethod]
        public void CopyAndView_OwnOrShareThePointers()
        {
            using var a = Grid();
            using var c = a.copy();
            Layout(c).Should().Be("(2, 3) (24, 8) C=True F=False own=True W=True");
            c.@base.Should().BeNull();
            using var f = a.copy('F');
            Layout(f).Should().Be("(2, 3) (8, 16) C=False F=True own=True W=True");
            Py(f.tolist()).Should().Be(Py(a.tolist()));
            using var v = a.view();
            Layout(v).Should().Be("(2, 3) (24, 8) C=True F=False own=False W=True");
            using var row = Row();
            using var reversed = row["::-1"];
            using var reversedCopy = reversed.copy();
            Layout(reversedCopy).Should().Be("(4) (8) C=True F=True own=True W=True");
            Py(reversedCopy.tolist()).Should().Be("[['d', 'e', 'f'], [], ['c'], ['a', 'b']]");
        }

        /// <summary>
        ///     <c>take</c> / <c>compress</c> / <c>repeat</c> copy the pointers they select (the lists shared); one index gives a
        ///     0-d list array, whose <c>item()</c> is NumPy's list; a <c>diagonal</c> is a READ-ONLY view; the errors are
        ///     NumPy's.
        /// </summary>
        [TestMethod]
        public void Selection_CopiesThePointersItSelects()
        {
            using var a = Grid();
            using var taken = a.take(new long[] { 0, 2 });
            Py(taken.tolist()).Should().Be("[['a', 'b'], ['x', 'y']]");
            taken.flags.owndata.Should().BeTrue();
            using var column = a.take(new long[] { 0, 2 }, 1);
            Py(column.tolist()).Should().Be("[[['a', 'b'], ['x', 'y']], [[], ['g']]]");
            using var wrapped = a.take(7, mode: "wrap");
            wrapped.ndim.Should().Be(0);
            Py(wrapped.item()).Should().Be("['c']");
            using var clipped = a.take(7, mode: "clip");
            Py(clipped.item()).Should().Be("['g']");
            Action bounds = () => a.take(7);
            bounds.Should().Throw<IndexError>().Where(e => e.Message == "index 7 is out of bounds for axis 0 with size 6");

            using var compressed = a.compress(new[] { true, false, true }, 1);
            Py(compressed.tolist()).Should().Be("[[['a', 'b'], ['x', 'y']], [[], ['g']]]");
            Action past = () => a.compress(new long[] { 1, 0, 1, 1, 1, 1, 1 });
            past.Should().Throw<IndexError>().Where(e => e.Message == "index 6 is out of bounds for axis 0 with size 6");

            using var repeated = a.repeat(new long[] { 1, 2 }, 0);
            Py(repeated.tolist()).Should().Be("[[['a', 'b'], ['c'], ['x', 'y']], [[], ['d', 'e', 'f'], ['g']], [[], ['d', 'e', 'f'], ['g']]]");
            Layout(repeated).Should().Be("(3, 3) (24, 8) C=True F=False own=True W=True");

            using var diagonal = a.diagonal();
            Py(diagonal.tolist()).Should().Be("[['a', 'b'], ['d', 'e', 'f']]");
            Layout(diagonal).Should().Be("(2) (32) C=False F=False own=False W=False");
            using var above = a.diagonal(1);
            Py(above.tolist()).Should().Be("[['c'], ['g']]");
            using var row = Row();
            Action flat = () => row.diagonal();
            flat.Should().Throw<ValueError>().Where(e => e.Message.StartsWith("diag requires an array of at least two dimensions"));
        }

        /// <summary>
        ///     The outer array's attributes: byte strides of 8-byte pointers, <c>itemsize</c> 8, <c>nbytes</c> the pointers'
        ///     bytes, <c>device</c> <c>"cpu"</c>; <c>to_device("cpu")</c> is the receiver itself, another device NumPy's
        ///     error.
        /// </summary>
        [TestMethod]
        public void Attributes_DescribeThePointers()
        {
            using var a = Grid();
            a.strides.Should().Equal(24L, 8L);
            a.itemsize.Should().Be(8);
            a.nbytes.Should().Be(48);
            a.device.Should().Be("cpu");
            a.to_device("cpu").Should().BeSameAs(a);
            Action gpu = () => a.to_device("gpu");
            gpu.Should().Throw<ArgumentException>().Where(e => e.Message.StartsWith("Unsupported device: gpu. Only 'cpu' is accepted."));
            using var zero = np.@char.split(np.array("p q"));
            zero.strides.Should().BeEmpty();
            zero.nbytes.Should().Be(8);
        }

        /// <summary>
        ///     A result outlives the array it came from: a view and a copy hold their own references on the shared pieces, so
        ///     disposing the source leaves them readable; a disposed list array refuses every member.
        /// </summary>
        [TestMethod]
        public void Results_OutliveTheirSource()
        {
            var a = Grid();
            using var t = a.T;
            using var c = a.copy();
            a.Dispose();
            Py(t.tolist()).Should().Be("[[['a', 'b'], []], [['c'], ['d', 'e', 'f']], [['x', 'y'], ['g']]]");
            Py(c.tolist()).Should().Be("[[['a', 'b'], ['c'], ['x', 'y']], [[], ['d', 'e', 'f'], ['g']]]");
            Action disposed = () => _ = a.T;
            disposed.Should().Throw<ObjectDisposedException>();
        }
    }
}
