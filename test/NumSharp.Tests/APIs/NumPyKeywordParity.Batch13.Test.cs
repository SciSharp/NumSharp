using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.IO;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     API-parity batch 13 (the owner approved fixes 1-4 and 7-15, not 5 and 6; fix 7 with NDListArray operands, fix 8
    ///     with NumPy's lower-case member names, fix 12 keeping the existing overloads): np.array of a string is an explicit
    ///     array, np.concatenate's rank check for a later 0-d array, np.delete / np.insert's typed forms, NumPy's
    ///     <c>np.broadcast</c> class constructor (over list arrays too), NpzFile's NumPy names, the np.lib.array_utils and
    ///     np.lib.format modules, np.ma.MaskedArray, np.arange's NDArray signature over NumPy's <c>PyArray_ArangeObj</c>,
    ///     NumPy's repeat / item texts, the two removed [Obsolete] members and the Shape-typed unravel_index /
    ///     ravel_multi_index. Every expected value, dtype and text is NumPy 2.4.2's (scratchpad batch13_numpy_evidence*.txt,
    ///     dbg13/arange13_numpy*.txt, format13_numpy.txt and tests13_numpy.txt), for the call exactly as the C# fix makes it.
    /// </summary>
    [TestClass]
    public class NumPyKeywordParityBatch13Tests
    {
        /// <summary>Asserts that <paramref name="act"/> throws <typeparamref name="T"/> with NumPy's exact text.</summary>
        /// <typeparam name="T">The exception type NumPy's error maps to.</typeparam>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw <typeparamref name="T"/> with that text.</exception>
        private static void Throws<T>(Action act, string message) where T : Exception
            => act.Should().Throw<T>().Which.Message.Should().Be(message);

        /// <summary>Asserts an <see cref="AxisError"/> whose text starts with NumPy's (.NET appends the parameter name to an
        ///     <see cref="ArgumentException"/>'s message).</summary>
        /// <param name="act">The call that must fail.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim.</param>
        /// <exception cref="AssertFailedException">The call does not throw an AxisError with that text.</exception>
        private static void ThrowsAxisError(Action act, string message)
            => act.Should().Throw<AxisError>().WithMessage(message + "*");

        /// <summary>The evidence's <c>d = np.arange(6)</c>.</summary>
        /// <returns>A new int64 (6,) array.</returns>
        private static NDArray Six() => np.array(new long[] { 0, 1, 2, 3, 4, 5 });

        /// <summary>The evidence's <c>x = np.arange(12).reshape(3, 4)</c> (int64).</summary>
        /// <returns>A new int64 (3, 4) array.</returns>
        private static NDArray Grid() => np.arange(12L).reshape(3, 4);

        // ------------------------------------------------------------------------------------------------------------
        // Fix 1: np.array(str) / np.asarray(str) are explicit 0-d arrays, not NumPy scalars
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>np.isscalar(np.array('x'))</c>, <c>np.isscalar(np.asarray('x'))</c> and
        ///     <c>np.isscalar(np.array('x', dtype='U3'))</c> are False; <c>repr(np.array('x'))</c> is
        ///     <c>array('x', dtype='&lt;U1')</c>.</summary>
        [TestMethod]
        public void ArrayOfString_IsAnExplicitZeroDimArray()
        {
            np.isscalar(np.array("x")).Should().BeFalse();
            np.isscalar(np.asarray("x")).Should().BeFalse();
            np.isscalar(np.array("x", np.dtype("U3"))).Should().BeFalse();
            np.array_repr(np.array("x")).Should().Be("array('x', dtype='<U1')");
            np.array("x").ndim.Should().Be(0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 2: np.concatenate - only the FIRST array's 0-d check, then the per-array rank check
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's PyArray_ConcatenateArrays order: a 0-d first array is <c>zero-dimensional arrays cannot be
        ///     concatenated</c>; a later 0-d array is the rank mismatch (NumSharp's IncorrectShapeException, NumPy's text);
        ///     without an axis the arrays ravel (<c>np.concatenate((np.zeros(2), np.array(5.0)), None)</c> is
        ///     <c>[0., 0., 5.]</c>).</summary>
        [TestMethod]
        public void Concatenate_ZeroDimChecks_FollowNumPysOrder()
        {
            Throws<ValueError>(() => np.concatenate(new[] { np.array(5.0), np.zeros(new Shape(1, 2)) }, 0),
                "zero-dimensional arrays cannot be concatenated");
            Throws<IncorrectShapeException>(() => np.concatenate(new[] { np.zeros(new Shape(1, 2)), np.array(5.0) }, 0),
                "all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 0 dimension(s)");
            np.concatenate(new[] { np.zeros(new Shape(2)), np.array(5.0) }, null).ToArray<double>().Should().Equal(0.0, 0.0, 5.0);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 3: np.delete - Slice, NDArray and the one integer form
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's <c>np.delete(d, obj)</c> for every obj kind, through the form each C# argument binds (an
        ///     integer the long form, an array or mask the NDArray form, a slice the Slice form).</summary>
        [TestMethod]
        public void Delete_EveryObjKind()
        {
            np.delete(Six(), 1).ToArray<long>().Should().Equal(0L, 2L, 3L, 4L, 5L);
            np.delete(Six(), -1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L);
            np.delete(Six(), (byte)2).ToArray<long>().Should().Equal(0L, 1L, 3L, 4L, 5L);
            np.delete(Six(), np.uint8(2)).ToArray<long>().Should().Equal(0L, 1L, 3L, 4L, 5L);
            np.delete(Six(), new[] { 1, 3 }).ToArray<long>().Should().Equal(0L, 2L, 4L, 5L);
            np.delete(Six(), np.array(1)).ToArray<long>().Should().Equal(0L, 2L, 3L, 4L, 5L);
            np.delete(Six(), new Slice(1, 4)).ToArray<long>().Should().Equal(0L, 4L, 5L);
            np.delete(Six(), new[] { true, false, true, false, false, false }).ToArray<long>().Should().Equal(1L, 3L, 4L, 5L);
            np.delete(Six(), new int[0]).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            np.delete(np.arange(6L).reshape(2, 3), 1, axis: 1).ToArray<long>().Should().Equal(0L, 2L, 3L, 5L);
        }

        /// <summary>NumPy 2.4.2's refusals: a bool scalar obj (<c>boolean array argument obj to delete must be one
        ///     dimensional and match the axis length of 6</c>), a float obj (<c>arrays used as indices must be of integer (or
        ///     boolean) type</c>) and an index past the axis (<c>index 7 is out of bounds for axis 0 with size 6</c>).</summary>
        [TestMethod]
        public void Delete_Refusals()
        {
            Throws<ValueError>(() => np.delete(Six(), true),
                "boolean array argument obj to delete must be one dimensional and match the axis length of 6");
            Throws<IndexError>(() => np.delete(Six(), 1.0), "arrays used as indices must be of integer (or boolean) type");
            new Action(() => np.delete(Six(), 7)).Should().Throw<IndexOutOfRangeException>()
                .Which.Message.Should().Be("index 7 is out of bounds for axis 0 with size 6");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 4: np.insert - Slice, NDArray and the one integer form, with NDArray values
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's <c>np.insert(d, obj, values)</c>: a scalar obj (the values' first axis moved to the axis),
        ///     a one-element list (no move), a slice, a negative and an appending index, None as NaN, a str parsed, a list
        ///     of values, and a float value truncated into int64.</summary>
        [TestMethod]
        public void Insert_EveryObjKind()
        {
            np.insert(Six(), 1, 9).ToArray<long>().Should().Equal(0L, 9L, 1L, 2L, 3L, 4L, 5L);
            np.insert(Six(), new[] { 1 }, 9).ToArray<long>().Should().Equal(0L, 9L, 1L, 2L, 3L, 4L, 5L);
            np.insert(Six(), new Slice(1, 3), new[] { 8, 9 }).ToArray<long>().Should().Equal(0L, 8L, 1L, 9L, 2L, 3L, 4L, 5L);
            np.insert(Six(), -1, 9).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 9L, 5L);
            np.insert(Six(), 6, 9).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L, 9L);
            np.insert(Six(), 1, 2.7).ToArray<long>().Should().Equal(0L, 2L, 1L, 2L, 3L, 4L, 5L);
            np.insert(Six(), 1, "7").ToArray<long>().Should().Equal(0L, 7L, 1L, 2L, 3L, 4L, 5L);
            np.insert(Six(), 1, new[] { 7, 8 }).ToArray<long>().Should().Equal(0L, 7L, 8L, 1L, 2L, 3L, 4L, 5L);
            var withNone = np.insert(np.arange(3.0), 1, (NDArray)null).ToArray<double>();
            withNone[0].Should().Be(0.0);
            double.IsNaN(withNone[1]).Should().BeTrue();
            withNone.Skip(2).Should().Equal(1.0, 2.0);
        }

        /// <summary>NumPy 2.4.2 along an axis: <c>np.insert(np.zeros((2, 2)), 1, [5, 6], axis=1)</c> and
        ///     <c>np.insert(np.zeros((2, 2)), [1], [[5], [6]], axis=1)</c> are <c>[[0, 5, 0], [0, 6, 0]]</c>,
        ///     <c>np.insert(np.zeros((2, 2)), 1, [5, 6], axis=0)</c> <c>[[0, 0], [5, 6], [0, 0]]</c>; an index past the end
        ///     is <c>index 7 is out of bounds for axis 0 with size 6</c>.</summary>
        [TestMethod]
        public void Insert_AlongAnAxis_AndOutOfBounds()
        {
            np.insert(np.zeros(new Shape(2, 2)), 1, new[] { 5, 6 }, axis: 1).ToArray<double>().Should().Equal(0.0, 5.0, 0.0, 0.0, 6.0, 0.0);
            np.insert(np.zeros(new Shape(2, 2)), new[] { 1 }, np.array(new[,] { { 5 }, { 6 } }), axis: 1).ToArray<double>()
                .Should().Equal(0.0, 5.0, 0.0, 0.0, 6.0, 0.0);
            var rows = np.insert(np.zeros(new Shape(2, 2)), 1, new[] { 5, 6 }, axis: 0);
            rows.shape.Should().Equal(3L, 2L);
            rows.ToArray<double>().Should().Equal(0.0, 0.0, 5.0, 6.0, 0.0, 0.0);
            new Action(() => np.insert(Six(), 7, 9)).Should().Throw<IndexOutOfRangeException>()
                .Which.Message.Should().Be("index 7 is out of bounds for axis 0 with size 6");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 7: new np.Broadcast(*arrays) - NumPy's class constructor, over list arrays too
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>np.broadcast(np.arange(3), np.arange(2)[:, None]).shape</c> is <c>(2, 3)</c>;
        ///     <c>np.broadcast()</c> is shape <c>()</c>, size 1, numiter 0, nd 0, and <c>list(np.broadcast())</c> is
        ///     <c>[()]</c>.</summary>
        [TestMethod]
        public void Broadcast_ClassConstructor()
        {
            var b = new np.Broadcast(np.arange(3), np.arange(2).reshape(2, 1));
            b.shape.dimensions.Should().Equal(2L, 3L);
            b.numiter.Should().Be(2);

            var none = new np.Broadcast();
            none.shape.dimensions.Should().BeEmpty();
            none.size.Should().Be(1);
            none.numiter.Should().Be(0);
            none.nd.Should().Be(0);
            var steps = none.ToList();
            steps.Should().HaveCount(1);
            steps[0].Should().BeEmpty();
        }

        /// <summary>NumPy 2.4.2 over <c>np.char.split(np.array(['a b', 'c']))</c> (an object array of lists): numiter 1, size
        ///     2, and two such operands step through <c>(['a', 'b'], ['a', 'b'])</c>, <c>(['c'], ['c'])</c>.</summary>
        [TestMethod]
        public void Broadcast_ListArrays()
        {
            var lists = np.@char.split(np.array(new[] { "a b", "c" }));
            var one = np.broadcast(new[] { lists });
            one.numiter.Should().Be(1);
            one.size.Should().Be(2);

            var two = new np.Broadcast(new[] { lists, lists });
            two.shape.dimensions.Should().Equal(2L);
            var steps = two.ToList();
            steps.Should().HaveCount(2);
            ((List<object>)steps[0][0]).Should().Equal("a", "b");
            ((List<object>)steps[0][1]).Should().Equal("a", "b");
            ((List<object>)steps[1][0]).Should().Equal("c");
            new Action(() => _ = two.iters).Should().Throw<NotSupportedException>();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 8: NpzFile - NumPy's names
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>An in-memory archive of <c>np.savez(buf, a=np.arange(3), b=np.array([1.5, 2.5]))</c>.</summary>
        /// <returns>The archive's bytes.</returns>
        private static byte[] TwoMemberArchive()
        {
            using var ms = new MemoryStream();
            np.savez(ms, new Dictionary<string, NDArray> { ["a"] = np.arange(3L), ["b"] = np.array(new[] { 1.5, 2.5 }) });
            return ms.ToArray();
        }

        /// <summary>NumPy 2.4.2: <c>files</c> / <c>keys()</c> / the <c>items()</c> keys are <c>['a', 'b']</c>,
        ///     <c>values()[1]</c> and <c>get('b')</c> are <c>[1.5, 2.5]</c>, <c>get('zz')</c> None, <c>'a' in z</c> and
        ///     <c>'a.npy' in z</c> True, <c>z['a.npy']</c> <c>[0, 1, 2]</c>, <c>(allow_pickle, max_header_size)</c>
        ///     <c>(False, 10000)</c>, <c>len(z)</c> 2, and a borrowed file's <c>fid</c> None.</summary>
        [TestMethod]
        public void NpzFile_NumPysMembers()
        {
            using var stream = new MemoryStream(TwoMemberArchive());
            using var z = new NpzFile(stream);
            z.files.Should().Equal("a", "b");
            z.keys().Should().Equal("a", "b");
            z.items().Select(p => p.key).Should().Equal("a", "b");
            z.values().ElementAt(1).ToArray<double>().Should().Equal(1.5, 2.5);
            z.get("b").ToArray<double>().Should().Equal(1.5, 2.5);
            z.get("zz").Should().BeNull();
            z.get("zz", np.array(5L)).ToArray<long>().Should().Equal(5L);
            z.ContainsKey("a").Should().BeTrue();
            z.ContainsKey("a.npy").Should().BeTrue();
            z["a.npy"].ToArray<long>().Should().Equal(0L, 1L, 2L);
            z.allow_pickle.Should().BeFalse();
            z.max_header_size.Should().Be(10000L);
            z.files.Count.Should().Be(2);
            z.fid.Should().BeNull();
            z.zip.Should().NotBeNull();
            ((NDArray)z.f.a).ToArray<long>().Should().Equal(0L, 1L, 2L);
        }

        /// <summary>NumPy 2.4.2: <c>NpzFile(f, own_fid=True).fid</c> is the file, and after <c>close()</c> <c>zip</c>,
        ///     <c>fid</c> and <c>f</c> are None (a member read then fails: NumSharp's ObjectDisposedException, NumPy's
        ///     AttributeError).</summary>
        [TestMethod]
        public void NpzFile_OwnedFid_AndClose()
        {
            var stream = new MemoryStream(TwoMemberArchive());
            var z = new NpzFile(stream, own_fid: true, allow_pickle: false, max_header_size: 10000);
            z.fid.Should().BeSameAs(stream);
            z.close();
            z.zip.Should().BeNull();
            z.fid.Should().BeNull();
            ((object)z.f).Should().BeNull();
            new Action(() => _ = z["a"]).Should().Throw<ObjectDisposedException>();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 9: np.lib.array_utils
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's <c>byte_bounds</c>: a (3, 4) int64 array spans 96 bytes; <c>x[::2, ::-1]</c> reaches
        ///     <c>(0, 96)</c> from x's start, <c>x[1:, 1:]</c> <c>(40, 96)</c>; an empty array spans 0 and a broadcast
        ///     <c>(4, 3)</c> view of <c>arange(3)</c> 24.</summary>
        [TestMethod]
        public void ArrayUtils_ByteBounds()
        {
            var x = Grid();
            var (lo, hi) = np.lib.array_utils.byte_bounds(x);
            (hi - lo).Should().Be(96L);
            var (lo2, hi2) = np.lib.array_utils.byte_bounds(x["::2, ::-1"]);
            (lo2 - lo).Should().Be(0L);
            (hi2 - lo).Should().Be(96L);
            var (lo3, hi3) = np.lib.array_utils.byte_bounds(x["1:, 1:"]);
            (lo3 - lo).Should().Be(40L);
            (hi3 - lo).Should().Be(96L);
            var (lo4, hi4) = np.lib.array_utils.byte_bounds(np.zeros(new Shape(0)));
            (hi4 - lo4).Should().Be(0L);
            var (lo5, hi5) = np.lib.array_utils.byte_bounds(np.broadcast_to(np.arange(3L), new Shape(4, 3)));
            (hi5 - lo5).Should().Be(24L);
        }

        /// <summary>NumPy 2.4.2's <c>normalize_axis_index</c> / <c>normalize_axis_tuple</c> answers and texts.</summary>
        [TestMethod]
        public void ArrayUtils_NormalizeAxis()
        {
            np.lib.array_utils.normalize_axis_index(-1, 3).Should().Be(2);
            ThrowsAxisError(() => np.lib.array_utils.normalize_axis_index(3, 3), "axis 3 is out of bounds for array of dimension 3");
            ThrowsAxisError(() => np.lib.array_utils.normalize_axis_index(3, 3, "axes"), "axes: axis 3 is out of bounds for array of dimension 3");
            ThrowsAxisError(() => np.lib.array_utils.normalize_axis_index(-4, 3), "axis -4 is out of bounds for array of dimension 3");

            np.lib.array_utils.normalize_axis_tuple(-1, 3).Should().Equal(2);
            np.lib.array_utils.normalize_axis_tuple(new[] { 0, -1 }, 3).Should().Equal(0, 2);
            Throws<ValueError>(() => np.lib.array_utils.normalize_axis_tuple(new[] { 0, 0 }, 3), "repeated axis");
            np.lib.array_utils.normalize_axis_tuple(new[] { 0, 0 }, 3, allow_duplicate: true).Should().Equal(0, 0);
            ThrowsAxisError(() => np.lib.array_utils.normalize_axis_tuple(new[] { 0, 5 }, 3, "axes"),
                "axes: axis 5 is out of bounds for array of dimension 3");
            Throws<ValueError>(() => np.lib.array_utils.normalize_axis_tuple(new[] { 0, 0 }, 3, "axes"), "repeated axis in `axes` argument");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 10: np.lib.format
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>magic(1, 0)</c> is <c>b'\x93NUMPY\x01\x00'</c>, <c>magic(3, 0)</c>
        ///     <c>b'\x93NUMPY\x03\x00'</c>; a version byte out of range is NumPy's ValueError; <c>read_magic</c> returns the
        ///     version and refuses a wrong prefix with NumPy's text.</summary>
        [TestMethod]
        public void Format_MagicAndReadMagic()
        {
            np.lib.format.magic(1, 0).Should().Equal(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 });
            np.lib.format.magic(3, 0).Should().Equal(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 3, 0 });
            Throws<ValueError>(() => np.lib.format.magic(256, 0), "major version must be 0 <= major < 256");
            Throws<ValueError>(() => np.lib.format.magic(1, 256), "minor version must be 0 <= minor < 256");
            np.lib.format.read_magic(new MemoryStream(np.lib.format.magic(2, 0))).Should().Be((2, 0));
            Throws<ValueError>(() => np.lib.format.read_magic(new MemoryStream(Encoding.ASCII.GetBytes("abcdefgh"))),
                "the magic string is not correct; expected b'\\x93NUMPY', got b'abcdef'");
        }

        /// <summary>NumPy 2.4.2: <c>dtype_to_descr</c> is <c>dtype.str</c> (<c>'&lt;i8'</c>, <c>'&gt;f4'</c>,
        ///     <c>'&gt;U3'</c>, <c>'|b1'</c>) and <c>'|O'</c> for a StringDType; <c>descr_to_dtype</c> is <c>np.dtype(descr)</c>
        ///     (byte order kept; <c>'garbage'</c> is <c>data type 'garbage' not understood</c>).</summary>
        [TestMethod]
        public void Format_Descr()
        {
            np.lib.format.dtype_to_descr(np.dtype("<i8")).Should().Be("<i8");
            np.lib.format.dtype_to_descr(np.dtype(">f4")).Should().Be(">f4");
            np.lib.format.dtype_to_descr(np.dtype(">U3")).Should().Be(">U3");
            np.lib.format.dtype_to_descr(np.dtype("U3")).Should().Be("<U3");
            np.lib.format.dtype_to_descr(np.dtype("bool")).Should().Be("|b1");
            np.lib.format.dtype_to_descr(np.dtype("T")).Should().Be("|O");
            np.lib.format.descr_to_dtype("<f8").Should().Be(np.float64);
            np.lib.format.descr_to_dtype("|b1").Should().Be(np.@bool);
            var swapped = np.lib.format.descr_to_dtype(">i2");
            swapped.str.Should().Be(">i2");
            swapped.isnative.Should().BeFalse();
            Throws<TypeError>(() => np.lib.format.descr_to_dtype("garbage"), "data type 'garbage' not understood");
            np.lib.format.drop_metadata(np.int64).Should().Be(np.int64);
        }

        /// <summary>NumPy 2.4.2's header dictionaries: <c>x</c> is <c>{'descr': '&lt;i8', 'fortran_order': False, 'shape': (3,
        ///     4)}</c>, <c>x.T</c> Fortran-ordered <c>(4, 3)</c>, <c>x[::2]</c> (copied in C order) <c>(2, 4)</c> not
        ///     Fortran.</summary>
        [TestMethod]
        public void Format_HeaderData()
        {
            var x = Grid();
            var d = np.lib.format.header_data_from_array_1_0(x);
            d["descr"].Should().Be("<i8");
            d["fortran_order"].Should().Be(false);
            ((long[])d["shape"]).Should().Equal(3L, 4L);
            var t = np.lib.format.header_data_from_array_1_0(x.T);
            t["fortran_order"].Should().Be(true);
            ((long[])t["shape"]).Should().Equal(4L, 3L);
            var s = np.lib.format.header_data_from_array_1_0(x["::2"]);
            s["fortran_order"].Should().Be(false);
            ((long[])s["shape"]).Should().Equal(2L, 4L);
            // The dictionary's shape is a copy: writing it never reshapes the array.
            ((long[])d["shape"])[0] = 99;
            x.shape.Should().Equal(3L, 4L);
        }

        /// <summary>NumPy 2.4.2: <c>write_array</c> of the (3, 4) int64 array is 224 bytes and <c>read_array</c> gives it
        ///     back; the version 1.0 header is <c>b"\x93NUMPY\x01\x00v\x00{'descr': '&lt;i8', 'fortran_order': False, 'shape':
        ///     (3, 4), }"</c> padded to 128 bytes; <c>read_magic</c> then <c>read_array_header_1_0</c> is <c>(1, 0)</c>,
        ///     <c>((3, 4), False, int64)</c>; version 2.0 reads back the same; an unknown version is NumPy's ValueError.</summary>
        [TestMethod]
        public void Format_WriteAndReadArrays()
        {
            var x = Grid();
            var file = new MemoryStream();
            np.lib.format.write_array(file, x);
            file.Length.Should().Be(224);
            file.Position = 0;
            np.lib.format.read_array(file).ToArray<long>().Should().Equal(Enumerable.Range(0, 12).Select(i => (long)i));

            var header = new MemoryStream();
            np.lib.format.write_array_header_1_0(header, np.lib.format.header_data_from_array_1_0(x));
            string expected = "\x93NUMPY\x01\x00v\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 4), }";
            expected += new string(' ', 128 - 1 - expected.Length) + "\n";
            header.ToArray().Should().Equal(Encoding.Latin1.GetBytes(expected));
            header.Position = 0;
            np.lib.format.read_magic(header).Should().Be((1, 0));
            var (shape, fortran, dtype) = np.lib.format.read_array_header_1_0(header);
            shape.Should().Equal(3L, 4L);
            fortran.Should().BeFalse();
            dtype.Should().Be(np.int64);

            var header2 = new MemoryStream();
            np.lib.format.write_array_header_2_0(header2, np.lib.format.header_data_from_array_1_0(x));
            header2.Length.Should().Be(128);
            header2.Position = 0;
            np.lib.format.read_magic(header2).Should().Be((2, 0));
            np.lib.format.read_array_header_2_0(header2).shape.Should().Equal(3L, 4L);

            Throws<ValueError>(() => np.lib.format.write_array(new MemoryStream(), x, (4, 0)),
                "we only support format version (1,0), (2,0), and (3,0), not (4, 0)");
            Throws<ValueError>(() => np.lib.format.write_array(new MemoryStream(), x, (256, 0)),
                "we only support format version (1,0), (2,0), and (3,0), not (256, 0)");
        }

        /// <summary>NumPy 2.4.2: a header of a <c>'&gt;f4'</c> descr reads back as <c>dtype('&gt;f4')</c> (the header's own
        ///     descriptor), and a header past version 1.0's 2-byte length is <c>Header length 80069 too big for
        ///     version=(1, 0)</c>.</summary>
        [TestMethod]
        public void Format_HeaderEdges()
        {
            var header = new MemoryStream();
            np.lib.format.write_array_header_1_0(header, new Dictionary<string, object>
            {
                ["descr"] = ">f4",
                ["fortran_order"] = false,
                ["shape"] = new long[] { 2 },
            });
            header.Position = 0;
            np.lib.format.read_magic(header);
            np.lib.format.read_array_header_1_0(header).dtype.str.Should().Be(">f4");

            var huge = new Dictionary<string, object>
            {
                ["descr"] = "<i8",
                ["fortran_order"] = false,
                ["shape"] = Enumerable.Repeat(100000L, 10000).ToArray(),
            };
            Throws<ValueError>(() => np.lib.format.write_array_header_1_0(new MemoryStream(), huge),
                "Header length 80069 too big for version=(1, 0)");
        }

        /// <summary>NumPy 2.4.2's constants: 64, 262144, 8, 21, <c>b'\x93NUMPY'</c> and the three expected keys.</summary>
        [TestMethod]
        public void Format_Constants()
        {
            np.lib.format.ARRAY_ALIGN.Should().Be(64);
            np.lib.format.BUFFER_SIZE.Should().Be(262144);
            np.lib.format.MAGIC_LEN.Should().Be(8);
            np.lib.format.GROWTH_AXIS_MAX_DIGITS.Should().Be(21);
            np.lib.format.MAGIC_PREFIX.Should().Equal(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
            np.lib.format.EXPECTED_KEYS.OrderBy(k => k, StringComparer.Ordinal).Should().Equal("descr", "fortran_order", "shape");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 11: np.ma.MaskedArray(...) - the class call
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>ma.MaskedArray([1., 2.], mask=[0, 1])</c> is <c>[1.0 --]</c>;
        ///     <c>ma.MaskedArray([1, 2], dtype=float32)</c> float32, <c>fill_value=7</c> kept,
        ///     <c>hard_mask=True</c> hard.</summary>
        [TestMethod]
        public void MaMaskedArray_IsMaskedArray()
        {
            var m = np.ma.MaskedArray(new[] { 1.0, 2.0 }, mask: new[] { false, true });
            m.data.ToArray<double>().Should().Equal(1.0, 2.0);
            m.mask.ToArray<bool>().Should().Equal(false, true);
            np.ma.MaskedArray(new[] { 1, 2 }, dtype: np.float32).dtype.Should().Be(np.float32);
            Convert.ToInt64(np.ma.MaskedArray(new[] { 1, 2 }, fill_value: 7).fill_value).Should().Be(7L);
            np.ma.MaskedArray(new[,] { { 1, 2 } }, hard_mask: true).hardmask.Should().BeTrue();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 12: np.arange - NumPy's signature over PyArray_ArangeObj
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's keyword forms: <c>np.arange(5, step=2)</c> <c>[0, 2, 4]</c>, <c>np.arange(1,
        ///     step=0.5)</c> <c>[0., 0.5]</c>, <c>np.arange(5, device='cpu')</c> and <c>np.arange(5, step=None)</c> int64
        ///     <c>[0..4]</c>, <c>np.arange(1, None, 2)</c> <c>[0]</c>, <c>np.arange()</c> <c>arange() requires stop to be
        ///     specified.</c></summary>
        [TestMethod]
        public void Arange_KeywordForms()
        {
            var stepped = np.arange(5, step: 2);
            stepped.dtype.Should().Be(np.int64);
            stepped.ToArray<long>().Should().Equal(0L, 2L, 4L);
            np.arange(1, step: 0.5).ToArray<double>().Should().Equal(0.0, 0.5);
            np.arange(5, device: "cpu").ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L);
            np.arange(5, step: null).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L);
            np.arange(1, null, 2).ToArray<long>().Should().Equal(0L);
            Throws<TypeError>(() => np.arange(), "arange() requires stop to be specified.");
        }

        /// <summary>NumPy 2.4.2's dtype discovery (int64 promoted with each argument's own dtype) and its scalar arithmetic:
        ///     <c>np.arange(np.float32(3.5))</c> float64 <c>[0, 1, 2, 3]</c>; <c>np.arange(0, 1, np.float32(0.1))[3]</c>
        ///     <c>0.30000000447034836</c>; <c>np.arange(np.int8(3))</c> int64; <c>np.arange(np.uint64(5))</c> float64;
        ///     <c>np.arange(np.float16(3))</c> float64; <c>np.arange(True)</c> int64 <c>[0]</c>;
        ///     <c>np.arange(np.int8(-128), np.int8(127), np.int8(100))</c> empty (int8 <c>stop - start</c> wraps);
        ///     <c>np.arange(1, 2, np.float32(0.3))</c> <c>[1.0, 1.2999999523162842, ...]</c>.</summary>
        [TestMethod]
        public void Arange_NumPyScalars()
        {
            var a = np.arange(np.float32(3.5));
            a.dtype.Should().Be(np.float64);
            a.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);
            var b = np.arange(0, 1, np.float32(0.1));
            b.size.Should().Be(10);
            b.GetDouble(3).Should().Be(0.30000000447034836);
            b.GetDouble(9).Should().Be(0.9000000134110451);
            np.arange(np.int8(3)).dtype.Should().Be(np.int64);
            np.arange(np.uint64(5)).dtype.Should().Be(np.float64);
            np.arange(np.float16(3)).ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
            var t = np.arange(true);
            t.dtype.Should().Be(np.int64);
            t.ToArray<long>().Should().Equal(0L);
            np.arange(np.int8(100), np.int8(127), np.int8(10)).ToArray<long>().Should().Equal(100L, 110L, 120L);
            np.arange(np.int8(-128), np.int8(127), np.int8(100)).size.Should().Be(0);
            np.arange(1, 2, np.float32(0.3)).ToArray<double>().Should().Equal(1.0, 1.2999999523162842, 1.5999999046325684, 1.8999998569488525);
            np.arange(np.float32(0.1), 1, 0.1).GetDouble(8).Should().Be(0.9000000134110451);
            np.arange(np.array(2.5), np.array(7)).ToArray<double>().Should().Equal(2.5, 3.5, 4.5, 5.5, 6.5);
            np.arange(np.@bool(true), 3).ToArray<long>().Should().Equal(1L, 2L);
        }

        /// <summary>NumPy 2.4.2 over complex arguments: the smaller ceiling of the quotient's two parts -
        ///     <c>np.arange(1+2j)</c> <c>[0j]</c>, <c>np.arange(0, 3+3j, 1+1j)</c> empty, <c>np.arange(0, 3+3j, 1)</c>
        ///     <c>[0j, 1+0j, 2+0j]</c>, <c>np.arange(np.complex128(3))</c> empty, <c>np.arange(1+2j, dtype=float)</c>
        ///     <c>must be real number, not complex</c>, <c>np.arange(np.complex128(3+1j), dtype=float)</c> the real part's
        ///     <c>[0., 1., 2.]</c>, <c>np.arange(0, 5, 0j)</c> <c>complex division by zero</c>.</summary>
        [TestMethod]
        public void Arange_Complex()
        {
            var one = np.arange(new Complex(1, 2));
            one.dtype.Should().Be(np.complex128);
            one.ToArray<Complex>().Should().Equal(Complex.Zero);
            np.arange(0, new Complex(3, 3), new Complex(1, 1)).size.Should().Be(0);
            np.arange(0, new Complex(3, 3), 1).ToArray<Complex>().Should().Equal(new Complex(0, 0), new Complex(1, 0), new Complex(2, 0));
            np.arange(np.complex128(3)).size.Should().Be(0);
            Throws<TypeError>(() => np.arange(new Complex(1, 2), dtype: np.float64), "must be real number, not complex");
            np.arange(np.complex128(new Complex(3, 1)), dtype: np.float64).ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
            Throws<ZeroDivisionError>(() => np.arange(0, 5, new Complex(0, 0)), "complex division by zero");
        }

        /// <summary>NumPy 2.4.2's length and argument errors: <c>np.arange(np.array([5]))</c> <c>only 0-dimensional arrays
        ///     can be converted to Python scalars</c>; a two-element and an empty array their truth-value texts; a NumPy
        ///     scalar's zero step <c>Maximum allowed size exceeded</c> (an infinite length), as is a Python int a NumPy integer
        ///     cannot hold (<c>np.arange(np.int8(5), 300)</c>); <c>np.arange(0, nan)</c> <c>arange: cannot compute
        ///     length</c>; a text argument or dtype <c>arange() not supported for inputs with DType ...</c>.</summary>
        [TestMethod]
        public void Arange_Errors()
        {
            Throws<TypeError>(() => np.arange(np.array(new[] { 5 })), "only 0-dimensional arrays can be converted to Python scalars");
            Throws<ValueError>(() => np.arange(np.array(new[] { 1, 2 })),
                "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            Throws<ValueError>(() => np.arange(np.array(new double[0])),
                "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            Throws<ValueError>(() => np.arange(np.float32(0), 5, np.float32(0)), "Maximum allowed size exceeded");
            Throws<ValueError>(() => np.arange(np.int8(5), step: 0), "Maximum allowed size exceeded");
            Throws<ValueError>(() => np.arange(np.int8(5), 300), "Maximum allowed size exceeded");
            Throws<ValueError>(() => np.arange(0, double.NaN), "arange: cannot compute length");
            Throws<TypeError>(() => np.arange("5"), "arange() not supported for inputs with DType <class 'numpy.dtypes.StrDType'>.");
            Throws<TypeError>(() => np.arange(5, dtype: "U"), "arange() not supported for inputs with DType <class 'numpy.dtypes.StrDType'>.");
        }

        /// <summary>NumPy 2.4.2 over Python ints past int64 (a C# <c>ulong</c>): <c>np.arange(18446744073709551610,
        ///     18446744073709551615, 2)</c> is three float64 elements, <c>np.arange(2**64 - 5, 2**64 - 1, 2)</c> two,
        ///     <c>np.arange(0, 2**63)</c> an empty float64 array (its length's ceiling is exactly 2**63).</summary>
        [TestMethod]
        public void Arange_PythonIntsPastInt64()
        {
            var three = np.arange(18446744073709551610UL, 18446744073709551615UL, 2UL);
            three.dtype.Should().Be(np.float64);
            three.ToArray<double>().Should().Equal(1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19);
            np.arange(18446744073709551611UL, 18446744073709551615UL, 2UL).size.Should().Be(2);
            var empty = np.arange(0, 9223372036854775808UL);
            empty.dtype.Should().Be(np.float64);
            empty.size.Should().Be(0);
            np.arange(5UL).dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     The kept <c>(double stop, DType dtype)</c> overload wins C#'s overload resolution for <c>np.arange(5, null)</c>
        ///     and <c>np.arange(5, dtype: null)</c> (an int converts to double by a standard conversion, to NDArray only by a
        ///     user-defined one), and a null dtype there is NumPy's discovery for a Python FLOAT: float64. NumPy reads 5 as a
        ///     Python int: <c>np.arange(5, None)</c> is int64 <c>[0..4]</c>. Write <c>np.arange(5)</c>, or pass a NumPy
        ///     argument (<c>np.arange((NDArray)5, dtype: null)</c>), for NumPy's int64. The owner kept the existing overloads
        ///     in batch 13.
        /// </summary>
        [TestMethod, Misaligned]
        public void Arange_IntStopWithANullDtype_BindsTheKeptDoubleOverload()
        {
            np.arange(5, null).dtype.Should().Be(np.float64, "the kept (double stop, DType dtype) overload binds");
            np.arange((NDArray)5, dtype: null).dtype.Should().Be(np.int64, "NumPy's signature reads the Python int");
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 13: np.repeat's per-element text and ndarray.item's ValueErrors
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2: <c>np.repeat([5, 1, 4], np.array([1, -1, 2]))</c> is <c>ValueError: repeats may not
        ///     contain negative values.</c>; <c>np.arange(5).item()</c>, <c>np.zeros(0).item()</c> and a masked array's are
        ///     <c>can only convert an array of size 1 to a Python scalar</c>; <c>item(1, 1, 1)</c> of a 2-D array is
        ///     <c>incorrect number of indices for array</c>.</summary>
        [TestMethod]
        public void RepeatAndItem_NumPysValueErrors()
        {
            Throws<ValueError>(() => np.repeat(np.array(new long[] { 5, 1, 4 }), np.array(new[] { 1, -1, 2 })),
                "repeats may not contain negative values.");
            Throws<ValueError>(() => np.arange(5).item(), "can only convert an array of size 1 to a Python scalar");
            Throws<ValueError>(() => np.arange(5).item<long>(), "can only convert an array of size 1 to a Python scalar");
            Throws<ValueError>(() => np.zeros(new Shape(0)).item(), "can only convert an array of size 1 to a Python scalar");
            Throws<ValueError>(() => np.ma.masked_array(np.arange(3.0)).item(), "can only convert an array of size 1 to a Python scalar");
            Throws<ValueError>(() => np.arange(6).reshape(2, 3).item(1, 1, 1), "incorrect number of indices for array");
            Throws<ValueError>(() => np.arange(6).item(1, 1), "incorrect number of indices for array");
            np.arange(6).reshape(2, 3).item(1, 2).Should().Be(5L);
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 14: the two [Obsolete] members are gone
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The owner's rule (a removed member is deleted, never left [Obsolete]): <c>NDArray.Normalize()</c> (no
        ///     NumPy function) and <c>AxisOutOfRangeException</c> (never raised: every axis check raises NumPy's
        ///     <see cref="AxisError"/>) no longer exist.</summary>
        [TestMethod]
        public void ObsoleteMembers_AreDeleted()
        {
            typeof(NDArray).GetMethod("Normalize", Type.EmptyTypes).Should().BeNull();
            typeof(NDArray).Assembly.GetType("NumSharp.AxisOutOfRangeException").Should().BeNull();
        }

        // ------------------------------------------------------------------------------------------------------------
        // Fix 15: np.unravel_index / np.ravel_multi_index take a Shape (and NumPy's intp dims)
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumPy 2.4.2's idiom <c>np.unravel_index(np.argmax(m), m.shape)</c> is <c>(2, 3)</c> for the (3, 4)
        ///     <c>arange(12.)</c>; <c>np.unravel_index([5, 11], m.shape)</c> <c>([1, 2], [1, 3])</c>;
        ///     <c>np.ravel_multi_index(([1, 2], [1, 3]), m.shape)</c> <c>[5, 11]</c>; <c>np.ravel_multi_index((1, 1),
        ///     m.shape)</c> 5 - with <c>m.shape</c> (<c>long[]</c>), <c>m.Shape</c> and an <c>int[]</c> alike.</summary>
        [TestMethod]
        public void UnravelRavel_TakeTheArraysShape()
        {
            var m = np.arange(12.0).reshape(3, 4);
            np.unravel_index(Convert.ToInt64(np.argmax(m)), m.shape).Should().Equal(2L, 3L);
            np.unravel_index(Convert.ToInt64(np.argmax(m)), m.Shape).Should().Equal(2L, 3L);
            np.unravel_index(11L, new[] { 3, 4 }).Should().Equal(2L, 3L);
            var coords = np.unravel_index(np.array(new long[] { 5, 11 }), m.shape);
            coords[0].ToArray<long>().Should().Equal(1L, 2L);
            coords[1].ToArray<long>().Should().Equal(1L, 3L);
            np.ravel_multi_index(new NDArray[] { np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 3 }) }, m.shape)
                .ToArray<long>().Should().Equal(5L, 11L);
            np.ravel_multi_index(new NDArray[] { np.array(new long[] { 1, 2 }), np.array(new long[] { 1, 3 }) }, m.Shape,
                new[] { "raise", "clip" }).ToArray<long>().Should().Equal(5L, 11L);
            np.ravel_multi_index(new long[] { 1, 1 }, m.shape).Should().Be(5L);
        }

        /// <summary>NumPy's intp dimensions, past int32: <c>np.unravel_index(5, (2**40, 3))</c> is <c>(1, 2)</c>,
        ///     <c>np.unravel_index(2**41, (2**40, 3))</c> <c>(733007751850, 2)</c>, <c>np.ravel_multi_index((2**39, 1), (2**40,
        ///     3))</c> <c>1649267441665</c>; a product past int64 is <c>dimensions are too large; ...</c> /
        ///     <c>invalid dims: ...</c>.</summary>
        [TestMethod]
        public void UnravelRavel_Int64Dimensions()
        {
            var big = new long[] { 1L << 40, 3 };
            np.unravel_index(5L, big).Should().Equal(1L, 2L);
            np.unravel_index(1L << 41, big).Should().Equal(733007751850L, 2L);
            np.ravel_multi_index(new[] { 1L << 39, 1L }, big).Should().Be(1649267441665L);
            Throws<ValueError>(() => np.unravel_index(np.array(new long[] { 0 }), new long[] { 1L << 62, 4 }),
                "dimensions are too large; arrays and shapes with a total size greater than 'intp' are not supported.");
            Throws<ValueError>(() => np.ravel_multi_index(new NDArray[] { np.array(new long[] { 0 }), np.array(new long[] { 0 }) },
                    new long[] { 1L << 62, 4 }),
                "invalid dims: array size defined by dims is larger than the maximum possible size.");
        }
    }
}
