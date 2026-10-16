using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     NumPy's C-type classes that share a storage lane with the canonical class of their width and kind —
    ///     <c>np.longdouble</c> / <c>np.clongdouble</c> on every host, Windows' <c>np.intc</c> / <c>np.uintc</c> (int32 is
    ///     the C long there), Linux's <c>np.longlong</c> / <c>np.ulonglong</c> (int64 is the C long there) — as
    ///     NumSharp's lane-sharing classes (<see cref="DTypeRegistry.CInt"/> and its siblings): their descriptors, equality,
    ///     casting, promotion, the hierarchy and its tables, and the arrays that carry them.
    /// </summary>
    /// <remarks>
    ///     Every NumPy fact was probed against <c>numpy==2.4.2</c> on Windows and on Linux (WSL), and each test asserts the
    ///     answer of the host's own C <c>long</c> model. NumSharp's long double is the 8-byte double on every host — Windows
    ///     NumPy's — so the long double answers are Windows NumPy's on Linux too (Linux NumPy's long double is the 16-byte
    ///     x87 type, <c>float128</c>).
    /// </remarks>
    [TestClass]
    public class CTypeClassesTests
    {
        /// <summary>The characters of this host's C-type classes that are classes of their own (not a lane's owner).</summary>
        private static string[] LaneSharingChars => DTypeRegistry.CLongIs32Bit ? new[] { "i", "I", "g", "G" } : new[] { "q", "Q", "g", "G" };

        // ---- descriptors --------------------------------------------------------------------------------------------

        /// <summary>
        ///     Each C-type descriptor reports NumPy's type number, character, kind, size, name, repr, scalar type and class;
        ///     the C long and the sized names stay the canonical classes.
        /// </summary>
        [TestMethod]
        public void Descriptors_ArePerModel()
        {
            void Check(string spec, int num, char ch, char kind, int size, string name, string type, string cls)
            {
                var d = np.dtype(spec);
                d.num.Should().Be(num, spec);
                d.@char.Should().Be(ch, spec);
                d.kind.Should().Be(kind, spec);
                d.itemsize.Should().Be(size, spec);
                d.name.Should().Be(name, spec);
                d.ToString(true).Should().Be($"dtype('{name}')", spec);
                ((NDScalarType)d).__name__.Should().Be(type, spec);
                d.Meta.Name.Should().Be(cls, spec);
                d.isbuiltin.Should().Be(1, spec);
            }

            Check("g", 13, 'g', 'f', 8, "float64", "longdouble", "LongDoubleDType");
            Check("longdouble", 13, 'g', 'f', 8, "float64", "longdouble", "LongDoubleDType");
            Check("G", 16, 'G', 'c', 16, "complex128", "clongdouble", "CLongDoubleDType");
            Check("d", 12, 'd', 'f', 8, "float64", "float64", "Float64DType");
            if (DTypeRegistry.CLongIs32Bit)
            {
                Check("i", 5, 'i', 'i', 4, "int32", "intc", "IntDType");
                Check("I", 6, 'I', 'u', 4, "uint32", "uintc", "UIntDType");
                Check("l", 7, 'l', 'i', 4, "int32", "int32", "Int32DType");
                Check("int32", 7, 'l', 'i', 4, "int32", "int32", "Int32DType");
                Check("q", 9, 'q', 'i', 8, "int64", "int64", "Int64DType");
                Check("longlong", 9, 'q', 'i', 8, "int64", "int64", "Int64DType");
            }
            else
            {
                Check("q", 9, 'q', 'i', 8, "int64", "longlong", "LongLongDType");
                Check("Q", 10, 'Q', 'u', 8, "uint64", "ulonglong", "ULongLongDType");
                Check("l", 7, 'l', 'i', 8, "int64", "int64", "Int64DType");
                Check("int64", 7, 'l', 'i', 8, "int64", "int64", "Int64DType");
                Check("i", 5, 'i', 'i', 4, "int32", "int32", "Int32DType");
                Check("intc", 5, 'i', 'i', 4, "int32", "int32", "Int32DType");
            }
        }

        /// <summary>
        ///     The C-type names and the type numbers resolve to the class objects: <c>np.longdouble</c> is a class of its own
        ///     everywhere, and <see cref="DType.From(NPY_TYPES)"/> gives each number's class.
        /// </summary>
        [TestMethod]
        public void NamesAndTypeNumbers_ResolveToTheClasses()
        {
            np.dtype("longdouble").Should().BeSameAs(np.longdouble);
            np.dtype("clongdouble").Should().BeSameAs(np.clongdouble);
            np.dtype("intc").Should().BeSameAs(np.intc);
            np.dtype("uintc").Should().BeSameAs(np.uintc);
            np.dtype("longlong").Should().BeSameAs(np.longlong);
            np.dtype("ulonglong").Should().BeSameAs(np.ulonglong);
            DType.From(NPY_TYPES.NPY_LONGDOUBLE).Should().BeSameAs(np.longdouble);
            DType.From(NPY_TYPES.NPY_CLONGDOUBLE).Should().BeSameAs(np.clongdouble);
            DType.From(NPY_TYPES.NPY_INT).Should().BeSameAs(np.intc);
            DType.From(NPY_TYPES.NPY_LONGLONG).Should().BeSameAs(np.longlong);
            np.dtypes.LongDoubleDType.singleton.Should().BeSameAs(np.longdouble);
            np.longdouble.Should().NotBeSameAs(np.float64);

            // 'n' / 'p' are intp, the canonical 8-byte class: 'q' on Windows, 'l' on Linux (probed).
            np.dtype("n").@char.Should().Be(DTypeRegistry.CLongIs32Bit ? 'q' : 'l');
            np.dtype("P").@char.Should().Be(DTypeRegistry.CLongIs32Bit ? 'Q' : 'L');

            // The extended-precision names are Windows NumPy's TypeError: NumSharp has no 16-byte long double.
            foreach (var spec in new[] { "float128", "f16", "complex256", "c32", "float96" })
                new Action(() => np.dtype(spec)).Should().Throw<NotSupportedException>().WithMessage($"data type '{spec}' not understood");
        }

        // ---- equality, casting, promotion ---------------------------------------------------------------------------

        /// <summary>
        ///     A C-type descriptor EQUALS the canonical descriptor of its kind and size (<c>PyArray_EquivTypes</c>) and
        ///     hashes like it, without being the same object; other pairs are unequal.
        /// </summary>
        [TestMethod]
        public void Equality_AndHash_AreNumPys()
        {
            void Equal(string a, string b)
            {
                var da = np.dtype(a);
                var db = np.dtype(b);
                (da == db).Should().BeTrue($"{a} == {b}");
                (db == da).Should().BeTrue($"{b} == {a}");
                da.GetHashCode().Should().Be(db.GetHashCode(), $"hash({a}) == hash({b})");
                ReferenceEquals(da, db).Should().BeFalse($"{a} is not {b}");
                da.Equals(b).Should().BeTrue($"{a}.Equals(\"{b}\")");
            }

            Equal("g", "d");
            Equal("G", "D");
            (np.dtype("g") == np.dtype("f")).Should().BeFalse();
            (np.dtype("i") == np.dtype("q")).Should().BeFalse();
            (np.dtype("i") == np.dtype("I")).Should().BeFalse();
            if (DTypeRegistry.CLongIs32Bit)
            {
                Equal("i", "l");
                Equal("I", "L");
                (np.dtype("q") == np.dtype("l")).Should().BeFalse();
            }
            else
            {
                Equal("q", "l");
                Equal("Q", "L");
                (np.dtype("i") == np.dtype("l")).Should().BeFalse();
            }
        }

        /// <summary>
        ///     <c>np.can_cast</c>: a C-type class and its lane's owner cast to each other under <c>'no'</c> (NumPy's
        ///     <c>numeric_copy_or_byteswap</c>, an equivalence); everything else follows the owner's rules.
        /// </summary>
        [TestMethod]
        public void Casting_IsNumPys()
        {
            np.can_cast(np.longdouble, np.float64, "no").Should().BeTrue();
            np.can_cast(np.float64, np.longdouble, "no").Should().BeTrue();
            np.can_cast(np.clongdouble, np.complex128, "no").Should().BeTrue();
            np.can_cast(np.longdouble, np.float32, "same_kind").Should().BeTrue();
            np.can_cast(np.longdouble, np.float32, "safe").Should().BeFalse();
            np.can_cast(np.float32, np.longdouble, "safe").Should().BeTrue();
            np.can_cast(np.longdouble, np.complex128, "safe").Should().BeTrue("an 8-byte long double, as on Windows NumPy");
            np.can_cast(np.longdouble, np.clongdouble, "safe").Should().BeTrue();
            np.can_cast(np.clongdouble, np.longdouble, "same_kind").Should().BeFalse();
            np.can_cast(np.clongdouble, np.float64, "same_kind").Should().BeFalse();
            np.can_cast(np.intc, np.longlong, "safe").Should().BeTrue();
            np.can_cast(np.longlong, np.intc, "safe").Should().BeFalse();
            np.can_cast(np.longlong, np.intc, "same_kind").Should().BeTrue();
            if (DTypeRegistry.CLongIs32Bit)
            {
                np.can_cast(np.intc, np.int32, "no").Should().BeTrue();
                np.can_cast(np.int32, np.intc, "no").Should().BeTrue();
                np.can_cast(np.longlong, np.int32, "no").Should().BeFalse();
            }
            else
            {
                np.can_cast(np.longlong, np.int64, "no").Should().BeTrue();
                np.can_cast(np.int64, np.longlong, "no").Should().BeTrue();
                np.can_cast(np.intc, np.int64, "no").Should().BeFalse();
            }

            // The implementation is NumPy's: the copy-or-byteswap cast at the equivalence level, a no-cast for natives.
            var impl = np.dtypes.LongDoubleDType.GetCastingImpl(np.dtypes.Float64DType);
            impl.name.Should().Be("numeric_copy_or_byteswap");
            impl.casting.Should().Be(NumSharp.Backends.Iteration.NPY_CASTING.NPY_EQUIV_CASTING);
            np.dtypes.LongDoubleDType.GetCastingImpl(np.dtypes.Float32DType).name.Should().Be("numeric_cast");
        }

        /// <summary>
        ///     <c>np.promote_types</c> follows NumPy's table of type numbers, so its answer is a CLASS: <c>longdouble</c> wins
        ///     over float64, and two canonical classes can meet in a C-type class (Windows' int16 with uint16 is the C int,
        ///     <c>'i'</c>).
        /// </summary>
        [TestMethod]
        public void Promotion_ByTypeNumber_IsNumPys()
        {
            char P(DType a, DType b) => np.promote_types(a, b).@char;
            P(np.longdouble, np.float64).Should().Be('g');
            P(np.float64, np.longdouble).Should().Be('g');
            P(np.longdouble, np.float32).Should().Be('g');
            P(np.longdouble, np.int64).Should().Be('g');
            P(np.longdouble, np.complex128).Should().Be('D', "an 8-byte long double, as on Windows NumPy");
            P(np.longdouble, np.clongdouble).Should().Be('G');
            P(np.clongdouble, np.float64).Should().Be('G');
            if (DTypeRegistry.CLongIs32Bit)
            {
                P(np.int16, np.uint16).Should().Be('i');
                P(np.int8, np.uint16).Should().Be('i');
                P(np.intc, np.int32).Should().Be('l', "both cast safely: the larger type number");
                P(np.intc, np.uintc).Should().Be('q');
                P(np.uint32, np.int8).Should().Be('q');
                (np.promote_types(np.int16, np.uint16) == np.int32).Should().BeTrue("the C int class equals int32");
            }
            else
            {
                P(np.int16, np.uint16).Should().Be('i');
                P(np.longlong, np.int64).Should().Be('q', "both cast safely: the larger type number");
                P(np.int32, np.uint32).Should().Be('l');
                P(np.uint32, np.int8).Should().Be('l');
                P(np.longlong, np.uint64).Should().Be('d');
            }

            // Weak C# literals keep the class (NEP 50): a float literal stays long double, a complex one lifts to clongdouble.
            np.result_type(np.longdouble, 1.0).@char.Should().Be('g');
            np.result_type(np.longdouble, new System.Numerics.Complex(0, 1)).@char.Should().Be('G');
            np.result_type(np.clongdouble, 1.0).@char.Should().Be('G');
            np.result_type(np.intc, 1).@char.Should().Be('i');
            np.result_type(np.longlong, 1).@char.Should().Be('q');
        }

        /// <summary>
        ///     NumPy's promotion table by type number agrees with NumSharp's storage-lane table on every pair of canonical
        ///     classes — the invariant that lets the kernels keep promoting by lane while the dtype API reports classes.
        /// </summary>
        [TestMethod]
        public void PromotionByTypeNumber_NamesTheLaneOfTheStorageTable()
        {
            var owners = DTypeRegistry.NumericStorage.Where(m => m.IsNumPyNumeric && m.OwnsTypeCode).ToArray();
            owners.Length.Should().Be(13);
            foreach (var a in owners)
            foreach (var b in owners)
            {
                var promoted = np.promote_types(a.singleton, b.singleton);
                promoted.typecode.Should().Be(np._nptypemap_arr_arr[(a.TypeCode, b.TypeCode)], $"promote({a.Name}, {b.Name})");
            }
        }

        // ---- the hierarchy and its tables ---------------------------------------------------------------------------

        /// <summary>
        ///     Each C-type class is a class of the hierarchy on its own: <c>np.issubdtype(np.float64, np.longdouble)</c> is
        ///     False, as in NumPy, while the abstract classes contain it.
        /// </summary>
        [TestMethod]
        public void Hierarchy_IsNumPys()
        {
            np.issubdtype(np.float64, np.longdouble).Should().BeFalse();
            np.issubdtype(np.longdouble, np.float64).Should().BeFalse();
            np.issubdtype(np.longdouble, np.floating).Should().BeTrue();
            np.issubdtype(np.clongdouble, np.complexfloating).Should().BeTrue();
            np.issubdtype(np.complex128, np.clongdouble).Should().BeFalse();
            np.isdtype(np.longdouble, "real floating").Should().BeTrue();
            np.isdtype(np.clongdouble, "complex floating").Should().BeTrue();
            np.isdtype(np.intc, "signed integer").Should().BeTrue();
            np.isdtype(np.longlong, "signed integer").Should().BeTrue();
            ((NDScalarType)np.longdouble).__mro__.Select(c => c.__name__).Should().Equal("longdouble", "floating", "inexact", "number", "generic", "object");
            ((NDScalarType)np.clongdouble).__mro__.Select(c => c.__name__).Should().Equal("clongdouble", "complexfloating", "inexact", "number", "generic", "object");
            if (DTypeRegistry.CLongIs32Bit)
            {
                np.issubdtype(np.int32, np.intc).Should().BeFalse();
                np.issubdtype(np.intc, np.int32).Should().BeFalse();
                np.issubdtype(np.int64, np.longlong).Should().BeTrue("the same class on Windows");
                ((NDScalarType)np.intc).__mro__.Select(c => c.__name__).Should().Equal("intc", "signedinteger", "integer", "number", "generic", "object");
            }
            else
            {
                np.issubdtype(np.int32, np.intc).Should().BeTrue("the same class on Linux");
                np.issubdtype(np.int64, np.longlong).Should().BeFalse();
                np.issubdtype(np.longlong, np.int64).Should().BeFalse();
                ((NDScalarType)np.longlong).__mro__.Select(c => c.__name__).Should().Equal("longlong", "signedinteger", "integer", "number", "generic", "object");
            }
        }

        /// <summary>
        ///     <c>np.ScalarType</c> is NumPy's list on the host's model: Python's seven classes, then every class once,
        ///     sorted by kind and size (Windows lists <c>intc</c> before <c>int32</c>, Linux <c>longlong</c> after <c>int64</c>).
        /// </summary>
        [TestMethod]
        public void ScalarType_IsNumPysList()
        {
            var names = np.ScalarType.Select(c => c.IsBuiltin ? c.__name__ : c.__module__ + "." + c.__name__).ToArray();
            string[] head = { "int", "float", "complex", "bool", "bytes", "str", "memoryview", "numpy.bool", "numpy.complex64",
                              "numpy.complex128", "numpy.clongdouble", "numpy.float16", "numpy.float32", "numpy.float64",
                              "numpy.longdouble", "numpy.int8", "numpy.int16" };
            string[] tail = DTypeRegistry.CLongIs32Bit
                ? new[] { "numpy.intc", "numpy.int32", "numpy.int64", "numpy.datetime64", "numpy.timedelta64", "numpy.object_",
                          "numpy.bytes_", "numpy.str_", "numpy.uint8", "numpy.uint16", "numpy.uintc", "numpy.uint32",
                          "numpy.uint64", "numpy.void" }
                : new[] { "numpy.int32", "numpy.int64", "numpy.longlong", "numpy.datetime64", "numpy.timedelta64",
                          "numpy.object_", "numpy.bytes_", "numpy.str_", "numpy.uint8", "numpy.uint16", "numpy.uint32",
                          "numpy.uint64", "numpy.ulonglong", "numpy.void" };
            names.Should().Equal(head.Concat(tail));
            np.ScalarType.Should().HaveCount(31);
        }

        /// <summary>
        ///     <c>np.sctypeDict</c>'s keys are NumPy's, in NumPy's order for the host's model (the C integer names differ
        ///     between Windows and Linux), less the dtype-less classes' names and Linux's extended-precision names.
        /// </summary>
        [TestMethod]
        public void SctypeDict_KeysAreNumPysPerModel()
        {
            string[] numpy = DTypeRegistry.CLongIs32Bit
                ? new[] { "bool", "float16", "float32", "float64", "longdouble", "complex64", "complex128", "clongdouble", "bytes_",
                          "str_", "void", "object_", "datetime64", "timedelta64", "int8", "byte", "uint8", "ubyte", "int16", "short",
                          "uint16", "ushort", "intc", "uintc", "int32", "long", "uint32", "ulong", "int64", "longlong", "uint64",
                          "ulonglong", "intp", "uintp", "double", "cdouble", "single", "csingle", "half", "bool_", "int_", "uint",
                          "float", "complex", "object", "bytes", "a", "int", "str", "unicode" }
                : new[] { "bool", "float16", "float32", "float64", "longdouble", "complex64", "complex128", "clongdouble", "bytes_",
                          "str_", "void", "object_", "datetime64", "timedelta64", "int8", "byte", "uint8", "ubyte", "int16", "short",
                          "uint16", "ushort", "int32", "intc", "uint32", "uintc", "int64", "long", "uint64", "ulong", "longlong",
                          "ulonglong", "intp", "uintp", "double", "cdouble", "single", "csingle", "half", "bool_", "int_", "uint",
                          "float", "complex", "object", "bytes", "a", "int", "str", "unicode", "float128", "complex256" };
            var absent = new HashSet<string> { "complex64", "csingle", "void", "object_", "object", "float128", "complex256" };
            np.sctypeDict.Keys.Should().Equal(numpy.Where(k => !absent.Contains(k)));
        }

        // ---- arrays -------------------------------------------------------------------------------------------------

        /// <summary>
        ///     An array of a C-type class keeps it through the routines that create, view, copy, cast to it, join or gather
        ///     (probed: NumPy keeps it through all of them), and an array viewed or cast to the lane's owner is the owner.
        /// </summary>
        [TestMethod]
        public void Arrays_KeepTheClass_ThroughCreationViewsCopiesJoinsAndGathers()
        {
            foreach (var c in LaneSharingChars)
            {
                DType d = np.dtype(c);
                DType owner = np.dtype(d.name);
                char want = d.@char;
                using var x = np.zeros(new Shape(3), d);
                using var y = np.zeros(new Shape(3), owner);
                var cases = new List<(string, Func<NDArray>)>
                {
                    ("zeros", () => np.zeros(new Shape(3), d)), ("ones", () => np.ones(new Shape(3), d)),
                    ("empty", () => np.empty(new Shape(3), d)), ("full", () => np.full(new Shape(3), 7, d)),
                    ("arange", () => np.arange(3.0, d)), ("zeros_like", () => np.zeros_like(x)), ("ones_like", () => np.ones_like(x)),
                    ("empty_like", () => np.empty_like(x)), ("full_like", () => np.full_like(x, 1)), ("array", () => np.array(x)),
                    ("asarray", () => np.asarray(x)), ("copy", () => x.copy()), ("np.copy", () => np.copy(x)),
                    ("slice", () => x["1:"]), ("x[0]", () => x[0]), ("T", () => x.T), ("reshape", () => x.reshape(3, 1)),
                    ("ravel", () => x.ravel()), ("flatten", () => x.flatten()), ("flatten F", () => x.reshape(3, 1).flatten('F')),
                    ("flatten K", () => x.reshape(3, 1).flatten('K')), ("broadcast_to", () => np.broadcast_to(x, new Shape(2, 3))),
                    ("astype", () => x.astype(d)), ("owner.astype", () => y.astype(d)), ("owner.view", () => y.view(d)),
                    ("concatenate", () => np.concatenate(new[] { x, x })), ("stack", () => np.stack(new[] { x, x })),
                    ("hstack", () => np.hstack(new[] { x, x })), ("fancy", () => x[np.array(new[] { 0, 1 })]),
                    ("mask", () => x[x == 0]), ("tile", () => np.tile(x, 2)), ("repeat", () => np.repeat(x, 2)),
                    ("flip", () => np.flip(x)), ("sort", () => np.sort(x)), ("roll", () => np.roll(x, 1)),
                    ("squeeze", () => x.reshape(1, 3).squeeze()), ("expand_dims", () => np.expand_dims(x, 0)),
                    ("ascontiguousarray", () => np.ascontiguousarray(x)), ("asfortranarray", () => np.asfortranarray(x)),
                };
                foreach (var (name, make) in cases)
                    make().dtype.@char.Should().Be(want, $"{name} of a '{c}' array");

                x.astype(owner).dtype.@char.Should().Be(owner.@char, $"'{c}'.astype(owner)");
                x.view(owner).dtype.@char.Should().Be(owner.@char, $"'{c}'.view(owner)");
                np.result_type(x, x).@char.Should().Be(want);
                np.result_type(new[] { x }).@char.Should().Be(want);
            }
        }

        /// <summary>
        ///     A computing routine returns its storage lane's class: NumPy's <c>x + x</c> of a long double array is a long
        ///     double array, NumSharp's the float64 the kernels allocate. The values are equal, and the dtypes compare equal.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Computing_ReturnsTheLaneClass()
        {
            using var x = np.ones(new Shape(3), np.longdouble);
            using var sum = x + x;
            sum.dtype.@char.Should().Be('d', "NumPy: 'g' (its loops are registered per class)");
            (sum.dtype == np.longdouble).Should().BeTrue();
            sum.GetDouble(0).Should().Be(2.0);
        }

        /// <summary>
        ///     The repr names the dtype for a C-type class (it is not the default of its kind), with NumPy's name, and
        ///     <c>np.finfo</c> / <c>np.iinfo</c> report the class.
        /// </summary>
        [TestMethod]
        public void PrintingAndInfo_AreNumPys()
        {
            using var g = np.zeros(new Shape(3), np.longdouble);
            g.ToString(true).Should().Be("array([0., 0., 0.], dtype=float64)");
            g.ToString(false).Should().Be("[0. 0. 0.]");
            using var cg = np.zeros(new Shape(2), np.clongdouble);
            cg.ToString(true).Should().Be("array([0.+0.j, 0.+0.j], dtype=complex128)");
            using var intc = np.zeros(new Shape(3), np.intc);
            intc.ToString(true).Should().Be("array([0, 0, 0], dtype=int32)");
            using var longlong = np.zeros(new Shape(3), np.longlong);
            longlong.ToString(true).Should().Be(DTypeRegistry.CLongIs32Bit ? "array([0, 0, 0])" : "array([0, 0, 0], dtype=int64)");

            np.finfo(np.longdouble).dtype.@char.Should().Be('g');
            np.finfo(np.clongdouble).dtype.@char.Should().Be('g');
            np.iinfo(np.intc).dtype.@char.Should().Be('i');
            np.iinfo(np.longlong).dtype.@char.Should().Be('q');
        }
    }
}
