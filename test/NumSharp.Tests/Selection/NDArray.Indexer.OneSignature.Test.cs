using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NumSharp.Generic;

namespace NumSharp.Tests
{
    /// <summary>
    ///     NumPy's ONE indexing signature — <c>ndarray.__getitem__(key)</c> / <c>__setitem__(key, value)</c> — as the one C#
    ///     indexer of every array class, and the single-signature <c>__getitem__</c> / <c>__setitem__</c> members. Every
    ///     expectation was probed on NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>What replaced what.</b> <see cref="NDArray"/> had five public indexers (<c>this[string]</c>,
    ///     <c>this[params Slice[]]</c>, <c>this[NDArray&lt;bool&gt;]</c>, <c>this[params NDArray&lt;int&gt;[]]</c>,
    ///     <c>this[long*, int]</c>) beside <c>this[params object[]]</c>; <see cref="NDArray{T}"/> four (two typed SCALAR
    ///     ones, <c>T this[params long[]]</c> / <c>T this[int[]]</c>); the subclasses up to eight. C# picked among them at
    ///     compile time, so one call could reach different code depending on the static type of an argument. Now each class
    ///     has one indexer (<see cref="NDMatrix"/> keeps its typed element access <c>m[row, column]</c>), and the one
    ///     indexer sorts the key at run time into the bodies the old indexers had, so every call keeps its result.
    ///     </para>
    ///     <para>
    ///     <b>The traps these tests pin.</b> (1) A lone <see cref="long"/>, <see cref="short"/> or <see cref="char"/> used to
    ///     bind <c>this[params Slice[]]</c> at compile time; the general path has no case for them, so the dispatch must
    ///     sort them as integers. (2) <see cref="NDArray{T}"/>'s typed selection must keep the ownership of the untyped one:
    ///     a gather owns its data (NumPy's <c>flags.owndata</c> true) where a typed alias of it would read as a view —
    ///     found by the strings oracle (<c>np.ma.flatten_mask</c> over <c>S</c> ends in a typed <c>nonzero[keep]</c>).
    ///     (3) An <c>int[]</c> key of <c>__getitem__</c> is NumPy's LIST — an index array — where the removed
    ///     <c>__getitem__(params int[])</c> read coordinates. (4) <see cref="NDCharArray"/> and <see cref="NDMatrix"/> hide
    ///     the indexer instead of overriding it: through an <see cref="NDArray"/> reference they index as plain arrays.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class NDArrayIndexerOneSignatureTests
    {
        /// <summary>The public instance indexers a type declares itself (inherited ones excluded).</summary>
        /// <param name="type">The type.</param>
        /// <returns>The declared indexers.</returns>
        private static PropertyInfo[] DeclaredIndexers(Type type)
            => type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                   .Where(p => p.GetIndexParameters().Length > 0).ToArray();

        /// <summary>The public instance methods named <paramref name="name"/> a type declares itself.</summary>
        /// <param name="type">The type.</param>
        /// <param name="name">The method name.</param>
        /// <returns>The declared methods.</returns>
        private static MethodInfo[] Declared(Type type, string name)
            => type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                   .Where(m => m.Name == name).ToArray();

        // ------------------------------------------------------------------------------------------------ the surface

        [TestMethod]
        public void EveryArrayClass_DeclaresOneIndexer_WithNumPysKey()
        {
            var nd = DeclaredIndexers(typeof(NDArray)).Should().ContainSingle().Subject;
            nd.GetIndexParameters().Select(p => p.ParameterType).Should().Equal(typeof(object[]));
            nd.GetIndexParameters()[0].Name.Should().Be("key");
            nd.GetMethod.IsVirtual.Should().BeTrue("NDArray<T> overrides it to type its result");

            // NDArray<T>: the override narrows the GETTER only (a read-write override must keep NDArray's type).
            var typed = DeclaredIndexers(typeof(NDArray<double>)).Should().ContainSingle().Subject;
            typed.PropertyType.Should().Be(typeof(NDArray<double>));
            typed.GetIndexParameters().Select(p => p.ParameterType).Should().Equal(typeof(object[]));
            // A covariant-return override is a new slot with an explicit override (reflection's GetBaseDefinition does not
            // see through it): it shows as a virtual getter and no setter of its own (the inherited one is NDArray's).
            typed.GetMethod.IsVirtual.Should().BeTrue();
            typed.CanWrite.Should().BeFalse("the setter is NDArray's");

            var chararray = DeclaredIndexers(typeof(NDCharArray)).Should().ContainSingle().Subject;
            chararray.GetIndexParameters().Select(p => p.ParameterType).Should().Equal(typeof(object[]));
            chararray.GetMethod.IsVirtual.Should().BeFalse("it hides NDArray's indexer instead of overriding it");

            // NDMatrix: the general indexer, and the element access typed as the scalar it returns.
            DeclaredIndexers(typeof(NDMatrix)).Select(p => string.Join(",", p.GetIndexParameters().Select(q => q.ParameterType.Name)))
                .Should().BeEquivalentTo("Object[]", "Int64,Int64");

            var lists = DeclaredIndexers(typeof(NDListArray)).Should().ContainSingle().Subject;
            lists.GetIndexParameters().Select(p => p.ParameterType).Should().Equal(typeof(object[]));
        }

        [TestMethod]
        public void TheDunders_HaveOneSignatureEach()
        {
            Declared(typeof(NDArray), "__getitem__").Select(m => m.GetParameters().Single().ParameterType)
                .Should().Equal(typeof(object));
            Declared(typeof(NDArray), "__setitem__").Select(m => string.Join(",", m.GetParameters().Select(p => p.ParameterType.Name)))
                .Should().Equal("Object,Object");
            Declared(typeof(NDCharArray), "__getitem__").Should().ContainSingle()
                .Which.GetParameters().Single().Name.Should().Be("obj", "NumPy's chararray.__getitem__(self, obj)");
            Declared(typeof(NDCharArray), "__setitem__").Should().BeEmpty("chararray inherits ndarray's __setitem__");
            Declared(typeof(NDMatrix), "__getitem__").Should().ContainSingle()
                .Which.GetParameters().Single().Name.Should().Be("index", "NumPy's matrix.__getitem__(self, index)");
            Declared(typeof(NDMatrix), "__setitem__").Should().ContainSingle();
        }

        // ------------------------------------------------------------------------------------------- the dispatch

        [TestMethod]
        public void EveryIntegerType_IsAnInteger_Index()
        {
            // a = np.arange(12).reshape(3, 4): a[1] = [4, 5, 6, 7], a[1, 2] = 6, a[-1, ::2] = [8, 10]
            var a = np.arange(12).reshape(3, 4);
            foreach (object one in new object[] { 1, 1L, (short)1, (ushort)1, (byte)1, (sbyte)1, 1u, (nint)1 })
            {
                a[one].ToArray<long>().Should().Equal(new long[] { 4, 5, 6, 7 }, $"a[({one.GetType().Name})1]");
                a[one, 2].GetInt64().Should().Be(6);
            }
            // ulong has no implicit conversion to long: the general path's own case (NumPy indexes with a uint64 too).
            a[1UL].ToArray<long>().Should().Equal(4, 5, 6, 7);
            a[-1L, "::2"].ToArray<long>().Should().Equal(8, 10);
            a[(short)-1, Slice.All].ToArray<long>().Should().Equal(8, 9, 10, 11);
        }

        [TestMethod]
        public void SliceNotation_LoneString_IsTheWholeIndex_AnEntryString_IsOneSlice()
        {
            var a = np.arange(12).reshape(3, 4);
            a["1, 2"].GetInt64().Should().Be(6, "a lone string holds the whole index");
            a["::2, 1:3"].shape.Should().Equal(2L, 2L);
            a["1", "1:3"].ToArray<long>().Should().Equal(5, 6);
            a[new Slice[] { Slice.Index(2), new Slice(null, null, -1) }].ToArray<long>().Should().Equal(11, 10, 9, 8);
        }

        [TestMethod]
        public void BasicIndex_IsAView_AndTheSetterWritesThrough()
        {
            var a = np.arange(12).reshape(3, 4);
            var row = a[1L];
            row.flags.owndata.Should().BeFalse("a basic index is a view");
            row[0] = 40;
            a.GetInt64(1, 0).Should().Be(40);
            a[(short)2, "1:3"] = -1;
            a[2].ToArray<long>().Should().Equal(8, -1, -1, 11);
        }

        [TestMethod]
        public void AMask_TypedOrNot_SelectsTheSame_AndOwnsIt()
        {
            // a[m] = [0, 2, 4, 6, 8, 10], owndata True
            var a = np.arange(12).reshape(3, 4);
            NDArray<bool> typed = a % 2 == 0;
            NDArray plain = typed.copy();
            foreach (NDArray m in new[] { (NDArray)typed, plain })
            {
                var r = a[m];
                r.ToArray<long>().Should().Equal(0, 2, 4, 6, 8, 10);
                r.flags.owndata.Should().BeTrue();
            }
            a[typed] = 0;
            a.ToArray<long>().Should().Equal(0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11);
        }

        [TestMethod]
        public void TheSetter_TestsWriteabilityFirst_ForEveryKeyKind()
        {
            // One key of each kind the old typed indexers took: an integer, slice notation, slices, a typed mask, an index
            // array. NumPy's setitem raises "assignment destination is read-only" before it looks at the index.
            var ro = np.broadcast_to(np.arange(4), new Shape(3, 4));
            foreach (object[] key in new[]
                     {
                         new object[] { 0 }, new object[] { "0:2" }, new object[] { Slice.All, 1 },
                         new object[] { np.arange(12).reshape(3, 4) > 5 },
                         new object[] { np.array(new[] { 0, 2 }) },
                     })
            {
                new Action(() => ro[key] = 1).Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }
        }

        [TestMethod]
        public void ANullKey_IsRefused()
        {
            var a = np.arange(3);
            new Action(() => _ = a[(object[])null]).Should().Throw<ArgumentNullException>();
            new Action(() => a[(object[])null] = 1).Should().Throw<ArgumentNullException>();
            new Action(() => a.__getitem__(null)).Should().Throw<ArgumentNullException>();
        }

        // ---------------------------------------------------------------------------------------------- the dunders

        [TestMethod]
        public void GetItem_TakesPythonsKeys()
        {
            var a = np.arange(12).reshape(3, 4);
            a.__getitem__(1).ToArray<long>().Should().Equal(4, 5, 6, 7);
            a.__getitem__(2L).ToArray<long>().Should().Equal(8, 9, 10, 11);
            a.__getitem__((2, 1)).GetInt64().Should().Be(9, "a C# tuple is NumPy's index tuple");
            a.__getitem__(new object[] { 2, 1 }).GetInt64().Should().Be(9, "an object[] is the index tuple");
            a.__getitem__("-1, ::2").ToArray<long>().Should().Equal(8, 10);
            a.__getitem__(Slice.Index(0)).ToArray<long>().Should().Equal(0, 1, 2, 3);
            // NumPy's list key: an index array or a mask (the removed params int[] form read COORDINATES).
            a.__getitem__(new[] { 0, 2 }).ToArray<long>().Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);
            a.__getitem__(new[] { true, false, true }).ToArray<long>().Should().Equal(0, 1, 2, 3, 8, 9, 10, 11);
        }

        [TestMethod]
        public void SetItem_TakesPythonsKeys_AndAnyValue()
        {
            var a = np.arange(12).reshape(3, 4);
            a.__setitem__((0, 1), 99);
            a[0].ToArray<long>().Should().Equal(0, 99, 2, 3);
            a.__setitem__(new[] { 0, 2 }, -1);
            a.ToArray<long>().Should().Equal(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1);
            a.__setitem__(1, new[] { 7, 8, 9, 10 });
            a[1].ToArray<long>().Should().Equal(7, 8, 9, 10);
            a.__setitem__("2, 1:3", 5L);
            a[2].ToArray<long>().Should().Equal(-1, 5, 5, -1);
            // A C# literal is NumPy's Python scalar: range-checked, never wrapped.
            var i8 = np.zeros(new Shape(3), np.int8);
            new Action(() => i8.__setitem__(0, 300)).Should().Throw<OverflowException>();
        }

        // --------------------------------------------------------------------------------------------- NDArray<T>

        [TestMethod]
        public void TypedArray_EverySelectionIsTyped_AndAnElementIsA0dView()
        {
            NDArray<double> a = np.arange(6.0).reshape(2, 3).MakeGeneric<double>();
            NDArray<double> row = a[1];
            row.ToArray<double>().Should().Equal(3.0, 4.0, 5.0);
            NDArray<double> element = a[1, 2];
            element.ndim.Should().Be(0);
            ((double)element).Should().Be(5.0);
            element.SetAtIndex(50.0, 0);
            ((double)a[1, 2]).Should().Be(50.0, "an element is a VIEW, as for every array");
            new Action(() => _ = element[0]).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 0-dimensional, but 1 were indexed");
        }

        [TestMethod]
        public void TypedArray_AGather_OwnsItsData_AViewDoesNot()
        {
            // a[[2, 0]].flags.owndata True, a[m].flags.owndata True, a[1:].flags.owndata False
            NDArray<long> a = np.arange(6).MakeGeneric<long>();
            NDArray<bool> m = a > 2;
            NDArray<long> masked = a[m];
            masked.ToArray<long>().Should().Equal(3, 4, 5);
            masked.flags.owndata.Should().BeTrue("a typed alias of the gather would read as a view");
            NDArray<long> gathered = a[np.array(new[] { 2, 0 })];
            gathered.ToArray<long>().Should().Equal(2, 0);
            gathered.flags.owndata.Should().BeTrue();
            a["1:"].flags.owndata.Should().BeFalse();
        }

        [TestMethod]
        public void TypedArray_ThroughAnNDArrayReference_StillIndexesTyped()
        {
            NDArray<int> typed = np.arange(4).astype(np.int32).MakeGeneric<int>();
            NDArray plain = typed;
            plain[1].Should().BeOfType<NDArray<int>>();
            plain["1:3"].Should().BeOfType<NDArray<int>>();
            plain[np.array(new[] { true, false, true, false })].Should().BeOfType<NDArray<int>>();
        }

        [TestMethod]
        public void TypedArray_ATypedValue_Assigns()
        {
            NDArray<double> a = np.zeros(4).MakeGeneric<double>();
            a[0] = 3.14;
            a[1] = 2;            // an int widens to double, then converts to a 0-d NDArray<double>
            NDArray<bool> mask = a > 1;
            mask[mask] = false;
            ((double)a[0]).Should().Be(3.14);
            ((double)a[1]).Should().Be(2.0);
            mask.ToArray<bool>().Should().Equal(false, false, false, false);
            // Any other value goes through an NDArray reference or __setitem__.
            NDArray u = a;
            u["2:"] = np.array(new[] { 5.0, 6.0 });
            a.__setitem__(0, np.array(7.0));
            a.ToArray<double>().Should().Equal(7.0, 2.0, 5.0, 6.0);
        }

        // ----------------------------------------------------------------------------- NDCharArray and NDMatrix

        [TestMethod]
        public void CharArray_OneIndexer_StripsElements_AndIsStatic()
        {
            // c = np.char.array([['a ', 'b'], ['c', 'd ']]): c[1, 1] 'd', c.__getitem__((1, 1)) 'd',
            // np.asarray(c)[1, 1] 'd ' (a plain view keeps the trailing space)
            var c = np.@char.array(new[,] { { "a ", "b" }, { "c", "d " } });
            c[1, 1].ToString().Should().Be("d");
            c[(short)1, 1L].ToString().Should().Be("d");
            c.__getitem__((1, 1)).ToString().Should().Be("d");
            NDCharArray rows = c.__getitem__(new[] { 1, 0 });
            rows.shape.Should().Equal(2L, 2L);
            ((string)rows[0, 0].ToString()).Should().Be("c");
            NDArray plain = c;
            plain[1, 1].Should().NotBeOfType<NDCharArray>();
            plain[1, 1].ToString().Should().Be("d ", "through an NDArray reference a chararray indexes as a plain array");
        }

        [TestMethod]
        public void Matrix_OneIndexer_PlusTheTypedElementAccess_AndIsStatic()
        {
            // m = np.matrix([[1, 2], [3, 4]]): m[0] matrix([[1, 2]]), m[0, 1] 2, m.__getitem__((1, 0)) 3,
            // m[:, 0] matrix([[1], [3]])
            var m = np.matrix(new[,] { { 1L, 2L }, { 3L, 4L } });
            NDMatrix row = m[0];
            row.shape.Should().Equal(1L, 2L);
            NDArray element = m[0, 1];      // two ints bind m[long row, long column]
            element.ndim.Should().Be(0);
            element.GetInt64().Should().Be(2);
            m.__getitem__((1, 0)).GetInt64().Should().Be(3);
            m[":", 0].shape.Should().Equal(2L, 1L);
            m.__getitem__(0).shape.Should().Equal(1L, 2L);
            NDArray plain = m;
            plain[0].shape.Should().Equal(new long[] { 2 }, "through an NDArray reference a matrix indexes as a plain 2-d array");
        }

        // ------------------------------------------------------------------------------------------- NDListArray

        [TestMethod]
        public void ListArray_OneIndexer_ElementIsTheList_AdvancedIndexSharesTheLists()
        {
            // r = np.char.split(np.array(['a b', 'c', 'd e f', ''])):
            // r[m] (m = [T, F, T, F]) -> [list(['a', 'b']), list(['d', 'e', 'f'])], owndata True, r[m][0] is r[0]
            // r[[2, 0]] -> [list(['d', 'e', 'f']), list(['a', 'b'])]; r[np.array(1)] -> ['c'] (the list)
            using var r = np.@char.split(np.array(new[] { "a b", "c", "d e f", "" }));
            using var element = r[1];
            element.IsStandIn.Should().BeTrue();
            element.item().Cast<string>().Should().Equal("c");
            using var masked = r[np.array(new[] { true, false, true, false })];
            masked.shape.Should().Equal(2L);
            masked.IsStandIn.Should().BeFalse();
            ListsOf(masked).Should().BeEquivalentTo(new[] { new[] { "a", "b" }, new[] { "d", "e", "f" } }, o => o.WithStrictOrdering());
            using var gathered = r[np.array(new[] { 2, 0 })];
            ListsOf(gathered).Should().BeEquivalentTo(new[] { new[] { "d", "e", "f" }, new[] { "a", "b" } }, o => o.WithStrictOrdering());
            using var zeroD = r[np.array(1)];
            zeroD.IsStandIn.Should().BeTrue("a 0-d integer array is part of a full integer index");
            zeroD.item().Cast<string>().Should().Equal("c");
        }

        [TestMethod]
        public void ListArray_AssignThroughAnAdvancedIndex_CopiesTheLists()
        {
            // b = r.copy(); b[[0, 1]] = r[[2, 3]] -> [list(['d', 'e', 'f']), list([]), list(['d', 'e', 'f']), list([])]
            // b[np.array(1)] = r[2] -> element 1 IS the list r[2]
            using var r = np.@char.split(np.array(new[] { "a b", "c", "d e f", "" }));
            using var b = r.copy();
            using (var source = r[np.array(new[] { 2, 3 })])
                b[np.array(new[] { 0, 1 })] = source;
            ListsOf(b).Should().BeEquivalentTo(new[] { new[] { "d", "e", "f" }, Array.Empty<string>(), new[] { "d", "e", "f" }, Array.Empty<string>() },
                o => o.WithStrictOrdering());
            using var c = r.copy();
            using (var list = r[2])
                c[np.array(1)] = list;
            ListsOf(c)[1].Should().Equal("d", "e", "f");
            // An array of lists that does not broadcast: the MapIter text, as NumPy's.
            using (var three = r[np.array(new[] { 0, 1, 2 })])
                new Action(() => b[np.array(new[] { 0, 1 })] = three).Should().Throw<ValueError>()
                    .WithMessage("shape mismatch: value array of shape (3,) could not be broadcast to indexing result of shape (2,)");
        }

        [TestMethod]
        public void ListArray_AListThroughAnAdvancedIndex_IsItsItems_NumPysErrors()
        {
            // NumPy reads a list assigned through an index array as the sequence of its items:
            // b[[0, 1]] = ['d', 'e', 'f'] -> ValueError could not broadcast input array from shape (3,) into shape (2,)
            // b[[0, 1]] = []              -> ValueError could not broadcast input array from shape (0,) into shape (2,)
            // b[m] = ['d', 'e', 'f'] (m of b's shape, 2 true) -> ValueError NumPy boolean array indexing assignment cannot
            //     assign 3 input values to the 2 output values where the mask is true
            // b[[0, 1]] = ['c']           -> stores the str 'c' in both elements, which a list array cannot hold
            using var r = np.@char.split(np.array(new[] { "a b", "c", "d e f", "" }));
            using var b = r.copy();
            using var three = r[2];
            using var empty = r[3];
            using var one = r[1];
            var idx = np.array(new[] { 0, 1 });
            new Action(() => b[idx] = three).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (3,) into shape (2,)");
            new Action(() => b[idx] = empty).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (0,) into shape (2,)");
            new Action(() => b[np.array(new[] { true, false, true, false })] = three).Should().Throw<ValueError>()
                .WithMessage("NumPy boolean array indexing assignment cannot assign 3 input values to the 2 output values where the mask is true");
            new Action(() => b[idx] = one).Should().Throw<NotSupportedException>();
            // A selection of no element stores nothing.
            b[np.array(new[] { false, false, false, false })] = one;
            ListsOf(b)[0].Should().Equal("a", "b");
        }

        /// <summary>The lists of a 1-d list array as string arrays (NumPy's <c>tolist()</c>).</summary>
        /// <param name="lists">The list array.</param>
        /// <returns>Each element's pieces.</returns>
        private static string[][] ListsOf(NDListArray lists)
            => ((System.Collections.IList)lists.tolist()).Cast<List<object>>().Select(l => l.Cast<string>().ToArray()).ToArray();
    }
}
