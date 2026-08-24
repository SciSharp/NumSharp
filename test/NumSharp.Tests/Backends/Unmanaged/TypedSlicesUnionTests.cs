using System;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     Gates for the <see cref="UnmanagedStorage"/> typed-slice union
    ///     (docs/UNMANAGED_STORAGE_UNION_DESIGN.md): the 15 per-dtype <c>ArraySlice&lt;T&gt;</c>
    ///     fields collapsed into one 64 B explicit-layout union whose lanes all overlap at offset 0.
    ///     Proves the union type LOADS on this runtime (a mis-formed GC ref map is a
    ///     <see cref="TypeLoadException"/> at first touch — the CLR's own legality check), that every
    ///     lane still reads through the direct typed getters on every layout the storage can take
    ///     (owned / sliced / aliased / cloned), that <c>Alias(Shape)</c>'s single struct copy carries
    ///     the live lane the retired IL field-copier used to mirror, and that the union is exactly
    ///     one slice wide.
    /// </summary>
    [TestClass]
    public class TypedSlicesUnionTests
    {
        private static readonly NPTypeCode[] AllDtypes =
        {
            NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
            NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64,
            NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
            NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
        };

        /// <summary>Reads through the dtype-matched DIRECT getter — the only path that touches the union lane.</summary>
        private static object DirectGet(UnmanagedStorage st, params long[] indices)
        {
            switch (st.TypeCode)
            {
                case NPTypeCode.Boolean: return st.GetBoolean(indices);
                case NPTypeCode.SByte: return st.GetSByte(indices);
                case NPTypeCode.Byte: return st.GetByte(indices);
                case NPTypeCode.Int16: return st.GetInt16(indices);
                case NPTypeCode.UInt16: return st.GetUInt16(indices);
                case NPTypeCode.Int32: return st.GetInt32(indices);
                case NPTypeCode.UInt32: return st.GetUInt32(indices);
                case NPTypeCode.Int64: return st.GetInt64(indices);
                case NPTypeCode.UInt64: return st.GetUInt64(indices);
                case NPTypeCode.Char: return st.GetChar(indices);
                case NPTypeCode.Half: return st.GetHalf(indices);
                case NPTypeCode.Single: return st.GetSingle(indices);
                case NPTypeCode.Double: return st.GetDouble(indices);
                case NPTypeCode.Decimal: return st.GetDecimal(indices);
                case NPTypeCode.Complex: return st.GetComplex(indices);
                default: throw new NotSupportedException(st.TypeCode.ToString());
            }
        }

        /// <summary>Same lane read through the int[] getter overloads.</summary>
        private static object DirectGet(UnmanagedStorage st, int[] indices)
        {
            switch (st.TypeCode)
            {
                case NPTypeCode.Boolean: return st.GetBoolean(indices);
                case NPTypeCode.SByte: return st.GetSByte(indices);
                case NPTypeCode.Byte: return st.GetByte(indices);
                case NPTypeCode.Int16: return st.GetInt16(indices);
                case NPTypeCode.UInt16: return st.GetUInt16(indices);
                case NPTypeCode.Int32: return st.GetInt32(indices);
                case NPTypeCode.UInt32: return st.GetUInt32(indices);
                case NPTypeCode.Int64: return st.GetInt64(indices);
                case NPTypeCode.UInt64: return st.GetUInt64(indices);
                case NPTypeCode.Char: return st.GetChar(indices);
                case NPTypeCode.Half: return st.GetHalf(indices);
                case NPTypeCode.Single: return st.GetSingle(indices);
                case NPTypeCode.Double: return st.GetDouble(indices);
                case NPTypeCode.Decimal: return st.GetDecimal(indices);
                case NPTypeCode.Complex: return st.GetComplex(indices);
                default: throw new NotSupportedException(st.TypeCode.ToString());
            }
        }

        [TestMethod]
        public void Union_TypeLoads_And_IsOneSliceWide()
        {
            var union = typeof(UnmanagedStorage).GetNestedType("TypedSlices", BindingFlags.NonPublic);
            union.Should().NotBeNull("UnmanagedStorage must carry the typed-slice union");
            union!.GetFields(BindingFlags.Public | BindingFlags.Instance).Length.Should().Be(15, "one lane per supported dtype");

            // Every lane overlaps at offset 0 — the union is exactly one ArraySlice wide, not fifteen.
            if (IntPtr.Size == 8)
            {
                var size = (int)typeof(Unsafe).GetMethod(nameof(Unsafe.SizeOf))!
                    .MakeGenericMethod(union).Invoke(null, null)!;
                size.Should().Be(64);
                Unsafe.SizeOf<ArraySlice<int>>().Should().Be(64);
            }

            // Instantiating any storage forces the union field's type to load: a GC-ref-map violation
            // in the explicit layout would be a TypeLoadException right here, not silent corruption.
            var nd = np.ones(new Shape(3), typeof(double));
            nd.Storage.GetDouble(1L).Should().Be(1.0);
        }

        [TestMethod]
        public void AllDtypes_LaneAgreesWithAddress_OnOwnedAndSlicedStorage()
        {
            foreach (var tc in AllDtypes)
            {
                var nd = np.arange(6).astype(tc.AsType()).reshape(2, 3);
                var st = nd.Storage;
                for (long i = 0; i < 6; i++)
                {
                    // The direct getter reads the union lane; GetValue reads the synchronized Address
                    // cache. They address the same memory, so they must agree bit-for-bit.
                    DirectGet(st, i / 3, i % 3).Should().Be(st.GetValue(i / 3, i % 3), $"{tc} lane vs Address at flat {i}");
                    DirectGet(st, new[] { (int)(i / 3), (int)(i % 3) }).Should().Be(st.GetValue((int)(i / 3), (int)(i % 3)), $"{tc} int[] lane vs Address at flat {i}");
                }

                // A sliced view routes the same lane through a shifted Shape.
                var view = nd["1:"];
                DirectGet(view.Storage, 0, 2).Should().Be(st.GetValue(1, 2), $"{tc} sliced-view lane");
            }
        }

        [TestMethod]
        public void Alias_StructCopy_CarriesTheLiveLane_AllDtypes()
        {
            foreach (var tc in AllDtypes)
            {
                var nd = np.arange(6).astype(tc.AsType());
                var st = nd.Storage;

                // Alias(Shape) is the path whose typed-lane mirror moved from the IL field-copier to
                // the single union struct copy — the alias's DIRECT getters read that copied lane.
                var alias = st.Alias(st.Shape);
                alias.InternalArray.Should().BeSameAs(st.InternalArray, $"{tc} alias shares the boxed slice");
                for (long i = 0; i < 6; i++)
                    DirectGet(alias, i).Should().Be(DirectGet(st, i), $"{tc} alias lane at {i}");

                // Alias() and Alias(ref Shape) route through SetInternalArray's switch — same lane.
                var alias2 = st.Alias();
                DirectGet(alias2, 3L).Should().Be(DirectGet(st, 3L), $"{tc} Alias() lane");
            }
        }

        [TestMethod]
        public void View_WriteThrough_IsVisibleThroughTheLane()
        {
            var nd = np.arange(10).astype(typeof(double));
            var view = nd["::2"];

            view.Storage.SetAtIndex(99.0, 1); // logical element 1 of the view == base element 2

            nd.Storage.GetDouble(2L).Should().Be(99.0, "the view writes through to the base lane");
            view.Storage.GetDouble(1L).Should().Be(99.0);
        }

        [TestMethod]
        public void Clone_IsIndependent_WidestLanesIncluded()
        {
            // decimal and Complex are the 16-byte lanes — the widest the union carries.
            foreach (var tc in new[] { NPTypeCode.Int32, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex })
            {
                var nd = np.arange(4).astype(tc.AsType());
                var st = nd.Storage;
                var clone = st.Clone();
                var before = DirectGet(clone, 0L);

                st.SetAtIndex(st.GetAtIndex(1), 0); // mutate original slot 0 (0 -> 1) in its own dtype

                DirectGet(clone, 0L).Should().Be(before, $"{tc} clone must not see the original's mutation");
                DirectGet(st, 0L).Should().Be(DirectGet(st, 1L), $"{tc} original mutated");
            }
        }

        [TestMethod]
        public void DisposeBase_ViewLaneStillReads()
        {
            var nd = np.arange(8).astype(typeof(double));
            var view = nd["2:5"];

            nd.Dispose();

            // The view's union lane carries the same Disposer reference the boxed InternalArray does,
            // and its counted ref keeps the block alive past the base's dispose.
            view.Storage.GetDouble(0L).Should().Be(2.0);
            view.Storage.GetDouble(2L).Should().Be(4.0);
        }
    }
}
