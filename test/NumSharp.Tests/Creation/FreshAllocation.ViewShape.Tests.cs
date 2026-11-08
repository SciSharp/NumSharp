using System;
using System.Collections.Generic;
using NumSharp.Backends;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Regression pins for the view-<see cref="Shape"/> allocation overrun (catalog open item, found 2026-09-24): every
    ///     entry point that allocates a FRESH array from a caller's <see cref="Shape"/> — <c>np.empty</c>, <c>np.zeros</c>,
    ///     <c>np.ones</c>, <c>np.full</c>, <c>new NDArray(dtype, shape[, fillZeros])</c> and the <see cref="UnmanagedStorage"/>
    ///     <c>Allocate</c> overloads — handed a view's shape (<c>x.Shape</c> of a stepped, offset, reversed, broadcast or sliced
    ///     view) must lay out a buffer of exactly <c>size</c> elements that every logical index stays inside.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before the fix the allocators installed the view's strides and offset over the <c>size</c>-element buffer: for
    ///     <c>arange(20)["::4"]</c> element i sat at i * 4, so reads returned heap memory and writes corrupted it —
    ///     <c>np.polyval(p, arange(20.0)["::4"])</c>, which allocates <c>np.zeros(x.Shape)</c> and accumulates into it, ended the
    ///     process with an internal CLR error. NumPy's <c>np.empty(x.shape)</c> takes dimensions only, so every check here is
    ///     either NumPy's own value (polyval, clip) or the memory-safety contract (offset 0, a buffer of exactly <c>size</c>
    ///     elements, every element addressable once).
    ///     </para>
    ///     <para>
    ///     A dense layout (C, F, or a KEEPORDER permutation at offset 0 over the whole buffer) is kept, since NumSharp allocates
    ///     its <c>order='K'</c> outputs that way; <c>ShapeForFreshAllocationTests</c> (View tests) pins that rule itself.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class FreshAllocationViewShapeTests
    {
        /// <summary>
        ///     The hostile view layouts, each with the dimensions NumPy would see (<c>x.shape</c>). Built from real views and
        ///     returned as shapes only — a <see cref="Shape"/> holds no storage, so the source arrays are disposed at once.
        /// </summary>
        /// <returns>(name, view shape) pairs.</returns>
        private static IEnumerable<(string name, Shape shape)> HostileShapes()
        {
            static Shape Of(Func<NDArray> make)
            {
                using var view = make();
                return view.Shape;
            }

            yield return ("stepped", Of(() => np.arange(20)["::4"]));
            yield return ("offset stepped", Of(() => np.arange(20)["3::4"]));
            yield return ("offset stepped 2-D", Of(() => np.zeros(new Shape(7, 9))["1::2, 1::2"]));
            yield return ("reversed", Of(() => np.zeros(new Shape(4, 5))["::-1"]));
            yield return ("broadcast", Of(() => np.broadcast_to(np.arange(3.0), new Shape(4, 3))));
            yield return ("scalar broadcast", Of(() => np.broadcast_to(np.zeros(new Shape(1)), new Shape(6L))));
            yield return ("contiguous slice, larger buffer", Of(() => np.zeros(new Shape(23))[":20"].reshape(4, 5)));
            yield return ("column slice", Of(() => np.arange(12).reshape(3, 4)[":, 1:2"]));
            yield return ("empty, parent offset", Of(() => np.arange(20)["::-1"]["0:0"]));
            yield return ("user strides with a gap", new Shape(new long[] { 2, 3 }, new long[] { 1, 4 }));
            yield return ("transposed (dense, kept)", Of(() => np.zeros(new Shape(5, 4)).T));
            yield return ("permuted 3-D (dense, kept)", Of(() => np.transpose(np.zeros(new Shape(3, 2, 4)), new[] { 1, 0, 2 })));
        }

        /// <summary>
        ///     The memory-safety contract of a fresh array allocated for <paramref name="requested"/>: same dimensions, offset 0,
        ///     a buffer of exactly <c>size</c> elements that it owns, writeable, and every logical element at its own address
        ///     inside the buffer.
        /// </summary>
        /// <param name="a">The fresh array.</param>
        /// <param name="requested">The shape it was allocated for.</param>
        /// <param name="what">The call, for the assertion messages.</param>
        private static void ShouldBeSafeFreshArray(NDArray a, Shape requested, string what)
        {
            var s = a.Shape;
            s.dimensions.Should().Equal(requested.dimensions, what);
            s.offset.Should().Be(0, what);
            s.BufferSize.Should().Be(s.size, what);
            s.IsWriteable.Should().BeTrue(what);
            s.IsBroadcasted.Should().BeFalse(what);
            a.Storage.OwnsData.Should().BeTrue(what);
            a.Storage.InternalArray.Count.Should().Be(s.size, what + ": the buffer holds exactly the elements");

            var seen = new bool[s.size];
            for (long i = 0; i < s.size; i++)
            {
                long off = s.GetOffset(s.GetCoordinates(i));
                off.Should().BeInRange(0, s.size - 1, $"{what}: logical element {i} must lie inside the buffer");
                seen[off].Should().BeFalse($"{what}: logical element {i} must not share an address");
                seen[off] = true;
            }
        }

        /// <summary>
        ///     Writes 0..size-1 in logical C order through the array's own layout and reads it back — the access that overran
        ///     the heap before the fix.
        /// </summary>
        /// <param name="a">A fresh float64 array.</param>
        /// <param name="what">The call, for the assertion messages.</param>
        private static void ShouldRoundTripEveryElement(NDArray a, string what)
        {
            using var src = np.arange((double)a.size).reshape(a.shape);
            np.copyto(a, src);
            for (long i = 0; i < a.size; i++)
                a.GetAtIndex<double>(i).Should().Be(i, what);
        }

        [TestMethod]
        public void Zeros_OfAViewShape_IsSafeAndAllZero()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                using var a = np.zeros(shape);
                ShouldBeSafeFreshArray(a, shape, $"np.zeros({name})");
                for (long i = 0; i < a.size; i++)
                    a.GetAtIndex<double>(i).Should().Be(0.0, $"np.zeros({name})[{i}] must read the new buffer, not the heap");
                ShouldRoundTripEveryElement(a, $"np.zeros({name})");

                using var typed = np.zeros(shape, np.int32);
                ShouldBeSafeFreshArray(typed, shape, $"np.zeros({name}, int32)");
            }
        }

        [TestMethod]
        public void Empty_OfAViewShape_IsSafe()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                using var a = np.empty(shape);
                ShouldBeSafeFreshArray(a, shape, $"np.empty({name})");
                ShouldRoundTripEveryElement(a, $"np.empty({name})");

                using var typed = np.empty(shape, np.uint8);
                ShouldBeSafeFreshArray(typed, shape, $"np.empty({name}, uint8)");
            }
        }

        [TestMethod]
        public void Ones_OfAViewShape_IsSafeAndAllOne()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                using var a = np.ones(shape);
                ShouldBeSafeFreshArray(a, shape, $"np.ones({name})");
                for (long i = 0; i < a.size; i++)
                    a.GetAtIndex<double>(i).Should().Be(1.0, $"np.ones({name})[{i}]");

                using var typed = np.ones(shape, np.int64);
                ShouldBeSafeFreshArray(typed, shape, $"np.ones({name}, int64)");
            }
        }

        [TestMethod]
        public void Full_OfAViewShape_IsSafeAndFilled()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                using var a = np.full(shape, 7.5);
                ShouldBeSafeFreshArray(a, shape, $"np.full({name})");
                for (long i = 0; i < a.size; i++)
                    a.GetAtIndex<double>(i).Should().Be(7.5, $"np.full({name})[{i}]");

                using var typed = np.full(shape, 3, np.int16);
                ShouldBeSafeFreshArray(typed, shape, $"np.full({name}, int16)");
            }
        }

        [TestMethod]
        public void NDArrayConstructors_OfAViewShape_AreSafe()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                using (var a = new NDArray(np.float64, shape))
                {
                    ShouldBeSafeFreshArray(a, shape, $"new NDArray(float64, {name})");
                    ShouldRoundTripEveryElement(a, $"new NDArray(float64, {name})");
                }

                using (var a = new NDArray(np.float64, shape, false))
                    ShouldBeSafeFreshArray(a, shape, $"new NDArray(float64, {name}, fillZeros: false)");
                using (var a = new NDArray(typeof(double), shape, true))
                    ShouldBeSafeFreshArray(a, shape, $"new NDArray(typeof(double), {name}, fillZeros: true)");
                using (var a = new NDArray(NPTypeCode.Complex, shape, true))
                {
                    ShouldBeSafeFreshArray(a, shape, $"new NDArray(complex, {name})");
                    // AliasComplexLane sizes its float64 lane by the buffer size: a stale parent size ran past the buffer.
                    using var re = a.real;
                    re.Storage.InternalArray.Count.Should().Be(2 * a.size, $"complex({name}).real lane length");
                }
            }
        }

        [TestMethod]
        public void StorageAllocate_EveryOverload_IsSafe()
        {
            foreach (var (name, shape) in HostileShapes())
            {
                var s1 = new UnmanagedStorage(typeof(double));
                s1.Allocate(shape);
                using (var a = new NDArray(s1)) ShouldBeSafeFreshArray(a, shape, $"Allocate({name})");

                var s2 = new UnmanagedStorage(typeof(double));
                s2.Allocate(shape, typeof(double), false);
                using (var a = new NDArray(s2)) ShouldBeSafeFreshArray(a, shape, $"Allocate({name}, Type, false)");

                var s3 = new UnmanagedStorage(NPTypeCode.Double);
                s3.Allocate(shape, NPTypeCode.Double, false);
                using (var a = new NDArray(s3)) ShouldBeSafeFreshArray(a, shape, $"Allocate({name}, NPTypeCode, false)");

                var s4 = new UnmanagedStorage(np.float64);
                s4.Allocate(shape, np.float64, true);
                using (var a = new NDArray(s4)) ShouldBeSafeFreshArray(a, shape, $"Allocate({name}, DType, true)");
            }
        }

        [TestMethod]
        public void DenseViewLayouts_AreKept()
        {
            // The KEEPORDER callers depend on this: a dense F / permuted layout over the whole buffer survives allocation.
            using var t = np.zeros(new Shape(5, 4)).T;
            using var a = np.empty(t.Shape);
            a.Shape.IsFContiguous.Should().BeTrue();
            a.Shape.IsContiguous.Should().BeFalse();

            using var p = np.transpose(np.zeros(new Shape(3, 2, 4)), new[] { 1, 0, 2 });
            using var b = new NDArray(np.float64, p.Shape);
            b.Shape.strides.Should().Equal(p.Shape.strides);
        }

        [TestMethod]
        public void ReadOnlySourceShape_GivesAWriteableFreshArray()
        {
            // np.full / np.ones wrap a fresh buffer directly; a read-only source's shape must not make them read-only.
            using var ro = np.zeros(new Shape(3, 4));
            ro.Shape.IsWriteable.Should().BeTrue();
            var roShape = ro.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            using var f = np.full(roShape, 2.0);
            f.Shape.IsWriteable.Should().BeTrue();
            using var o = np.ones(roShape);
            o.Shape.IsWriteable.Should().BeTrue();
            using var z = np.zeros(roShape);
            z.Shape.IsWriteable.Should().BeTrue();
        }

        [TestMethod]
        public void Polyval_OfASteppedView_MatchesNumPy()
        {
            // NumPy 2.4.2: np.polyval([1, 2, 3], np.arange(20.)[::4]) -> [3., 27., 83., 171., 291.]
            // Before the fix np.zeros(x.Shape) gave y the view's stride 4 and the Horner loop wrote past the buffer
            // (the test process died with an internal CLR error).
            using var p = np.array(new double[] { 1, 2, 3 });
            using var x = np.arange(20.0)["::4"];
            using var y = np.polyval(p, x);
            y.ToArray<double>().Should().Equal(3.0, 27.0, 83.0, 171.0, 291.0);
            y.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void ClipWithNaNBound_OfASteppedView_MatchesNumPy()
        {
            // NumPy 2.4.2: np.clip(np.arange(20.)[::4], np.nan, 5) -> [nan nan nan nan nan]
            // The NaN-bound shortcut returns np.full(lhs.Shape, NaN): before the fix four of the five reads were heap memory.
            using var x = np.arange(20.0)["::4"];
            using var r = np.clip(x, double.NaN, 5.0);
            r.shape.Should().Equal(new long[] { 5 });
            foreach (var v in r.ToArray<double>())
                double.IsNaN(v).Should().BeTrue();
        }
    }
}
