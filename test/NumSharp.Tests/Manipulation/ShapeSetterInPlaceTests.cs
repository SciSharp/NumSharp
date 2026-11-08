using System;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     Pins the IN-PLACE shape setters (<c>ndarray.shape = …</c> and <see cref="NDArray.Shape"/>) to NumPy's
    ///     <c>array_shape_set</c>: reshape in C order and ADOPT the result's dimensions and byte strides only when
    ///     that reshape is a no-copy view of the array's own memory; otherwise raise <see cref="AttributeError"/>
    ///     with NumPy's text verbatim. Every expectation below was probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     The setters used to route through <c>UnmanagedStorage.Reshape</c>, which COPIED a non-contiguous array
    ///     in place and swapped the storage's buffer under the array's counted reference — a NumPy divergence
    ///     (NumPy refuses <c>t.shape = (12,)</c> on a transposed view) and two pooled buffers stranded per
    ///     assignment (found by the leak audit's property read gate). The pins therefore hold both halves: the
    ///     NumPy verdict (adopt vs <see cref="AttributeError"/>, the resulting strides and flags) and that the
    ///     array keeps its OWN buffer and reference.
    /// </remarks>
    [TestClass]
    public class ShapeSetterInPlaceTests
    {
        /// <summary>NumPy's <c>array_shape_set</c> refusal text, verbatim.</summary>
        private const string Incompatible =
            "Incompatible shape for in-place modification. Use `.reshape()` to make a copy with the desired shape.";

        /// <summary>
        ///     Re-assigning a transposed view its own dims adopts them unchanged. NumPy 2.4.2:
        ///     <c>t = arange(12.).reshape(3,4).T; t.shape = (4,3)</c> → shape (4,3), strides (8,32), F-contiguous, writeable.
        /// </summary>
        [TestMethod]
        public void Transposed_SameDims_AdoptsTheFStrides()
        {
            using var b = np.arange(12.0).reshape(3, 4);
            using var t = b.T;

            t.shape = new long[] { 4, 3 };

            t.shape.Should().Equal(4L, 3L);
            t.strides.Should().Equal(8L, 32L);
            t.flags.f_contiguous.Should().BeTrue();
            t.flags.c_contiguous.Should().BeFalse();
            t.flags.writeable.Should().BeTrue();
        }

        /// <summary>
        ///     Inserting unit axes into a transposed view is still a no-copy view. NumPy 2.4.2: <c>t.shape = (4,3,1)</c>
        ///     → strides (8,32,32); <c>t.shape = (1,4,3)</c> → strides (32,8,32); <c>t.shape = (4,-1)</c> infers (4,3).
        /// </summary>
        [TestMethod]
        public void Transposed_UnitAxesAndUnknownDim_AdoptNoCopyStrides()
        {
            using var b = np.arange(12.0).reshape(3, 4);

            using (var t = b.T)
            {
                t.shape = new long[] { 4, 3, 1 };
                t.shape.Should().Equal(4L, 3L, 1L);
                t.strides.Should().Equal(8L, 32L, 32L);
            }

            using (var t = b.T)
            {
                t.shape = new long[] { 1, 4, 3 };
                t.shape.Should().Equal(1L, 4L, 3L);
                t.strides.Should().Equal(32L, 8L, 32L);
            }

            using (var t = b.T)
            {
                t.shape = new long[] { 4, -1 };
                t.shape.Should().Equal(4L, 3L);
                t.strides.Should().Equal(8L, 32L);
            }
        }

        /// <summary>
        ///     A shape that would need a COPY of a transposed view is refused and leaves the view untouched. NumPy
        ///     2.4.2: <c>t.shape = (12,)</c>, <c>(2,6)</c>, <c>(6,2)</c>, <c>(-1,)</c> → AttributeError. The
        ///     <see cref="NDArray.Shape"/> setter applies the same rule.
        /// </summary>
        [TestMethod]
        public void Transposed_ShapeNeedingACopy_RaisesAttributeError_AndLeavesTheViewIntact()
        {
            foreach (var requested in new[] { new long[] { 12 }, new long[] { 2, 6 }, new long[] { 6, 2 }, new long[] { -1 } })
            {
                using var b = np.arange(12.0).reshape(3, 4);
                using var t = b.T;

                new Action(() => t.shape = requested).Should().ThrowExactly<AttributeError>().WithMessage(Incompatible);

                t.shape.Should().Equal(4L, 3L);
                t.strides.Should().Equal(8L, 32L);
            }

            using (var b = np.arange(12.0).reshape(3, 4))
            using (var t = b.T)
                new Action(() => t.Shape = new Shape(12)).Should().ThrowExactly<AttributeError>().WithMessage(Incompatible);
        }

        /// <summary>
        ///     A strided (every-other-column) view accepts every shape its uniform stride can express. NumPy 2.4.2,
        ///     <c>s = arange(24.).reshape(3,8)[:, ::2]</c> (strides (64,16)): (2,6)→(96,16), (12,)→(16,),
        ///     (3,2,2)→(64,32,16), (6,2)→(32,16), (-1,)→(12,) (16,), (2,3,2)→(96,32,16); never contiguous, writeable.
        /// </summary>
        [TestMethod]
        public void StridedView_CombinableShapes_AdoptNoCopyStrides()
        {
            var cases = new (long[] requested, long[] shape, long[] strides)[]
            {
                (new long[] { 2, 6 }, new long[] { 2, 6 }, new long[] { 96, 16 }),
                (new long[] { 12 }, new long[] { 12 }, new long[] { 16 }),
                (new long[] { 3, 2, 2 }, new long[] { 3, 2, 2 }, new long[] { 64, 32, 16 }),
                (new long[] { 6, 2 }, new long[] { 6, 2 }, new long[] { 32, 16 }),
                (new long[] { -1 }, new long[] { 12 }, new long[] { 16 }),
                (new long[] { 2, 3, 2 }, new long[] { 2, 3, 2 }, new long[] { 96, 32, 16 }),
            };
            foreach (var (requested, shape, strides) in cases)
            {
                using var b = np.arange(24.0).reshape(3, 8);
                using var s = b[":, ::2"];
                s.strides.Should().Equal(64L, 16L);

                s.shape = requested;

                s.shape.Should().Equal(shape);
                s.strides.Should().Equal(strides);
                s.flags.c_contiguous.Should().BeFalse();
                s.flags.f_contiguous.Should().BeFalse();
                s.flags.writeable.Should().BeTrue();
            }
        }

        /// <summary>
        ///     The adopted shape views the SAME memory — no in-place copy was taken — so a write through the
        ///     reshaped view lands in the base. Flat element 5 of <c>arange(24.).reshape(3,8)[:, ::2]</c> is
        ///     <c>b[1, 2]</c>.
        /// </summary>
        [TestMethod]
        public void AdoptedShape_WritesThroughToTheBase()
        {
            using var b = np.arange(24.0).reshape(3, 8);
            using var s = b[":, ::2"];

            s.shape = new long[] { 12 };
            s[5] = 99.0;

            ((double)b[1, 2]).Should().Be(99.0);
        }

        /// <summary>
        ///     The setter never swaps the buffer under the array's counted reference: the view keeps the base's
        ///     buffer, and disposing it releases exactly the one reference it took, leaving the base the sole owner.
        ///     (The old copy-and-swap route left the view's reference on the base's buffer forever.)
        /// </summary>
        [TestMethod]
        public void InPlaceShape_KeepsTheArraysOwnBufferAndReference()
        {
            // The base must be its buffer's ONLY owner for the reference counts below to mean anything:
            // `np.arange(24.0).reshape(3, 8)` would leave the undisposed arange parent holding a second
            // reference, so the base is built in place instead (a C-contiguous relabel adopts, no copy).
            using var b = np.arange(24.0);
            b.shape = new long[] { 3, 8 };
            b.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the base is its buffer's only owner");
            var s = b[":, ::2"];
            var buffer = s.Storage.InternalArray;

            s.shape = new long[] { 2, 6 };

            ReferenceEquals(buffer, s.Storage.InternalArray).Should().BeTrue("an adopted view keeps its own buffer");
            b.Storage.InternalArray.IsUniquelyReferenced.Should().BeFalse("the base and the view both hold a reference");
            s.Dispose();
            b.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("disposing the view released its one reference");
            LeakGuards.StillUsable(b);
        }

        /// <summary>
        ///     The shapes NumPy adopts outside the view-of-a-view case. NumPy 2.4.2: a C-contiguous (12,) →
        ///     (3,4) with strides (32,8), still owning its data; a 0-d array → (1,) strides (8,) and back to ();
        ///     an empty (0,3) → (3,0), C- and F-contiguous.
        /// </summary>
        /// <remarks>
        ///     The empty case asserts no strides, by the repo's standing size-0 policy (see
        ///     <c>LayoutParityOracleTests.Verify</c>): NumPy's own byte strides for an empty array depend on the
        ///     construction path — this setter/reshape yields <c>(8,8)</c> while <c>np.zeros((3,0))</c> yields
        ///     <c>(0,0)</c> — so they are not a parity contract; NumSharp reports <c>(0,8)</c> on every path.
        /// </remarks>
        [TestMethod]
        public void Contiguous_ZeroD_AndEmpty_AdoptTheNewShape()
        {
            using (var c = np.arange(12.0))
            {
                c.shape = new long[] { 3, 4 };
                c.shape.Should().Equal(3L, 4L);
                c.strides.Should().Equal(32L, 8L);
                c.flags.c_contiguous.Should().BeTrue();
                c.flags.owndata.Should().BeTrue();
            }

            using (var z = np.array(5.0))
            {
                z.shape = new long[] { 1 };
                z.shape.Should().Equal(1L);
                z.strides.Should().Equal(8L);
                z.shape = Array.Empty<long>();
                z.ndim.Should().Be(0);
                ((double)z).Should().Be(5.0);
            }

            using (var e = np.zeros(new Shape(0, 3)))
            {
                e.shape = new long[] { 3, 0 };
                e.shape.Should().Equal(3L, 0L);
                e.flags.c_contiguous.Should().BeTrue();
                e.flags.f_contiguous.Should().BeTrue();
            }
        }

        /// <summary>
        ///     A broadcast (read-only, stride-0) view accepts only shapes its strides can express, and stays
        ///     read-only. NumPy 2.4.2, <c>b = broadcast_to(arange(4.), (3,4))</c>: <c>b.shape = (12,)</c> →
        ///     AttributeError; <c>b.shape = (3,2,2)</c> → strides (0,16,8), writeable False.
        /// </summary>
        [TestMethod]
        public void BroadcastView_KeepsStrideZeroAndReadOnly_OrRefuses()
        {
            using var src = np.arange(4.0);

            using (var b = np.broadcast_to(src, new Shape(3, 4)))
                new Action(() => b.shape = new long[] { 12 }).Should().ThrowExactly<AttributeError>().WithMessage(Incompatible);

            using (var b = np.broadcast_to(src, new Shape(3, 4)))
            {
                b.shape = new long[] { 3, 2, 2 };
                b.shape.Should().Equal(3L, 2L, 2L);
                b.strides.Should().Equal(0L, 16L, 8L);
                b.flags.writeable.Should().BeFalse();
            }
        }

        /// <summary>
        ///     Size and unknown-dimension errors keep NumPy's reshape texts (with NumSharp's house exception type
        ///     for the size mismatch). NumPy 2.4.2: <c>arange(6.).shape = (4,)</c> → "cannot reshape array of size 6
        ///     into shape (4,)"; <c>(-1,-1)</c> → ValueError "can only specify one unknown dimension".
        /// </summary>
        [TestMethod]
        public void SizeMismatch_AndTwoUnknownDims_KeepNumPyReshapeTexts()
        {
            using var x = np.arange(6.0);

            new Action(() => x.shape = new long[] { 4 })
                .Should().ThrowExactly<IncorrectShapeException>()
                .WithMessage("cannot reshape array of size 6 into shape (4,)");
            new Action(() => x.shape = new long[] { -1, -1 })
                .Should().ThrowExactly<ValueError>()
                .WithMessage("can only specify one unknown dimension");
            x.shape.Should().Equal(6L);
        }

        /// <summary>
        ///     Reshaping a VIEW in place leaves its base's shape alone while the two keep sharing memory. NumPy 2.4.2:
        ///     <c>v = base[:]; v.shape = (3,4)</c> → v (3,4), base (12,), shares_memory True.
        /// </summary>
        [TestMethod]
        public void ViewReshapedInPlace_LeavesTheBaseShape()
        {
            using var b = np.arange(12.0);
            using var v = b[":"];

            v.shape = new long[] { 3, 4 };

            v.shape.Should().Equal(3L, 4L);
            b.shape.Should().Equal(12L);
            np.shares_memory(v, b).Should().BeTrue();
        }
    }
}
