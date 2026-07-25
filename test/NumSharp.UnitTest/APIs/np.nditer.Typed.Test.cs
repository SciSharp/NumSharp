using System;
using System.Collections.Generic;
using System.Globalization;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    ///     Tests for the typed, allocation-free iterators <see cref="np.nditer{T}"/> and
    ///     <see cref="np.nditer_chunks{T}"/>.
    ///
    ///     <para>
    ///     These are a NumSharp EXTENSION — NumPy has no typed iteration, so there is no oracle
    ///     tier and no bit-comparable result. What is pinned instead is (a) equivalence with the
    ///     boxed <c>np.nditer</c> / <c>ndenumerate</c> the engine already ships, across every
    ///     memory layout and all 15 dtypes, and (b) the handful of behaviours that ARE
    ///     NumPy-observable: iteration order under <c>'K'</c> vs <c>'C'</c>, and the read-only
    ///     broadcast rejection. Those expected values come from running NumPy 2.4.2.
    ///     </para>
    /// </summary>
    [TestClass]
    public class np_nditer_Typed_Test
    {
        private static NDArray A() => np.arange(6).astype(np.float64).reshape(2, 3);

        private static string Walk(NDArray a, char order = 'K')
        {
            var parts = new List<string>();
            foreach (ref double x in np.nditer<double>(a, order: order))
                parts.Add(x.ToString(CultureInfo.InvariantCulture));

            return string.Join(" ", parts);
        }

        // ------------------------------------------------------------------ order (NumPy-probed)

        [TestMethod]
        public void Typed_DefaultOrderIsK_MemoryOrder()
        {
            Walk(A()).Should().Be("0 1 2 3 4 5");
        }

        [TestMethod]
        public void Typed_OrderF()
        {
            // numpy: list(np.nditer(a, order='F')) -> [0, 3, 1, 4, 2, 5]
            Walk(A(), 'F').Should().Be("0 3 1 4 2 5");
        }

        [TestMethod]
        public void Typed_ReversedView_K_IsMemoryOrder_C_IsLogical()
        {
            // Probed against NumPy 2.4.2 on a[:, ::-1] of arange(6).reshape(2,3):
            //   default (order='K') -> 0 1 2 3 4 5     (memory order)
            //   order='C'           -> 2 1 0 5 4 3     (logical order)
            // This is the one trap of the 'K' default, so it is pinned explicitly.
            var rev = A()[":, ::-1"];
            Walk(rev).Should().Be("0 1 2 3 4 5");
            Walk(rev, 'C').Should().Be("2 1 0 5 4 3");
        }

        [TestMethod]
        public void Typed_OrderC_MatchesNdEnumerate_OnEveryLayout()
        {
            // ndenumerate is the LOGICAL C-order walk, so order:'C' must agree with it everywhere.
            var b = np.arange(24).astype(np.float64).reshape(4, 6);
            foreach (var a in new[] {b, b.T, b[":, ::2"], b[":, ::-1"], b["1:3, 2:5"]})
            {
                var expected = new List<string>();
                foreach (var (_, v) in np.ndenumerate<double>(a))
                    expected.Add(v.ToString(CultureInfo.InvariantCulture));

                Walk(a, 'C').Should().Be(string.Join(" ", expected));
            }
        }

        // ------------------------------------------------------------------ layout coverage

        [TestMethod]
        public void Typed_AllLayouts_MatchBoxedNdIter()
        {
            var b = np.arange(100).astype(np.float64).reshape(10, 10);
            var bcast = np.broadcast_to(np.arange(10).astype(np.float64), new Shape(10, 10));

            var cases = new (string name, NDArray arr)[]
            {
                ("contiguous", b),
                ("F-order", b.T),
                ("strided", b[":, ::2"]),
                ("reversed", b[":, ::-1"]),
                ("sliced offset", b["2:8, 1:9"]),
                ("broadcast", bcast),
                ("0-d", np.array(3.5)),
                ("empty", np.zeros(new Shape(0, 5), np.float64)),
                ("one element", np.array(new[] {7.0})),
                ("3-D strided", np.arange(1000).astype(np.float64).reshape(10, 10, 10)["::2, ::3, ::4"]),
            };

            foreach (var (name, arr) in cases)
            {
                // The boxed np.nditer is the reference: same engine, same flags, same order.
                // It needs zerosize_ok for the empty case (NumPy's rule); the typed form does not
                // — see Typed_EmptyArray_IteratesZeroTimes_UnlikeBoxedNdIter.
                var expected = new List<string>();
                using (var it = arr.size == 0 ? np.nditer(arr, flags: new[] {"zerosize_ok"}) : np.nditer(arr))
                {
                    while (!it.finished)
                    {
                        expected.Add(Convert.ToString(it[0].GetAtIndex(0), CultureInfo.InvariantCulture));
                        it.iternext();
                    }
                }

                // np.nditer on an empty operand yields nothing but reports finished only after a
                // step, so normalise: the typed walk must visit exactly size elements.
                long count = 0;
                foreach (ref double _ in np.nditer<double>(arr))
                    count++;

                count.Should().Be(arr.size, $"{name} must visit every element");
                if (arr.size > 0)
                    Walk(arr).Should().Be(string.Join(" ", expected), $"{name} must match np.nditer");
            }
        }

        [TestMethod]
        public void Typed_EmptyArray_IteratesZeroTimes_UnlikeBoxedNdIter()
        {
            // DELIBERATE divergence from the boxed np.nditer (and from NumPy, which raises
            //   ValueError: Iteration of zero-sized operands is not enabled
            // unless you pass flags=['zerosize_ok']). Throwing would force every caller of a
            // `foreach` to guard `if (a.size > 0)`, which is not how C# collections behave. The
            // typed form is a NumSharp extension, not a parity surface, so empty means empty.
            var empty = np.zeros(new Shape(0, 5), np.float64);

            Assert.ThrowsException<ArgumentException>(() => np.nditer(empty));

            long n = 0;
            foreach (ref double _ in np.nditer<double>(empty)) n++;
            n.Should().Be(0);

            int chunks = 0;
            foreach (Span<double> _ in np.nditer_chunks<double>(empty)) chunks++;
            chunks.Should().Be(0);
        }

        [TestMethod]
        public void Typed_EmptyAndZeroD()
        {
            long n = 0;
            foreach (ref double _ in np.nditer<double>(np.zeros(new Shape(0, 5), np.float64)))
                n++;
            n.Should().Be(0);

            var seen = new List<double>();
            foreach (ref double x in np.nditer<double>(np.array(3.5)))
                seen.Add(x);
            seen.Should().Equal(3.5);
        }

        // ------------------------------------------------------------------ all 15 dtypes

        [TestMethod]
        public void Typed_AllFifteenDtypes_VisitEveryElement()
        {
            Count<bool>(np.array(new[] {true, false, true})).Should().Be(3);
            Count<byte>(np.arange(10).astype(np.uint8)).Should().Be(10);
            Count<sbyte>(np.arange(10).astype(np.int8)).Should().Be(10);
            Count<short>(np.arange(10).astype(np.int16)).Should().Be(10);
            Count<ushort>(np.arange(10).astype(np.uint16)).Should().Be(10);
            Count<int>(np.arange(10).astype(np.int32)).Should().Be(10);
            Count<uint>(np.arange(10).astype(np.uint32)).Should().Be(10);
            Count<long>(np.arange(10).astype(np.int64)).Should().Be(10);
            Count<ulong>(np.arange(10).astype(np.uint64)).Should().Be(10);
            Count<char>(np.arange(10).astype(typeof(char))).Should().Be(10);
            Count<Half>(np.arange(10).astype(np.float16)).Should().Be(10);
            Count<float>(np.arange(10).astype(np.float32)).Should().Be(10);
            Count<double>(np.arange(10).astype(np.float64)).Should().Be(10);
            Count<decimal>(np.arange(10).astype(np.@decimal)).Should().Be(10);
            Count<System.Numerics.Complex>(np.arange(10).astype(np.complex128)).Should().Be(10);
        }

        private static long Count<T>(NDArray a) where T : unmanaged
        {
            long n = 0;
            foreach (ref T _ in np.nditer<T>(a))
                n++;

            return n;
        }

        [TestMethod]
        public void Typed_ReadsCorrectValues_AcrossDtypeFamilies()
        {
            var ints = new List<long>();
            foreach (ref int x in np.nditer<int>(np.arange(5).astype(np.int32)))
                ints.Add(x);
            ints.Should().Equal(0L, 1L, 2L, 3L, 4L);

            var halves = new List<double>();
            foreach (ref Half x in np.nditer<Half>(np.arange(4).astype(np.float16)))
                halves.Add((double)x);
            halves.Should().Equal(0d, 1d, 2d, 3d);

            var decs = new List<decimal>();
            foreach (ref decimal x in np.nditer<decimal>(np.arange(4).astype(np.@decimal)))
                decs.Add(x);
            decs.Should().Equal(0m, 1m, 2m, 3m);

            var reals = new List<double>();
            foreach (ref System.Numerics.Complex x in np.nditer<System.Numerics.Complex>(np.arange(4).astype(np.complex128)))
                reals.Add(x.Real);
            reals.Should().Equal(0d, 1d, 2d, 3d);
        }

        // ------------------------------------------------------------------ write-through

        [TestMethod]
        public void Typed_WriteThrough_Contiguous()
        {
            var a = np.arange(6).astype(np.float64);
            foreach (ref double x in np.nditer<double>(a, writeable: true))
                x *= 10;

            for (int i = 0; i < 6; i++)
                a.GetAtIndex<double>(i).Should().Be(i * 10);
        }

        [TestMethod]
        public void Typed_WriteThrough_StridedView_ReachesTheParent()
        {
            var parent = np.arange(12).astype(np.float64).reshape(3, 4);
            foreach (ref double x in np.nditer<double>(parent[":, ::2"], writeable: true))
                x = -1;

            // columns 0 and 2 overwritten, 1 and 3 untouched
            parent.GetAtIndex<double>(0).Should().Be(-1);
            parent.GetAtIndex<double>(1).Should().Be(1);
            parent.GetAtIndex<double>(2).Should().Be(-1);
            parent.GetAtIndex<double>(3).Should().Be(3);
        }

        [TestMethod]
        public void Typed_WriteThrough_EveryMutableDtype()
        {
            var i32 = np.arange(4).astype(np.int32);
            foreach (ref int x in np.nditer<int>(i32, writeable: true)) x += 1;
            i32.GetAtIndex<int>(3).Should().Be(4);

            var f16 = np.arange(4).astype(np.float16);
            foreach (ref Half x in np.nditer<Half>(f16, writeable: true)) x = (Half)((double)x + 1);
            ((double)f16.GetAtIndex<Half>(3)).Should().Be(4);

            var dec = np.arange(4).astype(np.@decimal);
            foreach (ref decimal x in np.nditer<decimal>(dec, writeable: true)) x += 1m;
            dec.GetAtIndex<decimal>(3).Should().Be(4m);

            var cpx = np.arange(4).astype(np.complex128);
            foreach (ref System.Numerics.Complex x in np.nditer<System.Numerics.Complex>(cpx, writeable: true)) x += 1;
            cpx.GetAtIndex<System.Numerics.Complex>(3).Real.Should().Be(4);
        }

        // ------------------------------------------------------------------ guards

        [TestMethod]
        public void Typed_DtypeMismatch_Throws_RatherThanReinterpretingBytes()
        {
            // The whole point: `ref` cannot convert, so a silent reinterpret would hand out garbage.
            var ex = Assert.ThrowsException<ArgumentException>(
                () => { foreach (ref double _ in np.nditer<double>(np.arange(4).astype(np.int32))) { } });
            ex.Message.Should().Contain("called on a Int32 array");
        }

        [TestMethod]
        public void Typed_WriteToBroadcastView_Throws_WithNumPysMessage()
        {
            // numpy: np.nditer(np.broadcast_to(...), op_flags=['readwrite'])
            //     -> ValueError: operand array with iterator write flag set is read-only
            var bcast = np.broadcast_to(np.arange(4).astype(np.float64), new Shape(3, 4));
            var ex = Assert.ThrowsException<ArgumentException>(
                () => { foreach (ref double _ in np.nditer<double>(bcast, writeable: true)) { } });
            ex.Message.Should().Contain("operand array with iterator write flag set is read-only");
        }

        [TestMethod]
        public void Typed_ReadingABroadcastView_IsAllowed()
        {
            var bcast = np.broadcast_to(np.arange(4).astype(np.float64), new Shape(3, 4));
            double sum = 0;
            foreach (ref double x in np.nditer<double>(bcast))
                sum += x;

            sum.Should().Be(18);   // (0+1+2+3) * 3
        }

        [TestMethod]
        public void Typed_BadOrder_Throws()
        {
            var ex = Assert.ThrowsException<ArgumentException>(
                () => { foreach (ref double _ in np.nditer<double>(np.arange(4).astype(np.float64), order: 'Q')) { } });
            ex.Message.Should().Contain("order must be one of 'C', 'F', 'A', or 'K'");
        }

        [TestMethod]
        public void Typed_NullOperand_Throws()
        {
            Assert.ThrowsException<ArgumentNullException>(
                () => { foreach (ref double _ in np.nditer<double>(null)) { } });
        }

        // ------------------------------------------------------------------ re-enumeration

        [TestMethod]
        public void Typed_ReEnumeration_RestartsAndDoesNotUseFreedState()
        {
            // Unlike the class-based np.nditer (NumPy's `iter(x) is x`, which RESUMES), the typed
            // form builds a fresh iterator per foreach. If GetEnumerator returned `this`, the
            // second pass would walk state the first pass had already freed.
            var a = np.arange(4).astype(np.float64);
            var iter = np.nditer<double>(a);

            double first = 0, second = 0;
            foreach (ref double x in iter) first += x;
            foreach (ref double x in iter) second += x;

            first.Should().Be(6);
            second.Should().Be(6);
        }

        // ------------------------------------------------------------------ nditer_chunks

        [TestMethod]
        public void Chunks_ContiguousIsASingleChunk()
        {
            var a = np.arange(100).astype(np.float64).reshape(10, 10);
            int chunks = 0;
            long total = 0;
            foreach (Span<double> c in np.nditer_chunks<double>(a))
            {
                chunks++;
                total += c.Length;
            }

            chunks.Should().Be(1);
            total.Should().Be(100);
        }

        [TestMethod]
        public void Chunks_CoverEveryElement_OnSpannableLayouts()
        {
            var b = np.arange(100).astype(np.float64).reshape(10, 10);
            var cases = new[]
            {
                b, b.T, b[":, ::-1"], b["2:8, 1:9"],
                np.broadcast_to(np.arange(10).astype(np.float64), new Shape(10, 10)),
                np.array(3.5),
                np.zeros(new Shape(0, 5), np.float64),
            };

            foreach (var a in cases)
            {
                long n = 0;
                double sum = 0;
                foreach (Span<double> c in np.nditer_chunks<double>(a))
                {
                    n += c.Length;
                    for (int i = 0; i < c.Length; i++)
                        sum += c[i];
                }

                n.Should().Be(a.size);
                if (a.size > 0)
                    sum.Should().BeApproximately((double)np.sum(a), 1e-9);
            }
        }

        [TestMethod]
        public void Chunks_SteppedView_ThrowsUpFront_NotMidLoop()
        {
            // A Span is contiguous by definition; a[:, ::2] iterates with element stride 2.
            var b = np.arange(100).astype(np.float64).reshape(10, 10);
            var ex = Assert.ThrowsException<NotSupportedException>(
                () => { foreach (Span<double> _ in np.nditer_chunks<double>(b[":, ::2"])) { } });

            ex.Message.Should().Contain("unit-stride inner loop");
            ex.Message.Should().Contain("element stride 2");
        }

        [TestMethod]
        public void Chunks_WriteThrough()
        {
            var a = np.arange(8).astype(np.float64);
            foreach (Span<double> c in np.nditer_chunks<double>(a, writeable: true))
                for (int i = 0; i < c.Length; i++)
                    c[i] += 100;

            a.GetAtIndex<double>(7).Should().Be(107);
        }

        [TestMethod]
        public void Chunks_AgreeWithElementWalk()
        {
            var b = np.arange(60).astype(np.float64).reshape(6, 10);
            foreach (var a in new[] {b, b.T, b[":, ::-1"], b["1:5, 2:8"]})
            {
                var byElement = new List<double>();
                foreach (ref double x in np.nditer<double>(a))
                    byElement.Add(x);

                var byChunk = new List<double>();
                foreach (Span<double> c in np.nditer_chunks<double>(a))
                    for (int i = 0; i < c.Length; i++)
                        byChunk.Add(c[i]);

                byChunk.Should().Equal(byElement);
            }
        }
    }
}
