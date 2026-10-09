using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     <c>np.nditer</c>'s ITERATION PROTOCOL — the getters and setters of <c>nditer_pywrap.c</c>, the seeks and their
    ///     range checks, <c>it[i]</c> reads and writes, <c>copy()</c>, the closed state, <c>remove_axis</c> /
    ///     <c>remove_multi_index</c> / <c>enable_external_loop</c>, empty and 0-d walks, buffered chunking, and a walk of every
    ///     dtype — the parts the earlier suites (<see cref="np_nditer_Test"/>, <see cref="np_nditer_NumPyParity_Test"/>) do not
    ///     pin. Every expected value and every error text was probed against NumPy 2.4.2 (2026-10-09).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The protocol is NumPy's advance-on-entry one (<c>npyiter_next</c>): the first pull publishes element 0 without
    ///     moving, every later pull moves first. Inside a <c>foreach</c> body the position getters therefore read the element
    ///     the body is handling, and after <c>k</c> pulls a <c>copy()</c> resumes with element <c>k</c>.
    ///     </para>
    ///     <para>
    ///     NumPy's exception types map to the house types of the same name; NumPy's OverflowError is
    ///     <see cref="OverflowException"/>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class np_nditer_Protocol_Test
    {
        /// <summary>Walks an iterator to its end and reads operand 0 at every step.</summary>
        /// <param name="it">The iterator to walk; it is finished afterwards.</param>
        /// <returns>The boxed scalar of each 0-d view, in iteration order.</returns>
        private static List<object> Walk(np.NDIterator it) => it.Select(v => v[0].GetAtIndex(0)).ToList();

        /// <summary>The iterator's current multi-index as text (<c>"1,2"</c>), for compact sequence assertions.</summary>
        /// <param name="it">An iterator that tracks a multi-index and is not past its end.</param>
        /// <returns>The coordinates joined by commas.</returns>
        private static string Mi(np.NDIterator it) => string.Join(",", it.multi_index);

        // ------------------------------------------------------------------ the end of a walk

        /// <summary>
        ///     <c>iternext()</c> reports the move it made: true while an element remains, false on the move past the last one —
        ///     and false again on every later call, with <c>finished</c> staying true (NumPy's probed trace).
        /// </summary>
        [TestMethod]
        public void Iternext_ReturnsFalseAtTheEnd_AndKeepsReturningFalse()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            var trace = new List<bool>
            {
                it.finished, it.iternext(), it.iternext(), it.iternext(), it.finished, it.iternext(), it.finished,
            };
            trace.Should().Equal(false, true, true, false, true, false, true);
        }

        /// <summary>
        ///     Past the end, every POSITION member raises NumPy's "Iterator is past the end" — the views, the indices, the
        ///     shape and the setter alike — while the size members still answer.
        /// </summary>
        [TestMethod]
        public void PositionMembers_PastTheEnd_AreValueErrors_SizeMembersAreNot()
        {
            using var a = np.arange(2);
            using var it = np.nditer(a, new[] { "multi_index", "c_index" }, new[] { "readwrite" });
            foreach (var _ in it) { }

            const string pastTheEnd = "Iterator is past the end";
            new Action(() => _ = it.value).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            new Action(() => _ = it[0]).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            new Action(() => _ = it.iterindex).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            new Action(() => _ = it.multi_index).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            new Action(() => _ = it.index).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            new Action(() => _ = it.shape).Should().Throw<ValueError>().WithMessage(pastTheEnd);
            using var nine = NDArray.Scalar(9L);
            new Action(() => it[0] = nine).Should().Throw<ValueError>().WithMessage(pastTheEnd);

            it.itersize.Should().Be(2);
            it.ndim.Should().Be(1);
            it.nop.Should().Be(1);
            it.finished.Should().BeTrue();
            it.iterrange.Should().Be((0L, 2L));
        }

        /// <summary><c>reset()</c> restarts a finished walk and a seeked one alike: the next pull yields element 0.</summary>
        [TestMethod]
        public void Reset_RestartsAFinishedOrASeekedWalk()
        {
            using var a = np.arange(4);
            using var it = np.nditer(a);
            Walk(it).Should().Equal(0L, 1L, 2L, 3L);
            it.reset();
            it.finished.Should().BeFalse();
            Walk(it).Should().Equal(0L, 1L, 2L, 3L);

            it.reset();
            it.iterindex = 2;
            it.reset();
            Walk(it).Should().Equal(0L, 1L, 2L, 3L);
        }

        // ------------------------------------------------------------------ iterindex

        /// <summary>
        ///     Setting <c>iterindex</c> moves the iterator, and the NEXT pull yields that element (NumPy's
        ///     <c>npyiter_iterindex_set</c> clears <c>started</c>): position 4 of <c>arange(10, 16)</c> walks <c>14, 15</c>, the
        ///     last position walks its one element.
        /// </summary>
        [TestMethod]
        public void IterIndex_Seek_NextPullYieldsThatElement()
        {
            using var a = np.arange(10, 16);
            using (var it = np.nditer(a))
            {
                it.iterindex = 4;
                Walk(it).Should().Equal(14L, 15L);
            }

            using var b = np.arange(10, 15);
            using (var it = np.nditer(b))
            {
                it.iterindex = 4;
                Walk(it).Should().Equal(14L);
            }
        }

        /// <summary>An <c>iterindex</c> outside <c>[0, itersize)</c> — past the end or negative — is NumPy's IndexError.</summary>
        [TestMethod]
        public void IterIndex_OutsideTheIteration_IsAnIndexError()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            foreach (var bad in new long[] { 3, -1 })
            {
                var act = () => it.iterindex = bad;
                act.Should().Throw<IndexError>()
                   .WithMessage("Iterator GotoIterIndex called with an iterindex outside the iteration range.");
            }
        }

        /// <summary>An external-loop iterator moves by chunks, so it cannot seek to an element.</summary>
        [TestMethod]
        public void IterIndex_WithExternalLoop_IsRefused()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a, new[] { "external_loop" });
            var act = () => it.iterindex = 1;
            act.Should().Throw<ValueError>().WithMessage("Cannot call GotoIterIndex on an iterator which has the flag EXTERNAL_LOOP");
        }

        /// <summary>
        ///     <c>iterindex</c> counts in the iterator's OWN order: an F-order walk of a C <c>(2, 3)</c> array reads
        ///     <c>0 … 5</c> while the values go <c>0 3 1 4 2 5</c>.
        /// </summary>
        [TestMethod]
        public void IterIndex_DuringAWalk_CountsTheIterationOrder()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, order: 'F');
            it.Select(_ => it.iterindex).ToList().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        // ------------------------------------------------------------------ multi_index

        /// <summary>
        ///     Setting <c>multi_index</c> moves the iterator there: <c>(1, 2)</c> of a <c>(3, 4)</c> array is iteration
        ///     position 6, and the next pulls yield <c>(1, 2) 6</c>, <c>(1, 3) 7</c>, <c>(2, 0) 8</c>.
        /// </summary>
        [TestMethod]
        public void MultiIndex_Seek_NextPullYieldsThatElement()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var it = np.nditer(a, new[] { "multi_index" });
            it.multi_index = new long[] { 1, 2 };
            it.iterindex.Should().Be(6);
            it.Take(3).Select(v => $"{Mi(it)}:{v[0].GetAtIndex(0)}").Should().Equal("1,2:6", "1,3:7", "2,0:8");
        }

        /// <summary>
        ///     The <c>multi_index</c> setter's checks, in NumPy's order and texts: the length (ValueError), each coordinate's
        ///     range — a negative coordinate included, nothing wraps (IndexError) — and a null value (ValueError).
        /// </summary>
        [TestMethod]
        public void MultiIndex_SetterChecks_AreNumPys()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index" });

            var shortIndex = () => it.multi_index = new long[] { 1 };
            shortIndex.Should().Throw<ValueError>().WithMessage("Wrong number of indices");

            foreach (var bad in new[] { new long[] { 2, 0 }, new long[] { 0, -1 } })
            {
                var act = () => it.multi_index = bad;
                act.Should().Throw<IndexError>().WithMessage("Iterator GotoMultiIndex called with an out-of-bounds multi-index");
            }

            var none = () => it.multi_index = null;
            none.Should().Throw<ValueError>().WithMessage("multi_index must be set with a sequence");
        }

        /// <summary>A buffered iterator tracks a multi-index but cannot seek to one.</summary>
        [TestMethod]
        public void MultiIndex_Buffered_CannotSeek()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index", "buffered" });
            var act = () => it.multi_index = new long[] { 1, 0 };
            act.Should().Throw<ValueError>().WithMessage("Cannot call GotoMultiIndex on an iterator which is buffered");
        }

        /// <summary>Without the <c>multi_index</c> flag both the getter and the setter refuse with NumPy's text.</summary>
        [TestMethod]
        public void MultiIndex_NotTracked_IsRefused()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            new Action(() => _ = it.multi_index).Should().Throw<ValueError>().WithMessage("Iterator is not tracking a multi-index");
            var set = () => it.multi_index = new long[] { 0 };
            set.Should().Throw<ValueError>().WithMessage("Iterator is not tracking a multi-index");
        }

        // ------------------------------------------------------------------ index

        /// <summary>Without <c>c_index</c> / <c>f_index</c> both the getter and the setter refuse with NumPy's text.</summary>
        [TestMethod]
        public void Index_NotTracked_IsRefused()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a);
            new Action(() => _ = it.index).Should().Throw<ValueError>().WithMessage("Iterator does not have an index");
            var set = () => it.index = 0;
            set.Should().Throw<ValueError>().WithMessage("Iterator does not have an index");
        }

        /// <summary>An index outside <c>[0, itersize)</c> is NumPy's IndexError; a buffered iterator cannot seek at all.</summary>
        [TestMethod]
        public void Index_SetterChecks_AreNumPys()
        {
            using var a = np.arange(3);
            using (var it = np.nditer(a, new[] { "c_index" }))
            {
                var act = () => it.index = 3;
                act.Should().Throw<IndexError>().WithMessage("Iterator GotoIndex called with an out-of-bounds index");
            }

            using (var it = np.nditer(a, new[] { "c_index", "buffered" }))
            {
                var act = () => it.index = 1;
                act.Should().Throw<ValueError>().WithMessage("Cannot call GotoIndex on an iterator which is buffered");
            }
        }

        /// <summary>
        ///     Setting an <c>f_index</c> seeks to that F-order flat position: index 3 of a C <c>(2, 3)</c> array is element 4,
        ///     and the walk goes on in the iterator's C order — <c>(3, 4)</c>, then <c>(5, 5)</c>.
        /// </summary>
        [TestMethod]
        public void Index_FIndexSeek_YieldsFromThatFlatPosition()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "f_index" });
            it.index = 3;
            it.Select(v => $"{it.index}:{v[0].GetAtIndex(0)}").Should().Equal("3:4", "5:5");
        }

        /// <summary>
        ///     After an <c>iterindex</c> seek the tracked <c>f_index</c> is that element's F-order position: iteration
        ///     position 4 of a C-order walk is element 4, whose F index is 3.
        /// </summary>
        [TestMethod]
        public void Index_AfterAnIterIndexSeek_IsTheFlatPositionOfThatElement()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "f_index" }, order: 'C');
            it.iterindex = 4;
            it.Select(v => $"{it.iterindex}:{it.index}:{v[0].GetAtIndex(0)}").Should().Equal("4:3:4", "5:5:5");
        }

        // ------------------------------------------------------------------ iterrange

        /// <summary>
        ///     The <c>iterrange</c> setter's checks: the <c>ranged</c> flag, the bounds, the order of the ends, and a closed
        ///     iterator (NumPy's four texts).
        /// </summary>
        [TestMethod]
        public void IterRange_SetterChecks_AreNumPys()
        {
            using var a = np.arange(5);
            using (var plain = np.nditer(a))
            {
                var notRanged = () => plain.iterrange = (1, 3);
                notRanged.Should().Throw<ValueError>()
                         .WithMessage("Cannot call ResetToIterIndexRange on an iterator without requesting ranged iteration support in the constructor");
            }

            using (var it = np.nditer(a, new[] { "ranged" }))
            {
                var outOfBounds = () => it.iterrange = (0, 7);
                outOfBounds.Should().Throw<ValueError>().WithMessage("Out-of-bounds range [0, 7) passed to ResetToIterIndexRange");
                var reversed = () => it.iterrange = (3, 1);
                reversed.Should().Throw<ValueError>().WithMessage("Invalid range [3, 1) passed to ResetToIterIndexRange");
            }

            var closed = np.nditer(a, new[] { "ranged" });
            closed.close();
            var afterClose = () => closed.iterrange = (0, 1);
            afterClose.Should().Throw<ValueError>().WithMessage("Iterator is invalid");
        }

        /// <summary>An empty range is finished at once and walks nothing; the default range is the whole iteration.</summary>
        [TestMethod]
        public void IterRange_EmptyIsFinishedAtOnce_DefaultIsEverything()
        {
            using var a = np.arange(5);
            using (var it = np.nditer(a, new[] { "ranged" }))
            {
                it.iterrange = (2, 2);
                it.finished.Should().BeTrue();
                Walk(it).Should().BeEmpty();
            }

            using var seven = np.arange(7);
            using (var it = np.nditer(seven))
                it.iterrange.Should().Be((0L, 7L));
        }

        /// <summary>
        ///     A sub-range walks only its positions, with the multi-index of each: <c>[5, 9)</c> of a C <c>(3, 4)</c> array is
        ///     <c>(1, 1) 5</c> … <c>(2, 0) 8</c>, and the getter reports the range set.
        /// </summary>
        [TestMethod]
        public void IterRange_WithMultiIndex_WalksTheSubRange()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var it = np.nditer(a, new[] { "ranged", "multi_index" });
            it.iterrange = (5, 9);
            it.iterrange.Should().Be((5L, 9L));
            it.Select(v => $"{Mi(it)}:{v[0].GetAtIndex(0)}").Should().Equal("1,1:5", "1,2:6", "1,3:7", "2,0:8");
        }

        /// <summary>
        ///     Inside a restricted range every seek is checked against the RANGE, not the whole iteration — three IndexError
        ///     texts, NumPy's own (one ends in a period, one does not).
        /// </summary>
        [TestMethod]
        public void Seeks_OutsideARestrictedRange_AreIndexErrors()
        {
            using var a = np.arange(10);
            using var it = np.nditer(a, new[] { "ranged", "multi_index", "c_index" });
            it.iterrange = (2, 5);

            var iterIndex = () => it.iterindex = 7;
            iterIndex.Should().Throw<IndexError>()
                     .WithMessage("Iterator GotoIterIndex called with an iterindex outside the iteration range.");
            var multiIndex = () => it.multi_index = new long[] { 7 };
            multiIndex.Should().Throw<IndexError>()
                      .WithMessage("Iterator GotoMultiIndex called with a multi-index outside the restricted iteration range");
            var index = () => it.index = 7;
            index.Should().Throw<IndexError>()
                 .WithMessage("Iterator GotoIndex called with an index outside the restricted iteration range.");
        }

        // ------------------------------------------------------------------ it[i] and value

        /// <summary>
        ///     <c>it[i]</c> takes a negative index from the last operand, and an index outside <c>[-nop, nop)</c> is NumPy's
        ///     IndexError reporting the index AS GIVEN.
        /// </summary>
        [TestMethod]
        public void OperandIndex_NegativeCountsFromTheEnd_OutsideIsAnIndexError()
        {
            using var a = np.arange(3);
            using var b = np.arange(10, 13);
            using var it = np.nditer(new[] { a, b });
            it[-1].GetAtIndex(0).Should().Be(10L);
            it[-2].GetAtIndex(0).Should().Be(0L);
            new Action(() => _ = it[2]).Should().Throw<IndexError>().WithMessage("Iterator operand index 2 is out of bounds");
            new Action(() => _ = it[-3]).Should().Throw<IndexError>().WithMessage("Iterator operand index -3 is out of bounds");
        }

        /// <summary>
        ///     <c>value</c> holds one view per operand at the CURRENT position: <c>(0, 10)</c> at the start, <c>(2, 12)</c>
        ///     after a seek to position 2. Under <c>external_loop</c> it is the whole chunk.
        /// </summary>
        [TestMethod]
        public void Value_HoldsEveryOperandAtThePosition()
        {
            using var a = np.arange(4);
            using var b = np.arange(10, 14);
            using (var it = np.nditer(new[] { a, b }))
            {
                it.value.Select(x => x.GetAtIndex(0)).Should().Equal(0L, 10L);
                it.iterindex = 2;
                it.value.Select(x => x.GetAtIndex(0)).Should().Equal(2L, 12L);
            }

            using var six = np.arange(6);
            using (var it = np.nditer(six, new[] { "external_loop" }))
                it.value[0].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        /// <summary>
        ///     The setter packs a Python float into an integer operand by truncation (NumPy's <c>PyArray_Pack</c>):
        ///     <c>it[0] = 2.7</c> stores 2.
        /// </summary>
        [TestMethod]
        public void Setter_PythonFloat_TruncatesIntoAnIntegerOperand()
        {
            using var a = np.arange(3);
            using (var it = np.nditer(a, op_flags: new[] { "readwrite" }))
            {
                using var value = NDArray.Scalar(2.7);
                it[0] = value;
            }

            a.ToArray<long>().Should().Equal(2L, 1L, 2L);
        }

        /// <summary>A NaN cannot be packed into an integer operand: NumPy's ValueError, nothing is stored.</summary>
        [TestMethod]
        public void Setter_NaNIntoAnIntegerOperand_IsAValueError()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a, op_flags: new[] { "readwrite" });
            using var nan = NDArray.Scalar(double.NaN);
            var act = () => it[0] = nan;
            act.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
        }

        /// <summary>A text value is PARSED into a numeric operand (NumPy's <c>int('12')</c>): <c>it[0] = '12'</c> stores 12.</summary>
        [TestMethod]
        public void Setter_TextIntoAnIntegerOperand_IsParsed()
        {
            using var a = np.arange(3);
            using (var it = np.nditer(a, op_flags: new[] { "readwrite" }))
            {
                using var text = np.array("12");
                it[0] = text;
            }

            a.ToArray<long>().Should().Equal(12L, 1L, 2L);
        }

        /// <summary>
        ///     A value with dimensions is assigned to a 1-D view of the current inner loop — one element without
        ///     <c>external_loop</c> — so two values do not fit: NumPy's broadcast ValueError.
        /// </summary>
        [TestMethod]
        public void Setter_ArrayIntoOneElement_IsABroadcastError()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a, op_flags: new[] { "readwrite" });
            using var two = np.array(new long[] { 1, 2 });
            var act = () => it[0] = two;
            act.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,) into shape (1,)");
        }

        /// <summary>
        ///     Under <c>external_loop</c> the setter assigns the WHOLE chunk: a chunk of 3 takes <c>[7, 8, 9]</c>, and a
        ///     2-element value is the broadcast ValueError for shape (3,).
        /// </summary>
        [TestMethod]
        public void Setter_ExternalLoop_AssignsTheWholeChunk()
        {
            using var a = np.arange(6);
            using (var it = np.nditer(a, new[] { "external_loop", "buffered" }, new[] { "readwrite" }, buffersize: 3))
            {
                using var two = np.array(new long[] { 1, 2 });
                var act = () => it[0] = two;
                act.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,) into shape (3,)");

                using var chunk = np.array(new long[] { 7, 8, 9 });
                it[0] = chunk;
            }

            a.ToArray<long>().Should().Equal(7L, 8L, 9L, 3L, 4L, 5L);
        }

        /// <summary>
        ///     A <c>writeonly</c> operand gets writeable views and every write lands: <c>[0, 10, 20, 30]</c>.
        /// </summary>
        [TestMethod]
        public void WriteOnlyOperand_WritesThrough()
        {
            using var a = np.arange(4);
            using (var it = np.nditer(a, op_flags: new[] { "writeonly" }))
            {
                it[0].flags.writeable.Should().BeTrue();
                long i = 0;
                foreach (var v in it)
                {
                    using var value = NDArray.Scalar(10 * i++);
                    np.copyto(v[0], value);
                }
            }

            a.ToArray<long>().Should().Equal(0L, 10L, 20L, 30L);
        }

        // ------------------------------------------------------------------ copy

        /// <summary>
        ///     A copy taken after two pulls resumes with the THIRD element (it carries NumPy's <c>started</c> flag), and it moves
        ///     independently of the original: both walk <c>2, 3, 4</c>.
        /// </summary>
        [TestMethod]
        public void Copy_MidWalk_ResumesWithTheNextElement_Independently()
        {
            using var a = np.arange(5);
            using var it = np.nditer(a);
            it.MoveNext();
            it.MoveNext();
            using var copy = it.copy();
            Walk(copy).Should().Equal(2L, 3L, 4L);
            Walk(it).Should().Equal(2L, 3L, 4L);
        }

        /// <summary>A copy carries the tracked flat index too: after two pulls it reads <c>(2, 2)</c>, <c>(3, 3)</c>.</summary>
        [TestMethod]
        public void Copy_CarriesTheTrackedIndex()
        {
            using var a = np.arange(4);
            using var it = np.nditer(a, new[] { "c_index" });
            it.MoveNext();
            it.MoveNext();
            using var copy = it.copy();
            copy.Select(v => $"{copy.index}:{v[0].GetAtIndex(0)}").Should().Equal("2:2", "3:3");
        }

        /// <summary>
        ///     A copy taken before the first pull starts at element 0; a copy of a finished iterator is finished and walks
        ///     nothing.
        /// </summary>
        [TestMethod]
        public void Copy_BeforeTheFirstPull_OrAfterTheEnd()
        {
            using var a = np.arange(3);
            using (var it = np.nditer(a))
            {
                using var copy = it.copy();
                Walk(copy).Should().Equal(0L, 1L, 2L);
            }

            using (var it = np.nditer(a))
            {
                foreach (var _ in it) { }
                using var copy = it.copy();
                copy.finished.Should().BeTrue();
                Walk(copy).Should().BeEmpty();
            }
        }

        // ------------------------------------------------------------------ the closed state

        /// <summary>
        ///     A closed iterator reads <c>finished</c> false and <c>iternext()</c> false, refuses the members that need it
        ///     with "Iterator is invalid", and <c>value</c> with "Iterator is past the end" (NumPy's position guard runs
        ///     first).
        /// </summary>
        [TestMethod]
        public void ClosedIterator_Contract()
        {
            using var a = np.arange(3);
            var it = np.nditer(a);
            it.close();
            it.finished.Should().BeFalse();
            it.iternext().Should().BeFalse();

            const string invalid = "Iterator is invalid";
            new Action(() => _ = it.operands).Should().Throw<ValueError>().WithMessage(invalid);
            new Action(() => _ = it.has_index).Should().Throw<ValueError>().WithMessage(invalid);
            new Action(() => it.reset()).Should().Throw<ValueError>().WithMessage(invalid);
            new Action(() => it.copy()).Should().Throw<ValueError>().WithMessage(invalid);
            new Action(() => _ = it.value).Should().Throw<ValueError>().WithMessage("Iterator is past the end");
        }

        // ------------------------------------------------------------------ properties

        /// <summary>
        ///     The flag and size properties of a tracked <c>(2, 3)</c> walk: an index and a multi-index, no delayed
        ///     allocation, no Python API, one operand, two axes of six elements in the operand's own shape.
        /// </summary>
        [TestMethod]
        public void FlagAndSizeProperties_ReportTheConstruction()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index", "c_index" });
            it.has_index.Should().BeTrue();
            it.has_multi_index.Should().BeTrue();
            it.has_delayed_bufalloc.Should().BeFalse();
            it.iterationneedsapi.Should().BeFalse();
            it.nop.Should().Be(1);
            it.ndim.Should().Be(2);
            it.itersize.Should().Be(6);
            it.shape.Should().Equal(2L, 3L);
        }

        /// <summary>
        ///     Without a multi-index <c>shape</c> lists the iterator's own axes INNERMOST first: <c>a[::2, ::2]</c> of a
        ///     <c>(4, 6)</c> array (a <c>(2, 3)</c> view that cannot coalesce) reports <c>(3, 2)</c>.
        /// </summary>
        [TestMethod]
        public void Shape_WithoutMultiIndex_ListsTheInnermostAxisFirst()
        {
            using var a = np.arange(24).reshape(4, 6);
            using var stepped = a["::2, ::2"];
            using var it = np.nditer(stepped);
            it.shape.Should().Equal(3L, 2L);
            it.ndim.Should().Be(2);
        }

        /// <summary>
        ///     Each operand keeps its own dtype (an int8 and a float64 operand read as <see cref="sbyte"/> and
        ///     <see cref="double"/>), and an allocated output takes their result type.
        /// </summary>
        [TestMethod]
        public void Operands_KeepTheirDtypes_AnAllocatedOutputTakesTheResultType()
        {
            using var i8 = np.arange(3).astype(np.int8);
            using var f8 = np.array(new[] { 0.5, 1.5, 2.5 });
            using (var it = np.nditer(new[] { i8, f8 }))
            {
                it.Select(v => (v[0].GetAtIndex(0), v[1].GetAtIndex(0))).Should()
                  .Equal(((object)(sbyte)0, (object)0.5), ((sbyte)1, 1.5), ((sbyte)2, 2.5));
            }

            using (var it = np.nditer(new NDArray[] { i8, f8, null }))
                it.dtypes.Select(d => d.str).Should().Equal("|i1", "<f8", "<f8");
        }

        /// <summary>
        ///     <c>operands</c> holds the caller's array itself and the output the iterator allocated (shape <c>(3,)</c>, the
        ///     input's int64).
        /// </summary>
        [TestMethod]
        public void Operands_HoldTheCallersArrayAndTheAllocatedOutput()
        {
            using var a = np.arange(3);
            using var it = np.nditer(new NDArray[] { a, null });
            var ops = it.operands;
            ops.Should().HaveCount(2);
            ReferenceEquals(ops[0], a).Should().BeTrue();
            ops[1].shape.Should().Equal(3L);
            ops[1].dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     <c>itviews</c> exposes each operand in the iterator's axis order — a transposed <c>(3, 2)</c> view walked in
        ///     memory order is a <c>(2, 3)</c> view <c>[[0, 1, 2], [3, 4, 5]]</c> — and refuses under buffering.
        /// </summary>
        [TestMethod]
        public void ItViews_FollowTheIterationOrder_AndRefuseUnderBuffering()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var t = a.T;
            using (var it = np.nditer(t, new[] { "multi_index" }))
            {
                var view = it.itviews[0];
                view.shape.Should().Equal(2L, 3L);
                view.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            }

            using var three = np.arange(3);
            using (var it = np.nditer(three, new[] { "buffered" }))
            {
                new Action(() => _ = it.itviews).Should().Throw<ValueError>()
                                                 .WithMessage("cannot provide an iterator view when buffering is enabled");
            }
        }

        // ------------------------------------------------------------------ 0-d, empty and broadcast operands

        /// <summary>
        ///     A 0-d operand is one element with no axes: <c>ndim</c> 0, an empty shape and multi-index, flat index 0 — and
        ///     under <c>external_loop</c> ONE chunk of shape <c>(1,)</c>.
        /// </summary>
        [TestMethod]
        public void ZeroD_WalksOneElement()
        {
            using var five = NDArray.Scalar(5L);
            using (var it = np.nditer(five, new[] { "multi_index", "c_index" }))
            {
                it.ndim.Should().Be(0);
                it.shape.Should().BeEmpty();
                it.itersize.Should().Be(1);
                it.multi_index.Should().BeEmpty();
                it.index.Should().Be(0);
                Walk(it).Should().Equal(5L);
            }

            using (var it = np.nditer(five, new[] { "external_loop" }))
            {
                var chunks = it.Select(v => (v[0].shape, v[0].ToArray<long>())).ToList();
                chunks.Should().HaveCount(1);
                chunks[0].Item1.Should().Equal(1L);
                chunks[0].Item2.Should().Equal(5L);
            }
        }

        /// <summary>
        ///     A zero-size operand under <c>zerosize_ok</c> is finished from the start, walks nothing, stays finished after
        ///     <c>reset()</c>, and has no shape to report ("Iterator is past the end").
        /// </summary>
        [TestMethod]
        public void ZeroSizeOk_EmptyIteration_IsFinishedFromTheStart()
        {
            using var empty = np.zeros(new Shape(2, 0));
            using var it = np.nditer(empty, new[] { "zerosize_ok" });
            it.itersize.Should().Be(0);
            it.finished.Should().BeTrue();
            Walk(it).Should().BeEmpty();
            it.finished.Should().BeTrue();
            it.reset();
            it.finished.Should().BeTrue();
            new Action(() => _ = it.shape).Should().Throw<ValueError>().WithMessage("Iterator is past the end");
        }

        /// <summary>A broadcast (stride-0) operand repeats its row, and the multi-index walks the broadcast shape.</summary>
        [TestMethod]
        public void BroadcastOperand_RepeatsItsValues()
        {
            using var row = np.array(new long[] { 1, 2, 3 });
            using var b = np.broadcast_to(row, new Shape(2, 3));
            using var it = np.nditer(b, new[] { "multi_index" });
            it.Select(v => $"{Mi(it)}:{v[0].GetAtIndex(0)}").Should()
              .Equal("0,0:1", "0,1:2", "0,2:3", "1,0:1", "1,1:2", "1,2:3");
        }

        /// <summary>
        ///     Under <c>order='F'</c> the multi-index and a <c>c_index</c> follow the F walk while still naming C positions:
        ///     <c>(0,0) (1,0) (0,1) …</c> and C indices <c>0 3 1 4 2 5</c>.
        /// </summary>
        [TestMethod]
        public void MultiIndexAndCIndex_UnderFOrder()
        {
            using var a = np.arange(6).reshape(2, 3);
            using (var it = np.nditer(a, new[] { "multi_index" }, order: 'F'))
            {
                it.Select(v => $"{Mi(it)}:{v[0].GetAtIndex(0)}").Should()
                  .Equal("0,0:0", "1,0:3", "0,1:1", "1,1:4", "0,2:2", "1,2:5");
            }

            using (var it = np.nditer(a, new[] { "c_index" }, order: 'F'))
                it.Select(v => $"{it.index}:{v[0].GetAtIndex(0)}").Should().Equal("0:0", "3:3", "1:1", "4:4", "2:2", "5:5");
        }

        /// <summary>
        ///     A reversed view walks MEMORY order under 'K' while <c>c_index</c> still counts LOGICAL positions: values
        ///     <c>0 … 5</c> with indices <c>5 … 0</c>.
        /// </summary>
        [TestMethod]
        public void ReversedView_CIndex_CountsLogicalPositions()
        {
            using var a = np.arange(6);
            using var reversed = a["::-1"];
            using var it = np.nditer(reversed, new[] { "c_index" });
            it.Select(v => $"{it.index}:{v[0].GetAtIndex(0)}").Should().Equal("5:0", "4:1", "3:2", "2:3", "1:4", "0:5");
        }

        // ------------------------------------------------------------------ buffering and the inner loop

        /// <summary>
        ///     <c>grow_inner</c> cannot grow a CAST operand: its chunks stay buffer-sized (<c>3 3 3 1</c> for ten elements and
        ///     a buffer of 3), as without the flag. (The operand that needs NO buffer, which NumPy grows into one chunk, is
        ///     pinned in <c>OpenBugs.NditerGrowInner.cs</c>.)
        /// </summary>
        [TestMethod]
        public void GrowInner_CastOperand_KeepsBufferSizedChunks()
        {
            using var i32 = np.arange(10).astype(np.int32);
            using var it = np.nditer(i32, new[] { "buffered", "external_loop", "grow_inner" }, op_dtypes: new[] { np.float64 },
                buffersize: 3);
            it.Select(v => v[0].size).Should().Equal(3L, 3L, 3L, 1L);
        }

        /// <summary>
        ///     Without <c>grow_inner</c> every buffered walk is capped at the buffer size, even when no operand needs the
        ///     buffer: a contiguous <c>(100, 100)</c> array walks in chunks of <c>8192</c> and <c>1808</c> at NumPy's default
        ///     buffer size.
        /// </summary>
        [TestMethod]
        public void Buffered_WithoutGrowInner_CapsEveryChunkAtTheBufferSize()
        {
            using var a = np.zeros(new Shape(100, 100));
            using var it = np.nditer(a, new[] { "buffered", "external_loop" });
            it.Select(v => v[0].size).Should().Equal(8192L, 1808L);
        }

        /// <summary>
        ///     A buffered cast walks in buffer-sized chunks of the requested dtype — <c>[0..3] [4..7] [8, 9]</c> as float64 —
        ///     and a reversed operand is buffered in MEMORY order under 'K'.
        /// </summary>
        [TestMethod]
        public void BufferedCast_ExternalLoop_ChunksFollowTheBufferSize()
        {
            using var a = np.arange(10).astype(np.int32);
            using (var it = np.nditer(a, new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.float64 }, buffersize: 4))
            {
                var chunks = it.Select(v => v[0].ToArray<double>()).ToList();
                chunks.Should().HaveCount(3);
                chunks[0].Should().Equal(0.0, 1.0, 2.0, 3.0);
                chunks[1].Should().Equal(4.0, 5.0, 6.0, 7.0);
                chunks[2].Should().Equal(8.0, 9.0);
            }

            using var six = np.arange(6).astype(np.int32);
            using var reversed = six["::-1"];
            using (var it = np.nditer(reversed, new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.float64 },
                       buffersize: 4))
            {
                var chunks = it.Select(v => v[0].ToArray<double>()).ToList();
                chunks.Should().HaveCount(2);
                chunks[0].Should().Equal(0.0, 1.0, 2.0, 3.0);
                chunks[1].Should().Equal(4.0, 5.0);
            }
        }

        /// <summary>
        ///     A buffered writeonly cast flushes EVERY buffer, not only the last one: int16 values scaled by 1.6 in float64
        ///     with a buffer of 2 store <c>[0, 1, 3, 4, 6]</c> (unsafe truncation on the way back).
        /// </summary>
        [TestMethod]
        public void BufferedWriteOnlyCast_FlushesEveryBuffer()
        {
            using var a = np.arange(5).astype(np.int16);
            using var b = np.zeros(new Shape(5), np.int16);
            using (var it = np.nditer(new[] { a, b }, new[] { "buffered" }, new[] { new[] { "readonly" }, new[] { "writeonly" } },
                       new[] { np.float64, np.float64 }, casting: "unsafe", buffersize: 2))
            {
                foreach (var v in it)
                {
                    using var scaled = v[0] * 1.6;
                    np.copyto(v[1], scaled);
                }
            }

            b.ToArray<short>().Should().Equal(0, 1, 3, 4, 6);
        }

        /// <summary>A buffered int64 → complex128 cast reads <c>0+0j, 1+0j, 2+0j</c>.</summary>
        [TestMethod]
        public void BufferedCast_IntToComplex()
        {
            using var a = np.arange(3);
            using var it = np.nditer(a, new[] { "buffered" }, op_dtypes: new[] { np.complex128 });
            Walk(it).Should().Equal(new object[] { new Complex(0, 0), new Complex(1, 0), new Complex(2, 0) });
        }

        // ------------------------------------------------------------------ remove_axis / remove_multi_index / external loop

        /// <summary>
        ///     <c>remove_axis</c>'s refusals, NumPy's texts and types: no multi-index, an index tracked or a buffer
        ///     (RuntimeError), an axis outside <c>[0, ndim)</c> — negative ones included (ValueError).
        /// </summary>
        [TestMethod]
        public void RemoveAxis_Refusals_AreNumPys()
        {
            using var a = np.arange(6).reshape(2, 3);
            using (var it = np.nditer(a))
            {
                new Action(() => it.remove_axis(0)).Should().Throw<RuntimeError>()
                                                   .WithMessage("Iterator RemoveAxis may only be called if a multi-index is being tracked");
            }

            using (var it = np.nditer(a, new[] { "multi_index" }))
            {
                new Action(() => it.remove_axis(2)).Should().Throw<ValueError>().WithMessage("axis out of bounds in iterator RemoveAxis");
                new Action(() => it.remove_axis(-1)).Should().Throw<ValueError>().WithMessage("axis out of bounds in iterator RemoveAxis");
            }

            using (var it = np.nditer(a, new[] { "multi_index", "buffered" }))
            {
                new Action(() => it.remove_axis(0)).Should().Throw<RuntimeError>()
                                                   .WithMessage("Iterator RemoveAxis may not be called on a buffered iterator");
            }

            using (var it = np.nditer(a, new[] { "multi_index", "c_index" }))
            {
                new Action(() => it.remove_axis(0)).Should().Throw<RuntimeError>()
                                                   .WithMessage("Iterator RemoveAxis may not be called on an index is being tracked");
            }
        }

        /// <summary>
        ///     Removing the INNER axis of a <c>(3, 4)</c> walk leaves one axis of 3 at column 0: <c>(0,) 0</c>, <c>(1,) 4</c>,
        ///     <c>(2,) 8</c>.
        /// </summary>
        [TestMethod]
        public void RemoveAxis_InnerAxis_WalksTheOuterOneAtIndexZero()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var it = np.nditer(a, new[] { "multi_index" });
            it.remove_axis(1);
            it.ndim.Should().Be(1);
            it.shape.Should().Equal(3L);
            it.itersize.Should().Be(3);
            it.Select(v => $"{Mi(it)}:{v[0].GetAtIndex(0)}").Should().Equal("0:0", "1:4", "2:8");
        }

        /// <summary>
        ///     After <c>remove_multi_index</c> the multi-index is gone (NumPy's text), and <c>enable_external_loop</c> — refused
        ///     while it was tracked — then walks the coalesced array as one chunk.
        /// </summary>
        [TestMethod]
        public void RemoveMultiIndex_ThenEnableExternalLoop()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, new[] { "multi_index" });
            new Action(() => it.enable_external_loop()).Should().Throw<ValueError>()
                .WithMessage("Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");

            it.remove_multi_index();
            new Action(() => _ = it.multi_index).Should().Throw<ValueError>().WithMessage("Iterator is not tracking a multi-index");

            it.enable_external_loop();
            var chunks = it.Select(v => v[0].ToArray<long>()).ToList();
            chunks.Should().HaveCount(1);
            chunks[0].Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
        }

        // ------------------------------------------------------------------ every dtype

        /// <summary>
        ///     Every NumSharp dtype iterates as 0-d views of its own dtype holding the array's values in order — the boxed
        ///     view, not a converted one.
        /// </summary>
        /// <param name="code">The dtype under test.</param>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void EveryDtype_WalksItsOwnValues(NPTypeCode code)
        {
            using var a = np.arange(4).astype(code);
            using var it = np.nditer(a);
            it.dtypes[0].Should().Be(a.dtype);
            var views = it.Select(v => (Value: v[0].GetAtIndex(0), Code: v[0].typecode, Rank: v[0].ndim)).ToList();
            views.Select(t => t.Value).Should().Equal(Enumerable.Range(0, 4).Select(i => a.GetAtIndex(i)));
            views.Should().OnlyContain(t => t.Code == code && t.Rank == 0);
        }

        /// <summary>
        ///     Every dtype reaches float64 through the buffers under <c>'unsafe'</c>, reading what <c>astype(float64)</c>
        ///     gives (a complex value its real part, a bool 0 or 1).
        /// </summary>
        /// <param name="code">The dtype under test.</param>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void EveryDtype_BufferedCastToFloat64(NPTypeCode code)
        {
            using var a = np.arange(4).astype(code);
            using var expected = a.astype(np.float64);
            using var it = np.nditer(a, new[] { "buffered" }, op_dtypes: new[] { np.float64 }, casting: "unsafe");
            Walk(it).Should().Equal(expected.ToArray<double>().Cast<object>());
        }

        /// <summary>
        ///     Every dtype takes a Python int through the setter (NumPy's <c>PyArray_Pack</c>): writing 1 into each element
        ///     of a zero array gives what <c>np.ones</c> gives.
        /// </summary>
        /// <param name="code">The dtype under test.</param>
        [TestMethod]
        [DataRow(NPTypeCode.Boolean)]
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        [DataRow(NPTypeCode.Complex)]
        public void EveryDtype_SetterStoresAPythonInt(NPTypeCode code)
        {
            using var a = np.zeros(new Shape(3), code);
            using (var it = np.nditer(a, op_flags: new[] { "readwrite" }))
            {
                using var one = NDArray.Scalar(1);
                foreach (var _ in it)
                    it[0] = one;
            }

            using var ones = np.ones(new Shape(3), code);
            Enumerable.Range(0, 3).Select(i => a.GetAtIndex(i)).Should()
                      .Equal(Enumerable.Range(0, 3).Select(i => ones.GetAtIndex(i)));
        }
    }
}
