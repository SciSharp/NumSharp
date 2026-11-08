using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     <c>np.nditer</c>'s CONSTRUCTION contracts that the earlier suites (<see cref="np_nditer_Test"/>,
    ///     <see cref="np_nditer_NumPyParity_Test"/>) do not pin: NumPy's argument conversion, the global and per-operand flag
    ///     checks, the allocation and cast rules, broadcasting, <c>op_axes</c> / <c>itershape</c>, and the documented
    ///     reduction idiom. Every expected value and every error text was probed against NumPy 2.4.2 (2026-10-09).
    /// </summary>
    /// <remarks>
    ///     NumPy's exception types map to the house types of the same name (<see cref="ValueError"/>, <see cref="TypeError"/>,
    ///     <see cref="IndexError"/>, <see cref="RuntimeError"/>). The texts are verbatim, NumPy's typos included ("should be be
    ///     used", "buffering  without"), because a caller that matches on them ports unchanged.
    /// </remarks>
    [TestClass]
    public class np_nditer_Construction_Test
    {
        /// <summary>A one-dimensional int64 operand <c>[0, 1, 2]</c>, fresh for each use.</summary>
        /// <returns>The new array (the caller owns it).</returns>
        private static NDArray Three() => np.arange(3);

        /// <summary>Walks an iterator to its end and reads operand 0 at every step.</summary>
        /// <param name="it">The iterator to walk; it is finished afterwards.</param>
        /// <returns>The boxed scalar of each 0-d view, in iteration order.</returns>
        private static List<object> Walk(np.NDIterator it) => it.Select(v => v[0].GetAtIndex(0)).ToList();

        // ------------------------------------------------------------------ global flags

        /// <summary>
        ///     <c>c_index</c> and <c>f_index</c> together are refused by NumPy's <c>npyiter_check_global_flags</c> before any
        ///     operand is read: an iterator tracks at most one flat index.
        /// </summary>
        [TestMethod]
        public void BothIndexFlags_AreRefused()
        {
            var act = () => np.nditer(Three(), new[] { "c_index", "f_index" });
            act.Should().Throw<ValueError>().WithMessage("Iterator flags C_INDEX and F_INDEX cannot both be specified");
        }

        /// <summary>
        ///     <c>ranged</c> with <c>external_loop</c> needs <c>buffered</c>: without a buffer the inner loop of an
        ///     arbitrary range cannot be one fixed-size chunk.
        /// </summary>
        [TestMethod]
        public void RangedExternalLoop_WithoutBuffering_IsRefused()
        {
            var act = () => np.nditer(Three(), new[] { "ranged", "external_loop" });
            act.Should().Throw<ValueError>()
               .WithMessage("Iterator flag RANGED cannot be used with the flag EXTERNAL_LOOP unless BUFFERED is also enabled");
        }

        /// <summary>
        ///     With <c>buffered</c> the combination works: <c>iterrange = (1, 5)</c> over <c>arange(6)</c> with a buffer of 4
        ///     yields ONE chunk <c>[1, 2, 3, 4]</c> (NumPy's probed answer).
        /// </summary>
        [TestMethod]
        public void RangedExternalLoop_WithBuffering_WalksTheRangeInOneChunk()
        {
            using var a = np.arange(6);
            using var it = np.nditer(a, new[] { "ranged", "external_loop", "buffered" }, buffersize: 4);
            it.iterrange = (1, 5);
            var chunks = it.Select(v => v[0].ToArray<long>()).ToList();
            chunks.Should().HaveCount(1);
            chunks[0].Should().Equal(1L, 2L, 3L, 4L);
        }

        // ------------------------------------------------------------------ per-operand flags

        /// <summary>
        ///     Two read/write modes on one operand are refused with NumPy's single text, whichever pair is given
        ///     (<c>npyiter_check_per_op_flags</c>).
        /// </summary>
        [TestMethod]
        public void TwoReadWriteModes_AreRefused()
        {
            const string onlyOne =
                "Only one of the iterator flags READWRITE, READONLY, and WRITEONLY may be specified for an operand";
            var readonlyReadwrite = () => np.nditer(Three(), op_flags: new[] { "readonly", "readwrite" });
            readonlyReadwrite.Should().Throw<ValueError>().WithMessage(onlyOne);
            var readwriteWriteonly = () => np.nditer(Three(), op_flags: new[] { "readwrite", "writeonly" });
            readwriteWriteonly.Should().Throw<ValueError>().WithMessage(onlyOne);
        }

        /// <summary>
        ///     An <c>op_flags</c> list is used exactly as written — nothing is defaulted — so a list without a read/write mode
        ///     is refused, even when the other flags are valid.
        /// </summary>
        [TestMethod]
        public void NoReadWriteMode_IsRefused()
        {
            var act = () => np.nditer(Three(), op_flags: new[] { "copy" });
            act.Should().Throw<ValueError>()
               .WithMessage("None of the iterator flags READWRITE, READONLY, or WRITEONLY were specified for an operand");
        }

        /// <summary>
        ///     A written operand cannot use <c>copy</c>: a plain copy would never be written back, so NumPy demands
        ///     <c>updateifcopy</c>.
        /// </summary>
        [TestMethod]
        public void CopyOnAWrittenOperand_IsRefused()
        {
            var act = () => np.nditer(Three(), op_flags: new[] { "readwrite", "copy" });
            act.Should().Throw<ValueError>()
               .WithMessage("If an iterator operand is writeable, must use the flag UPDATEIFCOPY instead of COPY");
        }

        /// <summary>
        ///     The masked-write flags pair up: <c>writemasked</c> needs a write mode and may not sit with <c>arraymask</c> on
        ///     one operand, and each of the two needs the other somewhere in the iterator (NumPy's four texts, in its order).
        /// </summary>
        [TestMethod]
        public void WriteMaskedFlagRules_AreNumPys()
        {
            var readonlyMasked = () => np.nditer(Three(), op_flags: new[] { "readonly", "writemasked" });
            readonlyMasked.Should().Throw<ValueError>()
                          .WithMessage("The iterator flag WRITEMASKED may only be used with READWRITE or WRITEONLY");

            var maskedAndMask = () => np.nditer(Three(), op_flags: new[] { "readwrite", "writemasked", "arraymask" });
            maskedAndMask.Should().Throw<ValueError>()
                         .WithMessage("The iterator flag WRITEMASKED may not be used together with ARRAYMASK");

            var noMask = () => np.nditer(new[] { Three() }, op_flags: new[] { new[] { "readwrite", "writemasked" } });
            noMask.Should().Throw<ValueError>()
                  .WithMessage("An iterator operand was flagged as WRITEMASKED, but no ARRAYMASK operand was given to supply the mask");

            var noMasked = () => np.nditer(new[] { Three(), np.ones(new Shape(3), np.@bool) },
                op_flags: new[] { new[] { "readwrite" }, new[] { "readonly", "arraymask" } });
            noMasked.Should().Throw<ValueError>()
                    .WithMessage("An iterator operand was flagged as the ARRAYMASK, but no WRITEMASKED operands were given to use the mask");
        }

        /// <summary>One mask per iterator: a second <c>arraymask</c> operand is refused as it is reached.</summary>
        [TestMethod]
        public void TwoArrayMasks_AreRefused()
        {
            using var mask = np.ones(new Shape(3), np.@bool);
            var act = () => np.nditer(new[] { Three(), mask, mask },
                op_flags: new[] { new[] { "readwrite", "writemasked" }, new[] { "readonly", "arraymask" }, new[] { "readonly", "arraymask" } });
            act.Should().Throw<ValueError>().WithMessage("Only one iterator operand may receive an ARRAYMASK flag");
        }

        /// <summary>
        ///     A null operand (NumPy's <c>None</c>) is an output the iterator ALLOCATES, so it needs <c>allocate</c> (or
        ///     <c>virtual</c>), a write mode, and — under buffering without <c>delay_bufalloc</c> — must not be read
        ///     (NumPy's text has a doubled space before "without").
        /// </summary>
        [TestMethod]
        public void NullOperandFlagRules_AreNumPys()
        {
            var noAllocate = () => np.nditer(new NDArray[] { Three(), null },
                op_flags: new[] { new[] { "readonly" }, new[] { "readonly" } });
            noAllocate.Should().Throw<ValueError>()
                      .WithMessage("Iterator operand was NULL, but neither the ALLOCATE nor the VIRTUAL flag was specified");

            var notWritten = () => np.nditer(new NDArray[] { Three(), null },
                op_flags: new[] { new[] { "readonly" }, new[] { "readonly", "allocate" } });
            notWritten.Should().Throw<ValueError>()
                      .WithMessage("Automatic allocation was requested for an iterator operand, but it wasn't flagged for writing");

            var readableBuffered = () => np.nditer(new NDArray[] { Three(), null }, new[] { "buffered" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } });
            readableBuffered.Should().Throw<ValueError>()
                            .WithMessage("Automatic allocation was requested for an iterator operand, and it was flagged as " +
                                         "readable, but buffering  without delayed allocation was enabled");
        }

        /// <summary>
        ///     With <c>delay_bufalloc</c> a readable allocated output is allowed: the caller fills it, <c>reset()</c> allocates
        ///     the buffers, and the walk reads and writes it — NumPy's answer is <c>[0, 10, 20]</c> in the inferred int64.
        /// </summary>
        [TestMethod]
        public void ReadableAllocatedOutput_WithDelayedBuffering_IsFilledAfterReset()
        {
            using var a = Three();
            using var it = np.nditer(new NDArray[] { a, null }, new[] { "buffered", "delay_bufalloc" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } });
            using var output = it.operands[1];
            np.copyto(output, 0L);
            it.reset();
            foreach (var v in it)
            {
                using var tenfold = v[0] * 10;
                np.copyto(v[1], tenfold);
            }

            output.dtype.Should().Be(np.int64);
            output.ToArray<long>().Should().Equal(0L, 10L, 20L);
        }

        /// <summary>
        ///     <c>virtual</c> marks an operand with no array at all: a real array with the flag is refused, and the flag needs
        ///     <c>readwrite</c> (NumPy's message says "should be be used" — verbatim).
        /// </summary>
        [TestMethod]
        public void VirtualFlagRules_AreNumPys()
        {
            var notNull = () => np.nditer(new[] { Three() }, op_flags: new[] { new[] { "readwrite", "virtual" } });
            notNull.Should().Throw<ValueError>()
                   .WithMessage("Iterator operand flag VIRTUAL was specified, but the operand was not NULL");

            var notReadWrite = () => np.nditer(new NDArray[] { Three(), null },
                op_flags: new[] { new[] { "readonly" }, new[] { "readonly", "virtual" } });
            notReadWrite.Should().Throw<ValueError>()
                        .WithMessage("The iterator flag VIRTUAL should be be used together with READWRITE");
        }

        /// <summary>
        ///     A write mode on a read-only operand (a broadcast view) is refused at construction, before anything is
        ///     written — not when the first write lands.
        /// </summary>
        [TestMethod]
        public void WriteFlagOnAReadOnlyOperand_IsRefused()
        {
            using var row = np.broadcast_to(Three(), new Shape(2, 3));
            var act = () => np.nditer(row, op_flags: new[] { "readwrite" });
            act.Should().Throw<ValueError>().WithMessage("operand array with iterator write flag set is read-only");
        }

        // ------------------------------------------------------------------ the argument lists

        /// <summary>
        ///     A broadcast mismatch lists EVERY operand's shape, each followed by a space (so the text ends with one):
        ///     NumPy's <c>npyiter_fill_axisdata</c> message.
        /// </summary>
        [TestMethod]
        public void BroadcastMismatch_ListsEveryOperandShape()
        {
            var two = () => np.nditer(new[] { np.arange(3), np.arange(4) });
            two.Should().Throw<ValueError>().WithMessage("operands could not be broadcast together with shapes (3,) (4,) ");

            var three = () => np.nditer(new[] { np.zeros(new Shape(2, 3)), np.arange(3), np.zeros(new Shape(4, 1)) });
            three.Should().Throw<ValueError>()
                 .WithMessage("operands could not be broadcast together with shapes (2,3) (3,) (4,1) ");
        }

        /// <summary>
        ///     <c>op_flags</c> must hold one list per operand, or ONE list for all of them; two lists for three operands are
        ///     refused.
        /// </summary>
        [TestMethod]
        public void OpFlagsOfTheWrongCount_AreRefused()
        {
            var act = () => np.nditer(new[] { Three(), Three(), Three() },
                op_flags: new[] { new[] { "readonly" }, new[] { "readonly" } });
            act.Should().Throw<ValueError>().WithMessage("op_flags must be a tuple or array of per-op flag-tuples");
        }

        /// <summary>
        ///     <c>op_dtypes</c> of another length than the operand list is read by NumPy as ONE structured dtype, so the error
        ///     is the structured-dtype field error naming the first entry.
        /// </summary>
        [TestMethod]
        public void OpDtypesOfTheWrongCount_AreNumPysFieldError()
        {
            var act = () => np.nditer(new[] { Three(), Three() }, op_dtypes: new[] { np.float64 });
            act.Should().Throw<TypeError>().WithMessage("Field elements must be 2- or 3-tuples, got 'dtype('float64')'");
        }

        /// <summary>
        ///     <c>op_axes</c> needs one entry per operand, entries of one length, and at least one entry that is not null
        ///     (NumPy's three texts).
        /// </summary>
        [TestMethod]
        public void OpAxesErrors_AreNumPys()
        {
            var count = () => np.nditer(new[] { Three(), Three() }, op_axes: new[] { new[] { 0 } });
            count.Should().Throw<ValueError>().WithMessage("op_axes must be a tuple/list matching the number of ops");

            var sizes = () => np.nditer(new[] { Three(), Three() }, op_axes: new[] { new[] { 0 }, new[] { 0, -1 } });
            sizes.Should().Throw<ValueError>().WithMessage("Each entry of op_axes must have the same size");

            var allNull = () => np.nditer(new[] { Three(), Three() }, op_axes: new int[][] { null, null });
            allNull.Should().Throw<ValueError>()
                   .WithMessage("If op_axes is provided, at least one list of axes must be contained within it");
        }

        // ------------------------------------------------------------------ the shape pass: op_axes and itershape

        /// <summary>Asserts that constructing an iterator raises a ValueError with exactly NumPy's text.</summary>
        /// <param name="ops">The operands (temporaries of the call; the iterator is never built, so nothing else holds them).</param>
        /// <param name="opAxes">The <c>op_axes</c> argument, or null.</param>
        /// <param name="itershape">The <c>itershape</c> argument, or null.</param>
        /// <param name="message">NumPy 2.4.2's text, verbatim (no wildcard is used in these texts).</param>
        /// <param name="flags">The global flags, or null.</param>
        /// <param name="opFlags">The per-operand flags, or null.</param>
        private static void Refused(NDArray[] ops, int[][] opAxes, long[] itershape, string message,
                                    string[] flags = null, string[][] opFlags = null)
        {
            var act = () => np.nditer(ops, flags, opFlags, op_axes: opAxes, itershape: itershape);
            act.Should().Throw<ValueError>().WithMessage(message);
        }

        /// <summary>A zero-filled float64 array of the given shape.</summary>
        /// <param name="dims">The shape.</param>
        /// <returns>The new array.</returns>
        private static NDArray Z(params long[] dims) => np.zeros(new Shape(dims));

        /// <summary>
        ///     With <c>op_axes</c> a broadcast mismatch is NumPy's REMAPPED text: each mapped operand as
        ///     <c>original->remapped</c>, an unmapped one as its shape alone, a newaxis entry as NumPy's unknown dimension
        ///     (dropped when leading). It used to be the engine's IncorrectShapeException "shape mismatch: objects cannot be
        ///     broadcast to a single shape".
        /// </summary>
        [TestMethod]
        public void OpAxesBroadcastMismatch_ListsOriginalAndRemappedShapes()
        {
            const string head = "operands could not be broadcast together with remapped shapes [original->remapped]: ";
            Refused(new[] { Z(2, 3), Z(4, 3) }, new[] { new[] { 0, 1 }, new[] { 0, 1 } }, null, head + "(2,3)->(2,3) (4,3)->(4,3) ");
            Refused(new[] { Z(2, 3), Z(4, 3) }, new[] { null, new[] { 0, 1 } }, null, head + "(2,3) (4,3)->(4,3) ");
            Refused(new[] { Z(3), Z(4, 2) }, new[] { new[] { -1, 0 }, new[] { 0, 1 } }, null, head + "(3,)->(3) (4,2)->(4,2) ");
        }

        /// <summary>
        ///     With <c>itershape</c> the text names the requested shape after the operands — two spaces before it, the
        ///     separator's and the text's — with a negative entry printed as NumPy's unknown dimension. An itershape entry of 1
        ///     that an operand would stretch, and an entry of 0 against a longer operand, are the same refusal.
        /// </summary>
        [TestMethod]
        public void ItershapeMismatch_NamesTheRequestedShape()
        {
            const string head = "operands could not be broadcast together with remapped shapes [original->remapped]: ";
            Refused(new[] { Z(2, 3) }, null, new long[] { 4, 3 }, head + "(2,3)  and requested shape (4,3)");
            Refused(new[] { Z(2, 3) }, null, new long[] { 1, 3 }, head + "(2,3)  and requested shape (1,3)");
            Refused(new[] { Z(2, 3) }, null, new long[] { 0, 3 }, head + "(2,3)  and requested shape (0,3)");
            Refused(new[] { Z(2, 3) }, new[] { new[] { 0, 1 } }, new long[] { 1, 3 }, head + "(2,3)->(2,3)  and requested shape (1,3)");

            var pair = new[] { new[] { 0, 1 }, new[] { 0, 1 } };
            Refused(new[] { Z(2, 3), Z(4, 3) }, pair, new long[] { -1, 3 }, head + "(2,3)->(2,3) (4,3)->(4,3)  and requested shape (3)");
            Refused(new[] { Z(2, 3), Z(4, 3) }, pair, new long[] { 2, -1 },
                head + "(2,3)->(2,3) (4,3)->(4,3)  and requested shape (2,newaxis)");
        }

        /// <summary>
        ///     A NEGATIVE <c>itershape</c> entry is deduced from the operands (it was refused): <c>(-1, 3)</c> over a
        ///     <c>(2, 3)</c> array walks its six elements, <c>(-2, -1)</c> deduces both, and an allocated output takes the
        ///     deduced shape.
        /// </summary>
        [TestMethod]
        public void ItershapeNegativeEntries_AreDeducedFromTheOperands()
        {
            using var grid = np.arange(6).reshape(2, 3);
            using (var it = np.nditer(grid, new[] { "multi_index" }, itershape: new long[] { -1, 3 }))
            {
                it.shape.Should().Equal(2L, 3L);
                Walk(it).Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            }

            using (var it = np.nditer(grid, itershape: new long[] { -2, -1 }))
                it.itersize.Should().Be(6);

            using var row = Three();
            using var withOutput = np.nditer(new NDArray[] { row, null },
                op_flags: new[] { new[] { "readonly" }, new[] { "writeonly", "allocate" } }, itershape: new long[] { 2, -1 });
            using var output = withOutput.operands[1];
            output.shape.Should().Equal(2L, 3L);
            output.dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     An <c>itershape</c> without <c>op_axes</c> sets the iterator rank, and the operands are right-aligned into it: a
        ///     <c>(3,)</c> operand repeats along the leading axis of <c>itershape=(2, 3)</c>.
        /// </summary>
        [TestMethod]
        public void ItershapeWithoutOpAxes_RightAlignsTheOperands()
        {
            using var row = Three();
            using var it = np.nditer(row, new[] { "multi_index" }, itershape: new long[] { 2, 3 });
            var seen = new List<string>();
            foreach (var v in it)
                seen.Add($"{string.Join(",", it.multi_index)}:{v[0].GetAtIndex(0)}");
            seen.Should().Equal("0,0:0", "0,1:1", "0,2:2", "1,0:0", "1,1:1", "1,2:2");
        }

        /// <summary>
        ///     <c>op_axes</c> and <c>itershape</c> of different lengths are refused by NumPy's <c>npyiter_init</c> — before the
        ///     duplicate check, which would also fail here. They used to be accepted (the itershape's rank was iterated).
        /// </summary>
        [TestMethod]
        public void ItershapeAndOpAxesOfDifferentLengths_AreRefused()
        {
            const string message = "'op_axes' and 'itershape' must have the same number of entries equal to the iterator ndim";
            Refused(new[] { Z(2, 3) }, new[] { new[] { 0, 1 } }, new long[] { 2, 3, 4 }, message);
            Refused(new[] { Z(3, 3) }, new[] { new[] { 0, 0 } }, new long[] { 3, 3, 3 }, message);
        }

        /// <summary>
        ///     An <c>op_axes</c> entry naming an axis the operand lacks is NumPy's shape-pass ValueError (trailing space).
        ///     NumPy prints the entry's index REVERSED (<c>ndim - j - 1</c>): the second of two entries reads
        ///     <c>op_axes[0][0]</c>, the first of three <c>op_axes[0][2]</c>. The quirk is reproduced verbatim. It used to be
        ///     the engine's IncorrectShapeException "op_axes refers to non-existent axis".
        /// </summary>
        [TestMethod]
        public void OpAxesNamingAMissingAxis_IsNumPysValueError_WithTheReversedIndex()
        {
            Refused(new[] { Z(3) }, new[] { new[] { 1 } }, null,
                "Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which has 1 dimensions ");
            Refused(new[] { Z(2, 3) }, new[] { new[] { 0, 5 } }, null,
                "Iterator input op_axes[0][0] (==5) is not a valid axis of op[0], which has 2 dimensions ");
            Refused(new[] { Z(2, 3, 4) }, new[] { new[] { 7, 0, 1 } }, null,
                "Iterator input op_axes[0][2] (==7) is not a valid axis of op[0], which has 3 dimensions ");
            Refused(new[] { Z(2, 3, 4), Z(2, 3, 4) }, new[] { new[] { 0, 1, 2 }, new[] { 0, 9, 2 } }, null,
                "Iterator input op_axes[1][1] (==9) is not a valid axis of op[1], which has 3 dimensions ");
            Refused(new[] { Z(3) }, new[] { new[] { 63 } }, null,
                "Iterator input op_axes[0][0] (==63) is not a valid axis of op[0], which has 1 dimensions ");
        }

        /// <summary>
        ///     The shape pass walks each operand's entries in order, so an EARLIER entry's broadcast mismatch is reported
        ///     before a later entry's missing axis (<c>(3,)</c> mapped <c>[0, 2]</c> against <c>(2, 3)</c>: the remapped
        ///     text, its missing axis printed <c>newaxis</c>), and a missing axis in the first entry before a later mismatch.
        /// </summary>
        [TestMethod]
        public void ShapePass_ReportsTheFirstFailingEntryInOrder()
        {
            Refused(new[] { Z(2, 3), Z(3) }, new[] { new[] { 0, 1 }, new[] { 0, 2 } }, null,
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,3)->(2,3) (3,)->(3,newaxis) ");
            Refused(new[] { Z(2, 3), Z(4) }, new[] { new[] { 0, 1 }, new[] { 5, 0 } }, null,
                "Iterator input op_axes[1][1] (==5) is not a valid axis of op[1], which has 1 dimensions ");
        }

        /// <summary>
        ///     An operand named twice in one <c>op_axes</c> entry is refused by NumPy's <c>npyiter_check_op_axes</c> — an
        ///     explicit reduction entry counts as the axis it encodes, and the check runs BEFORE the global and per-operand
        ///     flag checks. Repeated newaxis entries are allowed. A duplicate used to be accepted and iterated a wrong grid.
        /// </summary>
        [TestMethod]
        public void OpAxesDuplicateAxis_IsRefusedBeforeTheFlagChecks()
        {
            const string op0 = "The 'op_axes' provided to the iterator constructor for operand 0 contained duplicate value 0";
            Refused(new[] { Z(3, 3) }, new[] { new[] { 0, 0 } }, null, op0);
            Refused(new[] { Z(1, 3), Z(2, 3) }, new[] { new[] { NumSharp.Backends.Iteration.NDIterUtils.ReductionAxis(0), 0 }, new[] { 0, 1 } },
                null, op0, flags: new[] { "reduce_ok" });
            Refused(new[] { Z(3), Z(3, 3) }, new[] { new[] { 0, -1 }, new[] { 1, 1 } }, null,
                "The 'op_axes' provided to the iterator constructor for operand 1 contained duplicate value 1");
            Refused(new[] { Z(3, 3) }, new[] { new[] { 0, 0 } }, null, op0, flags: new[] { "c_index", "f_index" });
            Refused(new[] { Z(3, 3) }, new[] { new[] { 0, 0 } }, null, op0, opFlags: new[] { new[] { "readonly", "readwrite" } });

            // Two newaxis entries pass the check; this map then fails the broadcast against the (2, 3, 4) operand.
            Refused(new[] { Z(3), Z(2, 3, 4) }, new[] { new[] { -1, -1, 0 }, new[] { 0, 1, 2 } }, null,
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3) (2,3,4)->(2,3,4) ");
        }

        /// <summary>
        ///     An <c>op_axes</c> value past NumPy's 64-dimension limit is NumPy's "contained invalid values" error — for an
        ///     axis no NumPy operand can have; an explicit reduction entry is decoded first. NumSharp has no dimension cap, so
        ///     an operand that does have the axis accepts it.
        /// </summary>
        [TestMethod]
        public void OpAxesValuePastNumPysDimensionLimit_IsRefused()
        {
            Refused(new[] { Z(3) }, new[] { new[] { 64 } }, null,
                "The 'op_axes' provided to the iterator constructor for operand 0 contained invalid values 64");
            Refused(new[] { Z(3) }, new[] { new[] { NumSharp.Backends.Iteration.NDIterUtils.ReductionAxis(70) } }, null,
                "The 'op_axes' provided to the iterator constructor for operand 0 contained invalid values 70");
        }

        /// <summary>
        ///     An UNMAPPED operand with more axes than the remapping (or than an <c>itershape</c>) is refused with NumPy's
        ///     text; a 0-d operand fits an empty <c>itershape</c> or an empty map and walks its one element.
        /// </summary>
        [TestMethod]
        public void UnmappedOperandWithMoreAxesThanTheRemapping_IsRefused()
        {
            const string message = "input operand has more dimensions than allowed by the axis remapping";
            Refused(new[] { Z(2, 3), Z(3) }, new[] { null, new[] { 0 } }, null, message);
            Refused(new[] { Z(3) }, null, new long[0], message);

            using var scalar = np.array(5.0);
            using (var it = np.nditer(scalar, itershape: new long[0]))
            {
                it.itersize.Should().Be(1);
                it.ndim.Should().Be(0);
            }

            using (var it = np.nditer(scalar, op_axes: new[] { new int[0] }))
            {
                it.itersize.Should().Be(1);
                it.ndim.Should().Be(0);
            }
        }

        // ------------------------------------------------------------------ casting

        /// <summary>
        ///     A requested dtype the operand cannot reach under the rule is refused at construction, naming the operand, both
        ///     dtypes and the rule: float64 → int32 is not <c>'safe'</c>.
        /// </summary>
        [TestMethod]
        public void RefusedReadCast_NamesTheOperandAndTheRule()
        {
            using var a = np.arange(3).astype(np.float64);
            var act = () => np.nditer(a, new[] { "buffered" }, op_dtypes: new[] { np.int32 });
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator operand 0 dtype could not be cast from dtype('float64') to dtype('int32') according to the rule 'safe'");
        }

        /// <summary>
        ///     A WRITTEN operand needs the cast back too: int32 → float64 reads safely, but float64 → int32 is not
        ///     <c>'safe'</c>, so NumPy refuses with its write-back text (which names the REQUESTED dtype first).
        /// </summary>
        [TestMethod]
        public void RefusedWriteBackCast_NamesTheRequestedDtype()
        {
            using var a = np.arange(3).astype(np.int32);
            var act = () => np.nditer(a, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.float64 });
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator requested dtype could not be cast from dtype('float64') to dtype('int32'), the operand 0 " +
                            "dtype, according to the rule 'safe'");
        }

        /// <summary>
        ///     <c>'no'</c> and <c>'equiv'</c> refuse even a widening int32 → int64 cast (only identical dtypes, or byte-order
        ///     changes, pass them); the text names the rule given.
        /// </summary>
        [TestMethod]
        public void NoAndEquivRules_RefuseAWideningCast()
        {
            using var a = np.arange(3).astype(np.int32);
            foreach (var rule in new[] { "no", "equiv" })
            {
                var act = () => np.nditer(a, new[] { "buffered" }, op_dtypes: new[] { np.int64 }, casting: rule);
                act.Should().Throw<TypeError>()
                   .WithMessage($"Iterator operand 0 dtype could not be cast from dtype('int32') to dtype('int64') according to the rule '{rule}'");
            }
        }

        /// <summary>
        ///     <c>common_dtype</c> with buffering iterates every operand in their result type: int32 and float32 both become
        ///     float64, and the walk reads them as doubles.
        /// </summary>
        [TestMethod]
        public void CommonDtype_Buffered_PromotesEveryOperand()
        {
            using var i = np.arange(3).astype(np.int32);
            using var f = np.arange(3).astype(np.float32);
            using var it = np.nditer(new[] { i, f }, new[] { "common_dtype", "buffered" });
            it.dtypes.Select(d => d.str).Should().Equal("<f8", "<f8");
            it.Select(v => (v[0].GetAtIndex(0), v[1].GetAtIndex(0))).Should()
              .Equal(((object)0.0, (object)0.0), (1.0, 1.0), (2.0, 2.0));
        }

        /// <summary>
        ///     Without buffering, <c>common_dtype</c>'s casts have no buffer to happen in: NumPy's "required copying or
        ///     buffering" TypeError, not a silent iteration in the operands' own dtypes.
        /// </summary>
        [TestMethod]
        public void CommonDtype_Unbuffered_NeedsACopyOrABuffer()
        {
            using var i = np.arange(3).astype(np.int32);
            using var f = np.arange(3).astype(np.float32);
            var act = () => np.nditer(new[] { i, f }, new[] { "common_dtype" });
            act.Should().Throw<TypeError>()
               .WithMessage("Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
        }

        /// <summary>
        ///     A read-only 0-d operand that needs a cast is copied without being asked (NumPy's
        ///     <c>npyiter_allocate_arrays</c>), so <c>np.nditer(np.array(7), op_dtypes=['float64'])</c> iterates <c>7.0</c>
        ///     with no <c>buffered</c> and no <c>copy</c> flag.
        /// </summary>
        [TestMethod]
        public void ZeroDOperand_IsCopiedForACastWithoutBuffering()
        {
            using var seven = np.array(7);
            using var it = np.nditer(seven, op_dtypes: new[] { np.float64 });
            it.dtypes[0].Should().Be(np.float64);
            Walk(it).Should().Equal(7.0);
        }

        /// <summary>
        ///     A buffered readwrite cast writes back through the rule given: under <c>'unsafe'</c> an int32 operand scaled by
        ///     2.5 in float64 stores the truncated <c>[0, 2, 5, 7]</c>.
        /// </summary>
        [TestMethod]
        public void BufferedReadWriteCast_WritesBackThroughTheCastingRule()
        {
            using var a = np.arange(4).astype(np.int32);
            using (var it = np.nditer(a, new[] { "buffered" }, new[] { "readwrite" }, new[] { np.float64 }, casting: "unsafe"))
            {
                foreach (var v in it)
                {
                    using var scaled = v[0] * 2.5;
                    np.copyto(v[0], scaled);
                }
            }

            a.ToArray<int>().Should().Equal(0, 2, 5, 7);
        }

        // ------------------------------------------------------------------ allocation

        /// <summary>
        ///     An allocated output takes its requested dtype: <c>op_dtypes=[None, 'float32']</c> allocates float32, and the
        ///     walk stores <c>x * 1.5</c> into it as <c>[0.0, 1.5, 3.0]</c>.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_TakesTheRequestedDtype()
        {
            using var a = Three();
            using var it = np.nditer(new NDArray[] { a, null }, op_dtypes: new DType[] { null, np.float32 });
            foreach (var v in it)
            {
                using var scaled = v[0] * 1.5;
                np.copyto(v[1], scaled);
            }

            using var output = it.operands[1];
            output.dtype.Should().Be(np.float32);
            output.ToArray<float>().Should().Equal(0f, 1.5f, 3f);
        }

        /// <summary>
        ///     Without a request, an allocated output takes the inputs' result type — an int32 input allocates int32, not the
        ///     default int64 — and <c>dtypes</c> reports it for both operands.
        /// </summary>
        [TestMethod]
        public void AllocatedOutput_InheritsTheInputDtype()
        {
            using var a = np.arange(3).astype(np.int32);
            using var it = np.nditer(new NDArray[] { a, null });
            using var output = it.operands[1];
            output.dtype.Should().Be(np.int32);
            it.dtypes.Select(d => d.str).Should().Equal("<i4", "<i4");
        }

        /// <summary>Asserts that a two-operand iteration with a <c>no_broadcast</c> first operand is refused with a text.</summary>
        /// <param name="first">The <c>no_broadcast</c> operand.</param>
        /// <param name="firstMode">Its read/write mode (<c>readonly</c> / <c>readwrite</c> / <c>writeonly</c>).</param>
        /// <param name="second">The read-only operand that sets the broadcast shape.</param>
        /// <param name="opAxes">The axis maps, or null for natural broadcasting.</param>
        /// <param name="message">NumPy's exact ValueError text.</param>
        private static void NoBroadcastRefused(NDArray first, string firstMode, NDArray second, int[][] opAxes, string message)
        {
            var act = () => np.nditer(new[] { first, second },
                op_flags: new[] { new[] { firstMode, "no_broadcast" }, new[] { "readonly" } }, op_axes: opAxes);
            act.Should().Throw<ValueError>().WithMessage(message);
        }

        /// <summary>
        ///     <c>no_broadcast</c> refuses a read operand the iteration would stretch, with NumPy's ValueError: an operand
        ///     axis of 1 under a longer one, and a MISSING axis — even where the iteration's extent is 1 (<c>(3,)</c> next to
        ///     <c>(1, 3)</c>) and for a 0-d operand (printed <c>()</c>).
        /// </summary>
        [TestMethod]
        public void NoBroadcastFlag_RefusesAStretchedReadOperand()
        {
            NoBroadcastRefused(Three(), "readonly", np.zeros(new Shape(2, 3)), null,
                "non-broadcastable operand with shape (3,) doesn't match the broadcast shape (2,3)");
            NoBroadcastRefused(np.zeros(new Shape(1, 3)), "readonly", np.zeros(new Shape(2, 3)), null,
                "non-broadcastable operand with shape (1,3) doesn't match the broadcast shape (2,3)");
            NoBroadcastRefused(np.array(5.0), "readonly", np.zeros(new Shape(3)), null,
                "non-broadcastable operand with shape () doesn't match the broadcast shape (3,)");
            NoBroadcastRefused(np.zeros(new Shape(3)), "readonly", np.zeros(new Shape(1, 3)), null,
                "non-broadcastable operand with shape (3,) doesn't match the broadcast shape (1,3)");
        }

        /// <summary>A WRITTEN <c>no_broadcast</c> operand is named an "output operand" in NumPy's text.</summary>
        [TestMethod]
        public void NoBroadcastFlag_OnAWrittenOperand_SaysOutputOperand()
        {
            foreach (var mode in new[] { "readwrite", "writeonly" })
            {
                NoBroadcastRefused(np.zeros(new Shape(1, 3)), mode, np.zeros(new Shape(2, 3)), null,
                    "non-broadcastable output operand with shape (1,3) doesn't match the broadcast shape (2,3)");
            }
        }

        /// <summary>
        ///     With <c>op_axes</c> NumPy also prints the remapped shape — with the iteration axes REVERSED, a NumPy quirk kept
        ///     verbatim (a <c>(1, 3)</c> operand mapped <c>[0, 1]</c> reads "[remapped to (3,1)]"; a <c>-1</c> entry prints as
        ///     <c>newaxis</c>). A <c>-1</c> entry itself is never refused: <c>(3,)</c> mapped <c>[-1, 0]</c> constructs.
        /// </summary>
        [TestMethod]
        public void NoBroadcastFlag_WithOpAxes_PrintsTheRemappedShape()
        {
            var both = new[] { new[] { 0, 1 }, new[] { 0, 1 } };
            NoBroadcastRefused(np.zeros(new Shape(1, 3)), "readonly", np.zeros(new Shape(2, 3)), both,
                "non-broadcastable operand with shape (1,3) [remapped to (3,1)] doesn't match the broadcast shape (2,3)");
            NoBroadcastRefused(np.zeros(new Shape(1, 3)), "readwrite", np.zeros(new Shape(2, 3)), both,
                "non-broadcastable output operand with shape (1,3) [remapped to (3,1)] doesn't match the broadcast shape (2,3)");

            var newAxisFirst = new[] { new[] { -1, 0 }, new[] { 0, 1 } };
            NoBroadcastRefused(np.zeros(new Shape(1)), "readonly", np.zeros(new Shape(2, 3)), newAxisFirst,
                "non-broadcastable operand with shape (1,) [remapped to (1,newaxis)] doesn't match the broadcast shape (2,3)");

            using var row = np.zeros(new Shape(3));
            using var grid = np.zeros(new Shape(2, 3));
            using var it = np.nditer(new[] { row, grid },
                op_flags: new[] { new[] { "readonly", "no_broadcast" }, new[] { "readonly" } }, op_axes: newAxisFirst);
            it.itersize.Should().Be(6);
        }

        /// <summary>
        ///     With two stretched <c>no_broadcast</c> operands NumPy reports the one stretched along the INNER axis first, whatever
        ///     its position: next to <c>(2, 3)</c>, the <c>(2, 1)</c> operand is named, not the earlier <c>(1, 3)</c> one.
        /// </summary>
        [TestMethod]
        public void NoBroadcastFlag_ReportsTheInnerAxisStretchFirst()
        {
            var act = () => np.nditer(new[] { np.zeros(new Shape(1, 3)), np.zeros(new Shape(2, 1)), np.zeros(new Shape(2, 3)) },
                op_flags: new[] { new[] { "readonly", "no_broadcast" }, new[] { "readonly", "no_broadcast" }, new[] { "readonly" } });
            act.Should().Throw<ValueError>()
               .WithMessage("non-broadcastable operand with shape (2,1) doesn't match the broadcast shape (2,3)");
        }

        /// <summary>
        ///     NumPy checks <c>no_broadcast</c> and the reduction rule in ONE loop — iteration axes innermost first, operands in
        ///     order within an axis — so a written operand stretched along the inner axis is reported before a
        ///     <c>no_broadcast</c> operand stretched along an outer one, and on one axis the earlier operand wins. A mapped
        ///     operand names its operand axis, an unmapped one the axis counted from the innermost. NumSharp used to check
        ///     <c>no_broadcast</c> first and the reduction operand by operand (two written operands named the first).
        /// </summary>
        [TestMethod]
        public void StridePass_RefusalsFollowNumPysAxisThenOperandOrder()
        {
            const string reduction = "output operand requires a reduction along dimension {0}, but the reduction is not enabled. " +
                                     "The dimension size of 1 does not match the expected output shape.";
            string reductionAlong0 = string.Format(reduction, 0);
            const string stretched21 = "non-broadcastable operand with shape (2,1) doesn't match the broadcast shape (2,3)";
            var nb = new[] { "readonly", "no_broadcast" };
            var rw = new[] { "readwrite" };
            var ro = new[] { "readonly" };

            Refused(new[] { Z(1, 3), Z(2, 1), Z(2, 3) }, null, null, reductionAlong0, opFlags: new[] { nb, rw, ro });
            Refused(new[] { Z(2, 1), Z(1, 3), Z(2, 3) }, null, null, stretched21, opFlags: new[] { nb, rw, ro });
            Refused(new[] { Z(2, 1), Z(2, 1), Z(2, 3) }, null, null, reductionAlong0, opFlags: new[] { rw, nb, ro });
            Refused(new[] { Z(2, 1), Z(2, 1), Z(2, 3) }, null, null, stretched21, opFlags: new[] { nb, rw, ro });
            Refused(new[] { Z(3), Z(2, 1), Z(2, 3) }, null, null, reductionAlong0, opFlags: new[] { nb, rw, ro });
            Refused(new[] { Z(1, 3), Z(2, 1), Z(2, 3) }, new[] { new[] { 0, 1 }, new[] { 0, 1 }, new[] { 0, 1 } }, null,
                string.Format(reduction, 1), opFlags: new[] { nb, rw, ro });
            Refused(new[] { Z(1, 3), Z(2, 1), Z(2, 3) }, null, null, reductionAlong0, opFlags: new[] { rw, rw, ro });
        }

        /// <summary>
        ///     A ZERO-length iteration axis stretches an operand of extent 1 too: a written <c>(1, 3)</c> operand next to
        ///     <c>(0, 3)</c> is a refused reduction (NumSharp used to accept it, its check testing an extent &gt; 1), an empty
        ///     walk with <c>reduce_ok</c>, and a <c>no_broadcast</c> operand there is refused.
        /// </summary>
        [TestMethod]
        public void ZeroLengthIterationAxis_StretchesAnOperandOfExtentOne()
        {
            var sizeOk = new[] { "zerosize_ok" };
            Refused(new[] { Z(1, 3), Z(0, 3) }, null, null,
                "output operand requires a reduction along dimension 1, but the reduction is not enabled. " +
                "The dimension size of 1 does not match the expected output shape.",
                sizeOk, new[] { new[] { "readwrite" }, new[] { "readonly" } });
            Refused(new[] { Z(1, 3), Z(0, 3) }, null, null,
                "non-broadcastable operand with shape (1,3) doesn't match the broadcast shape (0,3)",
                sizeOk, new[] { new[] { "readonly", "no_broadcast" }, new[] { "readonly" } });

            using var output = Z(1, 3);
            using var empty = Z(0, 3);
            using var it = np.nditer(new[] { output, empty }, new[] { "zerosize_ok", "reduce_ok" },
                new[] { new[] { "readwrite" }, new[] { "readonly" } });
            it.itersize.Should().Be(0);
            it.finished.Should().BeTrue();
        }

        /// <summary>
        ///     An explicit reduction axis of an extent other than 1 is a STRIDE-pass refusal: a broadcast mismatch is
        ///     reported before it, and within the stride pass the inner axis decides — a <c>no_broadcast</c> operand stretched
        ///     along the inner axis before an explicit reduction along the outer one, and the reverse. The length used to be
        ///     checked with the shapes, so the reduction was reported first.
        /// </summary>
        [TestMethod]
        public void ExplicitReductionLength_IsAStridePassRefusal()
        {
            static int R(int axis) => NumSharp.Backends.Iteration.NDIterUtils.ReductionAxis(axis);
            const string lengthOf3 = "operand was set up as a reduction along axis 1, but the length of the axis is 3 (it has to be 1)";
            var reduceOk = new[] { "reduce_ok" };
            var nb = new[] { "readonly", "no_broadcast" };
            var rw = new[] { "readwrite" };
            var ro = new[] { "readonly" };

            Refused(new[] { Z(2, 3), Z(2, 3) }, new[] { new[] { 0, 1 }, new[] { 0, R(1) } }, null, lengthOf3, reduceOk, new[] { ro, rw });
            Refused(new[] { Z(2, 3), Z(4, 3) }, new[] { new[] { 0, 1 }, new[] { R(0), 1 } }, null,
                "operands could not be broadcast together with remapped shapes [original->remapped]: (2,3)->(2,3) (4,3)->(4,3) ",
                reduceOk, new[] { ro, rw });
            Refused(new[] { Z(2, 1), Z(2, 3) }, new[] { new[] { 0, 1 }, new[] { R(0), 1 } }, null,
                "non-broadcastable operand with shape (2,1) [remapped to (1,2)] doesn't match the broadcast shape (2,3)",
                reduceOk, new[] { nb, rw });
            Refused(new[] { Z(1, 3), Z(2, 3) }, new[] { new[] { 0, 1 }, new[] { 0, R(1) } }, null, lengthOf3, reduceOk, new[] { nb, rw });
        }

        /// <summary>
        ///     A written operand the iteration stretches is a reduction, refused in NumPy's order of three checks:
        ///     <c>reduce_ok</c> first (a write-only operand and the ARRAYMASK without it get the "dimension" text), then a
        ///     write-only operand, then the ARRAYMASK. A newaxis map entry stretches an operand still to be allocated, named
        ///     "dimension -1".
        /// </summary>
        [TestMethod]
        public void ReductionRefusals_AreNumPysThreeChecksInOrder()
        {
            string Along(int dim) => $"output operand requires a reduction along dimension {dim}, but the reduction is not " +
                                     "enabled. The dimension size of 1 does not match the expected output shape.";
            const string writeOnly = "output operand requires a reduction, but is flagged as write-only, not read-write";
            const string arrayMask = "output operand requires a reduction, but is flagged as the ARRAYMASK operand which is " +
                                     "not permitted to be the result of a reduction";
            var ro = new[] { "readonly" };
            var wo = new[] { "writeonly" };
            var reduceOk = new[] { "reduce_ok" };
            var maskFlags = new[] { ro, new[] { "readwrite", "arraymask" }, new[] { "readwrite", "writemasked" } };

            Refused(new[] { Z(1, 3), Z(2, 3) }, null, null, Along(1), opFlags: new[] { wo, ro });
            Refused(new[] { Z(1, 3), Z(2, 3) }, null, null, writeOnly, reduceOk, new[] { wo, ro });
            Refused(new[] { Z(2, 3), np.zeros(new Shape(1, 3), np.@bool), Z(2, 3) }, null, null, Along(1), opFlags: maskFlags);
            Refused(new[] { Z(2, 3), np.zeros(new Shape(1, 3), np.@bool), Z(2, 3) }, null, null, arrayMask, reduceOk, maskFlags);

            // The mask stretched along the INNER axis is reported before an earlier no_broadcast operand stretched along the
            // outer one: the ordered pass itself must refuse the mask (the setup's own per-operand check would come too late).
            Refused(new[] { Z(1, 3), np.zeros(new Shape(2, 1), np.@bool), Z(2, 3) }, null, null, arrayMask, reduceOk,
                new[] { new[] { "readonly", "no_broadcast" }, new[] { "readwrite", "arraymask" }, new[] { "readwrite", "writemasked" } });

            var newAxis = new[] { new[] { 0, 1 }, new[] { 0, -1 } };
            Refused(new[] { Z(2, 3), null }, newAxis, null, Along(-1), opFlags: new[] { ro, new[] { "readwrite", "allocate" } });
            Refused(new[] { Z(2, 3), null }, newAxis, null, writeOnly, reduceOk, new[] { ro, new[] { "writeonly", "allocate" } });
        }

        /// <summary>
        ///     The explicit reduction length is refused on a READ-ONLY operand too, with or without <c>reduce_ok</c> (the
        ///     reduction rule's checks apply to written operands only; the length check does not), and a read-only operand
        ///     reduced along an axis of length 1 needs no <c>reduce_ok</c>.
        /// </summary>
        [TestMethod]
        public void ExplicitReductionLength_IsRefusedOnAReadOnlyOperandToo()
        {
            var map = new[] { new[] { 0, 1 }, new[] { 0, NumSharp.Backends.Iteration.NDIterUtils.ReductionAxis(1) } };
            var bothRead = new[] { new[] { "readonly" }, new[] { "readonly" } };
            const string lengthOf3 = "operand was set up as a reduction along axis 1, but the length of the axis is 3 (it has to be 1)";
            Refused(new[] { Z(2, 3), Z(2, 3) }, map, null, lengthOf3, new[] { "reduce_ok" }, bothRead);
            Refused(new[] { Z(2, 3), Z(2, 3) }, map, null, lengthOf3, opFlags: bothRead);

            using var grid = Z(2, 3);
            using var column = Z(2, 1);
            using var it = np.nditer(new[] { grid, column }, op_flags: bothRead, op_axes: map);
            it.itersize.Should().Be(6);
        }

        /// <summary>
        ///     The broadcast shape a <c>no_broadcast</c> refusal names is the DEDUCED one when <c>itershape</c> has a negative
        ///     entry: a <c>(1, 3)</c> operand against <c>itershape=(-1, 3)</c> and a <c>(2, 3)</c> operand names <c>(2,3)</c>.
        /// </summary>
        [TestMethod]
        public void NoBroadcastFlag_WithADeducedItershape_NamesTheDeducedShape()
        {
            Refused(new[] { Z(1, 3), Z(2, 3) }, null, new long[] { -1, 3 },
                "non-broadcastable operand with shape (1,3) doesn't match the broadcast shape (2,3)",
                opFlags: new[] { new[] { "readonly", "no_broadcast" }, new[] { "readonly" } });
        }

        /// <summary>
        ///     <c>op_axes</c> with <c>-1</c> (newaxis) entries builds NumPy's documented outer product: <c>[0, 1, 2]</c> and
        ///     <c>[1, 2]</c> into an allocated <c>(3, 2)</c> output <c>[[0, 0], [1, 2], [2, 4]]</c>.
        /// </summary>
        [TestMethod]
        public void OpAxes_NewAxisEntries_BuildAnOuterProduct()
        {
            using var a = Three();
            using var b = np.arange(1, 3);
            using var it = np.nditer(new NDArray[] { a, b, null }, op_axes: new[] { new[] { 0, -1 }, new[] { -1, 0 }, null });
            foreach (var v in it)
            {
                using var product = v[0] * v[1];
                np.copyto(v[2], product);
            }

            using var output = it.operands[2];
            output.shape.Should().Equal(3L, 2L);
            output.ToArray<long>().Should().Equal(0L, 0L, 1L, 2L, 2L, 4L);
        }

        /// <summary>
        ///     <c>itershape</c> sizes an iteration that has no input at all: one allocated int64 operand with
        ///     <c>op_axes=[[0, 1]]</c> and <c>itershape=(2, 3)</c> is a <c>(2, 3)</c> array the walk fills by its
        ///     multi-index (<c>10 * i + j</c>).
        /// </summary>
        [TestMethod]
        public void Itershape_SizesAnOutputWithNoInput()
        {
            using var it = np.nditer(new NDArray[] { null }, new[] { "multi_index" }, new[] { new[] { "writeonly", "allocate" } },
                new[] { np.int64 }, op_axes: new[] { new[] { 0, 1 } }, itershape: new long[] { 2, 3 });
            foreach (var _ in it)
            {
                var mi = it.multi_index;
                using var value = NDArray.Scalar(10 * mi[0] + mi[1]);
                it[0] = value;
            }

            using var output = it.operands[0];
            output.shape.Should().Equal(2L, 3L);
            output.ToArray<long>().Should().Equal(0L, 1L, 2L, 10L, 11L, 12L);
        }

        // ------------------------------------------------------------------ the reduction idiom

        /// <summary>
        ///     NumPy's documented reduction: <c>reduce_ok</c>, an allocated readwrite output with <c>op_axes=[None, [0, -1]]</c>
        ///     zeroed first, then <c>y += x</c> per element — the row sums of <c>arange(6).reshape(2, 3)</c>,
        ///     <c>[3, 12]</c>.
        /// </summary>
        [TestMethod]
        public void ReduceOk_PerElementWalk_SumsTheLastAxis()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(new NDArray[] { a, null }, new[] { "reduce_ok" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, op_axes: new[] { null, new[] { 0, -1 } });
            using var output = it.operands[1];
            np.copyto(output, 0L);
            foreach (var v in it)
            {
                using var sum = v[1] + v[0];
                np.copyto(v[1], sum);
            }

            output.ToArray<long>().Should().Equal(3L, 12L);
        }

        /// <summary>
        ///     The same idiom with <c>external_loop</c>, reducing the FIRST axis (<c>op_axes=[None, [-1, 0]]</c>): each chunk is
        ///     a row added into the output, so <c>arange(12).reshape(3, 4)</c> sums to <c>[12, 15, 18, 21]</c>.
        /// </summary>
        [TestMethod]
        public void ReduceOk_ExternalLoop_SumsTheFirstAxis()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var it = np.nditer(new NDArray[] { a, null }, new[] { "reduce_ok", "external_loop" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, op_axes: new[] { null, new[] { -1, 0 } });
            using var output = it.operands[1];
            np.copyto(output, 0L);
            foreach (var v in it)
            {
                using var sum = v[1] + v[0];
                np.copyto(v[1], sum);
            }

            output.ToArray<long>().Should().Equal(12L, 15L, 18L, 21L);
        }

        // ------------------------------------------------------------------ order and overlap

        /// <summary>
        ///     The order letter goes through NumPy's <c>PyArray_OrderConverter</c>, which takes either case: <c>'f'</c> walks
        ///     a C <c>(2, 3)</c> array in F order, <c>0 3 1 4 2 5</c>.
        /// </summary>
        [TestMethod]
        public void LowercaseOrder_IsNumPysOrderConverter()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var it = np.nditer(a, order: 'f');
            Walk(it).Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
        }

        /// <summary>
        ///     <c>copy_if_overlap</c> breaks the overlap between a read view and a written view of one buffer: shifting
        ///     <c>a[:-1] + 100</c> into <c>a[1:]</c> reads the ORIGINAL values, <c>[0, 100, 101, 102, 103, 104]</c> after
        ///     close — not a cascade of already-written elements.
        /// </summary>
        [TestMethod]
        public void CopyIfOverlap_ReadsTheValuesBeforeAnyWrite()
        {
            using var a = np.arange(6);
            using (var it = np.nditer(new[] { a[":-1"], a["1:"] }, new[] { "copy_if_overlap" },
                       new[] { new[] { "readonly" }, new[] { "writeonly" } }))
            {
                foreach (var v in it)
                {
                    using var shifted = v[0] + 100;
                    np.copyto(v[1], shifted);
                }
            }

            a.ToArray<long>().Should().Equal(0L, 100L, 101L, 102L, 103L, 104L);
        }
    }
}
