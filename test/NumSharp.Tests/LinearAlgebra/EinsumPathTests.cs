using System;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     <c>np.einsum_path</c> — the contraction planner, against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     A route-for-route port of NumPy's <c>einsumfunc.py</c> planner. Every path and every
    ///     numeric metric here was produced by running the same expression through NumPy 2.4.2; the
    ///     full <see cref="np.einsum_path(string, NDArray[])"/> surface was additionally
    ///     differential-fuzzed 1,525 random contraction networks bit-exact (path + printed string).
    ///     The info string is built with <c>'\n'</c> newlines, matching the implementation.
    /// </remarks>
    [TestClass]
    public class EinsumPathTests
    {
        private static NDArray Z(params int[] shape) => np.zeros(new Shape(shape));

        private static string Lines(params string[] lines) => string.Join("\n", lines);

        // ------------------------------------------------------------------ full byte-exact strings

        [TestMethod]
        public void Greedy_Chain_ByteExact()
        {
            var (path, repr) = np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, "greedy");

            path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
            repr.Should().Be(Lines(
                "  Complete contraction:  ij,jk,kl->il",
                "         Naive scaling:  4",
                "     Optimized scaling:  3",
                "      Naive FLOP count:  1.200e+02",
                "  Optimized FLOP count:  5.700e+01",
                "   Theoretical speedup:  2.105",
                "  Largest intermediate:  4.000e+00 elements",
                "--------------------------------------------------------------------------",
                "scaling                  current                                remaining",
                "--------------------------------------------------------------------------",
                "   3                   kl,jk->jl                                ij,jl->il",
                "   3                   jl,ij->il                                   il->il"));
        }

        [TestMethod]
        public void Greedy_FiveOperand_ByteExact()
        {
            var (path, repr) = np.einsum_path(
                "ea,fb,abcd,gc,hd->efgh",
                new[] { Z(10, 10), Z(10, 10), Z(10, 10, 10, 10), Z(10, 10), Z(10, 10) }, "greedy");

            path.ToString().Should().Be("['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]");
            repr.Should().Be(Lines(
                "  Complete contraction:  ea,fb,abcd,gc,hd->efgh",
                "         Naive scaling:  8",
                "     Optimized scaling:  5",
                "      Naive FLOP count:  5.000e+08",
                "  Optimized FLOP count:  8.000e+05",
                "   Theoretical speedup:  624.999",
                "  Largest intermediate:  1.000e+04 elements",
                "--------------------------------------------------------------------------",
                "scaling                  current                                remaining",
                "--------------------------------------------------------------------------",
                "   5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh",
                "   5               bcde,fb->cdef                         gc,hd,cdef->efgh",
                "   5               cdef,gc->defg                            hd,defg->efgh",
                "   5               defg,hd->efgh                               efgh->efgh"));
        }

        // ------------------------------------------------------------------ path per optimize mode

        [TestMethod]
        public void Optimal_MatchesGreedy_OnChain()
        {
            var greedy = np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, "greedy").path;
            var optimal = np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, "optimal").path;
            greedy.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
            optimal.Equals(greedy).Should().BeTrue();
        }

        [TestMethod]
        public void False_And_None_AreSingleContraction()
        {
            np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, false)
                .path.ToString().Should().Be("['einsum_path', (0, 1, 2)]");
            np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, (object)null)
                .path.ToString().Should().Be("['einsum_path', (0, 1, 2)]");
        }

        [TestMethod]
        public void True_IsGreedy()
        {
            np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, true)
                .path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
        }

        [TestMethod]
        public void OneOperand_And_TwoOperand_Shortcuts()
        {
            np.einsum_path("ii->i", new[] { Z(3, 3) }, "greedy").path.ToString()
                .Should().Be("['einsum_path', (0,)]");
            np.einsum_path("ij,jk->ik", new[] { Z(2, 2), Z(2, 3) }, "greedy").path.ToString()
                .Should().Be("['einsum_path', (0, 1)]");
        }

        [TestMethod]
        public void MemoryLimitTuple_IsAccepted()
        {
            np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, ("greedy", 5))
                .path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
        }

        [TestMethod]
        public void ExplicitPath_IsReturnedVerbatim()
        {
            var explicitPath = new EinsumPath(new[] { new[] { 1, 2 }, new[] { 0, 1 } });
            np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, explicitPath)
                .path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
        }

        // ------------------------------------------------------------------ ellipsis (path + metrics)

        [TestMethod]
        public void Ellipsis_PathAndMetrics_Match()
        {
            // NumPy's ellipsis PLACEHOLDER letters are per-process random, so only the path and the
            // letter-free numeric lines are contractual. Both match NumPy 2.4.2.
            var (path, repr) = np.einsum_path("...ij,jk->...ik", new[] { Z(2, 3, 4), Z(4, 5) }, "greedy");
            path.ToString().Should().Be("['einsum_path', (0, 1)]");
            repr.Should().Contain("         Naive scaling:  4");
            repr.Should().Contain("      Naive FLOP count:  2.400e+02");
            repr.Should().Contain("  Optimized FLOP count:  2.410e+02");
            repr.Should().Contain("   Theoretical speedup:  0.996");
            repr.Should().Contain("  Largest intermediate:  3.000e+01 elements");
        }

        // ------------------------------------------------------------------ sublist / variadic forms

        [TestMethod]
        public void Sublist_And_Variadic_Forms()
        {
            np.einsum_path(Z(2, 3), new[] { 0, 1 }, Z(3, 4), new[] { 1, 2 }, new[] { 0, 2 })
                .path.ToString().Should().Be("['einsum_path', (0, 1)]");
            np.einsum_path("ij,jk,kl->il", Z(2, 2), Z(2, 5), Z(5, 2))
                .path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
        }

        // ------------------------------------------------------------------ error taxonomy

        [TestMethod]
        public void WrongOperandCount_Raises()
        {
            new Action(() => np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 2) }))
                .Should().Throw<ValueError>()
                .WithMessage("Number of einsum subscripts must be equal to the number of operands.");
        }

        [TestMethod]
        public void InvalidSymbol_Raises()
        {
            new Action(() => np.einsum_path("i1,jk->ik", new[] { Z(2, 2), Z(2, 2) }))
                .Should().Throw<ValueError>().WithMessage("Character 1 is not a valid symbol.");
        }

        [TestMethod]
        public void SizeMismatch_Raises()
        {
            new Action(() => np.einsum_path("ij,jk->ik", new[] { Z(2, 3), Z(4, 5) }))
                .Should().Throw<ValueError>()
                .WithMessage("Size of label 'j' for operand 1 (3) does not match previous terms (4).");
        }

        [TestMethod]
        public void OutputCharacterTwice_Raises()
        {
            new Action(() => np.einsum_path("ij,jk->ii", new[] { Z(2, 2), Z(2, 2) }))
                .Should().Throw<ValueError>()
                .WithMessage("Output character i appeared more than once in the output.");
        }

        [TestMethod]
        public void OutputCharacterNotInInput_Raises()
        {
            new Action(() => np.einsum_path("ij,jk->ixk", new[] { Z(2, 2), Z(2, 2) }))
                .Should().Throw<ValueError>()
                .WithMessage("Output character x did not appear in the input");
        }

        [TestMethod]
        public void WrongNumberOfIndicesForOperand_Raises()
        {
            new Action(() => np.einsum_path("ij,jk->ik", new[] { Z(2, 2, 2), Z(2, 2) }))
                .Should().Throw<ValueError>()
                .WithMessage("Einstein sum subscript i does not contain the correct number of indices for operand 0.");
        }

        [TestMethod]
        public void IncompleteExplicitPath_RaisesRuntimeError()
        {
            var bad = new EinsumPath(new[] { new[] { 0, 1 } });
            new Action(() => np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 2), Z(2, 2) }, bad))
                .Should().Throw<RuntimeError>()
                .WithMessage("Invalid einsum_path is specified: 1 more operands has to be contracted.");
        }

        [TestMethod]
        public void UnknownPathName_RaisesKeyError()
        {
            new Action(() => np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 2), Z(2, 2) }, "banana"))
                .Should().Throw<KeyError>().WithMessage("('Path name %s not found', 'banana')");
        }

        [TestMethod]
        public void NotUnderstoodPath_RaisesTypeError()
        {
            new Action(() => np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 2), Z(2, 2) }, new object[] { 1, 2, 3 }))
                .Should().Throw<TypeError>().WithMessage("Did not understand the path: [1, 2, 3]");
        }

        [TestMethod]
        public void TwoArrows_Raises()
        {
            new Action(() => np.einsum_path("ij->jk->ik", new[] { Z(2, 2) }))
                .Should().Throw<ValueError>().WithMessage("Subscripts can only contain one '->'.");
        }

        // ------------------------------------------------------------------ EinsumPath value type

        [TestMethod]
        public void EinsumPath_Surface()
        {
            var path = new EinsumPath(new[] { new[] { 1, 2 }, new[] { 0, 1 } });
            path.Count.Should().Be(2);
            path[0].Should().Equal(1, 2);
            path[1].Should().Equal(0, 1);
            path.ToString().Should().Be("['einsum_path', (1, 2), (0, 1)]");
            path.ToList().Should().HaveCount(3);
            path.ToList()[0].Should().Be("einsum_path");
            path.Equals(new EinsumPath(new[] { new[] { 1, 2 }, new[] { 0, 1 } })).Should().BeTrue();
            path.Equals(new EinsumPath(new[] { new[] { 0, 1 }, new[] { 0, 1 } })).Should().BeFalse();

            // Single-step tuple renders with the Python trailing comma.
            new EinsumPath(new[] { new[] { 0 } }).ToString().Should().Be("['einsum_path', (0,)]");
        }

        [TestMethod]
        public void EinsumPath_Deconstructs()
        {
            var (path, repr) = np.einsum_path("ij,jk,kl->il", new[] { Z(2, 2), Z(2, 5), Z(5, 2) }, "greedy");
            var (a, b) = (path[0], path[1]);
            a.Should().Equal(1, 2);
            b.Should().Equal(0, 1);
            repr.Should().StartWith("  Complete contraction:  ij,jk,kl->il");
        }

        // ------------------------------------------------------------------ round-trip into einsum

        [TestMethod]
        public void PathRoundTripsIntoEinsum()
        {
            var a = np.arange(4).reshape(2, 2).astype(np.float64);
            var b = np.arange(10).reshape(2, 5).astype(np.float64);
            var c = np.arange(10).reshape(5, 2).astype(np.float64);

            var path = np.einsum_path("ij,jk,kl->il", new[] { a, b, c }, "greedy").path;
            var baseline = np.einsum("ij,jk,kl->il", new[] { a, b, c });

            np.array_equal(np.einsum("ij,jk,kl->il", new[] { a, b, c }, optimize: path), baseline).Should().BeTrue();
            np.array_equal(np.einsum("ij,jk,kl->il", new[] { a, b, c }, optimize: ("greedy", 8)), baseline).Should().BeTrue();
            np.array_equal(np.einsum("ij,jk,kl->il", new[] { a, b, c }, optimize: path.ToList()), baseline).Should().BeTrue();
        }
    }
}
