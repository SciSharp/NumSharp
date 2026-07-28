using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     Parameter NAMES, ORDER and DEFAULTS of the BLAS/LAPACK surface, against NumPy 2.4.2's own
    ///     <c>inspect.signature</c> output.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A signature is API: renaming <c>UPLO</c> to <c>uplo</c> or swapping <c>keepdims</c> and
    ///     <c>ord</c> compiles, passes every value test, and silently breaks every named-argument
    ///     call ported from Python. Nothing else in the suite would notice, so this asserts it
    ///     directly by reflection.
    ///     </para>
    ///     <para>
    ///     The expected sequences below are NumPy's, transcribed from its printed signatures — the
    ///     <c>/</c> and <c>*</c> markers drop out because C# has no positional-only or
    ///     keyword-only parameters, but the ORDER they impose is kept so a positional call
    ///     transcribes verbatim.
    ///     </para>
    ///     <para>
    ///     <b>Four NumPy ufunc keywords are deliberately absent</b> from the three gufuncs:
    ///     <c>casting</c>, <c>order</c>, <c>subok</c> and <c>signature</c>. NumSharp models none of
    ///     them anywhere in its ufunc surface — <c>signature</c> is what <c>dtype</c> already does
    ///     and <c>subok</c> concerns ndarray subclasses the library does not have — so accepting and
    ///     ignoring them would be worse than not offering them.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class LinAlgSignatureParityTests
    {
        // NumPy 2.4.2:  np.inner(a, b, /)   np.vdot(a, b, /)   np.tensordot(a, b, axes=2)
        // np.vecdot(x1, x2, /, out=None, *, axes, axis, keepdims=False, casting, order, dtype, subok, signature)
        private static readonly (string Method, string[] Parameters)[] MainNamespace =
        {
            ("inner", new[] {"a", "b"}),
            ("vdot", new[] {"a", "b"}),
            ("vecdot", new[] {"x1", "x2", "out", "axes", "axis", "keepdims", "dtype"}),
            ("matvec", new[] {"x1", "x2", "out", "axes", "axis", "keepdims", "dtype"}),
            ("vecmat", new[] {"x1", "x2", "out", "axes", "axis", "keepdims", "dtype"}),
            ("tensordot", new[] {"a", "b", "axes"}),
            // NumPy's printed signature is einsum(*operands, out, optimize, **kwargs); the DOCUMENTED
            // form it expands to is einsum(subscripts, *operands, out, dtype, order, casting, optimize).
            ("einsum", new[] {"subscripts", "operands", "out", "dtype", "order", "casting", "optimize"})
        };

        private static readonly (string Method, string[] Parameters)[] Linalg =
        {
            ("cholesky", new[] {"a", "upper"}),
            ("cond", new[] {"x", "p"}),
            ("cross", new[] {"x1", "x2", "axis"}),
            ("det", new[] {"a"}),
            ("diagonal", new[] {"x", "offset"}),
            ("eig", new[] {"a"}),
            ("eigh", new[] {"a", "UPLO"}),
            ("eigvals", new[] {"a"}),
            ("eigvalsh", new[] {"a", "UPLO"}),
            ("inv", new[] {"a"}),
            ("lstsq", new[] {"a", "b", "rcond"}),
            ("matmul", new[] {"x1", "x2"}),
            ("matrix_norm", new[] {"x", "keepdims", "ord"}),
            ("matrix_power", new[] {"a", "n"}),
            ("matrix_rank", new[] {"A", "tol", "hermitian", "rtol"}),
            ("matrix_transpose", new[] {"x"}),
            ("multi_dot", new[] {"arrays", "out"}),
            ("norm", new[] {"x", "ord", "axis", "keepdims"}),
            ("outer", new[] {"x1", "x2"}),
            ("pinv", new[] {"a", "rcond", "hermitian", "rtol"}),
            ("qr", new[] {"a", "mode"}),
            ("slogdet", new[] {"a"}),
            ("solve", new[] {"a", "b"}),
            ("svd", new[] {"a", "full_matrices", "compute_uv", "hermitian"}),
            ("svdvals", new[] {"x"}),
            ("tensordot", new[] {"x1", "x2", "axes"}),
            ("tensorinv", new[] {"a", "ind"}),
            ("tensorsolve", new[] {"a", "b", "axes"}),
            ("trace", new[] {"x", "offset", "dtype"}),
            ("vecdot", new[] {"x1", "x2", "axis"}),
            ("vector_norm", new[] {"x", "axis", "keepdims", "ord"})
        };

        [TestMethod]
        public void MainNamespace_ParameterNamesAndOrderMatchNumPy()
        {
            foreach (var (method, expected) in MainNamespace)
                AssertHasOverload(typeof(np), method, expected);
        }

        [TestMethod]
        public void Linalg_ParameterNamesAndOrderMatchNumPy()
        {
            foreach (var (method, expected) in Linalg)
                AssertHasOverload(typeof(np.linalg), method, expected);
        }

        [TestMethod]
        public void EveryPublicLinalgFunctionNumPyHas_Exists()
        {
            // The module's whole surface in NumPy 2.4.2, so a missing member is a failure rather
            // than something nobody notices until a port hits it.
            var expected = Linalg.Select(e => e.Method).ToHashSet();
            var actual = typeof(np.linalg)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Select(m => m.Name)
                .ToHashSet();

            expected.Except(actual).Should().BeEmpty("np.linalg must expose every NumPy 2.4.2 member");
            typeof(LinAlgError).Should().NotBeNull();
        }

        [TestMethod]
        public void DefaultsMatchNumPy()
        {
            // Only the defaults NumPy states as a concrete value; the sentinels are asserted below.
            DefaultOf(typeof(np), "tensordot", "axes").Should().Be(2);
            DefaultOf(typeof(np.linalg), "cholesky", "upper").Should().Be(false);
            DefaultOf(typeof(np.linalg), "cross", "axis").Should().Be(-1);
            DefaultOf(typeof(np.linalg), "diagonal", "offset").Should().Be(0);
            DefaultOf(typeof(np.linalg), "eigh", "UPLO").Should().Be('L');
            DefaultOf(typeof(np.linalg), "eigvalsh", "UPLO").Should().Be('L');
            DefaultOf(typeof(np.linalg), "matrix_rank", "hermitian").Should().Be(false);
            DefaultOf(typeof(np.linalg), "matrix_norm", "keepdims").Should().Be(false);
            DefaultOf(typeof(np.linalg), "norm", "keepdims").Should().Be(false);
            DefaultOf(typeof(np.linalg), "pinv", "hermitian").Should().Be(false);
            DefaultOf(typeof(np.linalg), "qr", "mode").Should().Be("reduced");
            DefaultOf(typeof(np.linalg), "svd", "full_matrices").Should().Be(true);
            DefaultOf(typeof(np.linalg), "svd", "compute_uv").Should().Be(true);
            DefaultOf(typeof(np.linalg), "svd", "hermitian").Should().Be(false);
            DefaultOf(typeof(np.linalg), "tensorinv", "ind").Should().Be(2);
            DefaultOf(typeof(np.linalg), "tensordot", "axes").Should().Be(2);
            DefaultOf(typeof(np.linalg), "vecdot", "axis").Should().Be(-1);
            DefaultOf(typeof(np.linalg), "vector_norm", "keepdims").Should().Be(false);
        }

        [TestMethod]
        public void TheTwoNonNullDefaultsCsharpCannotSpell_AreCarriedBySentinels()
        {
            // NumPy's matrix_norm defaults ord='fro' and vector_norm defaults ord=2, but a C#
            // `object` parameter may only default to null — so null MEANS those values, and that
            // has to be observable rather than merely documented.
            DefaultOf(typeof(np.linalg), "matrix_norm", "ord").Should().BeNull();
            DefaultOf(typeof(np.linalg), "vector_norm", "ord").Should().BeNull();

            var m = np.arange(6.0).reshape(2, 3);
            Scalar(np.linalg.matrix_norm(m)).Should().Be(Scalar(np.linalg.matrix_norm(m, ord: "fro")));
            Scalar(np.linalg.vector_norm(m)).Should().Be(Scalar(np.linalg.vector_norm(m, ord: 2)));

            // ...and they are NOT the same as some other plausible default.
            Scalar(np.linalg.vector_norm(m)).Should().NotBe(Scalar(np.linalg.vector_norm(m, ord: 1)));
        }

        private static double Scalar(NDArray a) => Convert.ToDouble(a.GetValue(0));

        private static void AssertHasOverload(Type owner, string method, string[] expected)
        {
            var overloads = owner.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == method)
                .ToArray();

            overloads.Should().NotBeEmpty($"{owner.Name}.{method} must exist");

            var found = overloads.Any(m => m.GetParameters().Select(p => p.Name).SequenceEqual(expected));

            found.Should().BeTrue(
                $"{owner.Name}.{method} must have an overload with parameters "
                + $"({string.Join(", ", expected)}) — NumPy 2.4.2's names in NumPy's order. Found: "
                + string.Join(" | ", overloads.Select(m => $"({string.Join(", ", m.GetParameters().Select(p => p.Name))})")));
        }

        private static object DefaultOf(Type owner, string method, string parameter)
        {
            var candidates = owner.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == method)
                .SelectMany(m => m.GetParameters())
                .Where(p => p.Name == parameter && p.HasDefaultValue)
                .ToArray();

            candidates.Should().NotBeEmpty($"{owner.Name}.{method} must take an optional '{parameter}'");
            return candidates[0].DefaultValue;
        }
    }
}
