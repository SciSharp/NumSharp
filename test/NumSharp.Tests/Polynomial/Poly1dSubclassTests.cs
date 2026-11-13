using System;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     <c>poly1d</c> is unsealed (the owner's decision of 2026-11-10, "Unseal poly1d only"), as NumPy's <c>poly1d</c> is a
    ///     plain Python class. NumPy's operators build <c>poly1d(...)</c> explicitly, so a subclass's results are plain
    ///     <c>poly1d</c> there too; probed on NumPy 2.4.2 with <c>class Named(np.poly1d): pass</c>.
    /// </summary>
    [TestClass]
    public class Poly1dSubclassTests
    {
        /// <summary>A consumer's subclass that counts the dispose calls reaching its override.</summary>
        private sealed class Named : poly1d
        {
            /// <summary>Builds the polynomial from its coefficients, as <c>poly1d(c)</c> does.</summary>
            /// <param name="c">The coefficients, highest power first.</param>
            public Named(NDArray c) : base(c) { }

            /// <summary>How many times <see cref="Dispose(bool)"/> ran.</summary>
            public int DisposeCalls;

            /// <inheritdoc/>
            protected override void Dispose(bool disposing)
            {
                DisposeCalls++;
                base.Dispose(disposing);
            }
        }

        /// <summary>
        ///     NumPy: <c>type(p + p)</c>, <c>type(p.deriv())</c> and <c>type(p * 2)</c> are <c>poly1d</c> for a subclass
        ///     instance <c>p</c>, and the values are the polynomial's.
        /// </summary>
        [TestMethod]
        public void Operations_ReturnAPlainPoly1d_AsNumPy()
        {
            using var p = new Named(np.array(new[] { 1.0, 2.0 }));
            using var sum = p + p;
            using var derivative = p.deriv();
            using var scaled = p * 2.0;
            sum.GetType().Should().Be(typeof(poly1d));
            derivative.GetType().Should().Be(typeof(poly1d));
            scaled.GetType().Should().Be(typeof(poly1d));
            sum.coeffs.ToArray<double>().Should().Equal(2.0, 4.0);
            derivative.coeffs.ToArray<double>().Should().Equal(1.0);
        }

        /// <summary>
        ///     <c>Dispose()</c> runs the subclass's <c>Dispose(bool)</c> override, and the base releases the coefficient array
        ///     once: a second dispose reaches the override again and releases nothing more.
        /// </summary>
        [TestMethod]
        public void Dispose_RunsTheOverride_AndReleasesTheCoefficientsOnce()
        {
            var p = new Named(np.array(new[] { 1.0, 2.0 }));
            p.Dispose();
            p.DisposeCalls.Should().Be(1);
            p.coeffs.Should().BeNull();
            p.Dispose();
            p.DisposeCalls.Should().Be(2);
            p.coeffs.Should().BeNull();
        }
    }
}
