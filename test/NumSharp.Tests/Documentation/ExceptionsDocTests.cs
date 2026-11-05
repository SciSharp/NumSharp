using System;
using NumSharp;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for <c>docs/website-src/docs/exceptions.md</c>: the hierarchy the page draws (each class's
    ///     .NET base and the <see cref="INumSharpException"/> marker), every row of its "What common failures raise"
    ///     table (type and NumPy 2.4.2's text), and its catching example. A change of exception type or text fails here
    ///     before the page misleads a <c>catch</c>.
    /// </summary>
    [TestClass]
    public class ExceptionsDocTests
    {
        /// <summary>The "Exception hierarchy" diagram: the NumPy classes sit on the .NET bases the page names, so a .NET
        /// catch of the base also catches them.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Hierarchy_AsDrawn()
        {
            typeof(TypeError).BaseType.Should().Be(typeof(NumSharpException));
            typeof(IncorrectTypeException).BaseType.Should().Be(typeof(TypeError));
            typeof(DTypePromotionError).BaseType.Should().Be(typeof(TypeError));
            typeof(IndexError).BaseType.Should().Be(typeof(NumSharpException));
            typeof(TooHardError).BaseType.Should().Be(typeof(RuntimeError));
            typeof(IncorrectShapeException).BaseType.Should().Be(typeof(NumSharpException));
            typeof(IncorrectSizeException).BaseType.Should().Be(typeof(NumSharpException));
            typeof(ValueError).BaseType.Should().Be(typeof(ArgumentException));
            typeof(UnicodeEncodeError).BaseType.Should().Be(typeof(UnicodeError));
            typeof(UnicodeError).BaseType.Should().Be(typeof(ValueError));
            typeof(LinAlgError).BaseType.Should().Be(typeof(ValueError));
            typeof(LookupError).BaseType.Should().Be(typeof(ArgumentException));
            typeof(AxisError).BaseType.Should().Be(typeof(ArgumentOutOfRangeException));
            typeof(AxisOutOfRangeException).BaseType.Should().Be(typeof(AxisError));
            typeof(OSError).BaseType.Should().Be(typeof(System.IO.IOException));
            typeof(OpenBlasMissingBackendException).BaseType.Should().Be(typeof(MissingBackendException));
            typeof(MissingBackendException).BaseType.Should().Be(typeof(NotSupportedException));
            typeof(StringComputationNotSupportedException).BaseType.Should().Be(typeof(NotSupportedException));
            typeof(MaskError).BaseType.Should().Be(typeof(MAError));
            foreach (var t in new[] { typeof(ValueError), typeof(AxisError), typeof(LookupError), typeof(OSError), typeof(TypeError),
                                      typeof(MissingBackendException), typeof(StringComputationNotSupportedException) })
                typeof(INumSharpException).IsAssignableFrom(t).Should().BeTrue(t.Name);
        }

        /// <summary>The "What common failures raise" table, row by row: the NumSharp type and NumPy's message.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void CommonFailures_AsTabulated()
        {
            Action broadcast = () => _ = np.ones((3, 4)) + np.ones(3);
            broadcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (3,4) (3,)*");
            Action reshape = () => np.arange(6).reshape(4);
            reshape.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 6 into shape (4,)");
            Action axis = () => np.sum(np.ones((2, 3)), 5);
            axis.Should().Throw<AxisError>().WithMessage("axis 5 is out of bounds for array of dimension 2*");
            Action index = () => _ = np.arange(3)[5];
            index.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 0 with size 3");
            Action empty = () => np.max(np.array(new double[0]));
            empty.Should().Throw<ValueError>().WithMessage("zero-size array to reduction operation maximum which has no identity");
            Action readOnly = () => { var b = np.broadcast_to(np.arange(3), (2, 3)); b[0, 0] = 1; };
            readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            Action text = () => np.sqrt(np.array(new[] { "a" }));
            text.Should().Throw<TypeError>().WithMessage("ufunc 'sqrt' not supported for the input types, *");
            Action noLoop = () => np.sqrt(np.array(new[] { 1.5 }), dtype: "U");
            noLoop.Should().Throw<IncorrectTypeException>().WithMessage("No loop matching the specified signature and casting was found for ufunc sqrt");
            Action promotion = () => np.promote_types(np.dtype("T"), np.int32);
            promotion.Should().Throw<DTypePromotionError>().WithMessage("The DTypes * do not have a common DType.*");
            Action overflow = () => np.zeros(3, dtype: np.int8).fill(300);
            overflow.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            Action parse = () => np.array(new[] { "x" }).astype(np.int64);
            parse.Should().Throw<ValueError>().WithMessage("invalid literal for int() with base 10: np.str_('x')");
            Action encode = () => np.array(new[] { "é" }).astype("S1");
            encode.Should().Throw<UnicodeEncodeError>().WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: *");
            Action codec = () => np.strings.encode(np.array(new[] { "a" }), "nope");
            codec.Should().Throw<LookupError>().WithMessage("unknown encoding: nope");
            Action singular = () => np.linalg.inv(np.zeros((2, 2)));
            singular.Should().Throw<LinAlgError>().WithMessage("Singular matrix");
            Action item = () => np.array(new[] { 1, 2 }).item();
            item.Should().Throw<IncorrectSizeException>().WithMessage("can only convert an array of size 1 to a Python scalar");
            Action cast = () => _ = (int)np.array(new[] { 1, 2 });
            cast.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
            Action truth = () => _ = (bool)np.array(new[] { 1, 2 });
            truth.Should().Throw<ValueError>().WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            Action emptyTruth = () => _ = (bool)np.array(new double[0]);
            emptyTruth.Should().Throw<ValueError>()
                .WithMessage("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            Action dtype = () => np.dtype("xyz");
            dtype.Should().Throw<TypeError>().WithMessage("data type 'xyz' not understood");
            // The table's row: a delegate inside np.evaluate given text (every other node computes on text or raises NumPy's
            // own error; the concatenation of two strings is NumPy's 'ab').
            Action fused = () => np.evaluate(NDExpr.Call((Func<double, double>)System.Math.Abs, np.array(new[] { "a" })));
            fused.Should().Throw<StringComputationNotSupportedException>().WithMessage("*NDExpr.Call delegate*");
            NDArray.AsStringArray(np.evaluate((NDExpr)np.array(new[] { "a" }) + np.array(new[] { "b" }))).Should().Equal("ab");
        }

        /// <summary>The "Catching exceptions" example: <c>catch (AxisError)</c> takes the axis failure, a .NET
        /// <c>catch (ArgumentException)</c> takes a <see cref="ValueError"/>, and the <see cref="INumSharpException"/>
        /// filter takes a <see cref="TypeError"/>, whose base is not a .NET argument exception.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void Catching_ByNumPyClassDotNetBaseOrMarker()
        {
            string caught = null;
            try { np.sum(np.ones((2, 3)), 5); }
            catch (AxisError e) { caught = e.Message; }
            caught.Should().StartWith("axis 5 is out of bounds for array of dimension 2");

            bool argument = false;
            try { np.max(np.array(new double[0])); }
            catch (ArgumentException) { argument = true; }
            argument.Should().BeTrue("ValueError is an ArgumentException");

            bool marker = false;
            try { np.sqrt(np.array(new[] { "a" })); }
            catch (Exception e) when (e is INumSharpException) { marker = true; }
            marker.Should().BeTrue();
        }
    }
}
