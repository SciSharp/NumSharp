using System;
using System.Collections.Generic;
using NumSharp.Generic;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     Bug 2d of <c>docs/plans/view-shape-open-bugs.md</c>: a CALLER'S array of dimensions (or strides) never becomes an
    ///     array's or a shape's own. NumPy copies every shape sequence it is given (<c>PyArray_IntpConverter</c>) and an array
    ///     owns its dimensions (<c>PyArray_NewFromDescr_int</c>), so changing the caller's list afterwards changes nothing:
    ///     <c>s = [2, 3]; a = np.zeros(s); s[0] = 100; a.shape == (2, 3)</c>. The C# analog of NumPy's list is the caller's
    ///     own <c>long[]</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every expected value is NumPy 2.4.2's, probed on 2026-10-05 for both a list and an int64 ndarray sequence (each
    ///     route KEPT its dimensions). Before the fix, 24 of the 32 NumSharp routes probed took the caller's array: the
    ///     dimensions became <c>(100, 3)</c> while the cached size stayed 6, so a read of a "new" row read past the buffer.
    ///     Every test therefore checks the shape (and size) only, never an element.
    ///     </para>
    ///     <para>
    ///     The three tests named <c>CallerDims_*</c> were <c>[OpenBugs]</c> pins in <c>OpenBugs.ViewShape.cs</c> (commit
    ///     <c>2f92d91d</c>).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ShapeCallerArrayTests
    {
        /// <summary>The caller's dimensions every test starts from, and NumPy's answer after the caller changes them.</summary>
        private static readonly long[] Expected = { 2, 3 };

        /// <summary>
        ///     <c>np.zeros(dims)</c> keeps its dimensions when the caller later writes <c>dims[0]</c> (formerly the pin
        ///     <c>CallerDims_AfterNpZeros_ShouldNotChangeTheArray</c>).
        /// </summary>
        [TestMethod]
        public void CallerDims_AfterNpZeros_DoNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var a = np.zeros(dims);
            dims[0] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 }, "np.zeros copies the caller's dimensions");
            a.size.Should().Be(12);
        }

        /// <summary>
        ///     <c>new NDArray(dtype, new Shape(dims))</c>, the allocating constructor every <c>np.*</c> creation function
        ///     reaches (formerly the pin <c>CallerDims_AfterNDArrayConstructor_ShouldNotChangeTheArray</c>).
        /// </summary>
        [TestMethod]
        public void CallerDims_AfterNDArrayConstructor_DoNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var a = new NDArray(np.float64, new Shape(dims));
            dims[1] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 }, "the Shape constructor copies the caller's dimensions");
        }

        /// <summary>
        ///     <c>rng.random(new Shape(dims))</c>: a sampler keeps a clean size as it is, so the copy has to happen when the
        ///     shape is built (formerly the pin <c>CallerDims_AfterRandomSize_ShouldNotChangeTheArray</c>).
        /// </summary>
        [TestMethod]
        public void CallerDims_AfterRandomSize_DoNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var r = np.random.default_rng(42).random(new Shape(dims));
            dims[0] = 100;

            r.shape.Should().Equal(new long[] { 3, 4 }, "the size shape copies the caller's dimensions");
        }

        /// <summary>
        ///     Every public <see cref="Shape"/> constructor and conversion that takes a <c>long[]</c> copies it: the
        ///     dimensions, the strides, the cached size and the flags stay what the shape was built with.
        /// </summary>
        [TestMethod]
        public void ShapeConstructorsAndConversions_CopyTheCallersArrays()
        {
            long[] d = { 2, 3 };
            var fromParams = new Shape(d);
            var fromOrder = new Shape(d, 'F');
            Shape fromConversion = d;
            Shape? fromNullableConversion = d;
            long[] dd = { 2, 3 }, ss = { 3, 1 };
            var fromStrides = new Shape(dd, ss);
            var fromOriginal = new Shape(dd, ss, new Shape(2, 3));

            d[0] = 100;
            dd[0] = 100;
            ss[0] = 7;

            fromParams.Dimensions.Should().Equal(Expected);
            fromParams.Size.Should().Be(6);
            fromOrder.Dimensions.Should().Equal(Expected);
            fromOrder.Strides.Should().Equal(new long[] { 1, 2 }, "F strides of (2, 3)");
            fromConversion.Dimensions.Should().Equal(Expected);
            fromNullableConversion.Value.Dimensions.Should().Equal(Expected);
            fromStrides.Dimensions.Should().Equal(Expected);
            fromStrides.Strides.Should().Equal(new long[] { 3, 1 });
            fromStrides.IsContiguous.Should().BeTrue("the flags were computed from the copies");
            fromOriginal.Dimensions.Should().Equal(Expected);
            fromOriginal.Strides.Should().Equal(new long[] { 3, 1 });
        }

        /// <summary>
        ///     The <c>np.*</c> creation functions over a caller's <c>long[]</c> (and over a shape built from one), generic
        ///     forms included. NumPy: every one keeps <c>(2, 3)</c>.
        /// </summary>
        [TestMethod]
        public void CreationFunctions_CopyTheCallersArray()
        {
            var routes = new List<(string name, Func<long[], NDArray> make)>
            {
                ("np.zeros(long[])", d => np.zeros(d)),
                ("np.zeros<double>(long[])", d => np.zeros<double>(d)),
                ("np.empty(long[])", d => np.empty(d)),
                ("np.empty<double>(long[])", d => np.empty<double>(d)),
                ("np.ones(long[])", d => np.ones(d)),
                ("np.full(long[], 7.0)", d => np.full(d, 7.0)),
                ("np.full<double>(long[], 7.0)", d => np.full<double>(d, 7.0)),
                ("np.zeros(new Shape(long[]))", d => np.zeros(new Shape(d))),
                ("new NDArray(double[6], long[])", d => new NDArray(new double[6], d)),
                ("new NDArray(double[6], new Shape(long[]), 'F')", d => new NDArray(new double[6], new Shape(d), 'F')),
            };

            foreach (var (name, make) in routes)
            {
                long[] dims = { 2, 3 };
                using var a = make(dims);
                dims[0] = 100;

                a.shape.Should().Equal(Expected, name);
                a.size.Should().Be(6, name);
            }
        }

        /// <summary>
        ///     <c>a.resize(dims)</c> (in place) keeps the requested dimensions when the caller later changes its array.
        ///     NumPy: <c>a.resize(s); s[0] = 100; a.shape == (2, 3)</c>.
        /// </summary>
        [TestMethod]
        public void Resize_CopiesTheCallersArray()
        {
            using var a = np.zeros(new Shape(6));
            long[] dims = { 2, 3 };
            a.resize(dims);
            dims[0] = 100;

            a.shape.Should().Equal(Expected);
            a.size.Should().Be(6);
        }

        /// <summary>
        ///     The random samplers that take a <c>long[]</c> size (or a shape built from one), legacy and
        ///     <see cref="Generator"/>. NumPy: every one keeps <c>(2, 3)</c> (the vector samplers append their own axis).
        /// </summary>
        [TestMethod]
        public void RandomSizes_CopyTheCallersArray()
        {
            var cov = new double[,] { { 1, 0 }, { 0, 1 } };
            var routes = new List<(string name, Func<long[], NDArray> make, long[] expected)>
            {
                ("np.random.rand(long[])", d => np.random.rand(d), Expected),
                ("np.random.randn(long[])", d => np.random.randn(d), Expected),
                ("np.random.random_sample(long[])", d => np.random.random_sample(d), Expected),
                ("np.random.normal(0, 1, long[])", d => np.random.normal(0, 1, d), Expected),
                ("np.random.pareto(2.0, long[])", d => np.random.pareto(2.0, d), Expected),
                ("np.random.power(2.0, long[])", d => np.random.power(2.0, d), Expected),
                ("np.random.logseries(0.5, long[])", d => np.random.logseries(0.5, d), Expected),
                ("np.random.hypergeometric(5, 5, 3, long[])", d => np.random.hypergeometric(5, 5, 3, d), Expected),
                ("np.random.dirichlet([1, 1], long[])", d => np.random.dirichlet(new double[] { 1, 1 }, d), new long[] { 2, 3, 2 }),
                ("np.random.multivariate_normal(mean, I, long[])", d => np.random.multivariate_normal(new double[] { 0, 0 }, cov, d), new long[] { 2, 3, 2 }),
                ("default_rng(0).integers(0, 9, long[])", d => np.random.default_rng(0).integers(0, 9, d), Expected),
            };

            foreach (var (name, make, expected) in routes)
            {
                long[] dims = { 2, 3 };
                using var a = make(dims);
                dims[0] = 100;

                a.shape.Should().Equal(expected, name);
            }
        }

        /// <summary>
        ///     The routes that already built their own dimensions keep doing so (guards): reshape (all three spellings, and
        ///     the typed wrapper's reshape and reshape_unsafe, which reach <c>UnmanagedStorage.Reshape(long[])</c>), the
        ///     storage's own in-place <c>Reshape</c>, the <c>shape</c> setter, <c>np.broadcast_to</c>, <c>np.indices</c> and
        ///     <c>np.tile</c>, plus the <see cref="Shape"/> constructors over an <c>int[]</c> and an
        ///     <see cref="IEnumerable{T}"/>. Probed on the 2c commit (95fd5149): every one KEPT its dimensions.
        /// </summary>
        [TestMethod]
        public void RoutesThatAlreadyCopied_StillDo()
        {
            var routes = new List<(string name, Func<long[], NDArray> make, long[] expected)>
            {
                ("arange(6).reshape(long[])", d => np.arange(6).reshape(d), Expected),
                ("np.reshape(arange(6), long[])", d => np.reshape(np.arange(6), d), Expected),
                ("arange(6).reshape_unsafe(long[])", d => np.arange(6).reshape_unsafe(d), Expected),
                ("NDArray<long>.reshape(long[])", d => np.arange(6).MakeGeneric<long>().reshape(d), Expected),
                ("NDArray<long>.reshape_unsafe(long[])", d => np.arange(6).MakeGeneric<long>().reshape_unsafe(d), Expected),
                ("a.Storage.Reshape(long[]) (in place)", d => { var a = np.arange(6); a.Storage.Reshape(d); return a; }, Expected),
                ("a.shape = long[]", d => { var a = np.arange(6); a.shape = d; return a; }, Expected),
                ("np.broadcast_to(arange(3), (Shape)long[])", d => np.broadcast_to(np.arange(3), (Shape)d), Expected),
                ("np.indices(long[])", d => np.indices(d), new long[] { 2, 2, 3 }),
                ("np.tile(arange(2), long[] reps)", d => np.tile(np.arange(2), d), new long[] { 2, 6 }),
            };

            foreach (var (name, make, expected) in routes)
            {
                long[] dims = { 2, 3 };
                using var a = make(dims);
                dims[0] = 100;

                a.shape.Should().Equal(expected, name);
            }

            int[] idims = { 2, 3 };
            var fromInts = new Shape(idims);
            var list = new List<long> { 2, 3 };
            var fromList = new Shape(list);
            idims[0] = 100;
            list[0] = 100;
            fromInts.Dimensions.Should().Equal(Expected);
            fromList.Dimensions.Should().Equal(Expected);
        }

        /// <summary>
        ///     Inline values (<c>new Shape(rows, cols)</c>) bind the <c>params ReadOnlySpan&lt;long&gt;</c> constructor, so
        ///     building the shape allocates exactly what adopting a fresh array does: the dimension and stride arrays, and no
        ///     params array to copy from.
        /// </summary>
        /// <remarks>
        ///     The values are locals on purpose. For CONSTANT values the compiler builds the span from static data
        ///     (<c>ldtoken</c> + <c>CreateSpan</c>), which allocates nothing in optimized code but a 72-byte field-handle stub
        ///     per call in an unoptimized (Debug) build, so a constant form would make this test depend on the configuration.
        ///     Measured in Release on 2026-10-05: <c>new Shape(2, 3)</c>, <c>new Shape(n, 3)</c> and
        ///     <c>Shape.Adopt(new long[] { 2, 3 })</c> all allocate 80 bytes.
        /// </remarks>
        [TestMethod]
        public void InlineValues_AllocateOnlyTheShapesOwnArrays()
        {
            static long Measure(Func<Shape> make)
            {
                for (int i = 0; i < 100; i++)
                    make();
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 1000; i++)
                    make();
                return GC.GetAllocatedBytesForCurrentThread() - before;
            }

            long rows = 2, cols = 3;
            long inline = Measure(() => new Shape(rows, cols));
            long adopted = Measure(() => Shape.Adopt(new long[] { rows, cols }));

            inline.Should().Be(adopted, "the inline form copies the values into the one array the shape keeps");
            new Shape(2, 3).Dimensions.Should().Equal(Expected);
        }

        /// <summary>
        ///     Inline values bind the <c>params ReadOnlySpan&lt;long&gt;</c> overloads of <c>reshape</c> and
        ///     <c>reshape_unsafe</c> (untyped and typed) and of <c>resize</c>: each call allocates exactly what the same call
        ///     over an adopted array does. So the copy the array overloads make of a caller's array (bug 2d) costs an inline
        ///     call nothing; before the span overloads, <c>m.reshape(16, 64)</c> allocated 528 bytes instead of 488.
        /// </summary>
        /// <remarks>Locals, not constants, for the reason given on <see cref="InlineValues_AllocateOnlyTheShapesOwnArrays"/>.</remarks>
        [TestMethod]
        public void InlineReshapeAndResize_AllocateWhatAdoptingTheDimsDoes()
        {
            static long Measure(Action body)
            {
                for (int i = 0; i < 100; i++)
                    body();
                long before = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 1000; i++)
                    body();
                return GC.GetAllocatedBytesForCurrentThread() - before;
            }

            long rows = 2, cols = 3;
            using var a = np.arange(6);
            using var t = a.MakeGeneric<long>();
            using var b = np.arange(6);

            Measure(() => { using var r = a.reshape(rows, cols); })
                .Should().Be(Measure(() => { using var r = a.reshape(Shape.Adopt(new long[] { rows, cols })); }), "reshape");
            Measure(() => { using var r = a.reshape_unsafe(rows, cols); })
                .Should().Be(Measure(() => { using var r = a.reshape_unsafe(Shape.Adopt(new long[] { rows, cols })); }),
                    "reshape_unsafe");
            Measure(() => { using var r = t.reshape(rows, cols); })
                .Should().Be(Measure(() => { using var r = t.reshape(Shape.Adopt(new long[] { rows, cols })); }), "typed reshape");
            Measure(() => { using var r = t.reshape_unsafe(rows, cols); })
                .Should().Be(Measure(() => { using var r = t.reshape_unsafe(Shape.Adopt(new long[] { rows, cols })); }),
                    "typed reshape_unsafe");
            Measure(() => b.resize(rows, cols))
                .Should().Be(Measure(() => b.resize(Shape.Adopt(new long[] { rows, cols }))), "resize");
        }

        /// <summary>
        ///     The span overloads give the array overloads' answers (NumPy 2.4.2, probed): an unknown dimension is inferred,
        ///     an empty list reshapes a one-element array to 0-d and leaves <c>resize</c> a no-op, a size mismatch raises
        ///     NumPy's text, and the typed wrapper's overloads keep its static type.
        /// </summary>
        [TestMethod]
        public void InlineReshapeAndResize_KeepTheArrayOverloadsAnswers()
        {
            using var a = np.arange(6);
            using var inferred = a.reshape(-1, 2);
            inferred.shape.Should().Equal(new long[] { 3, 2 }, "np.arange(6).reshape(-1, 2).shape == (3, 2)");

            using var one = np.arange(1);
            using var scalar = one.reshape();
            scalar.ndim.Should().Be(0, "np.arange(1).reshape(()).shape == ()");

            using var t = a.MakeGeneric<long>();
            using NDArray<long> typed = t.reshape(3, 2);
            typed.shape.Should().Equal(new long[] { 3, 2 });
            using NDArray<long> typedUnsafe = t.reshape_unsafe(2, 3);
            typedUnsafe.shape.Should().Equal(Expected);

            using var b = np.arange(6);
            b.resize();
            b.shape.Should().Equal(new long[] { 6 }, "b.resize() changes nothing");
            b.resize(2, 3);
            b.shape.Should().Equal(Expected);

            Action mismatch = () => a.reshape(4, 2);
            mismatch.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 6 into shape (4,2)");
        }

        /// <summary>
        ///     The internal <see cref="Shape.Adopt(long[])"/> family takes the array as it is (no copy): the form NumSharp
        ///     uses for arrays it built itself, which the public constructors copy.
        /// </summary>
        [TestMethod]
        public void Adopt_SharesTheArray_ThePublicConstructorDoesNot()
        {
            long[] d = { 2, 3 };
            ReferenceEquals(Shape.Adopt(d).dimensions, d).Should().BeTrue();
            ReferenceEquals(Shape.Adopt(d, 'F').dimensions, d).Should().BeTrue();
            long[] s = { 3, 1 };
            var both = Shape.Adopt(d, s);
            ReferenceEquals(both.dimensions, d).Should().BeTrue();
            ReferenceEquals(both.strides, s).Should().BeTrue();

            ReferenceEquals(new Shape(d).dimensions, d).Should().BeFalse();
            ReferenceEquals(new Shape(d, s).strides, s).Should().BeFalse();
        }

        /// <summary>
        ///     The argument checks of the copying constructors run before anything is copied, with the messages and types
        ///     they had: a null array, mismatched lengths, a logical order.
        /// </summary>
        [TestMethod]
        public void ConstructorArgumentChecks_AreUnchanged()
        {
            Action nullDims = () => new Shape(null, new long[] { 1 });
            Action nullStrides = () => new Shape(new long[] { 1 }, (long[])null);
            Action mismatch = () => new Shape(new long[] { 2, 3 }, new long[] { 1 });
            Action logicalOrder = () => new Shape(new long[] { 2, 3 }, 'A');

            nullDims.Should().Throw<ArgumentNullException>();
            nullStrides.Should().Throw<ArgumentNullException>();
            mismatch.Should().Throw<ArgumentException>().WithMessage("*does not match size (2 != 1)*");
            logicalOrder.Should().Throw<ArgumentException>().WithMessage("Physical order must be 'C' or 'F' (got 'A')*");

            new Shape((long[])null).IsScalar.Should().BeTrue("a null dims array is the 0-d shape, as before");
        }
    }
}
