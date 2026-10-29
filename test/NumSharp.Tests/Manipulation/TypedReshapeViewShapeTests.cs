using System;
using AwesomeAssertions;
using NumSharp.Generic;

namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     <c>NDArray&lt;T&gt;.reshape(Shape)</c> reads the DIMENSIONS of its argument, as NumPy's reshape reads a shape
    ///     tuple and as the untyped <c>NDArray.reshape(Shape)</c> does.
    /// </summary>
    /// <remarks>
    ///     Until 2026-10-10 the typed overload installed the argument's strides, offset and buffer size too. The Shape
    ///     of a view (<c>x[":, :, :4"].Shape</c>) then made the result address elements outside its own buffer: wrong
    ///     values read from other heap memory, and an access violation when the buffer ended where mapped memory ends.
    ///     A comparison returns <c>NDArray&lt;bool&gt;</c>, so the natural <c>(a % 3 == 0).reshape(x.Shape)</c> hit it.
    ///     Expected values are NumPy 2.4.2's: <c>np.arange(40).reshape((5, 2, 4))[i, j, k] == (i*2 + j)*4 + k</c>.
    /// </remarks>
    [TestClass]
    public class TypedReshapeViewShapeTests : TestClass
    {
        /// <summary>The shapes of five arrays of dims (5, 2, 4) whose layouts are not a plain C layout of 40 elements.</summary>
        /// <returns>A label and the Shape for each.</returns>
        private static (string Name, Shape Shape)[] ForeignShapes()
        {
            var wide = np.arange(80).reshape(5, 2, 8);
            var deep = np.arange(40).reshape(4, 2, 5);
            return new[]
            {
                ("a stepped view", wide[":, :, ::2"].Shape),                       // strides (16, 8, 2)
                ("a view of the first columns", wide[":, :, :4"].Shape),           // strides (16, 8, 1), buffer of 80
                ("a reversed view", np.arange(40).reshape(5, 2, 4)["::-1"].Shape), // negative stride, offset 32
                ("a transposed array", deep.T.Shape),                              // F layout: strides (1, 5, 10)
                ("a broadcast array", np.broadcast_to(np.arange(4), new Shape(5, 2, 4)).Shape), // strides (0, 0, 1)
            };
        }

        /// <summary>
        ///     A typed array reshaped with the Shape of another array's view gets a C layout over its own 40 elements,
        ///     the values NumPy gives, and the layout the untyped overload gives.
        /// </summary>
        [TestMethod]
        public void TypedReshape_WithTheShapeOfAnotherArraysView_UsesItsDimensionsOnly()
        {
            foreach (var (name, foreign) in ForeignShapes())
            {
                foreign.dimensions.Should().Equal(new long[] { 5, 2, 4 }, name);

                NDArray<long> typed = np.arange(40).AsGeneric<long>() ?? throw new InvalidOperationException("arange is int64");
                NDArray<long> reshaped = typed.reshape(foreign);
                NDArray untyped = ((NDArray)typed).reshape(foreign);

                reshaped.Shape.dimensions.Should().Equal(new long[] { 5, 2, 4 }, name);
                reshaped.Shape.strides.Should().Equal(new long[] { 8, 4, 1 }, $"{name}: C strides of (5, 2, 4)");
                reshaped.Shape.offset.Should().Be(0, name);
                reshaped.Shape.bufferSize.Should().Be(40, name);
                reshaped.Shape.IsContiguous.Should().BeTrue(name);
                reshaped.Shape.strides.Should().Equal(untyped.Shape.strides, $"{name}: the untyped overload's layout");

                for (int i = 0; i < 5; i++)
                for (int j = 0; j < 2; j++)
                for (int k = 0; k < 4; k++)
                    reshaped.GetInt64(i, j, k).Should().Be((i * 2 + j) * 4 + k, $"{name} [{i},{j},{k}]");
            }
        }

        /// <summary>
        ///     The result is a view of the typed array (a write shows in the source), and a comparison's
        ///     <c>NDArray&lt;bool&gt;</c> result reshaped with a view's Shape holds its own 40 answers.
        /// </summary>
        [TestMethod]
        public void TypedReshape_WithAViewsShape_IsAViewOfItsOwnBuffer()
        {
            var view = np.arange(80).reshape(5, 2, 8)[":, :, :4"];

            NDArray<double> source = np.zeros(new Shape(40), NPTypeCode.Double).AsGeneric<double>() ?? throw new InvalidOperationException("float64");
            NDArray<double> reshaped = source.reshape(view.Shape);
            reshaped.SetAtIndex(7.5, 39);                             // [4, 1, 3]: the last element of the 40
            source.GetDouble(39).Should().Be(7.5, "the reshape is a view of the source");
            reshaped.SetAtIndex(2.5, 8);                              // [1, 0, 0]
            source.GetDouble(8).Should().Be(2.5);

            // NumPy: (np.arange(40) % 3 == 0).reshape((5, 2, 4)).sum() == 14, and element [4, 1, 3] is 39 % 3 == 0.
            NDArray<bool> mask = ((np.arange(40) % 3) == 0).reshape(view.Shape);
            mask.Shape.strides.Should().Equal(new long[] { 8, 4, 1 });
            np.count_nonzero(mask).Should().Be(14);
            mask.GetBoolean(4, 1, 3).Should().BeTrue();
            mask.GetBoolean(4, 1, 2).Should().BeFalse();
        }

        /// <summary>
        ///     A plain dimension list still costs nothing extra and still infers a missing dimension and reports a size
        ///     mismatch in NumPy's words.
        /// </summary>
        [TestMethod]
        public void TypedReshape_WithAPlainShape_IsUnchanged()
        {
            NDArray<int> typed = np.arange(12).astype(NPTypeCode.Int32).AsGeneric<int>() ?? throw new InvalidOperationException("int32");

            var plain = new Shape(3, 4);
            typed.reshape(plain).Shape.strides.Should().Equal(new long[] { 4, 1 });
            typed.reshape(new Shape(-1, 4)).Shape.dimensions.Should().Equal(new long[] { 3, 4 });
            typed.reshape(new Shape(2, -1)).Shape.dimensions.Should().Equal(new long[] { 2, 6 });

            // NumPy: np.arange(12).reshape((5, 2)) -> ValueError: cannot reshape array of size 12 into shape (5,2)
            Action tooMany = () => typed.reshape(new Shape(5, 2));
            tooMany.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 12 into shape (5,2)");
        }
    }
}
