using System;

namespace NumSharp.Tests.Statistics
{
    /// <summary>
    ///     np.average's statements against NumPy 2.4.2 (<c>numpy/lib/_function_base_impl.py</c>, <c>average</c> and
    ///     <c>_weights_are_valid</c>), one test per D4 fix of the np-layer dtype plan's P2 family: NumPy's exception types,
    ///     NumPy's promotion of text weights and the refusals that follow it, the count np.average_returned gives without
    ///     weights (a scalar beside a scalar mean).
    /// </summary>
    [TestClass]
    public class np_average_NumPyPort_Tests
    {
        // ------------------------------------------------------------------------------- _weights_are_valid's types

        [TestMethod]
        public void WeightsOfAnotherShape_WithoutAnAxis_IsTypeError_OnEveryOperandKind()
        {
            // NumPy: `if axis is None: raise TypeError("Axis must be specified when shapes of a and weights differ.")`,
            // before the result dtype is computed - so numeric, U, S and StringDType operands all raise it.
            const string text = "Axis must be specified when shapes of a and weights differ.";
            var w = np.ones(new Shape(3));
            new Action(() => np.average(np.ones(new Shape(2, 3)), (int?)null, w)).Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }), (int?)null, w))
                .Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }).astype(np.dtype("S1")), (int?)null, w))
                .Should().ThrowExactly<TypeError>().WithMessage(text);
            new Action(() => np.average(np.array(2.5), (int?)null, np.ones(new Shape(1)))).Should().ThrowExactly<TypeError>().WithMessage(text);
        }

        [TestMethod]
        public void WeightsNotOfTheAxesShape_IsValueError_TheEmptyTupleIncluded()
        {
            // NumPy: `if wgt.shape != tuple(a.shape[ax] for ax in axis): raise ValueError(...)`. An empty tuple is an axis
            // given (not None): weights of another shape fail this check, also for a 0-d operand, whose only tuple is ().
            const string text = "Shape of weights must be consistent with shape of a along specified axis.";
            new Action(() => np.average(np.ones(new Shape(2, 3)), 0, np.ones(new Shape(3)))).Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(new[,] { { "a", "b", "c" }, { "d", "e", "f" } }), 0, np.ones(new Shape(3))))
                .Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(2.5), Array.Empty<int>(), np.ones(new Shape(1)))).Should().ThrowExactly<ValueError>().WithMessage(text);
            new Action(() => np.average(np.array(new[] { "a" }).astype(np.dtype("T")).reshape(Shape.Scalar), Array.Empty<int>(), np.ones(new Shape(1))))
                .Should().ThrowExactly<ValueError>().WithMessage(text);
        }

        // ------------------------------------------------------------------------------- the count without weights

        [TestMethod]
        public void ReturnedCount_WithoutWeights_IsAScalarBesideAScalarMean()
        {
            // scl = avg_as_array.dtype.type(a.size / avg_as_array.size): NumPy's scalar beside a scalar mean (every axis, as
            // None or as a tuple of them), a fresh array (np.broadcast_to(scl, avg.shape).copy()) beside a mean with
            // dimensions. It was a writeable 0-d array beside a scalar mean.
            using var x = np.arange(6.0).reshape(2, 3);
            foreach (var (label, call) in new (string, Func<(NDArray, NDArray)>)[]
                     {
                         ("axis=None", () => np.average_returned(x)),
                         ("axis=(0, 1)", () => np.average_returned(x, new[] { 0, 1 })),
                         ("an int32 operand", () => np.average_returned(np.arange(6).reshape(2, 3))),
                     })
            {
                var (avg, scl) = call();
                using (avg)
                using (scl)
                {
                    scl.ndim.Should().Be(0, label);
                    scl.dtype.Should().Be(np.float64, label);
                    scl.GetDouble().Should().Be(6.0, label);
                    scl.flags.writeable.Should().BeFalse($"{label}: NumPy's count is a scalar");
                    avg.flags.writeable.Should().BeFalse($"{label}: and so is the mean");
                }
            }

            var (avg0, scl0) = np.average_returned(x, 0);
            using (avg0)
            using (scl0)
            {
                scl0.shape.Should().Equal(3);
                scl0.flags.writeable.Should().BeTrue("the broadcast copy is a fresh array");
                scl0.flags.owndata.Should().BeTrue();
                scl0.ToArray<double>().Should().Equal(2.0, 2.0, 2.0);
            }

            var (avgK, sclK) = np.average_returned(x, keepdims: true);
            using (avgK)
            using (sclK)
            {
                sclK.shape.Should().Equal(1, 1);
                sclK.flags.writeable.Should().BeTrue("keepdims gives the mean dimensions, so the count is broadcast and copied");
                sclK.GetDouble(0, 0).Should().Be(6.0);
            }

            // The mean's dtype: a complex128 mean gives a complex128 count, a float16 mean a float16 one.
            var (avgC, sclC) = np.average_returned(np.ones(new Shape(2, 2), np.complex128));
            using (avgC)
            using (sclC)
            {
                sclC.dtype.Should().Be(np.complex128);
                sclC.flags.writeable.Should().BeFalse();
            }
        }

        [TestMethod]
        public void ReturnedCount_OfAStringDTypeNaMean_IsAScalar()
        {
            // A NaN-like NA sum is the one StringDType mean that returns (np.float64(nan)); NumPy's count beside it is
            // np.float64(3.0), a scalar.
            using var a = np.array(new object[] { "a", double.NaN, "b" }, np.dtypes.StringDType(na_object: double.NaN));
            var (avg, scl) = np.average_returned(a);
            using (avg)
            using (scl)
            {
                double.IsNaN(avg.GetDouble()).Should().BeTrue();
                avg.flags.writeable.Should().BeFalse();
                scl.dtype.Should().Be(np.float64);
                scl.GetDouble().Should().Be(3.0);
                scl.flags.writeable.Should().BeFalse("NumPy's count is a scalar");
            }
        }

        // ------------------------------------------------------------------------------- result_type over descriptors

        /// <summary>NumPy's refusal of a sized string <c>dtype=</c> in a ufunc (the weight sum's, for every sized text
        ///     result dtype).</summary>
        private const string GeneralDTypeOnly = "The `dtype` and `signature` arguments to ufuncs only select the general DType and not details "
            + "such as the byte order or time unit. You can avoid this error by using the scalar types `np.float64` or the dtype string notation.";

        /// <summary>Two-by-three text "1" in the given text dtype.</summary>
        /// <param name="dt">U5, S5 or T.</param>
        /// <returns>The array.</returns>
        private static NDArray Ones(string dt) => np.array(new[,] { { "1", "1", "1" }, { "1", "1", "1" } }).astype(np.dtype(dt));

        [TestMethod]
        public void SizedTextWeights_BesideANumber_PromoteAndTheWeightSumRefuses()
        {
            // result_type(float64, <U5) is <U32 (int32 / bool operands add 'f8': still <U32), a sized string the weight sum
            // wgt.sum(dtype=result_dtype) refuses by its class - where NumSharp's numeric table had no text entry
            // (NotSupportedException).
            foreach (var adt in new[] { "float64", "int32", "bool", "float16", "complex128" })
            foreach (var wdt in new[] { "U5", "S5" })
                new Action(() => np.average(np.ones(new Shape(2, 3), np.dtype(adt)), (int?)null, Ones(wdt)))
                    .Should().ThrowExactly<TypeError>($"{adt} beside {wdt}").WithMessage(GeneralDTypeOnly);
        }

        [TestMethod]
        public void StringDTypeWeights_BesideANumber_HaveNoCommonDType()
        {
            // np.result_type(a.dtype, wgt.dtype[, 'f8']): StringDType promotes with no number. The text lists the sequence's
            // DTypes in NumPy's order, 'f8' last for an integer or bool operand.
            new Action(() => np.average(np.ones(new Shape(2, 3)), (int?)null, Ones("T")))
                .Should().ThrowExactly<DTypePromotionError>().WithMessage(
                    "The DType <class 'numpy.dtypes.Float64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>. This means "
                    + "that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is "
                    + "`object`. The full list of DTypes is: (<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.StringDType'>)");
            new Action(() => np.average(np.ones(new Shape(2, 3), np.int32), (int?)null, Ones("T")))
                .Should().ThrowExactly<DTypePromotionError>().WithMessage(
                    "The DType <class 'numpy.dtypes.Float64DType'> could not be promoted by <class 'numpy.dtypes.StringDType'>. This means "
                    + "that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is "
                    + "`object`. The full list of DTypes is: (<class 'numpy.dtypes.Int32DType'>, <class 'numpy.dtypes.StringDType'>, "
                    + "<class 'numpy.dtypes.Float64DType'>)");
        }

        [TestMethod]
        public void FixedWidthTextOperands_TakeNumPysPromotionOfTheirWeights()
        {
            // A U operand beside StringDType weights promotes to StringDType, an instance the weight sum refuses as a new
            // user DType; an S operand beside StringDType has no common DType; U or S beside numbers is a sized string.
            new Action(() => np.average(Ones("U5"), (int?)null, Ones("T"))).Should().ThrowExactly<TypeError>()
                .WithMessage("Cannot pass a new user DType instance to the `dtype` or `signature` arguments of ufuncs. Pass the DType class instead.");
            new Action(() => np.average(Ones("S5"), (int?)null, Ones("T"))).Should().ThrowExactly<DTypePromotionError>()
                .WithMessage("The DType <class 'numpy.dtypes.BytesDType'> could not be promoted by <class 'numpy.dtypes.StringDType'>.*");
            new Action(() => np.average(Ones("S5"), (int?)null, np.ones(new Shape(2, 3)))).Should().ThrowExactly<TypeError>()
                .WithMessage(GeneralDTypeOnly);
            new Action(() => np.average(Ones("U5"), (int?)null, Ones("S5"))).Should().ThrowExactly<TypeError>()
                .WithMessage(GeneralDTypeOnly);
        }
    }
}
