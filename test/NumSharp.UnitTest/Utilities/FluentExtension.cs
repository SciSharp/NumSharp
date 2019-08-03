using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using FluentAssertions;
using FluentAssertions.Common;
using FluentAssertions.Execution;
using FluentAssertions.Primitives;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Utilities
{
    public static class FluentExtension
    {
        public static ShapeAssertions Should(this Shape shape)
        {
            return new ShapeAssertions(shape);
        }

        public static NDArrayAssertions Should(this NDArray arr)
        {
            return new NDArrayAssertions(arr);
        }

        public static NDArrayAssertions Should(this UnmanagedStorage arr)
        {
            return new NDArrayAssertions(arr);
        }
    }

    public class ShapeAssertions : ReferenceTypeAssertions<Shape, ShapeAssertions>
    {
        public ShapeAssertions(Shape instance)
        {
            Subject = instance;
        }

        protected override string Identifier => "shape";

        public AndConstraint<ShapeAssertions> BeOfSize(int size, string because = null, params object[] becauseArgs)
        {
            Subject.size.Should().Be(size, because, becauseArgs);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeShaped(params int[] dimensions)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(dimensions));

            Subject.dimensions.Should().BeEquivalentTo(dimensions);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> Be(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(Subject.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeEquivalentTo(int? size = null, int? ndim = null, ITuple shape = null)
        {
            if (size.HasValue)
            {
                BeOfSize(size.Value, null);
            }

            if (ndim.HasValue)
                HaveNDim(ndim.Value);

            if (shape != null)
                for (int i = 0; i < shape.Length; i++)
                {
                    Subject.dimensions[i].Should().Be((int)shape[i]);
                }

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBe(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(!Subject.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> HaveNDim(int ndim)
        {
            Subject.dimensions.Length.Should().Be(ndim);
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeSliced()
        {
            Subject.IsSliced.Should().BeTrue();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeScalar()
        {
            Subject.IsScalar.Should().BeTrue();
            return new AndConstraint<ShapeAssertions>(this);
        }


        public AndConstraint<ShapeAssertions> NotBeSliced()
        {
            Subject.IsSliced.Should().BeFalse();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> NotBeScalar()
        {
            Subject.IsScalar.Should().BeFalse();
            return new AndConstraint<ShapeAssertions>(this);
        }

        public AndConstraint<ShapeAssertions> BeNDim(int ndim)
        {
            Subject.dimensions.Length.Should().Be(ndim);
            return new AndConstraint<ShapeAssertions>(this);
        }
    }

    public class NDArrayAssertions : ReferenceTypeAssertions<NDArray, NDArrayAssertions>
    {
        public NDArrayAssertions(NDArray instance)
        {
            Subject = instance;
        }

        public NDArrayAssertions(UnmanagedStorage instance)
        {
            Subject = new NDArray(instance);
        }

        protected override string Identifier => "shape";

        public AndConstraint<NDArrayAssertions> BeOfSize(int size, string because = null, params object[] becauseArgs)
        {
            Subject.size.Should().Be(size, because, becauseArgs);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeShaped(params int[] dimensions)
        {
            if (dimensions == null)
                throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(dimensions));

            Subject.Shape.dimensions.Should().BeEquivalentTo(dimensions);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeShaped(int? size = null, int? ndim = null, ITuple shape = null)
        {
            if (size.HasValue)
            {
                BeOfSize(size.Value, null);
            }

            if (ndim.HasValue)
                HaveNDim(ndim.Value);

            if (shape != null)
                for (int i = 0; i < shape.Length; i++)
                {
                    Subject.Shape.dimensions[i].Should().Be((int)shape[i]);
                }

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeShaped(Shape shape, string because = null, params object[] becauseArgs)
        {
            Execute.Assertion
                .BecauseOf(because, becauseArgs)
                .ForCondition(!Subject.Shape.Equals(shape))
                .FailWith($"Expected shape to be {shape.ToString()} but got {Subject.ToString()}");

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> HaveNDim(int ndim)
        {
            Subject.Shape.dimensions.Length.Should().Be(ndim);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeSliced()
        {
            Subject.Shape.IsSliced.Should().BeTrue();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeScalar()
        {
            Subject.Shape.IsScalar.Should().BeTrue();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeScalar(object value)
        {
            Subject.Shape.IsScalar.Should().BeTrue();
            Subject.GetValue().Should().Be(value);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType(NPTypeCode typeCode)
        {
            Subject.typecode.Should().Be(typeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType(Type typeCode)
        {
            Subject.dtype.Should().Be(typeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfType<T>()
        {
            Subject.typecode.Should().Be(InfoOf<T>.NPTypeCode);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeSliced()
        {
            Subject.Shape.IsSliced.Should().BeFalse();
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> NotBeScalar()
        {
            Subject.Shape.IsScalar.Should().BeFalse();
            return new AndConstraint<NDArrayAssertions>(this);
        }


        public AndConstraint<NDArrayAssertions> BeNDim(int ndim)
        {
            Subject.Shape.dimensions.Length.Should().Be(ndim);
            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfValues(NDArray expected)
        {
            Execute.Assertion
                .ForCondition(np.array_equal(Subject, expected))
                .FailWith($"Expected the subject and other ndarray to be equals.");

            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> Be(NDArray expected)
        {
            return BeOfValues(expected);
        }

        public AndConstraint<NDArrayAssertions> BeOfValues(params object[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            Subject.size.Should().Be(values.Length, "the method BeOfValuesApproximately also confirms the sizes are matching with given values.");

#if _REGEN
            #region Compute
		    switch (Subject.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = Subject.AsIterator<#2>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");
                        
                        var expected = Convert.To#1(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: #1).", expected, nextval, i);
                    }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (Subject.typecode)
            {
                case NPTypeCode.Boolean:
                {
                    var iter = Subject.AsIterator<bool>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToBoolean(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Boolean).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    var iter = Subject.AsIterator<byte>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToByte(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Byte).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var iter = Subject.AsIterator<short>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int16).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var iter = Subject.AsIterator<ushort>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt16).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var iter = Subject.AsIterator<int>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int32).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var iter = Subject.AsIterator<uint>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt32).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var iter = Subject.AsIterator<long>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int64).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var iter = Subject.AsIterator<ulong>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt64).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    var iter = Subject.AsIterator<char>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToChar(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Char).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    var iter = Subject.AsIterator<double>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDouble(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Double).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    var iter = Subject.AsIterator<float>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToSingle(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Single).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var iter = Subject.AsIterator<decimal>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDecimal(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Decimal).", expected, nextval, i);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif


            return new AndConstraint<NDArrayAssertions>(this);
        }

        public AndConstraint<NDArrayAssertions> BeOfValuesApproximately(double sensitivity, params object[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            Subject.size.Should().Be(values.Length, "the method BeOfValuesApproximately also confirms the sizes are matching with given values.");

#if _REGEN
            #region Compute
		    switch (Subject.typecode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1: 
                {
                    var iter = Subject.AsIterator<#2>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");
                        
                        var expected = Convert.To#1(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: #1).", expected, nextval, i);
                    }
                    break;
                }
			    %
			    default:
				    throw new NotSupportedException();
		    }
            #endregion
#else

            #region Compute

            switch (Subject.typecode)
            {
                case NPTypeCode.Boolean:
                {
                    var iter = Subject.AsIterator<bool>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToBoolean(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(expected == nextval)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Boolean).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Byte:
                {
                    var iter = Subject.AsIterator<byte>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToByte(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Byte).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var iter = Subject.AsIterator<short>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int16).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var iter = Subject.AsIterator<ushort>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt16(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt16).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var iter = Subject.AsIterator<int>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int32).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var iter = Subject.AsIterator<uint>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt32(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt32).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var iter = Subject.AsIterator<long>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Int64).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var iter = Subject.AsIterator<ulong>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToUInt64(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs((double)(expected - nextval)) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: UInt64).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Char:
                {
                    var iter = Subject.AsIterator<char>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToChar(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Char).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Double:
                {
                    var iter = Subject.AsIterator<double>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDouble(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Double).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Single:
                {
                    var iter = Subject.AsIterator<float>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToSingle(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Single).", expected, nextval, i);
                    }

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var iter = Subject.AsIterator<decimal>();
                    var next = iter.MoveNext;
                    var hasnext = iter.HasNext;
                    for (int i = 0; i < values.Length; i++)
                    {
                        Execute.Assertion
                            .ForCondition(hasnext())
                            .FailWith($"Expected the NDArray to have atleast {values.Length} but in fact it has size of {i}.");

                        var expected = Convert.ToDecimal(values[i]);
                        var nextval = next();

                        Execute.Assertion
                            .ForCondition(Math.Abs(expected - nextval) <= (decimal)sensitivity)
                            .FailWith("Expected NDArray's {2}th value to be {0}, but found {1} (dtype: Decimal).", expected, nextval, i);
                    }

                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

#endif


            return new AndConstraint<NDArrayAssertions>(this);
        }
    }
}
