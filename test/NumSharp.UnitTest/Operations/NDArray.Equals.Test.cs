using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayEqualsTest
    {
        [TestMethod]
        public void IntTwo1D_NDArrayEquals()
        {
            var np0 = new NDArray(new[] {0, 0, 0, 0}, new Shape(4));
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));
            var np2 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 == np2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, true, true, true}, np3.Data<bool>()));
            var np3S = np.array_equal(np1, np2);
            Assert.IsTrue(np3S);

            var np4 = np0 == np2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {false, false, false, false}, np4.Data<bool>()));
            var np4S = np.array_equal(np0, np2);
            Assert.IsFalse(np4S);
        }

        [TestMethod]
        public void IntAnd1D_NDArrayEquals()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np2 = np1 == 2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] {false, true, false, false}, np2.Data<bool>()));
        }

        [TestMethod]
        public void IntTwo2D_NDArrayEquals()
        {
            var np1 = new NDArray(typeof(int), new Shape(2, 3));
            np1.ReplaceData(new[] {1, 2, 3, 4, 5, 6});

            var np2 = new NDArray(typeof(int), new Shape(2, 3));
            np2.ReplaceData(new[] {1, 2, 3, 4, 5, 6});

            var np3 = np1 == np2;

            // expected
            var np3S = np.array_equal(np1, np2);
            Assert.IsTrue(np3S);
            var np4 = new bool[] {true, true, true, true, true, true};
            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<bool>(), np4));


            var np5 = new NDArray(typeof(int), new Shape(2, 3));
            np5.ReplaceData(new[] {0, 0, 0, 0, 0, 0});

            var np6 = np1 == np5;
            // expected
            var np6S = np.array_equal(np1, np5);
            Assert.IsFalse(np6S);
            var np7 = new bool[] {false, false, false, false, false, false,};
            Assert.IsTrue(Enumerable.SequenceEqual(np6.Data<bool>(), np7));
        }

        [TestMethod]
        public void IntAnd2D_NDArrayEquals()
        {
            var np1 = new NDArray(typeof(int), new Shape(2, 3));
            np1.ReplaceData(new[] {1, 2, 3, 4, 5, 6});

            var result = np1 == 2;
            var results = result.Data<bool>();
            // expected
            var np3 = new bool[] {false, true, false, false, false, false};
            Assert.IsTrue(results.SequenceEqual(np3));
        }

#if _REGEN
        %a = except(supported_dtypes, "NDArray")
        %b = [true,"1","1","1","1","1u","1L","1UL","1","1d","1f","1m"]
        %foreach forevery(a,a,true), forevery(b,b,true)%
        [TestMethod]
        public void Compare_#1_To_#2()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.#1.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.#2.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        %
#else

        [TestMethod]
        public void Compare_Boolean_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Boolean_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Byte_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int16_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt16_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int32_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt32_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Int64_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_UInt64_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Double()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Char_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Int16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_UInt16()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt16.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Int32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_UInt32()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt32.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Int64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Int64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_UInt64()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.UInt64.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Char()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Char.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Single()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Double_To_Decimal()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Double.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Decimal.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Single_To_Boolean()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Boolean.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }
        
        [TestMethod]
        public void Compare_Single_To_Byte()
        {
            var left = np.array(new int[] {0, 0, 1, 1, 0, 0}, dtype: NPTypeCode.Single.AsType()).reshape(new Shape(3, 2));
            var right = np.array(new int[] {1, 0, 0, 1, 1, 0}, dtype: NPTypeCode.Byte.AsType()).reshape(new Shape(3, 2));
            NDArray<bool> ret = left == right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                //Convert.ToInt32(val).Should().Be(1);
                Console.WriteLine(val);
            }

            var a = ((NDArray)new int[] {0, 1, 0, 1, 0, 1}).astype(NPTypeCode.Boolean);
            ret.Array.Should().BeEquivalentTo(a.Array);
        }

        [TestMethod]
        public void EqualsNull()
        {
            NDArray nd = null;
            if (nd == null)
                Console.WriteLine("yes");
            else
                throw new NotSupportedException();
        }
        
#endif
    }
}
