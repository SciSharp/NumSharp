
using System;
using System.Linq;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged.Math
{
    [TestClass]
    public class np_mod_tests
    {
        #region Regular Tests
        
#if _REGEN
        %a = except(supported_dtypes, "NDArray", "Boolean")
        %foreach a
        [DataRow(NPTypeCode.Boolean, NPTypeCode.#1)]
#else
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Byte)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int16)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int32)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Int64)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Decimal)]
#endif
        [DataTestMethod]
        public void ModAllPossabilitiesBoolean(NPTypeCode ltc, NPTypeCode rtc)
        {
            var right = np.full(2, new Shape(5, 5), rtc);
            var left = np.full(3, new Shape(5, 5), ltc);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                Convert.ToInt32(val).Should().Be(rtc == NPTypeCode.Char ? '1' : 1);
                Console.WriteLine(val);
            }
        }

        [TestMethod]
        public void Mod_Two_Scalars()
        {
            var left = NDArray.Scalar(1d) % NDArray.Scalar(5d);
            left.GetDouble(0).Should().Be(1);
        }

        [TestMethod]
        public void Mod_ND_3_1_vs_1_3()
        {
            var left = (np.arange(3).reshape((3, 1)) + 1).astype(np.int32);
            var right = (np.arange(3).reshape((1, 3)) + 1).astype(np.int32);
            var ret = left % right;
            ret.size.Should().Be(9);
            ret.dtype.Should().Be<Int32>();
            ret.GetInt32(0, 0).Should().Be(0);
            ret.GetInt32(1, 1).Should().Be(0);
            ret.GetInt32(2, 2).Should().Be(0);
            ret.GetInt32(2, 1).Should().Be(1);
            ret.GetInt32(1, 2).Should().Be(2);

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Mod_ND_2_1_3_3_vs_1_3()
        {
            var left = np.arange(2 * 3 * 3).reshape((2, 1, 3, 3)).astype(np.float64) + 1;
            var right = np.arange(3).reshape((1, 3)).astype(np.float64) + 1;
            var ret = left % right;
            for (int i = 0; i < ret.size; i++) 
                Console.WriteLine(ret.GetAtIndex(i));

            ret.size.Should().Be(18);
            ret.dtype.Should().Be<double>();
            ret.GetDouble(0, 0, 0, 0).Should().Be(0);
            ret.GetDouble(0, 0, 1, 1).Should().Be(1);
            ret.GetDouble(0, 0, 2, 2).Should().Be(0);

            ret.GetDouble(1, 0, 0, 0).Should().Be(0);
            ret.GetDouble(1, 0, 1, 1).Should().Be(0);
            ret.GetDouble(1, 0, 2, 2).Should().Be(0);
        }

        [TestMethod]
        public void Mod_Rising()
        {
            var left = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                left.SetAtIndex<double>(i, i);
            }

            var right = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                right.SetAtIndex<double>(i, i);
            }

            var ret = left % right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Mod_RightScalar_Rising()
        {
            var left = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                left.SetAtIndex<double>(i, i);
            }

            var right = NDArray.Scalar(1d);
            var ret = left % right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        [TestMethod]
        public void Mod_LeftScalar_Rising()
        {
            var left = NDArray.Scalar(1d);
            var right = np.zeros(new Shape(5, 5), np.float64);
            for (int i = 0; i < 25; i++)
            {
                right.SetAtIndex<double>(i, i);
            }

            var ret = left % right;
            ret.Should().BeInAscendingOrder();

            for (int i = 0; i < ret.size; i++)
            {
                Console.WriteLine(ret.GetAtIndex(i));
            }
        }

        #endregion

#if _REGEN
        %a = ["Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Double","Single","Decimal"]
        %b = [true,"1","1","1","1","1u","1L","1UL","1d","1f","1m"]
        %mod = "%"
        %foreach forevery(a,a,true), forevery(b,b,true)%
        [TestMethod]
        public void Mod_#1_To_#2()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.#1);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.#2);
            var ret = left #(mod) right;

            for (int i = 0; i < ret.size; i++)
            {
                |#res = str(b[a.IndexOf("#2")]
                var val = ret.GetAtIndex(i);
                val.Should().Be(#(#2 == "Boolean" | "0" | (#1=="Boolean"|"1"|res) ));
                Console.WriteLine(val);
            }
        }
        %
#else

        [TestMethod]
        public void Mod_Boolean_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Boolean_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Boolean);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Byte_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Byte);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int16_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt16_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt16);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int32_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt32_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt32);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Int64_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Int64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Double()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Double);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1d);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_UInt64_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.UInt64);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Single()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Single);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1f);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Double_To_Decimal()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Double);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Decimal);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1m);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_Boolean()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Boolean);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(0);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_Byte()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Byte);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_Int16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_UInt16()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt16);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_Int32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_UInt32()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt32);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1u);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_Int64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.Int64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1L);
                Console.WriteLine(val);
            }
        }
        [TestMethod]
        public void Mod_Single_To_UInt64()
        {
            var left = np.full(4, new Shape(5, 5), NPTypeCode.Single);
            var right = np.full(3, new Shape(5, 5), NPTypeCode.UInt64);
            var ret = left % right;

            for (int i = 0; i < ret.size; i++)
            {
                var val = ret.GetAtIndex(i);
                val.Should().Be(1UL);
                Console.WriteLine(val);
            }
        }
#endif
    }
}
