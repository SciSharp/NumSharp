using System;
using System.Collections.Generic;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class np_find_common_type_test
    {
        [TestMethod]
        public void Case1()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.int64, np.float64});
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case2()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case3()
        {
            var r = np.find_common_type(new[] {np.float32}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case4()
        {
            var r = np.find_common_type(new[] {"f4", "f4", "i4",}, new[] {"c8"});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case5()
        {
            var r = np.find_common_type(new[] {"f4", "f4", "i4",}, new[] {"c8"});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case6()
        {
            var r = np.find_common_type(new[] {np.int32, np.float32});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case7()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case8()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.int32, np.float64});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case9()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.int32, np.float32});
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case10()
        {
            var r = np.find_common_type(new[] {np.int32, np.float64}, new[] {np.complex64});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case11()
        {
            var r = np.find_common_type(new[] {np.uint8, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case12()
        {
            var r = np.find_common_type(new[] {np.@byte, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case13()
        {
            var r = np.find_common_type(new[] {np.float32, np.float32}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case14()
        {
            var r = np.find_common_type(new[] {np.float32, np.@byte}, new Type[0]);
            r.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Case15()
        {
            var r = np.find_common_type(new[] {np.float64, np.float64}, new Type[0]);
            r.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Case17()
        {
            var r = np.find_common_type(new[] {np.@byte, np.@byte}, new Type[0]);
            r.Should().Be(NPTypeCode.Byte);
        }

        [TestMethod]
        public void Case18()
        {
            var r = np.find_common_type(new[] {np.complex128, np.@double}, new Type[0]);
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case19()
        {
            var r = np.find_common_type(new[] {np.complex128, np.complex128}, new Type[0]);
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case20()
        {
            var r = np.find_common_type(new[] {np.complex128, np.complex128}, new[] {np.@double});
            r.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void Case21()
        {
            var r = np.find_common_type(new[] {np.@decimal, np.@double}, new NPTypeCode[0]);
            r.Should().Be(NPTypeCode.Decimal);
        }

        [TestMethod]
        public void Case22()
        {
            var r = np.find_common_type(new[] {np.int16, np.int64}, new NPTypeCode[0]);
            r.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Case23()
        {
            var r = np.find_common_type(new[] {np.@char, np.int16}, new NPTypeCode[0]);
            r.Should().Be(NPTypeCode.Int16);
        }

        [TestMethod, Ignore]
        public void gen_typecode_map()
        {
            var r = np.find_common_type(new[] {np.float32, np.float64}, new NPTypeCode[0]);
            r.Should().Be(NPTypeCode.Double);
            var dict = new Dictionary<(Type, Type), Type>();
            dict.Add((np.@bool, np.@bool), np.@bool);
            dict.Add((np.@bool, np.uint8), np.uint8);
            dict.Add((np.@bool, np.int16), np.int16);
            dict.Add((np.@bool, np.uint16), np.uint16);
            dict.Add((np.@bool, np.int32), np.int32);
            dict.Add((np.@bool, np.uint32), np.uint32);
            dict.Add((np.@bool, np.int64), np.int64);
            dict.Add((np.@bool, np.uint64), np.uint64);
            dict.Add((np.@bool, np.float32), np.float32);
            dict.Add((np.@bool, np.float64), np.float64);
            dict.Add((np.@bool, np.complex64), np.complex64);

            dict.Add((np.uint8, np.@bool), np.uint8);
            dict.Add((np.uint8, np.uint8), np.uint8);
            dict.Add((np.uint8, np.int16), np.int16);
            dict.Add((np.uint8, np.uint16), np.uint8);
            dict.Add((np.uint8, np.int32), np.int32);
            dict.Add((np.uint8, np.uint32), np.uint8);
            dict.Add((np.uint8, np.int64), np.int64);
            dict.Add((np.uint8, np.uint64), np.uint8);
            dict.Add((np.uint8, np.float32), np.float32);
            dict.Add((np.uint8, np.float64), np.float64);
            dict.Add((np.uint8, np.complex64), np.complex64);

            dict.Add((np.int16, np.@bool), np.int16);
            dict.Add((np.int16, np.uint8), np.int16);
            dict.Add((np.int16, np.int16), np.int16);
            dict.Add((np.int16, np.uint16), np.int16);
            dict.Add((np.int16, np.int32), np.int16);
            dict.Add((np.int16, np.uint32), np.int16);
            dict.Add((np.int16, np.int64), np.int16);
            dict.Add((np.int16, np.uint64), np.int16);
            dict.Add((np.int16, np.float32), np.float32);
            dict.Add((np.int16, np.float64), np.float64);
            dict.Add((np.int16, np.complex64), np.complex64);

            dict.Add((np.uint16, np.@bool), np.uint16);
            dict.Add((np.uint16, np.uint8), np.uint16);
            dict.Add((np.uint16, np.int16), np.int32);
            dict.Add((np.uint16, np.uint16), np.uint16);
            dict.Add((np.uint16, np.int32), np.int32);
            dict.Add((np.uint16, np.uint32), np.uint16);
            dict.Add((np.uint16, np.int64), np.int64);
            dict.Add((np.uint16, np.uint64), np.uint16);
            dict.Add((np.uint16, np.float32), np.float32);
            dict.Add((np.uint16, np.float64), np.float64);
            dict.Add((np.uint16, np.complex64), np.complex64);

            dict.Add((np.int32, np.@bool), np.int32);
            dict.Add((np.int32, np.uint8), np.int32);
            dict.Add((np.int32, np.int16), np.int32);
            dict.Add((np.int32, np.uint16), np.int32);
            dict.Add((np.int32, np.int32), np.int32);
            dict.Add((np.int32, np.uint32), np.int32);
            dict.Add((np.int32, np.int64), np.int32);
            dict.Add((np.int32, np.uint64), np.int32);
            dict.Add((np.int32, np.float32), np.float64);
            dict.Add((np.int32, np.float64), np.float64);
            dict.Add((np.int32, np.complex64), np.complex128);

            dict.Add((np.uint32, np.@bool), np.uint32);
            dict.Add((np.uint32, np.uint8), np.uint32);
            dict.Add((np.uint32, np.int16), np.int64);
            dict.Add((np.uint32, np.uint16), np.uint32);
            dict.Add((np.uint32, np.int32), np.int64);
            dict.Add((np.uint32, np.uint32), np.uint32);
            dict.Add((np.uint32, np.int64), np.int64);
            dict.Add((np.uint32, np.uint64), np.uint32);
            dict.Add((np.uint32, np.float32), np.float64);
            dict.Add((np.uint32, np.float64), np.float64);
            dict.Add((np.uint32, np.complex64), np.complex128);

            dict.Add((np.int64, np.@bool), np.int64);
            dict.Add((np.int64, np.uint8), np.int64);
            dict.Add((np.int64, np.int16), np.int64);
            dict.Add((np.int64, np.uint16), np.int64);
            dict.Add((np.int64, np.int32), np.int64);
            dict.Add((np.int64, np.uint32), np.int64);
            dict.Add((np.int64, np.int64), np.int64);
            dict.Add((np.int64, np.uint64), np.int64);
            dict.Add((np.int64, np.float32), np.float64);
            dict.Add((np.int64, np.float64), np.float64);
            dict.Add((np.int64, np.complex64), np.complex128);

            dict.Add((np.uint64, np.@bool), np.uint64);
            dict.Add((np.uint64, np.uint8), np.uint64);
            dict.Add((np.uint64, np.int16), np.float64);
            dict.Add((np.uint64, np.uint16), np.uint64);
            dict.Add((np.uint64, np.int32), np.float64);
            dict.Add((np.uint64, np.uint32), np.uint64);
            dict.Add((np.uint64, np.int64), np.float64);
            dict.Add((np.uint64, np.uint64), np.uint64);
            dict.Add((np.uint64, np.float32), np.float64);
            dict.Add((np.uint64, np.float64), np.float64);
            dict.Add((np.uint64, np.complex64), np.complex128);

            dict.Add((np.float32, np.@bool), np.float32);
            dict.Add((np.float32, np.uint8), np.float32);
            dict.Add((np.float32, np.int16), np.float32);
            dict.Add((np.float32, np.uint16), np.float32);
            dict.Add((np.float32, np.int32), np.float32);
            dict.Add((np.float32, np.uint32), np.float32);
            dict.Add((np.float32, np.int64), np.float32);
            dict.Add((np.float32, np.uint64), np.float32);
            dict.Add((np.float32, np.float32), np.float32);
            dict.Add((np.float32, np.float64), np.float32);
            dict.Add((np.float32, np.complex64), np.complex64);

            dict.Add((np.float64, np.@bool), np.float64);
            dict.Add((np.float64, np.uint8), np.float64);
            dict.Add((np.float64, np.int16), np.float64);
            dict.Add((np.float64, np.uint16), np.float64);
            dict.Add((np.float64, np.int32), np.float64);
            dict.Add((np.float64, np.uint32), np.float64);
            dict.Add((np.float64, np.int64), np.float64);
            dict.Add((np.float64, np.uint64), np.float64);
            dict.Add((np.float64, np.float32), np.float64);
            dict.Add((np.float64, np.float64), np.float64);
            dict.Add((np.float64, np.complex64), np.complex128);

            dict.Add((np.complex64, np.@bool), np.complex64);
            dict.Add((np.complex64, np.uint8), np.complex64);
            dict.Add((np.complex64, np.int16), np.complex64);
            dict.Add((np.complex64, np.uint16), np.complex64);
            dict.Add((np.complex64, np.int32), np.complex64);
            dict.Add((np.complex64, np.uint32), np.complex64);
            dict.Add((np.complex64, np.int64), np.complex64);
            dict.Add((np.complex64, np.uint64), np.complex64);
            dict.Add((np.complex64, np.float32), np.complex64);
            dict.Add((np.complex64, np.float64), np.complex64);
            dict.Add((np.complex64, np.complex64), np.complex64);


#if _REGEN
            %a = ["bool","uint8","int16","uint16","int32","uint32","int64","uint64","float32","float64","complex64"]
            %foreach forevery(a,a,false)%
                print("{(np."+str(np.#1.__name__) + ", " + "np." + str(np.#2.__name__) + "),  " + "np." + str(np.find_common_type([np.#1], [np.#2]))+"},")
            %
#else

#endif
        }

        [TestMethod]
        [Ignore]
        public void gen_all_possible_combinations()
        {
#if _REGEN
            %a = supported_dtypes
            %b = supported_dtypes_lowercase

            %foreach forevery(b,b,false), forevery(a,a, false)%
            Console.WriteLine($"#3 <> #4  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.#3, NPTypeCode.#4}, Array.Empty<NPTypeCode>()));
            %
#else

            Console.WriteLine($"Boolean <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Boolean <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Boolean, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Byte <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Byte, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int16 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int16, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt16 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt16, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int32 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int32, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt32 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt32, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Int64 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Int64, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"UInt64 <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.UInt64, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Char <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Char, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Double <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Double, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Single <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Single, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Boolean  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Boolean}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Byte  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Byte}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Int16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Int16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> UInt16  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.UInt16}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Int32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Int32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> UInt32  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.UInt32}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Int64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Int64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> UInt64  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.UInt64}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Char  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Char}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Double  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Double}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Single  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Single}, Array.Empty<NPTypeCode>()));
            Console.WriteLine($"Decimal <> Decimal  ==  " + np._FindCommonType(new NPTypeCode[] {NPTypeCode.Decimal, NPTypeCode.Decimal}, Array.Empty<NPTypeCode>()));
#endif
        }
    }
}
