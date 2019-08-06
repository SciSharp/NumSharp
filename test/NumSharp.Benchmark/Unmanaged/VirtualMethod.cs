using BenchmarkDotNet.Attributes;

namespace NumSharp.Benchmark.Unmanaged
{
    //|                                 Method |      Mean |     Error |    StdDev |    Median |       Min |       Max | Ratio | RatioSD |
    //|--------------------------------------- |----------:|----------:|----------:|----------:|----------:|----------:|------:|--------:|
    //|                             BasicClass |  2.542 us | 0.0209 us | 0.0299 us |  2.526 us |  2.516 us |  2.613 us |  1.00 |    0.00 |
    //|                        InterfacedClass |  2.535 us | 0.0116 us | 0.0173 us |  2.528 us |  2.517 us |  2.570 us |  1.00 |    0.01 |
    //|                       SealedInterfaced |  2.541 us | 0.0159 us | 0.0218 us |  2.530 us |  2.522 us |  2.592 us |  1.00 |    0.02 |
    //|                 VirtualInterfacedClass | 15.096 us | 0.0339 us | 0.0475 us | 15.078 us | 15.055 us | 15.243 us |  5.94 |    0.07 |
    //|                    SecondInherietClass | 15.201 us | 0.1137 us | 0.1701 us | 15.156 us | 14.937 us | 15.586 us |  5.98 |    0.11 |
    //|              SealedSecondInherietClass |  2.515 us | 0.0095 us | 0.0136 us |  2.519 us |  2.474 us |  2.531 us |  0.99 |    0.01 |
    //|           ViaInterface_InterfacedClass | 24.386 us | 0.1441 us | 0.2157 us | 24.445 us | 23.501 us | 24.589 us |  9.59 |    0.13 |
    //|          ViaInterface_SealedInterfaced | 24.445 us | 0.1720 us | 0.2296 us | 24.490 us | 23.493 us | 24.661 us |  9.63 |    0.14 |
    //|    ViaInterface_VirtualInterfacedClass | 24.324 us | 0.2366 us | 0.3541 us | 24.439 us | 23.529 us | 24.643 us |  9.57 |    0.21 |
    //|       ViaInterface_SecondInherietClass | 24.585 us | 0.0511 us | 0.0749 us | 24.591 us | 24.370 us | 24.706 us |  9.67 |    0.11 |
    //| ViaInterface_SealedSecondInherietClass | 24.198 us | 0.2929 us | 0.4384 us | 24.393 us | 23.486 us | 24.629 us |  9.51 |    0.24 |
    //|                               Abstract | 15.004 us | 0.0409 us | 0.0586 us | 15.001 us | 14.833 us | 15.103 us |  5.90 |    0.07 |
    //|                         AbstractSealed |  2.506 us | 0.0030 us | 0.0043 us |  2.507 us |  2.493 us |  2.511 us |  0.99 |    0.01 |
    //|                       ViaBase_Abstract | 14.965 us | 0.0358 us | 0.0525 us | 14.981 us | 14.808 us | 15.006 us |  5.89 |    0.08 |
    //|                 ViaBase_AbstractSealed | 14.973 us | 0.0367 us | 0.0538 us | 14.990 us | 14.815 us | 15.018 us |  5.89 |    0.07 |

    [SimpleJob(launchCount: 1, warmupCount: 10, targetCount: 30)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    public class VirtualMethod
    {
        static BasicClass basicClass;
        static InterfacedClass interfacedClass;
        static SealedInterfacedClass sealedInterfaced;
        static VirtualInterfacedClass virtualInterfacedClass;
        static SecondInherietClass secondInherietClass;
        static SealedSecondInherietClass sealedSecondInherietClass;

        static IInterface iinterfacedClass;
        static IInterface isealedInterfaced;
        static IInterface ivirtualInterfacedClass;
        static IInterface isecondInherietClass;
        static IInterface isealedSecondInherietClass;

        static OfAbstractClass ofAbstractClass;
        static SealedOfAbstractClass sealedOfAbstractClass;
        static AbstractClass viaBase_ofAbstractClass;
        static AbstractClass viaBase_sealedOfAbstractClass;

        [GlobalSetup]
        public void Setup()
        {
            basicClass = new BasicClass();
            isealedInterfaced = sealedInterfaced = new SealedInterfacedClass();
            iinterfacedClass = interfacedClass = new InterfacedClass();
            ivirtualInterfacedClass = virtualInterfacedClass = new VirtualInterfacedClass();
            isecondInherietClass = secondInherietClass = new SecondInherietClass();
            isealedSecondInherietClass = sealedSecondInherietClass = new SealedSecondInherietClass();
            viaBase_ofAbstractClass = ofAbstractClass = new OfAbstractClass();
            viaBase_sealedOfAbstractClass = sealedOfAbstractClass = new SealedOfAbstractClass();
        }

        [Benchmark(Baseline = true)]
        public void BasicClass()
        {
            for (int i = 0; i < 10000; i++)
                basicClass.func();
        }

        [Benchmark]
        public void InterfacedClass()
        {
            for (int i = 0; i < 10000; i++)
                interfacedClass.func();
        }

        [Benchmark]
        public void SealedInterfaced()
        {
            for (int i = 0; i < 10000; i++)
                sealedInterfaced.func();
        }

        [Benchmark]
        public void VirtualInterfacedClass()
        {
            for (int i = 0; i < 10000; i++)
                virtualInterfacedClass.func();
        }

        [Benchmark]
        public void SecondInherietClass()
        {
            for (int i = 0; i < 10000; i++)
                secondInherietClass.func();
        }

        [Benchmark]
        public void SealedSecondInherietClass()
        {
            for (int i = 0; i < 10000; i++)
                sealedSecondInherietClass.func();
        }

        [Benchmark]
        public void ViaInterface_InterfacedClass()
        {
            for (int i = 0; i < 10000; i++)
                iinterfacedClass.func();
        }

        [Benchmark]
        public void ViaInterface_SealedInterfaced()
        {
            for (int i = 0; i < 10000; i++)
                isealedInterfaced.func();
        }

        [Benchmark]
        public void ViaInterface_VirtualInterfacedClass()
        {
            for (int i = 0; i < 10000; i++)
                ivirtualInterfacedClass.func();
        }

        [Benchmark]
        public void ViaInterface_SecondInherietClass()
        {
            for (int i = 0; i < 10000; i++)
                isecondInherietClass.func();
        }

        [Benchmark]
        public void ViaInterface_SealedSecondInherietClass()
        {
            for (int i = 0; i < 10000; i++)
                isealedSecondInherietClass.func();
        }

        [Benchmark]
        public void Abstract()
        {
            for (int i = 0; i < 10000; i++)
                ofAbstractClass.func();
        }

        [Benchmark]
        public void AbstractSealed()
        {
            for (int i = 0; i < 10000; i++)
                sealedOfAbstractClass.func();
        }

        [Benchmark]
        public void ViaBase_Abstract()
        {
            for (int i = 0; i < 10000; i++)
                viaBase_ofAbstractClass.func();
        }

        [Benchmark]
        public void ViaBase_AbstractSealed()
        {
            for (int i = 0; i < 10000; i++)
                viaBase_sealedOfAbstractClass.func();
        }
    }

    public class BasicClass
    {
        public int func()
        {
            return 1;
        }
    }

    public interface IInterface
    {
        int func();
    }

    public class InterfacedClass : IInterface
    {
        public int func()
        {
            return 1;
        }
    }

    public class VirtualInterfacedClass : IInterface
    {
        public virtual int func()
        {
            return 1;
        }
    }

    public sealed class SealedInterfacedClass : IInterface
    {
        public int func()
        {
            return 1;
        }
    }

    public class SecondInherietClass : VirtualInterfacedClass
    {
        public override int func()
        {
            return 1;
        }
    }

    public sealed class SealedSecondInherietClass : VirtualInterfacedClass
    {
        public override int func()
        {
            return 1;
        }
    }

    public abstract class AbstractClass
    {
        public abstract int func();
    }

    public class OfAbstractClass : AbstractClass
    {
        public override int func()
        {
            return 1;
        }
    }

    public sealed class SealedOfAbstractClass : AbstractClass
    {
        public override int func()
        {
            return 1;
        }
    }
}
