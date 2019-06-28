using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 10)]
    [SimpleJob(RunStrategy.Throughput, targetCount: 10)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class ArrayCopying
    {
        public double[] source;
        private double[] target;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            // first array
            source = new double[10_000_000];
            target = new double[10_000_000];

            for (int i = 0; i < source.Length; i++)
            {
                source[i] = i % 2;
            }
        }


        [Benchmark(Baseline = true)]
        public void ForLoopSetter()
        {
            var length = source.Length;
            for (int i = 0; i < length; i++)
            {
                target[i] = source[i];
            }
        }

        [Benchmark]
        public void ArrayCopy()
        {
            Array.Copy(source, 0, target, 0, target.Length);
        }

        [Benchmark]
        public void AsSpan()
        {
            source.AsSpan().CopyTo(target);
        }

        [Benchmark]
        public void BlockCopy()
        {
            Buffer.BlockCopy(source, 0, target, 0, target.Length * Marshal.SizeOf<Double>());
        }


        [Benchmark]
        public void CopyTo()
        {
            Array.Copy(source, target, source.Length);
        }

        [Benchmark]
        public void CopyConstraint()
        {
            Array.ConstrainedCopy(source, 0, target, 0, source.Length);
        }

        [Benchmark]
        public void UnsafeCopy()
        {
            FastByteCopy<double>(source, target, source.Length);
        }

        static void FastByteCopy<T>(Object source, Object destination, int length) where T : unmanaged
        {
            GCHandle h = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr p = h.AddrOfPinnedObject();

            GCHandle h2 = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr p2 = h2.AddrOfPinnedObject();

            unsafe
            {
                T* pByte = (T*)p;
                T* pByte2 = (T*)p2;

                for (int i = 0; i <= length; i++)
                    *pByte++ = *pByte2++;
            }

            if (h.IsAllocated) h.Free();
            if (h2.IsAllocated) h2.Free();
        }


        //[Benchmark]
        //public unsafe void memcpyimpl_Unsafe()
        //{
        //    Stopwatch sw = Stopwatch.StartNew();
        //    fixed (byte* pSrc = aSource)
        //    fixed (byte* pDest = aTarget)
        //        for (int i = 0; i < COUNT; i++)
        //            memcpyimpl(pSrc, pDest, SIZE);
        //    sw.Stop();
        //    Console.WriteLine("Buffer.memcpyimpl: {0:N0} ticks", sw.ElapsedTicks);
        //}
    }
}
