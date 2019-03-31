using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.UnitTest.Backends.VectorT
{
    [TestClass]
    public class StorageTest
    {
        public void allBench(int loop, int arraySize, int rerun)
        {
            Stopwatch sw = new Stopwatch();
            Random random = new Random();

            Func<int, int[]> genRandomIntArray = n => Enumerable.Repeat(0, n).Select(i => random.Next(2 * n)).ToArray();
            Func<int, int[]> genRandomIntArray2 = n =>
            {
                int[] ar = new int[n];
                for (int k = 0; k < n; ++k) ar[k] = random.Next(2 * n);
                return ar;
            };

            Action<int, int> bench1 = (K, N) =>
            {
                sw.Restart();
                int[] x = genRandomIntArray2(N);
                int[] y = genRandomIntArray2(N);
                int[] r = new int[N];

                for (int k = 0; k < K; ++k)
                {
                    for (int i = 0; i < N; ++i) r[i] = x[i] + y[i];
                }

                Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds,6} ms");
            };

            Action<int, int> bench2 = (K, N) =>
            {
                sw.Restart();
                int[] x = genRandomIntArray2(N);
                int[] y = genRandomIntArray2(N);
                int[] r = new int[N];
                Vector<int> v0 = Vector<int>.Zero;

                for (int k = 0; k < K; ++k)
                {
                    for (int i = 0; i < N; i += Vector<int>.Count)
                    {
                        v0 = new Vector<int>(x, i) + new Vector<int>(y, i);
                        v0.CopyTo(r, i);
                    }
                }

                Console.WriteLine($"Elapsed {sw.ElapsedMilliseconds,6} ms");
            };

            int m0 = rerun;
            int m1 = loop;
            int m2 = arraySize * 16;

            Console.WriteLine("#### Normal Computation");
            for (int i = 0; i < m0; ++i) bench1(m1, m2);
            Console.WriteLine();

            Console.WriteLine("#### SIMD Computation");
            for (int i = 0; i < m0; ++i) bench2(m1, m2);
            Console.WriteLine();
        }

        [TestMethod]
        public void SIMD_AVX()
        {
            allBench(loop: 10000, arraySize: 500, rerun: 10); // array size will be multiplied by 16
        }
    }
}

