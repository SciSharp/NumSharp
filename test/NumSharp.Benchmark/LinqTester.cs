
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 5)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class LinqTester
    {
        public int[] np1Int;
        public int[] np2Int;
        public int scalarInt;
        public int[] resultInt;
        public double[] np1Double;
        public double[] np2Double;
        public double scalarDouble;
        public double[] resultDouble;
        public Quaternion[] np1Quaternion;
        public Quaternion[] np2Quaternion;
        public Quaternion scalarQuaternion;
        public Quaternion[] resultQuaternion;
        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1Int = new int[100 * 100];
            np1Int = np1Int.Select((x,idx) => idx).ToArray();
            np1Double = np1Int.Select(x => (double) x ).ToArray();
            np1Quaternion = np1Double.Select( x => new Quaternion(new Vector3(0,0,0), (float) x )).ToArray();

            // second array
            np2Int = new int[100 * 100];
            np2Int = np2Int.Select((x,idx) => idx + 1).ToArray();
            np2Double = np2Int.Select(x => (double) x ).ToArray();
            np2Quaternion = np2Double.Select( x => new Quaternion(new Vector3(0,0,0), (float) x )).ToArray();

            // scalar 
            scalarInt = 25;
            scalarDouble = (double) scalarInt;
            scalarQuaternion = new Quaternion(new Vector3(0,0,0), (float) scalarDouble );
        }
        [Benchmark(Baseline = true)]
        public void NormalForLoopWithPreDefineResultArray()
        {
            resultInt = new int[np1Int.Length];

            // 2 arrays
            for (int idx = 0; idx < np1Int.Length;idx++)
            {
                resultInt[idx] = np1Int[idx] + np2Int[idx];
            }
            // 1 array and scalar
            for (int idx = 0; idx < np1Int.Length;idx++)
            {
                resultInt[idx] = np1Int[idx] + scalarInt;
            }
        }
        [Benchmark]
        public void LinqSelectWithPreDefineResultArray()
        {
            resultInt = new int[np1Int.Length];

            // 2 arrays
            resultInt = np1Int.Select((x,idx) => x + np2Int[idx]).ToArray();
            // 1 array and scalar
            resultInt = np1Int.Select((x,idx) => x + scalarInt).ToArray();
        }
        [Benchmark]
        public void LinqSelectWithoutPreDefineResultArray()
        {
            // 2 arrays
            resultInt = np1Int.Select((x,idx) => x + np2Int[idx]).ToArray();
            // 1 array and scalar
            resultInt = np1Int.Select((x,idx) => x + scalarInt).ToArray();
        }
    }
}
