
using System;
using System.Numerics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace NumSharp.Benchmark
{
    //[RPlotExporter, RankColumn]
    [SimpleJob(RunStrategy.ColdStart, targetCount: 20)]
    [MinColumn, MaxColumn, MeanColumn, MedianColumn]
    [HtmlExporter]
    public class LinqTesterQuaternion
    {
        public Quaternion[] np1;
        public Quaternion[] np2;
        public Quaternion scalar;
        public Quaternion[] result;
        [GlobalSetup]
        public void Setup()
        {
            // first array
            np1 = new Quaternion[100 * 100];
            np1 = np1.Select((x,idx) => new Quaternion(new Vector3(0,0,0),(float) idx)).ToArray();
            
            // second array
            np2 = new Quaternion[100 * 100];
            np2 = np2.Select((x,idx) => new Quaternion(new Vector3(0,0,0),(float) (idx+1))).ToArray();
            
            // scalar 
            scalar = new Quaternion(new Vector3(0,0,0),(float) 25);
        }
        [Benchmark(Baseline = true)]
        public void NormalForLoopWithPreDefineResultArray()
        {
            result = new Quaternion[np1.Length];

            // 2 arrays
            for (int idx = 0; idx < np1.Length;idx++)
            {
                result[idx] = np1[idx] + np2[idx];
            }
            // 1 array and scalar
            for (int idx = 0; idx < np1.Length;idx++)
            {
                result[idx] = np1[idx] + scalar;
            }
        }
        [Benchmark]
        public void LinqSelectWithPreDefineResultArray()
        {
            result = new Quaternion[np1.Length];

            // 2 arrays
            result = np1.Select((x,idx) => x + np2[idx]).ToArray();
            // 1 array and scalar
            result = np1.Select((x,idx) => x + scalar).ToArray();
        }
        [Benchmark]
        public void LinqSelectWithoutPreDefineResultArray()
        {
            // 2 arrays
            result = np1.Select((x,idx) => x + np2[idx]).ToArray();
            // 1 array and scalar
            result = np1.Select((x,idx) => x + scalar).ToArray();
        }
    }
}
