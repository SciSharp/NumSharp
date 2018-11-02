using System;
using System.Linq;

/*
[Fact]
        

        [Fact]
        
        [Fact]
        
 */

namespace NumSharp.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            ArrayTester.Access();
            ArrayTester.CheckPlusOperation();
            ArrayTester.CheckMatrixMultiplication();
            
            Console.ReadKey();
        }
    }
}
