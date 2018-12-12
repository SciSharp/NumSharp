using System;

namespace NumSharp.ConsumePackage
{
    class Program
    {
        static void Main(string[] args)
        {
            var np = new NumSharp.Core.NumPy();

            var A = np.array(new double[]{1,2,3,4}).reshape(2,2);

            var b = np.array(new double[] { 1,2}).reshape(2,1);

            var c = A.lstqr(b);

            Console.WriteLine("If there is no error --> package was sucessfully consumed! :)");
        }
    }
}
