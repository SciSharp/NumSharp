using System;
using NumSharp;

namespace ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            int i = 0;
            while (true)
            {
                var nd1 = np.arange(1, 10 * 1000);
                var nd2 = np.arange(1, 10 * 1000);
                var nd3 = nd1 % nd2;

                // 1000 ms
                i++;
                if (i == 10 * 1000)
                {
                    i = 0;
                }
            }

            Console.ReadLine();
        }
    }
}
