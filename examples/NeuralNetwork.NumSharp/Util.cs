using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.NumSharp
{
    public class Util
    {
        private static int counter = 0;

        public static int GetNext()
        {
            return counter++;
        }
    }
}
