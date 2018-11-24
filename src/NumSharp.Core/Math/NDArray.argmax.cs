using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public int argmax()
        {
            switch (dtype.Name)
            {
                case "Int32":
                    {
                        var max = Data<int>().Max();
                        return Data<int>().ToList().IndexOf(max);
                    }
                case "Double":
                    {
                        var max = Data<double>().Max();
                        return Data<double>().ToList().IndexOf(max);
                    }
            }

            return -1;
        }

        public int argmax<T>()
        {
            var max = Data<T>().Max();

            return Data<T>().ToList().IndexOf(max);
        }
    }
}
