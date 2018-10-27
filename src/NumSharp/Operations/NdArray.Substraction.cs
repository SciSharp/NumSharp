using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        public static NDArray<TData> operator -(NDArray<TData> np, NDArray<TData> np2)
        {
            dynamic sum = null;
            dynamic npDyn = np;
            dynamic np2Dyn = np2;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray<double>().Array(((NDArray<double>)npDyn).Data.Select((x,idx) => x - ((NDArray<double>)np2Dyn).Data[idx])); break;
                case ("Float"): sum = new NDArray<float>().Array(((NDArray<float>)npDyn).Data.Select((x,idx) => x - ((NDArray<float>)np2Dyn).Data[idx])); break;
                case ("Complex"): sum = new NDArray<Complex>().Array(((NDArray<Complex>)npDyn).Data.Select((x,idx) => x - ((NDArray<Complex>)np2Dyn).Data[idx])); break;
            }
            
            return (NDArray<TData>) sum;
        }
        public static NDArray<TData> operator -(NDArray<TData> np, TData scalar)
        {
            dynamic sum = null;
            dynamic npDyn = np;
            dynamic scalarDyn = scalar;

            var dataType = typeof(TData);

            switch (dataType.Name)
            {
                case ("Double"): sum = new NDArray<double>().Array(((NDArray<double>)npDyn).Data.Select((x) => x - (double)scalarDyn)); break;
                case ("Float"): sum = new NDArray<float>().Array(((NDArray<float>)npDyn).Data.Select((x,idx) => x - (float)scalarDyn)); break;
                case ("Complex"): sum = new NDArray<Complex>().Array(((NDArray<Complex>)npDyn).Data.Select((x,idx) => x - (Complex) scalarDyn )); break;
            }
            
            return (NDArray<TData>) sum;
        }
    }
}
