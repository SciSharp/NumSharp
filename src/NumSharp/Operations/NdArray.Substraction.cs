using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Shared;

namespace NumSharp
{
    public partial class NDArray<T>
    {
        public static NDArray<T> operator -(NDArray<T> np1, NDArray<T> np2)
        {
            dynamic sub = null;
            dynamic np1Dyn = np1.Data.ToArray();
            dynamic np2Dyn = np2.Data.ToArray();

            var dataType = typeof(T);

            switch (dataType.Name)
            {
                case ("Double"): sub = new NDArray<double>().Array(((double[])Substraction.SubDoubleArrayFromDoubleArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Float"): sub = new NDArray<float>().Array(((float[])Substraction.SubfloatArrayFromfloatArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Complex"): sub = new NDArray<Complex>().Array(((Complex[])Substraction.SubComplexArrayFromComplexArray(np1Dyn,np2Dyn)).ToList() ); break;
                case ("Quaternion"): sub = new NDArray<Quaternion>().Array(((Quaternion[])Substraction.SubQuaternionArrayFromQuaternionArray(np1Dyn,np2Dyn)).ToList() ); break;
            }
            
            return (NDArray<T>) sub;
        }
    }
}
