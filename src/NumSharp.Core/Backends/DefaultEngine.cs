using NumSharp.Interfaces;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace NumSharp.Backends
{
    /// <summary>
    /// Storage
    ///
    /// Responsible for :
    ///
    ///  - store data type, elements, Shape
    ///  - offers methods for accessing elements depending on shape
    ///  - offers methods for casting elements
    ///  - offers methods for change tensor order
    ///  - GetData always return reference object to the true storage
    ///  - GetData<T> and SetData<T> change dtype and cast storage
    ///  - CloneData always create a clone of storage and return this as reference object
    ///  - CloneData<T> clone storage and cast this clone 
    ///     
    /// </summary>
    public class DefaultEngine : ITensorEngine
    {
        public NDArray Add(NDArray x, NDArray y)
        {
            return x + y;
        }

        public NDArray Dot(NDArray x, NDArray y)
        {
            var dtype = x.dtype;

            if (x.ndim == 0 && y.ndim == 0)
            {
                switch (dtype.Name)
                {
                    case "Int32":
                        return y.Data<int>(0) * x.Data<int>(0);
                }
            }
            else if (x.ndim == 1 && x.ndim == 1)
            {
                int sum = 0;
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < x.size; i++)
                            sum += x.Data<int>(i) * y.Data<int>(i);
                        break;
                }
                return sum;
            }
            else if (x.ndim == 2 && y.ndim == 1)
            {
                var nd = new NDArray(dtype, new Shape(x.shape[0]));
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < x.shape[0]; i++)
                            for (int j = 0; j < y.shape[0]; j++)
                                nd.Data<int>()[i] += x.Data<int>(i, j) * y.Data<int>(j);
                        break;
                }
                return nd;
            }
            else if (x.ndim == 2 && y.ndim == 2)
            {
                return np.matmul(x, y);
            }

            throw new NotImplementedException($"dot {x.ndim} * {y.ndim}");
        }
    }
}