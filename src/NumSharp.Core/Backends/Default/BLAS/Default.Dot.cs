using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public abstract partial class DefaultEngine
    {
        public virtual NDArray Dot(NDArray x, NDArray y)
        {
            var dtype = x.dtype;

            if (x.ndim == 0 && y.ndim == 0)
            {
                switch (dtype.Name)
                {
                    case "Int32":
                        return y.Data<int>(0) * x.Data<int>(0);
                    case "Single":
                        return y.Data<float>(0) * x.Data<float>(0);
                }
            }
            else if (x.ndim == 1 && x.ndim == 1)
            {
                
                switch (dtype.Name)
                {
                    case "Int32":
                        {
                            int sum = 0;
                            for (int i = 0; i < x.size; i++)
                                sum += x.Data<int>(i) * y.Data<int>(i);
                            return sum;
                        }

                    case "Single":
                        {
                            float sum = 0;
                            for (int i = 0; i < x.size; i++)
                                sum += x.Data<float>(i) * y.Data<float>(i);
                            return sum;
                        }
                }
            }
            else if (x.ndim == 2 && y.ndim == 1)
            {
                // check size
                if (x.shape[1] != y.shape[0])
                    throw new IncorrectSizeException($"shapes ({x.shape[0]},{x.shape[1]}) and ({y.shape[0]},) not aligned: {x.shape[1]} (dim 1) != {y.shape[0]} (dim 0)");
                var nd = new NDArray(dtype, new Shape(x.shape[0]));
                switch (dtype.Name)
                {
                    case "Int32":
                        for (int i = 0; i < x.shape[0]; i++)
                            for (int j = 0; j < y.shape[0]; j++)
                                nd.Data<int>()[i] += x.Data<int>(i, j) * y.Data<int>(j);
                        break;
                    case "Single":
                        for (int i = 0; i < x.shape[0]; i++)
                            for (int j = 0; j < y.shape[0]; j++)
                                nd.Data<float>()[i] += x.Data<float>(i, j) * y.Data<float>(j);
                        break;
                }
                return nd;
            }
            else if (x.ndim == 2 && y.ndim == 2)
            {
                // check size
                if (x.shape[1] != y.shape[0])
                    throw new IncorrectSizeException($"shapes ({x.shape[0]},{x.shape[1]}) and ({y.shape[0]},{y.shape[1]}) not aligned: {x.shape[1]} (dim 1) != {y.shape[0]} (dim 0)");
                return np.matmul(x, y);
            }

            throw new NotImplementedException($"dot {x.ndim} * {y.ndim}");
        }
    }
}
