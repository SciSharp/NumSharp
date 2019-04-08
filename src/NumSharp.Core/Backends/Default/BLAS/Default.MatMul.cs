using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public abstract partial class DefaultEngine
    {
        public virtual NDArray MatMul(NDArray x, NDArray y)
        {
            if (x.ndim == 2 && y.ndim == 2)
            {
                var nd = new NDArray(x.dtype, new Shape(x.shape[0], y.shape[1]));
                switch (nd.dtype.Name)
                {
                    case "Int32":
                        Parallel.ForEach(Enumerable.Range(0, nd.shape[0]), (row) =>
                        {
                            for (int col = 0; col < nd.shape[1]; col++)
                            {
                                int sum = 0;
                                for (int s = 0; s < nd.shape[0]; s++)
                                    sum += x.Data<int>(row, s) * y.Data<int>(s, col);
                                nd[row, col] = sum;
                            }
                        });
                        break;

                    case "Single":
                        Parallel.ForEach(Enumerable.Range(0, nd.shape[0]), (row) =>
                        {
                            for (int col = 0; col < nd.shape[1]; col++)
                            {
                                float sum = 0;
                                for (int s = 0; s < nd.shape[0]; s++)
                                    sum += x.Data<float>(row, s) * y.Data<float>(s, col);
                                nd[row, col] = sum;
                            }
                        });
                        break;

                    case "Double":
                        Parallel.ForEach(Enumerable.Range(0, nd.shape[0]), (row) =>
                        {
                            for (int col = 0; col < nd.shape[1]; col++)
                            {
                                double sum = 0;
                                for (int s = 0; s < nd.shape[0]; s++)
                                    sum += x.Data<double>(row, s) * y.Data<double>(s, col);
                                nd[row, col] = sum;
                            }
                        });
                        break;
                }

                return nd;
            }

            throw new NotImplementedException("matmul");
        }
    }
}
