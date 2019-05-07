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
                        {
                            var datax = x.Data<int>();
                            var datay = y.Data<int>();
                            Parallel.For(0, nd.shape[0], (row) =>
                            {
                                for (int col = 0; col < nd.shape[1]; col++)
                                {
                                    int sum = 0;
                                    for (int s = 0; s < nd.shape[0]; s++)
                                        sum += datax[x.GetIndexInShape(row, s)] * datay[y.GetIndexInShape(s, col)];
                                    nd[row, col] = sum;
                                }
                            });
                        }

                        break;
                    case "Single":
                        {
                            var datax = x.Data<float>();
                            var datay = y.Data<float>();
                            Parallel.For(0, nd.shape[0], (row) =>
                            {
                                for (int col = 0; col < nd.shape[1]; col++)
                                {
                                    float sum = 0;
                                    for (int s = 0; s < nd.shape[0]; s++)
                                        sum += datax[x.GetIndexInShape(row, s)] * datay[y.GetIndexInShape(s, col)];
                                    nd[row, col] = sum;
                                }
                            });
                        }

                        break;
                    case "Double":
                        {
                            var datax = x.Data<double>();
                            var datay = y.Data<double>();
                            Parallel.For(0, nd.shape[0], (row) =>
                            {
                                for (int col = 0; col < nd.shape[1]; col++)
                                {
                                    double sum = 0;
                                    for (int s = 0; s < nd.shape[0]; s++)
                                        sum += datax[x.GetIndexInShape(row, s)] * datay[y.GetIndexInShape(s, col)];
                                    nd[row, col] = sum;
                                }
                            });
                        }

                        break;
                }

                return nd;
            }

            throw new NotImplementedException("matmul");
        }
    }
}
