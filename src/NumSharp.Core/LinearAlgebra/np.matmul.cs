using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray matmul(NDArray a, NDArray b)
        {
            if (a.ndim == b.ndim && a.ndim == 2)
            {
                var nd = new NDArray(a.dtype, new Shape(a.shape[0], b.shape[1]));
                switch (nd.dtype.Name)
                {
                    case "Int32":
                        Parallel.ForEach(Enumerable.Range(0, nd.shape[0]), (row) =>
                        {
                            for (int col = 0; col < nd.shape[1]; col++)
                            {
                                int sum = 0;
                                for (int s = 0; s < nd.shape[0]; s++)
                                    sum += a.Data<int>(row, s) * b.Data<int>(s, col);
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
                                    sum += a.Data<float>(row, s) * b.Data<float>(s, col);
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
                                    sum += a.Data<double>(row, s) * b.Data<double>(s, col);
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
