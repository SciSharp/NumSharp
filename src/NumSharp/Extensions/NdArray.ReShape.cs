using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace NumSharp
{
    public partial class NDArray<TData>
    {
        /// <summary>
        /// Gives a new shape to an array without changing its data.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <param name="np"></param>
        /// <param name="row"></param>
        /// <param name="dim"></param>
        /// <returns></returns>
        public NDArray<List<TData>> ReShape(int row, int dim)
        {
            var np = this;

            // loop row
            var rows = new List<List<TData>>();

            int index = 0;

            for (int r = 0; r < row; r++)
            {
                var dims = new List<TData>();

                for (int d = 0; d < dim; d++)
                {
                    dims.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom(np.Data[index].ToString()));

                    index++;
                }

                rows.Add(dims);
            }

            return new NDArray<List<TData>>
            {
                Data = rows
            };
        }
    }
}
