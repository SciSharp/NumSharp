using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray_Legacy<TData>
    {
        /// <summary>
        /// Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <typeparam name="TData"></typeparam>
        /// <param name="np"></param>
        /// <param name="rows"></param>
        /// <param name="dimenstions"></param>
        /// <returns></returns>
        public NDArray_Legacy<List<TData>> Zeros(int rows, int dimenstions)
        {
            dynamic np = this;

            np.Data = new List<TData>();

            for (int i = 0; i < rows * dimenstions; i++)
            {
                np.Data.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom("0"));
            }

            return np.ReShape(rows, dimenstions);
        }

        public NDArray_Legacy<TData> Zeros(int rows)
        {
            dynamic np = this;
            
            np.Data = new List<TData>();

            for (int i = 0; i < rows; i++)
            {
                np.Data.Add((TData)TypeDescriptor.GetConverter(typeof(TData)).ConvertFrom("0"));
            }

            return np;
        }
    }
}
