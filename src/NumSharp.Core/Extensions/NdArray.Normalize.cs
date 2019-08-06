using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Normalizes all entries into the range between 0 and 1
        /// 
        /// Note: this is not a numpy function.
        /// </summary>
        public void Normalize()
        {
            var min = this.min(0);
            var max = this.max(0);

            if (ndim == 2)
            {
                for (int col = 0; col < shape[1]; col++)
                {
                    double der = max.Storage.GetValue<double>(col) - min.Storage.GetValue<double>(col);
                    for (int row = 0; row < shape[0]; row++)
                    {
                        this[row, col] = (NDArray) (Storage.GetValue<double>(row, col) - min.Storage.GetValue<double>(col)) / der;
                    }
                }
            }
            else
            {
                throw new NotImplementedException();
            }
        }
    }
}
