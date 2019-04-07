using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray
    {

        // TODO: create an overload because np.power also allows to pass an array of exponents for every entry in the array

        /// <summary>
        /// Array elements raised to given powers, element-wise.
        /// </summary>
        public NDArray power(ValueType exponent)
        {
            var powerArray = new NDArray(this.dtype, this.shape);

            Array dataSysArr = this.Storage.GetData();
            Array powerDataSysArr = powerArray.Storage.GetData();

            switch (dataSysArr)
            {
                case double[] data:
                    {
                        var powerData = powerDataSysArr as double[];

                        for (int idx = 0; idx < data.Length; idx++)
                            powerData[idx] = Math.Pow(data[idx], (double)exponent);

                        break;
                    }
                case float[] data:
                    {
                        var powerData = powerDataSysArr as float[];

                        for (int idx = 0; idx < data.Length; idx++)
                            powerData[idx] = Convert.ToSingle(Math.Pow(Convert.ToDouble(data[idx]), Convert.ToDouble(exponent)));

                        break;
                    }
                case Complex[] data:
                    {
                        var powerData = powerDataSysArr as Complex[];

                        for (int idx = 0; idx < data.Length; idx++)
                            powerData[idx] = Complex.Pow(data[idx], Convert.ToDouble(exponent));

                        break;
                    }
                case int[] data:
                    {
                        var powerData = powerDataSysArr as int[];

                        for (int idx = 0; idx < data.Length; idx++)
                            powerData[idx] = Convert.ToInt32(Math.Pow(Convert.ToDouble(data[idx]), Convert.ToDouble(exponent)));

                        break;
                    }
                default:
                    {
                        throw new IncorrectTypeException();
                    }

            }
            return powerArray;
        }
    }

}
