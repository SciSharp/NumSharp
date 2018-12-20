using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray arange(int stop, int start = 0, int step = 1)
        {
            var list = new int[(int)Math.Ceiling((stop - start + 0.0) / step)];
            int index = 0;

            Storage.SetData(Array.CreateInstance(this.dtype,list.Length));

            Array data = Storage.GetData(); 

            for (int i = start; i < stop; i += step)
                list[index++] = i;

            switch (data)
            {
                case int[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                case long[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                case double[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                case float[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = list[idx];
                        break;
                    }
                case Complex[] dataArray:
                    {
                        // no performance critial operation
                        dataArray = list.Select(x => (Complex)x).ToArray();
                        break;
                    }
                case Quaternion[] dataArray:
                    {
                        // no performance critial operation
                        dataArray = list.Select(x => new Quaternion(new Vector3(0, 0, 0), x)).ToArray();
                        break;
                    }
                default:
                    {
                        throw new IncorrectTypeException();
                    }
            }

            this.Storage.Reshape(list.Length);

            return this;
        }
    }
}
