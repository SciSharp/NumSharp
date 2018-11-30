using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public int argmax()
        {
            var data = this.Storage.GetData();

            int index = -1;

            switch (data)
            {
                case double[] castedData : 
                {
                    index = castedData.ToList().IndexOf(castedData.Max());
                    break;
                }
                case float[] castedData : 
                {
                    index = castedData.ToList().IndexOf(castedData.Max());
                    break;
                }
                case int[] castedData : 
                {
                    index = castedData.ToList().IndexOf(castedData.Max());
                    break;
                }
                case Int64[] castedData : 
                {
                    index = castedData.ToList().IndexOf(castedData.Max());
                    break;
                }
                case decimal[] castedData : 
                {
                    index = castedData.ToList().IndexOf(castedData.Max());
                    break;
                }
                default :
                {
                    throw new IncorrectTypeException();
                }
            }
            
            return index;
        }

        public int argmax<T>()
        {
            var max = Data<T>().Max();

            return Data<T>().ToList().IndexOf(max);
        }
    }
}
