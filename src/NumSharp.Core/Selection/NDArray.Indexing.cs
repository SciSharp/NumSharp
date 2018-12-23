using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        /// <summary>
        /// Retrieve element
        /// low performance, use generic Data<T> method for performance sensitive invoke
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public object this[params int[] select]
        {
            get
            {
                return Storage.GetData(select);
            }

            set
            {
                Storage.SetData(value, select);
            }
        }
        public NDArray this[NumSharp.Generic.NDArray<bool> booleanArray]
        {
            get
            {
                return default(NDArray);
            }
            set
            {
                if (!Enumerable.SequenceEqual(this.shape,booleanArray.shape))
                {
                    throw new IncorrectShapeException();
                }

                object scalarObj = value.Storage.GetData().GetValue(0);

                bool[] boolDotNetArray = booleanArray.Storage.GetData() as bool[];

                int elementsAmount = booleanArray.size;
                Array data = this.Storage.GetData();

                for(int idx = 0; idx < elementsAmount;idx++)
                {
                    if (boolDotNetArray[idx])
                    {
                        int[] indexes = booleanArray.Storage.Shape.GetDimIndexOutShape(idx);
                        data.SetValue(scalarObj,this.Storage.Shape.GetIndexInShape(indexes));
                    }
                }

            } 
        }
      
    }

    
}
