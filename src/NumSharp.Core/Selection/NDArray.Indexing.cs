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
        public NDArray this[NDArray indexes]
        {
            get
            {
                if(!(indexes.ndim == 1))
                            throw new IncorrectShapeException();

                Array indexesArr = indexes.Storage.GetData();
                Array thisArr = this.Storage.GetData();
                
                NDArray selectedValues = null;

                switch (this.ndim)
                {
                    case 1 :
                    {
                        selectedValues = new NDArray(this.dtype, indexes.size);

                        Array selValArr = selectedValues.Storage.GetData();
                    
                        for(int idx = 0; idx < indexesArr.Length;idx++)
                            selValArr.SetValue(thisArr.GetValue(Convert.ToInt32(indexesArr.GetValue(idx))),idx);
                    
                        break;
                    }
                    case 2 :
                    {
                            selectedValues = new NDArray(this.dtype, new Shape(indexes.size, shape[1]));
                            for (int row = 0; row < selectedValues.shape[0]; row++)
                                for (int col = 0; col < selectedValues.shape[1]; col++)
                                    selectedValues[row, col] = this[indexes.Data<int>(row), col];

                            break;
                    }
                    case 4:
                    {
                        selectedValues = new NDArray(this.dtype, new Shape(indexes.size, shape[1], shape[2], shape[3]));

                        for (int item = 0; item < selectedValues.shape[0]; item++)
                            for (int row = 0; row < selectedValues.shape[1]; row++)
                                for (int col = 0; col < selectedValues.shape[2]; col++)
                                    for (int channel = 0; channel < selectedValues.shape[3]; channel++)
                                        selectedValues[item, row, col, channel] = this[indexes.Data<int>(item), row, col, channel];
                        break;
                    }
                }

                return selectedValues;
            }
        }
        public NDArray this[NumSharp.Generic.NDArray<bool> booleanArray]
        {
            get
            {
                if (!Enumerable.SequenceEqual(this.shape,booleanArray.shape))
                {
                    throw new IncorrectShapeException();
                }

                List<object> selectedList = new List<object>();

                bool[] boolDotNetArray = booleanArray.Storage.GetData() as bool[];

                int elementsAmount = booleanArray.size;

                Array data = this.Storage.GetData();

                for(int idx = 0; idx < elementsAmount;idx++)
                {
                    if (boolDotNetArray[idx])
                    {
                        int[] indexes = booleanArray.Storage.Shape.GetDimIndexOutShape(idx);
                        selectedList.Add(data.GetValue(this.Storage.Shape.GetIndexInShape(indexes)));
                    }
                }

                NDArray selected = new NDArray(this.dtype,selectedList.Count);
                selected.Storage.SetData(selectedList.ToArray());

                return selected;
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
