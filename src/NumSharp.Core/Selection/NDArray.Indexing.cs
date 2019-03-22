using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

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
                NDArray selectedValues = null;

                switch (ndim)
                {
                    case 1:
                        if (dtype.Name == "Byte")
                            selectedValues = setValue1D<byte>(indexes);
                        if (dtype.Name == "Int32")
                            selectedValues = setValue1D<int>(indexes);
                        if (dtype.Name == "Single")
                            selectedValues = setValue1D<float>(indexes);
                        else if (dtype.Name == "Double")
                            selectedValues = setValue1D<double>(indexes);
                        break;
                    case 2:
                        if (dtype.Name == "Byte")
                            selectedValues = setValue2D<byte>(indexes);
                        if (dtype.Name == "Int32")
                            selectedValues = setValue2D<int>(indexes);
                        if (dtype.Name == "Single")
                            selectedValues = setValue2D<float>(indexes);
                        else if (dtype.Name == "Double")
                            selectedValues = setValue2D<double>(indexes);
                        break;
                    case 3:
                        if (dtype.Name == "Byte")
                            selectedValues = setValue3D<byte>(indexes);
                        if (dtype.Name == "Int32")
                            selectedValues = setValue3D<int>(indexes);
                        if (dtype.Name == "Single")
                            selectedValues = setValue3D<float>(indexes);
                        else if (dtype.Name == "Double")
                            selectedValues = setValue3D<double>(indexes);
                        break;
                    case 4:
                        if (dtype.Name == "Byte")
                            selectedValues = setValue4D<byte>(indexes);
                        if (dtype.Name == "Int32")
                            selectedValues = setValue4D<int>(indexes);
                        if (dtype.Name == "Single")
                            selectedValues = setValue4D<float>(indexes);
                        else if (dtype.Name == "Double")
                            selectedValues = setValue4D<double>(indexes);
                        break;
                }

                return selectedValues;
            }
        }

        private NDArray setValue1D<T>(NDArray indexes)
        {
            var buf = Data<T>();
            var idx = indexes.Data<int>();
            var values = new T[indexes.size];

            Parallel.ForEach(Enumerable.Range(0, indexes.size), (row) =>
            {
                values[row] = buf[idx[row]];
            });

            return new NDArray(values, indexes.size);
        }

        private NDArray setValue2D<T>(NDArray indexes)
        {
            var buf = Data<T>();
            var idx = indexes.Data<int>();
            var selectedValues = new NDArray(this.dtype, new Shape(indexes.size, shape[1]));

            Parallel.ForEach(Enumerable.Range(0, selectedValues.shape[0]), (row) =>
            {
                for (int col = 0; col < selectedValues.shape[1]; col++)
                    selectedValues[row, col] = buf[Storage.Shape.GetIndexInShape(idx[row], col)];
            });

            return selectedValues;
        }

        private NDArray setValue3D<T>(NDArray indexes)
        {
            var buf = Data<T>();
            var selectedValues = new NDArray(this.dtype, new Shape(indexes.size, shape[1], shape[2]));
            var idx = indexes.Data<int>();

            Parallel.ForEach(Enumerable.Range(0, selectedValues.shape[0]), (item) =>
            {
                for (int row = 0; row < selectedValues.shape[1]; row++)
                    for (int col = 0; col < selectedValues.shape[2]; col++)
                            selectedValues[item, row, col] = buf[Storage.Shape.GetIndexInShape(idx[item], row, col)];
            });

            return selectedValues;
        }

        private NDArray setValue4D<T>(NDArray indexes)
        {
            var buf = Data<T>();
            var selectedValues = new NDArray(this.dtype, new Shape(indexes.size, shape[1], shape[2], shape[3]));
            var idx = indexes.Data<int>();

            Parallel.ForEach(Enumerable.Range(0, selectedValues.shape[0]), (item) =>
            {
                for (int row = 0; row < selectedValues.shape[1]; row++)
                    for (int col = 0; col < selectedValues.shape[2]; col++)
                        for (int channel = 0; channel < selectedValues.shape[3]; channel++)
                            selectedValues[item, row, col, channel] = buf[Storage.Shape.GetIndexInShape(idx[item], row, col, channel)];
            });

            return selectedValues;
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
