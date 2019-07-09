using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Return a copy of the array collapsed into one dimension.
        /// 
        /// A 1-D array, containing the elements of the input, is returned.A copy is made only if needed.
        /// </summary>
        public NDArray flatten(char order = 'C')
        {
            var nd = new NDArray(dtype, size);

            var s = new Shape(Storage.Shape.Dimensions);
            s.ChangeTensorLayout(order);

            switch (dtype.Name)
            {
                case "Int32":
                    var values1 = Array as int[];
                    var values2 = new int[size];
                    for (int i = 0; i < size; i++)
                    {
                        // Data<int>(s.GetDimIndexOutShape(i))
                        values2[i] = values1[Storage.Shape.GetIndexInShape(slice, s.GetDimIndexOutShape(i))];
                    }
                    nd.Array = values2;
                    break;
            }

            return nd;
        }
    }
}
