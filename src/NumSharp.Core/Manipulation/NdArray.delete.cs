using System.Collections;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray delete(NDArray array, int[] indexToDelete, int axis = 0)
        {

            int nToDelet = indexToDelete.Length;
            int x_shape = array.shape[0];
            int y_shape = array.shape[1];

            if (axis == 0)
                x_shape -= nToDelet;
            else
                y_shape -= nToDelet;

            int[,] newArray = new int[x_shape, y_shape];

            if (axis == 0)
            {
                int newRowIndex = 0;
                for (int i = 0; i < array.shape[0]; i++)
                {
                    if (!indexToDelete.Contains(i))
                    {
                        for (int j = 0; j < array.shape[1]; j++)
                        {
                            newArray[newRowIndex, j] = array[i, j];
                        }
                        newRowIndex++;
                    }
                }
            }
            else
            {
                int newColumnIndex = 0;
                for (int j = 0; j < array.shape[1]; j++)
                {
                    if (!indexToDelete.Contains(j))
                    {
                        for (int i = 0; i < array.shape[0]; i++)
                        {
                            newArray[i, newColumnIndex] = array[i, j];
                        }
                        newColumnIndex++;
                    }
                }
            }

            return np.array(newArray);
            
            //return null;

            //var sysArr = this.Storage.GetData();

            //NDArray res = null;

            //switch( sysArr)
            //{
            //    case double[] castedSysArr : 
            //    {
            //        var castedDelete = delete as double[];

            //        res = np.array(castedSysArr.Where(x => !castedDelete.Contains(x) ).ToArray());

            //        break;
            //    }
            //    case float[] castedSysArr : 
            //    {
            //        var castedDelete = delete as float[];

            //        res = np.array(castedSysArr.Where(x => !castedDelete.Contains(x) ).ToArray());

            //        break;
            //    }
            //    case int[] castedSysArr :
            //    {
            //        var castedDelete = delete as int[];

            //        res = np.array(castedSysArr.Where(x => !castedDelete.Contains(x) ).ToArray());

            //        break;
            //    }
            //    default : 
            //    {
            //        throw new IncorrectTypeException();
            //    }
            //}

            //return res;
        }
    }
}
