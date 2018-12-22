using System;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray copy(string order = null)
        {   
            NDArray puffer = this.Clone()  as NDArray;

            order = (order == null) ? ((this.Storage.TensorLayout == 1) ? "F" : "C" ) : order;

            switch (order)
            {
                case "C" :
                {
                    puffer.Storage.ChangeTensorLayout(2);
                    break;
                }
                case "F" :
                {
                    puffer.Storage.ChangeTensorLayout(1);
                    break;
                } 
                default :
                {
                    throw new Exception("Copy just accept C or F order.");
                } 
            }

            return puffer;
        }

    }
}