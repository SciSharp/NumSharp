using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="np1"></param>
        /// <param name="np2"></param>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public NDArray multi_dot(params NDArray[] np2Multi)
        {
            var np2 = np2Multi.Last(); 

            if ((this.shape.NDim == 1 ) & (np2.shape.NDim == 1))
                if (this.shape.Dimensions[0] != np2.shape.Dimensions[0])
                    throw new IncorrectShapeException(); 
                else 
                {
                    np2.Storage.Reshape(np2.Storage.GetData().Length,1);
                    this.Storage.Reshape(1,this.Storage.GetData().Length);
                }
            else
                if (this.shape.Dimensions[1] != np2.shape.Dimensions[0])
                    throw new IncorrectShapeException();
            
            var prod = this.dot(np2Multi[0]);

            for(int idx = 1;idx < np2Multi.Length;idx++)
            {
                prod = prod.dot(np2Multi[idx]);
            }
            
            return prod;
        }
    }
}
