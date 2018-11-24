using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArrayGeneric<T> 
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="np1"></param>
        /// <param name="np2"></param>
        /// <typeparam name="TData"></typeparam>
        /// <returns></returns>
        public NDArrayGeneric<T> multi_dot(params NDArrayGeneric<T>[] np2Multi)
        {
            var np2 = np2Multi.Last(); 

            if ((this.Shape.NDim == 1 ) & (np2.Shape.NDim == 1))
                if (this.Shape.Shapes[0] != np2.Shape.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented."); 
                else 
                {
                    np2.Shape = new Shape(np2.Data.Length,1);
                    this.Shape = new Shape(1,this.Data.Length);
                }
            else
                if (this.Shape.Shapes[1] != np2.Shape.Shapes[0])
                    throw new Exception("The Dot method does not work with this shape or was not already implemented.");
            
            var prod = this.dot(np2Multi[0]);

            for(int idx = 1;idx < np2Multi.Length;idx++)
            {
                prod = prod.dot(np2Multi[idx]);
            }
            
            return prod;
        }
    }
}
