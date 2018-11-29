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

            if ((this.Shape.NDim == 1 ) & (np2.Shape.NDim == 1))
                if (this.Shape.Shapes[0] != np2.Shape.Shapes[0])
                    throw new IncorrectShapeException(); 
                else 
                {
                    np2.Storage.Shape = new Shape(np2.Storage.GetData().Length,1);
                    this.Storage.Shape = new Shape(1,this.Storage.GetData().Length);
                }
            else
                if (this.Shape.Shapes[1] != np2.Shape.Shapes[0])
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
