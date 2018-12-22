using System;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray matrix_power(int power)
        {
            if (power < 0)
                throw new Exception("matrix_power just work with int >= 0");

            NDArray product = this.copy();

            for(int idx = 2; idx <= power;idx++)
                product = product.dot(this);

            product = (power == 0) ? np.eye(product.shape.Dimensions[0]) : product;
            
            return product;
        }
    }
}