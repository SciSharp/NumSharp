using System;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray matrix_power(int power)
        {
            if (power < 0)
                throw new Exception("matrix_power just work with int >= 0");

            NDArray product = this.copy();

            for (int idx = 2; idx <= power; idx++)
                product = TensorEngine.Dot(product, this);

            product = (power == 0) ? np.eye(product.shape[0]) : product;

            return product;
        }
    }
}
