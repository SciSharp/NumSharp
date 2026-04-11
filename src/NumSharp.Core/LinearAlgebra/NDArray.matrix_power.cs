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

            // np.eye expects int; matrix dimensions typically small but check anyway
            if (power == 0)
            {
                if (product.shape[0] > int.MaxValue)
                    throw new OverflowException($"Matrix dimension {product.shape[0]} exceeds int.MaxValue for np.eye");
                product = np.eye((int)product.shape[0]);
            }

            return product;
        }
    }
}
