using System.Linq;

namespace NumSharp
{
    public partial class NDArray
    {
        internal NDArray DetermineEmptyResult(NDArray np2, ref int scalarArrayNo)
        {
            scalarArrayNo = !(this.ndim == 0 || np2.ndim == 0) ? 0 : -1;

            if (scalarArrayNo == 0)
            {
                if (!Enumerable.SequenceEqual(this.shape, np2.shape))
                {
                    throw new IncorrectShapeException();
                }
            }
            else
            {
                if (this.ndim == 0)
                    scalarArrayNo = 1;
                else
                    scalarArrayNo = 2;
            }

            NDArray result = null;

            switch (scalarArrayNo)
            {
                case 1:
                {
                    result = new NDArray(np2.dtype, np2.shape);
                    break;
                }

                case 2:
                {
                    result = new NDArray(this.dtype, this.shape);
                    break;
                }

                default:
                {
                    result = new NDArray(this.dtype, this.shape);
                    break;
                }
            }

            return result;
        }
    }
}
