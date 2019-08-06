namespace NumSharp
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
            return null;
            //var np2 = np2Multi.Last(); 

            //if ((this.ndim == 1 ) & (np2.ndim == 1))
            //    if (this.shape[0] != np2.shape[0])
            //        throw new IncorrectShapeException(); 
            //    else 
            //    {
            //        np2.Storage.Reshape(np2.Storage.GetData().Length,1);
            //        this.Storage.Reshape(1,this.Storage.GetData().Length);
            //    }
            //else
            //    if (this.shape[1] != np2.shape[0])
            //        throw new IncorrectShapeException();

            //var prod = BackendFactory.GetEngine().Dot(this, np2Multi[0]);

            //for(int idx = 1;idx < np2Multi.Length;idx++)
            //{
            //    prod = BackendFactory.GetEngine().Dot(prod, np2Multi[idx]);
            //}

            //return prod;
        }
    }
}
