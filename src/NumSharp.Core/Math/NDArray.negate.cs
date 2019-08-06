namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Negates all values by performing: -x
        /// </summary>
        public NDArray negate()
        {
            return TensorEngine.Negate(this);
        }
    }
}
