namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray min(int? axis = null)
        {
            switch (dtype.Name)
            {
                case "Double":
                    return np.amin(this, axis);
            }

            return null;
        }

        public NDArray min<T>(int? axis = null)
        {
            return np.amin(this, axis);
        }
    }
}
