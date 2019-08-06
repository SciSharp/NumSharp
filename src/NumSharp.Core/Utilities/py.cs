namespace NumSharp.Utilities
{
    /// <summary>
    /// Implements Python utility functions that are often used in connection with numpy
    /// </summary>
    public static class py
    {
        public static int[] range(int n)
        {
            var a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = i;
            return a;
        }
    }
}
