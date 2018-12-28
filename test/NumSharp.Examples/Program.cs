using System;

namespace NumSharp.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            string method = $"NumSharp.Examples.ShallowWater";
            Type type = Type.GetType(method);
            // run example
            var example = (IExample)Activator.CreateInstance(type);
            example.Run();
        }
    }
}
