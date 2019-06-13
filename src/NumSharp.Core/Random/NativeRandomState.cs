using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NumSharp
{
    internal static class RandomStateExtensions
    {
        public static NativeRandomState Save(this System.Random random)
        {
            var binaryFormatter = new BinaryFormatter();
            using (var temp = new MemoryStream())
            {
                binaryFormatter.Serialize(temp, random);
                return new NativeRandomState(temp.ToArray());
            }
        }

        public static System.Random Restore(this NativeRandomState state)
        {
            var binaryFormatter = new BinaryFormatter();
            using (var temp = new MemoryStream(state.State))
            {
                return (System.Random)binaryFormatter.Deserialize(temp);
            }
        }
    }

    /// <summary>
    ///     Represents the stored state of <see cref="Random"/>.
    /// </summary>
    [Serializable]
    public struct NativeRandomState
    {
        public readonly byte[] State;

        public NativeRandomState(byte[] state)
        {
            State = state;
        }
    }
}
