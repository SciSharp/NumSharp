using System;

namespace NumSharp
{
    internal static class RandomStateExtensions
    {
        public static NativeRandomState Save(this Randomizer random)
        {
            return new NativeRandomState(random.Serialize());
        }

        public static Randomizer Restore(this NativeRandomState state)
        {
            return Randomizer.Deserialize(state.State);
        }
    }

    /// <summary>
    ///     Represents the stored state of <see cref="Randomizer"/>.
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
