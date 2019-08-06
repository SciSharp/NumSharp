namespace NumSharp
{
    public enum LAPACKProviderType
    {
        NetLib
    }

    public static partial class np
    {
        public static LAPACKProviderType LAPACKProvider = LAPACKProviderType.NetLib;
    }
}
