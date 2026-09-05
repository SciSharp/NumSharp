using System.Runtime.InteropServices;
using System.Threading.Channels;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Desktop;

/// <summary>Small original PCM cues, played off the UI thread. No downloaded audio or extra dependency.</summary>
internal sealed class WindowsGameAudio : IGameAudio
{
    private readonly Channel<byte[]> _sounds = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(1) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true });
    private readonly byte[][] _notes = Enumerable.Range(0, 9).Select(i => Wave(330 * Math.Pow(2, i / 12d), .07)).ToArray();
    private readonly byte[] _paddle = Wave(220, .035), _miss = Wave(92, .15), _birth = Wave(660, .10), _sector = Wave(880, .12);
    private volatile bool _disposed;
    public bool Available => OperatingSystem.IsWindows();
    public WindowsGameAudio()
    {
        if (Available) _ = Task.Run(Consume);
    }
    public void Play(ArcadeEvent item)
    {
        if (_disposed || !Available) return;
        byte[]? sound = item.Kind switch
        {
            ArcadeEventKind.Cell => _notes[Math.Clamp((int)Math.Log2(Math.Max(1, item.Value)), 0, 8)],
            ArcadeEventKind.Paddle => _paddle,
            ArcadeEventKind.Miss => _miss,
            ArcadeEventKind.Birth => _birth,
            ArcadeEventKind.Sector => _sector,
            _ => null
        };
        if (sound is not null) _sounds.Writer.TryWrite(sound);
    }
    private async Task Consume()
    {
        try
        {
            await foreach (var sound in _sounds.Reader.ReadAllAsync())
            {
                if (_disposed) break;
                // Synchronous memory playback keeps the marshalled buffer pinned until playback finishes.
                PlaySound(sound, IntPtr.Zero, 0x0004 | 0x0002);
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException) { }
    }
    private static byte[] Wave(double frequency, double duration)
    {
        const int rate = 22050;
        var count = (int)(rate * duration);
        using var memory = new MemoryStream();
        using var writer = new BinaryWriter(memory);
        writer.Write("RIFF"u8); writer.Write(36 + count * 2); writer.Write("WAVEfmt "u8);
        writer.Write(16); writer.Write((short)1); writer.Write((short)1); writer.Write(rate); writer.Write(rate * 2); writer.Write((short)2); writer.Write((short)16);
        writer.Write("data"u8); writer.Write(count * 2);
        for (var i = 0; i < count; i++)
        {
            var t = (double)i / rate;
            var envelope = Math.Min(1, t / .004) * Math.Pow(1 - (double)i / count, 2);
            var sample = Math.Sin(2 * Math.PI * frequency * t) + .18 * Math.Sin(4 * Math.PI * frequency * t);
            writer.Write((short)(sample * envelope * 5500));
        }
        return memory.ToArray();
    }
    public void Dispose() { _disposed = true; _sounds.Writer.TryComplete(); }
    [DllImport("winmm.dll", EntryPoint = "PlaySoundW")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool PlaySound(byte[] data, IntPtr module, uint flags);
}
