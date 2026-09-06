using Avalonia;

namespace NumSharp.LifeAndPong.Desktop;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        App.AudioFactory = () => new WindowsGameAudio();
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
