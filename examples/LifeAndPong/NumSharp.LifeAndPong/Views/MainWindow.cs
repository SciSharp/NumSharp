using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace NumSharp.LifeAndPong.Views;

public sealed class MainWindow : Window
{
    private readonly GameSurface _surface = new();

    public MainWindow()
    {
        Title = "NumSharp · Life + Pong";
        Width = 1440;
        Height = 900;
        MinWidth = 1120;
        MinHeight = 700;
        Background = new SolidColorBrush(Color.Parse("#070A12"));
        Content = _surface;
        WindowStartupLocation = WindowStartupLocation.CenterScreen;
        Deactivated += (_, _) => _surface.ReleaseTransientInput();
        Closed += (_, _) => _surface.Dispose();
    }
}
