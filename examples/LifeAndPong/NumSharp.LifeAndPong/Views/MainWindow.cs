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
        Opened += (_, _) => _surface.Focus();
        Deactivated += (_, _) => _surface.SuspendPlay();
        Closed += (_, _) => _surface.Dispose();
    }
}
