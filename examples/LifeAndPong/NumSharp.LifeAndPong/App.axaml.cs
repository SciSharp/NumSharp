using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using NumSharp.LifeAndPong.Views;

namespace NumSharp.LifeAndPong;

public sealed class App : Application
{
    public static Func<Models.IGameAudio> AudioFactory { get; set; } = () => new Models.SilentGameAudio();
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            desktop.MainWindow = new MainWindow();
        }
        else if (ApplicationLifetime is ISingleViewApplicationLifetime singleView)
        {
            singleView.MainView = new GameSurface();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
