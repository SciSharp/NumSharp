using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media.Imaging;
using NumSharp.LifeAndPong.Views;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class VisualSmokeTests
{
    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task Surface_Renders_And_Clears_Transient_Input_On_Focus_Loss()
    {
        await using var session = HeadlessUnitTestSession.StartNew(typeof(HeadlessAppBuilder));
        var output = Environment.GetEnvironmentVariable("LIFE_PONG_PREVIEW_PATH")
            ?? Path.Combine(Path.GetTempPath(), $"numsharp-life-pong-{Guid.NewGuid():N}.png");

        await session.Dispatch(() =>
        {
            using var surface = new GameSurface(startAnimation: false);
            var focusTarget = new Button
            {
                Width = 1,
                Height = 1,
                Opacity = 0.01,
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Bottom
            };
            var root = new Grid();
            root.Children.Add(surface);
            root.Children.Add(focusTarget);
            var window = new Window
            {
                Width = 1440,
                Height = 900,
                Content = root
            };

            window.Show();
            surface.Focus();
            using var frame = window.CaptureRenderedFrame();
            Assert.IsNotNull(frame);
            frame.Save(output, PngBitmapEncoderOptions.Default);

            window.KeyPress(Key.W, RawInputModifiers.None, PhysicalKey.None, "w");
            window.MouseDown(new Point(100, 300), MouseButton.Left, RawInputModifiers.None);
            Assert.IsTrue(surface.HasTransientInputForTesting);

            focusTarget.Focus();
            Assert.IsFalse(surface.HasTransientInputForTesting);
            window.Close();
        }, CancellationToken.None);

        var screenshot = new FileInfo(output);
        Assert.IsTrue(screenshot.Exists);
        Assert.IsTrue(screenshot.Length > 20_000, $"Rendered frame was unexpectedly small: {screenshot.Length} bytes.");
        TestContext.WriteLine($"Rendered preview: {output}");
    }
}

public static class HeadlessAppBuilder
{
    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UseSkia()
            .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false })
            .WithInterFont();
}
