using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using NumSharp.LifeAndPong.Views;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class VisualSmokeTests
{
    public TestContext TestContext { get; set; } = null!;

    [TestMethod]
    public async Task Complete_Surface_Renders_At_Desktop_Size()
    {
        await using var session = HeadlessUnitTestSession.StartNew(typeof(HeadlessAppBuilder));
        var output = Environment.GetEnvironmentVariable("LIFE_PONG_PREVIEW_PATH")
            ?? Path.Combine(Path.GetTempPath(), $"numsharp-life-pong-{Guid.NewGuid():N}.png");

        await session.Dispatch(() =>
        {
            using var surface = new GameSurface();
            var window = new Window
            {
                Width = 1440,
                Height = 900,
                Content = surface
            };

            window.Show();
            using var frame = window.CaptureRenderedFrame();
            Assert.IsNotNull(frame);
            frame.Save(output);
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
