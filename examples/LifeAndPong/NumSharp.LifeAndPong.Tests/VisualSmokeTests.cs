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

            Assert.IsTrue(surface.Pong.IsReady);
            window.KeyPress(Key.Space, RawInputModifiers.None, PhysicalKey.None, " ");
            window.KeyPress(Key.Space, RawInputModifiers.None, PhysicalKey.None, " ");
            Assert.IsFalse(surface.Pong.IsReady);
            Assert.IsFalse(surface.Pong.IsPaused, "Auto-repeat must not toggle pause.");
            window.KeyRelease(Key.Space, RawInputModifiers.None, PhysicalKey.None, " ");

            window.KeyPress(Key.W, RawInputModifiers.None, PhysicalKey.None, "w");
            window.KeyPress(Key.Up, RawInputModifiers.None, PhysicalKey.None, "");
            window.KeyRelease(Key.W, RawInputModifiers.None, PhysicalKey.None, "w");
            surface.AdvanceSimulation(0.05);
            Assert.IsTrue(surface.Pong.PlayerVelocity < 0, "Up must remain active after W is released.");
            window.KeyRelease(Key.Up, RawInputModifiers.None, PhysicalKey.None, "");

            Click(window, surface.ButtonBounds("life-clear").Center);
            var grid = surface.LifeBounds;
            var cellWidth = grid.Width / surface.Life.Columns;
            var cellHeight = grid.Height / surface.Life.Rows;
            window.MouseDown(new Point(grid.X + 2.5 * cellWidth, grid.Y + 5.5 * cellHeight), MouseButton.Left, RawInputModifiers.None);
            window.MouseMove(new Point(grid.X + 40.5 * cellWidth, grid.Y + 5.5 * cellHeight), RawInputModifiers.LeftMouseButton);
            window.MouseUp(new Point(grid.X + 40.5 * cellWidth, grid.Y + 5.5 * cellHeight), MouseButton.Left, RawInputModifiers.None);
            Assert.AreEqual(39, surface.Life.LiveCount);
            Click(window, surface.ButtonBounds("life-pulsar").Center);
            Assert.AreEqual(48, surface.Life.LiveCount);
            Click(window, surface.ButtonBounds("life-step").Center);
            Assert.AreEqual(1L, surface.Life.Generation);
            surface.AdvanceSimulation(0.05);
            Assert.AreEqual(1L, surface.Life.Generation, "Step must leave Life paused.");

            window.KeyPress(Key.W, RawInputModifiers.None, PhysicalKey.None, "w");
            window.MouseDown(surface.LifeBounds.Center, MouseButton.Left, RawInputModifiers.None);
            Assert.IsTrue(surface.HasTransientInputForTesting);

            focusTarget.Focus();
            Assert.IsFalse(surface.HasTransientInputForTesting);
            surface.Pong.PauseForDeactivation();
            var stopAt = surface.Pong.BallPosition;
            surface.AdvanceSimulation(0.05);
            Assert.AreEqual(stopAt, surface.Pong.BallPosition);
            surface.Focus();
            window.KeyPress(Key.Tab, RawInputModifiers.None, PhysicalKey.None, "");
            window.KeyRelease(Key.Tab, RawInputModifiers.None, PhysicalKey.None, "");
            var generation = surface.Life.Generation;
            window.KeyPress(Key.Enter, RawInputModifiers.None, PhysicalKey.None, "");
            window.KeyRelease(Key.Enter, RawInputModifiers.None, PhysicalKey.None, "");
            surface.AdvanceSimulation(0.1);
            Assert.IsTrue(surface.Life.Generation > generation, "Tab + Enter must activate the first control.");

            window.Width = 1120;
            window.Height = 700;
            using var minimum = window.CaptureRenderedFrame();
            Assert.IsNotNull(minimum);
            Assert.AreEqual(new PixelSize(1120, 700), minimum.PixelSize);
            Assert.IsTrue(surface.LifeBounds.Right < surface.PongBounds.Left);
            if (Environment.GetEnvironmentVariable("LIFE_PONG_PREVIEW_PATH") is not null)
                minimum.Save(Path.ChangeExtension(output, "minimum.png"), PngBitmapEncoderOptions.Default);
            root.Children.Remove(surface);
            Assert.IsFalse(surface.HasTransientInputForTesting);
            window.Close();
        }, CancellationToken.None);

        var screenshot = new FileInfo(output);
        Assert.IsTrue(screenshot.Exists);
        Assert.IsTrue(screenshot.Length > 20_000, $"Rendered frame was unexpectedly small: {screenshot.Length} bytes.");
        TestContext.WriteLine($"Rendered preview: {output}");
    }

    private static void Click(Window window, Point point)
    {
        window.MouseDown(point, MouseButton.Left, RawInputModifiers.None);
        window.MouseUp(point, MouseButton.Left, RawInputModifiers.None);
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
