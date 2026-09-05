using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Media.Imaging;
using NumSharp.LifeAndPong.Models;
using NumSharp.LifeAndPong.Views;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class VisualSmokeTests
{
    [TestMethod]
    public async Task Arcade_Real_Input_Ready_Play_Pause_Options_Miss_And_Retry()
    {
        await using var session = HeadlessUnitTestSession.StartNew(typeof(HeadlessAppBuilder));
        await session.Dispatch(() =>
        {
            using var surface = new GameSurface(false);
            var window = new Window { Width = 1440, Height = 900, Content = surface };
            window.Show(); surface.Focus();
            Capture(window, "preview.ready.png");
            Assert.AreEqual(RunState.Ready, surface.Session.State);
            // Actual routed mouse input to a real accessible button: one click must launch.
            Click(window, surface.PrimaryButton.TranslatePoint(new Point(surface.PrimaryButton.Bounds.Width / 2, surface.PrimaryButton.Bounds.Height / 2), window)!.Value);
            Assert.AreEqual(RunState.Playing, surface.Session.State);
            Press(window, Key.Space); Press(window, Key.Space);
            Assert.AreEqual(RunState.Paused, surface.Session.State, "Repeat must not resume."); Release(window, Key.Space);
            Press(window, Key.Space); Release(window, Key.Space); Assert.AreEqual(RunState.Playing, surface.Session.State);
            Press(window, Key.W); Press(window, Key.Up); Release(window, Key.W);
            surface.AdvanceSimulation(.05); Assert.IsTrue(surface.Session.PaddleVelocity < 0);
            Release(window, Key.Up);
            var world = surface.Arena.WorldBounds;
            var target = surface.Arena.TranslatePoint(new Point(world.X + world.Width * .90, world.Y + world.Height * .75), window)!.Value;
            window.MouseDown(target, MouseButton.Left, RawInputModifiers.None);
            Assert.IsTrue(surface.HasTransientInputForTesting);
            surface.SuspendPlay(); Assert.IsFalse(surface.HasTransientInputForTesting);
            var ball = surface.Session.Ball; var generation = surface.Session.Life.Generation;
            surface.AdvanceSimulation(.1); Assert.AreEqual(ball, surface.Session.Ball); Assert.AreEqual(generation, surface.Session.Life.Generation);
            window.MouseUp(target, MouseButton.Left, RawInputModifiers.None);
            surface.Focus(); Press(window, Key.Space); Release(window, Key.Space);
            for (var frame = 0; frame < 120 * 12; frame++)
            {
                surface.Session.SetPointerTarget(ArcadeSessionTests.Predict(surface.Session));
                surface.AdvanceSimulation(1d / 120);
            }
            Assert.IsTrue(surface.Session.Destroyed > 0);
            Capture(window, "preview.png");
            window.Width = 1120; window.Height = 700;
            Capture(window, "preview.minimum.png");
            Assert.IsTrue(surface.Arena.WorldBounds.Width > 800);
            surface.SuspendPlay(); Capture(window, "preview.paused.png");
            // Tab enters native controls and focus is visible; it must not launch accidentally.
            surface.Focus(); Press(window, Key.Tab); Release(window, Key.Tab);
            Assert.AreEqual(RunState.Paused, surface.Session.State);
            for (var i = surface.Session.Lives; i > 0; i--) ArcadeSessionTests.Miss(surface.Session);
            surface.AdvanceSimulation(1d / 120); Capture(window, "preview.gameover.png");
            Assert.AreEqual(RunState.GameOver, surface.Session.State);
            surface.Focus(); Press(window, Key.Enter); Release(window, Key.Enter);
            Assert.AreEqual(RunState.Playing, surface.Session.State); Assert.AreEqual(3, surface.Session.Lives); Assert.AreEqual(0L, surface.Session.Score);
            window.Content = null; Assert.IsFalse(surface.HasTransientInputForTesting); Assert.AreEqual(RunState.Paused, surface.Session.State);
            window.Close();
        }, CancellationToken.None);
    }

    [TestMethod]
    public async Task High_Contrast_Reduced_Motion_And_Long_Scores_Render()
    {
        await using var session = HeadlessUnitTestSession.StartNew(typeof(HeadlessAppBuilder));
        await session.Dispatch(() =>
        {
            var profile = new PlayerProfile { HighContrast = true, ReducedMotion = true };
            using var surface = new GameSurface(false, profile, new SilentGameAudio());
            var window = new Window { Width = 1120, Height = 700, Content = surface }; window.Show();
            surface.Session.LaunchOrResume(); surface.Session.SetScoreForTesting(long.MaxValue); surface.AdvanceSimulation(1d / 120);
            Capture(window, "preview.accessible.png");
            Assert.AreEqual(long.MaxValue, surface.Session.Score);
            window.Close();
        }, CancellationToken.None);
    }
    private static void Capture(Window window, string name)
    {
        using var frame = window.CaptureRenderedFrame(); Assert.IsNotNull(frame);
        Assert.IsTrue(frame.PixelSize.Width >= 1120);
        var directory = Environment.GetEnvironmentVariable("LIFE_ARCADE_PREVIEW_DIR");
        if (directory is not null) { Directory.CreateDirectory(directory); frame.Save(Path.Combine(directory, name), PngBitmapEncoderOptions.Default); }
    }
    private static void Press(Window w, Key k) => w.KeyPress(k, RawInputModifiers.None, PhysicalKey.None, "");
    private static void Release(Window w, Key k) => w.KeyRelease(k, RawInputModifiers.None, PhysicalKey.None, "");
    private static void Click(Window window, Point point) { window.MouseDown(point, MouseButton.Left, RawInputModifiers.None); window.MouseUp(point, MouseButton.Left, RawInputModifiers.None); }
}
public static class HeadlessAppBuilder
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UseSkia().UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false }).WithInterFont();
}
