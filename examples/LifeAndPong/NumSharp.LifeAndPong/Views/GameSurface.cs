using System.Diagnostics;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Threading;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Views;

/// <summary>
/// Custom vector surface for the complete split-screen experience.
/// </summary>
public sealed class GameSurface : Control, IDisposable
{
    private static readonly FontFamily s_interfaceFont = new("avares://Avalonia.Fonts.Inter/Assets#Inter");
    private static readonly IBrush s_ink = Solid("#070A12");
    private static readonly IBrush s_panel = Solid("#0C1220");
    private static readonly IBrush s_panelRaised = Solid("#111A2C");
    private static readonly IBrush s_hairline = Solid("#22304A");
    private static readonly IBrush s_textPrimary = Solid("#F4F7FB");
    private static readonly IBrush s_textSecondary = Solid("#8B9AB2");
    private static readonly IBrush s_mint = Solid("#68F6C6");
    private static readonly IBrush s_mintDim = Solid("#183E38");
    private static readonly IBrush s_coral = Solid("#FF7D6B");
    private static readonly IBrush s_coralDim = Solid("#48251F");
    private static readonly IBrush s_amber = Solid("#FFC857");
    private static readonly IBrush s_white08 = Solid("#14FFFFFF");
    private static readonly IBrush s_white14 = Solid("#24FFFFFF");
    private static readonly IBrush s_white22 = Solid("#38FFFFFF");
    private static readonly IPen s_hairlinePen = new Pen(s_hairline, 1);

    private readonly LifeSimulation _life = new();
    private readonly PongSimulation _pong = new();
    private readonly DispatcherTimer _timer;
    private readonly List<ButtonHit> _buttonHits = new(10);
    private readonly double[] _lifeRates = [3, 6, 12, 24, 40];

    private long _lastTimestamp;
    private double _lifeAccumulator;
    private double _pongAccumulator;
    private int _lifeRateIndex = 2;
    private bool _lifeRunning = true;
    private bool _upHeld;
    private bool _downHeld;
    private bool _paintingLife;
    private bool _paintAlive;
    private bool _disposed;
    private Rect _lifeGridRect;
    private Rect _pongWorldRect;

    public GameSurface()
    {
        Focusable = true;
        ClipToBounds = true;
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(8) };
        _timer.Tick += OnFrame;
        _lastTimestamp = Stopwatch.GetTimestamp();
        _timer.Start();
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        _buttonHits.Clear();

        var bounds = Bounds;
        context.FillRectangle(s_ink, bounds);
        DrawAmbientGrid(context, bounds);

        var scale = Math.Clamp(Math.Min(bounds.Width / 1440d, bounds.Height / 900d), 0.72, 1.25);
        var margin = 28 * scale;
        var headerHeight = 72 * scale;
        var gap = 18 * scale;
        var contentTop = margin + headerHeight;

        DrawHeader(context, new Rect(margin, margin, bounds.Width - margin * 2, headerHeight - 12 * scale), scale);

        var panelWidth = (bounds.Width - margin * 2 - gap) / 2;
        var panelHeight = bounds.Height - contentTop - margin;
        Rect lifePanel = new(margin, contentTop, panelWidth, panelHeight);
        Rect pongPanel = new(margin + panelWidth + gap, contentTop, panelWidth, panelHeight);

        DrawPanelShell(context, lifePanel, s_mint, scale);
        DrawPanelShell(context, pongPanel, s_coral, scale);
        DrawLife(context, lifePanel, scale);
        DrawPong(context, pongPanel, scale);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _timer.Stop();
        _timer.Tick -= OnFrame;
        _life.Dispose();
        _disposed = true;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        Focus();
        var point = e.GetPosition(this);

        foreach (var button in _buttonHits)
        {
            if (button.Bounds.Contains(point))
            {
                ActivateButton(button.Id);
                e.Handled = true;
                return;
            }
        }

        if (TryGetLifeCell(point, out var row, out var column))
        {
            _paintAlive = _life.ToggleCell(row, column);
            _paintingLife = true;
            e.Pointer.Capture(this);
            InvalidateVisual();
            e.Handled = true;
            return;
        }

        UpdatePointerPaddle(point);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var point = e.GetPosition(this);
        UpdatePointerPaddle(point);

        if (_paintingLife && TryGetLifeCell(point, out var row, out var column))
        {
            _life.SetCell(row, column, _paintAlive);
            InvalidateVisual();
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _paintingLife = false;
        e.Pointer.Capture(null);
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        switch (e.Key)
        {
            case Key.W:
            case Key.Up:
                _upHeld = true;
                UpdateKeyboardIntent();
                e.Handled = true;
                break;
            case Key.S:
            case Key.Down:
                _downHeld = true;
                UpdateKeyboardIntent();
                e.Handled = true;
                break;
            case Key.Space:
                _pong.TogglePause();
                e.Handled = true;
                break;
            case Key.R:
                _pong.ResetMatch();
                e.Handled = true;
                break;
        }
    }

    protected override void OnKeyUp(KeyEventArgs e)
    {
        base.OnKeyUp(e);
        switch (e.Key)
        {
            case Key.W:
            case Key.Up:
                _upHeld = false;
                UpdateKeyboardIntent();
                e.Handled = true;
                break;
            case Key.S:
            case Key.Down:
                _downHeld = false;
                UpdateKeyboardIntent();
                e.Handled = true;
                break;
        }
    }

    private void OnFrame(object? sender, EventArgs e)
    {
        var timestamp = Stopwatch.GetTimestamp();
        var elapsed = Math.Min(0.05, (timestamp - _lastTimestamp) / (double)Stopwatch.Frequency);
        _lastTimestamp = timestamp;

        if (_lifeRunning)
        {
            _lifeAccumulator += elapsed;
            var interval = 1d / _lifeRates[_lifeRateIndex];
            while (_lifeAccumulator >= interval)
            {
                _life.Step();
                _lifeAccumulator -= interval;
            }
        }

        _pongAccumulator += elapsed;
        const double FixedStep = 1d / 120d;
        while (_pongAccumulator >= FixedStep)
        {
            _pong.Advance((float)FixedStep);
            _pongAccumulator -= FixedStep;
        }

        InvalidateVisual();
    }

    private static void DrawHeader(DrawingContext context, Rect area, double scale)
    {
        DrawText(context, "NUMSHARP", new Point(area.X, area.Y - 2 * scale), 12 * scale, s_mint, FontWeight.Bold, 2.5 * scale);
        DrawText(context, "LIFE / PONG LAB", new Point(area.X, area.Y + 19 * scale), 28 * scale, s_textPrimary, FontWeight.SemiBold);

        var badgeWidth = 124 * scale;
        Rect badge = new(area.Right - badgeWidth, area.Y + 7 * scale, badgeWidth, 30 * scale);
        context.DrawRectangle(s_panelRaised, s_hairlinePen, badge, 15 * scale, 15 * scale);
        context.DrawEllipse(s_mint, null, new Point(badge.X + 16 * scale, badge.Center.Y), 3 * scale, 3 * scale);
        DrawText(context, "120 HZ  /  LIVE", new Point(badge.X + 27 * scale, badge.Y + 8 * scale), 9.5 * scale, s_textSecondary, FontWeight.SemiBold, 0.9 * scale);
    }

    private static void DrawAmbientGrid(DrawingContext context, Rect bounds)
    {
        var pen = new Pen(s_white08, 1);
        const double Spacing = 44;
        for (double x = 0; x < bounds.Width; x += Spacing)
            context.DrawLine(pen, new Point(x, 0), new Point(x, bounds.Height));
        for (double y = 0; y < bounds.Height; y += Spacing)
            context.DrawLine(pen, new Point(0, y), new Point(bounds.Width, y));

        context.DrawEllipse(new SolidColorBrush(Color.FromArgb(22, 104, 246, 198)), null, new Point(bounds.Width * 0.17, bounds.Height * 0.44), bounds.Width * 0.22, bounds.Width * 0.22);
        context.DrawEllipse(new SolidColorBrush(Color.FromArgb(17, 255, 125, 107)), null, new Point(bounds.Width * 0.82, bounds.Height * 0.52), bounds.Width * 0.21, bounds.Width * 0.21);
    }

    private static void DrawPanelShell(DrawingContext context, Rect panel, IBrush accent, double scale)
    {
        context.DrawRectangle(new SolidColorBrush(Color.FromArgb(55, 0, 0, 0)), null, panel.Translate(new Vector(0, 8 * scale)), 22 * scale, 22 * scale);
        context.DrawRectangle(s_panel, s_hairlinePen, panel, 22 * scale, 22 * scale);
        context.DrawRectangle(accent, null, new Rect(panel.X + 22 * scale, panel.Y, 52 * scale, 3 * scale), 1.5 * scale, 1.5 * scale);
    }

    private void DrawLife(DrawingContext context, Rect panel, double scale)
    {
        var inset = 20 * scale;
        DrawText(context, "CONWAY FIELD", new Point(panel.X + inset, panel.Y + 17 * scale), 16 * scale, s_textPrimary, FontWeight.SemiBold, 0.5 * scale);
        DrawText(context, $"GEN {_life.Generation:000000}  ·  {_life.LiveCount:0000} LIVE", new Point(panel.X + inset, panel.Y + 43 * scale), 10 * scale, s_textSecondary, FontWeight.Medium, 0.6 * scale);

        var buttonY = panel.Y + 67 * scale;
        var buttonX = panel.X + inset;
        buttonX = DrawButton(context, "life-run", _lifeRunning ? "PAUSE" : "RUN", buttonX, buttonY, 72 * scale, scale, s_mint, _lifeRunning);
        buttonX = DrawButton(context, "life-step", "STEP", buttonX + 7 * scale, buttonY, 58 * scale, scale, s_mint, false);
        buttonX = DrawButton(context, "life-seed", "SEED", buttonX + 7 * scale, buttonY, 62 * scale, scale, s_mint, false);
        _ = DrawButton(context, "life-clear", "CLEAR", buttonX + 7 * scale, buttonY, 66 * scale, scale, s_mint, false);

        var speedRight = panel.Right - inset;
        DrawButton(context, "life-faster", "+", speedRight - 32 * scale, buttonY, 32 * scale, scale, s_mint, false);
        DrawText(context, $"{_lifeRates[_lifeRateIndex]:0}×", new Point(speedRight - 77 * scale, buttonY + 8 * scale), 10 * scale, s_textSecondary, FontWeight.Bold);
        DrawButton(context, "life-slower", "−", speedRight - 116 * scale, buttonY, 32 * scale, scale, s_mint, false);

        Rect available = new(panel.X + inset, panel.Y + 113 * scale, panel.Width - inset * 2, panel.Height - 133 * scale);
        var cellSize = Math.Min(available.Width / _life.Columns, available.Height / _life.Rows);
        var gridWidth = cellSize * _life.Columns;
        var gridHeight = cellSize * _life.Rows;
        _lifeGridRect = new Rect(available.X + (available.Width - gridWidth) / 2, available.Y + (available.Height - gridHeight) / 2, gridWidth, gridHeight);

        context.DrawRectangle(Solid("#08110F"), new Pen(s_mintDim, 1), _lifeGridRect, 10 * scale, 10 * scale);

        var gridPen = new Pen(new SolidColorBrush(Color.FromArgb(24, 104, 246, 198)), Math.Max(0.65, scale * 0.7));
        for (var column = 1; column < _life.Columns; column++)
        {
            var x = _lifeGridRect.X + column * cellSize;
            context.DrawLine(gridPen, new Point(x, _lifeGridRect.Y), new Point(x, _lifeGridRect.Bottom));
        }
        for (var row = 1; row < _life.Rows; row++)
        {
            var y = _lifeGridRect.Y + row * cellSize;
            context.DrawLine(gridPen, new Point(_lifeGridRect.X, y), new Point(_lifeGridRect.Right, y));
        }

        var pad = Math.Max(1.2, cellSize * 0.13);
        for (var row = 0; row < _life.Rows; row++)
        {
            for (var column = 0; column < _life.Columns; column++)
            {
                if (!_life.IsAlive(row, column))
                    continue;

                Rect cell = new(
                    _lifeGridRect.X + column * cellSize + pad,
                    _lifeGridRect.Y + row * cellSize + pad,
                    Math.Max(1, cellSize - pad * 2),
                    Math.Max(1, cellSize - pad * 2));
                context.DrawRectangle(new SolidColorBrush(Color.FromArgb(45, 104, 246, 198)), null, cell.Inflate(2 * scale), 3 * scale, 3 * scale);
                context.DrawRectangle(s_mint, null, cell, 2 * scale, 2 * scale);
            }
        }
    }

    private void DrawPong(DrawingContext context, Rect panel, double scale)
    {
        var inset = 20 * scale;
        DrawText(context, "KINETIC PONG", new Point(panel.X + inset, panel.Y + 17 * scale), 16 * scale, s_textPrimary, FontWeight.SemiBold, 0.5 * scale);
        DrawText(context, "PLAYER  /  PREDICTIVE AI", new Point(panel.X + inset, panel.Y + 43 * scale), 10 * scale, s_textSecondary, FontWeight.Medium, 0.6 * scale);

        var buttonY = panel.Y + 67 * scale;
        var buttonX = panel.X + inset;
        buttonX = DrawButton(context, "pong-pause", _pong.IsPaused ? "RESUME" : "PAUSE", buttonX, buttonY, 76 * scale, scale, s_coral, _pong.IsPaused);
        DrawButton(context, "pong-reset", "NEW MATCH", buttonX + 7 * scale, buttonY, 96 * scale, scale, s_coral, false);
        DrawText(context, "2% DIRECTIONAL JITTER", new Point(panel.Right - 175 * scale, buttonY + 8 * scale), 9 * scale, s_textSecondary, FontWeight.Bold, 0.5 * scale);

        Rect available = new(panel.X + inset, panel.Y + 113 * scale, panel.Width - inset * 2, panel.Height - 133 * scale);
        var worldScale = Math.Min(available.Width / PongSimulation.WorldWidth, available.Height / PongSimulation.WorldHeight);
        var worldWidth = PongSimulation.WorldWidth * worldScale;
        var worldHeight = PongSimulation.WorldHeight * worldScale;
        _pongWorldRect = new Rect(available.X + (available.Width - worldWidth) / 2, available.Y + (available.Height - worldHeight) / 2, worldWidth, worldHeight);

        context.DrawRectangle(Solid("#110B0D"), new Pen(s_coralDim, 1), _pongWorldRect, 10 * scale, 10 * scale);
        DrawArenaMarkings(context, worldScale, scale);

        for (var i = 0; i < _pong.Trail.Count; i++)
        {
            var alpha = (i + 1f) / _pong.Trail.Count;
            var trailPoint = WorldToScreen(_pong.Trail[i], worldScale);
            var radius = PongSimulation.BallRadius * worldScale * (0.28 + alpha * 0.34);
            context.DrawEllipse(new SolidColorBrush(Color.FromArgb((byte)(alpha * 78), 255, 125, 107)), null, trailPoint, radius, radius);
        }

        DrawPaddle(context, 36, _pong.PlayerY, worldScale, s_mint, true, scale);
        DrawPaddle(context, PongSimulation.WorldWidth - 36, _pong.AiY, worldScale, s_coral, false, scale);

        var ball = WorldToScreen(_pong.BallPosition, worldScale);
        var ballRadius = PongSimulation.BallRadius * worldScale;
        context.DrawEllipse(new SolidColorBrush(Color.FromArgb(35, 255, 200, 87)), null, ball, ballRadius * 2.1, ballRadius * 2.1);
        context.DrawEllipse(new SolidColorBrush(Color.FromArgb(75, 255, 200, 87)), null, ball, ballRadius * 1.45, ballRadius * 1.45);
        context.DrawEllipse(s_amber, null, ball, ballRadius, ballRadius);
        context.DrawEllipse(s_textPrimary, null, new Point(ball.X - ballRadius * 0.25, ball.Y - ballRadius * 0.25), ballRadius * 0.26, ballRadius * 0.26);

        DrawScore(context, worldScale, scale);
        DrawPongOverlay(context, scale);
    }

    private void DrawArenaMarkings(DrawingContext context, double worldScale, double scale)
    {
        var centerX = _pongWorldRect.Center.X;
        var dashPen = new Pen(s_white14, Math.Max(1, 1.2 * scale));
        var dash = 10 * worldScale;
        for (var y = _pongWorldRect.Y + 20 * worldScale; y < _pongWorldRect.Bottom - 20 * worldScale; y += dash * 2)
            context.DrawLine(dashPen, new Point(centerX, y), new Point(centerX, Math.Min(y + dash, _pongWorldRect.Bottom)));

        context.DrawEllipse(null, new Pen(s_white08, 1), _pongWorldRect.Center, 82 * worldScale, 82 * worldScale);
        context.DrawLine(new Pen(s_coralDim, 2 * scale), new Point(_pongWorldRect.X, _pongWorldRect.Y), new Point(_pongWorldRect.Right, _pongWorldRect.Y));
        context.DrawLine(new Pen(s_coralDim, 2 * scale), new Point(_pongWorldRect.X, _pongWorldRect.Bottom), new Point(_pongWorldRect.Right, _pongWorldRect.Bottom));
    }

    private void DrawPaddle(DrawingContext context, float worldX, float worldY, double worldScale, IBrush color, bool player, double scale)
    {
        Rect paddle = new(
            _pongWorldRect.X + (worldX - PongSimulation.PaddleWidth / 2) * worldScale,
            _pongWorldRect.Y + (worldY - PongSimulation.PaddleHeight / 2) * worldScale,
            PongSimulation.PaddleWidth * worldScale,
            PongSimulation.PaddleHeight * worldScale);
        context.DrawRectangle(new SolidColorBrush(player ? Color.FromArgb(35, 104, 246, 198) : Color.FromArgb(35, 255, 125, 107)), null, paddle.Inflate(7 * scale), 7 * scale, 7 * scale);
        context.DrawRectangle(color, null, paddle, 5 * scale, 5 * scale);
        context.DrawRectangle(s_textPrimary, null, new Rect(paddle.X + paddle.Width * 0.25, paddle.Y + paddle.Height * 0.08, paddle.Width * 0.2, paddle.Height * 0.45), 2 * scale, 2 * scale);
    }

    private void DrawScore(DrawingContext context, double worldScale, double scale)
    {
        var player = _pong.PlayerScore.ToString(CultureInfo.InvariantCulture);
        var ai = _pong.AiScore.ToString(CultureInfo.InvariantCulture);
        var scoreY = _pongWorldRect.Y + 24 * worldScale;
        DrawTextCentered(context, player, new Point(_pongWorldRect.Center.X - 62 * worldScale, scoreY), 40 * scale, s_mint, FontWeight.Bold);
        DrawTextCentered(context, ai, new Point(_pongWorldRect.Center.X + 62 * worldScale, scoreY), 40 * scale, s_coral, FontWeight.Bold);
    }

    private void DrawPongOverlay(DrawingContext context, double scale)
    {
        string? title = null;
        string? detail = null;
        if (_pong.IsMatchOver)
        {
            title = _pong.PlayerScore > _pong.AiScore ? "YOU WIN" : "AI WINS";
            detail = "PRESS R OR NEW MATCH";
        }
        else if (_pong.IsPaused)
        {
            title = "PAUSED";
            detail = "SPACE TO RESUME";
        }
        else if (_pong.ServeCountdown > 0)
        {
            title = Math.Max(1, (int)Math.Ceiling(_pong.ServeCountdown * 3)).ToString(CultureInfo.InvariantCulture);
            detail = "SERVE";
        }

        if (title is null)
            return;

        Rect overlay = new(_pongWorldRect.Center.X - 100 * scale, _pongWorldRect.Center.Y - 47 * scale, 200 * scale, 94 * scale);
        context.DrawRectangle(new SolidColorBrush(Color.FromArgb(205, 7, 10, 18)), new Pen(s_white22, 1), overlay, 16 * scale, 16 * scale);
        DrawTextCentered(context, title, new Point(overlay.Center.X, overlay.Y + 16 * scale), 27 * scale, s_textPrimary, FontWeight.Bold);
        DrawTextCentered(context, detail!, new Point(overlay.Center.X, overlay.Y + 58 * scale), 9 * scale, s_textSecondary, FontWeight.Bold, 1 * scale);
    }

    private double DrawButton(DrawingContext context, string id, string label, double x, double y, double width, double scale, IBrush accent, bool active)
    {
        Rect bounds = new(x, y, width, 29 * scale);
        _buttonHits.Add(new ButtonHit(id, bounds));
        context.DrawRectangle(active ? accent : s_panelRaised, new Pen(active ? accent : s_hairline, 1), bounds, 8 * scale, 8 * scale);
        DrawTextCentered(context, label, bounds.Center.WithY(bounds.Y + 7.4 * scale), 9 * scale, active ? s_ink : s_textSecondary, FontWeight.Bold, 0.55 * scale);
        return bounds.Right;
    }

    private void ActivateButton(string id)
    {
        switch (id)
        {
            case "life-run":
                _lifeRunning = !_lifeRunning;
                _lifeAccumulator = 0;
                break;
            case "life-step":
                _life.Step();
                break;
            case "life-seed":
                _life.Reseed();
                break;
            case "life-clear":
                _life.Clear();
                break;
            case "life-slower":
                _lifeRateIndex = Math.Max(0, _lifeRateIndex - 1);
                break;
            case "life-faster":
                _lifeRateIndex = Math.Min(_lifeRates.Length - 1, _lifeRateIndex + 1);
                break;
            case "pong-pause":
                _pong.TogglePause();
                break;
            case "pong-reset":
                _pong.ResetMatch();
                break;
        }

        InvalidateVisual();
    }

    private bool TryGetLifeCell(Point point, out int row, out int column)
    {
        row = -1;
        column = -1;
        if (!_lifeGridRect.Contains(point) || _lifeGridRect.Width <= 0 || _lifeGridRect.Height <= 0)
            return false;

        column = Math.Clamp((int)((point.X - _lifeGridRect.X) / _lifeGridRect.Width * _life.Columns), 0, _life.Columns - 1);
        row = Math.Clamp((int)((point.Y - _lifeGridRect.Y) / _lifeGridRect.Height * _life.Rows), 0, _life.Rows - 1);
        return true;
    }

    private void UpdatePointerPaddle(Point point)
    {
        if (!_pongWorldRect.Contains(point) || _pongWorldRect.Height <= 0)
            return;

        var worldY = (float)((point.Y - _pongWorldRect.Y) / _pongWorldRect.Height * PongSimulation.WorldHeight);
        _pong.SetPointerTarget(worldY);
    }

    private void UpdateKeyboardIntent() => _pong.SetKeyboardIntent((_downHeld ? 1f : 0f) - (_upHeld ? 1f : 0f));

    private Point WorldToScreen(System.Numerics.Vector2 value, double scale) =>
        new(_pongWorldRect.X + value.X * scale, _pongWorldRect.Y + value.Y * scale);

    private static void DrawText(
        DrawingContext context,
        string text,
        Point point,
        double fontSize,
        IBrush brush,
        FontWeight weight,
        double letterSpacing = 0)
    {
        _ = letterSpacing;
        var formatted = new FormattedText(
            text,
            CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight,
            new Typeface(s_interfaceFont, FontStyle.Normal, weight),
            fontSize,
            brush);
        context.DrawText(formatted, point);
    }

    private static void DrawTextCentered(
        DrawingContext context,
        string text,
        Point anchor,
        double fontSize,
        IBrush brush,
        FontWeight weight,
        double letterSpacing = 0)
    {
        _ = letterSpacing;
        var formatted = new FormattedText(
            text,
            CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight,
            new Typeface(s_interfaceFont, FontStyle.Normal, weight),
            fontSize,
            brush);
        context.DrawText(formatted, new Point(anchor.X - formatted.Width / 2, anchor.Y));
    }

    private static SolidColorBrush Solid(string hex) => new(Color.Parse(hex));

    private readonly record struct ButtonHit(string Id, Rect Bounds);
}

internal static class PointExtensions
{
    public static Point WithY(this Point point, double y) => new(point.X, y);
}
