using System.Globalization;
using System.Numerics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Views;

/// <summary>All game marks are drawn here; effects have their own clock and never mutate gameplay.</summary>
internal sealed class ArenaView(ArcadeSession game, PlayerProfile profile) : Control
{
    internal static readonly IBrush Ink = Brush("#080F17"), Raised = Brush("#172734"), Line = Brush("#29404B");
    internal static readonly IBrush Mint = Brush("#79F3BD"), Coral = Brush("#FFAB7C"), Secondary = Brush("#ACBEC6");
    private static readonly IBrush Field = Brush("#0A1A1A"), Arena = Brush("#0B151F"), CellGlow = Brush("#2079F3BD"), HitGlow = Brush("#25FFAB7C"), BallGlow = Brush("#30DFF9EE");
    private static readonly Typeface Font = new(new FontFamily("avares://Avalonia.Fonts.Inter/Assets#Inter"), FontStyle.Normal, FontWeight.Medium);
    private readonly List<ParticleEffect> _effects = [];
    private readonly List<Vector2> _trail = [];
    private readonly Random _cosmetics = new(731);
    private double _clock, _birthGlow, _sectorGlow;
    private Vector2 _lastBall;
    private Rect _world;
    private double _scale;
    internal Rect WorldBounds => _world;
    internal Point ToWorld(Point point) => _scale <= 0 ? new(-1, -1) : new((point.X - _world.X) / _scale, (point.Y - _world.Y) / _scale);
    internal Point ToScreen(Vector2 point) => new(_world.X + point.X * _scale, _world.Y + point.Y * _scale);

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        var available = new Rect(0, 0, Math.Max(1, Bounds.Width), Math.Max(1, Bounds.Height));
        _scale = Math.Min(available.Width / ArcadeSession.Width, available.Height / ArcadeSession.Height);
        _world = new Rect((available.Width - ArcadeSession.Width * _scale) / 2, (available.Height - ArcadeSession.Height * _scale) / 2, ArcadeSession.Width * _scale, ArcadeSession.Height * _scale);
        context.DrawRectangle(Arena, new Pen(profile.HighContrast ? Secondary : Line, 1), _world, 14, 14);
        using (context.PushClip(_world))
        {
            context.DrawRectangle(Field, null, new Rect(_world.X, _world.Y, _world.Width / 2, _world.Height));
            var center = _world.Center.X;
            var phaseColor = game.Frozen ? Coral : Mint;
            var hairline = new Pen(profile.HighContrast ? Secondary : Line, 1);
            for (var y = _world.Y + 72 * _scale; y < _world.Bottom - 48 * _scale; y += 28 * _scale)
                context.DrawLine(hairline, new Point(center, y), new Point(center, y + 10 * _scale));
            var phase = game.ReturnAssist ? "RETURN" : game.Frozen ? "SHATTER" : game.Growing ? "GROW" : "COLONY";
            DrawText(context, phase, ToScreen(new Vector2(48, 20)), Math.Max(12, 17 * _scale), phaseColor);
            DrawText(context, game.Frozen ? "EVOLUTION FROZEN" : game.Growing ? "LIFE IS BREATHING" : "WAITING FOR YOUR SHOT", ToScreen(new Vector2(215, 23)), Math.Max(11, 12 * _scale), Secondary);
            DrawText(context, "YOU", ToScreen(new Vector2(1497, 23)), Math.Max(12, 16 * _scale), Brushes.White);
            DrawText(context, $"{game.Life.LiveCount} LIVING", ToScreen(new Vector2(48, 860)), Math.Max(11, 13 * _scale), Secondary);
            if (game.Replenishing || _birthGlow > 0)
                DrawText(context, "NEW LIFE", ToScreen(new Vector2(525, 860)), Math.Max(11, 13 * _scale), Mint);

            var opacity = game.Growing && !profile.ReducedMotion && !profile.HighContrast ? .88 + .12 * Math.Sin(_clock * 3.2) : 1;
            using (context.PushOpacity(opacity))
            {
                for (var row = 0; row < game.Life.Rows; row++) for (var col = 0; col < game.Life.Columns; col++)
                    {
                        if (!game.Life.IsAlive(row, col)) continue;
                        var point = ToScreen(new Vector2(ArcadeSession.FieldX + col * ArcadeSession.CellPitch + 1, ArcadeSession.FieldY + row * ArcadeSession.CellPitch + 1));
                        var cell = new Rect(point.X, point.Y, ArcadeSession.CellSize * _scale, ArcadeSession.CellSize * _scale);
                        if (!profile.ReducedMotion && !profile.HighContrast) context.DrawRectangle(game.Frozen ? HitGlow : CellGlow, null, cell.Inflate(2 * _scale), 4 * _scale, 4 * _scale);
                        context.DrawRectangle(profile.HighContrast ? Brushes.White : phaseColor, null, cell, 2 * _scale, 2 * _scale);
                        if (game.Frozen) context.DrawRectangle(Ink, null, new Rect(cell.X + 3 * _scale, cell.Y + 3 * _scale, Math.Max(1, cell.Width - 6 * _scale), 2 * _scale));
                    }
            }
            var wallPen = new Pen(profile.HighContrast ? Brushes.White : Mint, 2 * _scale);
            context.DrawLine(wallPen, ToScreen(new Vector2(0, 0)), ToScreen(new Vector2(1600, 0)));
            context.DrawLine(wallPen, ToScreen(new Vector2(0, 900)), ToScreen(new Vector2(1600, 900)));
            context.DrawLine(wallPen, ToScreen(new Vector2(0, 0)), ToScreen(new Vector2(0, 900)));
            context.DrawLine(new Pen(Coral, 1), ToScreen(new Vector2(1597, 0)), ToScreen(new Vector2(1597, 900)));
            if (!profile.ReducedMotion)
                for (var i = 0; i < _trail.Count; i++)
                    using (context.PushOpacity((double)i / _trail.Count * .28))
                    {
                        var radius = ArcadeSession.Radius * _scale * (.25 + .55 * i / _trail.Count);
                        context.DrawEllipse(Brushes.White, null, ToScreen(_trail[i]), radius, radius);
                    }
            var paddleTop = ToScreen(new Vector2(ArcadeSession.PaddleX - ArcadeSession.PaddleWidth / 2, game.PaddleY - ArcadeSession.PaddleHeight / 2));
            var paddle = new Rect(paddleTop.X, paddleTop.Y, ArcadeSession.PaddleWidth * _scale, ArcadeSession.PaddleHeight * _scale);
            if (!profile.HighContrast) context.DrawRectangle(BallGlow, null, paddle.Inflate(4 * _scale), 7 * _scale, 7 * _scale);
            context.DrawRectangle(Brushes.White, null, paddle, 4 * _scale, 4 * _scale);
            context.DrawRectangle(Mint, null, new Rect(paddle.X + 4 * _scale, paddle.Y + 10 * _scale, 3 * _scale, paddle.Height - 20 * _scale), 1, 1);
            var ball = ToScreen(game.Ball); var ballRadius = ArcadeSession.Radius * _scale;
            if (!profile.HighContrast) context.DrawEllipse(BallGlow, null, ball, ballRadius * 2.2, ballRadius * 2.2);
            context.DrawEllipse(Brushes.White, null, ball, ballRadius, ballRadius);
            foreach (var effect in _effects)
            {
                var point = ToScreen(effect.Position);
                using (context.PushOpacity(Math.Min(1, effect.Remaining * 2)))
                {
                    if (effect.Text is not null) DrawText(context, effect.Text, point, Math.Max(12, 21 * _scale), Coral);
                    else if (!profile.ReducedMotion) context.DrawRectangle(effect.Warm ? Coral : Mint, null, new Rect(point.X, point.Y, 3 * _scale, 3 * _scale));
                }
            }
            if (_sectorGlow > 0) DrawText(context, $"SECTOR {game.Sector:00}", ToScreen(new Vector2(1020, 110)), 22 * _scale, Mint);
        }
    }

    internal void OnEvent(ArcadeEvent item)
    {
        if (item.Kind == ArcadeEventKind.Birth) _birthGlow = 1;
        if (item.Kind == ArcadeEventKind.Sector) _sectorGlow = 2;
        if (item.Kind == ArcadeEventKind.Miss) _trail.Clear();
        if (item.Kind == ArcadeEventKind.Cell)
        {
            _effects.Add(new ParticleEffect(item.Position + new Vector2(5, -25), new Vector2(0, -35), .8, $"+{item.Value}", true));
            if (!profile.ReducedMotion)
                for (var i = 0; i < 7; i++)
                {
                    var angle = _cosmetics.NextDouble() * Math.PI * 2;
                    var speed = 35 + _cosmetics.NextDouble() * 90;
                    _effects.Add(new ParticleEffect(item.Position, new Vector2((float)(Math.Cos(angle) * speed), (float)(Math.Sin(angle) * speed)), .45, null, true));
                }
        }
        if (_effects.Count > 160) _effects.RemoveRange(0, _effects.Count - 160);
    }
    internal void AdvanceEffects(double dt)
    {
        _clock += dt; _birthGlow = Math.Max(0, _birthGlow - dt); _sectorGlow = Math.Max(0, _sectorGlow - dt);
        for (var i = _effects.Count - 1; i >= 0; i--)
        {
            var e = _effects[i]; e.Remaining -= dt;
            if (!profile.ReducedMotion) e.Position += e.Velocity * (float)dt;
            if (e.Remaining <= 0) _effects.RemoveAt(i);
        }
        if (game.Ball != _lastBall)
        {
            if (Vector2.Distance(game.Ball, _lastBall) > 200) _trail.Clear();
            _trail.Add(game.Ball); _lastBall = game.Ball;
            if (_trail.Count > 18) _trail.RemoveAt(0);
        }
    }
    internal void ClearEffects() { _effects.Clear(); _trail.Clear(); _birthGlow = _sectorGlow = 0; }
    private static void DrawText(DrawingContext context, string text, Point point, double size, IBrush brush)
    { context.DrawText(new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Font, size, brush), point); }
    private static IBrush Brush(string hex) => new SolidColorBrush(Color.Parse(hex));
    private sealed class ParticleEffect(Vector2 position, Vector2 velocity, double remaining, string? text, bool warm)
    {
        public Vector2 Position = position;
        public readonly Vector2 Velocity = velocity;
        public double Remaining = remaining;
        public readonly string? Text = text;
        public readonly bool Warm = warm;
    }
}
