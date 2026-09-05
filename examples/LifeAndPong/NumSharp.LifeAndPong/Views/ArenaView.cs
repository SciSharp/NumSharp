using System.Numerics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Immutable;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Views;

/// <summary>Text-free playfield. All effects are cosmetic and use an independent clock and RNG.</summary>
internal sealed class ArenaView(ArcadeSession game, PlayerProfile profile) : Control
{
    internal static readonly IBrush Ink = Brush("#080F17"), Raised = Brush("#172734"), Line = Brush("#29404B");
    internal static readonly IBrush Mint = Brush("#79F3BD"), Coral = Brush("#FFAB7C"), Secondary = Brush("#ACBEC6");
    internal static readonly IBrush Gold = Brush("#FFD46B"), Violet = Brush("#BA9CFF"), Pink = Brush("#FF83D5"), Cyan = Brush("#6FEFFF");
    private static readonly IBrush Field = Brush("#0A1A1A"), Arena = Brush("#0B151F"), CellGlow = Brush("#2079F3BD"), HitGlow = Brush("#25FFAB7C"), BallGlow = Brush("#30DFF9EE");
    private static readonly IBrush[] Prism = [Cyan, Violet, Pink, Gold];
    private readonly List<Spark> _sparks = [];
    private readonly List<Shockwave> _waves = [];
    private readonly List<Vector2> _trail = [];
    private readonly Random _cosmetics = new(731);
    private double _clock;
    private Vector2 _lastBall;
    private Rect _world;
    private double _scale;
    internal Rect WorldBounds => _world;
    internal double PhaseBoundaryX => _world.X + ArcadeSession.Midline * _scale;
    internal int SparkCount => _sparks.Count;
    internal int WaveCount => _waves.Count;
    internal static IBrush TierBrush(int tier) => tier switch { 1 => Gold, 2 => Violet, 3 => Pink, _ => Coral };
    internal static string TierName(int tier) => tier switch { 1 => "SURGE", 2 => "OVERDRIVE", 3 => "SUPERNOVA", _ => "BUILD YOUR SHOT" };
    internal Point ToWorld(Point point) => _scale <= 0 ? new(-1, -1) : new((point.X - _world.X) / _scale, (point.Y - _world.Y) / _scale);
    internal Point ToScreen(Vector2 point) => new(_world.X + point.X * _scale, _world.Y + point.Y * _scale);

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        if (game.IsDisposed) return;
        var width = Math.Max(1, Bounds.Width); var height = Math.Max(1, Bounds.Height);
        _scale = Math.Min(width / ArcadeSession.Width, height / ArcadeSession.Height);
        _world = new Rect((width - ArcadeSession.Width * _scale) / 2, (height - ArcadeSession.Height * _scale) / 2, ArcadeSession.Width * _scale, ArcadeSession.Height * _scale);
        context.DrawRectangle(Arena, new Pen(profile.HighContrast ? Secondary : Line, 1), _world, 12, 12);
        using (context.PushClip(_world))
        {
            context.DrawRectangle(Field, null, new Rect(_world.X, _world.Y, _world.Width * ArcadeSession.ColonyFraction, _world.Height));
            var tier = game.EffectTier;
            var accent = profile.HighContrast ? Brushes.White : TierBrush(tier);
            var phaseColor = game.Frozen ? accent : Mint;
            var hairline = new Pen(profile.HighContrast ? Secondary : Line, 1);
            for (var y = _world.Y + 18 * _scale; y < _world.Bottom - 18 * _scale; y += 28 * _scale)
                context.DrawLine(hairline, new Point(PhaseBoundaryX, y), new Point(PhaseBoundaryX, y + 10 * _scale));
            var opacity = game.Growing && !profile.ReducedMotion && !profile.HighContrast ? .88 + .12 * Math.Sin(_clock * 3.2) : 1;
            using (context.PushOpacity(opacity))
                for (var row = 0; row < game.Life.Rows; row++) for (var col = 0; col < game.Life.Columns; col++)
                {
                    if (!game.Life.IsAlive(row, col)) continue;
                    var point = ToScreen(new Vector2(ArcadeSession.FieldX + col * ArcadeSession.CellPitch + 1, ArcadeSession.FieldY + row * ArcadeSession.CellPitch + 1));
                    var cell = new Rect(point.X, point.Y, ArcadeSession.CellSize * _scale, ArcadeSession.CellSize * _scale);
                    if (!profile.ReducedMotion && !profile.HighContrast) context.DrawRectangle(game.Frozen ? HitGlow : CellGlow, null, cell.Inflate(2 * _scale), 4 * _scale, 4 * _scale);
                    context.DrawRectangle(profile.HighContrast ? Brushes.White : phaseColor, null, cell, 2 * _scale, 2 * _scale);
                    if (game.Frozen) context.DrawRectangle(Ink, null, new Rect(cell.X + 3 * _scale, cell.Y + 3 * _scale, Math.Max(1, cell.Width - 6 * _scale), 2 * _scale));
                }
            var wallPen = new Pen(profile.HighContrast ? Brushes.White : Mint, 2 * _scale);
            context.DrawLine(wallPen, ToScreen(Vector2.Zero), ToScreen(new Vector2(ArcadeSession.Width, 0)));
            context.DrawLine(wallPen, ToScreen(new Vector2(0, ArcadeSession.Height)), ToScreen(new Vector2(ArcadeSession.Width, ArcadeSession.Height)));
            context.DrawLine(wallPen, ToScreen(Vector2.Zero), ToScreen(new Vector2(0, ArcadeSession.Height)));
            context.DrawLine(new Pen(Coral, 1), ToScreen(new Vector2(ArcadeSession.Width - 3, 0)), ToScreen(new Vector2(ArcadeSession.Width - 3, ArcadeSession.Height)));

            // Persistent tier styling is edge-bound, so even the top tier leaves the ball readable.
            if (tier > 0)
            {
                var length = (45 + tier * 28) * _scale;
                var pen = new Pen(accent, (2 + tier) * _scale);
                foreach (var corner in new[] { _world.TopLeft, _world.TopRight, _world.BottomLeft, _world.BottomRight })
                {
                    var sx = corner.X < _world.Center.X ? 1 : -1;
                    var sy = corner.Y < _world.Center.Y ? 1 : -1;
                    context.DrawLine(pen, corner, new Point(corner.X + sx * length, corner.Y));
                    context.DrawLine(pen, corner, new Point(corner.X, corner.Y + sy * length));
                }
            }
            if (!profile.ReducedMotion)
                for (var i = 0; i < _trail.Count; i++)
                {
                    var progress = (double)i / _trail.Count;
                    using (context.PushOpacity(progress * (.24 + tier * .04)))
                    {
                        var radius = ArcadeSession.Radius * _scale * (.2 + progress * .55);
                        var point = ToScreen(_trail[i]);
                        context.DrawEllipse(tier == 0 ? Brushes.White : accent, null, point, radius, radius);
                        if (tier >= 2)
                        {
                            var offset = (2 + tier) * _scale * (1 - progress);
                            context.DrawEllipse(tier == 3 ? Cyan : Pink, null, point + new Avalonia.Vector(0, offset), radius * .55, radius * .55);
                            if (tier == 3) context.DrawEllipse(Gold, null, point - new Avalonia.Vector(0, offset), radius * .45, radius * .45);
                        }
                    }
                }
            foreach (var wave in _waves)
                if (!profile.ReducedMotion && !profile.HighContrast)
                    using (context.PushOpacity(Math.Clamp(wave.Remaining / .9, 0, .65)))
                    {
                        var radius = (1 - wave.Remaining / .9) * wave.Speed * _scale;
                        context.DrawEllipse(null, new Pen(wave.Color, 2 * _scale), ToScreen(wave.Position), radius, radius * wave.Flatness);
                    }
            foreach (var spark in _sparks)
                if (!profile.ReducedMotion)
                    using (context.PushOpacity(Math.Min(1, spark.Remaining * 2)))
                    {
                        var point = ToScreen(spark.Position);
                        if (spark.Streak) context.DrawLine(new Pen(spark.Color, 2 * _scale), point, ToScreen(spark.Position - spark.Velocity * .025f));
                        else context.DrawRectangle(spark.Color, null, new Rect(point.X, point.Y, 3 * _scale, 3 * _scale));
                    }
            var paddleTop = ToScreen(new Vector2(ArcadeSession.PaddleX - ArcadeSession.PaddleWidth / 2, game.PaddleY - ArcadeSession.PaddleHeight / 2));
            var paddle = new Rect(paddleTop.X, paddleTop.Y, ArcadeSession.PaddleWidth * _scale, ArcadeSession.PaddleHeight * _scale);
            if (!profile.HighContrast) context.DrawRectangle(BallGlow, null, paddle.Inflate(4 * _scale), 7 * _scale, 7 * _scale);
            context.DrawRectangle(Brushes.White, null, paddle, ArcadeSession.PaddleCornerRadius * _scale, ArcadeSession.PaddleCornerRadius * _scale);
            context.DrawRectangle(Mint, null, new Rect(paddle.X + 4 * _scale, paddle.Y + 10 * _scale, 3 * _scale, paddle.Height - 20 * _scale), 1, 1);
            // Draw the solid ball last: particles and shockwaves must never obscure it.
            var ball = ToScreen(game.Ball); var ballRadius = ArcadeSession.Radius * _scale;
            if (!profile.HighContrast)
                using (context.PushOpacity(.3)) context.DrawEllipse(tier > 0 ? accent : BallGlow, null, ball, ballRadius * (2.1 + tier * .3), ballRadius * (2.1 + tier * .3));
            context.DrawEllipse(Brushes.White, null, ball, ballRadius, ballRadius);
            var spinMark = new Avalonia.Vector(Math.Cos(game.BallAngle) * ballRadius * .5, Math.Sin(game.BallAngle) * ballRadius * .5);
            context.DrawLine(new Pen(Ink, Math.Max(1, 1.5 * _scale)), ball - spinMark, ball + spinMark);
        }
    }

    internal void OnEvent(ArcadeEvent item)
    {
        if (item.Kind is ArcadeEventKind.Paddle or ArcadeEventKind.Miss) { ClearEffects(); return; }
        if (profile.ReducedMotion) return;
        if (item.Kind == ArcadeEventKind.Cell)
        {
            var tier = ArcadeSession.TierForHits(item.ShotHits);
            AddSparks(item.Position, tier switch { 1 => 14, 2 => 24, 3 => 36, _ => 7 }, tier, false);
        }
        if (item.Kind == ArcadeEventKind.Milestone)
        {
            _waves.Clear();
            var tier = ArcadeSession.TierForHits(item.Value);
            AddSparks(item.Position, 12 + tier * 12, tier, true);
            for (var i = 0; i < tier; i++)
                _waves.Add(new Shockwave(item.Position, .9, 190 + i * 70, 1 - i * .16, tier == 3 ? Prism[i] : TierBrush(tier)));
        }
        if (_sparks.Count > 240) _sparks.RemoveRange(0, _sparks.Count - 240);
        if (_waves.Count > 6) _waves.RemoveRange(0, _waves.Count - 6);
    }
    private void AddSparks(Vector2 point, int count, int tier, bool burst)
    {
        for (var i = 0; i < count; i++)
        {
            var angle = _cosmetics.NextDouble() * Math.PI * 2;
            var speed = 40 + _cosmetics.NextDouble() * (100 + tier * 35) + (burst ? 80 : 0);
            var color = profile.HighContrast ? Brushes.White : tier == 3 ? Prism[i % Prism.Length] : TierBrush(tier);
            _sparks.Add(new Spark(point, new Vector2((float)(Math.Cos(angle) * speed), (float)(Math.Sin(angle) * speed)), burst ? .7 : .45, color, tier >= 2));
        }
    }
    internal void AdvanceEffects(double dt)
    {
        _clock += dt;
        if (profile.ReducedMotion) { _sparks.Clear(); _waves.Clear(); _trail.Clear(); return; }
        for (var i = _sparks.Count - 1; i >= 0; i--)
        {
            var spark = _sparks[i]; spark.Remaining -= dt; spark.Position += spark.Velocity * (float)dt;
            if (spark.Remaining <= 0) _sparks.RemoveAt(i);
        }
        for (var i = _waves.Count - 1; i >= 0; i--) { _waves[i].Remaining -= dt; if (_waves[i].Remaining <= 0) _waves.RemoveAt(i); }
        if (game.Ball != _lastBall)
        {
            if (Vector2.Distance(game.Ball, _lastBall) > 200) _trail.Clear();
            _trail.Add(game.Ball); _lastBall = game.Ball;
            if (_trail.Count > 28) _trail.RemoveAt(0);
        }
    }
    internal void ClearEffects() { _sparks.Clear(); _waves.Clear(); _trail.Clear(); }
    private static IBrush Brush(string hex) => new ImmutableSolidColorBrush(Color.Parse(hex));
    private sealed class Spark(Vector2 position, Vector2 velocity, double remaining, IBrush color, bool streak)
    {
        public Vector2 Position = position;
        public readonly Vector2 Velocity = velocity;
        public double Remaining = remaining;
        public readonly IBrush Color = color;
        public readonly bool Streak = streak;
    }
    private sealed class Shockwave(Vector2 position, double remaining, double speed, double flatness, IBrush color)
    {
        public readonly Vector2 Position = position;
        public double Remaining = remaining;
        public readonly double Speed = speed, Flatness = flatness;
        public readonly IBrush Color = color;
    }
}
