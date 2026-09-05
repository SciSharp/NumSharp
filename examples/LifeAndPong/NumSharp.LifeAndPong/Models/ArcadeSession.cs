using System.Numerics;

namespace NumSharp.LifeAndPong.Models;

public enum RunState { Ready, Playing, Paused, GameOver }
public enum ArcadeEventKind { Cell, Paddle, Wall, Birth, Miss, Sector, GameOver, Milestone }
public readonly record struct ArcadeEvent(ArcadeEventKind Kind, Vector2 Position, int Value = 0, int ShotHits = 0);

/// <summary>Authoritative seeded simulation. Continuous ball contacts, fixed-rate paddle control.</summary>
public sealed class ArcadeSession : IDisposable
{
    public const float Width = 1600, Height = 900, ColonyFraction = .70f, Midline = Width * ColonyFraction;
    public const float Radius = 10, PaddleX = 1544, PaddleWidth = 18, PaddleHeight = 144, PaddleCornerRadius = 9;
    public const float CellPitch = 24, CellSize = 22, CellCornerRadius = 2, FieldX = 48, FieldY = 66;
    public const int Columns = 42, Rows = 32, LowPopulation = 64, TargetPopulation = 160;
    public const float Jitter = .05f;
    public const string Version = "life-arcade-3";
    private const float MaxPaddleSpeed = 1180, Acceleration = 7000, Separation = .001f;
    private readonly Queue<ArcadeEvent> _events = new();
    private readonly List<WorldHit> _contacts = new(8);
    private readonly List<ContactConstraint> _constraints = new(8);
    private Random _random = null!;
    private float _intent;
    private float? _pointerTarget;
    private double _lifeClock, _birthClock, _birthCooldown;
    private bool _grow, _disposed;
    private RunState _beforePause;
    internal bool NoiseEnabled { get; set; } = true;

    public ArcadeSession(int seed = 73021) => NewRun(seed);
    public LifeSimulation Life { get; private set; } = null!;
    internal bool IsDisposed => _disposed;
    public int Seed { get; private set; }
    public RunState State { get; private set; }
    public Vector2 Ball { get; private set; }
    public Vector2 Velocity { get; private set; }
    public float BallSpin { get; private set; }
    public float BallAngle { get; private set; }
    public float PaddleY { get; private set; }
    public float PaddleVelocity { get; private set; }
    public long Score { get; private set; }
    public int NextAward { get; private set; }
    public int Lives { get; private set; }
    public int Destroyed { get; private set; }
    public int Chain { get; private set; }
    public int BestChain { get; private set; }
    public int EffectTier => TierForHits(Chain);
    public static int TierForHits(int hits) => hits >= 100 ? 3 : hits >= 50 ? 2 : hits >= 20 ? 1 : 0;
    public int Sector { get; private set; }
    public int PendingSector => 1 + Destroyed / 40;
    public float SectorSpeed => MathF.Min(1000, 640 * MathF.Pow(1.06f, Math.Min(Sector - 1, 20)));
    public double LifeRate => Math.Min(10, 6 + .5 * (Sector - 1));
    public bool Growing => State == RunState.Playing && _grow;
    public bool Frozen => State == RunState.Playing && !_grow;
    public bool Replenishing => Growing && _birthClock > 0;
    public double ActiveSeconds { get; private set; }
    public string? PhysicsIssue { get; private set; }

    public void NewRun(int seed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Life?.Dispose(); Life = new LifeSimulation(Rows, Columns, seed, wrapEdges: false);
        Life.Clear(); Life.ReplenishTo(TargetPopulation);
        Seed = seed; _random = new Random(seed ^ 0x2A471);
        Score = 0; NextAward = 1; Lives = 3; Destroyed = 0; Chain = 0; BestChain = 0; Sector = 1;
        _lifeClock = _birthClock = _birthCooldown = ActiveSeconds = 0;
        PhysicsIssue = null; _events.Clear(); PaddleY = Height / 2; PrepareServe();
    }
    public void LaunchOrResume()
    {
        if (PhysicsIssue is not null) return;
        if (State == RunState.Paused) { State = _beforePause; return; }
        if (State != RunState.Ready) return;
        AdoptSector();
        var slope = ((float)_random.NextDouble() * 2 - 1) * .35f;
        Velocity = Vector2.Normalize(new Vector2(-1, slope)) * SectorSpeed;
        State = RunState.Playing; _grow = true;
    }
    public void Pause()
    {
        ReleaseInput();
        if (State is RunState.Playing or RunState.Ready) { _beforePause = State; State = RunState.Paused; }
    }
    public void SetIntent(float direction)
    {
        if (!float.IsFinite(direction)) return;
        _intent = Math.Clamp(direction, -1, 1); if (_intent != 0) _pointerTarget = null;
    }
    public void SetPointerTarget(float y)
    { if (float.IsFinite(y)) _pointerTarget = Math.Clamp(y, PaddleHeight / 2, Height - PaddleHeight / 2); }
    public void ReleaseInput() { _intent = 0; _pointerTarget = null; PaddleVelocity = 0; }
    public bool TryTakeEvent(out ArcadeEvent item) => _events.TryDequeue(out item);

    public void Advance(double delta)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!double.IsFinite(delta) || delta < 0 || delta > .1) throw new ArgumentOutOfRangeException(nameof(delta));
        if (delta == 0 || State is RunState.Paused or RunState.GameOver) return;
        var remaining = delta;
        while (remaining > 1e-10 && State is RunState.Playing or RunState.Ready)
        {
            var step = Math.Min(remaining, 1d / 120);
            UpdatePaddleVelocity((float)step);
            if (State == RunState.Ready)
            { PaddleY = Math.Clamp(PaddleY + PaddleVelocity * (float)step, PaddleHeight / 2, Height - PaddleHeight / 2); AttachBall(); }
            else AdvanceContinuous(step);
            remaining -= step;
        }
    }

    private void AdvanceContinuous(double remaining)
    {
        var iterations = 0;
        while (remaining > 1e-10 && State == RunState.Playing)
        {
            if (++iterations > 128)
            {
                PhysicsIssue = "A contact could not be resolved safely. Restart this run."; Pause(); return;
            }
            if (Ball.X > Width + Radius) { LoseLife(); return; }
            if (Ball.X == Midline && Velocity.X != 0) SetPhase(Velocity.X > 0);
            var crossing = Velocity.X == 0 ? double.PositiveInfinity : (Midline - Ball.X) / (double)Velocity.X;
            if (crossing <= 0) crossing = double.PositiveInfinity;
            var stop = PaddleVelocity == 0 ? double.PositiveInfinity :
                ((PaddleVelocity > 0 ? Height - PaddleHeight / 2 : PaddleHeight / 2) - PaddleY) / (double)PaddleVelocity;
            if (stop <= 0) stop = double.PositiveInfinity;
            var goal = Velocity.X <= 0 ? double.PositiveInfinity : Math.Max(0, (Width + Radius - Ball.X) / (double)Velocity.X);
            var horizon = Math.Min(remaining, Math.Min(crossing, Math.Min(stop, goal)));
            var collision = FindContacts(horizon);
            var dt = Math.Min(horizon, collision);
            AdvancePhase(dt);
            Ball += Velocity * (float)dt;
            PaddleY = Math.Clamp(PaddleY + PaddleVelocity * (float)dt, PaddleHeight / 2, Height - PaddleHeight / 2);
            BallAngle = (BallAngle + BallSpin * (float)dt) % MathF.Tau;
            ActiveSeconds += dt; remaining -= dt;
            if (collision <= dt + CollisionMath.TimeTolerance) ResolveContacts();
            if (stop <= dt + CollisionMath.TimeTolerance) PaddleVelocity = 0;
            if (goal <= dt + CollisionMath.TimeTolerance && Ball.X >= Width + Radius - .0002f) { LoseLife(); return; }
            if (crossing <= dt + CollisionMath.TimeTolerance) { Ball = Ball with { X = Midline }; SetPhase(Velocity.X >= 0); }
        }
    }

    private double FindContacts(double horizon)
    {
        _contacts.Clear(); var earliest = double.PositiveInfinity;
        void Consider(SweepHit hit, ArcadeEventKind kind, Vector2 surface, int row = -1, int col = -1)
        {
            if (hit.Time > horizon + CollisionMath.TimeTolerance) return;
            if (hit.Time < earliest - CollisionMath.TimeTolerance) { _contacts.Clear(); earliest = hit.Time; }
            if (Math.Abs(hit.Time - earliest) <= CollisionMath.TimeTolerance) _contacts.Add(new WorldHit(hit, kind, surface, row, col));
        }
        void Wall(float distance, Vector2 normal)
        {
            var speed = Vector2.Dot(Velocity, normal); if (speed >= -1e-5f) return;
            var time = Math.Max(0, -distance / (double)speed);
            Consider(new SweepHit(time, normal, Math.Max(0, -distance)), ArcadeEventKind.Wall, Vector2.Zero);
        }
        Wall(Ball.X - Radius, Vector2.UnitX);
        Wall(Ball.Y - Radius, Vector2.UnitY);
        Wall(Height - Radius - Ball.Y, -Vector2.UnitY);
        var paddleMotion = new Vector2(0, PaddleVelocity);
        if (CollisionMath.SweepRoundedBox(Ball, Velocity - paddleMotion, horizon, Radius,
            PaddleX - PaddleWidth / 2, PaddleY - PaddleHeight / 2, PaddleWidth, PaddleHeight, PaddleCornerRadius, out var paddleHit))
            Consider(paddleHit, ArcadeEventKind.Paddle, paddleMotion);
        if (!_grow)
        {
            var end = Ball + Velocity * (float)horizon;
            var c0 = Math.Clamp((int)MathF.Floor((Math.Min(Ball.X, end.X) - Radius - FieldX) / CellPitch), 0, Columns - 1);
            var c1 = Math.Clamp((int)MathF.Floor((Math.Max(Ball.X, end.X) + Radius - FieldX) / CellPitch), 0, Columns - 1);
            var r0 = Math.Clamp((int)MathF.Floor((Math.Min(Ball.Y, end.Y) - Radius - FieldY) / CellPitch), 0, Rows - 1);
            var r1 = Math.Clamp((int)MathF.Floor((Math.Max(Ball.Y, end.Y) + Radius - FieldY) / CellPitch), 0, Rows - 1);
            for (var row = r0; row <= r1; row++) for (var col = c0; col <= c1; col++)
                if (Life.IsAlive(row, col) && CollisionMath.SweepRoundedBox(Ball, Velocity, horizon, Radius,
                    FieldX + col * CellPitch + 1, FieldY + row * CellPitch + 1, CellSize, CellSize, CellCornerRadius, out var cellHit))
                    Consider(cellHit, ArcadeEventKind.Cell, Vector2.Zero, row, col);
        }
        return earliest;
    }

    private void ResolveContacts()
    {
        if (_contacts.Count == 0) return;
        _constraints.Clear();
        foreach (var hit in _contacts) _constraints.Add(new ContactConstraint(hit.Geometry.Normal, hit.Surface));
        var incoming = Velocity;
        Velocity = CollisionMath.ElasticManifold(incoming, _constraints);
        if (_contacts.Count == 1 && _contacts[0].Kind == ArcadeEventKind.Paddle)
        {
            var hit = _contacts[0];
            (Velocity, BallSpin) = CollisionMath.PaddleFriction(incoming, Velocity, hit.Surface, hit.Geometry.Normal, BallSpin, Radius);
        }
        if (NoiseEnabled) Velocity = CollisionMath.SafeNoise(Velocity, ((float)_random.NextDouble() * 2 - 1) * Jitter, _constraints);
        // Consume all simultaneous distinct cells once. Their normals are constraints, not averaged steering.
        foreach (var hit in _contacts)
        {
            Ball += hit.Geometry.Normal * (hit.Geometry.Penetration + Separation);
            if (hit.Kind == ArcadeEventKind.Cell && Life.IsAlive(hit.Row, hit.Column))
            {
                Life.SetCell(hit.Row, hit.Column, false);
                AwardCell(new Vector2(FieldX + hit.Column * CellPitch + 12, FieldY + hit.Row * CellPitch + 12));
            }
        }
        if (_contacts.Any(hit => hit.Kind == ArcadeEventKind.Paddle))
        { AdoptSector(); NextAward = 1; Chain = 0; Emit(ArcadeEventKind.Paddle, Ball); }
        else if (_contacts.Any(hit => hit.Kind == ArcadeEventKind.Wall)) Emit(ArcadeEventKind.Wall, Ball);
    }
    private void SetPhase(bool grow)
    { if (_grow == grow) return; _grow = grow; if (!grow) _birthClock = 0; }
    private void AdvancePhase(double delta)
    {
        if (!_grow) return;
        _birthCooldown = Math.Max(0, _birthCooldown - delta); _lifeClock += delta;
        while (_lifeClock >= 1 / LifeRate) { Life.Step(); _lifeClock -= 1 / LifeRate; }
        if (Life.LiveCount >= LowPopulation) { _birthClock = 0; return; }
        if (_birthCooldown > 0) return;
        _birthClock += delta; if (_birthClock < .25) return;
        Life.ReplenishTo(TargetPopulation); _birthClock = 0; _birthCooldown = .75;
        Emit(ArcadeEventKind.Birth, new Vector2(Midline / 2, 450), Life.LiveCount);
    }
    private void UpdatePaddleVelocity(float dt)
    {
        var desired = _pointerTarget is float target ? Math.Clamp((target - PaddleY) * 14, -MaxPaddleSpeed, MaxPaddleSpeed) : _intent * MaxPaddleSpeed;
        PaddleVelocity += Math.Clamp(desired - PaddleVelocity, -Acceleration * dt, Acceleration * dt);
        if ((PaddleY <= PaddleHeight / 2 && PaddleVelocity < 0) || (PaddleY >= Height - PaddleHeight / 2 && PaddleVelocity > 0)) PaddleVelocity = 0;
    }
    internal static Vector2 AddJitter(Vector2 velocity, float fraction) => CollisionMath.DirectionNoise(velocity, fraction);
    private void AwardCell(Vector2 position)
    {
        var award = NextAward; Score = Score > long.MaxValue - award ? long.MaxValue : Score + award;
        NextAward = NextAward == 1 ? 2 : NextAward > int.MaxValue - 2 ? int.MaxValue : NextAward + 2;
        if (Destroyed < int.MaxValue - 1) Destroyed++; if (Chain < int.MaxValue) Chain++;
        BestChain = Math.Max(BestChain, Chain); Emit(ArcadeEventKind.Cell, position, award);
        if (Chain is 20 or 50 or 100) Emit(ArcadeEventKind.Milestone, position, Chain);
    }
    private void AdoptSector()
    { if (Sector == PendingSector) return; Sector = PendingSector; Emit(ArcadeEventKind.Sector, new Vector2(Midline, 450), Sector); }
    private void LoseLife()
    {
        Lives--; NextAward = 1; Chain = 0; Emit(ArcadeEventKind.Miss, new Vector2(Width - 25, Ball.Y), Lives);
        PrepareServe(); if (Lives == 0) { State = RunState.GameOver; Emit(ArcadeEventKind.GameOver, Ball); }
    }
    private void PrepareServe()
    {
        State = RunState.Ready; Velocity = Vector2.Zero; BallSpin = BallAngle = 0; _grow = true;
        _birthClock = 0; ReleaseInput(); AttachBall();
    }
    private void AttachBall() => Ball = new Vector2(PaddleX - PaddleWidth / 2 - Radius - 3, PaddleY);
    private void Emit(ArcadeEventKind kind, Vector2 position, int value = 0)
    { if (_events.Count == 128) _events.Dequeue(); _events.Enqueue(new ArcadeEvent(kind, position, value, Chain)); }
    internal void SetBallForTesting(Vector2 ball, Vector2 velocity)
    { Ball = ball; Velocity = velocity; State = RunState.Playing; _grow = ball.X >= Midline; }
    internal void SetScoreForTesting(long score) => Score = score;
    public void Dispose() { if (_disposed) return; _disposed = true; Life.Dispose(); _events.Clear(); }
    private readonly record struct WorldHit(SweepHit Geometry, ArcadeEventKind Kind, Vector2 Surface, int Row, int Column);
}
