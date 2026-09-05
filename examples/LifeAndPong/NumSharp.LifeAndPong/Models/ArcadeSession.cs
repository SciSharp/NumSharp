using System.Numerics;

namespace NumSharp.LifeAndPong.Models;

public enum RunState { Ready, Playing, Paused, GameOver }
public enum ArcadeEventKind { Cell, Paddle, Wall, Birth, Miss, Sector, GameOver, Milestone }
public readonly record struct ArcadeEvent(ArcadeEventKind Kind, Vector2 Position, int Value = 0, int ShotHits = 0);

/// <summary>Authoritative, seeded arcade state. Rendering and sound cannot advance this clock.</summary>
public sealed class ArcadeSession : IDisposable
{
    public const float Width = 1600, Height = 900, ColonyFraction = .70f, Midline = Width * ColonyFraction;
    public const float Radius = 10, PaddleX = 1544, PaddleWidth = 18, PaddleHeight = 144;
    public const float CellPitch = 24, CellSize = 22, FieldX = 48, FieldY = 66;
    public const int Columns = 42, Rows = 32, LowPopulation = 64, TargetPopulation = 160;
    public const float Jitter = .02f;
    public const string Version = "life-arcade-2";
    private const float MaxPaddleSpeed = 1180, Acceleration = 7000;
    private readonly Queue<ArcadeEvent> _events = new();
    private Random _random = null!;
    private float _intent;
    private float? _pointerTarget;
    private double _lifeClock, _birthClock, _birthCooldown, _idleClock;
    private bool _grow, _disposed;
    private int _recoveryReturns;
    private RunState _beforePause;

    public ArcadeSession(int seed = 73021) => NewRun(seed);
    public LifeSimulation Life { get; private set; } = null!;
    public int Seed { get; private set; }
    public RunState State { get; private set; }
    public Vector2 Ball { get; private set; }
    public Vector2 Velocity { get; private set; }
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
    public bool ReturnAssist => Frozen && _idleClock >= 6;
    public double ActiveSeconds { get; private set; }

    public void NewRun(int seed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Life?.Dispose();
        Life = new LifeSimulation(Rows, Columns, seed, wrapEdges: false);
        Life.Clear();
        Life.ReplenishTo(TargetPopulation);
        Seed = seed;
        _random = new Random(seed ^ 0x2A471);
        Score = 0; NextAward = 1; Lives = 3; Destroyed = 0; Chain = 0; BestChain = 0; Sector = 1;
        _lifeClock = _birthClock = _birthCooldown = _idleClock = ActiveSeconds = 0;
        _recoveryReturns = 3; _events.Clear(); PaddleY = Height / 2;
        PrepareServe();
    }

    public void LaunchOrResume()
    {
        if (State == RunState.Paused) { State = _beforePause; return; }
        if (State != RunState.Ready) return;
        AdoptSector();
        var slope = ((float)_random.NextDouble() * 2 - 1) * .35f;
        Velocity = Vector2.Normalize(new Vector2(-1, slope)) * CurrentSpeed();
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
        _intent = Math.Clamp(direction, -1, 1);
        if (_intent != 0) _pointerTarget = null;
    }

    public void SetPointerTarget(float y)
    {
        if (float.IsFinite(y)) _pointerTarget = Math.Clamp(y, PaddleHeight / 2, Height - PaddleHeight / 2);
    }

    public void ReleaseInput() { _intent = 0; _pointerTarget = null; PaddleVelocity = 0; }
    public bool TryTakeEvent(out ArcadeEvent item) => _events.TryDequeue(out item);

    public void Advance(double delta)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!double.IsFinite(delta) || delta < 0 || delta > .1) throw new ArgumentOutOfRangeException(nameof(delta));
        if (delta == 0 || State is RunState.Paused or RunState.GameOver) return;
        if (State == RunState.Ready) { MovePaddle((float)delta); AttachBall(); return; }
        // Travel <= 2.5 units per step, including relative moving-paddle travel.
        var remaining = delta;
        while (remaining > 1e-9 && State == RunState.Playing)
        {
            var step = Math.Min(remaining, 2.5 / (Velocity.Length() + MaxPaddleSpeed));
            var crosses = false;
            if (Velocity.X != 0)
            {
                var timeToLine = (Midline - Ball.X) / Velocity.X;
                if (timeToLine > 1e-8 && timeToLine <= step) { step = timeToLine; crosses = true; }
                else if (Math.Abs(Ball.X - Midline) < .001f) SetPhase(Velocity.X > 0);
            }
            AdvancePhase(step);
            MovePaddle((float)step);
            Ball += Velocity * (float)step;
            ActiveSeconds += step;
            remaining -= step;
            if (crosses) { Ball = Ball with { X = Midline }; SetPhase(Velocity.X > 0); }
            ResolveWalls();
            ResolvePaddle();
            if (!_grow) ResolveCells();
            if (Ball.X - Radius > Width) LoseLife();
        }
    }

    private void SetPhase(bool grow)
    {
        if (_grow == grow) return;
        _grow = grow; _idleClock = 0;
        if (!grow) _birthClock = 0;
    }

    private void AdvancePhase(double delta)
    {
        if (!_grow) { _idleClock += delta; return; }
        _birthCooldown = Math.Max(0, _birthCooldown - delta);
        _lifeClock += delta;
        while (_lifeClock >= 1 / LifeRate) { Life.Step(); _lifeClock -= 1 / LifeRate; }
        if (Life.LiveCount >= LowPopulation) { _birthClock = 0; return; }
        if (_birthCooldown > 0) return;
        _birthClock += delta;
        if (_birthClock < .25) return;
        Life.ReplenishTo(TargetPopulation); _birthClock = 0; _birthCooldown = .75;
        Emit(ArcadeEventKind.Birth, new Vector2(Midline / 2, 450), Life.LiveCount);
    }

    private void MovePaddle(float dt)
    {
        var desired = _pointerTarget is float target ? Math.Clamp((target - PaddleY) * 14, -MaxPaddleSpeed, MaxPaddleSpeed) : _intent * MaxPaddleSpeed;
        PaddleVelocity += Math.Clamp(desired - PaddleVelocity, -Acceleration * dt, Acceleration * dt);
        PaddleY = Math.Clamp(PaddleY + PaddleVelocity * dt, PaddleHeight / 2, Height - PaddleHeight / 2);
        if (PaddleY <= PaddleHeight / 2 || PaddleY >= Height - PaddleHeight / 2) PaddleVelocity = 0;
    }

    private void ResolveWalls()
    {
        var normal = Vector2.Zero;
        if (Ball.Y < Radius && Velocity.Y < 0) { Ball = Ball with { Y = Radius }; normal = Vector2.UnitY; }
        else if (Ball.Y > Height - Radius && Velocity.Y > 0) { Ball = Ball with { Y = Height - Radius }; normal = -Vector2.UnitY; }
        if (normal != Vector2.Zero) WallBounce(normal);
        if (Ball.X < Radius && Velocity.X < 0) { Ball = Ball with { X = Radius }; WallBounce(Vector2.UnitX); }
    }

    private void WallBounce(Vector2 normal)
    {
        Velocity = Vector2.Reflect(Velocity, normal);
        if (ReturnAssist)
        {
            var speed = Velocity.Length();
            var vx = Math.Max(.45f * speed, Math.Abs(Velocity.X));
            Velocity = new Vector2(vx, MathF.CopySign(MathF.Sqrt(Math.Max(0, speed * speed - vx * vx)), Velocity.Y));
        }
        Emit(ArcadeEventKind.Wall, Ball);
    }

    private void ResolvePaddle()
    {
        if (!Contact(PaddleX - PaddleWidth / 2, PaddleY - PaddleHeight / 2, PaddleWidth, PaddleHeight, out var normal, out var depth)) return;
        var motion = new Vector2(0, PaddleVelocity);
        if (Vector2.Dot(Velocity - motion, normal) >= 0) return;
        AdoptSector(); _recoveryReturns = Math.Min(3, _recoveryReturns + 1);
        var response = Vector2.Reflect(Velocity - motion, normal) + motion;
        var tangent = new Vector2(-normal.Y, normal.X);
        response += tangent * Vector2.Dot(motion, tangent) * .18f;
        Velocity = Rebound(response, CurrentSpeed(), normal);
        Ball += normal * (depth + .02f);
        NextAward = 1; Chain = 0;
        Emit(ArcadeEventKind.Paddle, Ball);
    }

    private void ResolveCells()
    {
        var c0 = Math.Clamp((int)MathF.Floor((Ball.X - Radius - FieldX) / CellPitch), 0, Columns - 1);
        var c1 = Math.Clamp((int)MathF.Floor((Ball.X + Radius - FieldX) / CellPitch), 0, Columns - 1);
        var r0 = Math.Clamp((int)MathF.Floor((Ball.Y - Radius - FieldY) / CellPitch), 0, Rows - 1);
        var r1 = Math.Clamp((int)MathF.Floor((Ball.Y + Radius - FieldY) / CellPitch), 0, Rows - 1);
        var summedNormal = Vector2.Zero;
        var fallback = Vector2.Zero;
        var deepest = 0f;
        var incoming = Velocity;
        var any = false;
        // Row-major tie-break for contacts occurring in the same micro-step.
        for (var row = r0; row <= r1; row++) for (var col = c0; col <= c1; col++)
            {
                if (!Life.IsAlive(row, col) || !Contact(FieldX + col * CellPitch + 1, FieldY + row * CellPitch + 1, CellSize, CellSize, out var normal, out var depth)) continue;
                if (Vector2.Dot(incoming, normal) >= 0) continue;
                Life.SetCell(row, col, false);
                summedNormal += normal; fallback = normal; deepest = Math.Max(deepest, depth); any = true;
                AwardCell(new Vector2(FieldX + col * CellPitch + 12, FieldY + row * CellPitch + 12));
            }
        if (!any) return;
        var contactNormal = summedNormal.LengthSquared() > .0001f ? Vector2.Normalize(summedNormal) : fallback;
        Velocity = Rebound(Vector2.Reflect(incoming, contactNormal), incoming.Length(), contactNormal);
        Ball += contactNormal * (deepest + .02f);
        _idleClock = 0;
    }

    private bool Contact(float x, float y, float w, float h, out Vector2 normal, out float depth)
    {
        var closest = new Vector2(Math.Clamp(Ball.X, x, x + w), Math.Clamp(Ball.Y, y, y + h));
        var difference = Ball - closest;
        var squared = difference.LengthSquared();
        normal = Vector2.Zero; depth = 0;
        if (squared > Radius * Radius) return false;
        if (squared > .000001f) { var distance = MathF.Sqrt(squared); normal = difference / distance; depth = Radius - distance; return true; }
        // Robust interior recovery; choose the closest face rather than an arbitrary sideways teleport.
        var distances = new[] { Ball.X - x, x + w - Ball.X, Ball.Y - y, y + h - Ball.Y };
        var i = Array.IndexOf(distances, distances.Min());
        normal = i switch { 0 => -Vector2.UnitX, 1 => Vector2.UnitX, 2 => -Vector2.UnitY, _ => Vector2.UnitY };
        depth = Radius + distances[i]; return true;
    }

    private Vector2 Rebound(Vector2 response, float speed, Vector2 normal)
    {
        if (response.LengthSquared() < .0001f) response = normal;
        var v = AddJitter(Vector2.Normalize(response) * speed, ((float)_random.NextDouble() * 2 - 1) * Jitter);
        // Contact safety and arcade horizontal assistance are distinct from random jitter.
        if (Vector2.Dot(v, normal) < 0) v = Vector2.Reflect(v, normal);
        if (Math.Abs(v.X) < speed * .3f)
        {
            var sx = Math.Abs(normal.X) > .01f ? MathF.Sign(normal.X) : v.X >= 0 ? 1 : -1;
            var candidate = new Vector2(sx * speed * .3f, MathF.CopySign(speed * MathF.Sqrt(.91f), v.Y));
            if (Vector2.Dot(candidate, normal) >= 0) v = candidate;
        }
        return Vector2.Normalize(v) * speed;
    }

    internal static Vector2 AddJitter(Vector2 velocity, float fraction)
    {
        fraction = Math.Clamp(fraction, -Jitter, Jitter);
        var p = new Vector2(-velocity.Y, velocity.X);
        return Vector2.Normalize(velocity + p * fraction) * velocity.Length();
    }

    private void AwardCell(Vector2 position)
    {
        var award = NextAward;
        Score = Score > long.MaxValue - award ? long.MaxValue : Score + award;
        NextAward = NextAward == 1 ? 2 : NextAward > int.MaxValue - 2 ? int.MaxValue : NextAward + 2;
        if (Destroyed < int.MaxValue - 1) Destroyed++;
        if (Chain < int.MaxValue) Chain++;
        BestChain = Math.Max(BestChain, Chain);
        Emit(ArcadeEventKind.Cell, position, award);
        if (Chain is 20 or 50 or 100) Emit(ArcadeEventKind.Milestone, position, Chain);
    }

    private void AdoptSector()
    {
        if (Sector == PendingSector) return;
        Sector = PendingSector; Emit(ArcadeEventKind.Sector, new Vector2(Midline, 450), Sector);
    }
    private float CurrentSpeed() => Math.Max(640, SectorSpeed * (.85f + .05f * _recoveryReturns));
    private void LoseLife()
    {
        Lives--; NextAward = 1; Chain = 0; _recoveryReturns = 0;
        Emit(ArcadeEventKind.Miss, new Vector2(Width - 25, Ball.Y), Lives);
        PrepareServe();
        if (Lives == 0) { State = RunState.GameOver; Emit(ArcadeEventKind.GameOver, Ball); }
    }
    private void PrepareServe()
    {
        State = RunState.Ready; Velocity = Vector2.Zero; _grow = true;
        _idleClock = _birthClock = 0; ReleaseInput(); AttachBall();
    }
    private void AttachBall() => Ball = new Vector2(PaddleX - PaddleWidth / 2 - Radius - 3, PaddleY);
    private void Emit(ArcadeEventKind kind, Vector2 position, int value = 0)
    {
        if (_events.Count == 128) _events.Dequeue();
        _events.Enqueue(new ArcadeEvent(kind, position, value, Chain));
    }
    internal void SetBallForTesting(Vector2 ball, Vector2 velocity)
    {
        Ball = ball; Velocity = velocity; State = RunState.Playing; _grow = ball.X >= Midline;
    }
    internal void SetScoreForTesting(long score) => Score = score;
    public void Dispose() { if (_disposed) return; _disposed = true; Life.Dispose(); _events.Clear(); }
}
