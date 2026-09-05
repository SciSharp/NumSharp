using System.Numerics;

namespace NumSharp.LifeAndPong.Models;

/// <summary>
/// Fixed-step Pong simulation with inertial paddles, adaptive collision sub-steps and bounded jitter.
/// </summary>
public sealed class PongSimulation
{
    public const float WorldWidth = 760f;
    public const float WorldHeight = 650f;
    public const float PaddleWidth = 18f;
    public const float PaddleHeight = 118f;
    public const float BallRadius = 11f;
    public const float DirectionalJitter = 0.02f;
    public const int WinningScore = 7;

    private const float PlayerX = 36f;
    private const float AiX = WorldWidth - 36f;
    private const float PaddleAcceleration = 4300f;
    private const float PlayerMaxSpeed = 700f;
    private const float AiMaxSpeed = 610f;
    private const float BallStartSpeed = 430f;
    private const float BallMaxSpeed = 920f;

    private readonly Random _random;
    private readonly List<Vector2> _trail = new(18);
    private float _playerIntent;
    private float? _pointerTargetY;
    private float _aiTargetY;
    private float _aiReactionClock;
    private float _serveClock;

    public PongSimulation(int seed = 918273)
    {
        _random = new Random(seed);
        ResetMatch();
    }

    public Vector2 BallPosition { get; private set; }

    public Vector2 BallVelocity { get; private set; }

    public float PlayerY { get; private set; }

    public float PlayerVelocity { get; private set; }

    public float AiY { get; private set; }

    public float AiVelocity { get; private set; }

    public int PlayerScore { get; private set; }

    public int AiScore { get; private set; }

    public bool IsPaused { get; private set; }

    public bool IsMatchOver { get; private set; }

    public float ServeCountdown => MathF.Max(0, _serveClock);

    public IReadOnlyList<Vector2> Trail => _trail;

    public void SetKeyboardIntent(float direction)
    {
        _playerIntent = Math.Clamp(direction, -1f, 1f);
        if (MathF.Abs(_playerIntent) > 0.01f)
            _pointerTargetY = null;
    }

    public void SetPointerTarget(float worldY)
    {
        _pointerTargetY = Math.Clamp(worldY, PaddleHeight / 2f, WorldHeight - PaddleHeight / 2f);
    }

    public void TogglePause()
    {
        if (!IsMatchOver)
            IsPaused = !IsPaused;
    }

    public void ResetMatch()
    {
        PlayerScore = 0;
        AiScore = 0;
        IsPaused = false;
        IsMatchOver = false;
        PlayerY = WorldHeight / 2f;
        AiY = WorldHeight / 2f;
        PlayerVelocity = 0;
        AiVelocity = 0;
        _pointerTargetY = null;
        _playerIntent = 0;
        BeginServe(_random.Next(2) == 0 ? -1 : 1);
    }

    public void Advance(float fixedDeltaSeconds)
    {
        if (fixedDeltaSeconds <= 0)
            return;
        if (IsPaused || IsMatchOver)
            return;

        UpdatePaddles(fixedDeltaSeconds);

        if (_serveClock > 0)
        {
            _serveClock -= fixedDeltaSeconds;
            if (_serveClock <= 0)
                LaunchServe();
            return;
        }

        var travel = BallVelocity.Length() * fixedDeltaSeconds;
        var subSteps = Math.Max(1, (int)MathF.Ceiling(travel / (BallRadius * 0.45f)));
        var subDelta = fixedDeltaSeconds / subSteps;

        for (var step = 0; step < subSteps; step++)
        {
            BallPosition += BallVelocity * subDelta;
            ResolveWalls();
            ResolvePaddle(PlayerX, PlayerY, PlayerVelocity, isLeft: true);
            ResolvePaddle(AiX, AiY, AiVelocity, isLeft: false);

            if (BallPosition.X < -BallRadius)
            {
                AwardPoint(playerScored: false);
                break;
            }

            if (BallPosition.X > WorldWidth + BallRadius)
            {
                AwardPoint(playerScored: true);
                break;
            }
        }

        if (_serveClock <= 0)
        {
            _trail.Add(BallPosition);
            if (_trail.Count > 18)
                _trail.RemoveAt(0);
        }
    }

    internal void SetBallForTesting(Vector2 position, Vector2 velocity)
    {
        BallPosition = position;
        BallVelocity = velocity;
        _serveClock = 0;
        IsPaused = false;
        IsMatchOver = false;
        _trail.Clear();
    }

    private void UpdatePaddles(float delta)
    {
        float playerDesiredVelocity;
        if (_pointerTargetY is float target)
        {
            var error = target - PlayerY;
            playerDesiredVelocity = Math.Clamp(error * 9f, -PlayerMaxSpeed, PlayerMaxSpeed);
        }
        else
        {
            playerDesiredVelocity = _playerIntent * PlayerMaxSpeed;
        }

        PlayerVelocity = MoveTowards(PlayerVelocity, playerDesiredVelocity, PaddleAcceleration * delta);
        PlayerY = ClampPaddle(PlayerY + PlayerVelocity * delta);
        if (PlayerY is <= PaddleHeight / 2f or >= WorldHeight - PaddleHeight / 2f)
            PlayerVelocity = 0;

        _aiReactionClock -= delta;
        if (_aiReactionClock <= 0)
        {
            _aiReactionClock = 0.075f;
            _aiTargetY = PredictAiTarget();
        }

        var aiError = _aiTargetY - AiY;
        var aiDesiredVelocity = Math.Clamp(aiError * 6.4f, -AiMaxSpeed, AiMaxSpeed);
        AiVelocity = MoveTowards(AiVelocity, aiDesiredVelocity, PaddleAcceleration * 0.82f * delta);
        AiY = ClampPaddle(AiY + AiVelocity * delta);
        if (AiY is <= PaddleHeight / 2f or >= WorldHeight - PaddleHeight / 2f)
            AiVelocity = 0;
    }

    private float PredictAiTarget()
    {
        if (BallVelocity.X <= 0 || _serveClock > 0)
            return WorldHeight / 2f;

        var time = MathF.Max(0, (AiX - BallPosition.X) / BallVelocity.X);
        var projected = BallPosition.Y + BallVelocity.Y * time;
        var span = WorldHeight - BallRadius * 2f;
        var folded = PositiveModulo(projected - BallRadius, span * 2f);
        var reflected = folded <= span ? folded : span * 2f - folded;
        return BallRadius + reflected + MathF.Sin(PlayerScore * 1.7f + AiScore) * 18f;
    }

    private void ResolveWalls()
    {
        if (BallPosition.Y - BallRadius < 0 && BallVelocity.Y < 0)
        {
            BallPosition = BallPosition with { Y = BallRadius };
            BallVelocity = BallVelocity with { Y = -BallVelocity.Y };
        }
        else if (BallPosition.Y + BallRadius > WorldHeight && BallVelocity.Y > 0)
        {
            BallPosition = BallPosition with { Y = WorldHeight - BallRadius };
            BallVelocity = BallVelocity with { Y = -BallVelocity.Y };
        }
    }

    private void ResolvePaddle(float paddleX, float paddleY, float paddleVelocity, bool isLeft)
    {
        if (isLeft ? BallVelocity.X >= 0 : BallVelocity.X <= 0)
            return;

        var halfWidth = PaddleWidth / 2f;
        var halfHeight = PaddleHeight / 2f;
        var closestX = Math.Clamp(BallPosition.X, paddleX - halfWidth, paddleX + halfWidth);
        var closestY = Math.Clamp(BallPosition.Y, paddleY - halfHeight, paddleY + halfHeight);
        var difference = BallPosition - new Vector2(closestX, closestY);
        if (difference.LengthSquared() > BallRadius * BallRadius)
            return;

        var impact = Math.Clamp((BallPosition.Y - paddleY) / halfHeight, -1f, 1f);
        var speed = MathF.Min(BallMaxSpeed, MathF.Max(BallStartSpeed, BallVelocity.Length() * 1.045f));
        var horizontalDirection = isLeft ? 1f : -1f;
        Vector2 response = new(horizontalDirection, impact * 0.78f + paddleVelocity / PlayerMaxSpeed * 0.24f);
        response = Vector2.Normalize(response) * speed;
        BallVelocity = ApplyDirectionalJitter(response);

        var outside = isLeft
            ? paddleX + halfWidth + BallRadius
            : paddleX - halfWidth - BallRadius;
        BallPosition = BallPosition with { X = outside };
    }

    private Vector2 ApplyDirectionalJitter(Vector2 velocity)
    {
        var signedFraction = ((float)_random.NextDouble() * 2f - 1f) * DirectionalJitter;
        var perpendicular = Vector2.Normalize(new Vector2(-velocity.Y, velocity.X));
        var perturbed = velocity + perpendicular * velocity.Length() * signedFraction;
        return Vector2.Normalize(perturbed) * velocity.Length();
    }

    private void AwardPoint(bool playerScored)
    {
        if (playerScored)
            PlayerScore++;
        else
            AiScore++;

        if (PlayerScore >= WinningScore || AiScore >= WinningScore)
        {
            IsMatchOver = true;
            BallVelocity = Vector2.Zero;
            BallPosition = new Vector2(WorldWidth / 2f, WorldHeight / 2f);
            _trail.Clear();
            return;
        }

        BeginServe(playerScored ? 1 : -1);
    }

    private void BeginServe(int horizontalDirection)
    {
        BallPosition = new Vector2(WorldWidth / 2f, WorldHeight / 2f);
        BallVelocity = new Vector2(horizontalDirection * BallStartSpeed, 0);
        _serveClock = 0.85f;
        _trail.Clear();
        _aiTargetY = WorldHeight / 2f;
        _aiReactionClock = 0;
    }

    private void LaunchServe()
    {
        float direction = MathF.Sign(BallVelocity.X);
        var vertical = ((float)_random.NextDouble() * 2f - 1f) * 0.42f;
        BallVelocity = Vector2.Normalize(new Vector2(direction, vertical)) * BallStartSpeed;
        _serveClock = 0;
    }

    private static float MoveTowards(float current, float target, float maxDelta)
    {
        if (MathF.Abs(target - current) <= maxDelta)
            return target;
        return current + MathF.Sign(target - current) * maxDelta;
    }

    private static float ClampPaddle(float value) =>
        Math.Clamp(value, PaddleHeight / 2f, WorldHeight - PaddleHeight / 2f);

    private static float PositiveModulo(float value, float modulus) => (value % modulus + modulus) % modulus;
}
