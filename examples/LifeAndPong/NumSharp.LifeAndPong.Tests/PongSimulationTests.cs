using System.Numerics;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class PongSimulationTests
{
    [TestMethod]
    public void Top_Wall_Reflects_Without_Changing_Speed()
    {
        var pong = new PongSimulation(42);
        pong.SetBallForTesting(new Vector2(380, PongSimulation.BallRadius + 0.2f), new Vector2(300, -240));
        var speed = pong.BallVelocity.Length();

        pong.Advance(1f / 120f);

        Assert.IsTrue(pong.BallVelocity.Y > 0);
        Assert.AreEqual(speed, pong.BallVelocity.Length(), 0.01f);
        Assert.IsTrue(pong.BallPosition.Y >= PongSimulation.BallRadius);
    }

    [TestMethod]
    public void Player_Paddle_Returns_Ball_And_Builds_Rally_Speed()
    {
        var pong = new PongSimulation(42);
        var contactX = 36f + PongSimulation.PaddleWidth / 2f + PongSimulation.BallRadius + 0.5f;
        pong.SetBallForTesting(new Vector2(contactX, pong.PlayerY), new Vector2(-500, 0));

        pong.Advance(1f / 120f);

        Assert.IsTrue(pong.BallVelocity.X > 0);
        Assert.IsTrue(pong.BallVelocity.Length() > 500);
        Assert.IsTrue(MathF.Abs(pong.BallVelocity.Y / pong.BallVelocity.X) <= PongSimulation.DirectionalJitter + 0.001f);
    }

    [TestMethod]
    public void Adaptive_Substeps_Prevent_A_Fast_Ball_From_Tunneling_Through_The_Paddle()
    {
        var pong = new PongSimulation(17);
        pong.SetBallForTesting(new Vector2(72, pong.PlayerY), new Vector2(-PongSimulation.WorldWidth * 1.2f, 0));

        pong.Advance(1f / 30f);

        Assert.IsTrue(pong.BallVelocity.X > 0);
        Assert.AreEqual(0, pong.AiScore);
    }

    [TestMethod]
    public void Missed_Ball_Awards_Ai_A_Point_And_Starts_Serve_Countdown()
    {
        var pong = new PongSimulation(42);
        pong.SetBallForTesting(new Vector2(-PongSimulation.BallRadius - 1, 40), new Vector2(-500, 0));

        pong.Advance(1f / 120f);

        Assert.AreEqual(1, pong.AiScore);
        Assert.IsTrue(pong.ServeCountdown > 0);
    }

    [TestMethod]
    public void Match_Ends_When_Either_Side_Reaches_Seven()
    {
        var pong = new PongSimulation(42);
        for (var point = 0; point < PongSimulation.WinningScore; point++)
        {
            pong.SetBallForTesting(
                new Vector2(PongSimulation.WorldWidth + PongSimulation.BallRadius + 1, 40),
                new Vector2(500, 0));
            pong.Advance(1f / 120f);
        }

        Assert.AreEqual(PongSimulation.WinningScore, pong.PlayerScore);
        Assert.IsTrue(pong.IsMatchOver);
    }
}
