using System.Numerics;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class PongSimulationTests
{
    [TestMethod]
    public void Ready_Pause_And_Reset_Do_Not_Run_Away_From_The_Player()
    {
        var pong = new PongSimulation();
        var position = pong.BallPosition;
        for (var i = 0; i < 2400; i++) pong.Advance(1f / 120);
        Assert.AreEqual(position, pong.BallPosition);
        Assert.AreEqual(0, pong.AiScore);
        pong.TogglePause();
        for (var i = 0; i < 120; i++) pong.Advance(1f / 120);
        Assert.AreNotEqual(position, pong.BallPosition);
        pong.PauseForDeactivation();
        position = pong.BallPosition;
        pong.Advance(1f / 120);
        Assert.AreEqual(position, pong.BallPosition);
        pong.ResetMatch();
        Assert.IsTrue(pong.IsReady);
    }

    [TestMethod]
    public void Jitter_Remains_Bounded_Across_Many_Impacts_And_Seeds()
    {
        var negative = 0;
        var positive = 0;
        for (var seed = 0; seed < 512; seed++)
        {
            var pong = new PongSimulation(seed);
            pong.SetBallForTesting(new Vector2(56.1f, pong.PlayerY), new Vector2(-500, 0));
            pong.Advance(1f / 120);
            Assert.AreEqual(522.5f, pong.BallVelocity.Length(), 0.01f);
            Assert.IsTrue(pong.BallVelocity.X > 0);
            Assert.IsTrue(MathF.Abs(pong.BallVelocity.Y / pong.BallVelocity.X) <= 0.02001f);
            if (pong.BallVelocity.Y < 0) negative++; else positive++;
        }
        Assert.IsTrue(negative > 150 && positive > 150);
    }

    [TestMethod]
    public void Paddle_End_Contact_Reflects_About_Its_Normal()
    {
        var pong = new PongSimulation(11);
        var x = 36f;
        pong.SetBallForTesting(new Vector2(x, pong.PlayerY - PongSimulation.PaddleHeight / 2 - 10.9f), new Vector2(0, 500));
        pong.Advance(1f / 120);
        Assert.IsTrue(pong.BallVelocity.Y < 0);
        Assert.IsTrue(Math.Abs(pong.BallPosition.X - x) < 1);
    }

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
