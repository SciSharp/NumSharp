using System.Numerics;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class ArcadeSessionTests
{
    public TestContext TestContext { get; set; } = null!;
    [TestMethod]
    public void Ready_Pause_And_Three_Lives_Form_A_Complete_Run()
    {
        using var game = new ArcadeSession();
        var field = Snapshot(game);
        game.Advance(.1); Assert.AreEqual(RunState.Ready, game.State); CollectionAssert.AreEqual(field, Snapshot(game));
        game.LaunchOrResume(); Assert.IsTrue(game.Velocity.X < 0);
        for (var lives = 2; lives >= 0; lives--)
        {
            Hit(game);
            var score = game.Score;
            game.Pause(); var ball = game.Ball; var generation = game.Life.Generation;
            game.SetIntent(1); game.Advance(.1); Assert.AreEqual(ball, game.Ball); Assert.AreEqual(generation, game.Life.Generation);
            game.LaunchOrResume(); Miss(game);
            Assert.AreEqual(lives, game.Lives); Assert.AreEqual(score, game.Score); Assert.AreEqual(1, game.NextAward);
            Assert.AreEqual(lives == 0 ? RunState.GameOver : RunState.Ready, game.State);
        }
        game.LaunchOrResume(); Assert.AreEqual(RunState.GameOver, game.State);
        game.NewRun(4); Assert.AreEqual(0L, game.Score); Assert.AreEqual(3, game.Lives); Assert.AreEqual(160, game.Life.LiveCount);
    }

    [TestMethod]
    public void Grow_Clock_Stops_In_Left_Half_And_Has_No_Catchup()
    {
        using var game = new ArcadeSession();
        game.Life.Clear();
        game.Life.SetCell(10, 10, true); game.Life.SetCell(10, 11, true); game.Life.SetCell(10, 12, true);
        Hold(game, false, 2);
        Assert.AreEqual(0L, game.Life.Generation); Assert.AreEqual(3, game.Life.LiveCount);
        Hold(game, true, .18); Assert.AreEqual(1L, game.Life.Generation);
        var generation = game.Life.Generation;
        Hold(game, false, 5); Assert.AreEqual(generation, game.Life.Generation);
        Hold(game, true, .01); Assert.AreEqual(generation, game.Life.Generation);
    }

    [TestMethod]
    public void Center_Crossing_Splits_Time_Without_A_Generation_On_The_Wrong_Side()
    {
        using var game = new ArcadeSession();
        Hold(game, true, .15);
        game.SetBallForTesting(new Vector2(801, 500), new Vector2(-640, 0));
        game.Advance(.08);
        Assert.IsTrue(game.Frozen); Assert.AreEqual(0L, game.Life.Generation);
        game.SetBallForTesting(new Vector2(799, 500), new Vector2(640, 0)); game.Advance(.03);
        Assert.IsTrue(game.Growing); Assert.AreEqual(1L, game.Life.Generation);
    }

    [TestMethod]
    public void Cell_Awards_Double_Across_Returns_Cap_And_Saturate()
    {
        using var game = new ArcadeSession();
        Hit(game); Assert.AreEqual(1L, game.Score);
        Hold(game, true, .02); Hit(game); Assert.AreEqual(3L, game.Score);
        Hit(game); Assert.AreEqual(7L, game.Score); Assert.AreEqual(8, game.NextAward);
        for (var i = 0; i < 12; i++) Hit(game);
        Assert.AreEqual(256, game.NextAward);
        var before = game.Score; Hit(game); Assert.AreEqual(before + 256, game.Score);
        game.SetScoreForTesting(long.MaxValue - 2); Hit(game); Assert.AreEqual(long.MaxValue, game.Score);
        Miss(game); Assert.AreEqual(1, game.NextAward); Assert.AreEqual(0, game.Chain);
    }

    [TestMethod]
    public void Shared_Cell_Seam_Scores_Distinct_Cells_Only_Once()
    {
        using var game = new ArcadeSession(); game.Life.Clear();
        game.Life.SetCell(15, 20, true); game.Life.SetCell(15, 21, true);
        game.SetBallForTesting(new Vector2(ArcadeSession.FieldX + 21 * 24, ArcadeSession.FieldY + 16 * 24 + 10.5f), new Vector2(0, -640));
        game.Advance(1d / 120);
        Assert.AreEqual(2, game.Destroyed); Assert.AreEqual(3L, game.Score); Assert.AreEqual(0, game.Life.LiveCount);
        Assert.IsTrue(game.Velocity.Y > 0);
        game.Advance(.1); Assert.AreEqual(3L, game.Score);
    }

    [TestMethod]
    public void Sparse_Replenishment_Preserves_Survivors_And_Waits_For_Grow()
    {
        using var game = new ArcadeSession(); game.Life.Clear();
        for (var r = 3; r < 5; r++) for (var c = 3; c < 5; c++) game.Life.SetCell(r, c, true);
        Hold(game, false, 2); Assert.AreEqual(4, game.Life.LiveCount);
        Hold(game, true, .27); Assert.AreEqual(160, game.Life.LiveCount);
        for (var r = 3; r < 5; r++) for (var c = 3; c < 5; c++) Assert.IsTrue(game.Life.IsAlive(r, c));
        Assert.AreEqual(0L, game.Score);
        game.Life.Clear(); Hold(game, false, 2); Assert.AreEqual(0, game.Life.LiveCount);
        Hold(game, true, .1); Assert.AreEqual(0, game.Life.LiveCount, "Cooldown advances only in GROW.");
    }

    [TestMethod]
    public void Pending_Birth_Is_Cancelled_If_Natural_Population_Recovers()
    {
        using var game = new ArcadeSession(); game.Life.Clear(); Hold(game, true, .1);
        Assert.IsTrue(game.Replenishing);
        game.Life.ReplenishTo(100); Hold(game, true, .02);
        Assert.IsFalse(game.Replenishing); Assert.AreEqual(100, game.Life.LiveCount);
    }

    [TestMethod]
    public void Colony_Uses_Dead_Boundaries_And_Tiny_Fields_Replenish_Safely()
    {
        using var life = new LifeSimulation(5, 5, 1, false); life.Clear();
        life.SetCell(0, 4, true); life.SetCell(4, 0, true); life.SetCell(4, 4, true); life.Step();
        Assert.IsFalse(life.IsAlive(0, 0));
        using var tiny = new LifeSimulation(3, 3); tiny.Clear(); tiny.ReplenishTo(9); Assert.AreEqual(9, tiny.LiveCount);
    }

    [TestMethod]
    public void Sector_Changes_Only_On_A_Paddle_Return_And_Speed_Is_Capped()
    {
        using var game = new ArcadeSession();
        for (var i = 0; i < 400; i++) Hit(game);
        Assert.AreEqual(1, game.Sector); Assert.AreEqual(11, game.PendingSector);
        game.SetBallForTesting(new Vector2(1515, game.PaddleY), new Vector2(1000, 0));
        game.Advance(.03);
        Assert.AreEqual(11, game.Sector); Assert.AreEqual(1000f, game.Velocity.Length(), .02f); Assert.IsTrue(game.Velocity.X < 0);
        Assert.AreEqual(10d, game.LifeRate);
    }

    [TestMethod]
    public void Moving_Paddle_And_Maximum_Speed_Cell_Contacts_Do_Not_Tunnel()
    {
        using var game = new ArcadeSession();
        game.SetIntent(-1); game.SetBallForTesting(new Vector2(1510, game.PaddleY - 30), new Vector2(1000, 0));
        game.Advance(.05); Assert.IsTrue(game.Velocity.X < 0); Assert.IsTrue(game.Velocity.Y < 0);
        Hit(game, 1000); Assert.IsTrue(game.Velocity.X > 0); Assert.IsTrue(float.IsFinite(game.Ball.X));
    }

    [TestMethod]
    public void Jitter_Is_Bounded_Symmetric_And_Preserves_Speed()
    {
        var random = new Random(101); var positive = 0; var negative = 0;
        for (var i = 0; i < 2048; i++)
        {
            var fraction = (float)(random.NextDouble() * .04 - .02);
            var v = ArcadeSession.AddJitter(new Vector2(1000, 0), fraction);
            Assert.AreEqual(1000f, v.Length(), .001f); Assert.IsTrue(Math.Abs(v.Y / v.X) <= .020001f);
            if (v.Y > 0) positive++; else negative++;
        }
        Assert.IsTrue(positive > 800 && negative > 800);
    }

    [TestMethod]
    public void Invalid_Deltas_Are_Rejected_And_Empty_Left_Wall_Returns_Ball()
    {
        using var game = new ArcadeSession();
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => game.Advance(double.NaN));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => game.Advance(-1));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => game.Advance(.2));
        game.Life.Clear(); game.SetBallForTesting(new Vector2(12, 500), new Vector2(-1000, 100));
        game.Advance(.01); Assert.IsTrue(game.Velocity.X > 0); Assert.AreEqual(0L, game.Score);
    }

    [TestMethod]
    public void Scripted_Skill_Player_Produces_Deterministic_Long_Rallies()
    {
        for (var seed = 1; seed <= 3; seed++)
        {
            using var a = new ArcadeSession(seed); using var b = new ArcadeSession(seed);
            a.LaunchOrResume(); b.LaunchOrResume();
            for (var frame = 0; frame < 120 * 120; frame++)
            {
                var target = Predict(a);
                a.SetPointerTarget(target); b.SetPointerTarget(target);
                a.Advance(1d / 120); b.Advance(1d / 120);
                if (a.State == RunState.Ready) { a.LaunchOrResume(); b.LaunchOrResume(); }
                Assert.IsTrue(float.IsFinite(a.Velocity.X) && float.IsFinite(a.Ball.Y));
                Assert.AreEqual(a.Ball, b.Ball); Assert.AreEqual(a.Score, b.Score);
            }
            Assert.IsTrue(a.Destroyed >= 8, $"Seed {seed}: arcade must produce cell contacts, not idle orbits.");
            TestContext.WriteLine($"seed {seed}: {a.Destroyed} cells, score {a.Score}, best chain {a.BestChain}, lives {a.Lives}, state {a.State}");
        }
    }
    internal static float Predict(ArcadeSession game)
    {
        if (game.Velocity.X <= 0) return 450;
        var t = Math.Max(0, (ArcadeSession.PaddleX - 20 - game.Ball.X) / game.Velocity.X);
        var p = game.Ball.Y + game.Velocity.Y * t - 10;
        var span = ArcadeSession.Height - 20; var folded = ((p % (span * 2)) + span * 2) % (span * 2);
        return 10 + (folded < span ? folded : span * 2 - folded);
    }
    internal static void Hit(ArcadeSession game, float speed = 640)
    {
        game.Life.Clear(); game.Life.SetCell(15, 20, true);
        game.SetBallForTesting(new Vector2(ArcadeSession.FieldX + 20 * 24 + 23 + 10.5f, ArcadeSession.FieldY + 15 * 24 + 12), new Vector2(-speed, 0));
        game.Advance(1d / 120);
    }
    internal static void Miss(ArcadeSession game)
    { game.SetBallForTesting(new Vector2(1612, 450), new Vector2(640, 0)); game.Advance(1d / 120); }
    private static void Hold(ArcadeSession game, bool right, double seconds)
    {
        while (seconds > 1e-8)
        {
            var dt = Math.Min(1d / 120, seconds);
            game.SetBallForTesting(new Vector2(right ? 1000 : 765, 450), new Vector2(right ? 640 : -640, 0));
            game.Advance(dt); seconds -= dt;
        }
    }
    private static bool[] Snapshot(ArcadeSession game) => Enumerable.Range(0, game.Life.Rows).SelectMany(r => Enumerable.Range(0, game.Life.Columns).Select(c => game.Life.IsAlive(r, c))).ToArray();
}
