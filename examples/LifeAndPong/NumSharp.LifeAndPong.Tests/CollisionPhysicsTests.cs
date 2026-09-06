using System.Numerics;
using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class CollisionPhysicsTests
{
    [TestMethod]
    public void Swept_Face_Contact_Has_Analytic_Time_And_Reflection()
    {
        Assert.IsTrue(CollisionMath.SweepRoundedBox(new Vector2(0, 5), new Vector2(10, 2), 1, 1, 10, 0, 4, 10, 0, out var hit));
        Assert.AreEqual(.9, hit.Time, 1e-7); Assert.AreEqual(-Vector2.UnitX, hit.Normal);
        var result = CollisionMath.ElasticManifold(new Vector2(10, 2), [new(hit.Normal, Vector2.Zero)]);
        Assert.AreEqual(new Vector2(-10, 2), result);
    }
    [TestMethod]
    public void Rounded_Corner_Does_Not_Collide_With_The_Empty_Aabb_Corner()
    {
        var position = new Vector2(5, 9.25f); var velocity = new Vector2(10, 0);
        Assert.IsFalse(CollisionMath.SweepRoundedBox(position, velocity, .45, 1, 10, 10, 10, 10, 2, out _));
        Assert.IsTrue(CollisionMath.SweepRoundedBox(position, velocity, 1, 1, 10, 10, 10, 10, 2, out var hit));
        var expectedTime = (12 - Math.Sqrt(9 - 2.75 * 2.75) - 5) / 10;
        Assert.AreEqual(expectedTime, hit.Time, 1e-7);
        Assert.IsTrue(hit.Normal.X < 0 && hit.Normal.Y < 0);
        var reflected = CollisionMath.ElasticManifold(velocity, [new(hit.Normal, Vector2.Zero)]);
        Assert.AreEqual(velocity.Length(), reflected.Length(), .0001f);
        Assert.IsTrue(reflected.Y < 0 && reflected.X > 0, "Glancing corner contact is not a forced horizontal reversal.");
    }
    [TestMethod]
    public void Moving_Paddle_Impulse_Conserves_Relative_Normal_Speed()
    {
        var incoming = new Vector2(20, -300); var surface = new Vector2(0, 100);
        var result = CollisionMath.ElasticManifold(incoming, [new(Vector2.UnitY, surface)]);
        Assert.AreEqual(new Vector2(20, 500), result);
        Assert.AreEqual(-Vector2.Dot(incoming - surface, Vector2.UnitY), Vector2.Dot(result - surface, Vector2.UnitY));
        Assert.AreEqual(80000f, .5f * (result.LengthSquared() - incoming.LengthSquared()), .01f, "Energy gain is work done by the moving paddle.");
    }
    [TestMethod]
    public void Paddle_Friction_Is_Coulomb_Limited_And_Accounts_For_Spin_Energy()
    {
        var incoming = new Vector2(1000, 300); var reflected = new Vector2(-1000, 300);
        var response = CollisionMath.PaddleFriction(incoming, reflected, Vector2.Zero, -Vector2.UnitX, 0, 10);
        Assert.AreEqual(new Vector2(-1000, 200), response.Velocity); Assert.AreEqual(-20f, response.Spin);
        var initialEnergy = .5 * incoming.LengthSquared();
        var finalEnergy = .5 * response.Velocity.LengthSquared() + .5 * 50 * response.Spin * response.Spin;
        Assert.IsTrue(finalEnergy <= initialEnergy);
        Assert.AreEqual(0f, -response.Velocity.Y - response.Spin * 10, .0001f, "Contact slip is removed, not arbitrary tangential acceleration.");
    }
    [TestMethod]
    public void Simultaneous_Corner_Constraints_Preserve_Elastic_Energy()
    {
        var incoming = new Vector2(-300, -400);
        var result = CollisionMath.ElasticManifold(incoming, [new(Vector2.UnitX, Vector2.Zero), new(Vector2.UnitY, Vector2.Zero)]);
        Assert.AreEqual(new Vector2(300, 400), result); Assert.AreEqual(500f, result.Length(), .0001f);
        var n1 = Vector2.Normalize(new Vector2(.1f, 1)); var n2 = Vector2.Normalize(new Vector2(-.1f, 1));
        result = CollisionMath.ElasticManifold(new Vector2(0, -640), [new(n1, Vector2.Zero), new(n2, Vector2.Zero)]);
        Assert.AreEqual(0f, result.X, .01f); Assert.AreEqual(640f, result.Y, .01f);
    }
    [TestMethod]
    public void Five_Percent_Noise_Preserves_Speed_And_Cannot_Reenter_Contacts()
    {
        Assert.AreEqual(.05f, ArcadeSession.Jitter);
        var original = new Vector2(1000, .001f);
        var perturbed = CollisionMath.DirectionNoise(new Vector2(1000, 0), .05f);
        Assert.AreEqual(.05f, perturbed.Y / perturbed.X, .000001f);
        Assert.AreEqual(1000f, perturbed.Length(), .001f);
        var safe = CollisionMath.SafeNoise(original, -.05f, [new(Vector2.UnitX, Vector2.Zero), new(Vector2.UnitY, Vector2.Zero)]);
        Assert.IsTrue(safe.Y >= -.0011f); Assert.AreEqual(original.Length(), safe.Length(), .001f);
        Assert.AreEqual(Vector2.Zero, CollisionMath.DirectionNoise(Vector2.Zero, .05f));
    }
    [TestMethod]
    public void Whole_Frame_Consumes_Remaining_Time_After_Wall_Impact()
    {
        using var game = new ArcadeSession { NoiseEnabled = false }; game.Life.Clear();
        game.SetBallForTesting(new Vector2(1200, 20), new Vector2(-40, -200)); game.Advance(.1);
        Assert.AreEqual(new Vector2(-40, 200), game.Velocity);
        Assert.AreEqual(1196f, game.Ball.X, .003f); Assert.AreEqual(20f, game.Ball.Y, .003f);
        Assert.IsNull(game.PhysicsIssue);
    }
    [TestMethod]
    public void Shallow_Trajectories_Are_Not_Clamped_Or_Automatically_Redirected()
    {
        using var game = new ArcadeSession { NoiseEnabled = false }; game.Life.Clear();
        game.SetBallForTesting(new Vector2(900, 450), new Vector2(1, 100));
        for (var i = 0; i < 60; i++) game.Advance(.1);
        Assert.AreEqual(1f, game.Velocity.X, .00001f); Assert.AreEqual(-100f, game.Velocity.Y, .00001f);
        Assert.AreEqual(906f, game.Ball.X, .05f); Assert.AreEqual(730f, game.Ball.Y, .05f);
        Assert.AreEqual(RunState.Playing, game.State); Assert.IsNull(game.PhysicsIssue);
    }
    [TestMethod]
    public void Very_Fast_Ball_Still_Hits_One_Cell_Without_Tunneling()
    {
        using var game = new ArcadeSession { NoiseEnabled = false }; game.Life.Clear(); game.Life.SetCell(15, 20, true);
        game.SetBallForTesting(new Vector2(1000, ArcadeSession.FieldY + 15 * 24 + 12), new Vector2(-100000, 0));
        game.Advance(.01);
        Assert.AreEqual(1, game.Destroyed); Assert.IsFalse(game.Life.IsAlive(15, 20));
        Assert.IsNull(game.PhysicsIssue); Assert.AreEqual(100000f, game.Velocity.Length(), .1f);
    }
    [TestMethod]
    public void Stationary_Paddle_Does_Not_Replace_An_Incoming_Speed_With_A_Sector_Target()
    {
        using var game = new ArcadeSession { NoiseEnabled = false };
        game.SetBallForTesting(new Vector2(1515, game.PaddleY), new Vector2(275, 0)); game.Advance(.05);
        Assert.AreEqual(-275f, game.Velocity.X, .01f); Assert.AreEqual(0f, game.Velocity.Y, .01f);
    }
}
