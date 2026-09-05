using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class LifeSimulationTests
{
    [TestMethod]
    public void Blinker_Advances_And_Returns_After_Two_Generations()
    {
        using var life = new LifeSimulation(7, 7);
        life.Clear();
        life.SetCell(3, 2, true);
        life.SetCell(3, 3, true);
        life.SetCell(3, 4, true);

        life.Step();

        Assert.IsTrue(life.IsAlive(2, 3));
        Assert.IsTrue(life.IsAlive(3, 3));
        Assert.IsTrue(life.IsAlive(4, 3));
        Assert.AreEqual(3, life.LiveCount);

        life.Step();

        Assert.IsTrue(life.IsAlive(3, 2));
        Assert.IsTrue(life.IsAlive(3, 3));
        Assert.IsTrue(life.IsAlive(3, 4));
        Assert.AreEqual(2L, life.Generation);
    }

    [TestMethod]
    public void Toroidal_Edges_Keep_Corner_Births_Consistent()
    {
        using var life = new LifeSimulation(5, 5);
        life.Clear();
        life.SetCell(0, 4, true);
        life.SetCell(4, 0, true);
        life.SetCell(4, 4, true);

        life.Step();

        Assert.IsTrue(life.IsAlive(0, 0));
    }
}
