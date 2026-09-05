using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class LifeSimulationTests
{
    [TestMethod]
    public void NumSharp_Field_Matches_Independent_Conway_Reference()
    {
        using var life = new LifeSimulation(13, 19, 101);
        var reference = new bool[13, 19];
        for (var row = 0; row < 13; row++)
            for (var col = 0; col < 19; col++)
                reference[row, col] = life.IsAlive(row, col);
        for (var generation = 0; generation < 120; generation++)
        {
            var next = new bool[13, 19];
            var count = 0;
            for (var row = 0; row < 13; row++)
                for (var col = 0; col < 19; col++)
                {
                    var neighbors = 0;
                    for (var dy = -1; dy <= 1; dy++)
                        for (var dx = -1; dx <= 1; dx++)
                            if ((dx != 0 || dy != 0) && reference[(row + dy + 13) % 13, (col + dx + 19) % 19]) neighbors++;
                    next[row, col] = neighbors == 3 || (neighbors == 2 && reference[row, col]);
                    if (next[row, col]) count++;
                }
            life.Step();
            for (var row = 0; row < 13; row++)
                for (var col = 0; col < 19; col++)
                    Assert.AreEqual(next[row, col], life.IsAlive(row, col), $"{generation}: {row}, {col}");
            Assert.AreEqual(count, life.LiveCount);
            reference = next;
        }
    }

    [TestMethod]
    public void Patterns_And_Strokes_Have_Expected_Evolution()
    {
        using var life = new LifeSimulation();
        life.LoadPattern(true);
        Assert.AreEqual(48, life.LiveCount);
        var initial = Enumerable.Range(0, life.Rows).SelectMany(row => Enumerable.Range(0, life.Columns).Select(col => life.IsAlive(row, col))).ToArray();
        for (var i = 0; i < 3; i++) life.Step();
        CollectionAssert.AreEqual(initial, Enumerable.Range(0, life.Rows).SelectMany(row => Enumerable.Range(0, life.Columns).Select(col => life.IsAlive(row, col))).ToArray());
        life.Clear();
        life.PaintLine(5, 2, 5, 40, true);
        Assert.AreEqual(39, life.LiveCount);
        life.PaintLine(5, 40, 5, 2, false);
        Assert.AreEqual(0, life.LiveCount);
        life.Reseed(0);
        Assert.AreEqual(0, life.LiveCount);
    }

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
