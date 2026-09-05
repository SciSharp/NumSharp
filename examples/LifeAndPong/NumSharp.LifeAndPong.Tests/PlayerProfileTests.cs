using NumSharp.LifeAndPong.Models;

namespace NumSharp.LifeAndPong.Tests;

[TestClass]
public sealed class PlayerProfileTests
{
    [TestMethod]
    public void Profile_RoundTrips_Settings_And_Only_Five_Best_Runs()
    {
        var directory = Path.Combine(Path.GetTempPath(), "life-arcade-profile-" + Guid.NewGuid().ToString("N"));
        var file = Path.Combine(directory, "profile.json");
        var profile = new PlayerProfile(file) { Sound = false, ReducedMotion = true, HighContrast = true };
        using var game = new ArcadeSession();
        for (var i = 0; i < 7; i++) { game.SetScoreForTesting(i * 10); profile.Record(game); }
        var loaded = new PlayerProfile(file);
        Assert.AreEqual(60L, loaded.Best); Assert.AreEqual(5, loaded.Results.Count);
        Assert.IsFalse(loaded.Sound); Assert.IsTrue(loaded.ReducedMotion); Assert.IsTrue(loaded.HighContrast);
        Assert.AreEqual(ArcadeSession.Version, loaded.Results[0].Version);
        File.Delete(file); Directory.Delete(directory);
    }
    [TestMethod]
    public void Unavailable_Save_Path_Is_Nonfatal()
    {
        var file = Path.GetTempFileName();
        var profile = new PlayerProfile(Path.Combine(file, "not-a-directory", "profile.json"));
        profile.Save(); Assert.IsNotNull(profile.SaveError); File.Delete(file);
    }
}
