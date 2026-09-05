using System.Text.Json;

namespace NumSharp.LifeAndPong.Models;

public sealed record RunResult(long Score, int Chain, int Destroyed, int Sector, int Seed, string Version);

public sealed class PlayerProfile
{
    private readonly string? _path;
    public bool Sound { get; set; } = true;
    public bool ReducedMotion { get; set; }
    public bool HighContrast { get; set; }
    public List<RunResult> Results { get; private set; } = [];
    public long Best => Results.Count == 0 ? 0 : Results.Max(r => r.Score);
    public string? SaveError { get; private set; }
    public PlayerProfile(string? path = null)
    {
        _path = path;
        if (path is null) return;
        try
        {
            if (!File.Exists(path) || new FileInfo(path).Length > 64_000) return;
            var data = JsonSerializer.Deserialize<ProfileData>(File.ReadAllText(path));
            if (data is null) return;
            Sound = data.Sound; ReducedMotion = data.ReducedMotion; HighContrast = data.HighContrast;
            Results = (data.Results ?? []).Where(r => r is not null && r.Score >= 0 && r.Chain >= 0 && r.Destroyed >= 0 && r.Sector >= 1)
                .OrderByDescending(r => r.Score).Take(5).ToList();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or NotSupportedException)
        { SaveError = "Local scores could not be loaded. Play is still available."; }
    }
    public static PlayerProfile OpenLocal() => new(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NumSharp", "LifeArcade", "profile.json"));
    public void Record(ArcadeSession session)
    {
        Results.Add(new RunResult(session.Score, session.BestChain, session.Destroyed, session.Sector, session.Seed, ArcadeSession.Version));
        Results = Results.OrderByDescending(r => r.Score).Take(5).ToList(); Save();
    }
    public void Save()
    {
        if (_path is null) return;
        var temporary = _path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(_path))!);
            File.WriteAllText(temporary, JsonSerializer.Serialize(new ProfileData(Sound, ReducedMotion, HighContrast, Results)));
            File.Move(temporary, _path, overwrite: true); SaveError = null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        { SaveError = "Local scores could not be saved. This run is still playable."; }
        finally
        {
            try { if (File.Exists(temporary)) File.Delete(temporary); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
    }
    private sealed record ProfileData(bool Sound, bool ReducedMotion, bool HighContrast, List<RunResult>? Results);
}
public interface IGameAudio : IDisposable { bool Available { get; } void Play(ArcadeEvent item); }
public sealed class SilentGameAudio : IGameAudio
{
    public bool Available => false;
    public void Play(ArcadeEvent item) { }
    public void Dispose() { }
}
