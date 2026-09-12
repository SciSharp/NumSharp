// Ten functional numerical ports; each implementation file starts with its original pinned Gist.
// Run: dotnet run --file examples/gist/run.cs -c Release -- all
#:project NumSharp.GistExamples.csproj
#:property PublishAot=false

using NumSharp.Examples.Gist;
using NumSharp.Interop.OpenBLAS;

// The least-squares interpolation companion needs LAPACK; the bundled backend also
// gives floating-point products the same implementation/thread count as the live oracle.
OpenBlasEngine.Enable(threads: 1);

var examples = new (string Name, Action Run)[]
{
    ("Forecasting metrics", ForecastingMetrics.Demo),
    ("Ranking metrics", RankingMetrics.Demo),
    ("Frequency estimation", FrequencyEstimation.Demo),
    ("Stable Diffusion numerical routines", StableDiffusionWalk.Demo),
    ("Catch environment and replay", CatchReinforcementLearning.Demo),
    ("Natural evolution strategies", NaturalEvolutionStrategies.Demo),
    ("Peak detection", PeakDetection.Demo),
    ("TensorBoard histogram payload", TensorboardHistogram.Demo),
    ("Time-series CNN data preparation", TimeseriesCnn.Demo),
    ("GoogLeNet numerical routines", GoogLeNet.Demo)
};

if (args.Length > 1 || (args.Length == 1 && args[0] != "all" &&
    (!int.TryParse(args[0], out int parsed) || parsed < 1 || parsed > examples.Length)))
{
    Console.Error.WriteLine("Choose all or an example number from 1 to 10.");
    return 2;
}

int selected = args.Length == 1 && int.TryParse(args[0], out int number) ? number : 0;
for (int i = 0; i < examples.Length; i++)
{
    if (selected != 0 && selected != i + 1) continue;
    Console.WriteLine($"\n{i + 1:D2}. {examples[i].Name}");
    examples[i].Run();
}
return 0;
