// Karpathy Gist numerical ports; source links and pins appear at the top of each implementation.
#:project ../NumSharp.GistExamples.csproj
#:property PublishAot=false

using NumSharp.Examples.Gist;
using NumSharp.Examples.Gist.Karpathy;
using NumSharp.Interop.OpenBLAS;

OpenBlasEngine.Enable(threads: 1);
string selected = args.Length == 0 ? "all" : args[0].ToLowerInvariant();
var demos = new (string Name, Action Run)[]
{
    ("rnn", MinimalCharacterRnn.Demo),
    ("pong", PongPolicyGradient.Demo),
    ("lstm", BatchedLstm.Demo),
    ("microgpt", MicroGpt.Demo),
    ("nes", NaturalEvolutionStrategies.Demo),
    ("walk", StableDiffusionWalk.Demo)
};
if (selected != "all" && !demos.Any(demo => demo.Name == selected))
    throw new ArgumentException("Choose all, rnn, pong, lstm, microgpt, nes or walk.");
foreach (var demo in demos)
{
    if (selected != "all" && selected != demo.Name) continue;
    Console.WriteLine($"\n--- {demo.Name} ---");
    demo.Run();
}
