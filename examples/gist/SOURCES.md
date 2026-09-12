# Pinned sources and attribution

This collection selects the **top ten gists**, not the top ten individual files, from our saved [stars × NumPy-usage ranking](../../docs/articles/gist/rankings/stars-x-usage-2026-09-09/RANKING.md). Each C# implementation begins with its original gist URL and pinned revision.

The user approved functional numerical routines only. Framework applications are not silently replaced with toy models: the ports expose useful numerical entry points and runnable demonstrations, while identifying the omitted model training/inference, audio/image IO, plotting and event-file writing. See [README.md](README.md) for scope, execution and test coverage.

## Selection is a measurement snapshot

The ranking's score is **saved gist stars × accepted NumPy call sites**, including ndarray methods and excluding import statements. Counts describe the whole measured gist at its pinned revision—not runtime executions, not the C# port, and not a per-file ranking. Stars were not refreshed for this porting task. The sample is our measured `complete-parse` corpus, not all public gists.

| Rank | Pinned gist | Saved stars | NumPy call sites | Score | C# implementation |
|---|---|---:|---:|---:|---|
| 1 | [bshishov/5dc237f59f019b26145648e2124ca1c9](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9/824c2f332d919185289ded2709752c67bd41244d) | 230 | 60 | 13800 | [bshishov.ForecastingMetrics.cs](bshishov.ForecastingMetrics.cs) |
| 2 | [bwhite/3726239](https://gist.github.com/bwhite/3726239/2c92e90259b01b4a657d20c0ad8390caadd59c8b) | 529 | 17 | 8993 | [bwhite.RankingMetrics.cs](bwhite.RankingMetrics.cs) |
| 3 | [endolith/255291](https://gist.github.com/endolith/255291/f94aa1d3d63a8ec3c0e80f30c6f9f4e375ceea0f) | 244 | 16 | 3904 | [endolith.FrequencyEstimation.cs](endolith.FrequencyEstimation.cs) |
| 4 | [karpathy/00103b0037c5aaea32fe1da1af553355](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355/cfec3bb84900cc3abfe5539d6d69e3fb1f2921b8) | 383 | 8 | 3064 | [karpathy.StableDiffusionWalk.cs](karpathy.StableDiffusionWalk.cs) |
| 5 | [EderSantana/c7222daa328f0e885093](https://gist.github.com/EderSantana/c7222daa328f0e885093/90a8dafa550b581abbe2d578eb10acec0fe3dcd3) | 157 | 13 | 2041 | [EderSantana.CatchReinforcementLearning.cs](EderSantana.CatchReinforcementLearning.cs) |
| 6 | [karpathy/77fbb6a8dac5395f1b73e7a89300318d](https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d/2d87bda1fe2a1711eac65c65a2d215f6cfb799e3) | 195 | 10 | 1950 | [karpathy.NaturalEvolutionStrategies.cs](karpathy.NaturalEvolutionStrategies.cs) |
| 7 | [endolith/250860](https://gist.github.com/endolith/250860/bd0936f3983d43b37472f7b59991fdd4bcbb35c5) | 178 | 10 | 1780 | [endolith.PeakDetection.cs](endolith.PeakDetection.cs) |
| 8 | [gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514/abe6172ae826c4c75fc344ba1def8b756ae9e8a7) | 226 | 7 | 1582 | [gyglim.TensorboardHistogram.cs](gyglim.TensorboardHistogram.cs) |
| 9 | [jkleint/1d878d0401b28b281eb75016ed29f2ee](https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee/b12c675bed3be4126fd9f502f0cd9b33ec4b2186) | 138 | 10 | 1380 | [jkleint.TimeseriesCnn.cs](jkleint.TimeseriesCnn.cs) |
| 10 | [joelouismarino/a2ede9ab3928f999575423b9887abd14](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/6c17b8dd160d1212f11a7918d00c7c0c0c85e812) | 215 | 6 | 1290 | [joelouismarino.GoogLeNet.cs](joelouismarino.GoogLeNet.cs) |

Machine-readable facts, original manifest entries, observation timestamps, full revisions, source URLs, cached paths and verification hashes are in [sources.json](sources.json). Port-specific interpretation is kept separately under each record's `port` field; upstream facts are under `facts`.

Selection inputs:

- [ranking.json](../../docs/articles/gist/rankings/stars-x-usage-2026-09-09/ranking.json), SHA-256 `bdfaddfd2fc517a5af5b45cf0d9e1b64e7f356663730e684d0eab5d65ce07e5a`.
- [provenance.json](../../docs/articles/gist/rankings/stars-x-usage-2026-09-09/provenance.json), SHA-256 `b3672421b6aca07191ed76e35356b5fbad9693bfb66638380283c543929a94ef`.
- The source manifest for each origin run, each pinned GitHub API response and their hashes are recorded per gist in `sources.json`.

## Fourteen Python source files, ten C# files

Ten main files and four companions were inventoried. `parabolic.py`, `lrn.py` and `pool_helper.py` contribute numerical routines to their associated C# file. Catch's `test.py` is retained as application-playback context; loading a trained Keras model and producing animation frames is not claimed as a NumSharp port.

Every row below matched both its saved `files.jsonl` byte count/SHA-256 and the UTF-8 content embedded in its pinned API response. Source hashes identify the exact reviewed bytes, not a mutable gist URL.

| Rank | Pinned source file | Role | Bytes | SHA-256 |
|---|---|---|---:|---|
| 1 | [forecasting_metrics.py](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9/824c2f332d919185289ded2709752c67bd41244d#file-forecasting_metrics-py) | Main | 9800 | `df7a6d0709b228cd8ae9b44a416da200e7f988ef2ef3e0479b89f89bc80a17be` |
| 2 | [rank_metrics.py](https://gist.github.com/bwhite/3726239/2c92e90259b01b4a657d20c0ad8390caadd59c8b#file-rank_metrics-py) | Main | 6204 | `b9abd30faa5fe1d4ff968769888bd774ad2220ce1d64365c66aa9b0a42055a1c` |
| 3 | [frequency_estimator.py](https://gist.github.com/endolith/255291/f94aa1d3d63a8ec3c0e80f30c6f9f4e375ceea0f#file-frequency_estimator-py) | Main | 3749 | `7be86bc481c64a7df601430b9ebc44e41224011eea65d6b02c7557972a8ac86d` |
| 3 | [parabolic.py](https://gist.github.com/endolith/255291/f94aa1d3d63a8ec3c0e80f30c6f9f4e375ceea0f#file-parabolic-py) | Numerical companion | 1576 | `38243bbbab3481a5839228bf71ddfedca79534fb49eaf921b6c644eb7367e5d0` |
| 4 | [stablediffusionwalk.py](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355/cfec3bb84900cc3abfe5539d6d69e3fb1f2921b8#file-stablediffusionwalk-py) | Main | 7696 | `cd5c47d7fbfa6d09fafbd6d6938c8a3217bf734c30ce9fb222b238aeb88c8451` |
| 5 | [qlearn.py](https://gist.github.com/EderSantana/c7222daa328f0e885093/90a8dafa550b581abbe2d578eb10acec0fe3dcd3#file-qlearn-py) | Main | 5535 | `cdb007b06d61b69c0207450920e3754787bcd3d422a0be2e145f2364ed05bfc7` |
| 5 | [test.py](https://gist.github.com/EderSantana/c7222daa328f0e885093/90a8dafa550b581abbe2d578eb10acec0fe3dcd3#file-test-py) | Playback reference (not an application port) | 1235 | `c577fb519cee9c000f99d1e015bb251ab309ef4326de5e1c9af4735f41063a80` |
| 6 | [nes.py](https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d/2d87bda1fe2a1711eac65c65a2d215f6cfb799e3#file-nes-py) | Main | 3306 | `4144833dab064335631040b233d19aea2392f1e460a2cae1491a5def5d912f3b` |
| 7 | [peakdetect.py](https://gist.github.com/endolith/250860/bd0936f3983d43b37472f7b59991fdd4bcbb35c5#file-peakdetect-py) | Main | 2452 | `7e7f6a87e7e7479a8d2d080c9806ef0e9249faa957b23c9269d781d62c228e48` |
| 8 | [tensorboard_logging.py](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514/abe6172ae826c4c75fc344ba1def8b756ae9e8a7#file-tensorboard_logging-py) | Main | 2973 | `6eb1095ec6d6af6316df0c1dfc86203d774fcdf5704ed72a6662496e07429aec` |
| 9 | [timeseries_cnn.py](https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee/b12c675bed3be4126fd9f502f0cd9b33ec4b2186#file-timeseries_cnn-py) | Main | 7062 | `d94a17b2a3d76ae273fe4f43812bd31fb40a0a4ac37dfcf7c1646df14f629965` |
| 10 | [googlenet.py](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/6c17b8dd160d1212f11a7918d00c7c0c0c85e812#file-googlenet-py) | Main | 18049 | `59833005f51947232b73beacd1047121aeeee275ac9fbaa8fcc3b65065f8a51d` |
| 10 | [lrn.py](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/6c17b8dd160d1212f11a7918d00c7c0c0c85e812#file-lrn-py) | Numerical companion | 1580 | `74ef3eb3e1828da44eb35fd4ff83ba83f79598cac0188145c67d72b15961f815` |
| 10 | [pool_helper.py](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/6c17b8dd160d1212f11a7918d00c7c0c0c85e812#file-pool_helper-py) | Numerical companion | 388 | `5fc18f52fb9c9c4bf3fdcf7015c163d09a82cc1e75c7fae9a4d8abe2689946c5` |

The six additional non-Python files retained in pinned API responses—frequency notes/readme, Catch readme, peak-detection readme/MATLAB original, and GoogLeNet readme—are inventoried separately as auxiliary evidence in `sources.json`. Their UTF-8 content hashes and sizes are recorded without reproducing their full content.

## Observed license declarations

Public availability and attribution are not license grants. The following are observations about the pinned evidence, not inferred permissions for unrelated versions or linked projects.

- **Rank 5, Catch:** [CATCH_Keras_RL.md](https://gist.githubusercontent.com/EderSantana/c7222daa328f0e885093/raw/abfee796899069fe3519d0e1eaa4ffe5e2e73734/CATCH_Keras_RL.md) declares MIT in its License section. It links to a description of MIT; the complete license text is not embedded in the pinned gist.
- **Rank 7, peak detection:** the Python file and MATLAB original attribute the algorithm to Eli Billauer (3 April 2005) and declare public-domain release. Crucially, [endolith's pinned readme](https://gist.githubusercontent.com/endolith/250860/raw/63205fa50e0cb7411ac82a93dc0db85f74fd2732/readme.md) explicitly extends that declaration to the Python translation.
- **Rank 8, TensorBoard logging:** the [pinned module header](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514/abe6172ae826c4c75fc344ba1def8b756ae9e8a7#file-tensorboard_logging-py) says `BSD License 2.0`, without complete license text. We preserve that wording and do **not** reinterpret it as BSD-2-Clause, BSD-3-Clause or another SPDX identifier.
- **Ranks 1, 2, 3, 4, 6, 9 and 10:** no explicit license declaration was observed in the reviewed pinned files. Their numerical methods are independently implemented in C#; this documentation does not label those gists public domain or invent a license.

No bulk upstream source or prose is copied into this provenance document or its JSON. The local research cache already contains the acquired evidence; it is not a new vendored source distribution.

## What was verified and how to repeat it

Verification timestamp: `2026-09-09T15:33:35.2836279Z`. This step made **zero network requests**, refetched no source files and refreshed no stars.

1. Read ranks 1–10 from the saved ranking, retaining the original row rather than recomputing a new selection.
2. Resolve each gist's `origin_run/files.jsonl`; require entity ID and revision to agree with the ranking.
3. Hash every local cached Python file and compare its byte count and SHA-256 with the manifest.
4. Independently UTF-8 encode each corresponding pinned API `files[filename].content`; require the same byte count and SHA-256 and no truncated content.
5. Record hashes for the ranking, ranking provenance, origin manifests and pinned API JSON. Require each API body's hash/byte count to match its original successful acquisition receipt (all ten matched), and retain that receipt's hash. Match each C# header to its gist URL, revision and main source filename.
6. Keep numerical validation separate: tests evaluate reviewed mathematical definitions or independently written NumPy/SciPy references. They do not execute original scripts as complete Keras, TensorFlow, Stable Diffusion, audio or plotting applications.

A read-only recheck of the fourteen cached Python files, from the repository root:

```powershell
$sourceInventory = Get-Content examples/gist/sources.json -Raw | ConvertFrom-Json
foreach ($gistRecord in $sourceInventory.gists) {
    foreach ($sourceRecord in $gistRecord.facts.source_files) {
        $sourcePath = $sourceRecord.verification.local_path
        $digest = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLower()
        $length = (Get-Item -LiteralPath $sourcePath).Length
        if ($digest -ne $sourceRecord.manifest_entry.sha256 -or
            $length -ne $sourceRecord.manifest_entry.bytes) {
            throw "Pinned source mismatch: $sourcePath"
        }
    }
}
```

Caches are local research artifacts and may be gitignored. If they are absent in another checkout, use the recorded immutable source links and acquisition receipts; do not substitute a gist's current head and continue claiming the pinned hashes. Evidence integrity does not itself prove numerical parity—consult the tested cases and explicit exceptions in the main README.
