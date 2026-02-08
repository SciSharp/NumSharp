# #421: Performance

- **URL:** https://github.com/SciSharp/NumSharp/issues/421
- **State:** OPEN
- **Author:** @mishun
- **Created:** 2020-08-16T01:11:02Z
- **Updated:** 2020-08-17T03:15:52Z

## Description

Hi!
I just tried to do some basic computational geometry with NumSharp and compare it with naive implementation like that:
```F#
﻿open System
open System.Diagnostics
open System.Numerics
open NumSharp

[<Struct>]
type Line2f = {
    Normal : Vector2
    Offset : single
}

let residualsNaive (points : ReadOnlySpan<Vector2>, line : Line2f) =
    let residuals = Array.create points.Length 0.0f
    for i in 0 .. points.Length - 1 do
        residuals.[i] <- abs (Vector2.Dot(line.Normal, points.[i]) + line.Offset)
    residuals


let residuals (points : NDArray, line : Line2f) =
    let l = np.array(line.Normal.X, line.Normal.Y)
    let signed = np.dot(&points, &l) + line.Offset
    np.abs &signed


[<EntryPoint>]
let main argv =
    let n = 10000000

    let rand = Random ()
    let points = Array.init n (fun _ -> Vector2 (single (rand.NextDouble ()), single (rand.NextDouble ())))
    let line =
        let a = Math.PI * rand.NextDouble ()
        {   Normal = Vector2 (single (cos a), single (sin a))
            Offset = single (rand.NextDouble ())
        }

    let swNaive = Stopwatch.StartNew ()
    let res1 = residualsNaive (ReadOnlySpan points, line)
    swNaive.Stop ()

    let pointsArray = np.array([| for p in points do yield p.X ; yield p.Y |]).reshape(n, 2)

    let swNumSharp = Stopwatch.StartNew ()
    let res2 = residuals (pointsArray, line)
    swNumSharp.Stop ()

    printfn "Naive: %A\nNumSharp: %A\n\n" swNaive.Elapsed swNumSharp.Elapsed
    printfn "Result:\n%A vs %A" (Array.sum res1) (np.sum (&res2))

    0
```
and got:
```
$ dotnet run -c release
Naive: 00:00:00.0762210
NumSharp: 00:00:09.7929785
```
Maybe I'm doing something very-very wrong, but 2-something orders of magnitude difference looks a bit too much even if there are just managed loops inside NumSharp's functions.

Tested with
```
<PackageReference Include="NumSharp" Version="0.20.5" />)
```

## Comments

### Comment 1 by @Oceania2018 (2020-08-16T12:49:32Z)

Can you try in Tensorflow.net preview3?

### Comment 2 by @mishun (2020-08-17T03:15:52Z)

You mean use NumSharp that is installed with Tensorflow.NET 0.20.0-preview3 nuget package? Runtime looks roughly the same except np.sum is now throwing an exception.
