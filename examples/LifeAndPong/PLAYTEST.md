# Life Arcade player acceptance playtest

Use this guide to evaluate the complete Windows player journey and the
subjective fun/balance questions that automated tests cannot answer.

## Evidence status when this guide was authored

- Code candidate: `ad8442bd16bc7d8d476409ec1b98e9f824787ea7`
- Delivery: `DU-life-arcade-004`
- Code QA: approved in `E-01a07513-643b-7a93-aa6a-41754178b785`
- Human acceptance evidence supplied to the Writer: **none**

The current automated, headless, physics, fixture, and local-package evidence
does not prove that the game is fun, balanced, understandable in the first
minute, or worth replaying. Completing or reading this checklist is not itself
an acceptance result; record what a human actually observes.

## Prepare the candidate

1. Build a fresh local Windows package from the documentation revision under
   review by running `publish-windows.ps1` as described in [README.md](README.md).
2. Compare the ZIP with the adjacent `SHA256SUMS` file and require a `True`
   checksum result.
3. Extract `NumSharp-LifeAndPong-win-x64.zip` to a new writable destination,
   then open the resulting top-level `NumSharp-LifeAndPong-win-x64` application
   directory. Confirm that this application directory contains
   `NumSharp.LifeAndPong.Desktop.exe`, the .NET runtime, NumSharp assemblies,
   `LICENSE`, both packaged previews, and every root Markdown guide including
   this one.
4. Record the code revision, documentation revision, ZIP SHA-256, Windows
   version, display scaling, window size, input device, and whether reduced
   motion, high contrast, or sound is enabled.

This verifies a local candidate only. Do not report a public or hosted release
unless its independent publishing evidence exists.

## Complete player journey

Perform the following through the executable inside the extracted
`NumSharp-LifeAndPong-win-x64` application directory, without developer or
test-only controls:

1. **Launch and Ready:** start `NumSharp.LifeAndPong.Desktop.exe`. From the Ready
   screen, identify the right-hand paddle, three lives, score, next-cell award,
   sector progress, GROW/SHATTER rhythm, and Start action.
2. **Launch and aim:** move the paddle before launch, then use Start or `Space`.
   Try mouse control and both keyboard alias pairs (`W`/`S` and Up/Down). Check
   whether aim feels intentional and whether the ball remains easy to track.
3. **Grow and shatter:** observe Life evolving while the ball is in the right
   30% GROW side and freezing into solid targets in the left 70% SHATTER side.
   Confirm that destroyed cells affect the colony that resumes growing.
4. **Score and return:** observe successive cell awards (`+1`, `+2`, `+4`,
   `+6`, ...). Confirm that crossing the phase boundary preserves the shot and
   a physical paddle contact resets the next award to `+1`.
5. **Miss and relaunch:** deliberately miss. Confirm that one life is removed,
   earned score is preserved, the shot chain resets, and the next ball waits in
   Ready until a deliberate launch.
6. **Pause and settings:** pause with `Space`, resume, then pause again with
   `Escape`. Check that all motion stops, restart asks for confirmation,
   sound/reduced-motion/high-contrast controls are keyboard reachable, and
   `Tab`/`Shift`+`Tab` plus `Enter`/`Space` operate the native controls. Resume
   explicitly. Deactivate and return to the window once; confirm the game waits
   paused without stuck keys or pointer capture.
7. **Game over and retry:** lose all three lives. Confirm the game-over summary
   shows score, best shot, destroyed cells, sector, and seed. Use `Enter` or
   Retry and confirm a new run starts with three lives and score zero. Restart
   the application and confirm settings and the current-ruleset best score
   survive. If the local profile is inspected, confirm the completed result is
   tagged `life-arcade-3`; older-version records must not affect the displayed
   best.

If a physics safety message appears, record the exact state and seed. The game
must pause and explain that the run needs restart; do not treat the safety pause
as ordinary balance.

## Ten-minute owner fun and balance checklist

Play at least ten minutes of active gameplay across one or more runs; paused
time does not count. For each item, record **pass**, **concern**, or **not
observed**, plus a short note.

- The goal and first action are understandable within the first minute without
  outside explanation.
- Mouse and keyboard aiming both feel responsive, deliberate, and consistent.
- The 70/30 arena and GROW/SHATTER phase change remain immediately legible.
- Cell destruction and the `+1`, `+2`, `+4`, `+6`, ... progression make a long
  shot satisfying, while paddle reset feels fair rather than arbitrary.
- The ball remains visible against the colony, trail, impacts, and any naturally
  reached 20/50/100 cosmetic tier. Mark an unreached tier **not observed**;
  controlled fixtures are not human play evidence.
- Misses feel attributable to the player's decision or timing. Life loss,
  score preservation, and relaunch are clear rather than punitive or surprising.
- Sparse or empty-colony periods recover without frustrating waits, confusing
  spawns, or a stranded ball.
- Sector pacing becomes more demanding without making paddle control or ball
  reading feel unfair.
- Pause, focus changes, settings, game over, and retry do not break the rhythm
  or lose trust in saved results.
- After ten minutes, the player would voluntarily start another run. Record why
  or why not; this is the decisive subjective replay signal.

The 20/50/100 effects are cosmetic by design. Any observed trajectory, scoring,
or collision change tied to an effect tier is a defect, not a balance opinion.

## Record the result

Capture these fields in the acceptance report:

```text
Date and tester:
Code revision:
Documentation revision:
ZIP SHA-256:
Windows/display/input:
Active play duration:
Journey defects:
Checklist observations:
Overall fun/balance verdict:
Would replay (yes/no and why):
Acceptance: accepted / changes requested / incomplete
Evidence attachments:
```

Do not mark acceptance as passed when required journey steps or the ten active
minutes were not completed. Subjective acceptance belongs to the human owner;
automated tests, static previews, seeded bots, and this document cannot grant it.
