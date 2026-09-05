# Life Arcade — adopted game-design baseline

Status: design adopted by the Lead under the owner's delegated design authority.
This document specifies the **next game**, not the behavior of the existing executable.

- Work: `WORK-legacy` — NumSharp Life Arcade
- Design job: `JOB-life-arcade-design-003`
- Baseline: `SPEC-life-arcade-003`
- Decision: `DEC-life-arcade-003`
- Supersedes the separate-game premise of `SPEC-numsharp-life-pong-001`.
- Existing implementation inspected: `journey3`, HEAD `2a7dbe74`.

## 1. The game in one sentence

Defend the right half with a paddle, send one ball into a living colony on the
left, and build a doubling score chain by shattering cells without missing.

The colony is the opponent. There is **one arena, one ball, one player paddle,
one score, and one shared rhythm**. There is no AI paddle and no separate Life
editor during a scored run.

### Owner requirements and Lead decisions

The owner established: player on the right; living Game of Life on the left;
Life runs and breathes while the ball is right; Life freezes while the ball is
left; the ball bounces off and destroys cells; destruction earns escalating
`1, 2, 4` rewards; a sparse colony receives new cells. The owner delegated the
remaining design to the Lead.

The following are **Lead design decisions**, not quotations of owner approval:
doubling per-cell awards; a chain that survives successful returns and resets on
a miss; a 256-point per-cell ceiling; three lives; population thresholds;
sector progression; boundary behavior; anti-stall assistance; presentation and
controls. Numeric values below are first-playtest tuning values, not evidence
of a proven difficulty balance.

## 2. The central rhythm

### GROW — ball in the right half

The colony advances through Conway B3/S23 generations and visibly breathes.
The player anticipates the returning ball and moves the right-hand paddle to
send it back. Breathing changes brightness and a subtle halo, **not collision
geometry**. Births and deaths interpolate visually between authoritative grid
states. Ordinary Life births, deaths and replenishment never award points.

### SHATTER — ball in the left half

The current colony freezes immediately. It becomes a solid destructible field.
The ball reflects from live cells, destroys contacted cells, emits a short
impact burst, and advances the score chain. Destroyed cells stay dead in that
snapshot; there are no generations or new spawns until GROW resumes.

### Crossing back

As the ball returns right, the damaged colony starts evolving from exactly the
surviving cells. Damage therefore changes the future opponent. Replenishment
can reinforce the survivors but does not replace the whole board.

The phase is determined by the ball's **center crossing the center line**, with
direction-aware crossing events. The center line is not a wall. A clear strip
between the colony and center prevents new cells from touching the ball as the
phase changes. The simulation splits a time step at a crossing: it never runs
a whole generation on the wrong side or accumulates a backlog while frozen.

## 3. Scoring and the reason to keep a rally alive

The awards for successive ball-destroyed cells are:

`+1, +2, +4, +8, +16, +32, +64, +128, +256, +256, ...`

These are **individual awards**, not cumulative totals. The first three hits
produce a total score of `7`. The HUD shows both total score and **Next cell +N**.

- The chain persists across center crossings and successful paddle returns.
- Missing the paddle resets the next award to `+1` and consumes one life.
- Previously earned score is never confiscated.
- Walls, paddle hits, Life deaths, births and spawns score zero.
- The per-cell ceiling keeps long rallies valuable without exponential overflow
  or unreadable numbers. Track total score in a saturating 64-bit integer.
- A cell can score once per live incarnation. A destroyed location that later
  becomes alive through Life or replenishment can legitimately score again.
- Simultaneous distinct cell contacts score separately in deterministic contact
  order; repeated overlap with a now-dead cell cannot duplicate a reward.

Resetting on every return right was considered and rejected: many honest shots
will hit one exposed cell and rebound immediately, making doubling almost
invisible. A rally-long chain rewards the player's sustained control instead.

## 4. The living opponent and replenishment

The colony uses a bounded 28-column by 32-row field: 896 cells. Conway's B3/S23
rules remain genuine; cells outside the field are dead, not wrapped to the
opposite edge. A bounded colony makes the arcade's physical boundary legible.

Begin with approximately 160 live cells in separated small clusters, gliders
and oscillators, with enough empty lanes for angled shots. Pattern placement
must fit the grid and never silently erase survivors.

When population drops below **64**, queue a replenishment burst:

1. During active GROW time, show a subtle 250 ms birth cue at available sites.
2. Recheck population. If it has recovered to at least 64 naturally, cancel.
3. Otherwise add cells to bring population to **160**, preserving survivors.
4. Use small separated growth-capable clusters first; fill remaining valid dead
   positions deterministically if complete clusters do not fit.
5. Enforce a 750 ms active-GROW cooldown after a burst.

The cue and cooldown stop advancing during SHATTER and pause. A crossing left
cancels the visible cue and leaves replenishment pending for the next GROW
interval. No birth materializes mid-attack. Placement uses bounded attempts and
a finite fallback scan, so a crowded field cannot cause a retry loop.

An empty colony is not a victory or a dead end. The left back wall returns the
ball; new life arrives in the next GROW interval. Spawn events award no points.

## 5. Physics: physical contacts, arcade pacing

Use one logical 1600 by 900 arena split equally at x=800. The cell field occupies
a padded area in the left half, with a clear approach strip of at least two
ball diameters before the center. Top, bottom and left boundaries reflect. The
right edge behind the player is the only loss boundary.

- Fixed 120 Hz simulation, independent of render rate; collision stepping must
  account for the smallest cell and moving paddle, not just ball travel.
- Ball radius: 10 logical units. Player paddle: 18 by 144, near the right edge.
- Rounded/circle-versus-rectangle contact normals determine reflection.
- Paddle acceleration and inertia are retained. Its motion transfers tangential
  velocity to the ball so a well-timed moving strike changes the next approach.
- Initial ball speed: 640 logical units/second; ceiling: 1000.
- On each distinct paddle or cell collision, add a uniformly sampled signed
  perpendicular perturbation bounded to **2% of speed**, then renormalize.
  Do not add jitter every frame or on ordinary wall reflections.
- Jitter may not point the ball back inside a resolved contact; deterministic
  outward correction and separation take priority, without drawing another
  random sample. Stationary contacts conserve the selected speed.
- Apply a minimum absolute horizontal component of 30% of speed after rebounds
  to avoid nearly vertical play. This is explicit arcade assistance, not a
  claim of perfectly Newtonian motion. Preserve direction except on the
  return-assistance rule below.
- Break ties at shared cell edges deterministically, remove contacted cells
  atomically, and resolve the contact set once. Never bounce twice solely
  because the ball still overlaps its previous contact.

If six seconds of active SHATTER time pass without a cell hit, show **RETURN**.
At the next wall contact, guide the rebound rightward with at least 45% of speed
in the horizontal component. This prevents watching a useless orbit; it does
not teleport, grant score, or move the paddle. Pause does not age this timer.

## 6. Run structure and difficulty

**Classic is an endless high-score run with three lives.** There is no seven-point
match and no requirement to eradicate a board that is designed to regenerate.

- Ready: a ball sits just in front of the right paddle. The player can position
  the paddle before pressing Space or Start to launch toward the colony.
- Play: GROW and SHATTER alternate according to the ball, not a fixed timer.
- Miss: retain score and colony state; subtract a life and reset the chain.
  Briefly show the lost-life result, then attach the new ball to the paddle.
  The next launch is deliberate, not a surprise while the player looks away.
- Game over: after the third miss, show score, local best, best chain, destroyed
  cells, and highest sector. Offer an immediate retry and a return to title.

Advance a **sector every 40 ball-destroyed cells**. Life deaths do not count.
Each sector increases target speed by 6%, capped at 1000, and Life rate by
0.5 generations/second from a base of 6, capped at 10. Keep paddle size stable;
the challenge comes from tempo and the evolving opponent, not shrinking input
tolerance. Apply a pending sector change on the next paddle return or serve,
never as an unexplained velocity jump inside the colony.

On a lost life, launch at 85% of the current sector speed, with a floor of 640,
and restore sector speed over the next three successful returns. A new run
resets difficulty. No automatic extra lives in the first version.

Randomness is seeded per run. Store the seed and gameplay version with local
results for reproducible debugging. Save a small local high-score list and
settings; no account, network leaderboard, monetization, or telemetry.

## 7. Look, sound and input

The arena dominates the window. A thin HUD contains score, next-cell award,
three life indicators and sector progress. The left is a coherent organism,
not a spreadsheet panel; the right has open space to read the incoming ball.
No editor toolbar, simulation statistics dashboard, AI score or stock game
widgets interrupt a run.

Visual direction: an ink-dark field, mint living cells, warm amber/coral impact
states, an unmistakable bright ball, and a slim player paddle. Use our own
vector drawing, disciplined contrast, subtle depth and restrained particles.
The phase has both a word and a visual change: **GROW / SHATTER**, not color alone.

- Breathing is a slow low-amplitude pulse. Growth fades in; frozen cells sharpen
  into a crisp stable target. Never pulse or resize the authoritative collider.
- Show small `+N` labels near impacts and a clear next-award change in the HUD.
- A short ball trail communicates direction without concealing the ball.
- Use brief pitched impact sounds, rising along the combo ladder, and distinct
  paddle, miss and replenishment cues. Cap pitch escalation with the award cap.
  Audio is optional, locally generated/bundled, and independently mutable.
- Screen movement is cosmetic and restrained. Reduced-motion disables shaking,
  breathing pulses and excessive particles; it preserves clear static phase cues.
  No full-screen flashing or mandatory hit-stop in the first version.

Desktop controls: W/S or Up/Down, or pointer targeting for the right paddle;
Space to launch/pause/resume; Escape to pause/open the menu. An active run must
not restart accidentally from one stray R key: Restart belongs in the pause
menu with confirmation. Game-over Retry can use Enter. Track key aliases
independently and suppress repeat on one-shot actions.

Use styled accessible controls for menus and custom drawing for the arena.
All menu actions support keyboard focus, activation and meaningful accessible
names. Master sound, reduced motion and high-contrast options are reachable
without leaving a run. Focus loss pauses **the entire game**, releases held
keys/capture and requires an explicit resume. Detachment/disposal stops clocks.

Windows desktop remains first. Preserve a shared C# simulation/input contract
and renderer boundary for later landscape mobile: relative dragging in the
right half, offset so a thumb need not cover the ball or paddle. This baseline
does not claim delivery or validation of a mobile binary.

## 8. Scope and implementation boundary

Keep the new game in `examples/LifeAndPong` on `journey3`, using the local
NumSharp project. Reuse the owned byte-typed NDArray double buffers, fixed-step
timing, input lifecycle work, rendering foundation and Windows release path.

Replace the independent `PongSimulation`/Life clocks with one authoritative
arcade session that owns phase, colony, ball, right-hand paddle, score, lives,
replenishment and progression. A single snapshot feeds drawing; effects and
audio consume events and do not mutate gameplay. Separate seeded gameplay RNG
from cosmetic RNG so changing particles cannot change a run.

Remove AI and first-to-seven assumptions from active play. Keep any debug Life
editor out of a scored run. Do not change NumSharp Core APIs for this game.

No bosses, multiball, weapons, shop, upgrades or extra game modes in the first
version. They can be evaluated after the basic rhythm is genuinely enjoyable.
Do not carry old code/documentation QA approval forward to the new game.

## 9. Acceptance and balancing plan

The replacement implementation must prove:

1. Conway steps occur only during active GROW; transitions, pauses and long
   frames cannot advance the frozen board or replay missed generations.
2. Destroyed cells mutate the next growing colony, and natural deaths score zero.
3. First hit awards 1, second 2, third 4; total is 7; crossing/paddle hits preserve
   the chain; a miss resets it; the 256 ceiling and total-score overflow are safe.
4. Duplicate contacts cannot re-score cells. Shared corners, fast travel and
   moving-paddle end contacts have deterministic, non-tunneling outcomes.
5. Low population queues a finite burst, preserves live cells, cancels if the
   colony recovers, and never spawns during SHATTER or pause.
6. Empty fields, stalled paths, simultaneous crossing/collision events and a
   miss during pending progression cannot strand the run.
7. Speed limits, horizontal assistance and 2% jitter are independently tested
   over deterministic seeds; cosmetic changes leave gameplay snapshots intact.
8. All three lives, serves, pause/resume, game-over and retry work through real
   input routes, including lost focus, aliased keys and pointer capture loss.
9. UI at default and minimum supported desktop sizes remains readable, including
   long scores, high-contrast and reduced-motion states.
10. A fresh self-contained Windows ZIP contains the replacement game and matching
    instructions, with independent code QA and documentation QA approvals.

Playtest the first minute for clarity without explanation, then a longer run
for chain satisfaction, meaningful aiming, fair misses and time spent waiting
on an empty board. Adjust the documented tuning values from that evidence;
do not call the arcade balanced merely because automated tests pass.

## 10. Delivery state

This turn adopts the game design and supplies a scripted visual study of the
shared-arena rhythm. The study illustrates three separate hits awarding 1, 2
and 4 across successive returns; it is **not the C#/NumSharp game or a physics
acceptance test**. Its field and pacing are scaled for illustration.

The existing Windows executable still implements the rejected separate-game
interpretation. Rebuilding it to this baseline, playtesting the balance, and
obtaining fresh code/documentation gates remain implementation work.
