# Life Arcade — adopted game-design baseline

Status: adopted baseline, amended by owner steering `DEC-life-arcade-steering-005`.
Implementation and fresh QA are in progress; historical packages are not evidence of this revision.

- Work: `WORK-legacy` — NumSharp Life Arcade
- Design job: `JOB-life-arcade-design-003`
- Baseline: `SPEC-life-arcade-003`
- Decision: `DEC-life-arcade-003`
- Supersedes the separate-game premise of `SPEC-numsharp-life-pong-001`.
- Existing implementation inspected: `journey3`, HEAD `2a7dbe74`.

## 1. The game in one sentence

Defend the right 30% with a paddle, send one ball into a living colony covering
the left 70%, and build an escalating score within each shot into the colony.

The colony is the opponent. There is **one arena, one ball, one player paddle,
one score, and one shared rhythm**. There is no AI paddle and no separate Life
editor during a scored run.

### Owner requirements and Lead decisions

The owner established: player on the right; living Game of Life on the left;
Life runs and breathes while the ball is right; Life freezes while the ball is
left; the ball bounces off and destroys cells; destruction earns escalating
`1, 2, 4` rewards; a sparse colony receives new cells. The owner delegated the
remaining design to the Lead.

The owner subsequently directed a 70/30 split, a per-paddle-contact reset,
the award sequence `1, 2, n+2`, a text-free playfield, and escalating effects at
20, 50 and 100 cells destroyed between paddle contacts. Interpret the explicit
sequence as `+1, +2, +4, +6, +8, ...`, not powers of two. These requirements
supersede the former rally-long doubling chain and 50/50 layout.

Three lives, population thresholds, sector progression, boundary behavior,
anti-stall assistance, effect art direction and controls remain Lead design
decisions. Numeric tuning values are not evidence of a proven difficulty balance.

## 2. The central rhythm

### GROW — ball in the right 30%

The colony advances through Conway B3/S23 generations and visibly breathes.
The player anticipates the returning ball and moves the right-hand paddle to
send it back. Breathing changes brightness and a subtle halo, **not collision
geometry**. Ordinary Life births, deaths and replenishment never award points.

### SHATTER — ball in the left 70%

The current colony freezes immediately. It becomes a solid destructible field.
The ball reflects from live cells, destroys contacted cells, emits a short
impact burst, and advances the score chain. Destroyed cells stay dead in that
snapshot; there are no generations or new spawns until GROW resumes.

### Crossing back

As the ball returns right, the damaged colony starts evolving from exactly the
surviving cells. Damage therefore changes the future opponent. Replenishment
can reinforce the survivors but does not replace the whole board.

The phase is determined by the ball's **center crossing the 70% boundary**, with
direction-aware crossing events. This boundary is not a wall. A clear strip
between the colony and boundary prevents new cells from touching the ball as the
phase changes. The simulation splits a time step at a crossing: it never runs
a whole generation on the wrong side or accumulates a backlog while frozen.

## 3. Scoring and cells per shot

The awards for successive ball-destroyed cells are:

`+1, +2, +4, +6, +8, +10, ...`

These are **individual awards**, not cumulative totals. The first three hits
produce a total score of `7`. The HUD shows both total score and **Next cell +N**.

- A physical paddle-ball collision resets the next award to `+1`, the shot's
  destroyed-cell count to zero, and its active effect tier to baseline.
- Crossing the 70% boundary by itself does not reset the counter.
- Missing the paddle resets the next award to `+1` and consumes one life.
- Previously earned score is never confiscated.
- Walls, paddle hits, Life deaths, births and spawns score zero.
- There is no 256-point gameplay ceiling. The next award adds two after the
  initial one-to-two step. Saturating 32-bit awards and 64-bit total score
  prevent integer wraparound in extreme runs.
- A cell can score once per live incarnation. A destroyed location that later
  becomes alive through Life or replenishment can legitimately score again.
- Simultaneous distinct cell contacts score separately in deterministic contact
  order; repeated overlap with a now-dead cell cannot duplicate a reward.

Twenty cell hits in one shot total 381 points; 50 total 2451; 100 total 9901.
The milestone-hit awards are respectively +38, +98 and +198. Store best shot
length separately from total cells destroyed. Keep old-version local records,
but do not compare their scores with this changed ruleset.

## 4. The living opponent and replenishment

The colony uses a bounded 42-column by 32-row field: 1344 cells. Conway's B3/S23
rules remain genuine; cells outside the field are dead, not wrapped to the
opposite edge. A bounded colony makes the arcade's physical boundary legible.

Begin with approximately 160 live cells in separated small clusters, gliders
and oscillators, with enough empty lanes for angled shots. Pattern placement
must fit the grid and never silently erase survivors.

When population drops below **64**, queue a replenishment burst:

1. During active GROW time, show a subtle 250 ms birth cue in the external HUD.
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

Use one logical 1600 by 900 arena split 70/30 at x=1120. The cell field occupies
a padded area in the left 70%, with a clear approach strip of at least two
ball diameters before the boundary. Top, bottom and left boundaries reflect. The
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
The phase has both a word in the external HUD and a visual change: **GROW / SHATTER**,
not color alone. All text is outside the playfield in every state, including
score feedback, phase labels, population counts, sector notices and menus.
Start, pause, confirmation and game-over controls occupy a dock beside the
arena rather than overlaying it. During play the dock collapses.

- Breathing is a slow low-amplitude pulse; frozen cells sharpen
  into a crisp stable target. Never pulse or resize the authoritative collider.
- Show the latest `+N`, shot count and next award in the external HUD, never
  floating over cells or the ball.
- A short ball trail communicates direction without concealing the ball.
- Use brief pitched impact sounds, rising along the combo ladder, and distinct
  paddle, miss and replenishment cues. Cap pitch escalation independently of score.
  Audio is optional, locally generated/bundled, and independently mutable.
- Screen movement is cosmetic and restrained. Reduced-motion disables shaking,
  breathing pulses and excessive particles; it preserves clear static phase cues.
  No full-screen flashing or mandatory hit-stop in the first version.

### Shot milestones: 20 / 50 / 100

Milestones count ball-destroyed cells since the last paddle hit, not lifetime
score, natural Life deaths, or number of paddle returns. Each fires once per shot:

- **20 — Surge:** golden impact shards, a brighter trail, one expanding ring,
  corner energy marks and a short ascending three-note cue.
- **50 — Overdrive:** violet/pink streak particles, dual-color trail, two
  expanding rings, stronger corner marks and a four-note arpeggio.
- **100 — Supernova:** prismatic shards, a three-color trail, three layered
  shockwaves, full tier corner marks and a five-note octave-spanning cue.

These tiers change neither ball trajectory nor collision rules. Draw the white
ball last; bound particles and rings; avoid full-screen fills/flashes. The HUD
names the tier outside the arena. Reduced-motion uses static tier colors/corner
marks and HUD feedback without trails, particles or expanding rings. A paddle
contact or miss immediately clears the active tier and residual attack effects.

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
3. Awards follow 1,2,4,6,8,...; first three total 7. Crossings preserve the
   counter; each paddle contact and each miss reset it. Awards and score cannot overflow.
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
9. UI at default and minimum supported desktop sizes remains readable, with all
   text outside the arena, including docked menus and milestone feedback.
   Verify exact 70/30 geometry, extended-cell collisions, 20/50/100 event
   thresholds, style resets, high-contrast and reduced-motion states.
10. A fresh self-contained Windows ZIP contains the replacement game and matching
    instructions, with independent code QA and documentation QA approvals.

Playtest the first minute for clarity without explanation, then a longer run
for chain satisfaction, meaningful aiming, fair misses and time spent waiting
on an empty board. Adjust the documented tuning values from that evidence;
do not call the arcade balanced merely because automated tests pass.

## 10. Delivery state

The initial scripted visual study and the e6b99006 executable predate the latest
owner steering. They are not current scoring/layout acceptance evidence. The
revised C# build must pass fresh code and documentation gates. Milestone image
fixtures exercise real event/rendering paths but do not claim human-earned
20/50/100-hit records. Human difficulty balance remains subject to playtesting.
