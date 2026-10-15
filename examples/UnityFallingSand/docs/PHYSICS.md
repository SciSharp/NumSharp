# The Applied Math Behind the Falling-Sand Game

This is a **cellular automaton** (CA): a grid of cells, each holding one material, updated by local
rules. What makes it a good NumSharp showcase — and a genuine bit of applied math — is that the update
is **fully vectorized** (whole-grid array operations, no per-cell loop) while still **conserving mass
exactly**, which the obvious vectorized approach silently gets wrong. Everything here is checked by the
`Verification/` harness.

The grid is an `(H, W)` int array; each cell is a [`Cell`](../UnityProject/Assets/Scripts/Simulation/Cell.cs)
id. Row 0 is the top, so "down" is increasing row.

---

## 1. One master rule: density

Every material has a **density**. The single rule "a cell sinks past a strictly lighter cell below it"
(Archimedes' principle) produces almost all the behaviour:

```
Smoke(-1) < Empty(0) < Oil(1) < Water(2) < Sand(3)          Wall = immovable
```

- **Sand** (heaviest) sinks to the bottom.
- **Water** and **oil** are liquids; oil is lighter, so **oil floats on water**.
- **Smoke** is lighter than air. Its density is **negative on purpose**: the empty cell above smoke is
  *denser* than the smoke, so the ordinary downward swap moves the empty down and the smoke up — "smoke
  rises" needs no separate rule.
- **Sand sinks through water**, because sand is denser than water: the same downward swap applies to a
  sand-over-water pair, not just sand-over-empty.

Drop a jumble of all four into a container and it settles into ordered layers — verified:

```
mean rows  smoke=3.1  oil=17.9  water=27.5  sand=38.2      (smaller row = higher up)
```

---

## 2. The vectorized swap, and why it conserves mass

Each rule selects a set of cells that want to move one step in a direction `(dr, dc)` and swaps each with
its neighbour. Doing that for the whole grid at once is the crux. The naive version — "for every cell
with empty below, copy it down" — **duplicates or deletes material** when two cells target the same
destination, or when a falling column all move at once.

The engine avoids that with a conflict-free swap
([`GridOps.ApplySwap`](../UnityProject/Assets/Scripts/Simulation/GridOps.cs)):

```
allowed[r,c] = move[r,c] AND NOT move[r+dr, c+dc]      # a mover may fire only if its target isn't also moving
```

**Why this is exactly mass-conserving.** Within one direction, each target cell `(r+dr, c+dc)` has exactly
one possible source (the cell at `-(dr,dc)`), so two movers can never collide on a target. The only other
hazard is a *chain* — a cell that is both a mover and someone's target. The guard removes it: if cell `X`
is an allowed mover then its neighbour is not moving, so the cell behind `X` (which targets `X`) is
suppressed. One can show each firing pair is therefore disjoint, so the whole update is a **permutation of
the grid** — every material count is invariant. The harness confirms it over 300 steps of a maximally busy
random grid:

```
Empty:676→676  Wall:317→317  Sand:343→343  Water:328→328  Oil:350→350  Smoke:290→290
```

A chain (a tall column of sand over a hole) therefore advances one cell per step — exactly how gravity
should propagate in a CA: the bottom grain falls first, the next follows next step.

The neighbour lookups themselves (`the cell below`, `the cell to the left`, a diagonal) are whole-grid
**shifts**, built with NumSharp slice assignment; off-grid neighbours are filled with a wall, so the four
edges behave as solid boundaries with no special-casing.

---

## 3. Granular vs fluid: the same swap, different masks

The three update passes differ only in which cells they select:

1. **Vertical density (everything).** `dens(cell) > dens(below)` → swap down. Falling, sinking, floating,
   rising, all at once.

2. **Diagonal slide (SAND only).** A grain that can't fall straight down slides to a lower diagonal cell.
   This is what makes sand **pile at ~45°** (an angle of repose) instead of leveling. Liquids do NOT do
   this.

3. **Horizontal spread (FLUIDS only).** A fluid that can't move vertically spills into an adjacent EMPTY
   cell. This is what makes liquids **level flat** and gases spread under ceilings.

The split *is* the difference between a powder and a liquid: sand slides and piles; water spreads and
levels.

---

## 4. Why fluid flow must be RANDOM (the subtle bug)

The horizontal spread has one non-obvious requirement. If a fluid cell always prefers the same side, a
lone surface cell — empty on both sides, unable to descend because the row below is full — moves one way,
then the deterministic rule moves it back, and it **oscillates in place forever**. The surface never
levels; water freezes into a stable sand-like ramp. (This sample hit exactly that during development.)

The fix is to give each fluid cell a **random flow direction each step**. A lone cell then performs a
random walk along the surface, finds a drop, and settles. That is what actually makes water level:

```
a 12-tall water column, released in a basin, settles to occupy just 2 rows (its flat depth)
```

The randomness uses NumSharp's seeded `np.random`, so a given seed replays bit-for-bit — verified by
running the same seed twice and comparing the whole grid.

---

## 5. Emitters (faucets) and boundaries

- **Emitters** drip a material into a fixed cell each step (only when it is empty, so a stream forms).
  They are the one thing that *adds* mass — a faucet you place and leave running.
- **Boundary walls** (left/right/bottom) keep loose material in the basin; with them and no emitter, the
  loose-material count is invariant (verified).

---

## Where each piece lives

| Concept | File |
|---|---|
| Materials, densities, colours (§1) | `Assets/Scripts/Simulation/Cell.cs` |
| Shifts + the conflict-free swap (§2) | `Assets/Scripts/Simulation/GridOps.cs` |
| The three update passes (§3, §4) | `Assets/Scripts/Simulation/PowderGrid.cs` |
| Brush, faucets, boundaries (§5) | `Assets/Scripts/Simulation/FallingSandWorld.cs` |
| The proofs | `Verification/Program.cs` |
