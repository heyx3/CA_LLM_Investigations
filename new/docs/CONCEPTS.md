# Concepts and vocabulary

The terms and maths used throughout `ca3d`, in one place. Each section says where the
idea lives in the code. If you know roughly what a cellular automaton is, you can read
this top to bottom in ten minutes.

**Suggested code reading order:** `rulesets/rules.py` → `wolfram.py` → `life.py` →
`hierarchy.py` → `cityscape.py`. Then `analysis/metrics.py` and `analysis/search.py` if
you want to measure or search, or `render3D/` if you want pictures.

## 1. Cellular automata

A **cellular automaton (CA)** is a grid of **cells**, each holding a **state**; here
always 0 (empty, "dead") or 1 (filled, "alive"). All cells update **at once** (a
**synchronous** update) by the same **rule**, which computes a cell's next state from
its **neighbourhood**: the cell itself and the cells around it. The grid's edges wrap
around (a ring in 1D, a torus in 2D) unless a function says otherwise.

| dimension | neighbourhood used | code |
| --- | --- | --- |
| 1D | a **window** of 2r+1 cells (radius r) | `rulesets/wolfram.py` |
| 2D | the **Moore neighbourhood**: the 3×3 block around the cell | `rulesets/life.py` |
| 3D | the 3×3×3 box (26 neighbours), or larger boxes | `analysis/dynamics.py`, `rulesets/deposition.py` |

## 2. Rules as lookup tables

A **rule table** is a 1D array whose entry *i* is the new state for neighbourhood
pattern *i*. The pattern is read as a binary number, so a table has 2^(cells in the
neighbourhood) entries. Everything else (rule numbers, B/S strings) is a way of writing
a table down.

**1D ("Wolfram") rules.** With radius 1 the window (left, centre, right) = (1, 1, 0) is
the number 0b110 = 6, so the table has 8 entries. "Rule 30" is the table whose bits,
read from entry 0 upwards, are the binary digits of 30 (`'lsb'` order). Some sources
write the table the other way round (`'msb'`), so a rule number is only meaningful with
its bit order. See `wolfram.rule_table`.

**Life-like ("B/S") rules, outer-totalistic.** The new state depends only on the
cell's own state and on *how many* of its 8 neighbours are alive (not which ones).
`B3/S23` (Conway's Game of Life) reads: a dead cell is **B**orn if it has 3 live
neighbours; a live cell **S**urvives if it has 2 or 3; otherwise the cell is dead next
step. The table has 18 entries, index = own × 9 + count. These rules treat all
directions alike (they are **isotropic**). See `life.parse_bs`.

**Non-totalistic ("Moore") rules.** The new state depends on the exact arrangement of
the 3×3 block. The table has 512 entries, index = own × 256 + neighbour_byte, where each
bit of the byte is one neighbour (the bit layout is drawn in `life.py`). These rules can
be **anisotropic**: they may prefer some directions. Any life-like rule **expands**
losslessly into a 512-entry table (`life.expand_to_moore`); **flipping** a few entries of
the expansion ("expand and perturb") breaks the symmetry while keeping the rule's
character. This is how the cityscape's rules were made.

**Rotating a rule.** Permuting the neighbour bits gives the rule that behaves like the
original on a turned lattice (`life.moore_permutation`). Rotating all rules by 90°
mid-run changes the direction in which structures grow (see §5).

## 3. Langton's lambda and mutations

**Lambda** (λ, `rules.langton_lambda`) is the fraction of table entries that output 1.
Table all zeros kills everything; all ones fills everything; interesting behaviour sits
between. It is a rough dial, not a classifier.

Mutation operators in `rulesets/rules.py`:

* `flip_bits`: invert k entries. Symmetric, so it leaves λ (and density) unchanged on
  average.
* `assign_bits`: overwrite k entries with fresh bits that are 1 with probability p.
  With small p this lowers λ.
* `set_lambda`, `clamp_lambda`: nudge a table to a chosen λ with the fewest changes.

## 4. Space-time volumes

Stack the successive states of a 2D CA along a third axis (time) and the history is a
3D **space-time volume**; stack the rows of a 1D CA and you get a 2D **space-time
diagram**. Structures that stay put become vertical **pillars**; a pattern that never
changes extrudes into a **prism**. Several constructions in `rulesets/` build volumes
from 1D CAs (`double_spacetime`, `triplanar`), from Life without Death (`lwd`), or
directly from 3D rules (`deposition`, `lattice_gas`). Volumes are boolean arrays;
`render3D/` draws them. Axis conventions are in the `ca3d/__init__.py` docstring.

## 5. The hierarchical CA (`rulesets/hierarchy.py`)

A **hierarchical** (or "multi-octave") CA is a stack of binary lattices at resolutions
1, 2, 4, 8 (the **scale**; finest first). The finest lattice (layer 0) is the one drawn.
**Coarser layers supply context to finer ones.**

* A cell reads its own neighbourhood pattern (say 512 possibilities) *and* one bit from
  each **parent** layer, taken from the parent cell that covers it.
* The parent bits, packed into an integer, are the **context**. The nearest parent is
  the most significant bit.
* A layer with P parents owns 2^P **rule banks** (`rules[context, pattern]`), and a cell
  uses the bank its context selects. Example: layer 0 with 3 parents has 8 contexts, so
  8 rule tables; a fine cell whose parents read (1, 0, 1) uses bank 0b101 = 5.
* **Wiring**: with `'all'` wiring every layer reads all coarser layers; with `'chain'`
  only the next one.
* **Schedule**: layer *i* updates on steps where `t % period == phase`. Coarse layers
  have larger periods, so they change slowly. Staggered phases (phase = log₂ period)
  stop two layers updating on the same step, which avoids horizontal banding.
* **Rotation**: a rotation policy (`RotateEvery`, `RotateOnDensityLadder`) turns every
  rule bank by 90° partway through. `RotateOnDensityLadder` turns each time the
  coarsest layer's density has climbed another `delta`, like rungs of a ladder.

**Rule families** (`rulesets/families.py`) classify rules by behaviour rather than by
table. The cityscape gives each of the fine layer's 8 contexts a rule from one family
(the **context plan**: 2 dead, 4 static, 2 complex):

| family | behaviour | what it does in the volume |
| --- | --- | --- |
| dead | density → 0 from a dense start | empty sky, carves voids |
| static | freezes at mid density | pillars |
| complex | partial damage spreading, sparse | ragged texture, horizontal streaks |
| slow | compact and slowly changing | used by the coarse layers |
| edge | partial damage spreading, mid density | edge-of-chaos rules (older configurations) |

## 6. Running and testing rules (`analysis/dynamics.py`)

* A **soup** is a random initial state: each cell alive with probability p0. A **dense
  start** is p0 = 0.85; rules that erode or freeze material only show themselves from a
  dense start.
* **Burn-in** is the initial stretch of steps ignored when averaging the **change
  rate**, so that start-up transients do not count.
* An **assay** fixes how every rule in a search is tested: lattice size, soup, run
  length, burn-in. All rules see the same soup.
* **Damage spreading**: run two copies of a lattice that differ in one cell and measure
  the fraction of cells that end up different. About 0 means ordered (the difference
  dies out); about 0.5 means chaotic (the copies become unrelated); in between
  (here 0.02 to 0.25) is the **complex** band, where rules are neither frozen nor
  noise. In 1D the difference can only spread inside the light cone, so give 1D rules
  enough steps.
* A **rule space** (`Wolfram`, `Totalistic`, `Moore`) knows how to step a whole batch of
  lattices, each under its own rule, in one numpy call.

## 7. Measures (`analysis/metrics.py`)

Every measure takes a binary array (a 2D grid, a 3D volume, or a batch of them) and
returns a number. `metrics.describe_measures()` lists them with what values read as.

| measure | definition | reads |
| --- | --- | --- |
| density | fraction of live cells | 3D: under 0.05 nothing to see; over ~0.35 opaque |
| entropy | Shannon entropy of the state distribution, bits | 0 uniform, 1 half full |
| block entropy | entropy of overlapping blocks (2×2 or 2×2×2), per cell | sees arrangement: a checkerboard has entropy 1 but block entropy 0.25 |
| gzip ratio | compressed size / raw size, relative to the same cells shuffled | ~0 ordered, 1 noise |
| compactness | fraction of cells whose whole 3^d neighbourhood equals their own state | 2D: <0.2 speckle, 0.4–0.7 regions with detail, >0.9 blobs |
| coherence | mean live face-neighbours of a live cell ÷ (2d × density) | 1 = random noise; >2 clustered |
| components | connected blobs of live cells (face-connected): count, median size, largest share | largest share >0.9: percolating |
| void | largest connected empty region ÷ array size | high = mostly open space |
| correlation length | smallest lag where autocorrelation falls below 1/e | 1 = decorrelates after one cell; per axis, so differences show anisotropy |
| anisotropy | spread of the autocorrelation at lag 3 across axes | zero on average for isotropic rules (a single state shows a little noise) |
| fractal dimension | slope of log(occupied boxes) vs log(1/box size) | 3D: 3 filling, ~2.5 aggregate, lower wispier |
| change rate | fraction of cells flipping per step | <0.001 frozen; 0.001–0.05 slow; >0.4 churn |
| pillar fraction | share of live cells in runs of ≥16 unchanged steps | ~0 nothing persists; ~0.7 prisms |
| overhang fraction | live cells with an empty cell below, along time | exactly 0 for a heightfield |
| streaks | live cells in horizontal runs that are short in time | texture from fast-changing rules |
| thinness | share of live cells that are long **and** thin | ladders ~0.7, blobs ~0.1 |

*Autocorrelation* at lag k is how strongly an array resembles itself shifted by k cells
(1 identical, 0 unrelated). *Enrichment* of a motif in a label (say a context) is the
share of motif cells with that label divided by the share of all cells with it; above 1
the motif concentrates there.

**The principle behind them:** interesting output lives in a **band** of a measure,
never at an extreme, and should be judged on two independent measures (compact *and*
slow; partial damage *and* mid density).

## 8. Searching (`analysis/search.py`)

* A **band** is an interval of a measure's values; a **criterion** is a measure plus a
  band; a **stage** applies criteria, in order, under one assay; a **pipeline** chains
  stages, so cheap measures cut the candidate list before expensive ones run.
* A **target** is a profile of measured values (say the cityscape's density, pillars,
  void, streaks); `Table.rank` sorts results by distance to it. Targeting a profile
  works better than maximising anything.
* `search.sweep` builds and measures every combination of parameters and seeds. Run
  several seeds: the spread between rule draws can be as large as the effect studied.

## 9. Glossary

| term | meaning |
| --- | --- |
| absorbing state | a state the dynamics never leave (an all-0 or all-1 row in a 1D CA) |
| ablation | removing one ingredient to see what it contributed |
| assay | the fixed test conditions for a batch of rules |
| bank (rule bank) | one rule table, selected by a cell's context |
| burn-in | initial steps ignored before measuring |
| context | the packed parent bits that select a rule bank |
| DDA | digital differential analyser: the voxel-by-voxel ray walk in `render3D/render.py` |
| heightfield | a surface with one height per (x, y), so no overhangs |
| lambda | fraction of table entries that output 1 |
| LWD | Life without Death: Life's B3 birth rule with every live cell surviving |
| ladder | (LWD) a long thin growth finger; (hierarchy) a rung of the density trigger for rotations |
| octave | one resolution level of a blend |
| outer-totalistic | depends on own state and the number of live neighbours |
| soup | random initial state |
| space-time | a CA's history, with time as an extra axis |
| voxel | a cell of the 3D grid |
