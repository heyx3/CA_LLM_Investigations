# Hand-coding Moore rules

How a Moore rule is written by hand for a desired effect, as the hierarchy kinds in
`ca3d/studies/hierarchy_kinds.py` do (`built_table` and the functions it calls). The
pool families (`studies/moore_families.py`) are selected on random soups; a hand-built
rule instead does one known thing to a territory at every update, which is what a
coarse layer that shapes a scene usually needs.

## 1. The table

A cell's next state depends on 9 bits: its own state and its 8 neighbours. A rule is
therefore a lookup table of 2⁹ = 512 entries, each 0 or 1, indexed by
`own * 256 + byte`, where `byte` packs the neighbours as in `rulesets/life.py`:

```
7 6 5      NW N NE
4 . 3  =   W  .  E
2 1 0      SW S SE
```

So N is bit 6, S bit 1, W bit 4 and E bit 3. Entries 0-255 say what a dead cell
becomes; entries 256-511 what a live cell becomes.

## 2. Writing a rule

Decide what one cell has to see to choose its next state, then write that test over
all 256 neighbourhoods at once, separately for the dead half and the live half:

```python
byte = np.arange(256)        # every neighbourhood
idx = np.arange(512)         # every (own, neighbourhood)
```

### Rules that ignore the neighbours

| rule | test | table |
| --- | --- | --- |
| hold | keep my state | `idx >= 256` |
| fill / clear | always 1 / always 0 | `np.ones(512)` / `np.zeros(512)` |
| toggle | flip my state | `idx < 256` |

hold keeps a footprint or a zone as it started; fill and clear give solid or void with
no texture; toggle makes a layer into a clock whose period sets how long each of its
two states lasts.

### Masks

The geometric rules test a mask, the set of neighbours that matter:

```python
N, S, W, E = 1 << 6, 1 << 1, 1 << 4, 1 << 3
SQUARE = 0xFF                # all 8
DIAMOND = N | S | W | E      # the 4 orthogonal
```

**Shrink: peel one cell of border per update.** A cell on a territory's edge has a
dead neighbour on the outside; an interior cell has none. So "a live cell survives
only if every neighbour in the mask is alive" kills exactly the edge, and nothing is
born:

```python
alive = (byte & MASK) == MASK
table = np.concatenate([np.zeros(256), alive])
```

The mask decides which edges go. Square kills a cell that touches the outside even at
a corner; diamond only one that touches it along a side. A single bit (side-N: `N`)
erodes only the edge facing that way, so the territory retreats in one direction and
whatever is built on it leans; two orthogonal bits (corner-NW: `N | W`) give a
diagonal lean.

**Spread: gain one cell of border per update.** The mirror of shrink: a dead cell is
born if any neighbour in the mask is alive, and nothing dies:

```python
born = (byte & MASK) != 0
table = np.concatenate([born, np.ones(256)])
```

From a seed, square spread grows a square, diamond a diamond, a single bit grows in
one direction only.

**Move: shift everything one cell.** Every cell copies one neighbour. To move east,
copy the west neighbour (bit 4):

```python
table = (idx >> 4) & 1
```

Every pattern translates rigidly and is otherwise unchanged. Pool families that
"drift" reshape their patterns as they go; a move table never does.

**Counts.** Instead of a mask, test how many neighbours are alive, e.g.
`popcount(byte) >= 6` for shrink-count6. A cell on a straight edge has 5 live
neighbours and dies, a cell in a concave corner has 7 and survives, so steps come out
irregular. spread-count2 (born with at least 2 live neighbours) is the spread
counterpart.

**Round fronts.** Because a count rule treats flat edges, convex corners and concave
corners differently, the right threshold rounds a front instead of squaring it:

* spread-count3 (born with at least 3 live neighbours): beside a flat edge a cell
  sees 3 and is born, beside a convex corner it sees 1 and waits, so a seed grows
  into a near-round disc. Seeds must be at least 3 wide.
* shrink-count6 (survive with at least 6): flat-edge cells (5) die and convex
  corners go faster, so a territory shrinks as a rounded octagon.
* shrink-count5 only rounds corners: flat-edge cells keep exactly 5 and survive, so
  it stalls. It smooths rather than shrinks.

Measured on discs, count3 and count6 move a border about 0.6 cells per update, not
1, which the periods must allow for (`ROUND_SPEED` in the organic kind).

## 3. From a rule to a shape

A rule can move a border at most one cell per update. Everything larger comes from the
hierarchy around it (`rulesets/hierarchy.py`):

* **Scale.** On a layer of scale s, one cell covers s × s fine cells, so one update
  moves a border s fine cells.
* **Period.** A layer that updates every p steps records its motion in the space-time
  volume (time up) as a slope of s / p fine cells per step. A shrink draws terraces s
  cells deep and p steps tall; a move leans s / p cells per step. The kinds choose p
  from the shape they want (a roof's slope, a chamber's radius, a helix's lean) rather
  than leaving it at s.
* **Where.** Parent layers choose which table runs where: each combination of parent
  bits (a context) selects its own table. Fixed patterns in a parent layer make zones,
  e.g. shear's layer 2 decides where layer 1 moves and where it holds.
* **When.** Toggle layers above are clocks. A context can depend on a clock bit (strata
  phases; caves' spread / shrink cycle), on the AND of several clocks (pagoda's pulse:
  fill on the footprint, else shrink), or on their binary count (helix's heading, one
  direction per count).
* **Material.** The fine layer usually turns the coarse result into material: a fill
  where the coarse layer is on, clear where it is off, sometimes a pool texture
  instead.

## 4. Pitfalls

* **Updates read the old state.** Every layer reads the other layers' states from
  before the step. A layer refilled during a pulse refills only if one of its own
  updates falls inside the pulse, so a pulse shorter than the layer's period can miss
  every update.
* **Seed sizes.** spread-count2 cannot start from a single cell: its neighbours see
  only one live cell. Its seeds are at least 2 wide.
* **Unaligned clocks.** `make_layers` staggers update steps by default (each layer
  fires at phase floor(log2 period)). Clocks meant to count together in binary need
  the layers aligned instead (a draw's `'stagger': False`), or the count skips and
  repeats states.
* **Scale multiples.** Each layer's scale must be a multiple of every finer layer's
  scale; a scale-3 layer under clocks needs the clocks at 48, not 16.
* **Wrapping.** The lattice wraps at its edges, so anything that moves or spreads
  eventually meets its own far side.

## 5. Summary

Decide what one cell must see; write that as a test over the 256 neighbourhoods for
each own state; then use scale, period and parent contexts to turn "one cell per
update" into the slope, size and timing wanted.
