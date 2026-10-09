# ca3d

Cellular-automaton experiments in Python: volume generators for each CA construction
that was tried, the hierarchical ("multi-octave") CA behind the cityscape, tools for
measuring a CA's output and searching for rules worth keeping, and a voxel renderer.

New to the vocabulary (lambda, B/S notation, soup, damage spreading, context...)?
Read [docs/CONCEPTS.md](docs/CONCEPTS.md) first. How the package relates to the earlier
exploratory scripts it was rebuilt from is in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # numpy, scipy, pillow, pytest
.venv/Scripts/python -m pytest                              # known-answer tests
.venv/Scripts/python scripts/render_gallery.py              # every scene -> out/gallery/
.venv/Scripts/python scripts/render_scene.py --list
.venv/Scripts/python scripts/render_scene.py cityscape --seed 3 --res 1200
.venv/Scripts/python scripts/find_rules.py --list-measures            # search heuristics
.venv/Scripts/python scripts/run_experiment.py --list                 # the experiments
.venv/Scripts/python scripts/render_gallery.py --saved                # saved rules -> out/gallery/saved/
```

The first hierarchical scene builds the rule pools (~2 min) and caches them in
`data/rule_pools.npz`.

## Layout

`ca3d` has three sub-packages; import modules from them, e.g.
`from ca3d.rulesets import cityscape` or `from ca3d.analysis import search`.
Suggested reading order: `rules`, `wolfram`, `life`, `hierarchy`, `cityscape`.

**`ca3d/rulesets/`**: the cellular automata.

| module | what it is |
| --- | --- |
| `rules.py` | rule-table operators: lambda, flip, biased assign, set/clamp lambda |
| `wolfram.py` | 1D binary CA of any radius; `lsb`/`msb` rule numbering |
| `life.py` | 2D B/S and 512-entry Moore rules, expansion, rule rotation |
| `hierarchy.py` | **the** hierarchical CA engine: 1D/2D lattices, all-parents or chain wiring, per-context rule banks, staggered schedules, rule rotation policies |
| `families.py` | dead / static / slow / edge / complex rule families and the pools drawn from them |
| `cityscape.py` | the cityscape configuration (`make`), its context plans and measured profile |
| `initial.py` | initial conditions: blobs, uniform, sparse points, rings, gradient, quadrants, half plane |
| `saved.py` | the catalogue of rules and configurations worth keeping (`data/saved_rules.json`) |
| `double_spacetime.py` | 1D CA extruded to 3D with per-slab rules; the row/column siblings; lambda terrain |
| `triplanar.py` | three 1D sheets combined on orthogonal planes |
| `octaves.py` | octave blending of any volume builder |
| `lwd.py` | Life without Death birth fields, heightfields, ladders, LWD/GoL hybrid |
| `lattice_gas.py` | 6-channel lattice-gas diffusion-limited aggregation |
| `deposition.py` | box sums, frozen-deposition CA (+ bleed), macro/micro hangar |

**`ca3d/analysis/`**: measuring output and searching for rules.

| module | what it is |
| --- | --- |
| `metrics.py` | every measure, for 2D grids and 3D volumes (batched), with a registry of what values read as |
| `dynamics.py` | rule spaces (1D Wolfram, 2D/3D outer-totalistic, 2D Moore), assays (how a rule is tested), lazily measured trials |
| `search.py` | bands, criteria, staged pipelines, candidate samplers, target profiles, parameter sweeps, result tables |
| `influence.py` | hierarchy coupling tests: pinning, one-cell perturbation, pattern traffic, uniform vs effective parent sensitivity, XOR coupling |
| `experiments.py` | named experiments, each asking one question, printed beside the numbers recorded earlier |

**`ca3d/render3D/`**: turning volumes into pictures.

| module | what it is |
| --- | --- |
| `render.py` | orthographic isometric voxel renderer (ray marching) with shadows |
| `color.py` | ramps, HSV layer palette, 2D colourisers |
| `scenes.py` | named recipes for every construction, plus scenes for saved rules |

Scripts: `render_scene.py` and `render_gallery.py` (images; both take saved-rule names),
`find_rules.py` (rule search from the command line), `run_experiment.py` (the
experiments; output to `out/experiments/`).

### The hierarchical CA in one paragraph

Layers are binary lattices at scales 1, 2, 4, 8 (finest first). A layer with P parent
layers owns `2**P` rule banks, `rules[context, pattern]`, where `pattern` is the cell's
3x3 Moore block (512 values) and `context` packs one bit from each parent, sampled at
the cell (nearest parent = most significant bit). Layer *i* updates when
`t % period == phase`, with `period = scale` and staggered `phase = log2(period)` so
no step updates every layer. The cityscape gives the fine layer's 8 contexts rules
from the families `dead, dead, static, static, static, static, complex, complex`, the
coarse layers rules from the slow pool, and starts layer 0 from patchy blobs.

Rotation turns every rule bank 90 degrees mid-run, changing the direction structures
grow in. `RotateEvery(period)` turns at fixed intervals; `RotateOnDensityLadder` turns
each time the coarsest layer's density has climbed another `delta`, so the turns come
at irregular, seed-dependent heights. See [docs/CONCEPTS.md](docs/CONCEPTS.md) §5.

## Searching for interesting rules

The search code is built on a few lessons: interesting output lives in a *band* of a
measure, never at an extreme; pair one band with an independent second measure (compact
*and* slow, partial damage *and* mid density); order filters by cost; target the
profile of an output you like rather than maximising anything; run several seeds.

**Measures** (`metrics`) work on a 2D grid (one state, or a 1D CA's space-time image) or
a 3D volume (a 2D CA's space-time, or one state of a native 3D CA). A leading batch axis
is allowed (`dims=` the sample's dimensions). `metrics.describe_measures()` lists them
with what their values read as; docs/CONCEPTS.md §7 defines each one.

**Rule search** (`dynamics` + `search`). A *rule space* steps thousands of candidate
rules at once. An *assay* fixes what every rule is tested under: lattice, soup (one
soup shared by all rules), run length, burn-in. A *pipeline* of *stages* applies
*criteria* (a measure and a band) in order; each criterion only sees what passed the
previous one, and every run is cached, so nothing is computed twice. Measures apply to
a rule's final state, or with `on='history'` to its space-time record (recorded only
for the rules that get that far). `'lambda'`, `'change'` and `'damage'` are rule-level.

```python
import numpy as np
from ca3d.analysis import dynamics, search
from ca3d.analysis.search import Criterion, Stage

space = dynamics.Totalistic()            # or Moore(), Wolfram(radius=2), Totalistic(ndim=3)
rules = search.random_rules(space, 6000, np.random.default_rng(0))
result = search.Pipeline([
    Stage([Criterion('density', (0.15, 0.85)), Criterion('damage', (0.02, 0.25))],
          dynamics.Assay(n=64)),                          # cheap first: small lattice
    Stage([Criterion('change', (0.001, 0.05)), Criterion('compactness', (0.4, 0.7)),
           Criterion('pillars', None, on='history')]),     # None: record, don't filter
]).run(space, rules)
print(result.report())                                    # funnel and survivors
```

Samplers: uniform random tables, fixed-lambda tables, interval rules (`Bb0-b1/Ss0-s1`,
for big totalistic spaces such as 3D), expand-and-perturb, rule-space walks. The same
from the shell: `scripts/find_rules.py --band NAME LO HI ...`, with `--render K` to
render the first K survivors (1D: space-time image; 2D: space-time volume; 3D: state).

**Configuration search.** `search.sweep(build, grid, seeds, measures)` builds and
measures every combination (any builder returning a grid or volume, e.g. a cityscape
with a given context plan); `Table.aggregate('plan')` gives the mean and spread over
seeds, and `Table.rank(search.Target({...}))` sorts by distance to a reference profile.
`plan_search` does this over all 45 dead/static/complex splits against the cityscape.

**Coupling** (`influence`): pin a layer, flip one of its cells, count pattern traffic,
compare uniform and traffic-weighted parent sensitivity, build XOR-coupled banks.

Measurement traps the code handles explicitly:
* a harsh soup can absorb a damage test's perturbation before the rule's own regime
  is reached; `Assay(damage_burn=...)` flips the cell after a burn-in;
* additive rules die on power-of-two rings and tori (ECA 90 on 256 cells, parity rules
  on 32^2), so censuses use other lattice sizes;
* sparse soups may not ignite at all, so the LWD census runs several soups per density.

## Saved rules

Finds worth keeping go into `data/saved_rules.json` (`ca3d.rulesets.saved`): the rule
(B/S notation, a hex rule number, or a 512-entry Moore table as hex), the settings it
was found and looked good under, what it measured, how it was found, and a note. A
search is cheap to rerun but hard to rerun identically; the catalogue keeps the result.

```bash
.venv/Scripts/python scripts/render_scene.py --list                   # scenes, then saved rules
.venv/Scripts/python scripts/render_scene.py tower_field --size 96
.venv/Scripts/python scripts/find_rules.py ... --keep my_rule         # save the first survivor
```

From Python: `saved.get(name).table()`, `.assay()`, `saved.keep(entry)`, `saved.drop(name)`.
Saved so far:

| name | what it is |
| --- | --- |
| `fossil_waves` | the frozen-deposition run (concentric wave fossils) |
| `tower_field` | B5/S234678 with 12 Moore entries flipped: an anisotropic tower field |
| `b46_s0135678`, `b47_s04567` | life-like rules that are compact, slow and in the complex band: plateaus with towers |
| `b4_s2347`, `b578_s0134578` | the same family: thin tower fields |
| `b11_14_s3_11` | the one survivor of the native 3D census |
| `cityscape_2_4_2`, `cityscape_2_5_1` | the cityscape and the plan nearest its profile |

A profile match is necessary, not sufficient: the all-complex plan also matched the
cityscape's density, pillars, void and streaks, but renders as a uniform block of
strands (coherence 1.8 vs 3.4), so it was not kept.

## Experiments

`ca3d/analysis/experiments.py` holds named experiments. `python scripts/run_experiment.py
NAME` runs one, and `--all --quick` smoke-tests them all. Each prints its tables with
the numbers recorded earlier underneath, and writes them to `out/experiments/`. How the
results compare with the recorded ones is in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).
