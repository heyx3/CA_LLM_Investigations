# ca3d

A clean reconstruction of the cellular-automaton experiments in `../raws`: volume
generators for each CA construction that was tried, the hierarchical ("multi-octave")
CA behind the cityscape, and the voxel DDA renderer.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # numpy, scipy, pillow, pytest
.venv/Scripts/python -m pytest                              # known-answer tests
.venv/Scripts/python scripts/render_gallery.py              # every scene -> out/gallery/
.venv/Scripts/python scripts/render_scene.py --list
.venv/Scripts/python scripts/render_scene.py cityscape --seed 3 --res 1200
.venv/Scripts/python scripts/find_rules.py --list-measures            # search heuristics
.venv/Scripts/python scripts/run_experiment.py --list                 # the original analyses
.venv/Scripts/python scripts/render_gallery.py --saved                # saved rules -> out/gallery/saved/
```

The first hierarchical scene builds the rule pools (~2 min) and caches them in
`data/rule_pools.npz`.

## Layout

`ca3d` has three sub-packages; import modules from them, e.g.
`from ca3d.rulesets import cityscape` or `from ca3d.analysis import search`.

**`ca3d/rulesets/`**: the cellular automata.

| module | what it is | replaces (raws) |
| --- | --- | --- |
| `rules.py` | rule-table operators: lambda, flip, biased assign, set/clamp lambda | parts of alt, exp12, nontot |
| `wolfram.py` | 1D binary CA of any radius; `lsb`/`msb` rule numbering | sweep (tested identical), `stack_search.py` (missing) |
| `life.py` | 2D B/S and 512-entry Moore rules, expansion, D4 rule rotation | nontot, rotate, hybrid |
| `hierarchy.py` | **the** hierarchical CA engine: 1D/2D lattices, all-parents or chain wiring, per-context rule banks, staggered schedules, rule rotation policies | ca2d, nt_stack, staggered, pillars, twin_fine, ablate, allparents, stack |
| `families.py` | dead / static / slow / edge / complex rule families; replays the original pool searches | pillars, nontot, ablate, the `pool_*.npy` files (missing) |
| `cityscape.py` | the cityscape configuration (`make`), its context plans and recorded profile | pillars `build_mixed` |
| `initial.py` | initial conditions: blobs, uniform, sparse points, rings, gradient, quadrants, half plane | seeds (tested identical) |
| `saved.py` | the catalogue of rules and configurations worth keeping (`data/saved_rules.json`) | -- |
| `double_spacetime.py` | 1D CA extruded to 3D with per-slab rules; the row/column siblings; lambda terrain | sweep, alt, exp12, baserules, `terrain_final.py` (missing) |
| `triplanar.py` | three 1D sheets combined on orthogonal planes | triplanar |
| `octaves.py` | octave blending of any volume builder | octaves, baserules |
| `lwd.py` | Life without Death birth fields, heightfields, ladders, LWD/GoL hybrid | lwd3d, lwd_seed, lwd_vis, hybrid, multiseed, ladders2, `lwd.py` (missing) |
| `lattice_gas.py` | 6-channel lattice-gas DLA | lgca3, `lgca.py` (missing) |
| `deposition.py` | box sums, frozen-deposition CA (+ bleed), macro/micro hangar | multi3, bleed3, hangar2, `hangar.py`/`bleed.py` (missing) |

**`ca3d/analysis/`**: measuring output and searching for rules.

| module | what it is | replaces (raws) |
| --- | --- | --- |
| `metrics.py` | every heuristic, for 2D grids and 3D volumes (batched), with a registry of what values read as | scattered through every script |
| `dynamics.py` | rule spaces (1D Wolfram, 2D/3D outer-totalistic, 2D Moore), assays (how a rule is tested), lazily measured trials | ca2d, nontot, pillars, sweep |
| `search.py` | bands, criteria, staged pipelines, candidate samplers, target profiles, parameter sweeps, result tables | the search loops in pillars, nontot, baserules, exp12, ... |
| `influence.py` | hierarchy coupling tests: pinning, one-cell perturbation, pattern traffic, uniform vs effective parent sensitivity, XOR coupling | allparents, stack, stack2, `influence.py` (missing) |
| `experiments.py` | 23 named experiments replaying the original analysis drivers, printed beside the recorded numbers | every `__main__` block in raws |

**`ca3d/render3D/`**: turning volumes into pictures.

| module | what it is | replaces (raws) |
| --- | --- | --- |
| `render.py` | orthographic isometric DDA renderer with shadows | voxel |
| `color.py` | ramps, HSV layer palette, 2D colourisers | hsvcolor, ca2d, stack3 |
| `scenes.py` | named recipes for every construction, plus scenes for saved rules | -- |

Scripts: `render_scene.py` and `render_gallery.py` (images; both take saved-rule names),
`find_rules.py` (rule search from the command line), `run_experiment.py` (the analyses;
output to `out/experiments/`).

### The hierarchical CA in one paragraph

Layers are binary lattices at scales 1, 2, 4, 8 (finest first). A layer with P parent
layers owns `2**P` rule banks, `rules[context, pattern]`, where `pattern` is the cell's
3x3 Moore block (512 values) and `context` packs one bit from each parent, sampled at
the cell (nearest parent = most significant bit). Layer *i* updates when
`t % period == phase`, with `period = scale` and staggered `phase = log2(period)` so
no step updates every layer. The cityscape gives the fine layer's 8 contexts rules
from the families `dead, dead, static, static, static, static, complex, complex`, the
coarse layers rules from the slow pool, and starts layer 0 from patchy blobs.

Rotation turns every rule bank 90 degrees mid-run. Two triggers were used:
`RotateEvery(period)` (the rotating cityscape; periods 8, 20 and 40 were rendered) and
the later `RotateOnDensityLadder`: orientation = `int((d - d_start) / delta) mod 4`,
where `d` is the coarsest layer's density, checked before each step. That layer
fills monotonically (~0.49 to ~0.80), so ladder turns come at irregular,
seed-dependent heights and cluster near the base.

### Reproduction status

The seed-3 cityscape is reproduced exactly except for its two `complex` contexts:

* the dead, static and slow pools replay `pillars.py`, `ablate.py` and `nontot.py` with
  the original seeds and draw order. The counts match the notes exactly: 156 dead,
  71 static, and slow survivors 240 / 80 / 39 / 12 (= 371) for 0 / 4 / 12 / 32 flips.
* coarse-layer densities 0.099 / 0.186 / 0.695 and pillar fractions 0.40 / 0.60 / 0.79
  match the notes to every printed digit.
* density-ladder rotation steps match the original chat for all six recorded seeds.
* the fine layer gives density 0.189 / pillars 0.47 / void 80.9% (original 0.196 /
  0.46 / 80.2%). The difference is the `complex` pool (`pool_nt_sparse.npy`), whose
  generator is lost; it is rebuilt from the documented criteria.

`tests/test_reproduction.py` pins all of this. `tests/test_raws_equivalence.py` runs the
original code that survives in raws (only its function definitions) against ours.

The patchy start the notes call essential is not: raws/seeds.py, which compared starts,
replaced only the finest layer's state, and the coarse layers (which never see the fine
layer) decide what grows where. `cityscape.make(coarse_start=...)` starts the coarse
layers from a pattern too, and that does shape the city.

## Searching for interesting rules

The search code follows what the notes learned the hard way
(`raws/Quantifying Cellular Automaton Output.md`): interesting output lives in a *band*
of a measure, never at an extreme; pair one band with an independent second measure
(compact *and* slow, partial damage *and* mid density); order filters by cost; target
the profile of an output you like rather than maximising anything; run several seeds.

**Measures** (`metrics`) work on whatever makes sense for them: a 2D grid (one state,
or a 1D CA's space-time image) or a 3D volume (a 2D CA's space-time, or one state of a
native 3D CA). A leading batch axis is allowed (`dims=` the sample's dimensions).
`metrics.describe_measures()` lists them with what their values read as.

| group | measures | 2D | 3D |
| --- | --- | --- | --- |
| occupancy | density, entropy, block entropy, gzip ratio (relative to the same cells shuffled) | yes | yes |
| spatial | compactness (3^d box), coherence, components (count / median / largest share), largest void, correlation length per axis, anisotropy, fractal dimension | yes | yes |
| temporal | change rate, temporal correlation length, pillar fraction, overhang fraction | a 1D CA's space-time | a 2D CA's space-time, or a native volume's "up" axis |
| motifs | thin-linear (long *and* thin), streaks, enrichment per label, schedule confound | yes | yes |
| | domain filter | a 1D CA's space-time | -- |
| slices | slice-density series, its autocorrelation and correlation length, degenerate slabs | -- | layered builds |

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

## The original analyses

`ca3d/analysis/experiments.py` replays the analysis drivers. `python scripts/run_experiment.py
NAME` runs one, and `--all --quick` smoke-tests them all. Each prints its tables with
the recorded numbers underneath, and writes them to `out/experiments/`. How the
results compare with the notes:

| experiment | replays | agreement with the notes |
| --- | --- | --- |
| `dead_static` | pillars.py | exact: 156 dead, 71 static |
| `slow_perturbation` | nontot.py | exact: kept 240 / 80 / 39 / 12 / 0; compactness 0.52 / 0.58 / 0.55 / 0.51; anisotropy 0.0117 / 0.0161 / 0.0257 / 0.0400 |
| `rule_census` | (census scripts lost) | close: 5,298 density-viable (5,252), 272 in the damage band (249); the compactness x change table is within a few rules per cell once restricted to density-viable rules (the notes' table was) |
| `wolfram_census`, `native3d_census` | new | the same heuristics on the 1D and native 3D rule spaces |
| `layer_contributions` | the per-layer table | coarse layers exact (density, change per step 0.0265 / 0.0110 / 0.0150, pillars, median blob 8 / 32 / 64); pinning changes 0.36 / 0.38 / 0.23 (0.345 / 0.353 / 0.209) |
| `initial_conditions` | seeds.py | the original changed only the finest layer's start, so every start gave the same city (ours: 0.184-0.189 vs 0.193-0.195, the offset being the complex pool). Starting every layer from the pattern changes the city completely: density 0.004 (sparse points) to 0.454 (half plane), rings give a ring-shaped city |
| `cityscape_contexts` | the streak and vocabulary analysis | vocabulary exact (31% / 65%, 69% / 78%); streak enrichment of the dead and static contexts matches when taken relative to each context's share of *solid* voxels (ctx0 2.78 vs 2.30, ctx3 0.22 vs 0.24, ctx4 3.62 vs 3.52) |
| `context_plans` | the plan-ratio table | 2/4/2 matches; the context order of the other plans was not recorded. Spread over five rule draws 0.10-0.15, as noted |
| `plan_search` | new | target-profile search over all 45 splits |
| `layer_ablation` | ablate.py | the original draws; no recorded numbers |
| `schedule_stagger` | staggered.py | exact: context change 0.0367 -> 0.0422; max firing 4 -> 3; banding drops |
| `wiring_influence` | allparents.py, stack.py, stack2.py | all-parents one-cell flips exactly 0.012-0.128, pinning 0.34-0.47 (0.33-0.50); chain wiring severs layer 3 for two of three seeds (pin 0.008 / 0.000), and one chain table has uniform sensitivity 0.25 but effective 0.004 |
| `base_rules` | baserules.py | coherence saturates (3.05-3.18 vs 3.06-3.21), as noted. Part counts swing wildly between rule draws (ECA 4: 10-387, k5 rule B: 30-1,012), so the recorded single-draw 900-1,020 for ordered rules is plausible but not reproduced; over medians the gzip-parts correlation is -0.15 (recorded -0.57). sweep.py's conventions are identical to ours, so this is not a convention issue |
| `mutation_strength` | sweep.py | exact draw-for-draw replay (tested): slab variety 0.038 -> 0.140 from 1 to 12 flips with density pinned at 0.47-0.50; the k=7 synchronisation rules give identical statistics, as noted |
| `double_spacetime_sweep` | exp12.py, alt.py | exact walk autocorrelation (0.92 at lag 1, ~0 by lag 16); full cascade with 4-bit walk degenerates 67 of 96 slabs (70) |
| `octave_persistence` | octaves.py | the same monotone persistence dial; (24, 96) coherence 1.77, recipe 3.12 (3.13); part counts within ~1.5x |
| `triplanar_combiners` | triplanar.py | exact: density 0.11, coherence 1.58-2.13, correlation length 1 |
| `lwd_soups` | lwd.py (missing), lwd_vis.py | exact: 639 steps at 0.005, 5 at 0.6, ~9,900 holes, fill 0.66-0.81 |
| `lwd_seeds`, `lwd_ladders` | lwd_seed.py, ladders2.py | ladders exact (thinness 0.704, bulk 0.094) |
| `hybrid_schedules` | hybrid.py, multiseed.py | exact: one seed 0.029 / coherence 28.0 / 7 parts / overhangs 0.0020; five seeds 0.075 / 10.6 / 22 / 0.0052 |
| `lattice_gas` | lgca3.py | fractal dimension 1.9-2.4 (1.93-2.28), coherence 10-21, one part; the isotropic ablation grows nothing |
| `deposition_sweep` | multi3.py, hangar2.py | exact: density 0.101, coherence 4.17, 3,434 parts, and all five refractory-ablation rows; macro chambers 53% |

## What had to be guessed

Eleven imported modules and all rule pools were missing from `raws`; they were rebuilt
from their call sites and the notes. Points of genuine uncertainty:

* **Complex and edge pools** are rebuilt from the documented criteria (see above).
* **`cityscape_twin`'s context plan** (16 contexts) is not recorded; it repeats the
  cityscape plan twice and comes out much denser than the cityscape.
* **Lambda terrain** ramps lambda *cumulatively* (each slab nudged from the previous),
  which reproduces the documented coherence (1.87 vs 1.85); independent ramping gave
  1.16.
* **Damage-spreading** settings (64², 60 steps, one flipped cell) are not recorded.

## Deliberate deviations from raws

* Lattice gas: bounce-back overwrote the bounced particles of half the channels; it now
  conserves particles (tested).
* Renderer: ray distances are tracked from each ray's origin throughout, which rules
  out the documented `t_hit` shadow bug by construction. The renderer also skips
  shadow rays for faces turned away from the light, and computes true face normals
  for rays that hit the grid boundary.
* Mutations pick distinct entries (the raws sampled with replacement, so repeated
  picks could cancel out). `double_spacetime.xor_mutants` keeps the original draw for
  exact replays.
* Frozen deposition defaults to p0 0.003, birth 8-18, freeze 10. multi3.py's own
  defaults (p0 0.001-0.002) cannot nucleate (failure mode #5); replaying its sweep
  showed the documented results came from this configuration.
* `metrics.components` defaults to 6-connectivity like the raws code, even though the
  notes say 26.

## Not ported (yet)

* `skew.py`, `skew5.py`: a side experiment (two-layer 1D CA with a 4-state coarse layer,
  and a cyclic CA) that only produced 2D images.
* The matplotlib panels (`render2d.py`, `lwd_vis.py`): figures rather than analyses.
