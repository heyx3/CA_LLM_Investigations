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
```

The first hierarchical scene builds the rule pools (~2 min) and caches them in
`data/rule_pools.npz`.

## Layout

| module | what it is | replaces (raws) |
| --- | --- | --- |
| `rules.py` | rule-table operators: lambda, flip, biased assign, set/clamp lambda | parts of alt, exp12, nontot |
| `wolfram.py` | 1D binary CA of any radius; `lsb`/`msb` rule numbering | `sweep.py`, `stack_search.py` (missing) |
| `life.py` | 2D B/S and 512-entry Moore rules, expansion, D4 rule rotation | nontot, rotate, hybrid |
| `hierarchy.py` | **the** hierarchical CA engine: 1D/2D lattices, all-parents or chain wiring, per-context rule banks, staggered schedules, rule rotation policies | ca2d, nt_stack, staggered, pillars, twin_fine, ablate, allparents, stack |
| `families.py` | dead / static / slow / edge / complex rule families; replays the original pool searches | pillars, nontot, ablate, the `pool_*.npy` files (missing) |
| `metrics.py` | rule profiling (batched) and volume measures | scattered through every script |
| `double_spacetime.py` | 1D CA extruded to 3D with per-slab rules; lambda terrain | alt, exp12, baserules, `terrain_final.py` (missing) |
| `triplanar.py` | three 1D sheets combined on orthogonal planes | triplanar |
| `octaves.py` | octave blending of any volume builder | octaves, baserules |
| `lwd.py` | Life without Death birth fields, heightfields, ladders, LWD/GoL hybrid | lwd3d, lwd_seed, lwd_vis, hybrid, multiseed, ladders2, `lwd.py` (missing) |
| `lattice_gas.py` | 6-channel lattice-gas DLA | lgca3, `lgca.py` (missing) |
| `deposition.py` | box sums, frozen-deposition CA (+ bleed), macro/micro hangar | multi3, bleed3, hangar2, `hangar.py`/`bleed.py` (missing) |
| `render.py` | orthographic isometric DDA renderer with shadows | voxel |
| `color.py` | ramps, HSV layer palette, 2D colourisers | hsvcolor, ca2d, stack3 |
| `scenes.py` | named recipes for every construction, used by the scripts | -- |

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

`tests/test_reproduction.py` pins all of this.

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
  picks could cancel out).
* Frozen deposition's default `p0` is 0.005: at the scripts' 0.001 the birth window is
  unreachable (failure mode #5) and nothing grows. At 0.005 it reproduces the
  documented density 0.10 and coherence 4.1.
* `metrics.components` defaults to 6-connectivity like the raws code, even though the
  notes say 26.

## Not ported (yet)

* `skew.py`, `skew5.py`: a side experiment (two-layer 1D CA with a 4-state coarse layer,
  and a cyclic CA) that only produced 2D images.
* The analysis drivers (sweeps, ablations, influence tests, matplotlib panels). Their
  building blocks are here (`metrics`, `families`, `Layer.pinned`, wiring options);
  the searches themselves come in the next phase.
