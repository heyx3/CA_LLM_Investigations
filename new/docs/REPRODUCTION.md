# Reproduction record

`ca3d` was rebuilt from earlier exploratory scripts and their recorded results. This
page keeps the bookkeeping: what matches the earlier results, what had to be guessed,
and where the rebuild deliberately differs. None of it is needed to use the package;
the tests (`tests/test_reproduction.py` and friends) pin the matches listed here.

## The seed-3 cityscape

Reproduced exactly except for its two `complex` contexts.  Its rule tables (rule draw 3
of the 2/4/2 plan from the rebuilt pools) are now stored in `cityscape.REFERENCE_BANKS`,
and the pool-search replays keep the recorded criteria in `analysis/experiments.py`, so
none of this depends on the family pools staying as they are.

* The `dead_static` and `slow_perturbation` experiments replay the earlier pool searches
  with the same seeds and draw order (the pools themselves are now built differently,
  see the deviations below). The counts match exactly: 156 dead, 71 static, and slow
  survivors 240 / 80 / 39 / 12 (= 371) for 0 / 4 / 12 / 32 flips.
* Coarse-layer densities 0.099 / 0.186 / 0.695 and pillar fractions 0.40 / 0.60 / 0.79
  match to every printed digit.
* Density-ladder rotation steps match for all six recorded seeds.
* The fine layer gives density 0.189 / pillars 0.47 / void 80.9% (recorded 0.196 /
  0.46 / 80.2%). The difference is its two `complex` banks: the generator of the
  original complex pool was lost, so they were drawn from one rebuilt from its
  documented criteria.

The patchy start was recorded as "essential", but it is not: the earlier comparison of
starts replaced only the finest layer's state, and the coarse layers (which never see
the fine layer) decide what grows where. `cityscape.make(coarse_start=...)` starts the
coarse layers from a pattern too, and that does shape the city.

## Experiments against the recorded results

`python scripts/run_experiment.py NAME` runs one experiment; `--all --quick`
smoke-tests them all. Each prints its tables with the recorded numbers underneath, and
writes them to `out/experiments/`.

| experiment | agreement with the recorded results |
| --- | --- |
| `dead_static` | exact: 156 dead, 71 static |
| `slow_perturbation` | exact: kept 240 / 80 / 39 / 12 / 0; compactness 0.52 / 0.58 / 0.55 / 0.51; anisotropy 0.0117 / 0.0161 / 0.0257 / 0.0400 |
| `rule_census` | close: 5,298 density-viable (5,252), 272 in the damage band (249); the compactness x change table is within a few rules per cell once restricted to density-viable rules (the recorded table was) |
| `wolfram_census`, `native3d_census` | new: the same measures on the 1D and native 3D rule spaces |
| `layer_contributions` | coarse layers exact (density, change per step 0.0265 / 0.0110 / 0.0150, pillars, median blob 8 / 32 / 64); pinning changes 0.36 / 0.38 / 0.23 (recorded 0.345 / 0.353 / 0.209) |
| `initial_conditions` | starting only the finest layer gave the same city every time (ours: 0.184-0.189 vs 0.193-0.195, the offset being the complex pool). Starting every layer from the pattern changes the city completely: density 0.004 (sparse points) to 0.454 (half plane), rings give a ring-shaped city |
| `cityscape_contexts` | vocabulary exact (31% / 65%, 69% / 78%); streak enrichment of the dead and static contexts matches when taken relative to each context's share of *solid* voxels (ctx0 2.78 vs 2.30, ctx3 0.22 vs 0.24, ctx4 3.62 vs 3.52) |
| `context_plans` | the reference cityscape (the recorded 2/4/2 draw) gives the seed-3 numbers above; every other draw comes from the current pools, so the recorded single draws are not reproduced (nor was the context order of the other plans recorded). Pillar fraction spreads 0.09-0.17 over five rule draws (recorded 0.07-0.15) |
| `plan_search` | new: target-profile search over all 45 splits |
| `layer_ablation` | the original draw procedure, from the current pools; no recorded numbers |
| `schedule_stagger` | exact: context change 0.0367 -> 0.0422; max firing 4 -> 3; banding drops |
| `wiring_influence` | all-parents one-cell flips exactly 0.012-0.128, pinning 0.34-0.47 (recorded 0.33-0.50); chain wiring severs layer 3 for two of three seeds (pin 0.008 / 0.000), and one chain table has uniform sensitivity 0.25 but effective 0.004 |
| `base_rules` | coherence saturates (3.05-3.18 vs 3.06-3.21), as recorded. Part counts swing wildly between rule draws (ECA 4: 10-387, k5 rule B: 30-1,012), so the recorded single-draw 900-1,020 for ordered rules is plausible but not reproduced; over medians the gzip-parts correlation is -0.15 (recorded -0.57). The conventions are identical to the earlier ones, so this is not a convention issue |
| `mutation_strength` | exact draw-for-draw replay (tested): slab variety 0.038 -> 0.140 from 1 to 12 flips with density pinned at 0.47-0.50; the k=7 synchronisation rules give identical statistics, as recorded |
| `double_spacetime_sweep` | exact walk autocorrelation (0.92 at lag 1, ~0 by lag 16); full cascade with 4-bit walk degenerates 67 of 96 slabs (recorded 70) |
| `octave_persistence` | the same monotone persistence dial; (24, 96) coherence 1.77, recipe 3.12 (3.13); part counts within ~1.5x |
| `triplanar_combiners` | exact: density 0.11, coherence 1.58-2.13, correlation length 1 |
| `lwd_soups` | exact: 639 steps at 0.005, 5 at 0.6, ~9,900 holes, fill 0.66-0.81 |
| `lwd_seeds`, `lwd_ladders` | ladders exact (thinness 0.704, bulk 0.094) |
| `hybrid_schedules` | exact: one seed 0.029 / coherence 28.0 / 7 parts / overhangs 0.0020; five seeds 0.075 / 10.6 / 22 / 0.0052 |
| `lattice_gas` | fractal dimension 1.9-2.4 (recorded 1.93-2.28), coherence 10-21, one part; the isotropic ablation grows nothing |
| `deposition_sweep` | exact: density 0.101, coherence 4.17, 3,434 parts, and all five refractory-ablation rows; macro chambers 53% |

## What had to be guessed

Eleven imported modules and all rule pools were missing from the earlier scripts; they
were rebuilt from their call sites and the recorded results. Points of genuine
uncertainty:

* **Complex and edge pools** are rebuilt from the documented criteria.
* **`cityscape_twin`'s context plan** (16 contexts) is not recorded; it repeats the
  cityscape plan twice and comes out much denser than the cityscape.
* **Lambda terrain** ramps lambda *cumulatively* (each slab nudged from the previous),
  which reproduces the documented coherence (1.87 vs 1.85); independent ramping gave
  1.16.
* **Damage-spreading** settings (64², 60 steps, one flipped cell) are not recorded.

## Deliberate deviations

* Rule pools: each family's members come from a census of all 262,144 life-like rules
  instead of the original random samples, and every perturbed or blended table is
  checked against its family again (see `rulesets/families.py`).  The reference
  cityscape and the replayed pool searches don't use the pools (see the seed-3
  cityscape above), so none of the reproduced numbers move.
* Lattice gas: bounce-back used to overwrite the bounced particles of half the channels;
  it now conserves particles (tested).
* Renderer: ray distances are tracked from each ray's origin throughout, which rules
  out a shadow-distance bug of the earlier renderer by construction. The renderer also
  skips shadow rays for faces turned away from the light, and computes true face
  normals for rays that hit the grid boundary.
* Mutations pick distinct entries (the earlier code sampled with replacement, so
  repeated picks could cancel out). `double_spacetime.xor_mutants` keeps the old draw
  for exact replays.
* Frozen deposition defaults to p0 0.003, birth 8-18, freeze 10. The earlier defaults
  (p0 0.001-0.002) cannot nucleate; replaying the sweep showed the documented results
  came from this configuration.
* `metrics.components` defaults to 6-connectivity (faces only) like the earlier code,
  even though the written notes say 26.

## Not carried over

Two side experiments (a two-layer 1D CA with a 4-state coarse layer, and a cyclic CA)
that only produced 2D images, and the matplotlib figure panels.
