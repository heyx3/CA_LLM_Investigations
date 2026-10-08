# CA 3D Rendering: Handoff Notes

Oct 4, 2026 · @Billy

## Purpose

The goal is aesthetically interesting, anisotropic 3D scenes generated cheaply from cellular automata, rendered as voxels. Not scientific visualisation — the criterion is that output should read as structure at a glance rather than as noise or uniform texture.

Fourteen volume constructions were tried. The best result (octave blending, 51 connected parts at 0.30 density) and the highest-coherence result (lattice-gas DLA, coherence 10-21) came from opposite ends of the cost spectrum. Most of the value in this document is in the failures: six distinct failure modes recur, and each has a measurement that reveals it early.

Read the failure catalogue and the metrics section before running anything. Several promising-looking results in this project were artifacts of badly chosen metrics, and two were outright bugs that produced plausible images.

## The renderer

`voxel.py` renders a boolean 3D array as voxels. Signature: `render(occ, W=900, H=900, zoom=1.0, attr=None)` returns a uint8 RGB array.

How it works:

- **Orthographic isometric camera** along `(-1,-1,-1)`. All rays share one direction; only origins vary across the image plane. No perspective divide.
- **Amanatides-Woo DDA**, vectorised across all \~810k rays in numpy rather than looped per pixel. A ray-AABB slab test advances each ray to the box boundary first, so empty space outside costs nothing. \~2s for a 96³ volume at 900×900.
- **Normals are free** — the axis last stepped and its sign give the face normal directly.
- **One shadow ray per pixel** from the hit point toward the key light, offset along the normal.
- **Ambient is a shadowless directional light**: a second direction contributing `max(0, N·L)` with no occlusion test, plus a small flat floor term.
- **`attr`** (optional) is a float array in \[0,1\] matching `occ`, used for the colour ramp instead of the z coordinate. Added late; useful for showing any per-voxel scalar field.

### The bug worth knowing about

The first version produced images with no shadows. `tMax` in the DDA is measured from the box-entry point, not the ray origin, so `origin + dir * t_hit` landed somewhere arbitrary and every shadow ray started in the wrong place.

What exposed it: shadow coverage came back as **exactly 1.0% for two completely different light directions**. Physics does not do that. A unit test against a floating block over a floor showed the DDA traversal itself was correct, localising the fault to the `t_hit` bookkeeping. Two-line fix; coverage went to 27.5% and became responsive to light angle.

The lesson generalises: a broken shadow ray produces a *plausible* image. For voxel work, test against a scene whose answer you know before trusting the picture.

### Rendering dense volumes

Anything above \~0.35 density is opaque from outside and shows nothing. Two workarounds, both used here:

- Cut the octant facing the camera (`vol[-c:,-c:,-c:] = False`) to expose three internal faces.
- Drop resolution so voxels are larger. At 128³ with a 900px image each voxel is \~3px and dense output reads as pure salt-and-pepper; 64³ at \~6px per voxel is legible.

## Files

All in the working directory. Several import each other, so keep them together.

| File | Contains | Reuse? |
| --- | --- | --- |
| `voxel.py` | The renderer plus a Life space-time volume generator | Yes — the core dependency |
| `sweep.py` | `table_of(rule_int, r, order)` and `spacetime(seed, table, r, steps)` — 1D CA primitives for any radius and bit-order | Yes — imported everywhere |
| `hangar.py` | 3D summed-area table `box_count(V, R)`, 8-corner inclusion-exclusion | Yes — makes large-radius 3D CA O(1) per cell |
| `bleed.py` | `box_count_aniso(V, Rx, Ry, Rz)` — per-axis radii | Yes — for anisotropic neighbourhoods |
| `octaves.py` | Multi-scale octave blending. **The best general-purpose generator** | Yes — start here |
| `baserules.py` | Octave blend swept over base rules of different radii, with λ/gzip correlation | Yes — the template for rule sweeps |
| `exp12.py` | Double-spacetime with rule-space walk + cascade seeding | Yes |
| `lgca3.py` | Lattice-gas DLA with directional states, fractal-dimension measurement | Yes — highest coherence of anything here |
| `multiseed.py` | Multi-seed staggered LWD/GoL hybrid with per-burst age tracking | Yes |
| `lwd.py`, `lwd_seed.py` | Life Without Death, soup and small-seed versions, birth-time fields | Yes |
| `ladders2.py` | Morphological ladder extraction from LWD | Yes |
| `triplanar.py` | Three 1D sheets combined into a volume from O(n²) work | Only as a texture layer |
| `multi3.py` | Multi-state CA with frozen deposition | Reference |
| `terrain_final.py` | λ-ramped terrain | Reference |
| `hybrid.py` | Single-seed LWD/GoL alternation | Superseded by `multiseed.py` |

### Bit-order conventions — read this before decoding any rule hex

Two incompatible conventions are in play and a rule number is meaningless without knowing which:

- `'lsb'`: bit *i* of the integer is the output for neighbourhood value *i*. Used for the k=5 rule `360a96f9` and the elementary rules.
- `'msb'`: leftmost bit of the binary string is the output for neighbourhood 0. Used for the k=7 rules from the Das et al. synchronisation paper.

Also note the `k` ambiguity: this project uses `v` for state count and `k` for neighbourhood size, but Wolfram's own convention is `k` = number of colours and `r` = radius. Record neighbourhood size, state count *and* bit-order alongside any hex you want to keep.

## Results

All measured on 96³ unless noted. Coherence = mean live-neighbour count among live voxels, divided by `6 × density`; 1.0 is the random-noise baseline. Part count is 26-connected components — the single most useful number in the table.

| # | Construction | Density | Coherence | Parts | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Life + time as 3rd axis | 0.17 | — | — | Works. Still lifes read as columns, gliders as diagonal rods |
| 2 | Double-spacetime, independent mutations | 0.47 | 1.04 | 13,222 | Opaque; the two siblings were indistinguishable |
| 3 | Sparsity-biased mutation | 0.19 | — | — | Siblings finally differ; see-through |
| 4 | λ-ramp terrain | 0.19 | 1.85 | — | Stratified geology, genuine surface |
| 5 | Macro/micro hangar | 0.29 | 2.47 | — | Largest chamber 53% of cube |
| 6 | Multistate + frozen deposition | 0.10 | 4.17 | 3,434 | Concentric wave fossils |
| 7 | Stochastic layer bleed | 0.08–0.12 | 2.7–3.4 | 20–30k | Regression — noise destroyed the fronts |
| 8 | Rule walk + full cascade | 0.46 | 1.81 | — | Monolithic slabs via absorbing states |
| 9 | Tri-planar | 0.11 | 1.58–2.13 | — | Texture, not structure. Corr length 1 on all axes |
| 10 | Lattice-gas DLA | 0.03 | **10–21** | **1** | Highest coherence; one connected aggregate; slow |
| 11 | **Octave blend** | 0.30 | 3.13 | **51** | **Best overall.** 10% cost over single-scale |
| 12 | LWD heightfield | 0.61 | — | — | Exact, not a metaphor — LWD space-time *is* a heightfield |
| 13 | LWD/GoL hybrid | 0.029 | 28.0 | 7 | First true overhangs (0.0020) |
| 14 | Multi-seed staggered hybrid | 0.075 | 10.6 | 22 | Overhangs 0.0052; colliding fronts |

### Cost

| Method | 96³ build time |
| --- | --- |
| Tri-planar | \~10 ms |
| Double-spacetime | \~250 ms |
| Octave blend (5 levels) | \~480 ms |
| 3D CA via SAT | \~40 ms per step (\~1 s for 25 steps) |
| Lattice-gas DLA | \~900 steps to reach 3% density |

The double-spacetime construction was originally adopted because it looked cheap. It is only \~5× cheaper than a full 3D CA with summed-area tables. Its real advantage is a tiny **rule search space** — 2³² for a k=5 1D rule versus astronomically large for a general 3D rule. Keep it if you want to search for rules; do not keep it for speed alone.

## Failure catalogue

Six recurring failure modes, each with the measurement that exposed it. These cost the most time and are the most likely to recur.

### 1. Symmetric mutation cannot change density

Flipping 1 to 12 bits of a rule table raised per-slab variety from 0.038 to 0.140, but density stayed pinned at 0.47 the whole way. Bit-flipping preserves the rule's balance, so you get more variety *inside* an opaque block.

**Fix:** bias the operator. Instead of flipping *m* bits, *assign* *m* bits with P(1)=0.05. Density dropped to 0.19 with 22 of 80 slabs dead. Mutation *bias* mattered; mutation *strength* did not.

### 2. Deterministic CAs do not decay, they hit attractors

The first terrain attempt searched for rules whose density falls over time. Every setting gave `base ≈ top`; slabs either reached the ceiling or died instantly. There is no gradual fade.

**Fix:** gradients must come from the *rules*, not the dynamics. Ramp Langton's λ across the slab index and stand that axis up vertically.

### 3. Growth rules without an upper bound saturate

A 3D Larger-than-Life rule with birth/survival windows bounded below but not above went 100% solid in a few steps. It is a growth process, not a pattern-former.

**Fix:** plain majority smoothing (`count × 2 > (2R+1)³`) holds density near 0.5 and only coarsens the blobs. That is what produced the chambers.

### 4. Positive feedback between structure and deposition

A freeze condition counting *frozen* neighbours made deposition breed deposition — every parameter setting saturated at 97–99% solid.

**Fix:** make the deposited material *block* new ignition. Growth becomes self-limiting and dendritic. This inhibitory gate, not the extra states, is what produced 90% void.

### 5. Changing coupling geometry invalidates every threshold

Switching from isotropic to in-plane coupling shrank the neighbourhood from 343 cells to 49. Thresholds tuned for the old geometry became unreachable and every run died at t=0. Several results logged as *the rule is dead* were nucleation failures.

**Fix:** express thresholds as a ratio to background density, not as absolute counts. Check that your birth threshold is reachable: with neighbourhood size N and seed density p, the mean neighbour count is `N × p`.

### 6. Absorbing states propagate through seeds, not rules

Cascade seeding (slab *i+1* seeded from slab *i*'s last row) is an absorbing-state machine. Once a row goes all-zero or all-one it is stuck. At full cascade with 4-bit mutation, **70 of 96 slabs were degenerate** — 40 empty, 30 solid. A λ band on the rules does not help, because the failure travels through the seed row.

**Useful anyway:** that failure mode produces large monolithic slabs with sharp edges, which is the most architectural output in the project. Degenerate-slab count is directly controllable via cascade probability.

### Smaller traps

- **Seeding too sparse never nucleates.** Below p0≈0.003 in a radius-3 neighbourhood, a random start cannot assemble 8 live neighbours anywhere.
- **The synchronisation-task rules all flood.** φsync and the GA-evolved k=7 rules share φ(0000000)=1, so a single-seed start floods to all-1s on step one and they all blink identically. They gave byte-identical statistics at every mutation strength. Use a random IC for that family.
- **Hard masking destroys gradient information.** A cascade of thresholded octaves (AND of masks) gave coherence 2.36 and 14,648 parts; the weighted blend of the same octaves gave 3.13 and 51 parts.

## Metrics

### What to measure

| Metric | Definition | Reads as |
| --- | --- | --- |
| Density | `V.mean()` | Above 0.35 is opaque; 0.05–0.30 is workable |
| Coherence | mean live-neighbour count among live voxels ÷ `6 × density` | 1.0 = noise; above 2 is clustered |
| **Part count** | 26-connected components | **The most discriminating single number** |
| Correlation length | smallest lag where autocorrelation drops below 1/e, per axis | 1 = noise; differing per axis = anisotropy |
| Overhang fraction | solid voxels with empty space directly below | Exactly 0 for any heightfield; above 0 means true 3D |
| Fractal dimension | box-counting slope | 3.0 = space-filling; \~2.5 = 3D DLA; lower = wispier |

### Two metrics that misled

**Coherence saturates.** Once the octave machinery is involved it stops discriminating. Every base rule swept — Wolfram classes 1 through 4, radii 1 to 3, including a rule that dies immediately — landed between 3.06 and 3.21. A 5% spread. The correlation between a rule's 1D gzip ratio and its 3D coherence was **+0.09**: no relationship at all.

Part count stayed informative over the same sweep: gzip ratio against log(parts) correlated at **−0.57**. Chaotic and class-4 rules consolidate (22–42 parts); ordered class-1 and class-2 rules fragment (900–1,020 parts). This is backwards from intuition — an ordered rule makes near-identical slabs, so the summed field is nearly uniform and quantile-thresholding slices it into scattered specks.

**A badly built shape metric reads high on anything dense.** Measuring *fraction of cells in runs ≥ 6* to detect Life Without Death ladders gave 0.890 at baseline, because any dense blob contains long horizontal runs. It made every threshold look equally good.

The fix was to require long **and thin**: survives an opening with a line element, destroyed by an opening with a square element. With the corrected metric, baseline thinness was 0.704 and bulk fraction only 0.094 — which immediately showed that time-based thresholds were doing nothing and length was the only discriminator.

### Rule of thumb

Measure a known-answer control alongside every new metric. The shadow-ray bug, the saturated coherence and the broken orthogonality metric were all caught by comparing against a baseline, never by looking at output.

## Ablations

Two attempts at *extra states that carry information while staying invisible*. They gave opposite answers, and the contrast is the most generalisable result in the project.

### Hidden refractory states: inert

A Generations-style multistate CA, states 0 dead / 1 alive / 2..C-1 refractory and invisible / frozen and solid. Varying the number of hidden states:

| Hidden states | Solid | Coherence | Steps |
| --- | --- | --- | --- |
| 0 (C=2) | 0.103 | 4.17 | never exhausts |
| 1 (C=3) | 0.100 | 4.19 | 25 |
| 3 (C=5) | 0.101 | 4.17 | 25 |
| 6 (C=8) | 0.103 | 4.15 | 24 |
| 10 (C=12) | 0.103 | 4.15 | 26 |

No measurable effect on the output. Their only function was termination — with zero hidden states the process never stops. **A refractory timer carries no spatial information.** The sparsity credited to them actually came from the frozen-blocks-ignition gate (failure mode 4).

### Directional states: load-bearing

A lattice gas: six momentum channels, particles stream, bounce back off deposited solid, and deposit on contact. The control replaces the directional channels with a scalar field transported by isotropic diffusion, same particle budget, same deposition rule.

With a point seed, the control grew **literally nothing** — 0.0000 solid after 900 steps at every particle density — while the directional version built a connected aggregate at every setting.

The mechanism: isotropic diffusion *averages* the concentration field, so the value beside the growing solid relaxes below the deposition threshold and growth stalls permanently. Discrete particles with momentum arrive as quantised packets, and a single arrival always clears the threshold. Averaging destroys the thing deposition depends on.

Results: fractal dimension 1.93–2.28 (space-filling is 3.0, 3D DLA is \~2.5), coherence 10–21, always exactly **one** connected part. Lower particle density gives lower fractal dimension and wispier dendrites.

### What did not work

Biasing a momentum channel (3× particles in +z or +x) changed growth *speed* — 172 steps down to 109 — but not structure: coherence 3.80 vs 3.90 vs 3.86, identical correlation lengths. The collision step re-permutes channels at every multi-particle cell, randomising momentum faster than bias accumulates. A momentum-conserving collision rule (proper HPP/FHP, where head-on pairs rotate rather than randomise) should preserve it. **This is the most promising unfinished thread in the project.**

### Takeaway

Extra states earn their keep only when they encode something the visible state cannot — a direction, a conserved quantity, a count of events. A timer does not qualify. Always ablate: run the same construction with the states removed and check the output actually changes.

## Recipes

### Default: octave blending — `octaves.py`

Build the same construction at several resolutions, trilinear-upsample each to full size, blend as weighted octaves, threshold at a quantile to fix density.

- Levels `(6, 12, 24, 48, 96)`, persistence **0.35**, density 0.30.
- Result: coherence 3.13, correlation length 12–48, **51 parts** versus 13,222 for the same construction at single scale.
- Cost: 480 ms versus 440 ms. Per-level build is 5 / 16 / 62 / 246 ms — every coarse octave combined costs a third of the fine level alone, because cost scales as n³.

**Persistence is the main dial and it is monotone:** 0.3 gives 60 parts, 0.5 gives 1,389, 0.7 gives 6,890, 0.9 gives 12,695 (the fine level dominates and you are back at noise). Lower means coarse levels dominate.

**Octave count matters more than spacing.** Two levels `(24, 96)` gave coherence 1.77 and 14,527 parts — barely better than baseline, because a single factor-of-4 gap leaves a hole in the spectrum. You need the intermediate scales.

### Choosing a base rule

Pick on **part count**, not coherence (which saturates — see Metrics).

- Want solid consolidated architecture → chaotic or class-4 rules. ECA 30 gave 42 parts, ECA 90 gave 37, the k=7 GA-failure rule gave 22.
- Want debris fields and floating fragments → class 1 or 2. ECA 108 gave 1,020 parts, ECA 4 gave 900.

### For the double-spacetime construction specifically

Three independent fixes, all cheap, addressing different failure modes:

1. **Rule-space walk** — make each slab's rule a mutation *of the previous* rather than a fresh mutation of the base. Independent mutations give autocorrelation ≈0 at every lag along the third axis, i.e. literally white noise. A walk gives 0.92 at lag 1 decaying to zero around lag 16. Walk step size is a direct correlation-length dial.
2. **Partial cascade** — seed slab *i+1* from slab *i*'s final row with probability \~0.5. Gives occasional monolithic slabs punctuating textured volume. Full cascade degenerates 70% of slabs (see failure 6).
3. **Octave blend on top** at persistence 0.3.

### For sparse, highly connected structure — `lgca3.py`

Lattice-gas DLA, point seed, particle density 0.02, sticking probability 0.15. Coherence 10–21, one connected part, fractal dimension \~2.26. Slow: 900 steps for 3% density, and growth rate is bounded by the aggregate's surface area, so it does not scale.

### For overhangs and non-heightfield structure — `multiseed.py`

Alternate Life Without Death (fills) with Game of Life (hollows), multiple seeds ignited at staggered times.

- **M=1 GoL step only.** One strike carves; two in a row destroy the remnant before LWD can regrow. M=1 gives fill 0.011–0.029, M=2 gives 0.002–0.007.
- LWD recovery phase \~30 steps; it saturates beyond that.
- Five scattered seeds at staggered start times: fill 0.075, 22 parts, overhang fraction 0.0052 versus 0.0020 for a single seed.
- Track per-burst birth sub-step and feed it to `render(attr=...)`. It reveals growth-ring stratification invisible in height colouring, and thresholding on it (keep burst-age ≤ 0.25) strips late infill, leaving hollow canopies.

### Visualising monotone CAs

Life Without Death never deletes, so its space-time set is exactly `{(x,y,t) : t ≥ birth(x,y)}` — a heightfield, not metaphorically. Record the **birth-time field** and treat it as a scalar landscape. Use height = `(T − birth)` so early arrivals stand tall; the inverse orientation buries the interesting features below the surrounding plain.

LWD does **not** fill to all-1s, despite the common claim. It fixates at 65–81% with \~9,900 permanent hole components, because a dead cell that reaches 4+ live neighbours can never be born. Sparse soups run much longer (639 steps at density 0.005 versus 5 steps at 0.6).

## Open threads

Ranked by expected payoff.

1. **Momentum-conserving collisions in the lattice gas.** The current collision step randomises channels, destroying any directional bias before it can accumulate. A proper HPP/FHP rule, where head-on pairs rotate rather than randomise, should preserve bias and give directional dendrites. This is the clearest path to anisotropy from a construction already producing the highest coherence in the project.
2. **Different rules per seed in the multi-seed hybrid.** All five cones currently point the same way because every seed grows isotropically under the same rule. Giving seeds different birth conditions would break that symmetry far more than position and timing did.
3. **Tri-planar as the micro layer in a macro/micro composition.** Tri-planar alone is a texture generator, not a scene generator — correlation length 1 on every axis. But it is 25× faster than the λ-ramped volume it would replace inside the hangar construction, where the coarse majority-smoothing pass supplies the macro form.
4. **Anisotropic neighbourhoods instead of anisotropic construction.** `box_count_aniso` already supports per-axis radii (radius 8 in x, 1 in y, 3 in z). Directional structure falls out natively at the same O(1)-per-cell cost, skipping the slab machinery entirely. Untested.
5. **Second-order / reversible CAs.** `s[t+1] = f(s[t]) XOR s[t-1]` cannot settle into a uniform attractor, which fixes the saturation problem behind failure modes 2 and 3. Same cost as a normal step plus one XOR. Untested.
6. **Ambient occlusion in the renderer.** A few short rays per hit point. Expensive but transformative on voxel work — currently the shading is one shadow ray plus a flat ambient term, which flattens interior detail.
7. **A better search criterion.** The original selection heuristics (compressibility, damage spreading, Langton's λ) were developed for raw 1D rules and stop discriminating once a rule is wrapped in octave blending. Part count works, but something better aimed at *scene quality* rather than *rule class* would be more useful.

### Not worth revisiting

- Stochastic layer bleed. It was a regression on every measure; injecting noise each step destroys the clean fronts that produce the good structures. The one salvageable finding is that a very low bleed probability (\~0.05) acts as a growth-rate throttle, extending a run from 16 steps to 143.
- Hidden refractory states as an information channel. Ablated to zero effect.
- Tri-planar as a standalone scene generator.
