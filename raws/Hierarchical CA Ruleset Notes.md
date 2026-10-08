# Hierarchical CA: Ruleset Notes

Oct 4, 2026 · @Billy

## Architecture

Four binary layers at halving resolution. Layer 0 is the finest and is the only one rendered as solid; layers 1 to 3 exist purely to supply context.

Each layer's rule reads two things: its own Moore neighbourhood, and one bit from **every** coarser layer, upsampled to its resolution. Direct wiring, not a chain — this matters, see below.

| layer | width | updates on | table size (non-totalistic) |
| --- | --- | --- | --- |
| 0 (finest, rendered) | n | every step | 512 × 8 = 4096 |
| 1 | n/2 | t mod 2 == 1 | 512 × 4 = 2048 |
| 2 | n/4 | t mod 4 == 2 | 512 × 2 = 1024 |
| 3 (coarsest) | n/8 | t mod 8 == 3 | 512 |

Table index is `(own_state × 256 + neighbour_byte) × 2^parents + context`, where the neighbour byte packs the 8 Moore neighbours in a fixed bit order.

### Two design decisions that were load-bearing

**All-parents wiring, not a chain.** An earlier version had each layer read only its immediate parent. Measured end-to-end, layer 3 then had *zero* influence on layer 0 — pinning it changed 0.000–0.020 of voxels. Each individual link was strong, but the parent bit was being discarded in transit: the tables were nominally parent-sensitive (0.62 uniform sensitivity) yet effectively parent-blind (0.008 weighted by actual neighbourhood traffic), because CA dynamics visit only a handful of neighbourhood patterns and those happened to map identically for both parent states. Direct wiring removes the problem structurally — every layer now registers 0.33–0.50 on pinning.

**Staggered update phases.** Layer *i* fires when `t mod 2^i == i`, not `== 2^i - 1`. The aligned schedule had all four layers updating simultaneously every 8th step, producing visible horizontal banding in the space-time render. Staggering drops max simultaneous updates from 4 to 3 and removes the artifact. It does *not* improve temporal coherence — change rate actually rose slightly, 0.0367 to 0.0422.

## Rule families

The fine layer assigns a *different rule family* to each of the 8 contexts. That mix is what produces structure rather than uniform texture.

| family | criterion | pool size | contributes |
| --- | --- | --- | --- |
| **dead** | density → 0 from a dense start | 156 | empty sky; carves voids |
| **static** | frozen (change/step < 0.0008), density 0.15–0.85 | 71 | pillars — a frozen 2D pattern extruded through time is a prism |
| **complex** | damage spreading in 0.02–0.25, density 0.03–0.22 | 35 | ragged texture, horizontal streaks |
| **slow/compact** (coarse layers) | compactness ≥ 0.4 *and* change/step 0.0005–0.05 | 4 → 371 after expansion | the territories |

### Finding them

**Dead and static rules must be searched from a dense start** (p0 = 0.85), not from p0 = 0.5. They are defined by what they do to existing material, and a sparse start never engages them.

**Complex rules come from damage spreading**, the standard ordered/chaotic/complex discriminator. Of 5,252 density-viable rules: 271 ordered, 4,356 chaotic, **249 in the partial-spreading band**. Under 5%.

**Slow-compact coarse rules are nearly nonexistent in totalistic space.** Compactness is spatial and change rate is temporal, and the two are close to mutually exclusive:

| compactness | <0.001 | .001–.01 | .01–.1 | .1–.4 | >0.4 |
| --- | --- | --- | --- | --- | --- |
| 0.0–0.2 | 63 | 161 | 307 | 1118 | 3470 |
| 0.4–0.6 | 4 | 3 | 1 | 10 | 16 |
| 0.8–1.0 | 0 | 0 | 1 | 0 | 6 |

The first threshold tried (compactness ≥ 0.55, change 0.05–3%) matched **0 of 9000** rules. Relaxing to ≥ 0.4 found exactly 4.

### Going non-totalistic

Direct search is hopeless — 2^512 per context, when only 4 of 6000 worked in the 18-bit totalistic space. Instead: **expand a known-good B/S rule to its full 512-entry table** (every neighbour pattern inherits the output for its popcount, so the expansion is exactly equivalent), then flip *k* entries.

| k flips | coarse rules kept (of 240) | compactness | anisotropy |
| --- | --- | --- | --- |
| 0 | 240 | 0.52 | 0.0117 |
| 4 | 80 | 0.58 | 0.0161 |
| 12 | 39 | 0.55 | 0.0257 |
| 32 | 12 | 0.51 | 0.0400 |
| 80 | 0 | — | — |

Anisotropy more than triples while compactness holds, then everything dies past 32 flips. Totalistic rules are isotropic by construction; only arrangement-sensitivity can break that symmetry. This took the coarse pool from 4 usable rules to 371, which is what lets different territories actually look different.

## The cityscape configuration

Seed 3, 160×160 grid, 160 timesteps, 4 layers. Density 0.196, void 80.2%, pillar fraction 0.46.

**Context plan for the fine layer** — this is the central dial:

```
['dead','dead','static','static','static','static','complex','complex']
```

Contexts 0–1 die out, 2–5 freeze into pillars, 6–7 run complex. Ratio effects:

| plan | density | pillar fraction | largest void |
| --- | --- | --- | --- |
| 2 dead / 4 static / 2 complex | 0.196 | 0.46 | 80.2% |
| 4 dead / 2 static / 2 complex | 0.227 | 0.45 | 77.2% |
| 3 dead / 2 static / 3 complex | 0.132 | 0.33 | 86.5% |

More static → more towers. More dead → more sky. More complex → more shaggy texture and more horizontal streaks.

**Patchy initial condition, not uniform noise.** Gaussian-smoothed low-resolution field thresholded at the 55th percentile and upsampled 8×, giving blobs that are \~100% alive with empty space between. This is essential: dying rules need material to erode. Seeded from uniform random noise they never engage, which is why earlier sparse attempts thinned everything uniformly instead of carving sky between structures.

### The actual rules

All coarse-layer rules derive from these four, the entire slow-compact pool found in totalistic space:

```
slow[0] = B356/S5678
slow[1] = B037/S245678
slow[2] = B5/S234678
slow[3] = B578/S1235678
```

All are high-survival, which is why they coarsen and persist. `B5/S234678` is closest to the classic cave-generation rule. Each is expanded to 512 entries and perturbed by k=4 flips before use.

For reference, an earlier totalistic configuration (render `sg_0`) used layer 3 = `B578/S1235678`, layer 2 = `B5/S234678` for both contexts, layer 1 cycling `B356/S5678, B578/S1235678, B5/S234678, B578/S1235678`, and layer 0 running eight edge-of-chaos rules: `B012458/S134568`, `B15/S012378`, `B05/S0268`, `B3/S245678`, `B058/S12458`, `B3456/S2567`, `B458/S24567`, `B06/S15`.

## What each layer contributes

Measured by rendering each layer's space-time volume separately, and by pinning each layer and comparing the final geometry.

| layer | density | changes/step | pillar fraction | median blob | pinning changes |
| --- | --- | --- | --- | --- | --- |
| 0 (rendered) | 0.196 | 0.0515 | 0.46 | 1 | — |
| 1 | 0.099 | 0.0265 | 0.40 | 8 | 0.345 |
| 2 | 0.186 | 0.0110 | 0.60 | 32 | 0.353 |
| 3 | 0.695 | 0.0150 | 0.79 | 64 | 0.209 |

**Layer 2 supplies the silhouette.** Rendered alone it is already a tower field — clean prisms with sharp tops, pillar fraction 0.60. The towers in the final image are substantially layer 2's geometry with layer 0's surface detail applied. This was the biggest surprise from the per-layer renders.

**Layer 3 gates the rule vocabulary rather than making geometry.** It is the lowest bit of the context index, so it halves the palette available everywhere:

|  | reachable contexts |
| --- | --- |
| layer3 = 0 (31% of volume) | 0, 2, 4, 6 |
| layer3 = 1 (69% of volume) | 1, 3, 5, 7 |

Within layer3=0 regions, 65% of volume runs context 0; within layer3=1, 78% runs context 1 — both dead rules. So layer 3 chooses which of two parallel rule-sets everything below draws from. It looks uncorrelated with the visible structure because it selects vocabulary, not form.

**Layers 1 and 2 act as suppressors.** Pinning either roughly *doubles* density, 0.209 → 0.404 and 0.396. Their churn repeatedly knocks cells back out. Consequence worth knowing: making layer 1 quieter gives a denser, blockier result, so any reduction in its noise needs compensating with more dead contexts.

**Pillar fractions show the division of labour.** Vertical coherence originates in the coarse layers (0.79 and 0.60) and layer 0 inherits a reduced 0.46. The fine layer is decorating structure handed down, not generating it.

### Horizontal streaks

Isolated morphologically — long in x or y, short in time — they are 2.9% of solid voxels. Enrichment relative to each context's share of the volume:

| ctx | family | enrichment |
| --- | --- | --- |
| 4 | static | 3.52× |
| 6 | complex | 3.30× |
| 7 | complex | 2.75× |
| 0 | dead | 2.30× |
| 3 | static | 0.24× |

Only 11.6% of streak mass falls on layer-2 update steps, which are 25% of all steps — so they are *anti*-correlated with the update schedule and are not a timing artifact. They come from the two complex contexts plus one static rule that locks in laterally-extended patterns.

## Search criteria

### What works

| measure | definition | use |
| --- | --- | --- |
| compactness | fraction of cells whose entire 3×3 neighbourhood matches them | spatial coherence; 0.12 is speckle, 0.97 is blobs |
| change/step | fraction of cells flipping per update | temporal persistence; needed *alongside* compactness |
| damage spreading | one-cell flip, two copies, measure divergence | finds the complex band between ordered and chaotic |
| anisotropy | abs difference of correlation at lag 3 horizontally vs vertically | detects broken rotational symmetry; 0 for any totalistic rule |
| pillar fraction | live voxels in a vertical run of 16+ unchanged steps | how much of the output is extruded static pattern |
| effective sensitivity | parent-bit sensitivity weighted by actual neighbourhood frequency | whether conditioning is real or nominal |

### Four metrics that misled

Every one was a mean or an aggregate over a heavy-tailed or one-sided distribution.

**Mean blob size.** Total live cells ÷ component count. A single percolating component inflates it without bound. Layer 1 read 1,275 cells mean blob while its *median* was 4 and one component held 99.4% of the mass — it was speckle, and the search had been selecting for speckle. Fixed by using median size, percolation share, and correlation length.

**Compactness optimised to its extreme.** Thresholding at ≥0.55 found rules at 0.97, meaning almost every cell's neighbourhood matches it — the definition of a boring rule. Both ends of this scalar are degenerate: 0.12 is noise, 0.97 is blobs. Interesting rules sit in a band, and the band is always small.

**Compactness without temporal persistence.** Purely spatial. Rules can make big compact blobs that completely rearrange every step. Context changed in 48% of voxel-steps while I was describing the output as static columns.

**Uniform parent sensitivity.** Counting how many of the 8 neighbourhood patterns respond to the parent bit overstates influence badly, because dynamics concentrate on a few patterns. Uniform 0.62, effective 0.008.

### The general lesson

Optimising any single scalar to its extreme gives degenerate output in both directions. Every useful family here was found by a *band* on one measure plus a second, independent measure — compactness **and** change rate, damage spreading **and** density. Single-criterion searches failed four times.

## Knobs and open leads

### Steering the output

| knob | effect |
| --- | --- |
| context plan (dead/static/complex mix) | the main dial — towers vs sky vs texture |
| perturbation k (4–32) | anisotropy; past 32 everything dies |
| initial-condition blob scale and fraction | size and spacing of the structures |
| coarse update periods | how much territories migrate within the window |
| number of layers | each layer added doubles the context count and halves the coarsest resolution |

### Colour

The renderer takes a per-voxel palette index plus an RGB palette, so colour can carry several independent channels. Hue from layer 3, value from layer 2, saturation from layer 1 works well — layer 0 cannot carry colour since it *is* the solid.

Binary layers give only 2×2×2 = 8 combinations, so box-smooth each layer before quantising to get continuous channels. Radius 5 on a 160³ volume uses all 96 palette slots; radius 11 oversmooths and collapses hue toward uniform. The smoothing radius should scale with volume size.

### Open leads

1. **Push the streaks.** They enrich 2.75–3.52× in complex contexts. A plan weighted to four or five complex contexts should make them the dominant motif rather than an accent.
2. **Towers that start partway up.** Currently all towers rise from the ground plane, because the initial condition is laid at t=0 and static contexts lock it immediately. Towers beginning mid-air need the context map to change during the run — the coarse layers move too slowly for that in 160 steps.
3. **Widen the slow-compact search.** Larger neighbourhoods or a smarter search than random sampling. The totalistic pool was 4 rules; everything coarse descends from those four.
4. **Sparser than \~0.3 needs a different approach.** Of 742 complex non-totalistic rules, zero had density below 0.05 and one was between 0.05 and 0.12. Complex behaviour and low density are close to mutually exclusive here. Rendering births instead of live cells gives 0.09–0.19 and shows activity rather than occupancy.
5. **Quieter layer 1** would reduce noise but roughly doubles density — needs compensating with more dead contexts.
