# Quantifying Cellular Automaton Output

Oct 4, 2026 · @Billy

## What this is for

Automatic quantification lets you search rule space without looking at every result. A rule space of any size is unviewable by hand — 2^18 for a life-like rule, 2^512 for a non-totalistic one — and the interesting fraction is always small. Measured hit rates from this work:

- 13 of 256 elementary rules make usable coarse layers (5%)
- 249 of 5,252 life-like rules sit in the complex band (4.7%)
- 4 of 6,000 are both spatially compact and temporally slow (0.07%)
- 64 of 8,000 are genuinely compact rather than percolating (0.8%)

### The one principle that matters

**Interesting output lives in a band, never at an extreme of any scalar.** Every measure below has two degenerate ends, and optimising toward either produces something boring:

| measure | low extreme | high extreme |
| --- | --- | --- |
| compactness | salt-and-pepper speckle | featureless blobs |
| damage spreading | frozen | indistinguishable from noise |
| change rate | static image | uncorrelated churn |
| density | nothing to see | opaque solid |
| pillar fraction | no persistent structure | extruded prisms, no life |

This sounds obvious and is still the most common failure. A search that maximises a scalar will find its extreme, and the extreme is degenerate by construction. Four separate searches in this project failed exactly this way.

### Two corollaries

**Use a band on one measure plus a second, independent measure.** Compactness *and* change rate. Damage spreading *and* density. A single criterion cannot distinguish "structured" from "uniform", because uniform scores perfectly on most structural measures.

**Target a profile, not a maximum.** If you have an output you like, measure it and search for rules landing near those values. In this project the preferred output sat at pillar fraction 0.36 and temporal correlation 16 — the middle of the observed ranges — while maximising either produced results the author rated worse.

## Regime measures

These place a rule on the ordered–complex–chaotic axis. They are the first filter: cheap, general, and they throw out the overwhelming majority of rule space.

&#91;embedded content: regime measures · 5 measures across 3 regimes\]

### Damage spreading

The most reliable of the set. Run two copies of the same rule from identical initial conditions, flip one cell in one copy, and measure how far the difference has spread after N steps.

Ordered rules absorb the perturbation (≈0). Chaotic rules let it fill the lattice — for a binary 2D system that saturates near 0.5, since two uncorrelated configurations differ in half their cells. Complex rules spread partially and intermittently.

This is the discrete analogue of a Lyapunov exponent, and it is the only measure here that directly tests the thing you care about: whether information propagates without being destroyed. Measured distribution over 5,252 density-viable life-like rules: 271 ordered, 4,356 chaotic, 249 in the 0.02–0.25 band.

Two practical notes. Compare the plateau against the *random baseline* for your system, not against 1.0 — for a ring of N binary cells the baseline is N/2, and for v states it is N(v−1)/v. And perturb at a single cell, not a region; region perturbations saturate too fast to discriminate.

### Compressibility

Gzip the space-time diagram and take the compressed/raw ratio. Near-total compression means order; near-zero compression means the output is indistinguishable from noise. The middle compresses partially.

Cheap and surprisingly effective as a first pass. Its weakness is that it conflates *spatial* and *temporal* redundancy — a static image and a periodic blinker both compress well for different reasons.

### Langton's λ

The fraction of rule-table entries producing a non-quiescent output. Not a measure of behaviour but of the rule itself, so it costs nothing to compute and can bias *sampling* rather than filter results.

Useful for steering a search toward the critical region, and useful as a construction tool — forcing a rule to a target λ is a clean way to generate a controlled gradient of behaviour. Unreliable as a predictor for any individual rule: λ correlates with regime only statistically.

### Entropy and density

Shannon entropy of cell states, or of local block patterns, over time. Mostly subsumed by the measures above, but worth computing because **density is a necessary gate for everything else**. A rule at density 0.99 or 0.01 will score misleadingly on every structural measure. Filter to a viable density window first — 0.15 to 0.85 is a reasonable default — then apply the regime measures.

## Spatial structure

Regime measures tell you a rule is not boring. These tell you what shape its output takes.

### Compactness

Fraction of cells whose entire 3×3 neighbourhood matches their own state. Cheap, local, and the best single indicator of whether output reads as *regions* or as *speckle*.

| compactness | appearance |
| --- | --- |
| under 0.2 | salt-and-pepper |
| 0.4 to 0.7 | structured regions with detail |
| over 0.9 | featureless blobs |

### Connected components

Label the live cells and look at the distribution of component sizes. **The single most discriminating measure found in this work**, and the one that kept working when others saturated.

Use three numbers, not one:

- **count** — how fragmented the output is. In one comparison this ranged 51 to 13,222 across configurations where coherence varied by only 3%.
- **median size** — resistant to percolation.
- **largest component as a share of live cells** — detects percolation directly.

A percolating structure has one component holding 97–99% of the mass and a median size of 1. Mean component size is then meaningless; see the failure catalogue.

### Correlation length

Smallest lag at which spatial autocorrelation drops below 1/e. Compute it **per axis** — differing values across axes is the cleanest detector of anisotropy, and a value of 1 on every axis means the output decorrelates after a single cell regardless of how structured it looks.

The per-axis version also catches a specific failure: tri-planar and similar constructions can produce visually busy output with correlation length 1 in all three directions, which is noise with extra steps.

### Fractal dimension

Box-counting slope. Coarsen the volume by factors of 2, 4, 8, 16, count occupied boxes at each scale, fit a line to log(count) against log(1/scale).

3.0 is space-filling in 3D; \~2.5 is diffusion-limited aggregation; lower is wispier. Directly measures whether structure is self-similar across scales rather than existing at one scale only. Useful as a *target* rather than a filter — pick the dimension you want and search for it.

### Anisotropy

Absolute difference between horizontal and vertical autocorrelation at a fixed lag. **Exactly zero for any totalistic rule**, since those depend only on neighbour counts and are isotropic by construction. Only arrangement-sensitive (non-totalistic) rules can register here.

This makes it a useful verification measure: if you have expanded a totalistic rule into non-totalistic space and anisotropy is still \~0, the expansion did nothing.

## Temporal structure

The most commonly skipped category, and the source of the worst errors. **Every spatial measure above can be satisfied by output that completely rearranges itself each step.** A rule can produce large compact blobs that bear no relation to the previous frame's blobs.

### Change rate

Fraction of cells flipping per update, measured after a burn-in. The simplest temporal measure and the one to pair with any spatial measure.

| change/step | behaviour |
| --- | --- |
| under 0.001 | frozen — useful for pillars, useless for dynamics |
| 0.001 to 0.05 | slow evolution, structure persists |
| 0.1 to 0.4 | active |
| over 0.4 | churn; no frame resembles the last |

The joint distribution with compactness is severely one-sided. Measured over 6,000 life-like rules, the cell for compactness ≥ 0.4 *and* change rate in 0.001–0.05 held **7 rules**; compactness under 0.2 with change over 0.4 held **3,470**. Spatially compact and temporally slow are close to mutually exclusive, which is exactly why you must test both.

### Temporal correlation length

Correlation length computed along the time axis. Reads how many steps structure survives.

The most useful diagnostic for layered or multi-scale systems, because it catches over-churn: in one case a four-layer stack read 3 steps while any three-layer version of the same system read 28–40. Nothing can persist when too many conditioning layers update on independent schedules.

### Pillar fraction

Fraction of live cells sitting inside a vertical run of N or more unchanged steps (N=16 is a reasonable default). A space-time-specific measure: it quantifies how much of a volume is *extruded static pattern* rather than evolving structure.

Useful as a target rather than a maximum. Pillar fraction near 0 means nothing persists; near 0.7 means the volume is prisms with no life in it.

### Overhang fraction

Fraction of live voxels with empty space directly below them in the time axis. **Exactly 0 for any monotone process**, because a monotone rule never deletes, so its space-time set is a heightfield.

This makes it a binary test: does the system produce genuine 3D structure, or merely a terrain? Values of 0.002–0.005 indicate real overhangs; 0 means you have a heightfield however complicated it looks.

## Morphological extraction

The measures above score a whole output. These isolate a *specific motif* so you can quantify how much of it is present — and, once isolated, trace it back to the rules that made it.

### Opening with structuring elements

Binary opening with a shape keeps only the parts of the output that can contain that shape. A line element of length L keeps runs of at least L; a square element keeps bulk.

The critical point is that **a single opening is almost always the wrong test**. "Long" is satisfied by any dense region. The useful test is a difference:

```
thin_linear = open(line_L) AND NOT open(square_S)
```

Long in one axis *and* thin in the other. The naive version — fraction of cells in runs of 6 or more — read 0.890 on output that was visually a solid blob, because any dense region contains long runs. The corrected version read 0.704 on genuinely filamentary output and 0.094 on bulk.

### Isolating a motif you like

The general recipe, used to trace horizontal streaks back to their source:

1. Express the motif as a morphological difference — long in x or y, short in time.
2. Measure its share of live cells.
3. Compute its share *per context, per rule, or per region*.
4. Divide by that context's share of the whole volume to get **enrichment**.

Enrichment above 1 means the motif concentrates there. Measured values of 2.3× to 3.5× identified exactly which rule families produced the feature, out of eight candidates.

### Checking against confounds

Once you have a motif isolated, test it against the mechanisms that could produce it artificially. For a periodic system the obvious one is the update schedule: compute what fraction of the motif's mass falls on steps where a layer updates, and compare against that fraction of all steps. In the streak case, 11.6% of mass fell on steps that were 25% of the timeline — *anti*-correlated, so not a timing artifact.

### Domain filtering

For 1D systems with a periodic background, XOR the configuration against a shifted copy matching the background's period and slope. Where the background holds, the result is 0; defects survive. This is the standard way to expose particles in a space-time diagram.

It only works when there *is* a clean periodic background. On output where mutation has destroyed the background, every shift direction tested gave 0.40 density or worse — no better than the unfiltered volume.

## Coupling and influence

For systems where one part is supposed to condition another — layered CAs, extra states, external fields — the question is whether the coupling is real or decorative. These answer it.

### Pinning

Freeze the component under test at its initial value, rerun with everything else identical, and measure how much the output differs. The bluntest and most informative test.

A component with no effect reads \~0. In one case layers 2 and 3 of a four-layer stack read 0.001 and 0.000 — they were doing nothing at all, and the architecture was effectively two layers with decoration.

Pinning is a strong intervention: it changes the component's *statistics*, not just its pattern, so a nonzero result confirms influence without quantifying it precisely. Pair it with perturbation.

### Single-cell perturbation

Flip one cell of the component at t=0, rerun, measure divergence in the output. The targeted version of damage spreading applied across components rather than within one.

More demanding than pinning and a better measure of whether information actually propagates. Values of 0.012–0.128 confirm a live channel; 0.000 confirms a severed one.

### Effective sensitivity

The subtle one, and the source of the most instructive failure in this work.

When a rule table takes an extra input, you can count how many table entries respond to that input — flip the input bit and see whether the output changes. Call this **uniform sensitivity**. It is almost always wrong.

CA dynamics do not visit neighbourhood patterns uniformly. In a measured case, three patterns carried **99% of the traffic**. Weight the sensitivity by observed pattern frequency:

```
effective = sum over patterns of  freq(p) × [table(p, 0) != table(p, 1)]
```

Measured on the same table: uniform 0.62, **effective 0.008**. The extra input was formally present and practically ignored, which is why the pinning test showed a severed chain despite tables that looked well-coupled.

Two uses. As a **diagnostic**, run a short pilot, collect the pattern histogram, and compute effective sensitivity before trusting any conditioned architecture. As a **construction constraint**, force `table(p, 1) = NOT table(p, 0)` to guarantee effective sensitivity 1.0 regardless of traffic — though that is blunt, since it makes the conditioned output the exact inverse rather than a different regime.

### Ablation

Remove a component entirely and compare. Cleaner than any influence metric because it tests necessity rather than correlation.

One caution: removing a layer from a hierarchy usually changes more than one thing at once — in the measured case it also halved the context count, so the comparison confounded architecture with rule-assignment. State the confound rather than pretending the ablation is clean.

## How these measures fail

Six failures, all observed, all of which produced plausible numbers and wrong conclusions. They fall into three recurring shapes.

### Shape 1: a mean over a heavy-tailed distribution

**Mean component size.** Total live cells ÷ component count. One percolating component inflates it without bound. Measured: mean 1,275 cells, **median 4**, largest component holding **99.4%** of mass. The output was speckle and the search had been selecting for speckle for several rounds.

*Fix:* report median and largest-share alongside any mean. If largest-share exceeds \~0.9, the mean is meaningless.

**Uniform parent sensitivity.** Counting responsive table entries treats all neighbourhood patterns as equally likely. They are not — three patterns carried 99% of traffic. Uniform 0.62, effective 0.008.

*Fix:* weight by observed frequency from a pilot run.

### Shape 2: a scalar optimised to its extreme

**Compactness.** Thresholding at ≥0.55 found rules at 0.97 — almost every cell's neighbourhood matching it, which is the definition of featureless. Both ends of this scalar are degenerate.

**Pillar fraction and temporal correlation.** Maximising these produced sparse spire forests the author rated clearly worse than output sitting mid-range on both.

*Fix:* band, not maximum. If you have a reference output you like, measure it and target that profile.

### Shape 3: measuring one dimension and assuming the other

**Compactness without temporal persistence.** Purely spatial. Rules make big compact blobs that rearrange completely each step. While describing output as static columns, context was in fact changing in **48% of voxel-steps**.

**A shape metric that any dense region satisfies.** "Fraction of cells in runs of 6 or more" read **0.890** at baseline on visually solid output, because any dense region contains long runs. It made every candidate look equally good.

*Fix:* require two properties that trade off — long **and** thin, compact **and** slow. A single-sided test is satisfied by the trivial case.

### The general diagnostic

Every one of these was caught by a control, never by looking at output:

- The same metric returning *identical* values for configurations that should differ. Shadow coverage reading exactly 1.0% for two different light directions; three different base rules giving byte-identical statistics. **Suspiciously stable numbers are the strongest bug signal available.**
- A baseline measurement. Coherence of 0.890 means nothing until you know random noise reads 0.85.
- A known-answer test scene before trusting a measurement pipeline on real data.

## Putting it together

### A staged filter

Order by cost. Each stage removes most of what reaches it, so the expensive measures only ever see a small remainder.

| stage | measure | typical pass rate |
| --- | --- | --- |
| 1 | density window (0.15–0.85) | \~85% |
| 2 | damage spreading in band | \~5% |
| 3 | change rate in band | varies |
| 4 | compactness or correlation length in band | \~1% |
| 5 | component count / median size | — |
| 6 | motif enrichment, if targeting one | — |

Stages 1–3 run on a small lattice with a short burn-in and cost almost nothing. Stage 4 onward is where you can afford a full-size run.

### Pick the right initial condition

A rule is only measurable under conditions that engage it. **Rules that erode need something to erode**: searching dying rules from p0 = 0.5 finds nothing, because they never start. Search from a dense start (p0 ≈ 0.85) when looking for dead or static families.

Similarly, nucleation failures masquerade as dead rules. If your birth threshold is B and your neighbourhood has N cells, a random start at density p gives mean neighbour count N·p — if that is far below B, every rule will read as dead regardless of its behaviour. Several results logged as "the rule dies" were this.

### Expect large variance, and measure it

Run every configuration across several seeds and report the spread. In one sweep the standard deviation across five rule draws was ±0.07–0.15 on pillar fraction — **comparable to the entire effect of changing the architecture**. A single-seed comparison would have attributed draw noise to design.

This also means a negative result from one seed is not a result.

### When a search space is too large

Random sampling fails once the space is large enough: 4 of 6,000 in an 18-bit space becomes hopeless at 512 bits. Two ways through:

**Expand and perturb.** Take a rule known to work in a smaller space, expand it losslessly into the larger one, then flip *k* entries. Perturbation strength becomes a tunable dial with a measurable ceiling — in one case anisotropy tripled from k=0 to k=32 while structure held, and everything died past k=80.

**Construct rather than search.** For properties with a direct expression — λ, parent sensitivity, a target density — build the rule to satisfy it instead of sampling until you find one. This is reliable but blunt; constructed rules tend to hit the property exactly and so land at an extreme.

### A note on what these measures cannot do

Everything here quantifies *structure*, not interest. In this work the configuration preferred by eye consistently sat mid-range on every scalar, and each attempt to optimise a measure moved away from it.

The honest use is as a **filter and a navigator**: cut 95% of rule space, characterise what survives, and target the profile of something you already know you like. Treating any of these as an objective function will reliably find its degenerate extreme.
