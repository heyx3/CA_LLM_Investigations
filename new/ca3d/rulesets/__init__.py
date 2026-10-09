"""The cellular automata: rule tables, CA engines, rule families and the volume
constructions built from them.

Suggested reading order: rules, wolfram, life, hierarchy, cityscape.  Terms are
explained in docs/CONCEPTS.md.

Dependencies: most modules here are self-contained.  `families` (and so `cityscape`)
and `saved` also use `ca3d.analysis`, because a rule family is defined by how a rule
measures when run, not by its table.  `analysis` in turn builds on this package's
rule engines, so the two packages depend on each other in that one place.

  rules             rule-table operators: lambda, flips, biased assignment
  wolfram           1D binary CAs of any radius
  life              2D B/S and 512-entry Moore rules, expansion, rule rotation
  hierarchy         the hierarchical (multi-octave) CA engine and rotation policies
  families          dead / static / slow / edge / complex rule pools
  cityscape         the cityscape configuration and its measured profile
  initial           initial conditions: blobs, uniform, sparse points, rings, ...
  saved             a catalogue of rules and configurations worth keeping
  double_spacetime  1D CA extruded to 3D with per-slab rules; lambda terrain
  triplanar         three 1D sheets combined on orthogonal planes
  octaves           octave blending of any volume builder
  lwd               Life without Death birth fields and the LWD / GoL hybrid
  lattice_gas       lattice-gas DLA
  deposition        box sums, frozen deposition, macro/micro hangar
"""
