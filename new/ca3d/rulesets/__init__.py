"""The cellular automata: rule tables, CA engines and the volume constructions built
from them.

Suggested reading order: rules, wolfram, life, hierarchy.  Terms are explained in
docs/CONCEPTS.md.

Nothing here depends on the rest of ca3d, or on any analysis of the CAs: what a rule
does when run (its family, say) belongs in ca3d.studies.

  rules             rule-table operators: lambda, flips, biased assignment
  wolfram           1D binary CAs of any radius
  life              2D B/S and 512-entry Moore rules, expansion, rule rotation
  hierarchy         the hierarchical (multi-octave) CA engine and rotation policies
  initial           initial conditions: blobs, uniform, sparse points, rings, ...
  double_spacetime  1D CA extruded to 3D with per-slab rules; lambda terrain
  triplanar         three 1D sheets combined on orthogonal planes
  octaves           octave blending of any volume builder
  lwd               Life without Death birth fields and the LWD / GoL hybrid
  lattice_gas       lattice-gas DLA
  deposition        box sums, frozen deposition, macro/micro hangar
"""
