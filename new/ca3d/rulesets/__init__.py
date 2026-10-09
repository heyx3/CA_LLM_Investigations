"""The cellular automata: rule tables, CA engines, rule families and the volume
constructions built from them.

  rules             rule-table operators: lambda, flips, biased assignment
  wolfram           1D binary CAs of any radius
  life              2D B/S and 512-entry Moore rules, expansion, D4 rotation
  hierarchy         the hierarchical (multi-octave) CA engine and rotation policies
  families          dead / static / slow / edge / complex rule pools
  cityscape         the cityscape configuration and its recorded profile
  initial           initial conditions: blobs, uniform, sparse points, rings, ... (seeds.py)
  saved             a catalogue of rules and configurations worth keeping
  double_spacetime  1D CA extruded to 3D with per-slab rules; lambda terrain
  triplanar         three 1D sheets combined on orthogonal planes
  octaves           octave blending of any volume builder
  lwd               Life without Death birth fields and the LWD / GoL hybrid
  lattice_gas       lattice-gas DLA
  deposition        box sums, frozen deposition, macro/micro hangar
"""
