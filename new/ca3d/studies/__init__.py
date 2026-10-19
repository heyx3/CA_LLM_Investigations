"""Particular CAs analysed in depth, and what was found: the conclusions of measuring
and searching, as opposed to the CAs themselves (ca3d.rulesets) and the general tools
(ca3d.analysis), both of which this package builds on.  A module specific to one kind
of CA is named after it.

  moore_families     families of 512-entry Moore rules (dead, static, complex, slow, ...),
                     their pools and traits
  hierarchy_designs  hierarchical CAs described by Moore families (Design) and one rule
                     draw of them (Instance)
  cityscape          the reference cityscape, its design and profile, and the bands
                     other hierarchies are judged by
  line_volumes       what the 1D-grown volumes share: the 1D rule and start (Sheet), the
                     random-rule screen, and the measures and bands cubes are judged by
  double_spacetime   double space-time designs, their survey, and the designs picked
  triplanar          triplanar designs, their survey, relief over combiner tables, and
                     the designs picked
  octave_mixes       octave blends of those designs: plans, pools, the structure and
                     texture presets, and the blends made so far
  deposition_shells  frozen deposition chosen and coloured by freeze time (Pick), the
                     shell searches and the configurations picked, and shapes for
                     structured starts
  saved              a catalogue of rules and configurations worth keeping
  experiments        named, re-runnable experiments, each asking one question
  scenes             named, ready-to-render recipes for every construction (and saved
                     finds), and hierarchy_scene for colouring any hierarchy's run
"""
