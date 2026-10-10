"""ca3d: cellular-automaton volume generators and a voxel renderer.

A cellular automaton (CA) is a grid of cells, each in one of a few states, updated in
lock step by a rule that looks only at a cell's neighbourhood.  Stacking a 2D CA's
successive states, or a 1D CA's rows, gives a 3D "space-time" volume, which this
package measures, searches for and renders as voxel scenes.  New to the vocabulary
(lambda, B/S notation, soup, damage spreading, context...)?  Start with
docs/CONCEPTS.md.

Four sub-packages, each importing only the ones above it:

  rulesets   the cellular automata: rule tables, CA engines and the volume constructions
  analysis   measuring output (metrics) and searching for rules (dynamics, search,
             banks, influence): general tools for any CA
  studies    particular CAs analysed in depth: the Moore-rule families, hierarchy
             designs, the cityscape, a catalogue of saved finds and re-runnable
             experiments
  render3D   the voxel renderer, colouring, contact sheets, and named scenes ready to
             render (the scenes draw on studies)

    from ca3d.studies import cityscape
    from ca3d.render3D import render
    fine = cityscape.make().run(160).fine          # the reference cityscape
    image = render.render(fine)

Conventions shared by every module:

* Cell states are binary uint8 (or bool) arrays unless stated otherwise.
* A rule table is a 1D uint8 array; entry i is the output for neighbourhood pattern i.
* A volume is a boolean occupancy array, normally indexed [x, y, z] with z pointing
  up on screen.  For space-time constructions the last axis is time and the other
  axes are lattice axes, so such a volume may be labelled [y, x, t] or
  [slab, x, t]: each generator's docstring says which.  Generators that make several
  states (frozen_deposition) return an integer array instead; compare with
  `== FROZEN` to get the occupancy.
* Randomness always comes from an explicit numpy Generator or integer seed.
* Docstrings state array shapes and dtypes; "batch" means a leading axis holding many
  independent samples (lattices, or rules each with its own lattice).
"""
