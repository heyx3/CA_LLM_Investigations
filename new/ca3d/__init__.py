"""ca3d: cellular-automaton volume generators and a voxel renderer.

Reconstructed from the exploratory scripts in ../raws.  Three sub-packages:

  rulesets   the cellular automata: rule tables, CA engines, rule families, the volume
             constructions, the cityscape and a catalogue of saved finds
  analysis   measuring output (metrics) and searching for rules (dynamics, search,
             influence), plus the original analyses as experiments
  render3D   the voxel renderer, colouring, and named scenes ready to render

    from ca3d.rulesets import cityscape
    from ca3d.render3D import render
    fine = cityscape.make(seed=3).run(160).fine
    image = render.render(fine)

Conventions shared by every module:

* Cell states are binary uint8 (or bool) arrays unless stated otherwise.
* A rule table is a 1D uint8 array; entry i is the output for neighbourhood pattern i.
* Every volume generator returns a boolean occupancy array indexed [x, y, z] with z
  pointing up on screen.  For space-time constructions z is time.
* Randomness always comes from an explicit numpy Generator or integer seed.
"""
