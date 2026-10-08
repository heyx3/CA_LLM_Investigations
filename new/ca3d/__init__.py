"""ca3d: cellular-automaton volume generators and a voxel renderer.

Reconstructed from the exploratory scripts in ../raws.

Conventions shared by every module:

* Cell states are binary uint8 (or bool) arrays unless stated otherwise.
* A rule table is a 1D uint8 array; entry i is the output for neighbourhood pattern i.
* Every volume generator returns a boolean occupancy array indexed [x, y, z] with z
  pointing up on screen.  For space-time constructions z is time.
* Randomness always comes from an explicit numpy Generator or integer seed.
"""
