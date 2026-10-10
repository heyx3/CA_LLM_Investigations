"""Regression tests against numbers recorded earlier for the cityscape.

They pin the reference cityscape (its hard-coded rule tables, cityscape.REFERENCE_BANKS,
and its seeded starting states) and the density-ladder rotation timings, so any change
to the hierarchy engine that alters these results is noticed.
"""
import pytest

from ca3d.analysis import metrics
from ca3d.rulesets import cityscape, hierarchy


def test_cityscape_coarse_layers_match_the_notes():
    st = cityscape.make(n=160).run(160, record_layers=True)
    coarse = st.layers[1:]
    assert [round(float(l.mean()), 3) for l in coarse] == [0.099, 0.186, 0.695]
    assert [round(metrics.pillar_fraction(l.astype(bool)), 2) for l in coarse] == [0.40, 0.60, 0.79]
    # the fine layer differs slightly: its two 'complex' contexts come from a rebuilt pool
    assert abs(st.fine.mean() - 0.196) < 0.02


@pytest.mark.parametrize('ic_seed, steps', [
    (3, [36, 52, 60, 68, 92]), (7, [20, 36, 52, 60, 76]), (11, [28, 36, 44, 76, 92]),
    (19, [12, 36, 60, 84, 100]), (23, [28, 52, 60, 76, 100]), (31, [12, 20, 44, 68, 100])])
def test_density_ladder_rotations_match_the_original(ic_seed, steps):
    ca = cityscape.make(n=160, ic_seed=ic_seed, rotation=hierarchy.RotateOnDensityLadder(0.05))
    ca.run(107)
    assert [t for t, _ in ca.rotation_log] == steps
