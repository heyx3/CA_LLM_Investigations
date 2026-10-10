"""Measuring CA output and searching for rules worth keeping.

  metrics      every heuristic, for 2D grids and 3D volumes, with a registry
  dynamics     rule spaces and assays: how candidate rules are run before measuring
  search       bands, staged pipelines, samplers, target profiles, sweeps, tables
  banks        a census of a rule set: every rule measured once, kept as a table
  influence    coupling tests for the hierarchy: pinning, perturbation, sensitivity
  experiments  named, re-runnable experiments, each asking one question

The vocabulary (lambda, soup, damage spreading, bands...) is in docs/CONCEPTS.md.
"""
