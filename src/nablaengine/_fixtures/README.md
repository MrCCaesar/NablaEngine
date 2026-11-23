# Packaged fixtures

The default fixtures shipped with the `nablaengine` wheel, so that the quick start runs without a source checkout:

| Path | Content |
|---|---|
| `mjlab/g1_train.mjb`, `mjlab/g1_meta.json` | The Unitree G1 model (mjlab configuration) and its metadata, copied from `fixtures/mjlab` of the repository. |
| `mjlab/LICENSE` | The licence of the Unitree G1 description (BSD 3-Clause), from MuJoCo Menagerie. |
| `mjlab/README.md` | The model's provenance and the changes made. |
| `data/mf_s0.npz` | The benchmark corpus `mf_s0` (4096 states of this G1 model), copied from `benchmarks/data`. |

Lookup order (`nablaengine.fixtures`): `$DK_FIXTURES` / `$DK_BENCH_DATA`, then `fixtures/mjlab` / `benchmarks/data` of
a source checkout, then this copy.

The reference data of the checks (the MuJoCo Warp references, the finite-difference truth, the corpora `mf_s1` ..
`mf_s3`), the terrain models and the other robots (`fixtures/robots`) are not packaged: the checks, the scripts and
these models need a checkout of <https://github.com/sailab-ai/NablaEngine> (or `DK_FIXTURES` / `DK_BENCH_DATA`
pointing into one). NablaEngine itself is licensed under the Apache License 2.0; `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md` are in the wheel's metadata (`nablaengine-<version>.dist-info/licenses`).
