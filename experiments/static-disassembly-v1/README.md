# Static disassembly validation v1

This experiment validates the first implemented MGL static-FFN profile on the neural XOR fixture and all six FFN + pre-LayerNorm regions of the pinned final Pythia-70M checkpoint.

[Report](../../docs/reports/static-disassembly-v1.md) · [Measurements](../../results/static-disassembly-v1/) · [Implementation guide](../../docs/development/static-disassembly.md)

## Run

From the repository root:

~~~bash
python -m pip install -e '.[safetensors]'
python -m pip install -r experiments/static-disassembly-v1/requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python experiments/static-disassembly-v1/run.py --cache model-cache/143000 --download --output replay-output/static-disassembly-v1
~~~

The output directory must be new or empty. The script fetches only missing public inputs when --download is supplied, checks them against the prior experiment's pinned manifest, and never trains a model. Downloaded weights stay outside Git.

The declared controls are three coherent and three broken pairings per layer, eight seeded synthetic residual vectors per layer, a 1e-12 coherent-error tolerance, and a 1e-6 minimum broken-control change. Failed checks are saved before a nonzero exit. Thresholds are not adjusted after results.

## Scope

The analyzer performs no forwards or backwards. The validation harness separately evaluates isolated region equations in NumPy float64. This is not native-framework or whole-model fidelity testing.

Static descriptors omit nonlinear activation regimes. The toy controls deliberately retain a descriptor collision between absolute value and the zero function, as well as an exact contribution-splitting relation across widths.

The HTML inspector and compact report can be rebuilt from saved measurements:

~~~bash
python experiments/static-disassembly-v1/build_report.py
~~~

The published Pythia inspector shows only 64 of 2,048 contributions in layer 0. The measurements cover all 12,288 contributions across six regions. Full analysis output can be regenerated with the CLI without committing model weights or large activation arrays.

