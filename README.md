# Fast Photonic-Crystal Band-Structure Prediction (TM Mode)

A compact forward-solver example using a Deep Eigenspace Network (DEN) and Rayleigh–Ritz calculation. The repository includes one trained TM mode checkpoint, its POD basis and FEM matrices, a training/evaluation script, a command-line predictor, and an interactive local browser demo. The raw training dataset and other experiments from the paper are not included.

> **Author review:** Keep this repository private until the authors approve redistribution of the checkpoint, POD basis, and FEM matrices. MIT covers the source code only; the binary assets do not yet have a separate reuse license.

## Interactive demo

Use Python 3.9+ with an appropriate PyTorch installation:

```bash
pip install -r requirements-demo.txt
python -m demo
```

Open <http://127.0.0.1:8000>. Paint the outlined editable triangle in the 16×16 unit cell; symmetry updates the other cells. You can adjust a selected cell with the slider, undo/redo, import/export a 36-value JSON design, choose 9–30 sampled k-points, and optionally verify the bands with FEM. The demo runs locally on CPU by default; `python -m demo --device cuda` selects CUDA if available.

## Command-line prediction and training

```bash
pip install -r requirements.txt
python examples/forward_tm.py --device cpu
python examples/forward_tm.py --fem --device cpu
```

The predictor accepts `--vector-file design.json` containing exactly 36 finite values in `[0,1]`. Without it, a deterministic illustrative design is used. Results go to `outputs/forward/`.

The full raw dataset is on https://drive.google.com/file/d/1n3hfDiQ_9ATn9v9wVQml98oiygzWeeCn/view?usp=sharing. Once `ev10-rand_uniform.mat` is available at `../data/ev10-rand_uniform.mat`, run `python train_tm.py --device cuda` (or pass `--data PATH`). The script saves the checkpoint, training history, and test metrics in JSON and CSV under `outputs/training/`. The bundled four-example synthetic file is only for checking the training pipeline; it cannot reproduce paper accuracy.

See [DETAILS.md](DETAILS.md) for data format, evaluation protocol, demo behavior, asset provenance, and reproducibility limits. Check binary digests with `python tests/verify_assets.py`.
