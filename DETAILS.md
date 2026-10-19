# DEN–Rayleigh–Ritz TM forward-solver example

This author-review package accompanies the draft “Band Structure Prediction of Photonic Crystals via Deep Eigenspace Networks.” It demonstrates TM forward prediction, optional FEM verification, model training and evaluation, and a local drawing interface. It is not a full reproduction package for every table, baseline, TE experiment, or raw dataset in the paper. Before public release, confirm coauthor approval, binary-asset rights, license holder, and the final citation.

## Prediction and browser interface

Install `requirements-demo.txt` and run `python -m demo` from the repository root. The server binds to `127.0.0.1`, starts one process, and loads the checkpoint and FEM matrices once. CPU is the default; choose CUDA in the page or with `--device cuda` if available. CPU inference uses up to four threads by default (`--cpu-threads N`). The page needs no Node installation or online frontend assets.

The outlined stair-step triangle in the upper-right quadrant of the 16×16 unit cell is editable. Its 36 values map through the solver's p4m index map; the remaining cells are symmetry previews. A click paints and selects a cell, and the slider changes that selected value. Predictions are debounced by 250 ms. Requests carry a design version so outdated prediction or FEM responses cannot overwrite the current design. FEM runs separately, allowing further editing. The page supports 9, 12, …, 30 computed k-points; the displayed closing Γ copies the first computed point. It shows all nine adjacent-band gap widths, including negative overlaps, and highlights positive open gaps. Solver and total wait times are separate.

The API provides `GET /api/meta`, `POST /api/predict`, and `POST /api/fem`. Calculation requests use `{"revision": 1, "design": [36 numbers], "k_points": 30}`; `k_points` defaults to 30. Designs must contain exactly 36 finite numeric values in `[0,1]`. The normalized design maps to permittivity `epsilon = 1 + 7.9 * design`. The built-in example is a deterministic illustrative pattern, not an optimized or paper-result design.

`python examples/forward_tm.py --fem --device cpu` writes `outputs/forward/bands.png` and `metrics.json`. Omit `--fem` for prediction only; use `--vector-file design.json` for your own vector. The plot marks the widest positive adjacent-band gap, if one exists. FEM comparison reports band-frequency MAE, RMSE, per-band MAE, per-gap IoU, and the mean IoU of gap pairs open in either result. Pairs closed in both results have no interval to compare and are omitted from that mean (`null` if all are closed). Metrics depend on the sampled k-path.

## Training and evaluation

`train_tm.py` defaults to `../data/ev10-rand_uniform.mat`. The original file has 1,000 material designs with 10 k-points each, but the loader prints material count, k-points per material, mesh nodes, modes, and total material-k samples rather than enforcing that exact dataset shape. Other data still must be compatible with the configured 1,089-node POD basis and 10-mode network. The raw dataset is deliberately not distributed. By default, the first 20% of material designs (up to 200) are the test set; remaining designs train the model. `--test-designs N` overrides that count.

Training settings follow the source workflow: model-initialization seed 42; 200 POD columns; output subspace dimension 20; four spectral layers; network widths/ranks/bandwidth from `config.yaml`; Adam learning rate 0.01 and weight decay `1e-6`; StepLR every 10 epochs with factor 0.7; batch size 32; 101 epochs. The checkpoint is saved every 10 epochs and at the end. `history.json` is updated after each epoch; `test_metrics.json` and `test_metrics.csv` are written after eigenpair evaluation. The console prints four paper-style metrics, while the CSV retains the full metric set. Undefined values appear as blank CSV cells and `null` in JSON. The one-time training-index shuffle precedes setting seed 42, so split membership is fixed but sample order can vary between runs.

The POD basis was generated from the uniform dataset. Its training-material pool overlaps the gradient-training pool; the first 200 test materials are separate. Test loss is computed each epoch, and manuscript hyperparameter comparisons used test performance. Consequently, reported test performance is **test-informed**, not an untouched holdout estimate.

The bundled `assets/smoke_training.mat` contains four synthetic examples, not paper training or test records. `python tests/smoke_training.py` checks loading and a gradient update only. A full training run requires the raw data and produces a new checkpoint rather than overwriting the bundled one.

## Scope and verification

The bundled checkpoint, POD basis, and FEM matrices are listed in `ASSET_SHA256.txt`. The source is derived from the author's `PhC/PC` implementation; portable wrappers replace hard-coded device transfers and cast mesh indices to a PyTorch-supported integer type without changing the model architecture or training parameters. This package is a representative forward calculation, not exact reproduction of the draft's Table 2; manuscript values and descriptions will be aligned separately. TE experiments, baseline networks, complete raw datasets, and the paper's speed benchmark are outside its scope.

For local checks, run `python tests/verify_assets.py`, `python tests/smoke_training.py`, `python tests/smoke_demo.py` (requires `httpx`), and `node tests/demo_frontend.test.js` (Node is needed for this test only). Earlier author checks covered CPU training/FEM and browser behavior; these checks do not substitute for a full training run on the undistributed raw dataset.
