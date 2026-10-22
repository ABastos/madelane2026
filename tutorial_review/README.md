# MaDeLaNe tutorial review (2024–2026)

A review of the tutorial code deposited on Box for the 2024, 2025 and 2026 workshops.
Representative tutorials are re-run here as Python ports.

| Path | What it is |
|---|---|
| `inventory.json` | Every deposited workflow, with its pipeline stage, lead, tools, years taught and Box folder |
| `make_workflow_graph.py` | Renders `outputs/workflow_map.png` / `.svg` from the inventory |
| `ports/vflip2.py` | Line-by-line port of `vFLIP2.m` (2026, Multi-Laminar Spectral Analysis) |
| `ports/run_spectrolaminar.py` | `Spectrolaminar_Tutorial_0.m` + `_1.m` (2026) and `CSDanalysis.m` (Maier, 2024) |
| `ports/run_spike_field.py` | Vinck spike–field tutorial: `create_data.m`, PPC1, STA, spike–field coherence (2024/2025) |
| `ports/run_connectivity.py` | Bastos FC tutorial: `sim_signaltonoise.m` (Granger + time reversal), `sim_volumeconduction.m` (2024/2025) |
| `outputs/` | Figures and `summary.json` files from each run |

## Data caveat

The Vinck and Bastos tutorials run on their own built-in simulations, as the originals do.
The recorded `.mat` files on Box (`TrialCutData_AreaMT.mat`, `NeuropixelsSnip_LPFC.mat`, 0.4–1.3 GB) were not available in the session that produced these outputs.
So `run_spectrolaminar.py` defaults to a **simulated** 16-channel laminar dataset with a planted ground truth: superficial gamma, deep alpha/beta, and an L4 current sink.
To run it on the real recording:

```bash
pip install numpy scipy matplotlib
python ports/run_spectrolaminar.py --mat /path/to/TrialCutData_AreaMT.mat
```

## Run everything

```bash
cd tutorial_review
python ports/run_spectrolaminar.py
python ports/run_spike_field.py
python ports/run_connectivity.py
python make_workflow_graph.py
```
