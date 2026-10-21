# Cyclone Genesis Predictor

Predicting, from a **single early infrared satellite image**, whether a weak and
disorganised tropical storm will later intensify into a typhoon/hurricane
(max sustained wind >= 64 kt). The model sees only the image - no future frames and no
wind or pressure numbers - so it is making the call before the storm visibly organises.

All training and experiments ran as Kaggle notebooks; this repo is the code from them
(35 notebooks).

## Highlights

- **Two datasets merged across 6 ocean basins**: Digital Typhoon (Western Pacific, 512x512
  IR) and TCIR/TCSA (Atlantic, East/Central Pacific, North Indian Ocean, Southern
  Hemisphere, West Pacific). The two use different storm-numbering schemes, so duplicates
  are found by an ID key **and** a best-track position match (same time, within 1.5 deg,
  >= 4 shared points); 392 duplicate storms were removed so validation can't leak.
- **Held-out by storm**, stratified by basin and label: 2,037 storms (1,627 train / 410 validation).
- **Model**: 1-channel ResNet-18 with weight EMA, physically-safe augmentation (no flips -
  storms in the two hemispheres spin in opposite directions), and Platt calibration folded
  directly into the final layer, kept only because 5-fold cross-checking showed it generalises.
- **Shipped model results** (410 held-out storms): storm-level **AUC 0.775**, frame-level
  AUC 0.742 (0.797 on Digital Typhoon); calibration error (ECE) cut from **0.103 to 0.030**.
- Systematic ablation sweeps (v5-v7), temporal architectures, self-supervised temporal
  pretraining (v8), extra data sources (HURSAT-B1, MOSDAC/INSAT, Digital Typhoon AU),
  a separate intensity model, and two input "gate" models that refuse to score images that
  aren't real satellite IR or don't contain an organised cyclone.

## Notebooks

### Shipped model

| Notebook | What it does | Kaggle |
|---|---|---|
| [`v6-ship.py`](shipped-model/v6-ship.py) | Trains and ships the final early-intensification model (calibrated ResNet-18). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v6-ship) |
| [`v5-final.py`](shipped-model/v5-final.py) | v5 final trainer: trains one shipping model with the configuration the v5 sweeps picked. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v5-final) |
| [`v5-fulldata.py`](shipped-model/v5-fulldata.py) | v5 final trainer, full-data variant. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v5-fulldata) |
| [`demo-render.py`](shipped-model/demo-render.py) | Demo video renderer. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-demo-render) |
| [`demo-picks.py`](shipped-model/demo-picks.py) | Picks real, in-distribution demo frames for the demo web app from the same data the models trained on. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-demo-picks) |

### Training

| Notebook | What it does | Kaggle |
|---|---|---|
| [`intensification-train.py`](training/intensification-train.py) | Early-intensification prediction for Western Pacific tropical cyclones. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensification-train) |
| [`train-v2-combined.py`](training/train-v2-combined.py) | Early-intensification prediction, COMBINED two-dataset version (v2). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-train-v2-combined) |
| [`intensification-train-v3.py`](training/intensification-train-v3.py) | Early-intensification prediction, v3. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensification-train-v3) |
| [`intensification-train-v4.py`](training/intensification-train-v4.py) | Early-intensification prediction, v4. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensification-train-v4) |

### Sweeps and experiments

| Notebook | What it does | Kaggle |
|---|---|---|
| [`v5-sweep.py`](sweeps-and-experiments/v5-sweep.py) | v5 ablation sweep: which training choices actually help (round 1). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v5-sweep) |
| [`v5b-sweep.py`](sweeps-and-experiments/v5b-sweep.py) | v5 ablation sweep, round 2 (adds an auxiliary peak-wind head). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v5b-sweep) |
| [`v5c-sweep.py`](sweeps-and-experiments/v5c-sweep.py) | v5 ablation sweep, round 3. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v5c-sweep) |
| [`v6-sweep.py`](sweeps-and-experiments/v6-sweep.py) | Early-intensification prediction, v6 ABLATION SWEEP (TEMPORAL). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v6-sweep) |
| [`v6-mosdac.py`](sweeps-and-experiments/v6-mosdac.py) | V6: does adding MOSDAC/INSAT Indian Ocean data actually help the Indian Ocean? | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v6-mosdac) |
| [`v7-smoke.py`](sweeps-and-experiments/v7-smoke.py) | v7 AUC campaign, round 1 (temporal architectures) - smoke test. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v7-smoke) |
| [`v7-sweep.py`](sweeps-and-experiments/v7-sweep.py) | v7 AUC campaign, round 1: temporal architectures compared under 4-fold grouped CV. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v7-sweep) |
| [`v7-r2-smoke.py`](sweeps-and-experiments/v7-r2-smoke.py) | v7 AUC campaign, round 2 (more data + regularisation) - smoke test. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v7-r2-smoke) |
| [`v7-r2-sweep.py`](sweeps-and-experiments/v7-r2-sweep.py) | v7 AUC campaign, round 2: more data (Digital Typhoon AU) + regularisation. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v7-r2-sweep) |
| [`v8-pretrain.py`](sweeps-and-experiments/v8-pretrain.py) | V8 -- self-supervised temporal pretraining for the cyclone encoder. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-v8-pretrain) |

### Intensity model

| Notebook | What it does | Kaggle |
|---|---|---|
| [`intensity-data-audit.py`](intensity-model/intensity-data-audit.py) | Cyclone INTENSITY model, data-audit pass ONLY. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensity-data-audit) |
| [`intensity-train.py`](intensity-model/intensity-train.py) | Separate intensity model: regresses storm wind speed from one IR frame - first training run. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensity-train) |
| [`intensity-v2-a100.py`](intensity-model/intensity-v2-a100.py) | Intensity model training run, v2 (A100 GPU). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensity-v2-a100) |
| [`intensity-v3-mt.py`](intensity-model/intensity-v3-mt.py) | Intensity model training run, v3 (MT variant). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensity-v3-mt) |
| [`intensity-v3-ordinal.py`](intensity-model/intensity-v3-ordinal.py) | Intensity model training run, v3 (ordinal variant). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-intensity-v3-ordinal) |

### Input gates

| Notebook | What it does | Kaggle |
|---|---|---|
| [`gates-data-audit.py`](input-gates/gates-data-audit.py) | Data audit for the two gate models, before spending GPU quota on training them. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-gates-data-audit) |
| [`gate1-satcheck.py`](input-gates/gate1-satcheck.py) | Gate 1: "is this real calibrated satellite IR data at all?" | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-gate1-satcheck) |
| [`gate2-cyclonecheck.py`](input-gates/gate2-cyclonecheck.py) | Gate 2: given a real satellite IR frame, is a cyclone actually present, or is it ordinary weather? | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-gate2-cyclonecheck) |
| [`gate2-cyclonecheck-v2.py`](input-gates/gate2-cyclonecheck-v2.py) | Gate 2 v2: "is a tropical cyclone ACTUALLY PRESENT and organized in this real satellite IR frame?" | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-gate2-cyclonecheck-v2) |
| [`needle-search.py`](input-gates/needle-search.py) | Searches for a real frame the intensification model scores high but Gate 2 says is not a cyclone. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-needle-search) |
| [`native-frame-extract.py`](input-gates/native-frame-extract.py) | Re-extract the top needle-search candidates at their NATIVE Digital Typhoon resolution (no downsizing to 224). | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-native-frame-extract) |

### Data probes

| Notebook | What it does | Kaggle |
|---|---|---|
| [`storm-survey.py`](data-probes/storm-survey.py) | First survey of the Digital Typhoon dataset layout and metadata. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-storm-survey) |
| [`storm-scan.py`](data-probes/storm-scan.py) | Ranks held-out storms that became typhoons by how diffuse their first frame looks, to pick honest demo storms. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-storm-scan) |
| [`tcir-probe.py`](data-probes/tcir-probe.py) | Probe of the TCIR/TCSA HDF5 file: keys, shapes, channel order. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-tcir-probe) |
| [`dt-au-overlap-probe.py`](data-probes/dt-au-overlap-probe.py) | Probe: does Digital Typhoon AU overlap with Digital Typhoon WP? | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-dt-au-overlap-probe) |
| [`hursat-ingest-a.py`](data-probes/hursat-ingest-a.py) | HURSAT-B1 v06 ingest -- turn NOAA's global storm archive into training frames. | [link](https://www.kaggle.com/code/vismaypattedar/cyclone-genesis-predictor-hursat-ingest-a) |

## Data

- [Digital Typhoon (Western Pacific)](https://www.kaggle.com/datasets/bwandowando/digital-typhoon-dataset-western-pacific-wp)
- [Tropical Cyclone Structure Analysis (TCIR/TCSA)](https://www.kaggle.com/datasets/boyochen/tropical-cyclone-structure-analysis-tcsa)

The scripts expect to run on Kaggle with these datasets attached.
