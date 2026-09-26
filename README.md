# Cardiac-Agent

An independent cardiac ultrasound research pipeline following `Overview-Pipeline.docx`:

```text
ED/ES images -> ROI -> Function -> RWMA head interface
             -> structured Knowledge Bridge -> Explanation -> QA/Evaluation
```

The pipeline accepts precomputed LV cavity masks or official EchoNet predictions. It computes ED/ES
areas, area ratio and FAC, retains optional cavity appearance/optical-flow
statistics, and writes typed evidence and checksum-verified run artifacts.
RWMA returns `unavailable` without a trained model and `abstained` on blocking QC.
There is no fabricated classifier output or FAC-based diagnostic threshold.

Real CAMUS validation covers three patients and six ED/ES pairs with manual masks.
The official EchoNet checkpoint was evaluated on three A4C pairs (six images):
**mean Dice 0.8509, mean IoU 0.7440, FAC MAE 8.14 percentage points, 0/3 failures**.
All three predicted-mask cases completed Qwen reporting with exact numerical checks;
a separate real manual-mask Qwen run also passed.

These are small, ordered cross-dataset pilot results, not a held-out clinical benchmark.
The largest observed FAC error was 15.22 percentage points. RWMA training, clinical
interpretation, open-ended QA and Web UI remain pending.

Documentation: [Chinese technical guide](docs/TECHNICAL_GUIDE.zh-CN.md),
[architecture mapping](docs/ARCHITECTURE.zh-CN.md),
[current progress and results](docs/PROGRESS_2026-09-18.zh-CN.md).

## Install and run

```bash
git clone https://github.com/IreliaNeu/Cardiac-Agent.git
cd Cardiac-Agent
conda env create -f environment.yml
conda activate cardiac-agent
cardiac-agent fixture data/synthetic-001
cardiac-agent run data/synthetic-001/study.json --output runs
cardiac-agent run data/synthetic-001/study.json --output runs --qwen --env .env
cardiac-agent verify runs/RUN_ID
pytest -q
```

Copy configuration from `.env.example` when creating a private `.env`. Alternatively,
pass an existing environment file with `--env`; this project does not modify it.
Default API: SiliconFlow, `Qwen/Qwen3-30B-A3B-Instruct-2507`. Set
`CARDIAC_BASE_URL`, `CARDIAC_QWEN_MODEL`, and `CARDIAC_API_KEY_ENV` to switch provider.
No raw images or patient identifiers are sent to the explanation API.

Evidence includes ED area, ES area, FAC, area ratio, RWMA availability and the
difference/motion method. When available, map mean, variance and units are included;
a configured predictor's class and probability are included only for actual predictions.
No clinical thresholds or diagnostic interpretation are inferred from these statistics.

Minimal offline acceptance (fresh output directory required):

```bash
python -m cardiac_agent.smoke --output runs/smoke-001
python -m cardiac_agent.smoke --study data/case-4CH/study.json --output runs/smoke-real
# Require successful Qwen reporting rather than accepting a template fallback:
python -m cardiac_agent.smoke --study data/case-4CH/study.json \
  --output runs/smoke-api --qwen --env .env --deformation farneback
```

The command writes `acceptance.json`, checks output and source hashes, verifies
arithmetic, and returns nonzero if explicitly requested Qwen reporting fails.
Passing this engineering check does not require a trained RWMA classifier.
See the [clinical collaboration checklist](docs/CLINICAL_CONFIRMATIONS.zh-CN.md).

The baseline Qwen adapter performs constrained evidence ordering. The model returns
JSON claims with exact evidence IDs and values; deterministic realization produces
the report. Invalid claims or API errors trigger an explicitly recorded template
fallback. The optional three-role workflow below adds model-written research prose,
not validated clinical interpretation. Open-ended QA remains a future stage.

## One-round three-role workflow

```text
ED/ES metadata -> Planner (select sparse flow or area-only)
ED/ES images -> EchoNet segmentation -> deterministic measurements and QC
plan + measured evidence -> Analyst (research draft)
plan + evidence + draft -> Reviewer (final report, limited/refer)
predicted masks + manual masks -> independent evaluation, after discussion
```

These are three separate role calls to the same configurable Qwen model, not three
independently trained models. Each role speaks once. There is no debate loop,
long-term memory, autonomous model selection or validated RWMA diagnosis.
Planner's tool choice changes execution; Reviewer can revise the draft and refer
a case, but cannot alter measurements. Blocking QC always forces referral.

```bash
conda activate cardiac-agent-gpu
python scripts/three_role_batch.py \
  --data data/camus-100 --output runs/three-role-100 \
  --count 100 --workers 4 --env .env
```

This downloads the first N alphabetically ordered official CAMUS patients, A4C
ED/ES only. Four workers overlap downloads/API calls; GPU inference is serialized.
Every case exports `original/`, `segmentation/` (binary masks and overlays),
`report.md`, `report.json`, `agent_trace.json`, independent evaluation and SHA256
checksums. Original PNGs are display conversions; original NIfTI files and download
provenance remain in the data directory. `index.csv` and `summary.json` summarize
the batch. Experiment outputs, images and private credentials are not committed.

Only view/phase metadata, computed measurements, QC, and previous role outputs go
to the API. Patient IDs, raw images, reference masks and reference metrics do not.
Reports must preserve the exact evidence map; numeric tokens in prose must match
an evidence value, allowing decimal half-up rounding to the displayed precision.
This lexical guard does not establish correct units, associations or clinical meaning.
An immutable limitations section and numerical table accompany model-written prose.

Received role responses are cached with request hashes. Re-running the same command
verifies completed cases and resumes partial cases without replaying received role
calls. A started/failed/invalid response is not silently regenerated; it remains a
visible failure requiring investigation. Transport retries (up to three attempts)
are logged separately and may repeat a provider request after a network timeout.
Batch failures produce a nonzero exit status. Keep output directories separate
when changing prompts, models, code, input data or segmentation configuration.

## Input contract

Use a JSON study manifest with `study_id`, `patient_id`, `view`, `ed_image`,
`es_image`, `ed_mask`, `es_mask`, `mask_source` and `foreground_label`.
Paths resolve relative to the manifest. Both images must share the pixel grid and
calibration; masks must match their respective image dimensions. Set the exact LV
label, e.g. 1 for indexed masks or 255 for 0/255 masks. `spacing_mm` is optional
and applies equally to both frames. Without it, areas are reported only in pixels.
`rwma_label` is optional independent ground truth, used only for evaluation.

`--deformation intensity_difference` is the dependency-light default. It is a
normalized appearance difference in the ED cavity, not physical deformation.
`--deformation none` skips it. Install `.[flow]` to use optional Farneback flow.
Optical flow is reported in pixels per ED/ES pair, not velocity or strain.

## Segmentation backend

First annotated-image target: [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html).
The importer accepts one extracted patient directory at a time, original MHD/raw or
NIfTI ED/ES files, and manual `_gt` labels. Install `.[medical_io]` for SimpleITK.
It preserves labels and spacing, applies joint ED/ES scaling only for non-uint8
images, and exports an audit record. It does not infer RWMA labels from EF or masks.

```bash
cardiac-agent import-camus /path/to/patient0001 --view 4CH --output data/patient0001-4CH
cardiac-agent run data/patient0001-4CH/study.json --output runs
```

The dataset itself is not bundled. Full training/test partitioning is patient-level
and must be defined before training. Manual-mask runs are an oracle feature path,
not a measurement of segmentation model accuracy.

Selected upstream: [EchoNet-Dynamic](https://github.com/echonet/dynamic),
one-output DeepLabV3-ResNet50 LV cavity segmentation trained on apical four-chamber
echo images. Use other views only after domain-specific validation.
The lazy adapter requires `.[segmentation]`, an official checkpoint and verified
normalization. It is not imported in the CPU mask-based path. GPU inference has
been verified with PyTorch 2.8.0+cu128, torchvision 0.23.0 and an RTX 4090.

```bash
conda env create -f environment.gpu.yml
conda activate cardiac-agent-gpu
mkdir -p models
curl -fL --retry 2 https://github.com/echonet/dynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt \
  -o models/deeplabv3_resnet50_random.pt
sha256sum models/deeplabv3_resnet50_random.pt
python scripts/camus_pilot.py --output data/camus-pilot
python scripts/evaluate_segmentation.py \
  --studies data/camus-pilot/converted/patient0001-4CH/study.json \
            data/camus-pilot/converted/patient0002-4CH/study.json \
            data/camus-pilot/converted/patient0003-4CH/study.json \
  --config configs/echonet.official.json \
  --output runs/echonet-evaluation --deformation farneback
```

Expected checkpoint SHA256:
`b32ea45b8200e11298ff4021166ff8d187204fb0c27a937bcdb4c518ffcbe1ed`.
Add `--qwen --env .env` to evaluate Qwen reporting on model-derived metrics.
Each evaluation needs a fresh output directory. It preserves independent manual
and model runs, reports all failed studies, and exits nonzero on any case failure.
Model inputs omit both manual masks. This evaluator restricts inputs to A4C.
Means are over successful cases, with the failure denominator explicitly reported.
The downloader selects the first three official patient folders by name; establish
patient-level training/test splits before formal experiments. Observe CAMUS terms
and cite Leclerc et al., IEEE TMI 2019, DOI: `10.1109/TMI.2019.2900516`.

With a study manifest that omits both masks, pass `--segmentation-config config.json`.
The config must contain `checkpoint`, `mean` and `std` (three RGB values on the
0-255 scale), and optional `device`. Normalization is mandatory because upstream
computes it from training data. No guessed normalization is supplied.
Checkpoint loading uses `weights_only=True`, strips DataParallel prefixes and
requires exact parameter compatibility. `configs/echonet.official.json` provides
parameters traced to the official EchoNet-Strain inference implementation.

## Status and outputs

Each run receives a new UUID directory containing request, masks, optional map,
measurements, RWMA status, knowledge, explanation, evaluation, run status and hashes.
Partial failures retain `failure.json`. No automatic clinical benchmark scores
are inferred from synthetic examples. Preset QA currently checks area reduction,
RWMA availability and FAC/LVEF distinction. Batch training, calibrated RWMA,
full-cycle motion, open questions, clinician scoring and Web UI are pending.

The batch evaluator is in `benchmark.py`; per-study stages are orchestrated by
`pipeline.py`. `camus.py` handles label-preserving medical image import, `roi.py`
segmentation, `function.py` measurements, `rwma.py` the predictor interface,
`explanation.py` Qwen and exact evidence checks, and `evaluation.py` mask/QA metrics.
Source evidence uses absolute paths; `cardiac-agent verify` expects the original
input locations to remain available. FAC and area ratio are algebraically redundant.
Farneback is sparse image displacement, not myocardial strain. CAMUS masks do not
provide independent RWMA labels, and systolic dysfunction is a distinct target.

## Provenance

Architecture lessons come from [IreliaNeu/RS-Agent](https://github.com/IreliaNeu/RS-Agent),
which acknowledges [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent).
This cardiac package is separately implemented and does not import legacy RS-Agent
or Lagent. The upstream EchoNet code remains in its own repository under its license;
see `docs/SEGMENTATION.md`. Project notes are in Chinese under `docs/`.
