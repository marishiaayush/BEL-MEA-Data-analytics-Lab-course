# HD-MEA Data Analysis Practical — Python

A hands-on course in analysing high-density microelectrode array (HD-MEA)
recordings of neuronal cultures, from the raw MaxWell `.h5` file to a drug
comparison. Everything runs in Jupyter notebooks, with one helper module
(`mea_helpers.py`) and a CSV manifest of the recordings. It replaces the 2024
MATLAB live scripts.

## Quick start (Windows)

You only need **git** installed. Everything else (uv, Python, the libraries) is
installed for you by the setup script.

1. **Get the files.** Open *Command Prompt* (press the Windows key, type `cmd`,
   Enter) and run:

   ```bat
   git clone https://github.com/marishiaayush/BEL-MEA-Data-analytics-Lab-course.git
   cd BEL-MEA-Data-analytics-Lab-course
   ```

   (Or on GitHub click the green **Code** button → **Download ZIP** and unzip it.)

2. **Install the environment.** Double-click **`setup.bat`** in that folder. The
   first run installs [uv](https://docs.astral.sh/uv/) if you don't have it, then
   downloads Python and all libraries — it needs internet and takes a few minutes.

3. **Open the notebooks.** Double-click **`start_jupyter.bat`**. JupyterLab opens
   in your browser; open the notebooks **in order**, starting with
   `00_setup_and_load.ipynb`.

4. **Add the data.** The recordings are not in this repo — put them in a `data/`
   folder as described under [Data](#data) below.

macOS / Linux users: see [Installation](#installation).

## Contents

```
.
├── mea_helpers.py           # the plumbing + all analysis functions (also the exercise answers)
├── recordings.csv           # manifest: one row per recording (this year's data)
├── recordings_template.csv  # empty template for a new manifest
├── pyproject.toml, uv.lock  # the Python environment (uv)
├── requirements.txt         # the same environment for pip / conda users
├── setup.sh / setup.bat     # one-time setup (macOS/Linux / Windows)
├── start_jupyter.bat        # open JupyterLab (Windows)
├── data/                    # the recordings — not included, see below
└── notebooks/
    ├── 00_setup_and_load.ipynb        your recording; online spikes: maps and a clip of the chip
    ├── 01_explore_hdf5_filter.ipynb   HDF5 file; raw traces; power spectrum; filtering; noise (RMS)
    ├── 02_spike_detection.ipynb       noise (MAD); threshold; spike detection; firing rate; activity map
    ├── 03_sorted_units_SUA.ipynb      sorted units; curation (refractory period); Fano/CV; feature maps
    ├── 04_waveforms_footprints.ipynb  mean waveform; amplitude; spatial footprint; spike width
    ├── 05_network_bursts.ipynb        population rate; network bursts; burst rate and regularity
    └── 06_compare_conditions.ipynb    baseline vs drug for drugs A and B; per-unit statistics; reporting
```

The notebooks are saved **with their outputs** (figures, results), so they can
be read without running them. Notebook 00 is the largest (~12 MB) because it
contains an interactive clip of the activity on the chip.

## How the notebooks work

Each notebook starts with a **Goal** and is split into numbered sections.
Three signs mark what students do:

| Sign | What it is |
|---|---|
| ✎ **Your turn** | an exercise: complete one line of code (look for `# TODO` and an obviously wrong placeholder such as `0.0`); a self-check, `mea.check(...)`, prints `[OK]` or a hint |
| 💬 **Discuss** | a question to discuss with a partner; no code, no check |
| 🔧 **Try it** | an optional experiment: change a value, re-run, set it back |

The explanation or worked example just above each exercise shows what is
needed, and every full answer lives in `mea_helpers.py`, so no one gets stuck.
Notebook 00 has a table of contents of the whole practical, listing every
exercise.

## Data

The recordings are **not** part of this folder. The notebooks look for them in
`data/` (set by `DATA_ROOT = "../data"` at the top of each notebook), with the
paths listed in `recordings.csv`, relative to `data/`:

- `raw_files` — the MaxWell `.h5` recording (raw signal + online-detected spikes);
- `sorted_files` — the Kilosort4 output folder of that recording (with
  `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `channel_map.npy`,
  `cluster_KSLabel.tsv`, …, or a `sorter_output/` subfolder containing them).

For this year's manifest, `data/` holds two folders (or links to them):
`raw/` and `sorted/`. Each row also gives the group, drug (A/B), condition
(Baseline/Drug), chip type and well (MaxTwo: one well per row).

To use other recordings, copy `recordings_template.csv` to `recordings.csv`
and add one row per recording, with the columns
`UID, GroupID, Chip, Drug, Conditions, Well, raw_files, sorted_files`.

## Installation

The environment is managed with [**uv**](https://docs.astral.sh/uv/) and
pinned by `uv.lock`, so every computer gets exactly the same versions
(Python 3.11, numpy, scipy, pandas, matplotlib, h5py, hdf5plugin,
spikeinterface, plotly, JupyterLab).

**Windows:** double-click `setup.bat` (installs uv if missing, then builds the
environment; first run needs internet and takes a few minutes), then
double-click `start_jupyter.bat`.

**macOS / Linux:**

```bash
git clone https://github.com/marishiaayush/BEL-MEA-Data-analytics-Lab-course.git
cd BEL-MEA-Data-analytics-Lab-course
bash setup.sh                    # installs uv if missing, then: uv sync
uv run jupyter lab notebooks     # open the notebooks
```

**With conda or pip instead:** `requirements.txt` lists the same pinned
versions.

```bash
conda create -n mea python=3.11 && conda activate mea
pip install -r requirements.txt
jupyter lab notebooks
```

**MaxWell's compression plugin.** MaxWell `.h5` files use their own
compression filter. The first time a recording is opened, it is downloaded
automatically (by `neo`, into `~/hdf5_plugin_path_maxwell`), so that first run
needs internet.

## Running the notebooks

1. Open the notebooks **in order**. In the settings cell at the top of each,
   enter `GROUP_ID`, `DRUG` and `CONDITION` (and `DATA_ROOT` if the data are
   elsewhere), then run all cells.
2. After changing `mea_helpers.py`, **restart the kernel**: Python only imports
   the module once, so a running kernel keeps the old version.
3. If the data are on a network share, reading can be slow; loading the
   recordings is the slowest step.

## Background and references

Curation (notebook 03) follows the refractory-period rule: Kilosort's `good`
label plus fewer than 1 % of inter-spike intervals below 1.5 ms, compared with
Hill et al.'s contamination estimate. References and links are in notebook 03
(Kilosort4: Pachitariu et al. 2024; Hill, Mehta & Kleinfeld 2011;
spikeinterface quality metrics; Allen Institute quality metrics).

## What changed from the 2024 MATLAB version

- MATLAB live scripts (`scripts/LS01–LS10`) → Jupyter notebooks.
- The SQLite database + `recordingsDB` class → a plain `recordings.csv`
  manifest, filtered with pandas.
- The hardware wrapper (`hd-wrap`) → spikeinterface/neo for reading the files;
  all lesson-facing analysis is reimplemented in plain numpy/scipy in
  `mea_helpers.py`.
- The MATLAB project machinery (`environment.m`, `.prj`, `resources/project/*`)
  is no longer needed.
