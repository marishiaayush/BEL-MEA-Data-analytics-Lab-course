"""
mea_helpers.py
==============
Helper functions for the HD-MEA data-analysis practical.

The goal of this module is to keep the notebooks short and readable: all of the
"plumbing" (finding a recording, reading HDF5, scaling to microvolts, drawing
rasters, ...) lives here, so the notebooks can focus on the *science*.

The scientific steps (filtering, spike detection, firing-rate / burst features)
are implemented here in a simple, transparent way. You are encouraged to open
this file and read how they work -- none of it is magic.

Dependencies (all pre-installed on the lab laptops):
    numpy, scipy, pandas, matplotlib, spikeinterface, h5py

Author: Practicals 2026
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from pathlib import Path

# The recordings usually live on a network share (SMB), often mounted
# read-only. HDF5 >= 1.10 tries to take a file lock when opening a file, which
# such shares refuse -- giving a confusing "unable to lock file, errno = 13,
# Permission denied" error. Disabling HDF5 file locking fixes it. This must be
# set before h5py opens any file, so we do it here, at import time.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

# spikeinterface imports tqdm, which warns (in red) that ipywidgets is missing;
# it only means progress bars are shown as text, so keep it out of the notebooks.
warnings.filterwarnings("ignore", message="IProgress not found")

# matplotlib and spikeinterface are imported lazily inside the functions that
# need them, so the core analysis functions work even if plotting/IO libraries
# are unavailable.


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _get_traces_uv(recording, **kwargs):
    """Call recording.get_traces asking for microvolts, across spikeinterface versions.

    spikeinterface renamed the keyword `return_scaled` (<=0.101) to
    `return_in_uV` (>=0.102). We try the new name first, then fall back.

    We also always pass explicit channel_ids. For legacy MaxOne files the neo
    reader returns the full raw stream (e.g. 1028 rows: electrodes + aux) when
    no channels are requested, which then fails to scale against the 960
    per-electrode gains ("operands could not be broadcast"). Asking for the
    recording's own channel ids makes spikeinterface subset correctly first.
    """
    if kwargs.get("channel_ids") is None:
        kwargs["channel_ids"] = recording.get_channel_ids()
    try:
        return recording.get_traces(return_in_uV=True, **kwargs)
    except TypeError:
        return recording.get_traces(return_scaled=True, **kwargs)


def _ensure_hdf5plugin():
    """Load hdf5plugin so h5py can read MaxWell's compressed datasets.

    MaxWell .h5 files are often stored with a compression filter that h5py only
    understands once hdf5plugin has been imported. Importing it is harmless if
    the file doesn't need it. We swallow the ImportError so analysis-only use
    (no file reading) still works without the package installed.
    """
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    # MaxWell's own filter is not in hdf5plugin: neo downloads it (once) and
    # sets HDF5_PLUGIN_PATH, but HDF5 has usually read that variable already,
    # so register the folder with h5py directly.
    try:
        import h5py
        from neo.rawio.maxwellrawio import auto_install_maxwell_hdf5_compression_plugin
        with contextlib.redirect_stdout(io.StringIO()):     # silence "already located"
            auto_install_maxwell_hdf5_compression_plugin(force_download=False)
        plugin_dir = os.environ["HDF5_PLUGIN_PATH"].encode()
        if plugin_dir not in [h5py.h5pl.get(i) for i in range(h5py.h5pl.size())]:
            h5py.h5pl.prepend(plugin_dir)
    except Exception:
        pass

# spikeinterface is only needed for *reading* the MaxWell .h5 files. We import
# it lazily (inside the functions that need it) so that notebooks which only do
# analysis on already-loaded arrays still work if spikeinterface is missing.


# ---------------------------------------------------------------------------
# 1. The recordings manifest (replaces last year's SQLite database)
# ---------------------------------------------------------------------------
#
# Instead of a database we keep a simple manifest with one entry per recording.
# You fill it in during the experiments. The recommended format is TOML
# (recordings.toml): it is easy to edit in Notepad, one labelled [[recording]]
# block per recording, and allows # comments. Plain CSV is still accepted.
#
# Each recording has these fields (TOML key -> manifest column):
#
#   uid        / UID          a short unique id for the recording
#   group_id   / GroupID      student group number (may be set once at the top)
#   chip       / Chip         chip type, "maxone" or "maxtwo" (default "maxtwo")
#   drug       / Drug         config / drug label, e.g. "A", "B", ...
#   condition  / Conditions   "Baseline" or "Drug"
#   well       / Well         well number on the chip (default 0)
#   raw        / raw_files    path to the raw data.raw.h5 (full path, or relative to DATA_ROOT)
#   sorted     / sorted_files path to the spike-sorted output folder (optional)
#
# See recordings_template.toml for a worked example.
#
MANIFEST_COLUMNS = [
    "UID", "GroupID", "Chip", "Drug", "Conditions",
    "Well", "raw_files", "sorted_files",
]


def _manifest_from_toml(path):
    """Parse a TOML manifest into the same rows a CSV manifest would give.

    A top-level `group_id` (optional) applies to every recording that does not
    set its own. Each `[[recording]]` block uses the short keys listed above.
    """
    import tomllib
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("recording")
    if not entries:
        raise ValueError(
            f"No [[recording]] blocks found in '{path}'. "
            f"See recordings_template.toml."
        )
    default_group = data.get("group_id")
    rows = []
    for i, r in enumerate(entries, start=1):
        try:
            rows.append({
                "UID":          r["uid"],
                "GroupID":      r.get("group_id", default_group),
                "Chip":         r.get("chip", "maxtwo"),
                "Drug":         r["drug"],
                "Conditions":   r["condition"],
                "Well":         r.get("well", 0),
                "raw_files":    r["raw"],
                "sorted_files": r.get("sorted", ""),
            })
        except KeyError as e:
            raise ValueError(
                f"Recording #{i} in '{path}' is missing the {e} field."
            ) from None
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def load_manifest(path="recordings.toml"):
    """Load the recordings manifest (TOML or CSV) into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the manifest '{path}'.\n"
            f"Create a recordings.toml (see recordings_template.toml) or a CSV "
            f"with these columns: {', '.join(MANIFEST_COLUMNS)}."
        )
    if path.suffix.lower() == ".toml":
        df = _manifest_from_toml(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return df


def find_recording(manifest, group_id, drug, condition):
    """Return the single manifest row matching (group, drug, condition).

    Raises a clear error if zero or more than one row matches, so students
    immediately see when their selection is ambiguous or absent.
    """
    sel = manifest[
        (manifest["GroupID"] == group_id)
        & (manifest["Drug"].astype(str).str.upper() == str(drug).upper())
        & (manifest["Conditions"].astype(str).str.lower() == str(condition).lower())
    ]
    if len(sel) == 0:
        raise LookupError(
            f"No recording for GroupID={group_id}, Drug={drug}, "
            f"Conditions={condition}. Check your entries and the manifest."
        )
    if len(sel) > 1:
        raise LookupError(
            f"Found {len(sel)} recordings for GroupID={group_id}, Drug={drug}, "
            f"Conditions={condition}; expected exactly one. "
            f"UIDs: {list(sel['UID'])}"
        )
    return sel.iloc[0]


# ---------------------------------------------------------------------------
# 2. Opening a recording (HDF5 -> spikeinterface RecordingExtractor)
# ---------------------------------------------------------------------------

def raw_path(row, data_root="."):
    """Path to a row's raw .h5 file, resolved against `data_root`.

    Use this instead of f"{data_root}/{row['raw_files']}": when the manifest
    stores a full path, that string form would produce a broken path like
    "./Volumes/..." whereas this returns the absolute path unchanged.
    """
    return str(Path(data_root) / row["raw_files"])


def open_recording(row, data_root="."):
    """Open the raw HD-MEA recording referenced by a manifest row.

    Parameters
    ----------
    row : pandas.Series
        A row from the manifest (e.g. from `find_recording`).
    data_root : str or Path
        Folder that the manifest's `raw_files` paths are relative to.

    Returns
    -------
    recording : spikeinterface RecordingExtractor
        Lazy object: no samples are read until you call `get_traces`.
    """
    _ensure_hdf5plugin()
    import spikeinterface.extractors as se

    h5_path = Path(data_root) / row["raw_files"]
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Data file not found:\n  {h5_path}\n"
            f"Is DATA_ROOT set correctly, and is the data copied/mounted?"
        )

    chip = str(row["Chip"]).lower()
    if chip == "maxtwo":
        # MaxTwo chips have several wells in one file; pick this row's well.
        well = int(row["Well"])
        stream_id = f"well{well:03d}"
        recording = se.read_maxwell(str(h5_path), stream_id=stream_id,
                                    install_maxwell_plugin=False)  # done above
    else:
        # MaxOne / single-well: one stream per file.
        recording = se.read_maxwell(str(h5_path), install_maxwell_plugin=False)

    return recording


def recording_summary(recording):
    """Print a short, human-readable summary of a recording."""
    fs = recording.get_sampling_frequency()
    n = recording.get_num_frames()
    print("HD-MEA recording")
    print(f"  channels          : {recording.get_num_channels()}")
    print(f"  sampling rate     : {fs:,.0f} Hz")
    print(f"  duration          : {n / fs:,.1f} s  ({n:,} samples)")
    try:
        locs = recording.get_channel_locations()
        print(f"  array extent (um) : x [{locs[:,0].min():.0f}, {locs[:,0].max():.0f}], "
              f"y [{locs[:,1].min():.0f}, {locs[:,1].max():.0f}]")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 2b. Looking inside the HDF5 file
# ---------------------------------------------------------------------------
#
# An HDF5 file is like a little file system *inside* one file: "groups" are
# folders and "datasets" are the arrays/tables they contain. Both groups and
# datasets can carry "attributes" (small bits of metadata). The helpers below
# let students walk that tree and peek at it directly, before handing the file
# to spikeinterface.

def explore_hdf5(path, max_items=200, show_attrs=True):
    """Print the group/dataset tree of an HDF5 file.

    Shows, for each entry, whether it is a GROUP or a DATASET, and for datasets
    their shape and dtype. Optionally prints attributes. This is the Python
    equivalent of browsing the file with HDFView.
    """
    _ensure_hdf5plugin()
    import h5py

    print(f"HDF5 file: {path}\n")
    count = [0]

    def _visit(name, obj):
        if count[0] >= max_items:
            return
        count[0] += 1
        indent = "  " * name.count("/")
        short = name.split("/")[-1] or "/"
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[DATASET] {short}  shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{indent}[GROUP]   {short}/")
        if show_attrs and len(obj.attrs):
            for k, v in obj.attrs.items():
                print(f"{indent}    .{k} = {v}")

    with h5py.File(path, "r") as f:
        # Top-level attributes (often hold sampling rate, gain, version, ...).
        if show_attrs and len(f.attrs):
            print("Root attributes:")
            for k, v in f.attrs.items():
                print(f"  .{k} = {v}")
            print()
        f.visititems(_visit)
    if count[0] >= max_items:
        print(f"\n... stopped after {max_items} items (raise max_items to see more).")


def read_hdf5_dataset(path, dataset_name):
    """Read one named dataset out of an HDF5 file into a numpy array.

    e.g. read_hdf5_dataset(h5path, "sig")  or a nested "data_store/.../groups".
    Use `explore_hdf5` first to find the exact dataset name.
    """
    _ensure_hdf5plugin()
    import h5py
    with h5py.File(path, "r") as f:
        return f[dataset_name][()]


def load_online_spikes(row, data_root="."):
    """Spikes the MaxLab software detected *online*, during the recording.

    These are threshold crossings flagged live by the hardware (stored in the
    .h5 next to the raw signal) -- a quick look at activity without any
    processing. Returns a DataFrame: time (s, from recording start),
    channel (same ids as the recording), amplitude_uv (negative peak, uV).

    Handles both MaxWell file layouts: the newer one (``recordings/rec0000/
    wellNNN/...``) and the legacy MaxOne one (top-level ``sig``/``mapping``/
    ``proc0``).
    """
    _ensure_hdf5plugin()
    import h5py

    h5_path = raw_path(row, data_root)
    with h5py.File(h5_path, "r") as f:
        if "recordings" in f:
            # ----- newer MaxWell format -----
            well = f"well{int(row['Well']):03d}" if str(row["Chip"]).lower() == "maxtwo" else "well000"
            w = f[f"recordings/rec0000/{well}"]
            spk = w["spikes"][()]
            framenos = spk["frameno"]
            channels = spk["channel"]
            amp_uv = spk["amplitude"] * (float(w["settings/lsb"][0]) * 1e6)  # ADC counts -> uV
            fs = float(w["settings/sampling"][0])
            first_frame = int(w["groups/routed/frame_nos"][0])
            routed = set(int(c) for c in w["settings/mapping"]["channel"])
        else:
            # ----- legacy MaxOne format -----
            spk = f["proc0/spikeTimes"][()]
            framenos = spk["frameno"]
            channels = spk["channel"]
            amp_uv = spk["amplitude"].astype(float)       # already in uV here
            fs = 20000.0                                  # MaxWell always samples at 20 kHz
            routed = set(int(c) for c in f["mapping"]["channel"])
            # This format has no stored start frame; the first online spike is
            # essentially at the recording start (sub-100 ms), so use it.
            first_frame = int(framenos.min())

    # The "channel" field is the hardware amplifier channel -- the same ids the
    # recording uses (recording.get_channel_ids()). A few online spikes are on
    # channels that were not routed into this recording (so they have no raw
    # signal and no electrode location); drop them so every spike has a matching
    # channel and can be placed on the activity map.
    keep = framenos >= first_frame                        # drop events before the raw data starts
    on_chip = np.array([int(c) in routed for c in channels[keep]], dtype=bool)
    return pd.DataFrame({
        "time": (framenos[keep][on_chip] - first_frame) / fs,
        "channel": channels[keep][on_chip].astype(str),
        "amplitude_uv": amp_uv[keep][on_chip],
    })


# ---------------------------------------------------------------------------
# 2c. Spike-sorted data  (single-unit activity, SUA)
# ---------------------------------------------------------------------------
#
# You (the instructor) spike-sort the raw recordings ahead of time. Spike
# sorting assigns each detected spike to a putative single neuron ("unit"),
# turning the raw multi-unit signal into something closer to "real" neuronal
# data. The sorted output is referenced by the `sorted_files` column.

def load_sorting(row, data_root=".", sorter="kilosort2"):
    """Load spike-sorted output referenced by a manifest row.

    The exact on-disk format depends on how the data was sorted. This tries the
    common cases (a spikeinterface-saved folder, or a raw sorter output folder)
    and raises a clear message if neither works -- adjust `sorter` to match the
    sorter you used this year.

    Returns a spikeinterface SortingExtractor.
    """
    import spikeinterface as si
    import spikeinterface.extractors as se

    folder = Path(data_root) / str(row["sorted_files"])
    if not folder.exists():
        raise FileNotFoundError(f"Sorted output not found:\n  {folder}")

    # 1) A spikeinterface-saved extractor folder.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return si.load(folder)
    except Exception:
        pass
    # 2) A spikeinterface sorter folder (e.g. .../kilosort4 from run_sorter).
    try:
        from spikeinterface.sorters import read_sorter_folder
        with contextlib.redirect_stdout(io.StringIO()):   # its failed attempts are chatty
            return read_sorter_folder(folder)
    except Exception:
        pass
    # 3) Kilosort / phy output, read straight from its .npy files. Step 2 fails
    #    when the folder still points at the recording's original (cluster)
    #    path, so also look inside its `sorter_output/` subfolder.
    if (folder / "sorter_output").is_dir():
        folder = folder / "sorter_output"
    if (folder / "spike_times.npy").exists():
        params = {}
        exec((folder / "params.py").read_text(), {}, params)
        samples = _load_npy(folder / "spike_times.npy").ravel().astype("int64")
        labels = _load_npy(folder / "spike_clusters.npy").ravel().astype("int64")
        sorting = si.NumpySorting.from_samples_and_labels(
            samples, labels, float(params["sample_rate"]))
        _add_template_properties(sorting, folder)
        return sorting
    # 4) Directly as a named sorter format.
    reader = getattr(se, f"read_{sorter.lower()}", None)
    if reader is not None:
        return reader(folder)
    raise RuntimeError(
        f"Could not load sorted data from {folder}. "
        f"Check the format and the `sorter` argument."
    )


def _load_npy(path):
    """np.load that survives a quirk of macOS network shares (smbfs): closing a
    file can fail with BlockingIOError ('Resource temporarily unavailable') even
    though it was read completely. Read everything first, then ignore that error."""
    f = open(path, "rb")
    try:
        data = f.read()
    finally:
        try:
            f.close()
        except BlockingIOError:
            pass
    return np.load(io.BytesIO(data))


def _add_template_properties(sorting, folder):
    """Attach each unit's peak channel and footprint size from Kilosort templates.

    Cheap (no raw data is read). `footprint_size` counts the channels where the
    template is larger than 30% of its peak; `pick_footprint_unit` uses it.
    """
    try:
        templates = _load_npy(folder / "templates.npy")   # (unit, time, channel)
        chan_map = _load_npy(folder / "channel_map.npy").ravel()
    except (FileNotFoundError, ValueError):
        return
    unit_ids = np.asarray(sorting.get_unit_ids())
    if unit_ids.max() >= len(templates):
        return  # clusters were merged/split after sorting: ids != templates
    ptp = np.ptp(templates, axis=1)[unit_ids]
    in_fp = ptp > 0.3 * ptp.max(axis=1, keepdims=True)
    sorting.set_property("peak_channel_index", chan_map[ptp.argmax(axis=1)])
    sorting.set_property("footprint_size", in_fp.sum(axis=1))
    # the footprint's channel indices, as text ("12,13,40") so the property stays 1-D
    sorting.set_property("footprint_channels",
                         np.array([",".join(map(str, chan_map[m])) for m in in_fp]))

    # Kilosort's own label per unit: "good" if its autocorrelogram shows a clear
    # refractory dip, else "mua" (likely several neurons mixed). Used by `curate_units`.
    label_file = folder / "cluster_KSLabel.tsv"
    if label_file.exists():
        labels = pd.read_csv(label_file, sep="\t").set_index("cluster_id")["KSLabel"]
        sorting.set_property("ks_label", labels.reindex(unit_ids).fillna("unknown").to_numpy())


def pick_footprint_unit(sorting, min_spikes=200, unit_ids=None):
    """A unit that makes a good footprint example: seen on many electrodes.

    Among units with at least `min_spikes` spikes (optionally only those in
    `unit_ids`, e.g. the curated ones), returns the one with the largest
    `footprint_size` (from the sorter's templates). Without templates, falls
    back to the unit with the most spikes.
    """
    all_ids = np.asarray(sorting.get_unit_ids())
    allowed = np.isin(all_ids, unit_ids) if unit_ids is not None else np.ones(len(all_ids), bool)
    unit_ids = all_ids
    n_spikes = np.array([len(sorting.get_unit_spike_train(unit_id=u)) if ok else 0
                         for u, ok in zip(unit_ids, allowed)])
    if "footprint_size" not in sorting.get_property_keys():
        return unit_ids[np.argmax(n_spikes)]
    size = np.where(n_spikes >= min_spikes, sorting.get_property("footprint_size"), -1)
    return unit_ids[np.argmax(size)]


def sorting_to_spikes(sorting):
    """Convert a SortingExtractor into a tidy DataFrame: time (s), unit.

    This is the SUA counterpart of `detect_spikes` (which produces MUA), so the
    same feature functions (`firing_rate`, `fano_factor`, `cv_isi`,
    `population_rate`, ...) work on it unchanged -- just use `unit` instead of
    `channel`.
    """
    fs = sorting.get_sampling_frequency()
    rows = []
    for uid in sorting.get_unit_ids():
        st = sorting.get_unit_spike_train(unit_id=uid) / fs
        for t in st:
            rows.append((t, uid))
    spikes = pd.DataFrame(rows, columns=["time", "unit"])
    return spikes.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Reading traces (into plain numpy arrays, in microvolts)
# ---------------------------------------------------------------------------

def get_traces(recording, t_start=0.0, t_stop=1.0, channel_ids=None):
    """Read a time slice of data as a numpy array scaled to microvolts.

    Parameters
    ----------
    recording : RecordingExtractor
    t_start, t_stop : float
        Start and stop time in seconds.
    channel_ids : list or None
        Which channels to read (default: all).

    Returns
    -------
    times : (n_samples,) array of time in seconds
    data  : (n_samples, n_channels) array in microvolts
    """
    fs = recording.get_sampling_frequency()
    start = int(round(t_start * fs))
    stop = int(round(t_stop * fs))
    start = max(start, 0)
    stop = min(stop, recording.get_num_frames())

    data = _get_traces_uv(
        recording, start_frame=start, end_frame=stop, channel_ids=channel_ids,
    )
    data = np.asarray(data, dtype=float)
    times = np.arange(start, stop) / fs
    return times, data


def most_active_channels(recording, n=25, t_window=10.0, threshold=5.0, chunk=100):
    """Rank channels by a quick spike-count estimate in the first `t_window` s.

    Used to pick a manageable subset of channels for interactive work
    (spike detection on all channels is slow).
    """
    fs = recording.get_sampling_frequency()
    channel_ids = np.asarray(recording.get_channel_ids())
    counts = []
    # in blocks of channels: all ~1000 channels at once would need several GB of RAM
    for i in range(0, len(channel_ids), chunk):
        _, data = get_traces(recording, 0.0, t_window, channel_ids=channel_ids[i:i + chunk])
        data = bandpass_filter(data, fs)
        counts += [len(detect_spikes_on_trace(data[:, c], fs, threshold=threshold))
                   for c in range(data.shape[1])]
    counts = np.array(counts)
    order = np.argsort(counts)[::-1][:n]
    return channel_ids[order], counts[order]


# ---------------------------------------------------------------------------
# 4. Filtering  (chapter step 1: band-pass 300-3000 Hz)
# ---------------------------------------------------------------------------

def bandpass_filter(data, fs, low=300.0, high=3000.0, order=3):
    """Zero-phase Butterworth band-pass filter.

    Keeps the spike band (~300-3000 Hz) and removes slow drifts/LFP (< 300 Hz)
    and high-frequency noise. `sosfiltfilt` filters forwards and backwards so no
    phase shift is introduced (spike times are preserved).

    Works on a 1-D trace or a 2-D (n_samples, n_channels) array.
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    axis = 0 if data.ndim == 2 else -1
    return sosfiltfilt(sos, data, axis=axis)


# ---------------------------------------------------------------------------
# 5. Spike detection  (chapter step 2: MAD-based negative threshold)
# ---------------------------------------------------------------------------

def noise_std_mad(trace):
    """Robust estimate of the noise standard deviation via the MAD.

    sigma ~= median(|x|) / 0.6745. The median absolute deviation is barely
    affected by the (sparse, large) spikes, so it does not get inflated by them
    the way a plain standard deviation would.
    """
    return np.median(np.abs(trace)) / 0.6745


def detect_spikes_on_trace(trace, fs, threshold=5.0, refractory_ms=1.0):
    """Detect spikes on a single filtered trace as negative threshold crossings.

    Parameters
    ----------
    trace : (n_samples,) filtered signal (microvolts)
    fs : sampling rate (Hz)
    threshold : threshold in units of noise SD (e.g. 5 -> -5 * sigma)
    refractory_ms : minimum spacing between detected spikes (ms)

    Returns
    -------
    peak_idx : array of sample indices of the detected spike troughs
    """
    sigma = noise_std_mad(trace)
    thr = -threshold * sigma

    below = trace < thr
    # Falling edges: sample where the trace first goes below threshold.
    crossings = np.flatnonzero((~below[:-1]) & below[1:]) + 1
    if crossings.size == 0:
        return np.array([], dtype=int)

    # For each crossing, find the local minimum (trough) within a short window.
    win = max(1, int(round(0.001 * fs)))  # 1 ms search window
    peaks = []
    for c in crossings:
        seg = trace[c:c + win]
        if seg.size:
            peaks.append(c + int(np.argmin(seg)))
    peaks = np.array(sorted(set(peaks)), dtype=int)

    # Enforce a refractory period: drop peaks too close to the previous one.
    refractory = int(round(refractory_ms * 1e-3 * fs))
    kept = []
    last = -refractory - 1
    for p in peaks:
        if p - last >= refractory:
            kept.append(p)
            last = p
    return np.array(kept, dtype=int)


def detect_spikes(data, fs, channel_ids=None, threshold=5.0, refractory_ms=1.0):
    """Detect spikes on every channel of a (n_samples, n_channels) array.

    Returns a tidy DataFrame with columns: time (s), channel.
    This "long" format is convenient for rasters and firing-rate calculations.
    """
    n_samples, n_channels = data.shape
    if channel_ids is None:
        channel_ids = np.arange(n_channels)

    rows = []
    for c in range(n_channels):
        idx = detect_spikes_on_trace(data[:, c], fs, threshold, refractory_ms)
        for i in idx:
            rows.append((i / fs, channel_ids[c]))
    spikes = pd.DataFrame(rows, columns=["time", "channel"])
    return spikes.sort_values("time").reset_index(drop=True)


def detect_spikes_recording(recording, t_start=0.0, duration=5.0, threshold=5.0,
                            refractory_ms=1.0, chunk=100):
    """Detect spikes on ALL channels of a recording over `duration` seconds.

    Reads and filters `chunk` channels at a time (all ~1000 at once would need
    several GB of RAM). Returns the same spike table as `detect_spikes`, with
    times relative to `t_start`.
    """
    fs = recording.get_sampling_frequency()
    channel_ids = np.asarray(recording.get_channel_ids())
    tables = []
    for i in range(0, len(channel_ids), chunk):
        ids = channel_ids[i:i + chunk]
        _, data = get_traces(recording, t_start, t_start + duration, channel_ids=ids)
        tables.append(detect_spikes(bandpass_filter(data, fs), fs, channel_ids=ids,
                                    threshold=threshold, refractory_ms=refractory_ms))
    spikes = pd.concat(tables, ignore_index=True)
    return spikes.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Spike-train features  (chapter step 2)
# ---------------------------------------------------------------------------

def firing_rate(spike_times, duration):
    """Mean firing rate in Hz = number of spikes / duration (s)."""
    spike_times = np.asarray(spike_times)
    if duration <= 0:
        return np.nan
    return spike_times.size / duration


def firing_rate_per_channel(spikes, duration):
    """Firing rate (Hz) per channel. `spikes` is the DataFrame from detect_spikes."""
    counts = spikes.groupby("channel").size()
    return (counts / duration).rename("firing_rate_hz")


def fano_factor(spike_times, duration, bin_s=1.0):
    """Fano factor = var / mean of spike counts in fixed `bin_s`-second windows.

    ~1 for a Poisson process, < 1 for more regular firing, > 1 for bursty firing.
    """
    spike_times = np.asarray(spike_times)
    edges = np.arange(0, duration + bin_s, bin_s)
    counts, _ = np.histogram(spike_times, bins=edges)
    if counts.mean() == 0:
        return np.nan
    return counts.var() / counts.mean()


def cv_isi(spike_times):
    """Coefficient of variation of inter-spike intervals (SD / mean of ISIs).

    ~1 for Poisson firing, near 0 for clock-like regular firing, > 1 for bursting.
    """
    spike_times = np.sort(np.asarray(spike_times))
    if spike_times.size < 3:
        return np.nan
    isi = np.diff(spike_times)
    if isi.mean() == 0:
        return np.nan
    return isi.std() / isi.mean()


def isi_violations(spike_times, refractory_ms=1.5):
    """Percentage of inter-spike intervals shorter than the refractory period.

    A real single neuron cannot fire twice within its refractory period (~1-2 ms),
    so a high value means the "unit" probably mixes spikes from several neurons.
    This is the simplest form of the refractory-period check (< 1 % is a rule of
    thumb; thresholds vary between labs). For a contamination *estimate* see
    Hill et al. 2011 (J Neurosci 31:8699) and spikeinterface's
    `compute_isi_violations` / `compute_refrac_period_violations`.
    """
    spike_times = np.sort(np.asarray(spike_times))
    if spike_times.size < 2:
        return np.nan
    isi = np.diff(spike_times)
    return 100.0 * np.mean(isi < refractory_ms / 1e3)


def isi_violation_ratio(spike_times, duration, refractory_ms=1.5):
    """Hill et al. (2011) contamination estimate from refractory violations.

    C = n_v * T / (2 * N^2 * t_r): n_v intervals shorter than t_r, N spikes,
    T recording duration (s). Assuming the contaminating spikes arrive at random,
    C approximates the fraction of the unit's spikes that are contamination
    (0.5 = contaminating spikes at half the rate of the true ones; it can exceed 1).
    This is spikeinterface's `isi_violations_ratio` (from UltraMegaSort2000); the
    Allen Institute keeps units with C < 0.5.
    """
    spike_times = np.sort(np.asarray(spike_times))
    n = spike_times.size
    if n < 2:
        return np.nan
    n_v = np.sum(np.diff(spike_times) < refractory_ms / 1e3)
    return n_v * duration / (2 * n ** 2 * refractory_ms / 1e3)


def unit_features(spikes, duration, sorting=None, recording=None):
    """One row per unit with its single-cell features.

    From the spike times (`spikes` = DataFrame from `sorting_to_spikes`):
      n_spikes, firing_rate_hz, cv_isi, fano, isi_violation_pct and
      isi_violation_ratio (Hill et al.).
    If `sorting` carries Kilosort info (see `load_sorting`), also: ks_label
      ("good"/"mua"), footprint_size (channels) and, with `recording`, the
      unit position x, y (um).
    """
    rows = []
    for uid, st in spikes.groupby("unit")["time"]:
        st = st.to_numpy()
        rows.append({
            "unit": uid,
            "n_spikes": st.size,
            "firing_rate_hz": firing_rate(st, duration),
            "cv_isi": cv_isi(st),
            "fano": fano_factor(st, duration, bin_s=1.0),
            "isi_violation_pct": isi_violations(st),
            "isi_violation_ratio": isi_violation_ratio(st, duration),
        })
    feats = pd.DataFrame(rows).set_index("unit")

    if sorting is not None and "ks_label" in sorting.get_property_keys():
        ids = np.asarray(sorting.get_unit_ids())
        feats["ks_label"] = pd.Series(sorting.get_property("ks_label"), index=ids)
    if sorting is not None and "footprint_size" in sorting.get_property_keys():
        ids = np.asarray(sorting.get_unit_ids())
        feats["footprint_size"] = pd.Series(sorting.get_property("footprint_size"), index=ids)
        if "footprint_channels" in sorting.get_property_keys():
            feats["footprint_channels"] = pd.Series(sorting.get_property("footprint_channels"),
                                                    index=ids)
        if recording is not None:
            peak = sorting.get_property("peak_channel_index")
            locs = np.asarray(recording.get_channel_locations())[peak]
            feats["x"] = pd.Series(locs[:, 0], index=ids)
            feats["y"] = pd.Series(locs[:, 1], index=ids)
    return feats


def curate_units(feats, max_isi_violation_pct=1.0):
    """Keep the units we trust as single neurons.

    Two checks: Kilosort labelled the unit "good" (if the label is available),
    and fewer than `max_isi_violation_pct` % of its inter-spike intervals are
    shorter than the 1.5 ms refractory period. Returns the matching rows of
    `feats` (from `unit_features`).
    """
    keep = feats["isi_violation_pct"] < max_isi_violation_pct
    if "ks_label" in feats.columns:
        keep &= feats["ks_label"] == "good"
    return feats[keep]


def curate_units_hill(feats, max_ratio=0.5):
    """Alternative curation: Kilosort "good" and Hill et al.'s contamination
    estimate (`isi_violation_ratio`) below `max_ratio` (0.5 = Allen default)."""
    keep = feats["isi_violation_ratio"] < max_ratio
    if "ks_label" in feats.columns:
        keep &= feats["ks_label"] == "good"
    return feats[keep]


def plot_isi_violations(feats, max_isi_violation_pct=1.0, xmax=5.0, bins=25,
                        metric="isi_violation_pct"):
    """Histograms of the % refractory violations (ISIs < 1.5 ms) over units.

    With `metric="isi_violation_ratio"` it shows Hill et al.'s contamination
    estimate instead; `max_isi_violation_pct` is then the limit on that ratio.

    One panel per Kilosort label (good, mua) when available, with the same x
    axis. Solid bars are the units curation keeps (`good` and below the limit);
    light hatched bars are excluded: `good` units above the limit (grey band)
    and every `mua` unit. The axis stops at `xmax` %, and units beyond it are
    counted in the last bar. Returns the axes.
    """
    plt = _plt()
    ink, muted = "#0b0b0b", "#52514e"
    colors = {"good": "#2a78d6", "mua": "#eb6834"}
    v = feats[metric]
    hill = metric == "isi_violation_ratio"
    xlabel = ("contamination estimate (Hill et al.)" if hill
              else "ISIs shorter than 1.5 ms (% of the unit's intervals)")
    limit_txt = f" limit {max_isi_violation_pct:g}" if hill else f" {max_isi_violation_pct:g}% limit"
    if "ks_label" in feats.columns:
        labels = sorted(feats["ks_label"].unique(), key=lambda l: (l not in colors, l))
        groups = [(l, v[feats["ks_label"] == l].dropna()) for l in labels]
    else:
        groups = [("all units", v.dropna())]

    _, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 3.8),
                           sharex=True, squeeze=False)
    axes = axes[0]
    edges = np.linspace(0, xmax, bins + 1)
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1)
    for ax, (label, vals) in zip(axes, groups):
        color = colors.get(label, colors["good"])
        label_ok = label in ("good", "all units")
        keep = (vals < max_isi_violation_pct) & label_ok
        x = np.clip(vals.to_numpy(), 0, xmax - 1e-9)
        ax.axvspan(max_isi_violation_pct, xmax, color="#ececea", zorder=0)
        if keep.any():
            ax.hist(x[keep.to_numpy()], bins=edges, color=color, edgecolor="white",
                    linewidth=1, label=f"kept ({int(keep.sum())})")
        ax.hist(x[~keep.to_numpy()], bins=edges, color=color, alpha=0.3, hatch="///",
                edgecolor="white", linewidth=1, label=f"excluded ({int((~keep).sum())})")
        ax.axvline(max_isi_violation_pct, color=ink, lw=1.2, ls="--")
        ax.text(max_isi_violation_pct, 0.97, limit_txt,
                transform=ax.get_xaxis_transform(), color=ink, fontsize=9,
                va="top", ha="left", bbox=box)
        if not label_ok:
            ax.text(0.5, 0.5, f"all excluded:\nnot Kilosort 'good'", transform=ax.transAxes,
                    color=muted, fontsize=10, ha="center", va="center", bbox=box)
        if (vals > xmax).any():
            n_over = int((vals > xmax).sum())
            ax.text(0.99, 0.62, f"{n_over} unit{'s' if n_over > 1 else ''} > {xmax:g}{'' if hill else '%'}\n(in last bar)",
                    transform=ax.transAxes, color=muted, fontsize=8, ha="right", bbox=box)
        name = f"Kilosort '{label}'" if label != "all units" else label
        ax.set_title(f"{name} (n={len(vals)})")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax)
        ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.82))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Units")
    return axes


# ---------------------------------------------------------------------------
# 7. Network activity & bursts  (chapter step 3)
# ---------------------------------------------------------------------------

def population_rate(spikes, duration, bin_s=0.025, smooth_bins=5):
    """Population firing rate: all spikes pooled, binned, and smoothed.

    Returns
    -------
    t : bin-centre times (s)
    rate : population rate in spikes/s
    """
    times = spikes["time"].to_numpy() if hasattr(spikes, "columns") else np.asarray(spikes)
    edges = np.arange(0, duration + bin_s, bin_s)
    counts, _ = np.histogram(times, bins=edges)
    rate = counts / bin_s
    if smooth_bins and smooth_bins > 1:
        kernel = np.ones(smooth_bins) / smooth_bins
        rate = np.convolve(rate, kernel, mode="same")
    centres = edges[:-1] + bin_s / 2
    return centres, rate


def detect_network_bursts(t, rate, threshold=None, threshold_factor=3.0,
                          min_duration_s=0.05):
    """Detect network bursts as periods where the population rate exceeds a threshold.

    Parameters
    ----------
    t, rate : output of `population_rate`
    threshold : absolute rate threshold (spikes/s). If None, it is set to
        `threshold_factor` times the median rate (a robust baseline).
    min_duration_s : bursts shorter than this are discarded.

    Returns
    -------
    bursts : DataFrame with columns start, end, duration, peak_rate.
    threshold : the threshold that was used.
    """
    rate = np.asarray(rate)
    if threshold is None:
        threshold = threshold_factor * np.median(rate)

    above = rate > threshold
    # Rising/falling edges of the "above threshold" mask.
    edges = np.diff(above.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]

    rows = []
    for s, e in zip(starts, ends):
        t0, t1 = t[s], t[e - 1]
        if t1 - t0 >= min_duration_s:
            rows.append((t0, t1, t1 - t0, rate[s:e].max()))
    bursts = pd.DataFrame(rows, columns=["start", "end", "duration", "peak_rate"])
    return bursts, threshold


def burst_features(bursts, duration, spikes=None, n_channels=None):
    """Summarise network bursts into the features from the chapter.

    Returns a dict with:
      burst_rate_per_min, mean_burst_duration_s, mean_ibi_s, cv_ibi,
      and (if `spikes` + `n_channels` given) mean channel participation.
    """
    feats = {}
    n = len(bursts)
    feats["n_bursts"] = n
    feats["burst_rate_per_min"] = n / (duration / 60.0) if duration > 0 else np.nan
    feats["mean_burst_duration_s"] = bursts["duration"].mean() if n else np.nan

    if n >= 2:
        ibi = bursts["start"].to_numpy()[1:] - bursts["end"].to_numpy()[:-1]
        feats["mean_ibi_s"] = ibi.mean()
        feats["cv_ibi"] = ibi.std() / ibi.mean() if ibi.mean() > 0 else np.nan
    else:
        feats["mean_ibi_s"] = np.nan
        feats["cv_ibi"] = np.nan

    if spikes is not None and n_channels and n:
        # Participation per unit (SUA) or per channel (MUA), as in plot_raster.
        key = "unit" if "unit" in spikes.columns else "channel"
        parts = []
        for _, b in bursts.iterrows():
            in_burst = spikes[(spikes["time"] >= b["start"]) & (spikes["time"] <= b["end"])]
            parts.append(in_burst[key].nunique() / n_channels)
        feats["mean_participation"] = float(np.mean(parts))
    return feats


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

def plot_traces(times, data, offset=200.0, ax=None, color="k", lw=0.5, labels=None):
    """Plot multi-channel traces stacked with a vertical offset (microvolts).

    `labels` names the traces on the y axis (e.g. the channel ids); default 0, 1, ...
    """
    if ax is None:
        _, ax = _plt().subplots(figsize=(10, 6))
    data = np.atleast_2d(data.T).T if data.ndim == 1 else data
    for c in range(data.shape[1]):
        ax.plot(times, data[:, c] + c * offset, color=color, lw=lw)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (offset)")
    ax.set_yticks([c * offset for c in range(data.shape[1])])
    ax.set_yticklabels(labels if labels is not None else range(data.shape[1]))
    return ax


def plot_spike_detection(trace, fs, threshold=5.0, ax=None):
    """Plot one filtered trace with the detection threshold and detected spikes."""
    if ax is None:
        _, ax = _plt().subplots(figsize=(10, 3))
    t = np.arange(trace.size) / fs
    sigma = noise_std_mad(trace)
    idx = detect_spikes_on_trace(trace, fs, threshold)
    ax.plot(t, trace, "k", lw=0.5)
    ax.axhline(-threshold * sigma, color="r", ls="--", lw=1,
               label=f"-{threshold:g}x MAD")
    ax.plot(idx / fs, trace[idx], "rv", ms=5, label="detected spikes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend(loc="upper right", frameon=False)
    return ax


def plot_raster(spikes, duration=None, ax=None, marker="|", color="k", key=None):
    """Raster plot of spikes, one row per channel (MUA) or per unit (SUA).

    `key` selects the row grouping column; if None it auto-picks "unit" if
    present, else "channel".
    """
    if ax is None:
        _, ax = _plt().subplots(figsize=(10, 6))
    if key is None:
        key = "unit" if "unit" in spikes.columns else "channel"
    rows_ = np.sort(spikes[key].unique())
    row = {ch: i for i, ch in enumerate(rows_)}
    y = spikes[key].map(row).to_numpy()
    ax.scatter(spikes["time"], y, marker=marker, s=20, c=color, linewidths=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(key.capitalize())
    if duration:
        ax.set_xlim(0, duration)
    return ax


def plot_population_raster(spikes, recording, duration=None, ax=None, match=None):
    """Raster of all recorded channels, one row each, ordered by y position.

    Ordering the rows bottom-to-top by the electrode's y position keeps them in
    roughly the same order as the activity map: put the map at the right and
    pass its axes as `match`, and the raster takes over its height.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ids = np.asarray(recording.get_channel_ids())
    locs = np.asarray(recording.get_channel_locations())
    order = np.lexsort((locs[:, 0], locs[:, 1]))           # by y, then x
    row = pd.Series(np.arange(len(ids)), index=ids[order])
    ax.scatter(spikes["time"], spikes["channel"].map(row), marker="|", s=4,
               c="k", linewidths=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (sorted by y position)")
    ax.set_ylim(-0.5, len(ids) - 0.5)
    if duration:
        ax.set_xlim(0, duration)
    if match is not None:
        ax.figure.canvas.draw()                    # let the map settle its final size
        r, m = ax.get_position(), match.get_position()
        ax.set_position([r.x0, m.y0, r.width, m.height])
    return ax


def plot_config_map(recording, highlight=None, ax=None):
    """Scatter of the routed electrode positions (um); optionally highlight some."""
    if ax is None:
        _, ax = _plt().subplots(figsize=(6, 6))
    locs = recording.get_channel_locations()
    ax.scatter(locs[:, 0], locs[:, 1], s=8, c="lightgray", label="electrodes")
    if highlight is not None:
        ids = np.asarray(recording.get_channel_ids())
        mask = np.isin(ids, np.asarray(highlight))
        ax.scatter(locs[mask, 0], locs[mask, 1], s=20, c="crimson", label="selected")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# 9. Power spectrum  (why we filter)
# ---------------------------------------------------------------------------

def power_spectrum(trace, fs, nperseg=None):
    """Power spectral density of a single trace (Welch's method).

    Returns (freqs_hz, power). The spectrum shows *how much* signal sits at each
    frequency -- handy for spotting mains hum (~50 Hz) and slow drift (low Hz),
    and for choosing a sensible filter band.
    """
    if nperseg is None:
        nperseg = int(min(len(trace), fs))  # ~1 s windows
    f, pxx = welch(np.asarray(trace, dtype=float), fs=fs, nperseg=nperseg)
    return f, pxx


def line_noise_ratio(f, power, f0=50.0, half_width=10.0):
    """How far the power at `f0` (e.g. 50 Hz mains) stands above its neighbours.

    = power in the bin closest to `f0` / median power within +-`half_width` Hz.
    Line noise is a *narrow* peak, so a ratio near 1 means there is none; a
    ratio of ~10 or more means a clear line-noise peak.
    """
    f, power = np.asarray(f), np.asarray(power)
    at_f0 = power[np.argmin(np.abs(f - f0))]
    around = power[(f >= f0 - half_width) & (f <= f0 + half_width)]
    return float(at_f0 / np.median(around))


def plot_power_spectrum(trace, fs, ax=None, flim=(1, 5000), mark_hz=None, color="k"):
    """Plot the power spectrum of a trace on log-log axes over a frequency range.

    `mark_hz` draws dotted reference lines, e.g. (50, 60) for mains frequencies.
    """
    if ax is None:
        _, ax = _plt().subplots(figsize=(8, 4))
    f, pxx = power_spectrum(trace, fs)
    m = (f >= flim[0]) & (f <= flim[1])
    ax.loglog(f[m], pxx[m], color=color, lw=0.8)
    ax.set_ylim(bottom=pxx[m].max() * 1e-6)              # 6 decades: hide numerical floor
    for k, hz in enumerate(mark_hz or ()):
        ax.axvline(hz, color="#52514e", lw=1, ls=":")
        ax.text(hz, 0.98 - 0.08 * (k % 2), f" {hz:g} Hz", transform=ax.get_xaxis_transform(),
                color="#52514e", fontsize=8, va="top", ha="left")   # staggered: 50/60 overlap
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (uV^2/Hz)")
    return ax


# ---------------------------------------------------------------------------
# 10. Waveforms & footprints  (what a spike looks like in space)
# ---------------------------------------------------------------------------

def bandpass_recording(recording, low=300.0, high=3000.0):
    """Return a *lazily* band-pass filtered view of a recording (spikeinterface).

    Reading cutouts from this is cleaner than filtering tiny snippets by hand.
    """
    import spikeinterface.preprocessing as spre
    # MaxWell stores uint16 samples; spikeinterface filters need signed ints.
    if np.dtype(recording.get_dtype()).kind == "u":
        recording = spre.unsigned_to_signed(recording)
    # float output: kept as integers, the filtered signal would be rounded to whole
    # ADC steps (~6 uV), coarser than the noise itself (~4-5 uV).
    return spre.bandpass_filter(recording, freq_min=low, freq_max=high, dtype="float32")


def channel_noise(recording, t_start=0.0, duration=2.0, low=300.0, high=3000.0):
    """Two noise estimates (uV) of the band-pass filtered signal on every channel.

    rms_uv : root mean square -- counts *everything*, spikes included.
    mad_uv : median(|x|) / 0.6745 (see `noise_std_mad`) -- barely affected by
             the (sparse, large) spikes.
    On a quiet electrode the two agree; spikes push the RMS above the MAD.
    Uses a short `duration`-second window. Returns a DataFrame indexed by channel id.
    """
    rec_f = bandpass_recording(recording, low=low, high=high)
    fs = rec_f.get_sampling_frequency()
    data = np.asarray(_get_traces_uv(rec_f, start_frame=int(t_start * fs),
                                     end_frame=int((t_start + duration) * fs)), dtype=float)
    return pd.DataFrame({"rms_uv": np.sqrt(np.mean(data ** 2, axis=0)),
                         "mad_uv": np.median(np.abs(data), axis=0) / 0.6745},
                        index=rec_f.get_channel_ids())


def channel_rms(recording, t_start=0.0, duration=2.0, low=300.0, high=3000.0):
    """RMS (uV) of the band-pass filtered signal on every channel.

    Uses a short `duration`-second window (enough for a stable estimate). In the
    spike band the RMS is dominated by noise, so this is each electrode's noise
    level -- what the spike-detection threshold will be scaled by.
    Returns a Series indexed by channel id.
    """
    return channel_noise(recording, t_start, duration, low, high)["rms_uv"]


def plot_noise_comparison(noise, rate_hz=None, ax=None):
    """Scatter of RMS against the MAD noise estimate, one dot per electrode.

    `noise` is the DataFrame from `channel_noise`; `rate_hz` (optional, Series
    by channel) colours the dots by firing rate. Dots on the dashed diagonal
    have RMS = MAD (no spikes to speak of); spikes lift the RMS above it.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    x, y = noise["mad_uv"], noise["rms_uv"]
    lim = [0, float(max(x.max(), y.max())) * 1.05]
    ax.plot(lim, lim, color="#52514e", lw=1, ls="--", zorder=1)
    ax.text(0.8 * lim[1], 0.76 * lim[1], "RMS = MAD", color="#52514e", fontsize=8,
            ha="left", va="top")                         # just below the line, clear of the title
    if rate_hz is not None:
        rate = rate_hz.reindex(noise.index).fillna(0).clip(lower=0.01)
        order = np.argsort(rate.to_numpy())                    # busiest drawn on top
        sc = ax.scatter(x.iloc[order], y.iloc[order], c=rate.iloc[order], s=14,
                        cmap=sequential_cmap(), norm=plt.matplotlib.colors.LogNorm(),
                        edgecolors="white", linewidths=0.3, zorder=2)
        _map_colorbar(ax, sc, "firing rate (Hz)")
    else:
        ax.scatter(x, y, s=14, color="#2a78d6", edgecolors="white", linewidths=0.3, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("MAD noise estimate (uV)")
    ax.set_ylabel("RMS (uV)")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_rms_distribution(rms, ax=None, bins=40):
    """Histogram of the per-electrode noise RMS, with the median marked.

    A dotted line at 2x the median flags unusually noisy electrodes.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    rms = np.asarray(rms, dtype=float)
    med = np.median(rms)
    ax.hist(rms, bins=bins, color="#1f5fbf", edgecolor="white", linewidth=1)   # filtered = blue
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1)
    ax.axvline(med, color="#0b0b0b", lw=1.2, ls="--")
    ax.text(med, 1.0, f" median {med:.2f} uV", transform=ax.get_xaxis_transform(),
            color="#0b0b0b", fontsize=9, va="top", ha="left", bbox=box)
    if rms.max() > 2 * med:
        ax.axvline(2 * med, color="#52514e", lw=1, ls=":")
        ax.text(2 * med, 0.86, f" 2x median ({(rms > 2 * med).sum()} electrodes above)",
                transform=ax.get_xaxis_transform(), color="#52514e", fontsize=8,
                va="top", ha="left", bbox=box)
    ax.set_xlabel("Noise RMS, 300-3000 Hz (uV)")
    ax.set_ylabel("Electrodes")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def neighbouring_channels(recording, center_id, n=9, radius_um=None):
    """The `n` channels closest (in x,y space) to `center_id`, including it.

    With `radius_um`, returns all channels within that distance instead -- better
    for HD-MEAs, where the routed electrodes come in patches and the "n closest"
    can be hundreds of um away.

    Returns (channel_ids, locations). Used to build a spatial footprint.
    """
    ids = np.asarray(recording.get_channel_ids())
    locs = np.asarray(recording.get_channel_locations())
    ci = int(np.flatnonzero(ids == center_id)[0])
    dist = np.linalg.norm(locs - locs[ci], axis=1)
    order = np.argsort(dist)
    order = order[dist[order] <= radius_um] if radius_um is not None else order[:n]
    return ids[order], locs[order]


def extract_waveforms(data, spike_samples, fs, pre_ms=1.0, post_ms=2.0):
    """Cut out short windows of `data` around each spike sample.

    `data` is a 1-D trace (one channel). Returns (waveforms, t_ms) where
    waveforms has shape (n_spikes, n_window_samples) and t_ms is the time axis
    in milliseconds relative to the spike (0 = spike sample).
    """
    data = np.asarray(data, dtype=float)
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    wfs = [data[s - pre:s + post] for s in spike_samples
           if s - pre >= 0 and s + post <= len(data)]
    wfs = np.array(wfs) if wfs else np.empty((0, pre + post))
    t_ms = np.arange(-pre, post) / fs * 1e3
    return wfs, t_ms


def waveform_metrics(mean_wf, fs):
    """Simple shape metrics of a (negative) spike waveform.

    Returns dict:
      amplitude_uv     : value at the trough (most negative point), in uV
      trough_to_peak_ms: time from the trough to the following positive peak
    """
    wf = np.asarray(mean_wf, dtype=float)
    trough = int(np.argmin(wf))
    after = wf[trough:]
    peak = trough + int(np.argmax(after)) if after.size else trough
    return {
        "amplitude_uv": float(wf[trough]),
        "trough_to_peak_ms": float((peak - trough) / fs * 1e3),
    }


def extract_footprint(recording_f, spike_times, channel_ids,
                      pre_ms=1.0, post_ms=2.0, max_spikes=200, seed=0):
    """Mean spike waveform on each of `channel_ids` (a spatial 'footprint').

    `recording_f` should be the (lazily) band-pass filtered recording. Reads a
    short window around up to `max_spikes` spikes, averages them per channel.

    Returns (mean_wf, t_ms) with mean_wf shape (n_window_samples, n_channels).
    """
    fs = recording_f.get_sampling_frequency()
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    n_frames = recording_f.get_num_frames()

    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size > max_spikes:
        rng = np.random.default_rng(seed)
        spike_times = rng.choice(spike_times, max_spikes, replace=False)

    acc = None
    count = 0
    for t in spike_times:
        s = int(round(t * fs))
        if s - pre < 0 or s + post > n_frames:
            continue
        tr = _get_traces_uv(recording_f, start_frame=s - pre, end_frame=s + post,
                            channel_ids=channel_ids)
        tr = np.asarray(tr, dtype=float)
        acc = tr if acc is None else acc + tr
        count += 1
    if count == 0:
        raise RuntimeError("No spikes far enough from the recording edges.")
    mean_wf = acc / count
    t_ms = np.arange(-pre, post) / fs * 1e3
    return mean_wf, t_ms


def unit_mean_waveform(recording_f, sorting, unit_id, max_spikes=100,
                       pre_ms=1.0, post_ms=2.0):
    """Mean waveform of one sorted unit, on its strongest (peak) channel.

    Returns (t_ms, waveform_uv, peak_channel_id).
    """
    fs = sorting.get_sampling_frequency()
    st = sorting.get_unit_spike_train(unit_id=unit_id) / fs
    ch_ids = recording_f.get_channel_ids()
    mean_wf, t_ms = extract_footprint(recording_f, st, ch_ids,
                                      pre_ms, post_ms, max_spikes)
    peak = int(np.argmin(mean_wf.min(axis=0)))  # channel with deepest trough
    return t_ms, mean_wf[:, peak], np.asarray(ch_ids)[peak]


# ---------------------------------------------------------------------------
# 11. More plotting (footprint, activity map)
# ---------------------------------------------------------------------------

def plot_footprint(mean_wf, t_ms, locs, ax=None, x_scale=None, y_scale=None):
    """Draw each channel's mean waveform at its spatial (x,y) position.

    Each waveform is centred on its electrode and coloured by its amplitude;
    a * marks the peak channel (deepest trough). Scale bars show 1 ms
    and the voltage scale. `x_scale` (um per ms) and `y_scale` (um per uV)
    default to fitting the waveforms to the electrode pitch.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    locs = np.asarray(locs, dtype=float)
    mean_wf = np.asarray(mean_wf, dtype=float)
    mean_wf = mean_wf - np.median(mean_wf, axis=0)        # baseline at 0 uV
    amp = -mean_wf.min(axis=0)                            # trough depth per channel
    peak = int(np.argmax(amp))

    # Electrode pitch = typical nearest-neighbour distance (17.5 um on MaxWell).
    if len(locs) > 1:
        d = np.linalg.norm(locs[:, None] - locs[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        pitch = float(np.median(d.min(axis=1)))
    else:
        pitch = 17.5
    if x_scale is None:
        x_scale = 0.9 * pitch / np.ptp(t_ms)              # one waveform ~ one pitch wide
    if y_scale is None:
        y_scale = 1.5 * pitch / (amp[peak] or 1.0)        # deepest trough ~ 1.5 pitch
    tx = np.asarray(t_ms) * x_scale                       # spike time (0 ms) on the electrode

    ax.scatter(locs[:, 0], locs[:, 1], s=12, c="0.85", zorder=0)
    cmap, norm = plt.cm.viridis_r, plt.Normalize(0, amp[peak])
    for c in np.argsort(amp):                             # strongest drawn on top
        ax.plot(locs[c, 0] + tx, locs[c, 1] + mean_wf[:, c] * y_scale,
                color=cmap(norm(amp[c])), lw=1.6 if c == peak else 1.0)
    ax.scatter(locs[peak, 0], locs[peak, 1], marker="*", s=220, c="black",
               edgecolors="white", linewidths=1, zorder=5, label="peak channel")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, shrink=0.6, label="trough amplitude (uV)")

    # Scale bars in the lower-left corner: 1 ms and a round number of uV.
    uv_bar = float(np.round(amp[peak] / 2, -1) or np.round(amp[peak] / 2) or 1)
    x0 = locs[:, 0].min() - pitch
    y0 = locs[:, 1].min() - 2 * pitch
    ax.plot([x0, x0 + x_scale], [y0, y0], "k", lw=2)
    ax.plot([x0, x0], [y0, y0 + uv_bar * y_scale], "k", lw=2)
    ax.text(x0 + x_scale / 2, y0 - 0.3 * pitch, "1 ms", ha="center", va="top")
    ax.text(x0 + 0.2 * pitch, y0 + uv_bar * y_scale / 2, f"{uv_bar:g} uV",
            ha="left", va="center")                       # right of the bar: clear of the axis

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    ax.margins(0.08)
    return ax


# Single-unit features that `plot_unit_features` can show:
# column -> (axis label, log scale?, (reference value, reference label) or None,
#            colour family: one single-hue colormap per feature)
UNIT_FEATURES = {
    "footprint_size": ("Footprint size (channels)", False, None, "Greys"),
    "firing_rate_hz": ("Firing rate (Hz)", True, None, "Blues"),
    "cv_isi": ("CV of ISI", False, (1.0, "Poisson"), "Greens"),
    "fano": ("Fano factor (1 s bins)", True, (1.0, "Poisson"), "Oranges"),
    "n_spikes": ("Number of spikes", True, None, "Reds"),
    "isi_violation_pct": ("ISI < 1.5 ms (%)", False, (1.0, "1% limit"), "Greys"),
    "isi_violation_ratio": ("Contamination (Hill et al.)", False, (0.5, "Allen limit"), "PuRd"),
}


def plot_unit_features(feats, recording=None,
                       features=("footprint_size", "firing_rate_hz", "cv_isi", "fano"),
                       bins=30):
    """Distributions of single-unit features (top row) and where they are on
    the chip (bottom row), one column per feature.

    `features` picks the columns of `feats` to show, from `UNIT_FEATURES`
    (e.g. also "n_spikes", "isi_violation_pct", "isi_violation_ratio"); each
    feature has its own colour, shared by its histogram and its map.
    Top: histograms with the median marked; dotted lines show a reference value
    (1 = Poisson for CV and Fano). Log-scaled features use log bins and a log
    colour scale. Bottom (with `recording` and unit positions in `feats`): each
    unit at its position, coloured by the feature above it — except the
    footprint size, where each unit's footprint electrodes are drawn in its own
    colour.
    """
    plt = _plt()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ink, muted = "#0b0b0b", "#52514e"
    unknown = [f for f in features if f not in UNIT_FEATURES or f not in feats.columns]
    if unknown:
        raise ValueError(f"unknown features {unknown}; choose from "
                         f"{[f for f in UNIT_FEATURES if f in feats.columns]}")
    show_map = recording is not None and {"x", "y"} <= set(feats.columns)
    nrows = 2 if show_map else 1
    if len(features) == 1 and show_map:               # one feature: histogram and map side by side
        fig, (h_ax, m_ax) = plt.subplots(1, 2, figsize=(12, 3.6), width_ratios=[1, 1.5])
        top, bottom = [h_ax], [m_ax]
    else:
        fig, axes = plt.subplots(nrows, len(features), figsize=(4.4 * len(features), 3.2 * nrows),
                                 squeeze=False, height_ratios=[1.15, 1] if show_map else None)
        top, bottom = list(axes[0]), list(axes[1]) if show_map else []
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1)
    if show_map:
        locs = np.asarray(recording.get_channel_locations())

    for k, col in enumerate(features):
        label, logx, ref, cmap_name = UNIT_FEATURES[col]
        bar = plt.colormaps[cmap_name](0.75)            # the feature's colour, histogram and map
        ax = top[k]
        v = feats[col].to_numpy(dtype=float)
        v = v[np.isfinite(v) & ((v > 0) if logx else True)]
        if logx:
            edges = np.logspace(np.log10(v.min()), np.log10(v.max()), bins + 1)
            ax.set_xscale("log")
        elif col == "footprint_size":
            edges = np.arange(v.min() - 0.5, v.max() + 1.5)     # one bar per channel count
        else:
            edges = bins
        med = np.median(v)
        ax.hist(v, bins=edges, color=bar, edgecolor="white", linewidth=1)
        ax.axvline(med, color=ink, lw=1.2, ls="--")
        ax.text(med, 1.0, f" median {med:.3g}", transform=ax.get_xaxis_transform(),
                color=ink, fontsize=9, va="top", ha="left", bbox=box)
        if ref is not None and v.min() < ref[0] < v.max():
            ax.axvline(ref[0], color=muted, lw=1, ls=":")
            ax.text(ref[0], 0.86, f" {ref[1]}", transform=ax.get_xaxis_transform(),
                    color=muted, fontsize=8, va="top", ha="left", bbox=box)
        ax.set_xlabel(label)
        ax.set_ylabel("Units" if k == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)

        if not show_map:
            continue
        ax = bottom[k]
        ax.scatter(locs[:, 0], locs[:, 1], s=1, c="0.88", zorder=0)
        if col == "footprint_size" and "footprint_channels" in feats.columns:
            # identity, not magnitude: each unit's footprint electrodes in its own colour
            palette = plt.colormaps["tab20"](np.r_[0:14, 16:20])   # without its two greys
            colours = palette[np.arange(len(feats)) % len(palette)]
            for colour, chans in zip(colours, feats["footprint_channels"]):
                idx = [int(c) for c in str(chans).split(",") if c]
                ax.scatter(locs[idx, 0], locs[idx, 1], s=6, color=colour, linewidths=0)
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
            cax.set_visible(False)                       # same width as the maps with a colorbar
            ax.set_title("each unit's footprint, own colour", fontsize=9, color=muted)
        else:
            vals = feats[col].astype(float)
            ok = np.isfinite(vals) & ((vals > 0) if logx else True)
            lo, hi = np.nanpercentile(vals[ok], [2, 98])      # a few extremes don't wash out the rest
            norm = (plt.matplotlib.colors.LogNorm if logx else plt.matplotlib.colors.Normalize)(lo, hi)
            sc = ax.scatter(feats.loc[ok, "x"], feats.loc[ok, "y"], c=vals[ok], s=10,
                            cmap=sequential_cmap(cmap_name), norm=norm, edgecolors="white",
                            linewidths=0.3)
            cbar = _map_colorbar(ax, sc, label)
            if logx:                                       # plain numbers, not 6x10^0
                tk = plt.matplotlib.ticker
                cbar.ax.yaxis.set_major_locator(tk.LogLocator(subs=(1, 2, 5)))
                cbar.ax.yaxis.set_major_formatter(tk.FuncFormatter(lambda x, _: f"{x:g}"))
                cbar.ax.yaxis.set_minor_formatter(tk.NullFormatter())
        ax.set_aspect("equal")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)" if k == 0 else "")
        if k:
            ax.tick_params(labelleft=False)
        if len(features) == 1:
            ax.set_title("on the chip: " + ax.get_title() if ax.get_title() else "on the chip",
                         fontsize=9, color=muted)

    if len(features) == 1:
        fig.suptitle(f"{UNIT_FEATURES[features[0]][0].split(' (')[0]} of {len(feats)} units",
                     color=ink)
    else:
        fig.suptitle(f"Single-unit features ({len(feats)} units)" +
                     (": distribution (top) and on the chip (bottom)" if show_map else ""),
                     color=ink)
    fig.tight_layout()
    return fig


UNIT_FEATURE_LABELS = {
    "firing_rate_hz": ("Firing rate (Hz)", True),      # label, log axis?
    "cv_isi": ("CV of ISI", False),
    "fano": ("Fano factor (1 s bins)", True),
}


def compare_unit_features(units, features=("firing_rate_hz", "cv_isi", "fano")):
    """Baseline vs Drug, per drug, on the per-unit feature distributions.

    `units` has one row per curated unit with columns Drug, Conditions and the
    `features`. Uses the Mann-Whitney U test: *unpaired* (baseline and drug
    units are not the same neurons) and rank-based (the features are skewed).
    `p_bonferroni` corrects for running several tests (p x number of tests).
    """
    from scipy.stats import mannwhitneyu
    rows = []
    for drug, d in units.groupby("Drug"):
        base = d[d["Conditions"] == "Baseline"]
        treated = d[d["Conditions"] == "Drug"]
        for f in features:
            x, y = base[f].dropna(), treated[f].dropna()
            p = mannwhitneyu(x, y).pvalue if len(x) and len(y) else np.nan
            rows.append({
                "Drug": drug, "feature": f,
                "n_baseline": len(x), "n_drug": len(y),
                "median_baseline": x.median(), "median_drug": y.median(),
                "pct_change_median": 100 * (y.median() - x.median()) / x.median(),
                "p_value": p,
            })
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    return out


def plot_unit_comparison(units, stats, features=("firing_rate_hz", "cv_isi", "fano"), seed=0):
    """Box plots of per-unit features, Baseline vs Drug, one column per drug.

    Every dot is one unit; boxes show the median and quartiles. Each panel is
    annotated with the % change of the median and the corrected p-value from
    `compare_unit_features`.
    """
    plt = _plt()
    colors = {"Baseline": "#2a78d6", "Drug": "#eb6834"}
    ink, muted = "#0b0b0b", "#52514e"
    drugs = sorted(units["Drug"].unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(features), len(drugs), figsize=(4.5 * len(drugs), 3.2 * len(features)),
                             squeeze=False, sharey="row")
    for r, f in enumerate(features):
        label, logy = UNIT_FEATURE_LABELS.get(f, (f, False))
        for c, drug in enumerate(drugs):
            ax = axes[r, c]
            d = units[units["Drug"] == drug]
            data = [d.loc[d["Conditions"] == cond, f].dropna().to_numpy() for cond in colors]
            if logy:
                data = [v[v > 0] for v in data]
                ax.set_yscale("log")
            for i, (cond, v) in enumerate(zip(colors, data)):
                ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=6, color=colors[cond],
                           alpha=0.35, linewidths=0, zorder=1)
            ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False, zorder=2,
                       medianprops=dict(color=ink, lw=2), boxprops=dict(color=ink),
                       whiskerprops=dict(color=ink), capprops=dict(color=ink))
            ax.set_xticks([0, 1], [f"{cond}\n(n={v.size})" for cond, v in zip(colors, data)])
            s = stats[(stats["Drug"] == drug) & (stats["feature"] == f)].iloc[0]
            p = s["p_bonferroni"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.set_title(f"Drug {drug}: median {s['pct_change_median']:+.0f}%, {ptxt}",
                         fontsize=9, color=muted)
            if c == 0:
                ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-unit features, Baseline vs Drug (p: Mann-Whitney, Bonferroni-corrected)",
                 color=ink)
    fig.tight_layout()
    return fig


def sequential_cmap(name="Blues"):
    """Light -> dark colour scale, without the near-white end (invisible on white).

    `name` is any matplotlib sequential map, e.g. "Blues" or "YlOrRd" (yellow -> red).
    """
    plt = _plt()
    return plt.matplotlib.colors.ListedColormap(plt.colormaps[name](np.linspace(0.3, 1.0, 256)))


def _map_colorbar(ax, mappable, label):
    """Colorbar exactly as tall as the (equal-aspect) electrode map next to it."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
    cbar = ax.figure.colorbar(mappable, cax=cax, label=label)
    _plt().sca(ax)              # keep the map current, so plt.title() etc. land on it
    return cbar


def plot_activity_map(recording, values, channel_ids=None, ax=None,
                      cmap="viridis", label="firing rate (Hz)", log=False,
                      s=25, show_all=False, robust=False):
    """Colour each electrode by a per-channel value (e.g. firing rate) at its x,y.

    `log=True` uses a logarithmic colour scale (for values spanning orders of
    magnitude); `show_all=True` draws electrodes without a value in light grey;
    `robust=True` sets the colour range to the 2nd-98th percentile so a few
    extreme electrodes don't wash out the rest.
    """
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ids = np.asarray(recording.get_channel_ids())
    locs = np.asarray(recording.get_channel_locations())
    if show_all:
        ax.scatter(locs[:, 0], locs[:, 1], s=s, c="0.88", zorder=0)
    if channel_ids is not None:
        idx = np.array([int(np.flatnonzero(ids == c)[0]) for c in channel_ids])
        locs = locs[idx]
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.nanpercentile(values, [2, 98]) if robust else (None, None)
    colors = plt.matplotlib.colors
    norm = (colors.LogNorm if log else colors.Normalize)(vmin=vmin, vmax=vmax)
    sc = ax.scatter(locs[:, 0], locs[:, 1], c=values, s=s, cmap=cmap, norm=norm)
    _map_colorbar(ax, sc, label)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    return ax


def online_activity_clip(recording, spikes, t_start=0.0, duration=10.0, bin_ms=50.0,
                         speed=0.5, s=10, dpi=85):
    """Interactive clip of the online-detected activity on the chip.

    Each frame is one `bin_ms` time bin: every electrode is coloured by how many
    spikes it detected in that bin (grey = none). Below, the population rate of
    the window with a cursor at the current frame. Plays at `speed` x real time
    (0.5 = 2x slower). Returns an interactive player (play/pause, slider, speed
    buttons) to show in the notebook. Every frame is stored in the notebook, so
    keep `duration` short (~10 s).
    """
    from IPython.display import HTML
    from matplotlib import animation
    plt = _plt()
    ink = "#0b0b0b"
    ids = np.asarray(recording.get_channel_ids())
    locs = np.asarray(recording.get_channel_locations())

    # spike counts per (frame, electrode)
    bin_s = bin_ms / 1e3
    edges = t_start + np.arange(int(round(duration / bin_s)) + 1) * bin_s
    sel = spikes[(spikes["time"] >= edges[0]) & (spikes["time"] < edges[-1])]
    ch = pd.Series(np.arange(len(ids)), index=ids).reindex(sel["channel"]).to_numpy()
    ok = ~np.isnan(ch)                                    # spikes on unrouted channels
    counts = np.histogram2d(sel["time"].to_numpy()[ok], ch[ok],
                            bins=[edges, np.arange(len(ids) + 1) - 0.5])[0]
    pop = counts.sum(axis=1) / bin_s                       # spikes/s over the chip

    # One colour per spike count (1, 2, ...); the top step means "this many or more".
    vmax = int(max(2, np.percentile(counts[counts > 0], 99))) if (counts > 0).any() else 2
    cmap = sequential_cmap().resampled(vmax)
    cmap.set_under("0.9")                                  # 0 spikes -> light grey
    cmap.set_over(cmap(vmax - 1))
    norm = plt.matplotlib.colors.BoundaryNorm(np.arange(0.5, vmax + 1), vmax)

    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), dpi=dpi,
                                     gridspec_kw=dict(height_ratios=[3.2, 1]))
    sc = ax_m.scatter(locs[:, 0], locs[:, 1], c=counts[0], s=s, cmap=cmap, norm=norm)
    cbar = _map_colorbar(ax_m, sc, f"spikes per {bin_ms:g} ms")
    cbar.set_ticks(np.arange(1, vmax + 1))
    cbar.set_ticklabels([str(k) for k in range(1, vmax)] + [f"{vmax}+"])
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("x (um)")
    ax_m.set_ylabel("y (um)")
    title = ax_m.set_title("Online spikes, t = 0.00-0.00 s", color=ink, loc="left")  # sized for layout
    ax_p.plot(edges[:-1] + bin_s / 2, pop, color="#2a78d6", lw=1)
    cursor = ax_p.axvline(edges[0], color=ink, lw=1.2)
    ax_p.set_xlim(edges[0], edges[-1])
    ax_p.set_ylim(bottom=0)
    ax_p.set_xlabel("Time (s)")
    ax_p.set_ylabel("Spikes/s")
    ax_p.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()                                     # line the trace up with the map
    m, p = ax_m.get_position(), ax_p.get_position()
    ax_p.set_position([m.x0, p.y0, m.width, p.height])

    def update(k):
        t = edges[k]
        sc.set_array(counts[k])
        cursor.set_xdata([t, t])
        title.set_text(f"Online spikes, t = {t:.2f}-{t + bin_s:.2f} s")
        return sc, cursor, title

    anim = animation.FuncAnimation(fig, update, frames=len(counts), interval=bin_ms / speed)
    # JPEG frames are ~4x smaller than PNG; the default 20 MB limit would cut clips short
    with plt.rc_context({"animation.frame_format": "jpeg", "animation.embed_limit": 100}):
        html = anim.to_jshtml(default_mode="once")
    plt.close(fig)                                        # no extra static copy
    return HTML(html)


# ---------------------------------------------------------------------------
# 12. Self-check helper (for the notebook exercises)
# ---------------------------------------------------------------------------

def check(student, expected, tol=1e-6, hint=""):
    """Compare a student's answer to the expected value and print feedback.

    Used by the notebook exercises so students get instant ✓ / ✗ feedback
    without anyone grading. Works on scalars or arrays.
    """
    try:
        ok = np.allclose(np.asarray(student, dtype=float),
                         np.asarray(expected, dtype=float),
                         rtol=1e-3, atol=tol, equal_nan=True)
    except Exception:
        ok = (student == expected)
    if ok:
        print("[OK] correct!")
    else:
        print("[X] not quite yet.")
        if hint:
            print("   hint:", hint)
        print(f"   you got: {student}")
        print(f"   expected ~ {expected}")
    return bool(ok)
