import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage


DATA_ROOT = Path("data/visualize_raw_samples_n70_original_fs/visualize_raw_samples_n70_original_fs")
SITES = {"pohang", "utah2019", "utah2023"}
SITE_ALIASES = {
    "utah2019": "utah_2019",
    "utah2023": "utah_2023",
}
LABEL_DIRS = {
    0: "0_noise",
    1: "1_event",
    2: "2_unlabel",
}
LABEL_NAMES = {
    0: "noise",
    1: "event",
    2: "unlabel",
}
ALLOWED_LABELS = [0, 1, 2]
SELECT_LABEL_NAMES = ["event"]
SELECT_SAMPLE_IDS = None
SELECT_ROW_INDICES = None
NORMALIZE = "none"
BATCH_SIZE = 4
MAX_BATCHES = None
ADD_CHANNEL_DIM = True

# raw와 filtered 비교
COMPARE_LEFT = "filtered"
COMPARE_RIGHT = "logscaled_after_scaled"

# filtered와 logscaled 비교
#COMPARE_LEFT = "filtered"
#COMPARE_RIGHT = "logscaled_after_scaled"

# raw와 logscaled 비교
#COMPARE_LEFT = "raw"
#COMPARE_RIGHT = "logscaled_after_scaled"

SAMPLING_RATE = 1000
LOG_BASE_LIST = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
SMOOTH_SIGMA = (1.0, 0.5)
EPS = 1e-8
SAVE_PLOTS = True
SHOW_PLOTS = False
FIGURE_SIZE = (14, 10)
FILTER_FIGURE_SIZE = (12, 4)
MAXIMIZE_PLOT_WINDOW = True
PLOT_VMIN = -2
PLOT_VMAX = 2
F_LOW = None
F_LOW_ORDER = 3
F_LOW_DECAY = 3
F_HIGH_LIST = [50, 100]
F_HIGH_ORDER = 1
F_HIGH_DECAY = 1
PAD_LENGTH = 500
TAPER_LENGTH = 300
ANALYSIS_ROOT = Path(f"analysis/{COMPARE_LEFT}_vs_{COMPARE_RIGHT}")

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def hilbert_1d(data):
    nt, _ = data.shape
    data_fft = np.fft.fftn(data, axes=(-2, -1))
    freqs_t = np.fft.fftfreq(nt)
    hilbert_filter_t = np.where(freqs_t > 0, 1j, np.where(freqs_t < 0, -1j, 0))[:, None]
    data_fft = hilbert_filter_t * data_fft
    return np.fft.ifftn(data_fft, axes=(-2, -1))


def envelope_1d(data):
    data_hilbert = hilbert_1d(data)
    data_env = np.real(np.sqrt(data_hilbert * data_hilbert + data * data))
    return data_env.astype(np.float32)


def hilbert_2d(data):
    nx, ny = data.shape
    data_fft = np.fft.fft2(data)
    freqs_x = np.fft.fftfreq(nx)
    freqs_y = np.fft.fftfreq(ny)

    epsilon = 1e-8
    kx_grid, ky_grid = np.meshgrid(freqs_x, freqs_y, indexing="ij")
    k_magnitude = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid + epsilon)
    is_dc = k_magnitude < 2 * epsilon

    h_riesz_x = np.zeros_like(kx_grid, dtype=np.complex64)
    h_riesz_y = np.zeros_like(ky_grid, dtype=np.complex64)
    h_riesz_x[~is_dc] = -1j * kx_grid[~is_dc] / k_magnitude[~is_dc]
    h_riesz_y[~is_dc] = -1j * ky_grid[~is_dc] / k_magnitude[~is_dc]

    rx_data = np.fft.ifft2(data_fft * h_riesz_x)
    ry_data = np.fft.ifft2(data_fft * h_riesz_y)
    return rx_data, ry_data


def envelope_2d(data):
    rt_data, rr_data = hilbert_2d(data)
    data_env = np.real(np.sqrt(rt_data * rt_data + rr_data * rr_data + data * data))
    return data_env.astype(np.float32)


def calculate_logscale(data, log_base=1, smooth_sigma=5.0, eps=1e-10, is_1d_envelope=True):
    data_env = envelope_1d(data) if is_1d_envelope else envelope_2d(data)
    data_env = data_env + log_base

    data_env_log = np.log10(data_env)
    data_env_log -= data_env_log.min()
    data_env_log += eps

    scale = data_env_log / data_env
    scale = np.log10(scale)
    scale = ndimage.gaussian_filter(scale, sigma=smooth_sigma)
    scale = np.power(10, scale)
    return scale


def calculate_norscale_inversion(data_log, log_base=1, smooth_sigma=5.0, eps=1e-10, iterations=50, is_1d_envelope=True):
    data_est = np.copy(data_log)
    est_scale = None

    for _ in range(iterations):
        est_scale = calculate_logscale(data_est, log_base, smooth_sigma, eps, is_1d_envelope)
        data_est = data_log / est_scale

    return est_scale, data_est


def build_csv(site, allowed_labels=None):
    rows = []
    site_root = DATA_ROOT / site
    for label, dirname in LABEL_DIRS.items():
        for path in sorted((site_root / dirname).glob("*.npy")):
            rows.append({
                "npy_path": str(path.resolve()),
                "label": label,
                "label_name": LABEL_NAMES[label],
                "sample_id": path.stem,
                "site": site,
            })

    if not rows:
        expected = "\n".join(f"  {site_root / dirname}" for dirname in LABEL_DIRS.values())
        raise FileNotFoundError(f"No .npy files were found for site '{site}'. Expected paths:\n{expected}")

    df = pd.DataFrame(rows)
    if allowed_labels is not None:
        df = df[df["label"].isin(allowed_labels)].reset_index(drop=True)

    csv_path = site_root / "pretrain.csv"
    df.to_csv(csv_path, index=False)
    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"[INFO] CSV created: {csv_path}")
    print(f"[INFO] label counts: {counts}")
    return df


def select_samples(df, label_names=None, sample_ids=None, row_indices=None):
    selected = df.copy()

    if label_names is not None:
        selected = selected[selected["label_name"].isin(label_names)]
    if sample_ids is not None:
        sample_ids = [str(sample_id).zfill(4) for sample_id in sample_ids]
        selected = selected[selected["sample_id"].isin(sample_ids)]
    if row_indices is not None:
        selected = selected.iloc[row_indices]

    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            "No samples matched the current selection. "
            "Check SELECT_LABEL_NAMES, SELECT_SAMPLE_IDS, and SELECT_ROW_INDICES."
        )
    return selected


def format_setting(value):
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    if isinstance(value, tuple):
        return "x".join(format_setting(v) for v in value)
    if value is None:
        return "None"
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "")


def selection_name(values, fallback="all"):
    if values is None:
        return fallback
    return "-".join(str(value).zfill(4) if isinstance(value, int) else str(value) for value in values)


def safe_name(value):
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def validate_compare_targets(signal_map):
    missing = [name for name in [COMPARE_LEFT, COMPARE_RIGHT] if name not in signal_map]
    if missing:
        available = ", ".join(signal_map.keys())
        raise ValueError(f"Unknown compare target(s): {missing}. Available targets: {available}")


def normalize_sites(sites):
    if isinstance(sites, str):
        sites = [sites]

    normalized = []
    for site in sites:
        site = SITE_ALIASES.get(site, site)
        if site not in normalized:
            normalized.append(site)
    return sorted(normalized)


def build_run_dir(site, log_base, f_high):
    label_part = selection_name(SELECT_LABEL_NAMES, fallback="all_labels")
    sample_part = selection_name(SELECT_SAMPLE_IDS, fallback="all_samples")
    row_part = selection_name(SELECT_ROW_INDICES, fallback="all_rows")
    run_name = (
        f"{label_part}"
        f"__compare_{COMPARE_LEFT}_vs_{COMPARE_RIGHT}"
        f"__log{format_setting(log_base)}"
        f"_sm{format_setting(SMOOTH_SIGMA)}"
        f"_eps{format_setting(EPS)}"
        f"_f{format_setting(F_LOW)}-{format_setting(f_high)}"
        f"_clip{format_setting(PLOT_VMIN)}-{format_setting(PLOT_VMAX)}"
        f"_samples{sample_part}"
        f"_rows{row_part}"
    )

    return ANALYSIS_ROOT / run_name / site


def save_settings(run_dir, site, log_base, f_high, selected_count):
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "site": site,
        "sites": normalize_sites(SITES),
        "allowed_labels": ALLOWED_LABELS,
        "select_label_names": SELECT_LABEL_NAMES,
        "select_sample_ids": SELECT_SAMPLE_IDS,
        "select_row_indices": SELECT_ROW_INDICES,
        "selected_count": selected_count,
        "normalize": NORMALIZE,
        "compare_left": COMPARE_LEFT,
        "compare_right": COMPARE_RIGHT,
        "batch_size": BATCH_SIZE,
        "max_batches": MAX_BATCHES,
        "sampling_rate": SAMPLING_RATE,
        "log_base": log_base,
        "log_base_list": LOG_BASE_LIST,
        "smooth_sigma": SMOOTH_SIGMA,
        "eps": EPS,
        "plot_vmin": PLOT_VMIN,
        "plot_vmax": PLOT_VMAX,
        "f_low": F_LOW,
        "f_low_order": F_LOW_ORDER,
        "f_low_decay": F_LOW_DECAY,
        "f_high": f_high,
        "f_high_list": F_HIGH_LIST,
        "f_high_order": F_HIGH_ORDER,
        "f_high_decay": F_HIGH_DECAY,
        "pad_length": PAD_LENGTH,
        "taper_length": TAPER_LENGTH,
        "figure_size": FIGURE_SIZE,
        "filter_figure_size": FILTER_FIGURE_SIZE,
    }
    path = run_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"[INFO] settings saved: {path}")


def normalize_array(x, mode="none", eps=1e-8):
    if mode in [None, "none"]:
        return x
    if mode in ["zscore", "standard"]:
        return (x - x.mean()) / (x.std() + eps)
    if mode in ["rms", "rms_normalize"]:
        return x / (np.sqrt(np.mean(x * x)) + eps)
    if mode == "minmax":
        return 2.0 * (x - x.min()) / (x.max() - x.min() + eps) - 1.0
    raise ValueError(f"Unsupported normalize mode: {mode}")


def load_sample(path, normalize="none", add_channel_dim=True):
    x = np.load(path).astype(np.float32, copy=False)
    x = normalize_array(x, normalize).astype(np.float32, copy=False)
    if add_channel_dim:
        x = x[None, ...]
    return x


def iter_batches(df, batch_size, normalize="none", add_channel_dim=True):
    paths = df["npy_path"].tolist()
    for start in range(0, len(paths), batch_size):
        batch_df = df.iloc[start:start + batch_size].reset_index(drop=True)
        batch_paths = batch_df["npy_path"].tolist()
        samples = [load_sample(path, normalize, add_channel_dim) for path in batch_paths]
        yield np.stack(samples, axis=0), batch_df


def inspect_batch(batch):
    print("\n" + "=" * 50)
    print(f"shape   : {tuple(batch.shape)}")
    print(f"dtype   : {batch.dtype}")
    print(f"min/max : {batch.min():.4f} / {batch.max():.4f}")
    print(f"mean    : {batch.mean():.4f}  std: {batch.std():.4f}")


def mirror_padding(signal, pad_length=1000, taper_length=500, axis=-1):
    def _slice(arr, start, stop, axis):
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(start, stop)
        return arr[tuple(idx)]

    zero_length = pad_length - taper_length
    mirror = np.flip(signal, axis=axis)

    taper = np.hanning(2 * taper_length)
    shape = [1] * signal.ndim
    shape[axis] = 2 * taper_length
    taper = taper.reshape(shape)

    taper_top = _slice(taper, None, taper_length, axis)
    taper_bot = _slice(taper, taper_length, None, axis)

    mirror_tapered_top = _slice(mirror, -taper_length, None, axis) * taper_top
    mirror_tapered_bot = _slice(mirror, None, taper_length, axis) * taper_bot

    shape_zero = list(signal.shape)
    shape_zero[axis] = zero_length
    zeros = np.zeros(shape_zero, dtype=signal.dtype)

    return np.concatenate([zeros, mirror_tapered_top, signal, mirror_tapered_bot, zeros], axis=axis)


def f_filter(nt, dt, f_cut, order, decay, is_lowpass=True, max_clip=400.0):
    f = np.abs(np.fft.fftfreq(nt, dt))
    x = f - f_cut if is_lowpass else f_cut - f
    x = np.clip(x, -max_clip, max_clip)
    m = -(decay / order) * np.log2(1 + 2 ** (order * x))
    return 2 ** m


def bandpass_filter(nt, dt, f_low, order_low, decay_low, f_high, order_high, decay_high, max_clip=400.0):
    if f_low is None:
        m_low = np.ones(nt)
    else:
        m_low = f_filter(nt, dt, f_low, order_low, decay_low, is_lowpass=False, max_clip=max_clip)

    if f_high is None:
        m_high = np.ones(nt)
    else:
        m_high = f_filter(nt, dt, f_high, order_high, decay_high, is_lowpass=True, max_clip=max_clip)

    return m_low * m_high


def f_filtering(data, mask, is_zeroout=False):
    shape = data.shape
    nt = shape[-1]
    m_f = mask.copy()
    if is_zeroout:
        m_f[0] = 0.0

    data_unfold = data.copy()[None, :] if len(shape) == 1 else data.reshape((-1, nt))
    data_unfold = np.fft.fft(data_unfold, axis=-1)
    data_unfold *= m_f[None]
    data_unfold = np.real(np.fft.ifft(data_unfold, axis=-1))

    return data_unfold.reshape(shape) if len(shape) > 1 else data_unfold.squeeze()


def render_filter_plot(freq_mask, out_dir=None):
    fig, ax = plt.subplots(figsize=FILTER_FIGURE_SIZE)
    ax.plot(freq_mask)
    ax.set_title("Frequency mask")
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("Gain")
    fig.tight_layout()

    if SAVE_PLOTS:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "frequency_mask.png"
        fig.savefig(path, dpi=150)
        print(f"[INFO] saved: {path}")
    if SHOW_PLOTS:
        show_plot(fig)

    plt.close(fig)


def render_contrast_pair(
    x1,
    x2,
    out_path=None,
    title="",
    first_label="filtered",
    second_label="logscaled",
    vmin=None,
    vmax=None,
):
    if vmin is None:
        vmin = PLOT_VMIN
    if vmax is None:
        vmax = PLOT_VMAX

    a = x1.transpose().copy()
    b = x2.transpose().copy()
    a *= 0.25 / (np.sqrt(np.mean(a * a)) + 1e-8)
    b *= 0.25 / (np.sqrt(np.mean(b * b)) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    for ax, arr, label in zip(axes, [a, b, a - b], [first_label, second_label, "difference"]):
        im = ax.imshow(arr, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Channels")
        ax.set_ylabel("Time samples")

    axes[0].text(
        0.02,
        0.98,
        title,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 4},
    )

    fig.tight_layout()
    if SAVE_PLOTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[INFO] saved: {out_path}")
    if SHOW_PLOTS:
        show_plot(fig)

    plt.close(fig)


def show_plot(fig):
    manager = plt.get_current_fig_manager()
    if MAXIMIZE_PLOT_WINDOW:
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.resize(*manager.window.maxsize())
            except Exception:
                pass
    plt.show()


def process_site(site, log_base, f_high):
    df = build_csv(site, allowed_labels=ALLOWED_LABELS)
    df = select_samples(
        df,
        label_names=SELECT_LABEL_NAMES,
        sample_ids=SELECT_SAMPLE_IDS,
        row_indices=SELECT_ROW_INDICES,
    )
    n_batches = int(np.ceil(len(df) / BATCH_SIZE))

    print(f"\n[INFO] data root    : {DATA_ROOT}")
    print(f"[INFO] site         : {site}")
    print(f"[INFO] dataset size : {len(df)}")
    print(f"[INFO] batch_size   : {BATCH_SIZE}")
    print(f"[INFO] normalize    : {NORMALIZE}")
    print(f"[INFO] f_high       : {f_high}")
    print(f"[INFO] log_base     : {log_base}")

    nt = 2000
    dt = 1 / SAMPLING_RATE
    freq_mask = bandpass_filter(
        nt + 2 * PAD_LENGTH,
        dt,
        f_low=F_LOW,
        order_low=F_LOW_ORDER,
        decay_low=F_LOW_DECAY,
        f_high=f_high,
        order_high=F_HIGH_ORDER,
        decay_high=F_HIGH_DECAY,
    )

    run_dir = build_run_dir(site, log_base, f_high)
    save_settings(run_dir, site, log_base, f_high, selected_count=len(df))
    render_filter_plot(freq_mask, run_dir)

    for i, (batch, batch_df) in enumerate(iter_batches(df, BATCH_SIZE, NORMALIZE, ADD_CHANNEL_DIM)):
        if MAX_BATCHES is not None and i >= MAX_BATCHES:
            break

        print(f"\nBatch {i + 1}/{n_batches}")
        print(f"[INFO] first file   : {batch_df.iloc[0]['npy_path']}")
        inspect_batch(batch)

        for j, row in batch_df.iterrows():
            x = batch[j].squeeze()
            x_pad = mirror_padding(x, pad_length=PAD_LENGTH, taper_length=TAPER_LENGTH)
            filtered = f_filtering(x_pad, freq_mask, is_zeroout=True)
            filtered = filtered[..., PAD_LENGTH:-PAD_LENGTH]

            scale = calculate_logscale(filtered, log_base=log_base, smooth_sigma=SMOOTH_SIGMA, eps=EPS)
            logscaled_after_scaled = filtered * scale

            signal_map = {
                "raw": x,
                "filtered": filtered,
                "logscaled_after_scaled": logscaled_after_scaled,
            }
            validate_compare_targets(signal_map)

            left = signal_map[COMPARE_LEFT]
            right = signal_map[COMPARE_RIGHT]
            file_suffix = f"{safe_name(COMPARE_LEFT)}_vs_{safe_name(COMPARE_RIGHT)}"

            label_name = row["label_name"]
            sample_id = row["sample_id"]
            title = f"{site} / {label_name} / {sample_id}.npy"
            out_path = run_dir / label_name / f"{sample_id}_{file_suffix}.png"
            render_contrast_pair(
                left,
                right,
                out_path,
                title=title,
                first_label=COMPARE_LEFT,
                second_label=COMPARE_RIGHT,
                vmin=PLOT_VMIN,
                vmax=PLOT_VMAX,
            )


def main():
    sites = normalize_sites(SITES)
    print(f"[INFO] sites        : {sites}")
    print(f"[INFO] f_high_list  : {F_HIGH_LIST}")
    print(f"[INFO] log_base_list: {LOG_BASE_LIST}")
    for f_high in F_HIGH_LIST:
        for log_base in LOG_BASE_LIST:
            for site in sites:
                process_site(site, log_base, f_high)


if __name__ == "__main__":
    main()
