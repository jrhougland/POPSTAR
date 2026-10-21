# %%
# packages and paths
import os
import subprocess
from json import dump, load
from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat
from scipy.signal import firwin

from spectrum_analysis import parameterize_spectrum

# change pwd to script directory to ensure relative paths work correctly
pwd = Path(__file__).parent
os.chdir(pwd)


with open("settings.json", "r") as f:
    settings = load(f)

sub_id = settings["sub_id"]
sess_id = settings["sess_id"]
data_root = Path(settings["data_root"])
sess_root = data_root / f"POPSTAR_{sub_id[-3:]}"
latest_session = max(
    (path for path in sess_root.iterdir() if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
    default=None,
)
data_load = str(latest_session)

script_dir = Path(__file__).resolve().parent
repo_root = script_dir
data_save = Path(settings["data_save"])
subject_dir = data_save / sub_id
rs_file = subject_dir / f"{sub_id}.set"
coeffs_path = data_save / f"bpfilter_{sub_id}.mat"
psd_path = repo_root / "figures" / f"{sub_id}_psd.png"
dec = 10
print(f"root: {repo_root}")
subject_dir.mkdir(parents=True, exist_ok=True)
psd_path.parent.mkdir(exist_ok=True)

# save new paths into settings.json for use in MATLAB
settings["coeffs_path"] = str(coeffs_path)
with open("settings.json", "w") as f:
    dump(settings, f, indent=4)

# %%
matlab_cmd = (
    f"root='{repo_root}'; "
    f"sub_id='{sub_id}'; "
    f"data_load='{data_load}'; "
    f"data_save='{data_save.as_posix()}'; "
    f"sess_id={sess_id}; "
    "run('convert_to_eeglab.m');"
)

result = subprocess.run(
    ["matlab", "-batch", matlab_cmd],
    capture_output=True,
    text=True,
    cwd=str(script_dir),
)

print(result.stdout)
print(result.stderr)

if result.returncode != 0:
    raise RuntimeError(
        f"MATLAB conversion failed with exit code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

if not rs_file.exists():
    raise FileNotFoundError(
        f"Expected EEGLAB file was not created: {rs_file}.\n"
        "Check MATLAB conversion logs above and confirm input data path/session are correct."
    )

# %%
# c3 hjorth filter
pre_rs = mne.io.read_raw_eeglab(str(rs_file), preload=True)

pre_rs.crop(tmin=5, tmax=305)  # take only the first 5 minutes of data
center = "C3"
neighbors = ["FC1", "CP1", "FC5", "CP5"]

# get the data
c3 = pre_rs.get_data(picks=center)
surround = pre_rs.get_data(picks=neighbors)

c3_hjorth = c3 - surround.mean(axis=0, keepdims=True)

# create a new Raw object for the virtual channel
info = mne.create_info(
    ch_names=["C3_Hjorth"], sfreq=pre_rs.info["sfreq"], ch_types=["eeg"]
)
raw_c3_hjorth = mne.io.RawArray(c3_hjorth, info).filter(1, 60, fir_design="firwin")

raw_c3_hjorth.resample(pre_rs.info["sfreq"] / dec)

n_per_seg = int(4 * raw_c3_hjorth.info["sfreq"])  # 4 s windows
n_fft = 4096
spectrum = raw_c3_hjorth.compute_psd(
    fmin=1, fmax=60, n_fft=n_fft, n_overlap=n_per_seg // 2, n_per_seg=n_per_seg
)

freqs = spectrum.freqs
psd = spectrum.get_data()

periodic_params, aperiodic_params = parameterize_spectrum(
    [freqs, psd[0, :]],
    save_fig=repo_root / "figures" / f"{sub_id}_psd.png",
    freq_range=[1, 60],
)

# get snr peak
lower_alpha, upper_alpha = 8, 13
peak_snr = max(psd[0, (freqs >= lower_alpha) & (freqs <= upper_alpha)])
peak_idx = np.where(psd[0, :] == peak_snr)[0][0]
peak_power_log = np.log10(peak_snr)
aperiodic_background_log = aperiodic_params[0] - aperiodic_params[1] * np.log10(
    freqs[peak_idx]
)
peak_snr = 10 * (peak_power_log - aperiodic_background_log)
print("\n" + 100 * "-")
print(f"Peak SNR: {peak_snr:.2f} dB at {freqs[peak_idx]:.2f} Hz")

# compute peak bands with +/-1 fixed bandwidth
band_peaks = [
    param
    for param in periodic_params
    if param[0] < upper_alpha and param[0] > lower_alpha
]
if len(band_peaks) > 0:
    peak = max(band_peaks, key=lambda x: x[1])  # get the peak with the highest power
    peak_f = peak[0]
else:
    print("No peaks found in the alpha band. Using the peak SNR frequency instead.")
    peak_f = freqs[peak_idx]
print(f"Peak frequency: {peak_f:.2f} Hz")
lower, upper = peak_f - 1, peak_f + 1

# save snr
with open(data_save / "sub_snr.csv", "a") as f:
    f.write(
        f"{sub_id}, Session: {sess_id}, Peak frequency: {peak_f} Hz, SNR: {peak_snr:.2f} dB\n"
    )

# use for fitted width instead of fixed 1 Hz bandwidth
# upper, lower = [x[0] for x in compute_peak_bands(periodic_params, [[8, 12, 0]])[:2]]


# design FIR filter and retrieve coefficients
fs = raw_c3_hjorth.info["sfreq"]
numtaps = (
    80 + 1
)  # Filter order (according to Zrenner et al. 2020) + 1 (must be odd for bandpass)

print(
    f"Designing FIR filter with fs={fs}, numtaps={numtaps}, lower={lower}, upper={upper}"
)

coefficients = firwin(numtaps, [lower, upper], fs=fs, pass_zero="bandpass")

# save coefficients to .mat file
savemat(str(coeffs_path), {"coefficients": coefficients})

# run git push to sync with remote repository
subprocess.run(["git", "add", str(coeffs_path), str(psd_path)], cwd=str(repo_root))
subprocess.run(
    ["git", "commit", "-m", f"Add FIR filter coefficients for {sub_id}"],
    cwd=str(repo_root),
)
subprocess.run(["git", "push"], cwd=str(repo_root))
