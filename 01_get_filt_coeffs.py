# %%
# packages and paths
import subprocess, os
from json import dump, load
from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat
from scipy.signal import firwin
from spectrum_analysis import compute_peak_bands, parameterize_spectrum

# change pwd to script directory to ensure relative paths work correctly
pwd = Path(__file__).parent
os.chdir(pwd)


with open("settings.json", "r") as f:
    settings = load(f)

sub_id = settings["sub_id"]
sess_id = settings["sess_id"]
data_root = Path(settings["data_root"])
sess_root = data_root / f"LAVA_{sub_id[-3:]}"
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

pre_rs.crop(tmin=30, tmax=630)  # take only the first 10 minutes of data
center = "C3"
neighbors = ["FC3", "CP3", "C1", "C5"]

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

spectrum = raw_c3_hjorth.compute_psd(fmin=1, fmax=60)

freqs = spectrum.freqs
psd = spectrum.get_data()

periodic_params, aperiodic_params = parameterize_spectrum(
    [freqs, psd[0, :]], save_fig=psd_path, fmax=60
)

# compute peak bands
# peak_index = np.argmax(
#     [
#         periodic_params[index][1]
#         for index, param in enumerate(periodic_params)
#         if param[0] < 12 and param[0] > 8
#     ]
# )

# peak = periodic_params[peak_index][0]
# print(f"peak: {peak}")
# lower, upper = peak - 1, peak + 1
upper, lower, _ = compute_peak_bands(periodic_params, [[8, 12, 0]])

upper, lower = upper[0], lower[0]

print(f"upper, lower: {upper}, {lower}")
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
