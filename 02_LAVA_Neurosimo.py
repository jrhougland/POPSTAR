"""
Phastimate decider module for NeuroSimo.

Reference:
Zrenner, C., Galevska, D., Nieminen, J.O., Baur, D., Stefanou, M.I., Ziemann, U. (2020). 
The shaky ground truth of real-time phase estimation. NeuroImage, 214, 116761.
https://doi.org/10.1016/j.neuroimage.2020.116761

Available at:
https://www.sciencedirect.com/science/article/pii/S1053811920302262

MATLAB implementation:
https://github.com/bnplab/phastimate

This version is based on MATLAB adaptation of Phastimate by Joonas Laurinoja at Aalto University.
"""

import csv
import os, subprocess
import time
import socket
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import filtfilt, hilbert
from scipy.io import loadmat
from spectrum import aryule


# EEG channel indices for C3 referencing matches BOSSDevice
C3_CHANNEL_INDEX = 4 # add 1 to convert to MATLAB (1-based)
REFERENCE_CHANNEL_INDICES = [20, 22, 24, 26]  # Reference channels for C3, again 0-based
REFERENCE_WEIGHT = 0.25 

# Phase estimation constants
REPO_PATH = os.path.dirname(os.path.abspath(__file__))  # This directory is the POPSTAR Git repository (for filter coefficient updates)
DATA_PATH = f'{REPO_PATH}/data'

# Phase schedule: trough (pi) and peak (0) only, shuffled within balanced blocks
N_TRIALS = 148 # MUST BE DIVISIBLE BY len(PHASE_CONDITIONS) * BLOCK_MULTIPLIER
PHASE_CONDITIONS = [0, np.pi]
BLOCK_MULTIPLIER = 2  # each block has 4 trials (2 peak, 2 trough)

DEFAULT_PHASE_TOLERANCE = np.pi / 40  # Single tolerance value for all phases (pi/40)

# Phastimate algorithm parameters (Zrenner et al. 2020)
# Note: BOSSdevice firmware parameters are compiled in Simulink model (not accessible via API)
# BOSSdevice examples show offset_samples=3-6 (depends on loop delay)
# Zrenner2020: Uses 500ms windows, 128-sample Hilbert window at 500Hz (after 10x decimation from 5kHz)
DEFAULT_HILBERT_WINDOW_SIZE = 128  
DEFAULT_EDGE_SAMPLES = 65
# Note: BOSS offset_samples=3-6 at 500Hz, scaled to your sampling rate this equals ~30-60 at 5000Hz pre-decimation
DEFAULT_AR_MODEL_ORDER = 25  # AR model order - good balance between under/overfitting
DEFAULT_DOWNSAMPLE_RATIO = 10  # Matches BOSSDevice: 5000Hz -> 500Hz

# Processing timing constants
DEFAULT_PROCESSING_INTERVAL_SECONDS = 0.05

DEFAULT_BUFFER_SIZE_SECONDS = 0.719  # ??

# NOTE: There used to be TRIGGER_COOLDOWN_SECONDS here, which was set to 2.0 s. Nowadays, the equivalent variable is
#   set in the protocol (LAVA.yaml), and it is called minimum_trial_interval. It is currently set to 2.0 s, but can
#   be adjusted as needed.

MINIMUM_TRIGGER_DELAY_SECONDS = 0.005  # Minimum 5ms delay required for system to process and send trigger
FIRST_ROUND_WAIT_SECONDS = 0.5  # Wait time before first UDP trigger to ensure system is ready

class Decider:
    """
    Real-time EEG phase estimation and trigger scheduling using Phastimate algorithm.
    
    This class implements the Phastimate algorithm for real-time phase estimation
    of EEG signals and schedules triggers based on target phase detection.
    """
    
    def __init__(self, subject_id: int, num_eeg_channels: int, num_emg_channels: int, sampling_frequency: int):
        """
        Initialize the Decider with parameters and filter design.
        
        Args:
            subject_id: ID of the subject
            num_eeg_channels: Number of EEG channels (unused but kept for interface compatibility)
            num_emg_channels: Number of EMG channels (unused but kept for interface compatibility)
            sampling_frequency: Sampling frequency in Hz
        """
        self.subject_number = subject_id
        self.subject_id = f"sub-{subject_id:03d}"

        results = subprocess.run('git pull', cwd=REPO_PATH, shell=True, capture_output=True, text=True)
        print(results.stdout)
        print(results.stderr) if results.returncode != 0 else print("Git pull successful")

        os.makedirs(DATA_PATH, exist_ok=True)

        mat_data = loadmat(f'{DATA_PATH}/bpfilter_{self.subject_id}.mat')
        self.bandpass_filter_coefficients = np.array(mat_data['coefficients'].flatten()) # correct, matches BOSSDevice
        self.sampling_frequency = sampling_frequency

        # Phastimate algorithm parameters
        self.hilbert_window_size = DEFAULT_HILBERT_WINDOW_SIZE
        self.edge_samples = DEFAULT_EDGE_SAMPLES
        self.ar_model_order = DEFAULT_AR_MODEL_ORDER
        self.downsample_ratio = DEFAULT_DOWNSAMPLE_RATIO

        # Processing timing parameters
        self.processing_interval_seconds = DEFAULT_PROCESSING_INTERVAL_SECONDS
        self.buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS

        

        self.phase_tolerance = DEFAULT_PHASE_TOLERANCE
        
        # Maximum number of future samples to consider for trigger scheduling
        self.max_future_samples = int(self.edge_samples / 2)

        # Leave empty when mTMS device is not used
        self.targets = []
        

        # Initialize logs - single structure for all phases
        self.timestamp_log = []
        self.triggertimes_log = []
        self.phases_log = []  # Track which phase was targeted

        # Number of warm-up rounds to prevent first-call delays (see README.md for details)
        self.warm_up_rounds = 2

        self.first_call = True  # Track first call for UDP trigger

        # Warm-up time tracking (initialized when get_configuration is called)
        self.warm_up_start_time = None
        self.warm_up_duration = FIRST_ROUND_WAIT_SECONDS

        # Diagnostics for debugging trigger skips
        self.last_phase_error = None
        
        # Create the phase schedule and save it as CSV
        self.phases = self._create_phases(f'{DATA_PATH}/{self.subject_id}_phase_schedule.csv')

        self.target_phase_radians = self.phases[0]
        
        # Total trials
        self.total_trials = len(self.phases)
        self.trials_left = self.total_trials

    def get_configuration(self) -> dict[str, Any]:
        """
        Return the configuration for the processing interval and sample window.
        
        Returns:
            Dictionary containing processing configuration parameters
        """
        # Import time module for warm-up timing
        
        # Start the warm-up timer when configuration is first requested
        if self.warm_up_start_time is None:
            self.warm_up_start_time = time.time()
            print(f"Warm-up period started: {self.warm_up_duration} seconds")
                
        return {
            # Data configuration
            'sample_window': [-self.buffer_size_seconds, 0],
            'warm_up_rounds': 0,  # Number of warm-up rounds to perform (0 to disable)

            # Periodic processing
            'periodic_processing_interval': self.processing_interval_seconds,
        }


    def _create_phases(self, csv_path: str) -> np.ndarray:
        """
        Create the peak/trough phase schedule and save it to a CSV file.

        Trials are shuffled within blocks so both conditions stay balanced over the session.
        The RNG is seeded with the subject number, so the order is reproducible per subject.

        Args:
            csv_path: Path of the CSV file to write the phase targets (in radians) to

        Returns:
            Array of phase targets
        """
        rng = np.random.default_rng(self.subject_number)
        block = np.array(PHASE_CONDITIONS * BLOCK_MULTIPLIER, dtype=float)
        num_blocks = N_TRIALS // len(block)

        phases_array = np.concatenate([rng.permutation(block) for _ in range(num_blocks)])
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for phase in phases_array:
                writer.writerow([phase])

        print(f"Created {len(phases_array)} phase targets (peak/trough) and saved them to {csv_path}")

        return phases_array

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, pulse_count: int, is_warm_up: bool) -> dict[str, Any] | None:
        """
        Process the EEG data to estimate phase and schedule a trigger.
        """
        print("decider process_periodic called with pulse_count:", pulse_count)
        # Check if warm-up period has elapsed
        if self.warm_up_start_time is not None:
            elapsed_time = time.time() - self.warm_up_start_time
            if elapsed_time < self.warm_up_duration:
                # Still in warm-up period, skip processing
                return None
            elif elapsed_time < self.warm_up_duration:
                self.warm_up_start_time = None  # Clear warm-up timer
                print("Warm-up period completed")

        if self.first_call:
        
            # send UDP trigger
            udp_host = os.environ.get('UDP_TRIGGER_HOST', '192.168.2.136') # adjust if necessary, e.g. if control PC got assigned a new IPv4 adress.
            udp_port = int(os.environ.get('UDP_TRIGGER_PORT', '5555'))
            udp_msg = os.environ.get('UDP_TRIGGER_MESSAGE', 'phastimate_trigger')
            try:
                # Send UDP datagram
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1.0)
                sock.sendto(udp_msg.encode('utf-8'), (udp_host, udp_port))
                sock.close()
                print(f"Sent UDP trigger to {udp_host}:{udp_port} msg='{udp_msg}'")
            except Exception as e:
                print(f"Warning: UDP trigger failed: {e}")
            self.first_call = False
                   
        # Extract C3 channel with common average reference
        c3_referenced_data = self._extract_c3_referenced_data(eeg_buffer)
        if c3_referenced_data is None:
            return None

        # Update target phase from the schedule
        self.target_phase_radians = self.phases[pulse_count]

        # Preprocess the data
        preprocessed_data = self._preprocess_eeg_data(c3_referenced_data)

        # Estimate future phases using Phastimate algorithm
        estimated_phases = self._estimate_phases(preprocessed_data)
        if estimated_phases is None:
            print(f"DEBUG: Phase estimation failed (trial {pulse_count + 1})")
            return None

        # Find optimal trigger timing
        trigger_timing = self._find_optimal_trigger_timing(estimated_phases, reference_time)

        if trigger_timing is None:
            return None
                
        # log timestamp, trigger time, and phase
        self.timestamp_log.append(reference_time)

        absolute_trigger_time = reference_time + trigger_timing['trigger_offset']
        self.triggertimes_log.append(absolute_trigger_time)  # Save just the time value, not the dict
        self.phases_log.append(self.target_phase_radians)
        
        
        print(
            f"Delivered Trigger for Trial {pulse_count + 1}/{self.total_trials} "
            f"(Phase target: {self.target_phase_radians:.2f} rad)", flush=True
        )

        return trigger_timing
    
    def __del__(self):
        print("\n=== Session finished — saving logs ===")
        self._save_logs(self.subject_id)

    def _extract_c3_referenced_data(self, eeg_buffer: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract C3 channel data with common average reference.
        
        Args:
            eeg_buffer: EEG data buffer (samples x channels)
            
        Returns:
            Referenced C3 data or None if extraction fails
        """
        try:
            c3_data = eeg_buffer[:, C3_CHANNEL_INDEX]
            reference_data = np.sum(eeg_buffer[:, REFERENCE_CHANNEL_INDICES], axis=1)
            return c3_data - REFERENCE_WEIGHT * reference_data
        except IndexError:
            print("Error: EEG buffer does not have expected number of channels")
            return None

    def _preprocess_eeg_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data by demeaning and downsampling.
        
        Args:
            data: Input EEG data
            
        Returns:
            Preprocessed and downsampled data
        """
        # Remove DC component
        demeaned_data = data - np.mean(data)
        
        # Downsample the data
        return demeaned_data[::self.downsample_ratio]

    def _estimate_phases(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate future phases using the Phastimate algorithm.
        
        Args:
            data: Preprocessed EEG data
            
        Returns:
            Array of estimated phases or None if estimation fails
        """
        estimated_phases, _ = self.phastimate(
            data,
            self.bandpass_filter_coefficients, 
            [1.0], 
            self.edge_samples, 
            self.ar_model_order, 
            self.hilbert_window_size
        )

        return estimated_phases

    def _find_optimal_trigger_timing(self, estimated_phases: np.ndarray, reference_time: float) -> Optional[Dict[str, Any]]:
        """
        Find optimal trigger timing based on estimated phases.
        
        Args:
            estimated_phases: Array of estimated phase values
            reference_time: Current timestamp
            
        Returns:
            Dictionary with trigger timing or None if no suitable timing found
        """
        # Reset diagnostics for this search
        self.last_phase_error = None

        # Extract future phase estimates (second half of the estimation window)
        num_samples = estimated_phases.shape[0]
        future_phase_estimates = estimated_phases[num_samples // 2:]
        
        # Limit to maximum future samples
        future_phase_estimates = future_phase_estimates[:self.max_future_samples]

        # Calculate phase differences from target
        phase_differences = np.angle(np.exp(1j * (future_phase_estimates - self.target_phase_radians)))

        print(f"DEBUG: Future phase differences (radians): {phase_differences}")
        # Find the sample with minimum phase difference
        optimal_sample_index = np.argmin(np.abs(phase_differences))

        min_phase_difference = phase_differences[optimal_sample_index]
        phase_error = np.abs(min_phase_difference)

        print(f"DEBUG: Optimal sample index: {optimal_sample_index}, Phase error: {phase_error:.4f} radians")
        self.last_phase_error = phase_error

        # Check if phase difference is within tolerance
        if phase_error > self.phase_tolerance:
            return None

        # Calculate trigger execution time
        time_offset_seconds = (optimal_sample_index * self.downsample_ratio) / self.sampling_frequency
        
        # Check if trigger delay is sufficient for system processing
        if time_offset_seconds < MINIMUM_TRIGGER_DELAY_SECONDS:
            return None
        
        print(f'Trigger scheduled {time_offset_seconds:.3f} seconds from now', flush=True)

        return {
            'trigger_offset': time_offset_seconds,
        }

    def phastimate(self, data: np.ndarray, filter_b: np.ndarray, filter_a: List[float], 
                   edge_samples: int, ar_order: int, hilbert_window_size: int,
                   offset_correction: int = 0, iterations: Optional[int] = None, 
                   ar_method: str = 'aryule') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate the phase of the EEG signal using autoregressive modeling and Hilbert transform.
        
        This is the core Phastimate algorithm that performs:
        1. Bandpass filtering of the input signal
        2. Autoregressive (AR) modeling for forward prediction
        3. Hilbert transform for phase extraction
        
        Args:
            data: Input EEG signal
            filter_b: Numerator coefficients of the bandpass filter
            filter_a: Denominator coefficients of the bandpass filter  
            edge_samples: Number of edge samples to remove after filtering
            ar_order: Order of the autoregressive model
            hilbert_window_size: Size of the window for Hilbert transform
            offset_correction: Offset correction (unused)
            iterations: Number of forward prediction iterations
            ar_method: Method for AR parameter estimation ('aryule')
            
        Returns:
            Tuple of (estimated_phases, estimated_amplitudes) or (None, None) if estimation fails
        """
        # Calculate default number of iterations if not specified
        if iterations is None:
            iterations = edge_samples + int(np.ceil(hilbert_window_size / 2))

        # Validate data length for filtering
        min_padding_length = 3 * (max(len(filter_a), len(filter_b)) - 1)
        if data.shape[0] <= min_padding_length:
            print(f"Insufficient data for filtering: {data.shape[0]} <= {min_padding_length}")
            return None, None

        # Apply bandpass filter
        filtered_data = filtfilt(filter_b, filter_a, data)

        # Remove edge samples to mitigate filter transients
        if filtered_data.shape[0] <= 2 * edge_samples:
            print(f"Insufficient data after edge removal: {filtered_data.shape[0]} <= {2 * edge_samples}")
            return None, None

        edge_removed_data = filtered_data[edge_samples:-edge_samples]

        # Fit autoregressive model
        ar_coefficients = self._fit_ar_model(edge_removed_data, ar_order, ar_method)
        if ar_coefficients is None:
            return None, None

        # Perform forward prediction
        predicted_data = self._forward_predict(edge_removed_data, ar_coefficients, iterations)

        # Extract phase and amplitude using Hilbert transform
        return self._extract_phase_amplitude(predicted_data, hilbert_window_size)

    def _fit_ar_model(self, data: np.ndarray, ar_order: int, ar_method: str) -> Optional[np.ndarray]:
        """
        Fit autoregressive model to the data.
        
        Args:
            data: Input data for AR modeling
            ar_order: Order of the AR model
            ar_method: Method for AR parameter estimation
            
        Returns:
            AR coefficients or None if fitting fails
        """
        if len(data) < ar_order:
            print(f"Insufficient data for AR model: {len(data)} < {ar_order}")
            return None

        if ar_method == 'aryule':
            try:
                ar_params, _, _ = aryule(data, ar_order)
                # Flip and negate coefficients for prediction equation
                return -1 * ar_params[::-1]
            except Exception as e:
                print(f"AR model fitting failed: {e}")
                return None
        else:
            raise ValueError(f'Unknown AR method: {ar_method}')

    def _forward_predict(self, data: np.ndarray, ar_coefficients: np.ndarray, iterations: int) -> np.ndarray:
        """
        Perform forward prediction using AR model.
        
        Args:
            data: Historical data
            ar_coefficients: AR model coefficients
            iterations: Number of prediction steps
            
        Returns:
            Extended data with predictions
        """
        # Initialize prediction array
        total_length = len(data) + iterations
        predicted_data = np.zeros(total_length)
        predicted_data[:len(data)] = data

        # Perform iterative prediction
        ar_order = len(ar_coefficients)
        for i in range(iterations):
            prediction_index = len(data) + i
            data_window = predicted_data[prediction_index - ar_order:prediction_index]
            predicted_data[prediction_index] = np.sum(ar_coefficients * data_window)

        return predicted_data

    def _extract_phase_amplitude(self, data: np.ndarray, window_size: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract phase and amplitude using Hilbert transform.
        
        Args:
            data: Input signal data
            window_size: Size of the analysis window
            
        Returns:
            Tuple of (phases, amplitudes) or (None, None) if extraction fails
        """
        if data.shape[0] < window_size:
            print(f'Insufficient data for Hilbert transform: {data.shape[0]} < {window_size}')
            return None, None

        # Extract the analysis window (last window_size samples)
        analysis_window = data[-window_size:]

        # Compute analytic signal using Hilbert transform
        analytic_signal = hilbert(analysis_window)
        phases = np.angle(analytic_signal)
        amplitudes = np.abs(analytic_signal)

        return phases, amplitudes

    def _save_logs(self, filename_prefix: str, path: str = './data') -> None:
        """
        Save timestamp, trigger time, and phase logs to files.
        
        Args:
            filename_prefix: Prefix for the log filenames
            path: Directory path to save the log files
        """
        # Save combined log with all information
        with open(f'{path}/{filename_prefix}_trigger_data.csv', 'w', newline='') as combined_file:
            writer = csv.writer(combined_file)
            writer.writerow(['Timestamp', 'TriggerTime', 'Phase'])
            for ts, tt, phase in zip(self.timestamp_log, self.triggertimes_log, self.phases_log):
                writer.writerow([ts, tt, phase])

        print(f'Logs saved for subject {filename_prefix} in {path}, with {len(self.timestamp_log)} entries.')
