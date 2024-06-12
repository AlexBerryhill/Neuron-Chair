from typing import List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch

def beep() -> None:
    """Produce a simple beep sound."""
    print('\a')

def get_feature_names(ch_names: List[str]) -> List[str]:
    """Generate feature names based on channel names.

    Args:
        ch_names (List[str]): List of channel names.

    Returns:
        List[str]: List of feature names.
    """
    feature_names = []
    for ch in ch_names:
        feature_names.extend([
            f'delta - {ch}', f'theta - {ch}', f'alpha - {ch}', f'beta - {ch}'
        ])
    return feature_names

def epoch(data: np.ndarray, epoch_length: int, overlap_length: int) -> np.ndarray:
    """Segment data into epochs.

    Args:
        data (np.ndarray): Input data.
        epoch_length (int): Length of each epoch.
        overlap_length (int): Amount of overlap between epochs.

    Returns:
        np.ndarray: Array of epochs.
    """
    n_samples = len(data)
    step = epoch_length - overlap_length
    n_epochs = int((n_samples - overlap_length) / step)
    epochs = np.array([data[i * step:i * step + epoch_length] for i in range(n_epochs)])
    return epochs

def compute_feature_matrix(epochs: np.ndarray, fs: int) -> np.ndarray:
    """Compute feature matrix from epochs.

    Args:
        epochs (np.ndarray): Array of epochs.
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: Feature matrix.
    """
    n_epochs, _, n_channels = epochs.shape
    feature_matrix = np.zeros((n_epochs, n_channels * 4))
    for i in range(n_epochs):
        for j in range(n_channels):
            feature_matrix[i, j * 4:(j + 1) * 4] = compute_features(epochs[i, :, j], fs)
    return feature_matrix

def compute_features(data: np.ndarray, fs: int) -> np.ndarray:
    """Compute features for a single epoch of a single channel.

    Args:
        data (np.ndarray): EEG data for a single epoch.
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: Feature vector.
    """
    bandpowers = np.log10(np.var(data))
    return bandpowers

def train_classifier(feat_matrix0: np.ndarray, feat_matrix1: np.ndarray, classifier_type: str='SVM'):
    """
    Train a classifier using two sets of feature matrices.

    Args:
        feat_matrix0 (np.ndarray): Feature matrix for the first class.
        feat_matrix1 (np.ndarray): Feature matrix for the second class.
        classifier_type (str): Type of classifier to use.

    Returns:
        Any: Trained classifier.
    """
    X = np.vstack((feat_matrix0, feat_matrix1))
    y = np.hstack((np.zeros(len(feat_matrix0)), np.ones(len(feat_matrix1))))
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    classifier = SVC().fit(X_scaled, y)
    score = classifier.score(X_scaled, y)
    return classifier, scaler.mean_, scaler.scale_, score

def update_buffer(buffer, new_data, notch=False, filter_state=None):
    new_buffer = np.roll(buffer, -len(new_data), axis=0)
    new_buffer[-len(new_data):] = new_data
    return new_buffer, filter_state

def get_last_data(buffer, epoch_length):
    return buffer[-epoch_length:]

def compute_feature_vector(data_epoch, fs):
    n_channels = data_epoch.shape[1]
    feature_vector = np.zeros(n_channels * 4)
    for j in range(n_channels):
        feature_vector[j * 4:(j + 1) * 4] = compute_features(data_epoch[:, j], fs)
    return feature_vector

def test_classifier(classifier, feat_vector, mu_ft, std_ft):
    # Ensure feat_vector has the same shape as mu_ft and std_ft
    print(f'Feature vector shape: {feat_vector.shape}')
    print(f'Mean shape: {mu_ft.shape}, Std shape: {std_ft.shape}')
    feat_vector = (feat_vector - mu_ft) / std_ft
    return classifier.predict(feat_vector)

def calculate_dt(prev_timestamp, curr_timestamp):
    return curr_timestamp - prev_timestamp

def complementary_filter(gyro_data: np.ndarray, accel_data: np.ndarray, dt: float, alpha: float = 0.98) -> tuple:
    """
    Apply a complementary filter to combine gyroscope and accelerometer data to compute roll, pitch, and yaw.
    
    Parameters:
    - gyro_data: numpy.ndarray, gyroscope data with shape (N, 3), where N is the number of samples.
    - accel_data: numpy.ndarray, accelerometer data with shape (N, 3), where N is the number of samples.
    - dt: float, time difference between successive gyro measurements.
    - alpha: float, filter coefficient (0 < alpha < 1). Higher alpha relies more on gyro, lower alpha on accelerometer.
    
    Returns:
    - roll: numpy.ndarray, calculated roll angles.
    - pitch: numpy.ndarray, calculated pitch angles.
    - yaw: numpy.ndarray, calculated yaw angles.
    """
    # Initialize angles
    roll, pitch, yaw = np.zeros(gyro_data.shape[0]), np.zeros(gyro_data.shape[0]), np.zeros(gyro_data.shape[0])

    # Accelerometer-based initial angle calculations
    accel_roll = np.arctan2(accel_data[:, 1], accel_data[:, 2]) * 180 / np.pi
    accel_pitch = np.arctan2(-accel_data[:, 0], np.sqrt(accel_data[:, 1]**2 + accel_data[:, 2]**2)) * 180 / np.pi

    for i in range(gyro_data.shape[0]):
        if i == 0:
            roll[i] = accel_roll[i]
            pitch[i] = accel_pitch[i]
            yaw[i] = 0  # Initial yaw is set to 0
        else:
            # Gyroscope angles (integrated gyroscope rate data)
            roll_gyro = roll[i - 1] + gyro_data[i, 0] * dt
            pitch_gyro = pitch[i - 1] + gyro_data[i, 1] * dt
            yaw_gyro = yaw[i - 1] + gyro_data[i, 2] * dt

            # Complementary filter to combine accelerometer and gyroscope data
            roll[i] = alpha * roll_gyro + (1 - alpha) * accel_roll[i]
            pitch[i] = alpha * pitch_gyro + (1 - alpha) * accel_pitch[i]
            yaw[i] = yaw_gyro  # Yaw cannot be determined from accelerometer data

    return roll, pitch, yaw

class DataPlotter:
    def __init__(self, buffer_length, ch_names):
        self.buffer_length = buffer_length
        self.ch_names = ch_names
        self.fig, self.ax = plt.subplots()
        self.lines = {ch: self.ax.plot(np.zeros(buffer_length), label=ch)[0] for ch in ch_names}
        self.ax.legend()

    def update_plot(self, new_data):
        for i, ch in enumerate(self.ch_names):
            self.lines[ch].set_ydata(np.roll(self.lines[ch].get_ydata(), -len(new_data)))
            self.lines[ch].get_ydata()[-len(new_data):] = new_data[:, i]
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
