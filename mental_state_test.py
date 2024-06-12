# -*- coding: utf-8 -*-
"""
A basic Brain-Computer Interface
=============================================

Description:
We will show how to use an automatic algorithm to
recognize somebody's mental states from their EEG. We will use a classifier,
i.e., an algorithm that, provided some data, learns to recognize patterns,
and can then classify similar unseen information.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import serial
import time
import auxiliary_tools as BCIw
from typing import List, Tuple, Any

def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='BCI Workshop example 2')
    parser.add_argument('channels', metavar='N', type=int, nargs='*', default=[0, 1, 2, 3],
                        help='channel number to use. If not specified, all the channels are used')
    return parser.parse_args()

def connect_to_streams():
    """Connect to EEG, gyro, and accelerometer streams.

    Returns:
        Tuple[Any, Any, Any]: Tuple containing EEG, gyro, and accelerometer streams.
    """
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    gyro = resolve_byprop('type', 'Gyroscope', timeout=2)
    accel = resolve_byprop('type', 'Accelerometer', timeout=2)

    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")
    if len(gyro) == 0:
        raise RuntimeError("Can't find Gyroscope stream.")
    if len(accel) == 0:
        raise RuntimeError("Can't find Accelerometer stream.")

    return streams[0], gyro[0], accel[0]

def get_stream_info(inlet):
    info = inlet.info()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()
    ch = info.desc().child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))
    return fs, n_channels, ch_names

def record_training_data(inlet, training_length, fs, index_channel):
    BCIw.beep()
    print('\nRelax\n')
    eeg_data0, _ = inlet.pull_chunk(timeout=training_length + 1, max_samples=fs * training_length)
    eeg_data0 = np.array(eeg_data0)[:, index_channel]

    BCIw.beep()
    print('\nFocus!\n')
    eeg_data1, _ = inlet.pull_chunk(timeout=training_length + 1, max_samples=fs * training_length)
    eeg_data1 = np.array(eeg_data1)[:, index_channel]

    return eeg_data0, eeg_data1

def get_initial_roll_angle(inlet_gyro, gyro_fs, shift_length):
    """
    Get the initial roll angle using gyroscope data.

    Parameters:
    - inlet_gyro: StreamInlet, gyroscope inlet stream.
    - gyro_fs: int, gyroscope sampling frequency.
    - shift_length: float, length of the shift window.

    Returns:
    - initial_roll_angle: float, the initial roll angle.
    """
    gyro_data, gyro_timestamp = inlet_gyro.pull_chunk(timeout=1, max_samples=int(shift_length * gyro_fs))
    gyro_data = np.array(gyro_data)
    accel_data, _ = inlet_gyro.pull_chunk(timeout=1, max_samples=int(shift_length * gyro_fs))
    accel_data = np.array(accel_data)

    dt = 1.0 / gyro_fs  # Assuming the time step is the inverse of the sampling frequency

    roll, pitch, yaw = BCIw.complementary_filter(gyro_data, accel_data, dt)
    
    return roll[0]  # Return the first roll angle as the initial roll angle

def main():
    # Define shift_length and other parameters
    shift_length = 0.4

    try:
        arduino = serial.Serial(port='COM7', timeout=0)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        arduino = None

    # Parse arguments
    args = parse_arguments()
    
    # Connect to streams
    eeg_stream, gyro_stream, accel_stream = connect_to_streams()

    print("Start acquiring data")
    inlet = StreamInlet(eeg_stream, max_chunklen=12)
    inlet_gyro = StreamInlet(gyro_stream, max_chunklen=12)
    inlet_accel = StreamInlet(accel_stream, max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    fs, n_channels, ch_names = get_stream_info(inlet)
    gyro_fs, _, _ = get_stream_info(inlet_gyro)
    accel_fs, _, _ = get_stream_info(inlet_accel)

    # Get initial roll angle
    initial_roll_angle = get_initial_roll_angle(inlet_gyro, gyro_fs, shift_length)

    index_channel = args.channels
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    feature_names = BCIw.get_feature_names(ch_names)

    training_length = 20
    eeg_data0, eeg_data1 = record_training_data(inlet, training_length, fs, index_channel)

    eeg_epochs0 = BCIw.epoch(eeg_data0, int(2 * fs), int(0.8 * fs))
    eeg_epochs1 = BCIw.epoch(eeg_data1, int(2 * fs), int(0.8 * fs))

    feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
    feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)

    classifier, mu_ft, std_ft, score = BCIw.train_classifier(feat_matrix0, feat_matrix1)
    print(f'{score * 100}% correctly predicted')
    BCIw.beep()

    eeg_buffer = np.zeros((int(fs * 15), n_channels))
    decision_buffer = np.zeros((30, 1))
    plotter_decision = BCIw.DataPlotter(30, ['Decision'])

    print('Press Ctrl-C in the console to break the while loop.')
    try:
        previous_timestamp = 0
        position = [0, 0, 0]
        while True:
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(2 * fs))
            ch_data = np.array(eeg_data)[:, index_channel]

            eeg_buffer, _ = BCIw.update_buffer(eeg_buffer, ch_data, notch=True)

            data_epoch = BCIw.get_last_data(eeg_buffer, int(2 * fs))
            feat_vector = BCIw.compute_feature_vector(data_epoch, fs)

            y_hat = BCIw.test_classifier(classifier, feat_vector.reshape(1, -1), mu_ft, std_ft)

            gyro_data, gyro_timestamp = inlet_gyro.pull_chunk(timeout=1, max_samples=int(shift_length * gyro_fs))
            gyro_data = np.array(gyro_data)
            gyro_timestamp = np.array(gyro_timestamp)

            dt = BCIw.calculate_dt(previous_timestamp, gyro_timestamp[0]) if previous_timestamp != 0 else 0
            previous_timestamp = gyro_timestamp[-1]

            accel_data, _ = inlet_accel.pull_chunk(timeout=1, max_samples=int(shift_length * accel_fs))
            accel_data = np.array(accel_data)

            roll, pitch, yaw = BCIw.complementary_filter(gyro_data, accel_data, dt)
            mean_roll = np.mean(roll)
            mean_roll = mean_roll.round(1)
            print("Roll Angle:", mean_roll)

            position += gyro_data.mean(axis=0)
            if y_hat == 0:
                if arduino:
                    arduino.write(str.encode('0'))
                print('0')
            else:
                if mean_roll > 100:
                    if arduino:
                        arduino.write(str.encode('l'))
                    print('l')
                elif position[0] < -100:
                    if arduino:
                        arduino.write(str.encode('r'))
                    print('r')
                else:
                    if arduino:
                        arduino.write(str.encode('c'))
                    print('c')

            decision_buffer, _ = BCIw.update_buffer(decision_buffer, np.reshape(y_hat, (-1, 1)))
            plotter_decision.update_plot(decision_buffer)
            plt.pause(0.00001)

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        if arduino:
            arduino.close()
        print('Closed!')

if __name__ == "__main__":
    main()

