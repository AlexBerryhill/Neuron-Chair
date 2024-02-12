# -*- coding: utf-8 -*-
"""
A basic Brain-Computer Interface
=============================================

Description:
We will how to use an automatic algorithm to
recognize somebody's mental states from their EEG. We will use a classifier,
i.e., an algorithm that, provided some data, learns to recognize patterns,
and can then classify similar unseen information.

"""

import argparse
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import serial # Module to send data to Arduino
import time
import auxiliary_tools as BCIw  # Our own functions for the workshop
import math

if __name__ == "__main__":

        # """ Open Serial port to Arduino """
        # arduino = serial.Serial(port = 'COM7', timeout=0)
        time.sleep(2)  

        """ 0. PARSE ARGUMENTS """
        parser = argparse.ArgumentParser(description='BCI Workshop example 2')
        parser.add_argument('channels', metavar='N', type=int, nargs='*',
        default=[0, 1, 2, 3],
        help='channel number to use. If not specified, all the channels are used')

        args = parser.parse_args()

        """ 1. CONNECT TO EEG STREAM """

        # Search for active LSL stream
        print('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        gyro = resolve_byprop('type', 'GYRO', timeout=2)
        accel = resolve_byprop('type', 'Accelerometer', timeout=2)

        if len(streams) == 0:
                raise RuntimeError('Can\'t find EEG stream.')

        if len(gyro) == 0:
                gyro = resolve_byprop('type', 'Gyroscope', timeout=2)
                if len(gyro) == 0:
                        raise RuntimeError('Can\'t find Gyro stream.')

        if len(accel) == 0:
                gyro = resolve_byprop('type', 'Accelerometer', timeout=2)
                if len(accel) == 0:
                        raise RuntimeError('Can\'t find Accel stream.')

        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        inlet_gyro = StreamInlet(gyro[0], max_chunklen=12)
        inlet_accel = StreamInlet(accel[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()
        gyro_time_correction = inlet_gyro.time_correction()
        

        # Get the stream info, description, sampling frequency, number of channels
        info = inlet.info()
        description = info.desc()
        fs = int(info.nominal_srate())
        n_channels = info.channel_count()

        # Get the stream info, description, sampling frequency, number of channels
        gyro_info = inlet_gyro.info()
        gyro_description = gyro_info.desc()
        gyro_fs = int(gyro_info.nominal_srate())
        n_gyro_channels = gyro_info.channel_count()

        # Get the stream info, description, sampling frequency, number of channels
        accel_info = inlet_accel.info()
        accel_description = accel_info.desc()
        accel_fs = int(accel_info.nominal_srate())
        n_accel_channels = accel_info.channel_count()

        # Get names of all channels
        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]
        for i in range(1, n_channels):
                ch = ch.next_sibling()
                ch_names.append(ch.child_value('label'))

        """ 2. SET EXPERIMENTAL PARAMETERS """

        # Length of the EEG data buffer (in seconds)
        # This buffer will hold last n seconds of data and be used for calculations
        buffer_length = 15

        # Length of the epochs used to compute the FFT (in seconds)
        epoch_length = 2

        # Amount of overlap between two consecutive epochs (in seconds)
        overlap_length = 0.8

        # Amount to 'shift' the start of each next consecutive epoch
        shift_length = epoch_length - overlap_length

        # Index of the channel (electrode) to be used
        # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
        index_channel = args.channels
        # Name of our channel for plotting purposes
        ch_names = [ch_names[i] for i in index_channel]
        n_channels = len(index_channel)

        # Get names of features
        # ex. ['delta - CH1', 'pwr-theta - CH1', 'pwr-alpha - CH1',...]
        feature_names = BCIw.get_feature_names(ch_names)

        # Number of seconds to collect training data for (one class)
        training_length = 20

        """ 3. RECORD TRAINING DATA """

        # Record data for mental activity 0
        BCIw.beep()
        eeg_data0, timestamps0 = inlet.pull_chunk(
                timeout=training_length+1, max_samples=fs * training_length)
        eeg_data0 = np.array(eeg_data0)[:, index_channel]

        print('\nClose your eyes!\n')

        # Record data for mental activity 1
        BCIw.beep()  # Beep sound
        eeg_data1, timestamps1 = inlet.pull_chunk(
                timeout=training_length+1, max_samples=fs * training_length)
        eeg_data1 = np.array(eeg_data1)[:, index_channel]
        
        # Divide data into epochs
        eeg_epochs0 = BCIw.epoch(eeg_data0, epoch_length * fs,
                                overlap_length * fs)
        eeg_epochs1 = BCIw.epoch(eeg_data1, epoch_length * fs,
                                overlap_length * fs)
        
        """ 4. COMPUTE FEATURES AND TRAIN CLASSIFIER """
        
        feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
        feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)
        
        [classifier, mu_ft, std_ft, score] = BCIw.train_classifier(
                feat_matrix0, feat_matrix1, 'SVM')

        print(str(score * 100) + '% correctly predicted')

        BCIw.beep()

        """ 5. USE THE CLASSIFIER IN REAL-TIME"""

        # Initialize the buffers for storing raw EEG and decisions
        eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
        filter_state = None  # for use with the notch filter
        decision_buffer = np.zeros((30, 1))

        plotter_decision = BCIw.DataPlotter(30, ['Decision'])

        # The try/except structure allows to quit the while loop by aborting the
        # script with <Ctrl-C>
        print('Press Ctrl-C in the console to break the while loop.')
        try:
                previous_timestamp = 0 # The starting timestamp
                position = [0,0,0]
                starting_roll = [135, -45, -45]
                while True:

                        """ 3.1 ACQUIRE DATA """
                        # Obtain EEG data from the LSL stream
                        eeg_data, timestamp = inlet.pull_chunk(
                                timeout=1, max_samples=int(shift_length * fs))

                        # Only keep the channel we're interested in
                        ch_data = np.array(eeg_data)[:, index_channel]

                        # Update EEG buffer
                        eeg_buffer, filter_state = BCIw.update_buffer(
                                eeg_buffer, ch_data, notch=True,
                                filter_state=filter_state)

                        """ 3.2 COMPUTE FEATURES AND CLASSIFY """
                        # Get newest samples from the buffer
                        data_epoch = BCIw.get_last_data(eeg_buffer,
                                                        epoch_length * fs)

                        # Compute features
                        feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
                        y_hat = BCIw.test_classifier(classifier,
                                                        feat_vector.reshape(1, -1), mu_ft,
                                                        std_ft)
                        """ 3.3 Calculate Angle """
                        # Obtain gyroscope data
                        gyro_data, gyro_timestamp = inlet_gyro.pull_chunk(
                                timeout=1, max_samples=int(shift_length * gyro_fs))
                        gyro_data = np.array(gyro_data)
                        gyro_timestamp = np.array(gyro_timestamp)
                        
                        # Calculate current dt
                        dt = BCIw.calculate_dt(previous_timestamp, gyro_timestamp[0]) if previous_timestamp != 0 else 0
                        previous_timestamp = gyro_timestamp[-1]

                        # Obtain accelerometer data
                        accel_data, accel_timestamp = inlet_accel.pull_chunk(
                                timeout=1, max_samples=int(shift_length * accel_fs))
                        accel_data = np.array(accel_data)

                        # Process gyro_data using complementary filter or other methods
                        roll, pitch, yaw = BCIw.complementary_filter(gyro_data, accel_data, dt)
                        roll = roll.round(1)-starting_roll
                        # Print or use the angles as needed
                        print("Roll Angle:", roll.round(1)-starting_roll)
                        # print("Pitch Angle:", pitch.round(1))
                        # print("Yaw Angle:", yaw.round(1))
                        position += gyro_data.mean(axis=0)
                        if(y_hat == 0):
                                # arduino.write(str.encode('0'))
                                print('0')
                        else:
                                # arduino.write(str.encode('1'))
                                # print(gyro_data.mean(axis=0))
                                if (roll[1]>100):
                                        # arduino.write(str.encode('l')) #letter L
                                        print('l')
                                elif (position[0] < -100): #left
                                        # arduino.write(str.encode('r'))
                                        print('r')
                                else: #center
                                        # arduino.write(str.encode('c'))
                                        print('c')

                        decision_buffer, _ = BCIw.update_buffer(decision_buffer,
                                                                np.reshape(y_hat, (-1, 1)))

                        """ 3.3 VISUALIZE THE DECISIONS """
                        plotter_decision.update_plot(decision_buffer)
                        plt.pause(0.00001)

        except KeyboardInterrupt:
                print('Closed!')