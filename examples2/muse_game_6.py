# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
import numpy as np
import pyqtgraph as pg
import traceback

# Handy little enum to make code more readable

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

class App(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtWidgets.QVBoxLayout())
        self.setGeometry(0, 0, 1560, 800) 

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtWidgets.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 100, 100))

        ### Set up EEG stream
        """ EXPERIMENTAL PARAMETERS """
        # Modify these to change aspects of the signal processing

        # Length of the EEG data buffer (in seconds)
        # This buffer will hold last n seconds of data and be used for calculations
        self.BUFFER_LENGTH = 1

        # Length of the epochs used to compute the FFT (in seconds)
        self.EPOCH_LENGTH = 0.2

        # Amount of overlap between two consecutive epochs (in seconds)
        self.OVERLAP_LENGTH = 0.8 * self.EPOCH_LENGTH

        # Amount to 'shift' the start of each next consecutive epoch
        self.SHIFT_LENGTH = self.EPOCH_LENGTH - self.OVERLAP_LENGTH

        # Index of the channel(s) (electrodes) to be used
        # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
        self.INDEX_CHANNEL = [0]

        """ 1. CONNECT TO EEG STREAM """
        # Search for active LSL streams
        print('Looking for an EEG stream...')
        self.streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(self.streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')

        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        self.inlet = StreamInlet(self.streams[0], max_chunklen=12)
        self.eeg_time_correction = self.inlet.time_correction()

        # Get the stream info and description
        self.info = self.inlet.info()
        self.description = self.info.desc()

        # Get the sampling frequency
        # This is an important value that represents how many EEG data points are
        # collected in a second. This influences our frequency band calculation.
        # for the Muse 2016, this should always be 256
        self.fs = int(self.info.nominal_srate())

        """ 2. INITIALIZE BUFFERS """

        # Initialize raw EEG data buffer
        self.eeg_buffer = np.zeros((int(self.fs * self.BUFFER_LENGTH), 5))
        self.filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        self.n_win_test = int(np.floor((self.BUFFER_LENGTH - self.EPOCH_LENGTH) /
                                  self.SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        self.band_buffer = np.zeros((self.n_win_test, 4, 5))
        self.buffer_length_for_disp = 512
        self.psd_buffer = np.zeros((self.buffer_length_for_disp, int(160 * self.EPOCH_LENGTH), 5))

        # Selected data from a particular band for plotting purposes
        self.data_buffer1 = np.zeros(self.buffer_length_for_disp)
        self.data_buffer2 = np.zeros(self.buffer_length_for_disp)

        # The try/except structure allows to quit the while loop by aborting the
        # script with <Ctrl-C>
        print('Press Ctrl-C in the console to break the while loop.')

        ### Finish making GUI elements ###########
        #  define image transform
        self.tr = QtGui.QTransform()  # prepare ImageItem transformation:
        self.tr.scale(1,2)       # scale horizontal and vertical axes
        self.tr.translate(-200, 10) # translation
        #  image plot
        self.img = pg.ImageItem(border='w')
        self.img.setTransform(self.tr) 
        self.view.addItem(self.img)

        #  line plot 1
        self.canvas.nextRow()
        self.otherplot2 = self.canvas.addPlot()
        self.otherplot2.setYRange(-10, 10, padding=0)
        self.h2 = self.otherplot2.plot(pen='y')

        #  line plot 2
        self.canvas.nextRow()
        self.otherplot3 = self.canvas.addPlot()
        self.otherplot3.setYRange(-10, 10, padding=0)
        self.h3 = self.otherplot3.plot(pen='w')
        #### Set Data  #####################

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #### Start  #####################
        self._update()

    def _update(self):
        try:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            self.eeg_data, self.timestamp = self.inlet.pull_chunk(
                timeout=1, max_samples=int(self.SHIFT_LENGTH * self.fs))

            # Only keep the channel we're interested in
            try:
                # self.ch_data = np.array(self.eeg_data)[:, self.INDEX_CHANNEL]
                #             # Update EEG buffer with the new data
                # self.l_data = np.array(self.eeg_data)[:, 0:2]
                # self.r_data = np.array(self.eeg_data)[:, 2:4]
                self.eeg_buffer, self.filter_state = utils.update_buffer(
                    self.eeg_buffer, np.array(self.eeg_data), notch=True,
                    filter_state=self.filter_state)

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                self.data_epoch = utils.get_last_data(self.eeg_buffer,
                                                 self.EPOCH_LENGTH * self.fs)
                # print('data epoch')
                # print(self.data_epoch.shape)

                # Compute band powers
                n_bands = self.eeg_buffer.shape[1]
                self.band_powers, self.PSD, self.freqs, self.max_mag = utils.compute_band_powers(self.data_epoch, self.fs)
                # print('band  power')
                # print(self.band_powers.shape)
                # print('PSD')
                # print(self.PSD.shape)
                # print('freqs')
                # print(self.freqs.shape)
                # print('max mag')
                # print(self.max_mag.shape)
                # print('band buffer 1')
                # print(self.band_buffer.shape)
                self.band_buffer, _ = utils.update_buffer(self.band_buffer,
                                                    np.asarray([self.band_powers]))
                # print('band buffer 2')
                # print(self.band_buffer.shape)
                # print('psd buffer 1')
                # print(self.psd_buffer.shape)
                self.psd_buffer, _ = utils.update_buffer(self.psd_buffer,
                                                    np.asarray([self.PSD]))
                # print('psd buffer 2')
                # print(self.psd_buffer.shape)
                # Compute the average band powers for all epochs in buffer
                # This helps to smooth out noise
                self.smooth_band_powers = np.mean(self.band_buffer, axis=0)
                # print(self.smooth_band_powers)

                # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
                #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

                """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
                # These metrics could also be used to drive brain-computer interfaces
                self.BAND_TO_USE = -1

                # Multi Data Protocol:
                if self.BAND_TO_USE == -1:
                    # Raw signal display
                    # Index of the channel(s) (electrodes) to be used
                    # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
                    # Left channel display
                    self.otherplot2.setYRange(-256, 256, padding=0)
                    self.metric1 = self.data_epoch[-1,0]
                    # Right channel display
                    self.otherplot3.setYRange(-256, 256, padding=0)
                    self.metric2 = self.data_epoch[-1,3]
                    if self.max_mag[0] > 300:
                        print("CONGRATS YOU GET A LEFT PRIZE!!!")
                    if self.max_mag[3] > 300:
                        print("CONGRATS YOU GET A RIGHT PRIZE!!!")

                self.data_buffer1 = np.roll(self.data_buffer1, -1)
                self.data_buffer1[-1] = self.metric1
                self.data_buffer2 = np.roll(self.data_buffer2, -1)
                self.data_buffer2[-1] = self.metric2
            except Exception as e:
                print(traceback.format_exc())
                exit()
        except KeyboardInterrupt:
            print('Closing!')

        self.img.setImage(self.psd_buffer[:,0])
        self.h2.setData(self.data_buffer1)
        self.h3.setData(self.data_buffer2)

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        self.counter += 1

if __name__ == "__main__":

    
    """ Start visual aid """
    app = QtWidgets.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())





