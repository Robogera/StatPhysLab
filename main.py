import sys
from PyQt6 import QtWidgets
import design
from PyQt6.QtCore import QTimer,QDateTime

import threading
from scipy.integrate import simps
from scipy.stats import norm
import numpy as np

class LabApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):

        super().__init__()
        self.setupUi(self)
        
        self.experimentsTotal = 0
        self.detectionSuccess = 0

        self.threshold = 0
        self.snr = 1
        self.delay = 0

        self.noiseMode = 0
        self.signalMode = 0

        self.dialSNR.setMinimum(200)
        self.dialSNR.setMaximum(5000)
        self.dialSNR.setValue(self.snr * 1000)
        self.dialSNR.valueChanged.connect(self.dialMoved)

        self.dialDelay.setMinimum(-350)
        self.dialDelay.setMaximum(350)
        self.dialDelay.setValue(self.delay * 1000)
        self.dialDelay.valueChanged.connect(self.dialMoved)

        self.dialH.setMinimum(-3000)
        self.dialH.setMaximum(3000)
        self.dialH.setValue(self.threshold * 1000)
        self.dialH.valueChanged.connect(self.dialMoved)

        self.switchNoise.setMinimum(0)
        self.switchNoise.setMaximum(1)
        self.switchNoise.setValue(self.noiseMode)
        self.switchNoise.setTickInterval(1)
        self.switchNoise.valueChanged.connect(self.switchMoved)
        
        self.switchSignal.setMinimum(0)
        self.switchSignal.setMaximum(2)
        self.switchSignal.setValue(self.signalMode)
        self.switchSignal.setTickInterval(1)
        self.switchSignal.valueChanged.connect(self.switchMoved)

        self.buttonStop.setEnabled(False)
        
        self.timer=QTimer()
        self.timer.timeout.connect(self.experiment)

        self.timerPlot = QTimer()
        self.timerPlot.timeout.connect(self.plt)
        self.timerPlot.start(50)

        self.buttonStart.clicked.connect(self.startTimer)
        self.buttonStop.clicked.connect(self.endTimer)
        self.buttonReset.clicked.connect(self.reset)

        self.plot = self.graphicsView.plot()
        self.graphicsView.hideAxis('bottom')
        self.graphicsView.hideAxis('left')
        self.graphicsView.setMouseEnabled(x=False, y=False)


    def plt(self):
        T = 1
        samples = 256
        length = 0.2
        rc = 0.05
        fakenoise = self.noiseMode * np.random.normal(0,np.sqrt(1/self.snr), samples)
        time_points = np.linspace( 0, T, samples )
        if self.signalMode == 0:
            signal = np.array([0 for i in range(samples)])
        elif self.signalMode == 1:
            signal = np.array( [ 1 if T/2 + self.delay - length/2 <= time_points[ i ] <= T/2 + self.delay + length/2 else 0 for i in range( samples ) ] )
        elif self.signalMode == 2:
            signal = np.empty(shape = samples)
            signalMax = (1 - np.exp(-(T/2 + self.delay + length/2)/rc))
            for i in range( samples ):
                t = time_points[i]
                if 0 <= t < T/2 + self.delay - length/2:
                    signal[i] = 0
                elif t < T/2 + self.delay + length/2:
                    signal[i] = ( 1 - np.exp(-(t - (T/2 + self.delay - length/2))/rc))
                else:
                    signal[i] = signalMax * np.exp(-(t - (T/2 + self.delay + length/2))/rc)
        self.plot.setData(time_points, signal + fakenoise)

    def reset(self):
        self.experimentsTotal = 0
        self.detectionSuccess = 0
        self.lcdTotal.display(self.experimentsTotal)
        self.lcdSuccess.display(self.detectionSuccess)

    def startTimer(self):
        self.timer.start(50)
        self.buttonStart.setEnabled(False)
        self.buttonStop.setEnabled(True)

    def endTimer(self):
        self.timer.stop()
        self.buttonStart.setEnabled(True)
        self.buttonStop.setEnabled(False)
    
    def experiment(self):
        T = 1
        samples = 128
        length = 0.2
        rc = 0.05

        time_points = np.linspace( 0, T, samples )
        signalRef = np.array( [ 1 if T/2 - length/2 <= time_points[ i ] <= T/2 + length/2 else 0 for i in range( samples ) ] )
        if self.signalMode == 0:
            signal = np.array([0 for i in range(samples)])
        elif self.signalMode == 1:
            signal = np.array( [ 1 if T/2 + self.delay - length/2 <= time_points[ i ] <= T/2 + self.delay + length/2 else 0 for i in range( samples ) ] )
        elif self.signalMode == 2:
            signal = np.empty(shape = samples)
            signalMax = (1 - np.exp(-(T/2 + self.delay + length/2)/rc))
            for i in range( samples ):
                t = time_points[i]
                if 0 <= t < T/2 + self.delay - length/2:
                    signal[i] = 0
                elif t < T/2 + self.delay + length/2:
                    signal[i] = ( 1 - np.exp(-(t - (T/2 + self.delay - length/2))/rc))
                else:
                    signal[i] = signalMax * np.exp(-(t - (T/2 + self.delay + length/2))/rc)

        energy = simps( signal * signal, time_points )
        noise = self.noiseMode * np.random.normal( 0, np.sqrt( length * samples / T / self.snr), size = samples )
        integrand = (noise + signal ) * signalRef
        self.experimentsTotal += 1
        if simps( integrand, time_points ) >= self.threshold * energy/self.snr - energy:
            self.detectionSuccess += 1

        self.lcdTotal.display(self.experimentsTotal)
        self.lcdSuccess.display(self.detectionSuccess)

    def dialMoved(self):
        origin = self.sender()
        
        if origin == self.dialH:
            self.threshold = self.dialH.value() / 1000
            self.lcdH.display(self.threshold)
        elif origin == self.dialSNR:
            self.snr = self.dialSNR.value() / 1000
            self.lcdSNR.display(self.snr)
        elif origin == self.dialDelay:
            self.delay = self.dialDelay.value() / 1000
            self.lcdDelay.display(self.delay)


    def switchMoved(self):
        origin = self.sender()
        
        if origin == self.switchNoise:
            self.noiseMode = self.switchNoise.value()
        elif origin == self.switchSignal:
            self.signalMode = self.switchSignal.value()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = LabApp()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()