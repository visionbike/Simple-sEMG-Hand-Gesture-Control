from typing import Any
from collections import deque
import threading
import time
import argparse
import numpy as np
import serial
import torch
from utils import load_model
from network import SpectrogramAttResNet
from preprocessing import filter2d, normalize, spectrogram2d


class Serial:
    def __init__(self,
                 port: str = 'COM1',
                 baudrate: int = 115200,
                 num_channels: int = 4,
                 seq_len: int = 1000,
                 interval: int = 50,
                 timeout: float = 0.1,
                 clf: Any = None):
        self.serial = serial.Serial(port, baudrate)
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.interval = interval
        self.timeout = timeout
        self.clf = clf
        self.cmd = '0'
        self.buffer = deque(maxlen=int(num_channels * self.interval))
        self.window = deque(maxlen=int(num_channels * self.seq_len))

    def serial_open(self):
        if not self.serial.isOpen():
            self.serial.opent()

    def serial_close(self):
        if self.serial.isOpen():
            self.serial.close()

    def serial_send(self):
        # time.sleep(self.timeout)
        while True:
            self.serial.write(self.cmd.encode('utf-8'))

    def serial_read(self):
        print('Receiving signal...')
        cmd = '0'
        while True:
            string = self.serial.readline().decode('utf-8').rstrip()  # read and decode a byte string
            vals = [float(v) for v in string.split(' ')]
            self.buffer.extend(vals)    # add value into buffer first
            if len(self.buffer) == self.buffer.maxlen:
                self.window.extend(self.buffer)     # add buffer to window
            if len(self.window) == self.window.maxlen:
                # start prediction
                # convert 1D sequence to 2D signal in shape of (n_channel, n_len)
                emg = np.reshape(np.asarray(self.window, dtype=float), (self.num_channels, self.seq_len), order='F')
                emg = filter2d(emg, kernel_size=0, high_cut=5, order=4, mode='highpass')
                emg = normalize(emg)
                emg = spectrogram2d(emg, self.seq_len, self.interval)
                emg = torch.from_numpy(emg).unsqueeze(0).to('cpu', dtype=torch.float)
                out = clf(emg)
                score = torch.softmax(out, dim=1)
                prob = score.data.max(dim=1)[0]  # get the max probability
                idx = score.data.max(dim=1)[1].cpu().numpy()[0]
                if prob < 0.5:
                    self.cmd = cmd
                if prob > 0.5:
                    # send current predicted action if the confident > 0.5
                    cmd = str(idx)
                    self.cmd = cmd
                time.sleep(self.timeout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time robot-arm controlling')
    parser.add_argument('-p', type=str, default='COM1', help='COM port')
    parser.add_argument('-b', type=int, default=112500, help='Baud rate')
    parser.add_argument('-n', type=int, default=4, help='The number of channels')
    parser.add_argument('-s', type=int, default=1000, help='Sequence length')
    parser.add_argument('-i', type=int, default=50, help='The number of interval')
    parser.add_argument('-t', type=float, default=0.5, help='Time out (s)')
    args = parser.parse_args()

    # setup network
    clf = SpectrogramAttResNet(arch='resnet18', in_channels=4, out_channels=5, n_filters=64)
    clf = load_model(clf, path='./models/best_spectrum_resnet18_att.pth')
    clf = clf.to('cpu')
    clf.eval()

    # setup serial line
    serial_ = Serial(args.p, args.b, args.n, args.s, args.i, args.t, clf)
    t1 = threading.Thread(target=serial_.serial_read)
    t1.start()

    try:
        while True:
            serial_.serial_send()
    except KeyboardInterrupt:
        print('Press Ctrl-C to terminate while statement')
    serial_.serial_close()

