# Load nos files from EU-RADION devices v0.0 - B. Klis 2022

import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import struct
from pathlib import Path


class Load_nos:

    def __init__(self, filename=None) -> None:
        # file format specification:
        self.SECTION = 0  # current section
        self.FILEINFO_SECTION = 1
        self.DATA_SECTION = 2
        self.RAWDATA_SECTION = 3
        self.METHOD_SECTION = 4
        self.PARAMETER_SECTION = 5
        # Data:
        self.channels_const = 4092  # Number of channels
        self.spectra = None  # Data matrix
        self.time = 0  # Data lenght

        if (filename != None):
            self.load_spectra(filename)

    def save_to_csv(self, spectra):
        filename = "spectra.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(spectra)


    # Get array:
    def get_spectra_at_time(self, time) -> np.array:
        return self.spectra[time]

    # Plot spectra data:
    def simple_plot(self, time=None) -> None:
        if time == None:
            time = self.time - 1
        plt.plot(range(self.channels_const), self.get_spectra_at_time(time), '-r')
        plt.title('Spectra data')
        plt.xlabel('Channel number')
        plt.ylabel('Counts')
        plt.show()

    # Load only spectra data:
    def load_spectra(self, filename) -> None:
        channels_const = self.channels_const  # Number of channels
        scm_filename = re.sub('.nos', '.scm', filename)  # binary file with spectra data
        max_vectors = 0

        with open(filename, 'rt') as f:

            for line in f:
                if line == '':
                    continue
                if '[Data]' in line:
                    self.SECTION = self.DATA_SECTION
                if '[RAWDATA]' in line:
                    break
                if self.SECTION == self.DATA_SECTION:
                    words = line[0:7]
                    try:
                        float(words)
                        max_vectors += 1
                    except ValueError:
                        print("Not a float")
        print(max_vectors / 2)
        self.time = int(max_vectors / 2)
        self.spectra = np.zeros((self.time, channels_const))

        num_vector = 0  # Iterate trough vectors:
        spectra_pos = np.zeros(channels_const + 1)
        with open(scm_filename, 'rb') as f2:
            while num_vector < self.time:
                for i in range(channels_const + 1):
                    data_chunk = f2.read(4)
                    if len(data_chunk) == 4:
                        byte_string = struct.unpack('f', data_chunk)
                    else:
                        print("End of binary data!")
                    values = re.findall("\d+\.\d+", str(byte_string))
                    spectra_pos[i] = float(values[0])  # Float32
                if spectra_pos[0] > 1.5:  # God knows why... some space is not used
                    for i in range(channels_const):
                        self.spectra[num_vector, i] = spectra_pos[i + 1]
                num_vector += 1
            self.save_to_csv(spectra_pos)


def main():
    # Loading object:
    loader = Load_nos('euradion2_co60_cs137_11042022.nos')
    loader.load_spectra('euradion2_co60_cs137_11042022.nos')
    # Very simple ploting method:
    print(loader.spectra.shape)
    loader.simple_plot()

    # Get array for given moment:
    arr1 = loader.get_spectra_at_time(20)
    print(arr1)


if __name__ == '__main__': main()