from os import path
import csv
import numpy as np
import requests
from scipy.signal import argrelmax

class NuclideIdentifier:
    def __init__(self):
        self.sample = self.read_sample()
        self.peaks = self.read_peaks()
        self.database_array, self.database_dict = self.get_database()
        self.identify_nuclide()

    def get_energy_values(self):
        file_path =  "./res/gf_C12137_2.csv"
        energy_values = []
        with open(file_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                energy_values.append(float(row[0]))
                energy_values.append(float(row[0]))
        return energy_values

    def find_peaks(self):
        histogram = [float(line[1]) for line in self.sample]
        histogram = np.asarray(histogram)
        peaks = argrelmax(histogram)
        energy_values = self.get_energy_values()
        for peak in peaks:
            peak_energy = [energy_values[p] for p in peak]
        peak_energy_path = path.abspath("./res/Na_peaks.csv")
        with open(peak_energy_path, 'w') as f:
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(peak_energy)

    def read_peaks(self):
        file_path = path.abspath("./res/Na_peaks.csv")
        with open(file_path, 'r') as file:
            csv_peaks = csv.reader(file)
            peaks = [float(peak_value) for peak in csv_peaks for peak_value in peak]
        return peaks

    def read_sample(self):
        file_path = path.abspath("./res/meas.txt")
        Na_path = path.abspath("./res/Na.txt")
        with open(Na_path, 'r') as file:
            sample = [line.split() for line in file]
        return sample

    def reform_database(self, database):
        new_database = []
        #nuclides dictionary in a easier form for later calculations
        nuclides_dict = {}
        for nuclide in database:
            energy_peaks = []
            intensity = []
            for peaks in database[nuclide]:
                energy_peaks.append(peaks['energy'])
                intensity.append(peaks['intensity'])
                new_database.append([nuclide, peaks['energy'], peaks['intensity']])
            nuclides_dict[nuclide] = [energy_peaks, intensity]
        new_database = np.array(new_database)
        return new_database, nuclides_dict

    def get_database(self):
        x = requests.get('http://25.20.235.8:8000/lhnb/list')
        data = x.json()
        nuclides_peaks = {}
        for nuclide in data['available nuclides']:
            r = requests.get(f'http://25.20.235.8:8000/lhnb/nuclide/{nuclide}')
            nuclide_info = r.json()
            nuclides_peaks[nuclide] = nuclide_info['spectra_peaks']
        new_database_array, new_database_dict = self.reform_database(nuclides_peaks)
        return new_database_array, new_database_dict


    def check_nuclide_lines(self, nuclide, similar_peaks, energy_peaks):
        missing_peaks = [p for p in energy_peaks if p not in similar_peaks]
        if len(missing_peaks) > len(energy_peaks)//2:
            #print(f'Two many lines missing for this {nuclide} with {len(missing_peaks)} missing and {len(energy_peaks)} total')
            return False
        else:
            print(f'{nuclide} has {len(missing_peaks)} missing and {len(energy_peaks)} total')
            return True

    def check_candidate(self, candidate):
        energies, intensities= self.database_dict[candidate]
        sorted_peaks = [[e, i] for e, i in sorted(zip(energies, intensities))][:10]
        #make better ways of including similar peaks when more values are included
        missing_peaks = [p for p in sorted_peaks if np.around(p) not in np.around(self.peaks)]
        if len(missing_peaks) > 5:
            #print(f'Too many missing peaks {len(missing_peaks)} for {candidate}')
            return False
        else:
            return True


    def identify_nuclide(self):
        #candidates = [self.database_dict.keys()]
        similar_peaks = []
        all_peaks_energy = self.database_array[:,1].astype(np.float64)
        for sample_peak in self.peaks:
            difference = abs(all_peaks_energy - sample_peak)
            idx = np.argmin(difference)
            similar_peaks = [i for i in difference if i < 2]
            candidate = self.database_array[idx][0]
            if self.check_candidate(candidate):
                print(f'Match for peak {sample_peak} is  {candidate} with peak {self.database_array[idx][1]}')



        """nuclide_candidates = []
        similar_peaks = []
        energy_peaks = []
        intensity = []
        nuclide = None
        if energy_peaks:
            for sample_peak in self.peaks:
                similar_peak = min(energy_peaks, key=lambda x: abs(x - sample_peak))
                if abs(similar_peak - sample_peak) > 4:
                    continue
                else:
                    similar_peaks.append(similar_peak)
                    index = energy_peaks.index(similar_peak)
            if self.check_nuclide_lines(nuclide, similar_peaks, energy_peaks):
                nuclide_candidates.append((nuclide, intensity[index]))"""







