from os import path
import csv
import numpy as np
import requests
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelextrema
from collections import OrderedDict, Iterable

class NuclideIdentifier:
    def __init__(self):
        self.sample = self.read_sample()
        self.energy_values = self.get_energy_values()
        #self.peaks = self.find_peaks()
        self.peaks = self.interpolate_bkg(self.sample)
        self.database_array, self.database_dict = self.get_database()
        self.identify_nuclide()

    def get_energy_values(self):
        file_path =  "../res/gf_C12137_2.csv"
        energy_values = []
        with open(file_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                energy_values.append(float(row[0]))
        return energy_values

    def interpolate_bkg(self, counts, channels=None, window=5, order=3, scale=0.5, energy_cal=None):
        """
           Method to identify the background of a spectrum by looking at absolute values of the slopes of spectrum and applying
           a moving average with window size wndw, order times on the slopes. From this, an estimate of what is background and
           what not is made by looking at the resulting mean of the moving average with some scale. An interpolation of the data
           is made. If a calibration is provided, the channels are translated into energies beforehand.
           Parameters
           ----------
           counts : array
               array of counts
           channels : array or None
               array of channels; if None will be np.arange(len(counts))
           window : int
               window size ov moving average
           order : int
               order of how many time the average is applied. For each application i, the window decreases window*(order -i)
           scale : float
               scaling of mean
           energy_cal : func
               function that translates channels to energy
           Returns
           -------
           background_estimate : func
               interpolated function describing background
           """

        # some sanity checks for input data
        # check if input is np.array
        try:
            _ = counts.shape
            _cnts = counts[:]
        except AttributeError:
            _cnts = np.array(counts)

        # check for correct shape
        if len(_cnts.shape) != 1:
            raise ValueError('Counts must be 1-dimensional array')

        # check if channels are given
        if channels is None:
            print('Generating array of {} channels'.format(_cnts.shape[0]))
            _chnnls = np.arange(_cnts.shape[0])
        else:
            _chnnls = channels[:]

        # calibrate channels if a calibration is given
        _chnnls = _chnnls if energy_cal is None else energy_cal(_chnnls)

        # get slopes of spectrum
        dy = np.diff(_cnts)

        # initialize variable to calulate moving average along slopes; should be close to 0 for background
        dy_mv_avg = dy

        # list of  bkg estimates for each order
        bkg_estimates = []

        # init variables
        prev_ratio = None
        idx = 0

        # loop and smooth slopes o times
        for o in range(order):
            # move order times over the slopes in order to smooth
            dy_mv_avg = [np.mean(dy_mv_avg[i:i + (window * (order - o))]) for i in range(dy.shape[0])]

            # make mask
            bkg_mask = np.append(np.abs(dy_mv_avg) <= scale * np.mean(np.abs(dy_mv_avg)), np.array([0], dtype=np.bool))

            # interpolate the masked array into array, then create function and append to estimates
            bkg_estimates.append(
                interp1d(_chnnls, np.interp(_chnnls, _chnnls[bkg_mask], _cnts[bkg_mask]), kind='quadratic'))

            # mask wherever signal and bkg are 0
            zero_mask = ~(np.isclose(_cnts, 0) & np.isclose(bkg_estimates[o](_chnnls), 0))


            # make signal to noise ratio of current order
            sn = _cnts[zero_mask] / bkg_estimates[o](_chnnls)[zero_mask]

            # remove parts where signal equals bkg
            sn = sn[~np.isclose(sn, 1)]

            # mask areas which are peaks by interpolation
            peak_mask = sn >= np.mean(sn) + 2 * np.std(sn)

            # make ration of number of peak x values and their sum; gets smaller with every order until peaks are masked
            ratio = sn.shape[0] / np.sum(sn[peak_mask])

            if prev_ratio is None:
                prev_ratio = ratio
            # if peaks are started to get masked break and return
            elif prev_ratio < ratio:
                idx = o - 1
                break
            else:
                prev_ratio = ratio

        # return bkg estimate which has highest signal in smallest number of x vals
        _bkg = bkg_estimates[idx]
        counter = 0

        # counter to break fitting when runtime is too large
        runtime_counter = 0

        # maximum peaks in spectrum that are tried to be foundd
        _MAX_PEAKS = 100

        # result dict
        peaks = OrderedDict()

        # boolean masks
        # masking regions due to failing general conditions (peak_mask)
        # masking successfully fitted regions (peak_mask_fitted)
        peak_mask, peak_mask_fitted = np.ones_like(_cnts, dtype=np.bool), np.ones_like(_cnts, dtype=np.bool)

        # flag whether expected peaks have been checked
        checked_expected = False

        y_find_peaks = _cnts - _bkg(_chnnls)
        if n_peaks is None and expected_peaks is None:
            if not xray:
                expected_peaks = get_isotope_info(isp.gamma_table, info='lines')
                gx_msg = 'Finding isotopes from gamma table file %s...' % isp.gamma_table_file
            else:
                expected_peaks = isp.xray_table['xrays']
                gx_msg = 'Finding x-rays from x-ray table file %s...' % isp.xray_table_file
            tmp_n_peaks = total_n_peaks = len(expected_peaks)


            # make channel sigma iterable if not already iterable
        if not isinstance(ch_sigma, Iterable):
            ch_sigma = [ch_sigma] * total_n_peaks
        else:
            if len(ch_sigma) != total_n_peaks:
                raise IndexError('Not enough channel sigmas for number of peaks')
        peaks = []
        for idx, y in enumerate(y_find_peaks):
            mean =  np.mean(y_find_peaks[idx-15:idx+15])
            if idx<200:
                y=y-20
            if y>mean:
                peaks.append([self.energy_values[idx], y])
        print(peaks)
        return peaks



    def find_peaks(self):
        #histogram = [float(line[1]) for line in self.sample]
        histogram = self.sample
        histogram = np.asarray(histogram)
        #peaks = argrelextrema(histogram, np.greater, order=20)
        mean = np.mean(histogram)
        peaks = histogram - mean
        energy_values = self.get_energy_values()
        for idx,peak in enumerate(peaks):
            peak_energy = [energy_values[idx]]
            peak_energy_path = path.abspath("../res/Na_peaks.csv")
        with open(peak_energy_path, 'w') as f:
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(peak_energy)
        return peak_energy

    def read_peaks(self):
        file_path = path.abspath("../res/peaks.csv")
        with open(file_path, 'r') as file:
            csv_peaks = csv.reader(file)
            peaks = [float(peak_value) for peak in csv_peaks for peak_value in peak]
        return peaks

    def read_sample(self):
        file_path = path.abspath("spectra.csv")
        with open(file_path, 'r') as file:
            sample = csv.reader(file)
            #sample = [line.split() for line in file]
            sample = [float(count_value) for count in sample for count_value in count]
            print(sample)
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


    def check_candidate(self, candidate):
        energies, intensities = self.database_dict[candidate]
        sum_int=sum(intensities)
        zipped_list = zip(intensities, energies)
        sorted_zipped_list = sorted(zipped_list)
        energies = [candidate for _, candidate in sorted_zipped_list]
        intensities = [i for i, candidate in sorted_zipped_list]
        sorted_peaks = [[e, i] for (e, i) in zip(energies, intensities)]
        sorted_peaks = sorted_peaks[:-5]
        print(sorted_peaks)
        count = 0
        inten = 0
        all_peaks = len(sorted_peaks)
        if len(sorted_peaks)!=0:
            sorted_peaks = sorted_peaks[:5]
            for [p, value] in self.peaks:
                for cand_peak, intensity in sorted_peaks:
                    if cand_peak-1 < p <cand_peak+1:
                        count+=1
                        inten += intensity
                        sorted_peaks.remove([cand_peak, intensity])
                        continue
            print('Added intensity for peaks', inten, 'Added sum of intensities',  sum_int)
            return inten



    def identify_nuclide(self):
        all_peaks_energy = self.database_array[:,1].astype(np.float64)
        candidates = []
        intensities = []
        together = []
        for candidate in self.database_dict.keys():
            intensity = self.check_candidate(candidate)
            if intensity is not None:
                intensities.append(intensity)
                candidates.append(candidate)
        zipped_list = zip(intensities, candidates)
        sorted_zipped_list = sorted(zipped_list)
        candidates = [candidate for _, candidate in sorted_zipped_list]
        intensities = [i for i, candidate in sorted_zipped_list]
        print(candidates)
        print(len(candidates), candidates.index('Cs-137'), candidates.index('Co-60'))
        print(intensities)
        print(self.check_candidate('Co-60'))








