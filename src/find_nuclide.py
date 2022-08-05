from os import path
import csv
import numpy as np
import requests
import logging
from scipy.interpolate import interp1d
import warnings
from scipy.signal import argrelmax, argrelextrema
from collections import OrderedDict, Iterable
from scipy.optimize import curve_fit, fsolve, OptimizeWarning
from scipy.integrate import quad
import inspect

def lin(x, a):
    return a * x

def gauss(x, mu, sigma, h):
    return h * np.exp(-0.5 * np.power((x - mu) / sigma, 2.))

class NuclideIdentifier:
    def __init__(self):
        self.sample = self.read_sample()
        self.energy_values = self.get_energy_values()
        #self.peaks = self.find_peaks()
        self.database_array, self.database_dict, self.nuc_dic_irrad, self.nuc_dic_irrad_prob = self.get_database()
        self.peaks = self.interpolate_bkg(self.sample)
        self.identify_nuclide()

    def get_energy_values(self):
        file_path =  "../res/gf_C12137_2.csv"
        energy_values = []
        with open(file_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                energy_values.append(float(row[0]))
        return energy_values

    def gauss_general(self, x, mu, sigma, n, h):
        return h * np.exp(-np.power((x - mu) / sigma, n))

    def get_activity(self, observed_peaks):
        """
        Method to calculate activity isotope-wise. The peak-wise activities of all peaks of
        each isotope are added and scaled with their respectively summed-up probability
        """
        activities = OrderedDict()
        probability_peaks = self.nuc_dic_irrad_prob

        for peak in observed_peaks:
            isotope=peak
            if isotope not in activities:
                activities[isotope] = OrderedDict([('nominal', 0), ('sigma', 0),
                                                   ('probability', 0), ('unscaled', {'nominal': 0, 'sigma': 0})])
                activities[isotope]['unscaled']['nominal'] = 0

            if peak in probability_peaks:
                print('PROBABILITY', probability_peaks[peak])
                activities[isotope]['unscaled']['nominal'] += observed_peaks[peak]['activity']['nominal']
                # squared sum in order to get Gaussian error propagation right
                activities[isotope]['unscaled']['sigma'] += observed_peaks[peak]['activity']['sigma'] ** 2
                activities[isotope]['probability'] += probability_peaks[peak][-1]

        for iso in activities:
            try:
                activities[iso]['nominal'] = activities[iso]['unscaled']['nominal'] * 1. / activities[iso]['probability']
                # square root of sum of squared sigmas in order to get Gaussian error propagation right
                activities[iso]['sigma'] = activities[iso]['unscaled']['sigma'] ** 0.5 * 1. / activities[iso]['probability']

            # when no probability given
            except ZeroDivisionError:
                pass

        return activities

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
        _chnnls = _chnnls

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
            bkg_chnnls = [_chnnls[i] for i in range(len(_chnnls)) if bkg_mask[i]]
            bkg_counts = [_cnts[i] for i in range(len(_cnts)) if bkg_mask[i]]
            bkg_estimates.append(
                interp1d(_chnnls, np.interp(_chnnls, bkg_chnnls, bkg_counts), kind='quadratic'))

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
        t_spec = None
        peak_fit = gauss
        ch_sigma = 5
        energy_range = None
        logging.getLogger().setLevel(logging.DEBUG)
        local_bkg = True
        n_peaks = None
        efficiency_cal = None
        expected_accuracy = 0.02
        reliable = False
        _chnnls = np.asarray(self.get_energy_values())


        # ignore scipy throwing warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=OptimizeWarning)



        # boolean masks
        # masking regions due to failing general conditions (peak_mask)
        # masking successfully fitted regions (peak_mask_fitted)
        peak_mask, peak_mask_fitted = np.ones_like(_cnts, dtype=np.bool), np.ones_like(_cnts, dtype=np.bool)

        # flag whether expected peaks have been checked
        checked_expected = False

        y_find_peaks = _cnts - _bkg(_chnnls)
        expected_peaks = {}
        #BIERZE TYLKO OSTATNI PEAK!!!! NAPRAWIĆ
        for candidate in self.nuc_dic_irrad.keys():
            if self.nuc_dic_irrad_prob[candidate]:
                max_index = np.argmax(self.nuc_dic_irrad_prob[candidate])
                expected_peaks[candidate] = self.nuc_dic_irrad[candidate][max_index]
                print(candidate, expected_peaks[candidate])
        filtered = {k: v for k, v in expected_peaks.items() if v != []}
        expected_peaks.clear()
        expected_peaks.update(filtered)

        tmp_n_peaks = total_n_peaks = len(expected_peaks)

        # make channel sigma iterable if not already iterable
        if not isinstance(ch_sigma, Iterable):
            ch_sigma = [ch_sigma] * total_n_peaks
        else:
            if len(ch_sigma) != total_n_peaks:
                raise IndexError('Not enough channel sigmas for number of peaks')

        # check whether energy range cut should be applied for finding peaks
        if energy_range:
            _e_msg = None
            if energy_cal is None:
                _e_msg = 'No energy calibration provided. Setting an energy range has no effect.'
            else:
                # one range specified
                if isinstance(energy_range, Iterable) and not isinstance(energy_range[0], Iterable):
                    if len(energy_range) == 2:
                        _e_range_mask = (_chnnls <= energy_range[0]) | (_chnnls >= energy_range[1])
                        peak_mask[_e_range_mask] = peak_mask_fitted[_e_range_mask] = False
                    else:
                        _e_msg = 'Energy range {} must contain two elements of lower/upper limit.'.format(energy_range)
                # multiple energy sections specified; invert bool masks and set all specified ranges True
                elif isinstance(energy_range, Iterable) and isinstance(energy_range[0], Iterable):
                    if all(len(er) == 2 for er in energy_range):
                        peak_mask, peak_mask_fitted = ~peak_mask, ~peak_mask_fitted
                        for e_section in energy_range:
                            _e_sec_mask = (e_section[0] <= _chnnls) & (_chnnls <= e_section[1])
                            peak_mask[_e_sec_mask] = peak_mask_fitted[_e_sec_mask] = True
                    else:
                        _e_msg = 'Each element of {} must contain two elements of lower/upper limit.'.format(
                            energy_range)
                else:
                    _e_msg = 'Energy range {} must be an iterable of size two. No range set.'.format(energy_range)
            if _e_msg:
                logging.warning(_e_msg)

        # define tmp fit function of peak: either just gauss or gauss plus background
        def tmp_fit(x, *args):
            return peak_fit(x, *args) if local_bkg else peak_fit(x, *args) + _bkg(x)

        logging.info('Start fitting...')

        # loop over tmp_n_peaks
        while counter < tmp_n_peaks:

            # try to find the expected peaks by going through _MAX_PEAKS peaks in spectrum
            if runtime_counter > _MAX_PEAKS:
                runtime_msg = 'Not all peaks could be found! '
                if expected_peaks is not None:
                    missing_peaks = [str(p) for p in expected_peaks if p in expected_peaks and p not in peaks]
                    if missing_peaks:
                        if len(missing_peaks) <= 10:
                            runtime_msg += ', '.join(missing_peaks) + ' missing! '
                        else:
                            runtime_msg += 'Number of lines not found in spectrum: {}'.format(len(missing_peaks))
                    logging.warning(runtime_msg)

                    # check expected peaks first; then reset and look for another n_peaks
                    if n_peaks is not None:
                        msg = 'Finding additional %i peaks ...' % n_peaks
                        counter = 0
                        runtime_counter = 0
                        tmp_n_peaks = n_peaks
                        checked_expected = True
                        peak_mask = peak_mask_fitted
                        logging.info(msg)
                        continue
                else:
                    print(runtime_msg)
                break

            runtime_counter += 1

            # get peak aka maximum
            try:
                y_peak = np.max(y_find_peaks[peak_mask])
            # this happens for small energy ranges with all entries being masked after a couple fits
            except ValueError:
                msg = 'All data masked. '
                if expected_peaks is not None:
                    missing_peaks = [str(p) for p in expected_peaks if p in expected_peaks and p not in peaks]
                    if missing_peaks:
                        if len(missing_peaks) <= 10:
                            msg += ', '.join(missing_peaks) + ' missing! '
                        else:
                            msg += 'Number of lines not found in spectrum: {}'.format(len(missing_peaks))
                if n_peaks is not None:
                    msg += '' if counter == tmp_n_peaks else '{} of {} peaks found.'.format(counter, tmp_n_peaks)
                logging.info(msg)
                break

            # get corresponding channel numbers: MULTIPLE X FOR SAME Y POSSIBLE BUT RARE! NEED TO LOOP
            x_peaks = np.where(y_peak == y_find_peaks)[0]
            energy_values = self.get_energy_values()

            # loop over possibly multiple x peak positions
            for x_peak in x_peaks:

                # make fit environment; fit around x_peak +- some ch_sigma
                # make tmp_peak for whether we're fitting channels or already calibrated channels
                tmp_peak = np.where(np.isclose(_chnnls,energy_values[x_peak]))[0][0]
                low = tmp_peak - ch_sigma[counter] if tmp_peak - ch_sigma[counter] > 0 else 0
                high = tmp_peak + ch_sigma[counter] if tmp_peak + ch_sigma[counter] < len(_chnnls) else len(_chnnls) - 1

                # make fit regions in x and y; a little confusing to look at but we need the double indexing to
                # obtain the same shapes
                #x_fit = [_chnnls[i] for i in range(low, high+1) if peak_mask[i]]
                #y_fit = [_cnts[i] for i in range(low, high+1) if peak_mask[i]]
                x_fit, y_fit = np.asarray(_chnnls)[low:high][peak_mask[low:high]], np.asarray(_cnts)[low:high][peak_mask[low:high]]

                # check whether we have enough points to fit to
                if len(x_fit) < 5:  # skip less than 5 data points
                    logging.debug('Only %i data points in fit region. Skipping' % len(x_fit))
                    peak_mask[low:high] = False
                    continue

                # start fitting
                try:
                    # estimate starting parameters
                    _mu = energy_values[x_peak]
                    print(x_peak, _mu)
                    _sigma = 0
                    k = 2.0
                    while _sigma == 0 and k <= 10:
                        try:
                            sigma_x_fit = [x_fit[i] for i in range(len(x_fit)) if y_fit[i] >= y_peak / k]
                            _sigma = np.abs(np.asarray(x_fit)[y_fit >= y_peak / k][-1] - np.asarray(x_fit)[y_fit >= y_peak / k][0]) / 2.3548
                        except IndexError:
                            pass
                        finally:
                            k += .5
                    _p0 = {'mu': _mu, 'sigma': _sigma, 'h': y_peak}
                    fit_args = inspect.getargspec(peak_fit)[0][1:]
                    p0 = tuple(_p0[arg] if arg in _p0 else 1 for arg in fit_args)
                    popt, pcov = curve_fit(tmp_fit, x_fit, y_fit, p0=p0, sigma=np.sqrt(y_fit), absolute_sigma=True,
                                           maxfev=5000)
                    perr = np.sqrt(np.diag(pcov))  # get std deviation

                    # update
                    _mu, _sigma = [popt[fit_args.index(par)] for par in ('mu', 'sigma')]

                    # if fitting resulted in nan errors
                    if any(np.isnan(perr)):
                        peak_mask[low:high] = False
                        print('Fitted resulted in nan errors')
                        continue

                    """if any(_mu == peaks[p]['peak_fit']['popt'][0] for p in peaks):
                        logging.debug('Peak at %.2f already fitted. Skipping' % _mu)
                        peak_mask[low:high] = False
                        continue"""

                    # if fit is unreliable
                    if any(np.abs(perr / popt) > 1.0):
                        print(perr/popt)
                        if not reliable:
                            logging.warning(
                                'Unreliable fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                        else:
                            print('UNCERTAINTY FOR PEAK', _mu, perr, popt)
                            logging.debug('Skipping fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                            peak_mask[low:high] = False
                            continue

                    #if fit is indistinguishable from background
                    try:
                        _msk = ((_mu - 8 * _sigma <= _chnnls) & (_chnnls <= _mu - 6 * _sigma)) | \
                               ((_mu + 6 * _sigma <= _chnnls) & (_chnnls <= _mu + 8 * _sigma))
                        _msk[~peak_mask_fitted] = False
                        if np.max(_cnts[_msk]) > popt[fit_args.index('h')]:
                            print(np.max(_cnts[_msk]))
                            print(popt[fit_args.index('h')])
                            logging.debug('Peak at %.2f indistinguishable from background. Skipping' % _mu)
                            print('Peak at %.2f indistinguishable from background. Skipping' % _mu)
                            raise ValueError
                    except ValueError:
                        peak_mask[low:high] = False
                        continue

                # fitting did not succeed
                except RuntimeError:  # disable failed region for next iteration
                    logging.debug('Fitting failed. Skipping peak at %.2f' % _mu)
                    peak_mask[low:high] = False
                    continue

                # check if our fitted peak is expected
                if expected_peaks is not None and not checked_expected:

                    # make list for potential candidates
                    candidates = []
                    predictions = []

                    # loop over all expected peaks and check which check out as expected within the accuracy
                    for ep in expected_peaks:


                        # get upper and lower estimates
                        lower_est, upper_est = [(1 + sgn * expected_accuracy) * expected_peaks[ep] for sgn in (-1, 1)]
                        if ep == 'Cs-137':
                            print(ep, 'LOWER MU UPPER', lower_est, _mu, upper_est)
                        if ep == 'Co-60':
                            print(ep, 'LOWER MU UPPER', lower_est, _mu, upper_est)
                        # if current peak checks out set peak name and break
                        if lower_est <= _mu <= upper_est:
                            candidates.append(ep)
                            predictions.append([ep, expected_peaks[ep], _mu])

                    # if no candidates are found, current peak was not expected
                    if not candidates:
                        logging.debug('Peak at %.2f not expected. Skipping' % _mu)
                        peak_mask[low:high] = False
                        continue

                    # if all candidates are already found
                    if all(c in peaks for c in candidates):
                        logging.debug('Peak at %.2f already fitted. Skipping' % _mu)
                        peak_mask[x_peak] = False
                        continue
                else:
                    candidates = ['peak_%i' % counter]

                ### FROM HERE ON THE FITTED PEAK WILL BE IN THE RESULT DICT ###

                # get integration limits within 3 sigma for non local background
                low_lim, high_lim = _mu - 3 * _sigma, _mu + 3 * _sigma  # integrate within 3 sigma

                # get background via integration of background model
                if not local_bkg:
                    background = quad(_bkg, low_lim, high_lim)  # background integration

                # get local background and update limits
                else:
                    # find local background bounds; start looking at bkg from (+-3 to +-6) sigma
                    # increase bkg to left/right of peak until to avoid influence of nearby peaks
                    _i_dev = 6
                    _deviation = None
                    while _i_dev < int(_MAX_PEAKS / 2):
                        # Make tmp array of mean bkg values left and right of peak
                        _tmp_dev_array = [
                            np.mean(_cnts[(_mu - _i_dev * _sigma <= _chnnls) & (_chnnls <= _mu - 3 * _sigma)]),
                            np.mean(_cnts[(_mu + 3 * _sigma <= _chnnls) & (_chnnls <= _mu + _i_dev * _sigma)])]
                        # look at std. deviation; as long as it decreases for increasing bkg area update
                        if _deviation is None or np.std(_tmp_dev_array) < _deviation:
                            _deviation = np.std(_tmp_dev_array)
                        # if std. deviation increases again, break
                        elif np.std(_tmp_dev_array) >= _deviation:
                            _i_dev -= 1
                            break
                        # increment
                        _i_dev += 1

                    # get background from 3 to _i_dev sigma left of peak
                    lower_bkg = np.logical_and(_mu - _i_dev * _sigma <= _chnnls, _chnnls <= _mu - 3 * _sigma)
                    # get background from 3 to _i_dev sigma right of peak
                    upper_bkg = np.logical_and(_mu + 3 * _sigma <= _chnnls, _chnnls <= _mu + _i_dev * _sigma)
                    # combine bool mask
                    bkg_mask = np.logical_or(lower_bkg, upper_bkg)
                    # mask other peaks in bkg so local background fit will not be influenced by nearby peak
                    bkg_mask[~peak_mask] = False
                    # do fit; make sure we don't mask at least 2 points
                    bkg_chnnls = [_chnnls[i] for i in range(len(_chnnls)) if bkg_mask[i]]
                    bkg_counts = [_cnts[i] for i in range(len(_cnts)) if bkg_mask[i]]
                    if np.count_nonzero(bkg_mask) > 1:
                        bkg_opt, bkg_cov = curve_fit(lin, bkg_chnnls, bkg_counts)
                        bkg_interp = False
                    # if we do use interpolated bkg
                    else:
                        mask = (lower_bkg | upper_bkg)
                        print(lower_bkg, upper_bkg)
                        _chnnls_local = [_chnnls[i] for i in range(len(_chnnls)) if mask[i]]
                        _chnnls_local = np.asarray(_chnnls)[(lower_bkg | upper_bkg)]
                        #bkg_opt, bkg_cov = curve_fit(lin, _chnnls[(lower_bkg | upper_bkg)],
                                                     #_bkg(_chnnls[(lower_bkg | upper_bkg)]))
                        bkg_opt, bkg_cov = curve_fit(lin, np.asarray(_chnnls)[(lower_bkg | upper_bkg)], _bkg(np.asarray(_chnnls)[(lower_bkg | upper_bkg)]))
                        bkg_interp = True
                    # find intersections of line and gauss; should be in 3-sigma environment since background is not 0
                    # increase environment to 5 sigma to be sure
                    low_lim, high_lim = _mu - 5 * _sigma, _mu + 5 * _sigma
                    # _chnnls values of current peak
                    """_peak_chnnls = [_chnnl for _chnnl in _chnnls if (low_lim <= _chnnl) & (_chnnl <= high_lim)]
                    tmp = []
                    for i in range(len(_chnnls)):
                        if (low_lim <= _chnnls[i]) & (_chnnls[i] <= high_lim):
                            tmp.append(_cnts[i])
                    _peak_cnts = tmp"""
                    _peak_chnnls = np.asarray(_chnnls)[(low_lim <= _chnnls) & (_chnnls <= high_lim)]
                    _peak_cnts = np.asarray(_cnts)[(low_lim <= _chnnls) & (_chnnls <= high_lim)]
                    # fsolve heavily relies on correct start parameters; estimate from data and loop
                    try:
                        _i_tries = 0
                        found = False
                        while _i_tries < _MAX_PEAKS:
                            diff = np.abs(high_lim - low_lim) / _MAX_PEAKS * _i_tries

                            _x0_low = low_lim + diff / 2.
                            _x0_high = high_lim - diff / 2.

                            # find intersections; needs to be sorted since sometimes higher intersection is found first
                            _low, _high = sorted(
                                fsolve(lambda k: tmp_fit(k, *popt) - lin(k, *bkg_opt), x0=np.asarray([_x0_low, _x0_high])))

                            # if intersections have been found
                            if not np.isclose(_low, _high) and np.abs(_high - _low) <= 7 * _sigma:
                                low_lim, high_lim = _low, _high
                                found = True
                                break

                            # increment
                            _i_tries += 1

                        # raise error
                        if not found:
                            raise ValueError

                    except (TypeError, ValueError):
                        msg = 'Intersections between peak and local background for peak(s) %s could not be found.' % ', '.join(
                            candidates)
                        if bkg_interp:
                            msg += ' Use estimates from interpolated background instead.'
                            _y_low, _y_high = _bkg(_chnnls[lower_bkg]), _bkg(_chnnls[upper_bkg])
                        else:
                            msg += ' Use estimates from data instead.'
                            _y_low, _y_high = _cnts[lower_bkg], _cnts[upper_bkg]

                        # estimate intersections of background and peak from data
                        low_lim = _peak_chnnls[np.where(_peak_cnts >= np.mean(_y_low))[0][0]]
                        high_lim = _peak_chnnls[np.where(_peak_cnts >= np.mean(_y_high))[0][-1]]
                        logging.info(msg)


                # scale activity to compensate for dectector inefficiency at given energy
                if efficiency_cal is not None:
                    activity, activity_err = (efficiency_cal(popt[0]) * x for x in [activity, activity_err])

                # normalize to counts / s == Bq
                if t_spec is not None:
                    activity, activity_err = activity / t_spec, activity_err / t_spec
                probability_peaks = self.nuc_dic_irrad_prob

                # write current results to dict for every candidate
                for peak_name in candidates:
                    print(peak_name)
                    # make entry for current peak
                    peaks[peak_name] = popt.tolist()[0], expected_peaks[peak_name], max(probability_peaks[peak_name])


                    # entries for data

                    # write background to result dict


                    counter += 1  # increase counter

                runtime_counter = 0  # peak(s) were found; reset runtime counter

                # disable fitted region for next iteration
                #current_mask = (low_lim <= _chnnls) & (_chnnls <= high_lim)
                current_mask = [True if (low_lim <= _chnnls[i]) & (_chnnls[i] <= high_lim) else False for i in range(len(_chnnls))]
                peak_mask[current_mask] = peak_mask_fitted[current_mask] = False

       #activities = self.get_activity(peaks)

        #peaks[peak_name]['activity']= activities[peak_name]

        print(peaks)
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
        nuc_dic_irrad = {}
        nuc_dic_irrad_prob = {}
        for nuclide in database:
            energy_peaks = []
            intensity = []
            for peaks in database[nuclide]:
                energy_peaks.append(peaks['energy'])
                intensity.append(peaks['intensity'])
                new_database.append([nuclide, peaks['energy'], peaks['intensity']])
            nuclides_dict[nuclide] = [energy_peaks, intensity]
            nuc_dic_irrad[nuclide] = energy_peaks
            nuc_dic_irrad_prob[nuclide] = intensity
        new_database = np.array(new_database)
        return new_database, nuclides_dict, nuc_dic_irrad, nuc_dic_irrad_prob

    def get_database(self):
        x = requests.get('http://25.20.235.8:8000/lhnb/list')
        data = x.json()
        nuclides_peaks = {}
        for nuclide in data['available nuclides']:
            r = requests.get(f'http://25.20.235.8:8000/lhnb/nuclide/{nuclide}')
            nuclide_info = r.json()
            nuclides_peaks[nuclide] = nuclide_info['spectra_peaks']
        new_database_array, new_database_dict, nuc_dic_irrad, nuc_dic_irrad_prob = self.reform_database(nuclides_peaks)
        return new_database_array, new_database_dict, nuc_dic_irrad, nuc_dic_irrad_prob


    def check_candidate(self, candidate):
        energies, intensities = self.database_dict[candidate]
        sum_int=sum(intensities)
        zipped_list = zip(intensities, energies)
        sorted_zipped_list = sorted(zipped_list)
        energies = [candidate for _, candidate in sorted_zipped_list]
        intensities = [i for i, candidate in sorted_zipped_list]
        sorted_peaks = [[e, i] for (e, i) in zip(energies, intensities)]
        sorted_peaks = sorted_peaks[:-5]
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
            #print('Added intensity for peaks', inten, 'Added sum of intensities',  sum_int)
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








