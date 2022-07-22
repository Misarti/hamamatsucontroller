import logging
import json
import inspect
import numpy as np
from collections import OrderedDict, Iterable
from scipy.optimize import curve_fit, fsolve, OptimizeWarning
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path


class Spectrometry:
    def __init__(self):
        path = Path(__file__).parent.absolute()
        with open(path/  'gamma_table.json') as json_file:
            self.gamma_table = json.load(json_file)

        with open(path / "xray_coefficient_table.json") as json_file:
            self.xray_coefficient_table = json.load(json_file)

        with open(path /"xray_table.json") as json_file:
            self.xray_table = json.load(json_file)


        self.n_channels = 4095


    def decay_constant(self, half_life):
        return np.log(2.) / half_life

    def decay_law(self, t, x0, half_life):
        return x0 * np.exp(-self.decay_constant(half_life) * t)

    def lin(self, x, a, b):
        return (a * x + b)

    def gauss(self, x, mu, sigma, h):
        return h * np.exp(-0.5 * np.power((x - mu) / sigma, 2.))


    def get_isotope_info(self, filename='gamma_table.yaml', info='lines', iso_filter=None):
        """
        Method to return dict of isotope info from gamma table. Info can either be 'lines', 'probability', 'half_life',
        'decay_mode', 'name', 'A', or 'Z'. Keys of result dict are element symbols.

        Parameters
        ----------
            gamma table of isotopes with additional info. Default is isp.gamma_table
        info : str
            information which is needed. Default is 'lines' which corresponds to gamma energies. Can be either of the ones
            listed above
        iso_filter : str
            string of certain isotope whichs info you want to filter e.g. '65_Zn' or '65' or 'Zn'
        """

        if 'isotopes' not in self.gamma_table:
            raise ValueError('Gamma table must contain isotopes.')
        else:
            isotopes = self.gamma_table['isotopes']

            # init result dict and loop over different isotopes
            result = {}
            for symbol in isotopes:
                if info in isotopes[symbol]:
                    if not isinstance(isotopes[symbol][info], dict):
                        result[symbol] = isotopes[symbol][info]
                    else:
                        mass_nums = isotopes[symbol][info].keys()
                        result[symbol] = mass_nums if len(mass_nums) > 1 else mass_nums[0]

                else:
                    mass_number = isotopes[symbol]['A']
                    for A in mass_number:
                        identifier = '%s_%s' % (str(A), str(symbol))
                        if info in mass_number[A]:
                            if isinstance(mass_number[A][info], list):
                                for i, n in enumerate(mass_number[A][info]):
                                    result[identifier + '_%i' % i] = n
                            else:
                                result[identifier] = mass_number[A][info]

            if not result:
                raise ValueError('Gamma table does not contain info %s.' % info)

            if iso_filter:
                sortout = [k for k in result if iso_filter not in k]
                for s in sortout:
                    del result[s]

            return result


    def source_to_dict(self, source, info='lines'):
        """
        Method to convert a source dict to a dict containing isotope keys and info.
        """

        reqs = ('A', 'symbol', info)
        if not all(req in source for req in reqs):
            raise ValueError('Missing reuqired data in source dict: %s' % ', '.join(req for req in reqs if req not in source))
        return dict(('%i_%s_%i' % (source['A'], source['symbol'], i) , l) for i, l in enumerate(source[info]))

    def activity(self, n0, half_life):
        return self.decay_constant(half_life) * n0

    def mean_lifetime(self, half_life):
        return 1. / self.decay_constant(half_life)


    def gamma_dose_rate(self, energy, probability, activity, distance, material='air'):
        """
        Calculation of the per-gamma dose rate in air according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

        Parameters
        ----------
        energy : float
            gamma energy
        probability : float from 0 to 1
            probability of emitting this gamma per disintegration
        activity : float
            disintegrations per second (Bq)
        distance : float
            distance in cm from gamma source the dose rate should be calculated at
        material : str
            string of material the dose is to be calculated in. Must be key in xray_coeffs

        Returns
        -------

        dose_rate: float
            dose rate from gamma in uSv/h
        """


        if material not in self.xray_coefficient_table['material'].keys():
            msg = 'No x-Ray coefficient table for material "{}". Please add table to xray_coefficient_table.json.json '.format(material)
            raise KeyError(msg)

        # load values for energy-absorption coefficients from package
        xray_energies = np.array(self.xray_coefficient_table['material'][material]['energy'])
        xray_en_absorption = np.array(self.xray_coefficient_table['material'][material]['energy_absorption'])

        # factor for conversion of intermedate result to uSv/h
        # 1st: link above; 2nd: Roentgen to Sievert; 3rd: combination of Sv to uSv and keV to MeV
        custom_factor = 5.263e-6 * 1. / 107.185 * 1e3

        # find energy-absorption coefficient from coefficients file through linear interpolation
        idx = np.where(xray_energies <= energy)[0][-1]

        if idx == len(xray_en_absorption) - 1:
            msg = '{} keV larger than largest energy in x-Ray coefficient table.' \
                  ' Taking coefficient of largest energy available ({} keV) instead'.format(energy, xray_energies[-1])
            logging.warning(msg)
            tmp_xray_en_ab_interp = xray_en_absorption[-1]
        else:
            tmp_xray_en_ab_interp = np.interp(energy, xray_energies[idx:idx+2], xray_en_absorption[idx:idx+2])

        return custom_factor * energy * probability * tmp_xray_en_ab_interp * activity / distance**2.

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
            logging.info('Generating array of {} channels'.format(_cnts.shape[0]))
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
            bkg_estimates.append(interp1d(_chnnls, np.interp(_chnnls, _chnnls[bkg_mask], _cnts[bkg_mask]), kind='quadratic'))

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
        return bkg_estimates[idx]


    def fit_spectrum(self, counts, channels=None, bkg=None, local_bkg=True, n_peaks=None, ch_sigma=5, energy_cal=None, efficiency_cal=None, t_spec=None, expected_peaks=None, expected_accuracy=5e-3, peak_fit=gauss, energy_range=None, reliable=True, xray=False):
        """
        Method that identifies the first n_peaks peaks in a spectrum. They are identified in descending order from highest
        to lowest peak. A Gaussian is fitted to each peak within a fit region of +- ch_sigma of its peak center.
        If a calibration is provided, the channels are translated into energies. The integral of each peak within its fitting
        range is calculated as well as the error. If a dict of expected peaks is given, only these peaks are looked for
        within a accuracy of expected_accuracy.

        Parameters
        ----------

        counts : array
            array of counts
        channels : array or None
            array of channels; if None will be np.arange(len(counts))
        bkg : func
            function describing background; if local_background is True, background is only needed to find peaks and not for activity calculation
        local_bkg: bool
            if True, every peaks background will be determined by a linear fit right below the peak
        n_peaks : int
            number of peaks to identify
        ch_sigma : int
            defines fit region around center of peak in channels
        energy_cal : func
            calibration function that translates channels to energy
        efficiency_cal : func
            calibration function that scales activity as function of energy (compensation of detector inefficiencies)
        t_spec : float
            integrated time of measured spectrum y in seconds
        expected_peaks : dict
            dict of expected peaks with names (e.g. 40K_1 for 1 kalium peak) as keys and values either in channels or energies
        expected_accuracy : float from 0 to 1
            accuracy with which the expected peak has to be found
        peak_fit : func
            function to fit peaks in spectrum with. This is only fitted to peaks
        energy_range : iterable or iterable of iterables
            iterable (of iterables) with two values (per element) one for lower and upper limit. Is only applied when energy calibration is provided.
        reliable: bool
            whether to accept peaks with unreliable fits
        xray : bool
            whether or not the spectrum is pure xray, If False, spectrum is assumed to be gammas & xray from radioactive decays
            which causes validation of identified isotopes. Default is False

        Returns
        -------

        peaks : dict
            dictionary with fit parameters of gauss as well as errors and integrated counts (signal) of each peak with error

        _bkg : func, optional
            if bkg is None, bkg is interpolated and also returned
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
            logging.info('Generating array of {} channels'.format(_cnts.shape[0]))
            _chnnls = np.arange(_cnts.shape[0])
        else:
            _chnnls = channels[:]

        # calibrate channels if a calibration is given
        #TUTAJ ZAMIAST KANAŁÓW DAĆ ENERGIĘ (self.log_list)???
        _chnnls = _chnnls if energy_cal is None else energy_cal(_chnnls)

        # peak counter
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

        # make background model if None
        if bkg is None:
            logging.info('Interpolating background...')
            _bkg = self.interpolate_bkg(channels=_chnnls, counts=_cnts)
        else:
            _bkg = bkg

        # correct y by background to find peaks
        y_find_peaks = _cnts - _bkg(_chnnls)

        # make tmp variable for n_peaks in order to not change input
        if n_peaks is None and expected_peaks is None:
            if not xray:
                expected_peaks = self.get_isotope_info(self.gamma_table, info='lines')
                gx_msg = 'Finding isotopes from gamma table file..'
            else:
                expected_peaks = self.xray_table['xrays']
                gx_msg = 'Finding x-rays from x-ray table file ...'
            tmp_n_peaks = total_n_peaks = len(expected_peaks)
            logging.info(gx_msg)
        else:
            # expected peaks are checked first
            tmp_n_peaks = n_peaks if expected_peaks is None else len(expected_peaks)

            # total peaks which are checked
            total_n_peaks = n_peaks if expected_peaks is None else len(expected_peaks) if n_peaks is None else n_peaks + len(expected_peaks)

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
                        _e_msg = 'Each element of {} must contain two elements of lower/upper limit.'.format(energy_range)
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
                    logging.warning(runtime_msg)
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

            # loop over possibly multiple x peak positions
            for x_peak in x_peaks:

                # make fit environment; fit around x_peak +- some ch_sigma
                # make tmp_peak for whether we're fitting channels or already calibrated channels
                tmp_peak = x_peak if energy_cal is None else np.where(_chnnls == energy_cal(x_peak))[0][0]
                low = tmp_peak - ch_sigma[counter] if tmp_peak - ch_sigma[counter] > 0 else 0
                high = tmp_peak + ch_sigma[counter] if tmp_peak + ch_sigma[counter] < len(_chnnls) else len(_chnnls) - 1

                # make fit regions in x and y; a little confusing to look at but we need the double indexing to
                # obtain the same shapes
                x_fit, y_fit = _chnnls[low:high][peak_mask[low:high]], _cnts[low:high][peak_mask[low:high]]

                # check whether we have enough points to fit to
                if len(x_fit) < 5:  # skip less than 5 data points
                    logging.debug('Only %i data points in fit region. Skipping' % len(x_fit))
                    peak_mask[low:high] = False
                    continue

                # start fitting
                try:
                    # estimate starting parameters
                    _mu = x_peak if energy_cal is None else energy_cal(x_peak)
                    _sigma = 0
                    k = 2.0
                    while _sigma == 0 and k <= 10:
                        try:
                            _sigma = np.abs(x_fit[y_fit >= y_peak / k][-1] - x_fit[y_fit >= y_peak / k][0]) / 2.3548
                        except IndexError:
                            pass
                        finally:
                            k += .5
                    _p0 = {'mu': _mu, 'sigma': _sigma, 'h': y_peak}
                    fit_args = inspect.getfullargspec(peak_fit)[0][1:]
                    p0 = tuple(_p0[arg] if arg in _p0 else 1 for arg in fit_args)
                    popt, pcov = curve_fit(tmp_fit, x_fit.astype(np.float64), y_fit.astype(np.float64), p0=p0, sigma=np.sqrt(y_fit), absolute_sigma=True, maxfev=5000)
                    perr = np.sqrt(np.diag(pcov))  # get std deviation

                    # update
                    _mu, _sigma = [popt[fit_args.index(par)] for par in ('mu', 'sigma')]

                    # if fitting resulted in nan errors
                    if any(np.isnan(perr)):
                        peak_mask[low:high] = False
                        continue

                    if any(_mu == peaks[p]['peak_fit']['popt'][0] for p in peaks):
                        logging.debug('Peak at %.2f already fitted. Skipping' % _mu)
                        peak_mask[low:high] = False
                        continue

                    # if fit is unreliable
                    if any(np.abs(perr / popt) > 1.0):
                        if not reliable:
                            logging.warning('Unreliable fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                        else:
                            logging.debug('Skipping fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                            peak_mask[low:high] = False
                            continue

                    # if fit is indistinguishable from background
                    try:
                        _msk = ((_mu - 6 * _sigma <= _chnnls) & (_chnnls <= _mu - 3 * _sigma)) | \
                               ((_mu + 3 * _sigma <= _chnnls) & (_chnnls <= _mu + 6 * _sigma))
                        _msk[~peak_mask_fitted] = False
                        if np.max(_cnts[_msk]) > popt[fit_args.index('h')]:
                            logging.debug('Peak at %.2f indistinguishable from background. Skipping' % _mu)
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

                    # loop over all expected peaks and check which check out as expected within the accuracy
                    for ep in expected_peaks:

                        # get upper and lower estimates
                        lower_est, upper_est = [(1 + sgn * expected_accuracy) * expected_peaks[ep] for sgn in (-1, 1)]

                        # if current peak checks out set peak name and break
                        if lower_est <= _mu <= upper_est:
                            candidates.append(ep)

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
                if not local_bkg :
                    background, background_err = quad(_bkg, low_lim, high_lim)  # background integration

                # get local background and update limits
                else:
                    # find local background bounds; start looking at bkg from (+-3 to +-6) sigma
                    # increase bkg to left/right of peak until to avoid influence of nearby peaks
                    _i_dev = 6
                    _deviation = None
                    while _i_dev < int(_MAX_PEAKS / 2):
                        # Make tmp array of mean bkg values left and right of peak
                        _tmp_dev_array = [np.mean(_cnts[(_mu - _i_dev * _sigma <= _chnnls) & (_chnnls <= _mu - 3 * _sigma)]),
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
                    if np.count_nonzero(bkg_mask) > 1:
                        bkg_opt, bkg_cov = curve_fit(self.lin, _chnnls[bkg_mask], _cnts[bkg_mask])
                        bkg_interp = False
                    # if we do use interpolated bkg
                    else:
                        bkg_opt, bkg_cov = curve_fit(self.lin, _chnnls[(lower_bkg | upper_bkg)], _bkg(_chnnls[(lower_bkg | upper_bkg)]))
                        bkg_interp = True
                    # find intersections of line and gauss; should be in 3-sigma environment since background is not 0
                    # increase environment to 5 sigma to be sure
                    low_lim, high_lim = _mu - 5 * _sigma, _mu + 5 * _sigma
                    # _chnnls values of current peak
                    _peak_chnnls = _chnnls[(low_lim <= _chnnls) & (_chnnls <= high_lim)]
                    _peak_cnts  =  _cnts[(low_lim <= _chnnls) & (_chnnls <= high_lim)]
                    # fsolve heavily relies on correct start parameters; estimate from data and loop
                    try:
                        _i_tries = 0
                        found = False
                        while _i_tries < _MAX_PEAKS:
                            diff = np.abs(high_lim - low_lim) / _MAX_PEAKS * _i_tries

                            _x0_low = low_lim + diff / 2.
                            _x0_high = high_lim - diff / 2.

                            # find intersections; needs to be sorted since sometimes higher intersection is found first
                            _low, _high = sorted(fsolve(lambda k: tmp_fit(k, *popt) - self.lin(k, *bkg_opt), x0=[_x0_low, _x0_high]))

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
                        msg = 'Intersections between peak and local background for peak(s) %s could not be found.' % ', '.join(candidates)
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

                    # do background integration
                    background, background_err = quad(self.lin, low_lim, high_lim, args=tuple(bkg_opt))

                # get counts via integration of fit
                counts = quad(tmp_fit, low_lim, high_lim, args=tuple(popt))  # count integration

                # estimate lower uncertainty limit
                counts_low = quad(tmp_fit, low_lim, high_lim, args=tuple(popt - perr))  # lower counts limit

                # estimate lower uncertainty limit
                counts_high = quad(tmp_fit, low_lim, high_lim, args=tuple(popt + perr))  # lower counts limit

                low_count_err, high_count_err = np.abs(counts - counts_low), np.abs(counts_high - counts)

                max_count_err = high_count_err if high_count_err >= low_count_err else low_count_err

                # calc activity and error
                activity, activity_err = counts - background, np.sqrt(np.power(max_count_err, 2.) + np.power(background_err, 2.))

                # scale activity to compensate for dectector inefficiency at given energy
                if efficiency_cal is not None:
                    activity, activity_err = (efficiency_cal(popt[0]) * x for x in [activity, activity_err])

                # normalize to counts / s == Bq
                if t_spec is not None:
                    activity, activity_err = activity / t_spec, activity_err / t_spec

                # write current results to dict for every candidate
                for peak_name in candidates:
                    # make entry for current peak
                    peaks[peak_name] = OrderedDict()

                    # entries for data
                    peaks[peak_name]['background'] = OrderedDict()
                    peaks[peak_name]['peak_fit'] = OrderedDict()
                    peaks[peak_name]['activity'] = OrderedDict()

                    # write background to result dict
                    if local_bkg:
                        peaks[peak_name]['background']['popt'] = bkg_opt.tolist()
                        peaks[peak_name]['background']['perr'] = np.sqrt(np.diag(bkg_cov)).tolist()
                    peaks[peak_name]['background']['type'] = 'local' if local_bkg else 'global'

                    # write optimal fit parameters/errors for every peak to result dict
                    peaks[peak_name]['peak_fit']['popt'] = popt.tolist()
                    peaks[peak_name]['peak_fit']['perr'] = perr.tolist()
                    peaks[peak_name]['peak_fit']['int_lims'] = [float(low_lim), float(high_lim)]
                    peaks[peak_name]['peak_fit']['type'] = peak_fit.__name__

                    # write activity data to output dict
                    peaks[peak_name]['activity']['nominal'] = float(activity)
                    peaks[peak_name]['activity']['sigma'] = float(activity_err)
                    peaks[peak_name]['activity']['type'] = 'integrated' if t_spec is None else 'normalized'
                    peaks[peak_name]['activity']['unit'] = 'becquerel' if t_spec is not None else 'counts / t_spec'
                    peaks[peak_name]['activity']['calibrated'] = efficiency_cal is not None

                    counter += 1  # increase counter

                runtime_counter = 0  # peak(s) were found; reset runtime counter

                # disable fitted region for next iteration
                current_mask = (low_lim <= _chnnls) & (_chnnls <= high_lim)
                peak_mask[current_mask] = peak_mask_fitted[current_mask] = False

            # check whether we have found all expected peaks and there's still n_peaks to look after
            if counter == tmp_n_peaks and expected_peaks is not None:
                msg = 'All %s have been found!' % ('expected peaks' if not checked_expected else 'peaks')
                # expected peaks are all found if they're not None
                if n_peaks is not None and not checked_expected:
                    msg += 'Finding additional %i peaks ...' % n_peaks
                    counter = 0
                    runtime_counter = 0
                    tmp_n_peaks = n_peaks
                    checked_expected = True
                    peak_mask = peak_mask_fitted
                logging.info(msg)

        logging.info('Finished fitting.')

        # identify wrongly assigned isotopes and remove
        if not xray:
            if reliable:
                logging.info('Validate identified isotopes...')
                self.validate_isotopes(peaks=peaks, table=self.gamma_table, e_max=_chnnls[-1] if energy_cal is not None else None)
            else:
                msg = 'Isotopes not validated. Set "reliable=True" to validate!'
                logging.warning(msg)

        return peaks if bkg is not None else (peaks, _bkg)

    def validate_isotopes(self, peaks, table, e_max=None):
        """
        Methods that validates identified isotopes from fit_spectrum. It uses the gamma table in order to perform
        the following check: For each isotope in peaks, the respective line with the lowest probability per isotope is
        taken as start parameter. All remaining lines of the same isotope with higher probability must be in the spectrum
        as well.

        Parameters
        ----------

        peaks : dict
            return value of irrad_spectroscopy.fit_spectrum
        table : dict
            gamma table dictionary loaded from gamma_table.yaml
        e_max : float
            maximum energy of spectrum

        """

        table = self.gamma_table

        # get unique isotopes in sample
        isotopes_in_sample = set('_'.join(p.split('_')[:-1]) for p in peaks)

        # make list for logging info
        not_in_table = []
        removed = []

        # loop over all unique isotopes
        for isotope in isotopes_in_sample:
            # get all lines' probabilities of isotope from gamma table
            current_in_table = self.get_isotope_info(self.gamma_table, info='probability', iso_filter=isotope)

            # if current isotope is not in table
            if not current_in_table:
                not_in_table.append(isotope)
                continue

            # exclude out-of-range gamma lines from validation procedure
            if e_max is not None:
                # get all lines of isotope from gamma table
                tmp_e = self.get_isotope_info(self.gamma_table, info='lines', iso_filter=isotope)
                # find keys for which line not in energy range
                lines_not_in_e_range = [k for k in tmp_e if tmp_e[k] > e_max]
                # remove lines from current in table
                for l in lines_not_in_e_range:
                    del current_in_table[l]

            # get all lines' probabilities of current isotope in sample
            current_in_sample = dict((peak, current_in_table[peak]) for peak in peaks if isotope in peak)

            # sort both
            current_props_table, current_props_sample = sorted(current_in_table.values()), sorted(
                current_in_sample.values())

            # get index of lowest prob line in probs from table as start and len as end
            start_index = current_props_table.index(current_props_sample[0])

            # update table lines of current isotope to start from lowest prob line in sample
            current_props_table = current_props_table[start_index:]

            # sanity checks follow, each in distinct if/elif statement

            # now both lists should be equal if all higher probability lines are present; if not remove isotope
            if current_props_table != current_props_sample:
                for cis in current_in_sample:
                    removed.append(cis)
                    del peaks[cis]
                logging.info(
                    'Removed %s due to %i missing lines!' % (isotope, len(current_in_table) - len(current_in_sample)))

            # if peaks are scaled for efficiency the activities must increase
            elif all(peaks[pn]['activity']['calibrated'] for pn in current_in_sample):
                # helper funcs and vars
                _tmp = [None] * len(current_in_sample)
                for k in current_in_sample.keys():
                    _tmp[int(k.split('_')[-1])] = k
                _f = lambda x: peaks[x]['activity']['nominal']
                _g = lambda y: peaks[y]['activity']['sigma']
                _bool_arr = [_f(_tmp[j]) + _g(_tmp[j]) >= _f(_tmp[j + 1]) - _g(_tmp[j + 1]) for j in
                             range(len(_tmp) - 1)]

                if _bool_arr and np.count_nonzero(_bool_arr) / (1.0 * len(_bool_arr)) < 0.75:
                    for cis in current_in_sample:
                        removed.append(cis)
                        del peaks[cis]
                    logging.info('Removed %s due to faulty activitiy.'
                                 'Consecutively increasing probabilities do not show increasing activities!' % isotope)
            # current isotope is valid
            else:
                logging.info('Isotope %s valid!' % isotope)

        if not_in_table:
            logging.warning('Isotope(s) %s not contained in gamma table; not validated!' % ', '.join(not_in_table))
        else:
            if not removed:
                logging.info('All isotopes validated!')

    # analysis

    def get_activity(self, observed_peaks, probability_peaks=None):
        """
        Method to calculate activity isotope-wise. The peak-wise activities of all peaks of
        each isotope are added and scaled with their respectively summed-up probability
        """
        activities = OrderedDict()
        probability_peaks = self.get_isotope_info(info='probability') if probability_peaks is None else probability_peaks

        for peak in observed_peaks:
            isotope = '_'.join(peak.split('_')[:-1])
            if isotope not in activities:
                activities[isotope] = OrderedDict([('nominal', 0), ('sigma', 0),
                                                   ('probability', 0), ('unscaled', {'nominal': 0, 'sigma': 0})])

            if peak in probability_peaks:
                activities[isotope]['unscaled']['nominal'] += observed_peaks[peak]['activity']['nominal']
                # squared sum in order to get Gaussian error propagation right
                activities[isotope]['unscaled']['sigma'] += observed_peaks[peak]['activity']['sigma'] ** 2
                activities[isotope]['probability'] += probability_peaks[peak]

        for iso in activities:
            try:
                activities[iso]['nominal'] = activities[iso]['unscaled']['nominal'] * 1. / activities[iso][
                    'probability']
                # square root of sum of squared sigmas in order to get Gaussian error propagation right
                activities[iso]['sigma'] = activities[iso]['unscaled']['sigma'] ** 0.5 * 1. / activities[iso][
                    'probability']

            # when no probability given
            except ZeroDivisionError:
                pass

        return activities

    def get_dose(self, peaks, distance, time=None, material='air'):
        """
        Method to calculate the dose or dose rate at distance in given material which the sample exposes. The dose (rate) is
        calculated isotope-wise and subsequently summed up for all present isotopes.

        Parameters
        ----------

        peaks : dict
            return value of irrad_spectroscopy.fit_spectrum
        distance : float
            distance in cm at which the dose rate should be calculated
        time : float
            time in hours for which the total effective dose should be calculated

        Returns
        -------

        dose : dict
            dict with info about dose (rate)
        """

        # make result dict
        dose = OrderedDict([('nominal', 0.0), ('sigma', 0.0), ('unit', None), ('isotopes', OrderedDict())])
        # get unique isotopes in sample
        isotopes = set('_'.join(p.split('_')[:-1]) for p in peaks)
        # get gamma lines in sample
        lines = peaks.keys()
        # get activities in sample
        activities = self.get_activity(peaks)

        if time:
            _type = 'dose in {} hours'.format(time)
            dose['unit'] = 'uSv'
            half_lifes = self.get_isotope_info(info='half_life')
        else:
            _type = 'dose rate'
            dose['unit'] = 'uSv/h'

        # user feedback
        logging.info('Calculating {} in {} for distance of {} cm from point source'.format(_type, material, distance))

        for iso in isotopes:
            dose['isotopes'][iso] = OrderedDict([('nominal', 0.0), ('sigma', 0.0), ('lines', OrderedDict())])
            iso_activity_n = activities[iso]['nominal']
            iso_activity_s = activities[iso]['sigma']
            iso_probabilities = self.get_isotope_info(info='probability', iso_filter=iso)
            for l in lines:
                if time:
                    # integrate activitiy over time; convert half life to hours since dose rate given in uSv/h
                    tmp_activity_n, _ = quad(self.decay_law, 0, time,
                                             args=tuple([iso_activity_n, half_lifes[iso] / 60. ** 2]))
                    tmp_activity_s, _ = quad(self.decay_law, 0, time,
                                             args=(tuple[iso_activity_s, half_lifes[iso] / 60. ** 2]))
                else:
                    tmp_activity_n, tmp_activity_s = iso_activity_n, iso_activity_s

                if iso in l:
                    dose['isotopes'][iso]['lines'][l] = OrderedDict([('nominal', 0.0), ('sigma', 0.0)])
                    tmp_energy = peaks[l]['peak_fit']['popt'][0]
                    tmp_probability = iso_probabilities[l]
                    tmp_dose_n = self.gamma_dose_rate(tmp_energy, tmp_probability, tmp_activity_n, distance, material)
                    tmp_dose_s = self.gamma_dose_rate(tmp_energy, tmp_probability, tmp_activity_s, distance, material)
                    dose['isotopes'][iso]['lines'][l]['nominal'] = tmp_dose_n
                    dose['isotopes'][iso]['lines'][l]['sigma'] = tmp_dose_s
                    dose['isotopes'][iso]['nominal'] += tmp_dose_n
                    # squared sum in order to get Gaussian error propagation right
                    dose['isotopes'][iso]['sigma'] += tmp_dose_s ** 2

            # square root of sum of squared sigmas in order to get Gaussian error propagation right
            dose['isotopes'][iso]['sigma'] **= 0.5

            dose['nominal'] += dose['isotopes'][iso]['nominal']
            dose['sigma'] += dose['isotopes'][iso]['sigma'] ** 2

        # square root of sum of squared sigmas in order to get Gaussian error propagation right
        dose['sigma'] **= 0.5

        return dose
