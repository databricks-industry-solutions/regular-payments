import pandas as pd
import numpy as np
from scipy import fft
import copy


def create_calendar_df(min_date, max_date):
    dates = list(pd.date_range(min_date, max_date, freq='d'))
    dates = [d.date() for d in dates]
    dates_df = pd.DataFrame(dates, columns=['date'])
    dates_df.index = dates_df.date
    dates_df = dates_df.drop(columns=['date'], axis=1)
    return dates_df


def closest_enum(period):
    """Simple approximation from period to ENUMs"""
    if period <= 2:
        return "DAILY"
    if 6 <= period <= 8:
        return "WEEKLY"
    if 13 <= period <= 17:
        return "BIWEEKLY"
    if 25 <= period <= 32:
        return "MONTHLY"
    if 40 <= period <= 65:
        return "BIMONTHLY"
    if 80 <= period <= 95:
        return "QUARTERLY"
    return ""


def to_triangle_signal(xs):
  
    split = list(np.where(xs != 0)[0])
    triangular = []

    for i in range(len(split)):

        split_id = split[i:i+2]
        if len(split_id) == 2:

            src_id = split_id[0]
            src_vl = xs[src_id]

            dst_id = split_id[1]
            dst_vl = xs[dst_id]

            mid_id = src_id + int((dst_id - src_id) / 2)
            mid_vl = 0

            a1 = (mid_vl - src_vl) / (mid_id - src_id)  # slope
            a2 = (dst_vl - mid_vl) / (dst_id - mid_id)  # slope

            for j in np.arange(src_id, dst_id, 1):
                if j < mid_id:
                    j_vl = src_vl + (j - src_id) * a1
                else:
                    j_vl = mid_vl + (j - mid_id) * a2
                triangular.append(float(j_vl))

    return triangular


class FourierWrapper:
  
    def __init__(self, spectrum, frequencies, sample_size):
        import numpy as np
        self.spectrum = spectrum
        self.frequencies = frequencies
        self.amplitudes = np.abs(spectrum)
        self.sample_size = sample_size

    def filter(self, threshold):
        dominants = self.amplitudes >= threshold
        return FourierWrapper(self.spectrum[dominants], self.frequencies[dominants], self.sample_size)

    def to_df(self):
        import pandas as pd
        df = pd.DataFrame(
          zip(self.spectrum.real, self.spectrum.imag, self.frequencies),
          columns=['real', 'imag', 'freq']
        )
        df['size'] = self.sample_size
        return df
  
  
def fourier_transform(xs):
    spectrum = fft.fft(xs)
    frequencies = fft.fftfreq(len(xs), 1/len(xs))
    return FourierWrapper(spectrum, frequencies, len(xs))


def detect_anomalies(amplitudes):

    # to detect highly regular signals, it is safe to restrict space to top 2 frequencies only
    # we have both imaginary and real number, so times 2 + the zero frequency
    # we use our kernel density with the maximum distance between our top frequency with the rest
    a = copy.deepcopy(amplitudes)
    a.sort()
    threshold = np.max(a[:-7])

    # use density clustering to find anomalies
    from sklearn.cluster import DBSCAN
    clustered = DBSCAN(eps=threshold, min_samples=3).fit_predict([[x] for x in a])

    # we need at least 3 points within a given distance to group frequencies together
    # a frequency being further than 3 points within a distance will therefore be considered as outlier
    unclustered = a[clustered == -1]
    if np.size(unclustered) == 0:
        return np.inf
    else:
        return np.min(unclustered)