import unittest
import datetime
from datetime import timedelta

from utils.regpay_utils import *


class UtilTest(unittest.TestCase):

    def test_calendar_df(self):
        min_date = datetime.date(2022, 1, 1)
        max_date = datetime.date(2022, 2, 1)
        df = create_calendar_df(min_date, max_date)
        dates = df.index
        start_date = min_date
        while start_date <= max_date:
            self.assertTrue(start_date in dates)
            start_date = start_date + datetime.timedelta(days=1)
        self.assertFalse(start_date in dates)

    def test_enums(self):
        self.assertEqual(closest_enum(1), 'DAILY')
        self.assertEqual(closest_enum(2), 'DAILY')
        self.assertEqual(closest_enum(7), 'WEEKLY')
        self.assertEqual(closest_enum(8), 'WEEKLY')
        self.assertEqual(closest_enum(13), 'BIWEEKLY')
        self.assertEqual(closest_enum(15), 'BIWEEKLY')
        self.assertEqual(closest_enum(29), 'MONTHLY')
        self.assertEqual(closest_enum(31), 'MONTHLY')
        self.assertEqual(closest_enum(42), 'BIMONTHLY')
        self.assertEqual(closest_enum(89), 'QUARTERLY')
        self.assertEqual(closest_enum(123), '')

    def test_triangles(self):
        amounts = pd.Series([
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            9.99
        ])

        tri_df = to_triangle_signal(amounts)
        actual = [round(x, 2) for x in tri_df[0:32]]
        expected = [
            9.99, 9.32, 8.66, 7.99, 7.33, 6.66, 5.99, 5.33, 4.66, 4.0, 3.33, 2.66,
            2.0, 1.33, 0.67, 0.0, 0.62, 1.25, 1.87, 2.5, 3.12, 3.75, 4.37, 5.0,
            5.62, 6.24, 6.87, 7.49, 8.12, 8.74, 9.37, 9.99
        ]
        self.assertListEqual(actual, expected)


    def test_fourier(self):
        import pandas as pd
        transactions = pd.read_json('''
        {
          "amount":
            {
              "1547683200000":9.99,
              "1550361600000":9.99,
              "1552780800000":9.99,
              "1555459200000":9.99,
              "1558051200000":9.99,
              "1560729600000":9.99,
              "1563321600000":9.99,
              "1566000000000":9.99,
              "1568678400000":9.99,
              "1571270400000":9.99,
              "1573948800000":9.99,
              "1576540800000":9.99
            }
        }
        ''')

        # create a dense representation
        min_date = np.min(transactions.index)
        max_date = np.max(transactions.index)
        dates_df = create_calendar_df(min_date, max_date)
        transactions_ts = pd.merge(transactions, dates_df, left_index=True, right_index=True, how='right').fillna(0)

        # convert time series to triangles
        xs = to_triangle_signal(transactions_ts.amount)

        # extract fourier frequencies
        fft_ts_tri = fourier_transform(xs)

        # extract periods
        threshold = detect_anomalies(fft_ts_tri.amplitudes)
        fft_ts_tri_df = fft_ts_tri.filter(threshold).to_df()
        periods = fft_ts_tri_df[fft_ts_tri_df['freq'] > 0]
        self.assertEqual(periods.shape[0], 1)
        self.assertEqual(periods.freq.iloc[0], 11)


if __name__ == '__main__':
    unittest.main()
