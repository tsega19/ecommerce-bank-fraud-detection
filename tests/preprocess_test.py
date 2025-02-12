import unittest

from scripts.preprocess import handle_missing_values

class TestPreprocess(unittest.TestCase):

    def test_handle_missing_values(self):
        df = pd.DataFrame({'A': [1, 2, None, 4],
                           'B': [5, 6, 7, 8]})
        result = handle_missing_values(df)
        expected = pd.DataFrame({'A': [1, 2, 3, 4],
                                 'B': [5, 6, 7, 8]})
        pd.testing.assert_frame_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
