from unfoldingsuite.datahandler import UnfoldingDataHandler

import unittest
import os


# the directory of the test reference data
REF_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'testdata')


class FileUnitTest(unittest.TestCase):
    def setUp(self):
        self.unfolding_data = UnfoldingDataHandler(verbosity=0)
        self.unfolding_data.load_vector('reaction_rates', os.path.join(REF_DATA_DIR, "5ChannelReactionRates.csv"))
        self.data_reaction_rates = self.unfolding_data.vectors['reaction_rates']
        self.unfolding_data.load_matrix('response_matrix', os.path.join(REF_DATA_DIR, "5ChannelResponse.csv"))
        self.data_response_matrix = self.unfolding_data.matrices['response_matrix']

    def test_load_reaction_rate_as_vector_from_csv(self):
        vector_stored_in_file = [166, 96, 93, 130, 93]
        self.assertListEqual(self.data_reaction_rates, vector_stored_in_file, "Assert a 1d vector stored in ")

    def test_loaded_reaction_rate_as_vector_from_csv_is_1d_list_of_floats(self):
        self.assertEqual(type(self.data_reaction_rates), list)
        for value in self.data_reaction_rates:
            self.assertEqual(float, type(value))

    def test_loaded_response_matrix_as_matrix_from_csv(self):
        matrix_stored_in_file = [[2, 5, 7, 4, 1],
                                 [4, 3, 1, 2, 3],
                                 [3, 2, 2, 3, 5],
                                 [3, 2, 4, 6, 7],
                                 [2, 2, 4, 1, 5]]
        self.assertEqual(self.data_response_matrix, matrix_stored_in_file)

    def test_loaded_response_matrix_as_matrix_from_csv_is_2d_list_of_floats(self):
        self.assertEqual(type(self.data_response_matrix), list)
        for row in self.data_response_matrix:
            self.assertEqual(type(row), list)

