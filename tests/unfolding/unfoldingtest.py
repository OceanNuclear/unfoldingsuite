from unfoldingsuite.datahandler import UnfoldingDataHandler
from unfoldingsuite.nonlinearleastsquare import SAND_II, GRAVEL, SPUNIT
from unfoldingsuite.maximumentropy import MAXED

import unittest
import os


# the directory of the test reference data
REF_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'testdata')



class UnfoldingUnitTest(unittest.TestCase):
    def setUp(self):
        self.unfolding_data = UnfoldingDataHandler(verbosity=0)
        self.unfolding_data.load_vector('reaction_rates', os.path.join(REF_DATA_DIR, "5ChannelReactionRates.csv"))
        self.unfolding_data.load_vector_uncertainty('reaction_rates', os.path.join(REF_DATA_DIR, "5ChannelReactionRateUncertainties.csv"))
        self.unfolding_data.load_vector('a_priori', os.path.join(REF_DATA_DIR, "5ChannelAPriori.csv"))
        self.unfolding_data.load_matrix('response_matrix', os.path.join(REF_DATA_DIR, "5ChannelResponse.csv"))

    def test_run_sand_ii_5_channel_full_rank_response(self):
        sand_ii = SAND_II(verbosity=0)
        sand_ii.set_matrices_and_vectors(self.unfolding_data)
        sand_ii.run('n_trials', 100000)
        self.assertAlmostEqual(sand_ii.vectors['solution'][0], 8.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][1], 13.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][2], 9.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][3], 5.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][4], 2.0, 2)

    def test_run_gravel_5_channel_full_rank_response(self):
        gravel = GRAVEL(verbosity=0)
        gravel.set_matrices_and_vectors(self.unfolding_data)
        gravel.run('n_trials', 100000)
        self.assertAlmostEqual(gravel.vectors['solution'][0], 8.0, 2)
        self.assertAlmostEqual(gravel.vectors['solution'][1], 13.0, 2)
        self.assertAlmostEqual(gravel.vectors['solution'][2], 9.0, 2)
        self.assertAlmostEqual(gravel.vectors['solution'][3], 5.0, 2)
        self.assertAlmostEqual(gravel.vectors['solution'][4], 2.0, 2)
    
    def test_run_SPUNIT_5_channel_full_rank_response(self):
        sand_ii = SPUNIT(verbosity=0)
        sand_ii.set_matrices_and_vectors(self.unfolding_data)
        sand_ii.run('n_trials', 100000)
        self.assertAlmostEqual(sand_ii.vectors['solution'][0], 8.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][1], 13.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][2], 9.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][3], 5.0, 2)
        self.assertAlmostEqual(sand_ii.vectors['solution'][4], 2.0, 2)
