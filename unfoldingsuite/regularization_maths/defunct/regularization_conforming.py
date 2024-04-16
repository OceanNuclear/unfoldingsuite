from unfoldingsuite.datahandler import UnfoldingDataHandler_2
from scipy import optimize

class MAXED_2(UnfoldingDataHandler_2):
    """

    MAXED unfolding method, inheriting from common DataHandler class.

    Args:
        *args: arguments to pass to parent DataHandler class
        **kwargs: Key-word arguments to pass to parent DataHandler class
    
    """
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        print('\nregularization\n')

        self.matrix_dimensions = {**self.matrix_dimensions}
        self.vector_dimensions = {**self.vector_dimensions}

        # Define the omega constant (later defaults to
        self.overshoot_protection_factor = 2
        # preferably larger than 1.255 for non-oscillatory approach ,
        # but MUST be a number larger than 1 in order to not break the program.

        self.method = 'regularization'
        