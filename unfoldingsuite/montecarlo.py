from unfoldingsuite.datahandler import UnfoldingDataHandler
import math
from scipy.special import erfinv
import random


class MonteCarlo(UnfoldingDataHandler):
    def __init__(self, unfolding_method, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print('Monte Carlo')

        self.matrix_dimensions = {**self.matrix_dimensions}
        self.vector_dimensions = {**self.vector_dimensions}

        self.unfolding_method = unfolding_method
        self.method = 'Monte Carlo'
        self.trials = 0
        self.minimum_number_of_trials = 20
        self.tolerance = 0.1
        self.confidence = 0.95
        self.batch_size = 10
        self.solution_spectra = []
        self.vectors_to_randomise = ['reaction_rates']
        self.matrices_to_randomise = []

    @UnfoldingDataHandler._setup_and_summarise
    def run(self, run_type, *args, **kwargs):
        confidence_met = False

        # Tun the initial trial here
        method_instance = self._run_single_trial(run_type, *args, **kwargs)
        self.vectors['solution'] = [flux for flux in method_instance.vectors['solution']]
        self.trials = 1

        while (not confidence_met) or (self.trials < self.minimum_number_of_trials):
            method_instance = self._run_single_trial(run_type, *args, **kwargs)

            previous_weighted = [flux * self.trials / (self.trials + 1.0) for flux in self.vectors['solution']]
            new_weighted = [flux / (self.trials + 1.0) for flux in method_instance.vectors['solution']]
            self.vectors['solution'] = [prev + new for prev, new in zip(previous_weighted, new_weighted)]
            self.trials += 1
            self.solution_spectra.append([flux for flux in method_instance.vectors['solution']])

            if self.trials % self.batch_size == 0.0:
                self.vector_uncertainties['solution'] = [0.0 for _ in range(self.n)]
                print(self.vectors['solution'])
                print('\n')
                for solution in self.solution_spectra:
                    print(solution)
                    for j in range(self.n):
                        self.vector_uncertainties['solution'][j] += (solution[j] - self.vectors['solution'][j]) ** 2
                self.vector_uncertainties['solution'] = [math.sqrt(sigma_sum / self.trials) / mean for mean, sigma_sum in zip(self.vectors['solution'], self.vector_uncertainties['solution'])]
                print('\n')

                trials_required = [self._calculate_number_of_trials_required(mean, mean * mu, self.tolerance, self.confidence) for mean, mu in zip(self.vectors['solution'], self.vector_uncertainties['solution'])]

                print(self.vectors['solution'])
                print(self.vector_uncertainties['solution'])
                print(trials_required)

                if self.trials >= max(trials_required):
                    confidence_met = True
                else:
                    print('Trials complete: {0} Trials required: {1}'.format(self.trials, max(trials_required)))
            # Calculate the new running mean

        for solution in self.solution_spectra:
            print(solution)

    def _calculate_number_of_trials_required(self, sample_mean, sample_standard_deviation, margin_or_error, confidence_interval):
        if sample_mean != 0:
            delta = math.sqrt(2) * erfinv(2*(confidence_interval/2))
            return math.ceil((delta**2) * ((sample_standard_deviation**2)/((margin_or_error*sample_mean)**2)))
        else:
            return 0

    def _run_single_trial(self, run_type, *args, **kwargs):
        print('Trial: {0}'.format(self.trials))
        method_instance = self.unfolding_method(verbosity=0)
        method_instance.set_matrices_and_vectors(self)

        def random_skewed_gaussian(mu, sigma):
            random_value = 0.0
            while random_value <= 0.0:
                random_value = random.gauss(mu, sigma)
            return random_value

        def randomise_vector(vector, vector_uncertainty):
            return [random_skewed_gaussian(flux, flux * sigma_rel) for flux, sigma_rel in zip(vector, vector_uncertainty)]

        for vector_label in self.vectors_to_randomise:
            print(vector_label)
            method_instance.vectors[vector_label] = randomise_vector(method_instance.vectors[vector_label],
                                                                     method_instance.vector_uncertainties[vector_label])

        for matrix_label in self.matrices_to_randomise:
            for row_index, (vector, vector_uncertainty) in enumerate(zip(self.matrices[matrix_label], self.matrix_uncertainties[matrix_label])):
                self.matrices[matrix_label][row_index] = randomise_vector(vector, vector_uncertainty)

        method_instance.run(run_type, *args, **kwargs)
        return method_instance
