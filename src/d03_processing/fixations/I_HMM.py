from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixationProcessor import FixationProcessor
from hmmlearn import hmm
import numpy as np
from scipy.spatial import distance


class I_HMM(FixationProcessor):
    """
        Class for implementing hidden markov model for identification of fixations from saccades - Salvucci ref
        """

    def __init__(self, timepoints):
        super().__init__(timepoints)
        self.method_name = 'I_HMM'
        self.timestamp_units_per_second = 1000  # i.e. ms
        # the below params were estimated using a sample of the data
        self.startprob_ = np.array([0.9999999271603389, 7.283966106505518e-08])  # initial state probabilities
        self.transmat_ = np.array([[0.9499550174395448, 0.05004498256045514],
                                   [0.1894995436920064, 0.8105004563079936]])  # state transition matrix
        self.means_ = np.array([[0.00866825], [0.12547113]])  # mean of each state's Gaussian emission distribution
        self.covars_ = np.array(
            [[8.23893934e-05], [0.00949561]])  # variance of each state's Gaussian emission distribution
        self.fix_df, self.timepoints = self.get_fixations_missing_split(self.timepoints)

    def get_fixations(self, timepoints=None, missing_split_group_id=0):
        super().get_fixations(timepoints)
        tps = self.timepoints

        if self.skip_algo:
            return None

        # adjust negative time error in timestamp

        first_time = tps.eye_timestamp_ms[0]
        if first_time < 0:
            tps.eye_timestamp_ms = tps.eye_timestamp_ms + np.absolute(first_time) + 1
        t = timepoints.eye_timestamp_ms.to_numpy()
        # velocity (angular)
        v_ang = FixationProcessor.angular_velocity_vec(tps)

        # Define the HMM model
        model = hmm.GaussianHMM(n_components=2)

        # Initialize the model parameters
        model.startprob_ = self.startprob_
        model.transmat_ = self.transmat_
        model.means_ = self.means_
        model.covars_ = self.covars_

        # Decode the velocities with the HMM model to identify fixation and saccade points
        log_prob, state_sequence = model.decode(v_ang.reshape(-1, 1), algorithm="viterbi")

        # get fixation array - fixations will be smaller mean velocity
        fixation = np.where(state_sequence == np.argmin(model.means_), 1, 0)

        # correct length
        fixation = np.array(np.hstack([fixation, np.zeros([1, ])]), dtype=int)

        # time threshold
        fixation = FixationProcessor.threshold_fixboolarray_duration(fixation, self.fixation_threshold, t)

        # get new start and end
        fix_start, fix_end = TimepointProcessor.get_start_end(fixation)

        # add to original tp df
        tps['fixation'] = fixation
        tps['fixation_start'] = fix_start
        tps['fixation_end'] = fix_end
        return self.convert_fix_df_format(tps, missing_split_group_id), tps

    def train_model(self, timepoints=None):
        tps = self.timepoints

        # velocity (angular)
        v_ang = FixationProcessor.angular_velocity_vec(tps)

        # Define the HMM model
        model = hmm.GaussianHMM(n_components=2)

        # Train the model
        model.fit(v_ang.reshape(-1, 1))
        return model
