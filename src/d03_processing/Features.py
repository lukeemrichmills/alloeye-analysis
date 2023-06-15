from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.feature_calculate.PupilProcessor import PupilProcessor
from src.d03_processing.feature_calculate.edit_distance import lev_dist_xfix_s, lev_ratio_xfix_s
from src.d03_processing.feature_calculate.fix_sacc_calculations \
    import n_fix, dwell_time, t_first, n_sacc, velocity_mean, velocity_std, dispersion_mean, total_invalid, n_refix, \
    redwell, fix_duration_mean, refix_duration_mean
from src.d03_processing.feature_calculate.gaze_distance import average_distance_obj1, average_distance_pp, \
    average_distance_border, sum_gauss_duration_centroid, sum_gauss_duration_pp
from src.d03_processing.feature_calculate.viewing_compare_calcs import D_KL, ea_td, multimatch, mm_string
from src.d03_processing.feature_extract.group_trial_functions import n_trials, n_correct, p_correct, trial_mean, \
    confidence_mean, trial_median
from src.d03_processing.feature_calculate.transition_calculations import Hn, Hd, Ht, p_matrix, p_matrix_objects, \
    n_transitions, gini, gini_fix, gini_dwell, gini_refix, gini_redwell, Hn_s, Hd_s, Ht_s
from src.d03_processing.feature_calculate.trial_from_viewing_functions import return_enc, return_ret, return_diff
from src.d03_processing.FeatureBounds import FeatureBounds as fb

"""
Contains a class full of class dictionaries mapping feature names to their functions
"""

class Features:
    """dictionaries of functions with string key to tuple containing:
     (function, out_dtype, lower bound or string length, upper bound or string length)"""

    viewing_transition = {
        'hn': (Hn, float, 0, 2),   # entropy calculation using number of fixations as stationary distribution
        'hd': (Hd, float, 0, 2),    # entropy calculation using dwell time as stationary distribution
        'ht': (Ht, float, 0, 2),    # entropy calculation using number of transitions as stationary distribution
        'hn_s': (Hn_s, float, 0, 2),  # entropy calculation using number of fixations as stationary distribution
        'hd_s': (Hd_s, float, 0, 2),  # entropy calculation using dwell time as stationary distribution
        'ht_s': (Ht_s, float, 0, 2),  # entropy calculation using number of transitions as stationary distribution
        'p_matrix': (p_matrix, str, 5, 1000),
        'p_matrix_objects': (p_matrix_objects, str, 5, 1000),
        'n_transitions': (n_transitions, int, 0, 100),
        'gini_fix': (gini_fix, float, 0, 1),
        'gini_dwell': (gini_dwell, float, 0, 1),
        'gini_refix': (gini_refix, float, 0, 1),
        'gini_redwell': (gini_redwell, float, 0, 1),

    }
    viewing_fixation = {
        'n_fix_total': (n_fix, int, 2, 100),
        'n_fix_obj1': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_obj2': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_obj3': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_obj4': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_pp': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_table': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'n_fix_other': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        # 'n_fix_selected': (n_fix, int, 0, fb.n_fix_obj_upper_bound),
        'dwell_total': (dwell_time, float, fb.dwell_time_total_lower_bound, fb.dwell_time_total_upper_bound),
        'dwell_obj1': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_obj2': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_obj3': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_obj4': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_pp': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_table': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        'dwell_other': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        # 'dwell_selected': (dwell_time, float, 0, fb.dwell_time_obj_upper_bound),
        't_first_array': (t_first, float, fb.t_first_array_lower, fb.t_first_array_upper),
        't_first_obj1': (t_first, float, 0, fb.t_first_obj_upper),
        't_first_obj2': (t_first, float, 0, fb.t_first_obj_upper),
        't_first_obj3': (t_first, float, 0, fb.t_first_obj_upper),
        't_first_obj4': (t_first, float, 0, fb.t_first_obj_upper),
        'dispersion_mean': (dispersion_mean, float, fb.lower_easy, fb.upper_easy),
        'drop_out_total': (total_invalid, int, fb.lower_easy, fb.upper_easy),
        'n_refix_total': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_obj1': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_obj2': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_obj3': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_obj4': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_pp': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_table': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'n_refix_other': (n_refix, int, 0, fb.n_fix_obj_upper_bound),
        'redwell_total': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_obj1': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_obj2': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_obj3': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_obj4': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_pp': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_table': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'redwell_other': (redwell, float, 0, fb.dwell_time_obj_upper_bound),
        'fix_duration_mean': (fix_duration_mean, float, 0, fb.upper_easy),
        'refix_duration_mean': (refix_duration_mean, float, 0, fb.upper_easy),

    }
    viewing_saccade = {
        'n_sacc_total': (n_sacc, int, 0, 50),
        'n_sacc_obj1': (n_sacc, int, 0, fb.n_fix_obj_upper_bound),
        'n_sacc_obj2': (n_sacc, int, 0, fb.n_fix_obj_upper_bound),
        'n_sacc_obj3': (n_sacc, int, 0, fb.n_fix_obj_upper_bound),
        'n_sacc_obj4': (n_sacc, int, 0, fb.n_fix_obj_upper_bound),
        'n_sacc_table': (n_sacc, int, 0, fb.n_fix_obj_upper_bound),
        'velocity_mean_total': (velocity_mean, float, 0, 1000),
        'velocity_std_total': (velocity_std, float, 0, 1000)
    }
    viewing_pupil = {
        'pupil_diam_centre': (PupilProcessor.combined_diameter_median, float, 0, 10),
        'pupil_diam_spread': (PupilProcessor.combined_diameter_iqr, float, 0, 10),
        #'pupil_Hz': ...
    }
    viewing_blinks = {
        'n_blinks': (BlinkProcessor.n_blinks, int, 0, 100),
        'p_trackloss': (BlinkProcessor.total_p_trackloss, float, 0, fb.upper_easy),
    }
    viewing_gaze_distance = {
        'distance_obj1_median': (average_distance_obj1, float, 0, fb.upper_easy),
        'distance_pp_median': (average_distance_pp, float, 0, fb.upper_easy),
        'distance_border_median': (average_distance_border, float, 0, fb.upper_easy),
        'gauss_dwell_centroid': (sum_gauss_duration_centroid, float, 0, fb.upper_easy),
        'gauss_dwell_pp': (sum_gauss_duration_pp, float, 0, fb.upper_easy)
    }

    viewing_selection_delay = {
        # 'selection_delay': selection_delay
    }
    viewing_headset_loc = {}
    viewing_fix_sacc_dict = {**viewing_fixation, **viewing_saccade}
    viewing_fix_sacc = [*list(viewing_fixation.keys()),
                        *list(viewing_saccade.keys())]

    viewing_fix_sacc_derived = [*list(viewing_transition.keys()),
                                *viewing_fix_sacc]
    viewing_fix_sacc_derived_dict = {**viewing_transition, **viewing_fix_sacc_dict}
    viewing_from_timepoints = [*list(viewing_pupil.keys()),
                               *list(viewing_selection_delay.keys()),
                               *list(viewing_headset_loc.keys()),
                               *list(viewing_blinks.keys()),
                               *list(viewing_gaze_distance.keys())]

    viewing_from_timepoints_dict = {**viewing_pupil, **viewing_selection_delay,
                                    **viewing_headset_loc, **viewing_blinks, **viewing_gaze_distance}

    viewing = [*viewing_fix_sacc_derived, *viewing_from_timepoints]
    viewing_dict = {**viewing_from_timepoints_dict, **viewing_fix_sacc_derived_dict}

    # trial features from viewing features
    trial_enc, trial_ret, trial_diff = [], [], []
    trial_enc_dict, trial_ret_dict, trial_diff_dict = {}, {}, {}
    for i in range(len(viewing)):
        f = viewing[i]
        enc, ret, diff = f + '_enc', f + '_ret',  f + '_diff'
        trial_enc.append(enc)
        trial_ret.append(ret)
        trial_enc_dict[enc] = (return_enc, viewing_dict[f][1], fb.lower_easy, fb.upper_easy)
        trial_ret_dict[ret] = (return_ret, viewing_dict[f][1], fb.lower_easy, fb.upper_easy)
        if viewing_dict[f][1] is not str:
            trial_diff.append(diff)
            trial_diff_dict[diff] = (return_diff, viewing_dict[f][1], fb.lower_easy, fb.upper_easy)

    # trial features comparing retrieval and encoding eye patterns
    trial_view_compare_dict = {
        'd_kl': (D_KL, float, 0, fb.upper_easy),                            # relative entropy/KL distance (symmetric)
        'ea_td': (ea_td, float, fb.lower_easy, fb.upper_easy),                  # 'Eyeanalysis'
        'multimatch': (mm_string, str, 0, 500),
        'lev_dist_xfix_s': (lev_dist_xfix_s, int, 0, fb.upper_easy),
        'lev_ratio_xfix_s': (lev_ratio_xfix_s, float, 0, 1)

        # 'ScanMatch': (scan_match, float, lower_easy, upper_easy)       # 'ScanMatch' adapted string edit distance
        #

    }
    trial_from_viewing_dict = {**trial_enc_dict, **trial_ret_dict, **trial_diff_dict}
    trial_dict = {**trial_from_viewing_dict, **trial_view_compare_dict}
    trial = list(trial_dict.keys())

    trial_group = {
        'n_trials': (n_trials, int, 0, 1000),
        'n_correct': (n_correct, int, 0, 1000),
        'p_correct': (p_correct, float, 0, 1),
        'mean_confidence': (confidence_mean, float, 0, 10)
    }
    conditions_grouped_dict = {}
    for key, value in trial_dict.items():
        _type = value[1]
        if _type is not str:
            conditions_grouped_dict[key] = (trial_median, float, value[2], value[3])
            # note: all types float because all means

    conditions_multimatch = {
        'mm_vector': (trial_mean, float, 0, 1),
        'mm_direction': (trial_mean, float, 0, 1),
        'mm_length': (trial_mean, float, 0, 1),
        'mm_position': (trial_mean, float, 0, 1),
        'mm_duration': (trial_mean, float, 0, 1)
    }
    conditions_grouped_dict = {**conditions_grouped_dict, **conditions_multimatch}

    conditions_dict = {**trial_group, **conditions_grouped_dict}

    conditions = list(conditions_dict.keys())
    block = []
