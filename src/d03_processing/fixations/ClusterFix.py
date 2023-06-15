import warnings

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d

from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixationProcessor import FixationProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor


class ClusterFix(FixationProcessor):
    """
    lightly adapted from https://www.sciencedirect.com/science/article/abs/pii/S0165027014000491
    based on k-means clustering of acceleration, velocity and angular velocity
    """

    def __init__(self, timepoints):
        super().__init__(timepoints)
        self.method_name = 'ClusterFix'
        self.fixation_threshold = 25  # ms
        self.max_gap_length = 75  # ms for blinks Komogortsev et al Standardization of Automated Analyses of Oculomotor Fixation and Saccadic Behaviors
        self.n_clust = 10
        self.use_silhouette = False     # determine n_clust by max silhouette score
        self.fix_df = self.get_fixations(self.timepoints)

    def get_fixations(self, timepoints):
        super().get_fixations(timepoints)
        tps = self.timepoints
        if self.skip_algo:
            return None

        x = tps.gaze_collision_x.to_numpy()
        y = tps.gaze_collision_y.to_numpy()
        z = tps.gaze_collision_z.to_numpy()
        t = tps.eye_timestamp_ms.to_numpy()
        sig_mat = np.concatenate([x.reshape(-1, 1), z.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        sigproc = SignalProcessor(sig_mat, t)
        new_interval = 1    # 1000Hz
        t_new = sigproc.up_t(new_interval)
        up_sig = sigproc.upsample_multi(new_interval)
        filtered_sig = []
        for i in range(3):
            filtered_sig.append(sigproc.wiener_filter(up_sig[i]))
        point_matrix = np.concatenate(filtered_sig, axis=1)

        # project to 2d
        cam_x = tps.camera_x.to_numpy()
        cam_z = tps.camera_z.to_numpy()
        cam_y = tps.camera_y.to_numpy()
        head_loc_mat = np.concatenate([cam_x.reshape(-1, 1), cam_z.reshape(-1, 1), cam_y.reshape(-1, 1)], axis=1)

        head_loc_up = SignalProcessor(head_loc_mat, t).upsample_multi(new_interval)
        head_loc_up = np.concatenate([head_loc_up[0].reshape(-1, 1),
                                      head_loc_up[1].reshape(-1, 1),
                                      head_loc_up[2].reshape(-1, 1)], axis=1)

        rot_x = tps.cam_rotation_x.to_numpy()
        rot_z = tps.cam_rotation_z.to_numpy()
        rot_y = tps.cam_rotation_y.to_numpy()
        head_rot_mat = np.concatenate([rot_x.reshape(-1, 1), rot_z.reshape(-1, 1), rot_y.reshape(-1, 1)], axis=1)
        head_rot_up = SignalProcessor(head_rot_mat, t).upsample_multi(new_interval)
        head_rot_up = np.concatenate([head_rot_up[0].reshape(-1, 1),
                                      head_rot_up[1].reshape(-1, 1),
                                      head_rot_up[2].reshape(-1, 1)], axis=1)
        proj_points = FixationProcessor.head_project(point_matrix, head_loc_up)

        # distance, velocity, acceleration based on projected points
        distance = FixationProcessor.displacement_vector(proj_points)
        v, acc = ClusterFix.v_a(distance, t_new)

        # distance, velocity, acceleration based on angular shifts
        vectors = point_matrix - head_loc_up
        angles = []
        for i in range(1, len(vectors)):
            angles.append(FixationProcessor.angle_between(vectors[i - 1], vectors[i]))
        angles = np.array(angles)
        v_ang, acc_ang = ClusterFix.v_a(angles, t_new)

        # max 3sd and normalise from 0 to 1
        def max_sd(data, n):
            sd_max = n * np.std(data)
            return np.where(data < sd_max, data, sd_max)

        def NormalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        dat_3d = [v[1:], v_ang[1:], acc]
        for i in range(len(dat_3d)):
            dat_3d[i] = NormalizeData(max_sd(dat_3d[i], 3))

        data = np.concatenate([dat_3d[0].reshape(-1, 1),
                               dat_3d[1].reshape(-1, 1),
                               dat_3d[2].reshape(-1, 1)], axis=1)


        # kmeans clustering
        n_clust = self.n_clust

        if self.use_silhouette:
            sil = np.zeros([n_clust])
            range_clusters = range(2, n_clust)
            data_tenth = data[0:10:, :]
            for n_clust in range_clusters:
                kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=5).fit(data_tenth)
                silh = silhouette_score(data_tenth, kmeans.labels_)
                sil[n_clust] = np.mean(silh)
            n_clust = np.argmax(sil)

        kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=5).fit(data)

        fix_group = np.argmin(np.mean(kmeans.cluster_centers_, axis=1))
        new_labels = np.where(kmeans.labels_ == fix_group, 0, 1)

        # apply fixation threshold


        fix_threshold = self.fixation_threshold

        thresholded_labels = []
        thresholded_labels.extend(new_labels)
        thresholded_labels = np.array(thresholded_labels)

        thresholded_labels = FixationProcessor.threshold_fixboolarray_duration(thresholded_labels,
                                                                               fix_threshold,
                                                                               t_new,
                                                                               inverse=True)

        # fixation column
        fixation = np.where(thresholded_labels == 1, 0, 1)

        # downsample to original times
        fix_down = SignalProcessor.downsample_1d_threshold(t_new, t, fixation, 0.5)

        # add fix_start and fix_end
        fix_start, fix_end = TimepointProcessor.get_start_end(fix_down)

        # add to original tp df
        tps['fixation'] = fix_down
        tps['fixation_start'] = fix_start
        tps['fixation_end'] = fix_end
        self.timepoints = tps

        return self.convert_fix_df_format(tps)

    @staticmethod
    def v_a(d, t):
        time_diff = np.diff(t)
        v = d / time_diff
        delta_v = np.diff(v)
        acc = np.abs(delta_v / time_diff[1:])
        return v, acc

    @staticmethod
    def ClusterFix(eyedat, samprate=5e-3):
        """
        Converted to python using chatGPT - needs amending

        % Copyright 2013-2017 Seth Koenig (skoenig3@uw.edu) & Elizabeth Buffalo,
        % all rights reserved.
        %
        % Function detects periods of fixations and saccades using k-means
        % clustering. Code does not distinguish beteen saccades and micro-saccades.
        % Periods of fixation are distinguished between periods of saccade using 4
        % parameters dervied from Low Pass Filtered eye data that is resampled from
        % 200 Hz to 1000 Hz. Parameters are distance, velocity, acceleration, and
        % angular velocity. All parameters are necessary to increase the senitivity
        % of the code. Clustering is initialy done on all data points (global
        % clustering) during one presentation of images then clustering is redone on
        % perviously detected fixations (local re-clustering) to increase sensitivtiy
        % to small saccades. Saccades are required to be longer than 10 ms in duration
        % and fixations are required to be longer than 25 ms in duration. Typically
        % the only time that these durations are not met are during false
        % classification at peak acclerations or velocities when having low values in
        % the other parameter.These durations are free parameters (Lines 132,197,204).
        :param eyedat:
        :param samprate:
        :return:
        """
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        if len(eyedat) == 0:
            raise ValueError('No data file found')

        variables = ['Dist', 'Vel', 'Accel', 'Angular Velocity']

        fltord = 60
        lowpasfrq = 30
        nyqfrq = 1000 / 2
        flt = signal.firwin2(fltord, [0, lowpasfrq / nyqfrq, lowpasfrq / nyqfrq, 1], [1, 1, 0, 0])

        buffer = int(100 * samprate / 1000)
        fixationstats = []
        fixationstats  = np.zeros(len(eyedat))
        for cndlop in range(len(eyedat)):
            if eyedat[cndlop].shape[1] <= int(500 * samprate / 1000):

                # Filtering Extract Parameters from Eye Traces
                x = eyedat[cndlop][0, :] * 24 + 400
                y = eyedat[cndlop][1, :] * 24 + 300
                x = np.concatenate((x[buffer - 1::-1], x, x[-1:-buffer - 1:-1]))
                y = np.concatenate((y[buffer - 1::-1], y, y[-1:-buffer - 1:-1]))
                x = signal.resample(x, int(samprate * 1000))
                y = signal.resample(y, int(samprate * 1000))
                xss = signal.filtfilt(flt, 1, x)
                yss = signal.filtfilt(flt, 1, y)
                xss = xss[100:-100]
                yss = yss[100:-100]
                x = x[100:-100]
                y = y[100:-100]
                velx = np.diff(xss)
                vely = np.diff(yss)
                vel = np.sqrt(velx ** 2 + vely ** 2)
                accel = np.abs(np.diff(vel))
                angle = 180 * np.arctan2(vely, velx) / np.pi
                vel = vel[:-1]
                rot = np.zeros(len(xss) - 2)
                dist = np.zeros(len(xss) - 2)
                for a in range(len(xss) - 2):
                    rot[a] = np.abs(angle[a] - angle[a + 1])
                    dist[a] = np.sqrt((xss[a] - xss[a + 2]) ** 2 + (yss[a] - yss[a + 2]) ** 2)

                rot[rot > 180] = rot[rot > 180] - 180
                rot = 360 - rot

                points = np.column_stack((dist, vel, accel, rot))
                for ii in range(points.shape[1]):
                    thresh = np.mean(points[:, ii]) + 3 * np.std(points[:, ii])
                    points[(points[:, ii] > thresh), ii] = thresh
                    points[:, ii] = points[:, ii] - np.min(points[:, ii])
                    points[:, ii] = points[:, ii] / np.max(points[:, ii])

                # Global Clustering
                sil = np.zeros((5,))
                for numclusts in range(2, 6):
                    # Only using 3/4 parameters (-distance) and 1/10 of data points
                    T = KMeans(n_clusters=numclusts, n_init=5).fit(points[::10, 1:4])
                    silh = ClusterFix.InterVSIntraDist(points[::10, 1:4], T.labels_)
                    sil[numclusts - 1] = np.mean(silh)

                sil[sil > 0.9 * sil.max()] = 1
                numclusters = np.where(sil == sil.max())[0] + 2
                T = KMeans(n_clusters=numclusters[-1], n_init=5).fit(points)
                meanvalues = np.zeros((T.n_clusters, points.shape[1]))
                stdvalues = np.zeros((T.n_clusters, points.shape[1]))
                for TIND in range(T.n_clusters):
                    tc = np.where(T.labels_ == TIND)[0]
                    meanvalues[TIND, :] = points[tc, :].mean(axis=0)
                    stdvalues[TIND, :] = points[tc, :].std(axis=0)

                # determines fixation clusters by overlapping distributions in velocity and acceleration state space, here assumes gaussian distributions
                fixationcluster = np.argmin(np.sum(meanvalues[:, 1:2], axis=1))
                T[T == fixationcluster] = 100
                fixationcluster2 = \
                np.where(meanvalues[:, 1] < meanvalues[fixationcluster, 1] + 3 * stdvalues[fixationcluster, 1])[0]
                fixationcluster2 = np.delete(fixationcluster2, np.where(fixationcluster2 == fixationcluster))
                for iii in range(len(fixationcluster2)):
                    T[T == fixationcluster2[iii]] = 100
                T[T != 100] = 2
                T[T == 100] = 1

                fixationindexes = np.where(T == 1)[0]
                fixationtimes = ClusterFix.BehavioralIndex(fixationindexes)
                fixationtimes = fixationtimes[:,
                                np.diff(fixationtimes, axis=1) >= 25].squeeze()  # 25 ms duration threshold

                # Local Re-Clustering
                notfixations = []
                for ii in range(fixationtimes.shape[1]):
                    # select points left and right of fixation for comparison
                    altind = np.arange(fixationtimes[0, ii] - 50, fixationtimes[2, ii] + 50 + 1)
                    altind = altind[(altind >= 1) & (altind <= len(points))]
                    POINTS = points[altind, :]  # does not re-normalize
                    sil = np.zeros(5)
                    for numclusts in range(1, 6):
                        # can save a little bit of computation by only using 1/5 of data points
                        T = KMeans(n_clusters=numclusts, n_init=5).fit(POINTS[::5, :]).labels_
                        silh = ClusterFix.InterVSIntraDist(POINTS[::5, :], T)
                        sil[numclusts - 1] = np.mean(silh)
                    sil[sil > 0.9 * np.max(sil)] = 1
                    numclusters = np.where(sil == np.max(sil))[0]  # it's dangerous to have too many clusters
                    T = KMeans(n_clusters=int(np.median(numclusters)), n_init=5).fit(POINTS).labels_
                    rng = np.zeros((np.max(T), 2 * (POINTS.shape[1] - 1)))
                    # determines fixation clusters by overlapping median values in velocity
                    # and acceleration state space, here we DO NOT assume Gaussian distributions
                    # because there are not as many points and distributions rarely
                    # are normal
                    medianvalues = np.zeros((np.max(T), POINTS.shape[1]))
                    for TIND in range(1, np.max(T) + 1):
                        tc = np.where(T == TIND)[0]
                        if len(tc) == 1:
                            rng[TIND - 1, :] = np.ones(rng.shape[1])
                            medianvalues[TIND - 1, :] = POINTS[tc, :]
                        else:
                            rng[TIND - 1, :] = np.array([np.max(POINTS[tc, :-1]), np.min(POINTS[tc, :-1])])
                            medianvalues[TIND - 1, :] = np.median(POINTS[tc, :], axis=0)
                    fixationcluster = np.argmin(np.sum(medianvalues[:, 1:3], axis=1))
                    T[T == fixationcluster + 1] = 100
                    fixationcluster2 = np.where(
                        (medianvalues[fixationcluster, 1] < rng[:, 1::2]) & (
                                    medianvalues[fixationcluster, 1] > rng[:, 4::2]) &
                        (medianvalues[fixationcluster, 2] < rng[:, 2::2]) & (
                                    medianvalues[fixationcluster, 2] > rng[:, 5::2])
                    )[0]
                    fixationcluster2 = fixationcluster2

                # # Remove Points that are not fixations determing by Local Re-Clustering
                # ia = np.intersect1d(fixationindexes, notfixations, return_indices=True)[1]
                # fixationindexes = np.delete(fixationindexes, ia)
                # saccadeindexes = np.arange(len(points))
                # ib = np.intersect1d(fixationindexes, saccadeindexes, return_indices=True)[2]
                # saccadeindexes = np.delete(saccadeindexes, ib)

                # Remove Points that are not fixations determing by Local Re-Clustering
                fixationindexes, _, _ = np.intersect1d(fixationindexes, notfixations, return_indices=True)
                saccadeindexes = np.arange(points.shape[0])
                _, _, ib = np.intersect1d(fixationindexes, saccadeindexes, return_indices=True)
                saccadeindexes = np.delete(saccadeindexes, ib)

                # Consolidate & turn indexes into times
                saccadetimes = ClusterFix.BehavioralIndex(saccadeindexes)
                fixationtimes = ClusterFix.BehavioralIndex(fixationindexes)
                tooshort = np.where(np.diff(fixationtimes, axis=0, n=1) < 5)  # potential accidental fixationtimes
                notbehav = np.array([], dtype=np.int64)
                for ii in range(tooshort[0].shape[0]):
                    notbehav = np.concatenate(
                        [notbehav, np.arange(fixationtimes[0, tooshort[0][ii]], fixationtimes[1, tooshort[0][ii]] + 1)])
                saccadeindexes = np.sort(np.concatenate([saccadeindexes, notbehav]))
                tooshort = np.where(np.diff(saccadetimes, axis=0, n=1) < 10)  # 10 ms duration threshold for saccades
                notbehav = np.array([], dtype=np.int64)
                for ii in range(tooshort[0].shape[0]):
                    notbehav = np.concatenate(
                        [notbehav, np.arange(saccadetimes[0, tooshort[0][ii]], saccadetimes[1, tooshort[0][ii]] + 1)])
                fixationindexes = np.sort(np.concatenate([fixationindexes, notbehav]))
                fixationtimes, fixations = ClusterFix.BehavioralIndexXY(fixationindexes, x, y)

                saccadeindexes = np.arange(points.shape[0])
                _, _, ib = np.intersect1d(fixationindexes, saccadeindexes, return_indices=True)
                saccadeindexes = np.delete(saccadeindexes, ib)
                saccadetimes, _ = ClusterFix.BehavioralIndexXY(saccadeindexes, x, y)

                # Return indexes to previous sampling rate & Calculate mean fixation position
                round5 = np.remainder(fixationtimes, samprate * 1000)
                round5[0, round5[0, :] > 0] = samprate * 1000 - round5[0, round5[0, :] > 0]
                round5[1, :] = -round5[1, :]
                fixationtimes = np.round((fixationtimes + round5) / (samprate * 1000))
                fixationtimes[fixationtimes < 1] = 1

                round5 = np.remainder(saccadetimes, samprate * 1000)
                round5[0, :] = -round5[0, :]
                round5[1, round5[1, :] > 0] = samprate * 1000 - round5[1, round5[1, :] > 0]
                saccadetimes = np.round((saccadetimes + round5) / (samprate * 1000))
                saccadetimes[saccadetimes < 1] = 1

                x = eyedat[cndlop][0, :]  # * 24 + 400
                y = eyedat[cndlop][1, :]  # * 24 + 300
                saccadetimes[saccadetimes > len(x)] = len(x)
                fixationtimes[fixationtimes > len(x)] = len(x)

                # Calculate Whole Saccade and Fixation Parameters
                pointfix = np.full((fixationtimes.shape[1], 6), np.nan)
                for i in range(fixationtimes.shape[1]):
                    xss = x[fixationtimes[0, i]:fixationtimes[1, i]]
                    yss = y[fixationtimes[0, i]:fixationtimes[1, i]]
                    pp = ClusterFix.ExtractVariables(xss, yss, samprate)
                    pointfix[i, :] = pp

                pointsac = np.full((saccadetimes.shape[1], 6), np.nan)
                for i in range(saccadetimes.shape[1]):
                    xss = x[saccadetimes[0, i]:saccadetimes[1, i]]
                    yss = y[saccadetimes[0, i]:saccadetimes[1, i]]
                    pp = ClusterFix.ExtractVariables(xss, yss, samprate)
                    pointsac[i, :] = pp

                recalc_meanvalues = np.array([np.mean(pointfix, axis=0), np.mean(pointsac, axis=0)])
                if pointfix.shape[0] == 1:
                    recalc_stdvalues = np.full((2, 6), np.nan)
                else:
                    recalc_stdvalues = np.array([np.nanstd(pointfix, axis=0), np.nanstd(pointsac, axis=0)])

                fixationstats[cndlop]["fixationtimes"] = fixationtimes
                fixationstats[cndlop]["fixations"] = fixations
                fixationstats[cndlop]["saccadetimes"] = saccadetimes
                fixationstats[cndlop]["FixationClusterValues"] = pointfix
                fixationstats[cndlop]["SaaccadeClusterValues"] = pointsac
                fixationstats[cndlop]["MeanClusterValues"] = recalc_meanvalues
                fixationstats[cndlop]["STDClusterValues"] = recalc_stdvalues
                fixationstats[cndlop]["XY"] = [x, y]
                fixationstats[cndlop]["variables"] = variables
            else:
                x = eyedat[cndlop][0, :] * 24 + 400  # converts dva to pixels and data from [-400,400] to [0,800]
                y = eyedat[cndlop][1, :] * 24 + 300  # converts dva to pixels and from [-300,300] to [0,600]
                fixationstats[cndlop]['fixationtimes'] = []
                fixationstats[cndlop]['fixations'] = []
                fixationstats[cndlop]['saccadetimes'] = []
                fixationstats[cndlop]['FixationClusterValues'] = []
                fixationstats[cndlop]['SaaccadeClusterValues'] = []
                fixationstats[cndlop]['MeanClusterValues'] = []
                fixationstats[cndlop]['STDClusterValues'] = []
                fixationstats[cndlop]['XY'] = [x, y]
                fixationstats[cndlop]['variables'] = variables

        return fixationstats


    @staticmethod
    def BehavioralIndex(behavind):
        """chatGPT conversion of original code"""
        dind = np.diff(behavind)
        gaps = np.where(dind > 1)[0]
        behaveind = np.zeros((len(gaps), 50))
        if gaps.size > 0:
            for gapind in range(len(gaps) + 1):
                if gapind == 0:
                    temp = behavind[:gaps[gapind]]
                elif gapind == len(gaps):
                    temp = behavind[gaps[gapind - 1] + 1:]
                else:
                    temp = behavind[gaps[gapind - 1] + 1:gaps[gapind]]
                behaveind[gapind, :len(temp)] = temp
        else:
            behaveind = behavind
        behaviortime = np.zeros((2, behaveind.shape[0]))
        for index in range(behaveind.shape[0]):
            rowfixind = behaveind[index, :]
            rowfixind = rowfixind[rowfixind != 0]
            behaviortime[:, index] = [rowfixind[0], rowfixind[-1]]
        return behaviortime

    @staticmethod
    def BehavioralIndexXY(behavind, x, y):
        """chatGPT conversion of original code"""
        dind = np.diff(behavind)
        gaps = np.where(dind > 1)[0]
        behaveind = np.zeros((len(gaps), 50))
        if len(gaps) > 0:
            for gapind in range(len(gaps) + 1):
                if gapind == 0:
                    temp = behavind[:gaps[gapind]]
                elif gapind == len(gaps):
                    temp = behavind[gaps[gapind - 1] + 1:]
                else:
                    temp = behavind[gaps[gapind - 1] + 1:gaps[gapind]]
                behaveind[gapind, :len(temp)] = temp
        else:
            behaveind = behavind
        behaviortime = np.zeros((2, behaveind.shape[0]))
        behaviormean = np.zeros((2, behaveind.shape[0]))
        for index in range(behaveind.shape[0]):
            rowfixind = behaveind[index, :]
            rowfixind = rowfixind[rowfixind != 0]
            behaviortime[:, index] = [rowfixind[0], rowfixind[-1]]
            behaviormean[:, index] = [np.mean(x[rowfixind]), np.mean(y[rowfixind])]
        return (behaviortime, behaviormean)

    @staticmethod
    def extractVariables(xss, yss, samprate):
        """chatGPT conversion of original code"""
        n = len(xss)
        if n >= 3:
            vel = np.sqrt(np.diff(xss) ** 2 + np.diff(yss) ** 2) / samprate
            angle = 180 / np.pi * np.arctan2(np.diff(yss), np.diff(xss))
            accel = np.abs(np.diff(vel)) / samprate
            pp = np.zeros(6)
            pp[0] = np.max(vel)
            pp[1] = np.max(accel)
            dist = np.zeros(n - 2)
            rot = np.zeros(n - 2)
            for aa in range(n - 2):
                rot[aa] = np.abs(angle[aa] - angle[aa + 1])
                dist[aa] = np.sqrt((xss[aa] - xss[aa + 2]) ** 2 + (yss[aa] - yss[aa + 2]) ** 2)
            rot[rot > 180] = rot[rot > 180] - 180
            pp[2] = np.mean(dist)
            pp[3] = np.mean(vel)
            pp[4] = np.abs(np.mean(angle))
            pp[5] = np.mean(rot)
            return pp
        else:
            return np.full(6, np.nan)

    @staticmethod
    def InterVSIntraDist(X, clust):
        """chatGPT conversion of original code"""
        idx, cnames = stats.mode(clust)
        n = len(idx)
        k = len(cnames)
        count = np.histogram(idx[0], bins=range(1, k + 2))[0]
        mbrs = np.zeros((n, k), dtype=bool)
        for i in range(k):
            mbrs[:, i] = (clust == cnames[0][i])

        avgDWithin = np.full((n, 1), np.inf)
        avgDBetween = np.full((n, k), np.inf)
        for j in range(n):
            distj = np.sum((X - X[j]) ** 2, axis=1)
            for i in range(k):
                if i == idx[0][j]:
                    avgDWithin[j] = np.sum(distj[mbrs[:, i]]) / max(count[i] - 1, 1)
                else:
                    avgDBetween[j, i] = np.sum(distj[mbrs[:, i]]) / count[i]

        minavgDBetween = np.min(avgDBetween, axis=1)
        silh = (minavgDBetween - avgDWithin) / np.maximum(avgDWithin, minavgDBetween)
        return silh