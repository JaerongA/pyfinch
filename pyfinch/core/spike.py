"""
Module for neural analysis
"""
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


def get_isi(spk_ts_list: list):
    """
    Get inter-analysis interval of spikes

    Parameters
    ----------
    spk_ts_list : list

    Returns
    -------
    isi : class object
        class object for inter-spike intervals
    """

    isi = np.array([], dtype=np.float64)
    for spk in spk_ts_list:
        isi = np.append(isi, np.diff(spk))

    isi = ISI(isi)  # return the class object
    return isi


def get_peth(
    evt_ts_list: list,
    spk_ts_list: list,
    pre_evt_buffer=None,
    duration=None,
    bin_size=None,
    nb_bins=None,
):
    """
    Get peri-event histogram & firing rates

    Parameters
    ----------
    evt_ts_list : list
        Timestamps for behavioral events (e.g., syllable onset/offsets)
    spk_ts_list : list
        Spike timestamps
    pre_evt_buffer : int, default=None
        Size of buffer window prior to the first event (in ms)
    duration : int, optional
        Duration of the peth (in ms). Truncate the
    bin_size : int, default=None
        Time bin size
    nb_bins : int, default=None
        Number of bins

    Returns
    -------
    peth : np.ndarray
        Peri-event time histograms
    time_bin : np.ndarray
        Time bin vector
    parameter : dict
        Parameters for draw peth

    Notes
    -----
    If pre_evt_buffer, bin_size, nb_bins not specified,
    take values from analysis ..analysis.parameters
    """

    import copy
    import math

    from ..core.parameters import peth_parm

    parameter = peth_parm.copy()

    if pre_evt_buffer is None:
        pre_evt_buffer = parameter["buffer"]

    if bin_size is None:
        bin_size = parameter["bin_size"]

    if nb_bins is None:
        nb_bins = parameter["nb_bins"]

    time_bin = np.arange(0, nb_bins, bin_size) - pre_evt_buffer
    peth = np.zeros((len(evt_ts_list), nb_bins))  # nb of trials x nb of time bins

    for trial_ind, (evt_ts, spk_ts) in enumerate(zip(evt_ts_list, spk_ts_list)):
        spk_ts_new = copy.deepcopy(spk_ts)

        if not isinstance(evt_ts, np.float64):
            # evt_ts = np.asarray(list(map(float, evt_ts))) + pre_evt_buffer
            # spk_ts_new -= evt_ts[0]

            evt_ts = np.asarray(list(map(float, evt_ts)))
            spk_ts_new -= evt_ts[0]
            spk_ts_new += pre_evt_buffer

        else:
            spk_ts_new -= evt_ts
            spk_ts_new += pre_evt_buffer

        for spk in spk_ts_new:
            ind = math.ceil(spk / bin_size)
            # print("spk = {}, bin index = {}".format(spk, ind))  # for debugging
            if ind < 0:
                raise Exception("Index out of bound")
            peth[trial_ind, ind] += 1

    # Truncate the array leaving out only the portion of our interest
    if duration:
        ind = np.where(((0 - pre_evt_buffer) <= time_bin) & (time_bin < duration))[0]
        peth = peth[:, ind[0] : ind[-1] + 1]
        time_bin = time_bin[ind[0] : ind[-1] + 1]

    return peth, time_bin, parameter


def get_pcc(fr_array: np.ndarray) -> dict:
    """
    Get pairwise cross-correlation

    Parameters
    ----------
    fr_array : np.ndarray
        (trial x time_bin)

    Returns
    -------
    pcc_dict : dict
    """

    pcc_dict = {}
    pcc_arr = np.array([])

    for ind1, fr1 in enumerate(fr_array):
        for ind2, fr2 in enumerate(fr_array):
            if ind2 > ind1:
                if np.linalg.norm((fr1 - fr1.mean()), ord=1) * np.linalg.norm(
                    (fr2 - fr2.mean()), ord=1
                ):
                    if not np.isnan(np.corrcoef(fr1, fr2)[0, 1]):
                        pcc_arr = np.append(
                            pcc_arr, np.corrcoef(fr1, fr2)[0, 1]
                        )  # get correlation coefficient

    pcc_dict["array"] = pcc_arr
    pcc_dict["mean"] = round(pcc_arr.mean(), 3)
    return pcc_dict


def jitter_spk_ts(spk_ts_list, shuffle_limit, reproducible=True):
    """
    Add a random temporal jitter to the spike

    Parameters
    ----------
    reproducible : bool
        Make the results reproducible by setting the seed as equal to index
    """

    spk_ts_jittered_list = []
    for ind, spk_ts in enumerate(spk_ts_list):
        np.random.seed()
        if reproducible:  # randomization seed
            seed = ind
            np.random.seed(seed)  # make random jitter reproducible
        else:
            seed = np.random.randint(len(spk_ts_list), size=1)
            np.random.seed(seed)  # make random jitter reproducible
        nb_spk = spk_ts.shape[0]
        jitter = np.random.uniform(-shuffle_limit, shuffle_limit, nb_spk)
        spk_ts_jittered_list.append(spk_ts + jitter)
    return spk_ts_jittered_list


def pcc_shuffle_test(ClassObject, PethInfo, plot_hist=False, alpha=0.05):
    """
    Run statistical test to see if baseline pairwise cross-correlation obtained by spike time shuffling is significant

    Parameters
    ----------
    ClassObject : class object (e.g., NoteInfo, MotifInfo)
    PethInfo : peth info class object
    plot_hist : bool
        Plot histogram of bootstrapped pcc values (False by default)

    Returns
    -------
    p_sig : dict
        True if the pcc is significantly above the baseline
    """

    from collections import defaultdict
    from functools import partial

    import matplotlib.pyplot as plt
    import scipy.stats as stats

    from ..core.parameters import peth_shuffle

    pcc_shuffle = defaultdict(partial(np.ndarray, 0))
    for i in range(peth_shuffle["shuffle_iter"]):
        ClassObject.jitter_spk_ts(peth_shuffle["shuffle_limit"])
        pi_shuffle = ClassObject.get_note_peth(shuffle=True)  # peth object
        pi_shuffle.get_fr()  # get firing rates
        pi_shuffle.get_pcc()  # get pcc
        for context, pcc in pi_shuffle.pcc.items():
            pcc_shuffle[context] = np.append(pcc_shuffle[context], pcc["mean"])

    # One-sample t-test (one-sided)
    p_val = {}
    p_sig = {}

    for context in pcc_shuffle.keys():
        (_, p_val[context]) = stats.ttest_1samp(
            a=pcc_shuffle[context],
            popmean=PethInfo.pcc[context]["mean"],
            nan_policy="omit",
            alternative="less",
        )  # one-tailed t-test
    for context, value in p_val.items():
        p_sig[context] = value < alpha

    # Plot histogram
    if plot_hist:
        from ..utils.draw import remove_right_top

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plt.suptitle("PCC shuffle distribution", y=0.98, fontsize=10)
        for axis, context in zip(axes, pcc_shuffle.keys()):
            axis.set_title(context)
            axis.hist(pcc_shuffle[context], color="k")
            axis.set_xlim([-0.1, 0.6])
            axis.set_xlabel("PCC"), axis.set_ylabel("Count")
            if p_sig[context]:
                axis.axvline(
                    x=PethInfo.pcc[context]["mean"], color="r", linewidth=1, ls="--"
                )
            else:
                axis.axvline(
                    x=PethInfo.pcc[context]["mean"], color="k", linewidth=1, ls="--"
                )
            remove_right_top(axis)
        plt.tight_layout()
        plt.show()

    return p_sig


class ClusterInfo:
    def __init__(
        self,
        path,
        channel_nb,
        unit_nb,
        format="rhd",
        *name,
        update=False,
        time_unit="ms",
    ):
        """
        Load information about cluster

        Parameters
        ----------
        path : path
            path that contains recording files for the cluster
        channel_nb : int
            number of the channel that recorded the cluster
        unit_nb : int
            number id of the cluster (needed because multiple neurons could have been recorded in the same session & channel)
        format : str
            'rhd' by default (Intan)
        name : name of the cluster
            e.g., ('096-g70r40-Predeafening-D07(20191106)-S03-Ch17-Cluster01')
        update : bool
            If not exists, create a .npz cache file in the same folder so that it doesn't read from the raw data every time the class is called.
        time_unit : str
            'ms' by default
        """
        from ..core.load import load_song

        self.path = path
        if channel_nb:  # if a neuron was recorded
            if len(str(channel_nb)) == 1:
                self.channel_nb = "Ch0" + str(channel_nb)
            elif len(str(channel_nb)) == 2:
                self.channel_nb = "Ch" + str(channel_nb)
        else:
            self.channel_nb = "Ch"

        self.unit_nb = unit_nb
        self.format = format

        if name:
            self.name = name[0]
        else:
            self.name = self.path

        self._print_name()

        # Load events
        file_name = self.path / "ClusterInfo_{}_Cluster{}.npy".format(
            self.channel_nb, self.unit_nb
        )
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file
            song_info = load_song(self.path)
            # Save cluster_info as a numpy object
            np.save(file_name, song_info)
        else:
            song_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in song_info:
            setattr(self, key, song_info[key])

        # Load spike
        if channel_nb and unit_nb:
            self._load_spk(time_unit)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def _print_name(self) -> None:
        print("")
        print("Load cluster {self.name}".format(self=self))

    def list_files(self, ext: str):
        from ..utils.functions import list_files

        return list_files(self.path, ext)

    def _load_spk(self, time_unit, delimiter="\t") -> None:
        """
        Load spike information

        Parameters
        ----------
        time_unit : str
            time unit (e.g., 'ms')
        delimiter : str
            delimiter of the cluster file (tab (\t) by default)

        Returns
        -------
            sets spk_wf, spk_ts, nb_spk as attributes
        """

        spk_txt_file = list(self.path.glob("*" + self.channel_nb + "(merged).txt"))
        if not spk_txt_file:
            print("spk text file doesn't exist !")
            return

        spk_txt_file = spk_txt_file[0]
        spk_info = np.loadtxt(
            spk_txt_file, delimiter=delimiter, skiprows=1
        )  # skip header

        # Select only the unit (there could be multiple isolated units in the same file)
        if self.unit_nb:  # if the unit number is specified
            spk_info = spk_info[spk_info[:, 1] == self.unit_nb, :]

        spk_ts = spk_info[:, 2]  # analysis time stamps
        spk_wf = spk_info[:, 3:]  # analysis waveform
        nb_spk = spk_wf.shape[0]  # total number of spikes

        self.spk_wf = spk_wf  # individual waveforms
        self.nb_spk = nb_spk  # the number of spikes

        # Units are in second by default, but convert to  millisecond with the argument
        if time_unit == "ms":
            spk_ts *= 1e3

        # Output analysis timestamps per file in a list
        spk_list = []
        for file_start, file_end in zip(self.file_start, self.file_end):
            spk_list.append(
                spk_ts[np.where((spk_ts >= file_start) & (spk_ts <= file_end))]
            )

        self.spk_ts = spk_list  # analysis timestamps in ms
        # print("spk_ts, spk_wf, nb_spk attributes added")

    def analyze_waveform(self, align_wf=True, interpolate=True, interp_factor=None):
        """
        Perform waveform analysis

        Parameters
        ----------
        align_wf : bool
            align all spike waveforms relative to the max location
        interpolate : bool
            Set to true if waveform interpolation is needed
        interp_factor : int
            Factor by which to increase the sampling frequency of the waveform
            e.g., 100 if you want to increase the data points by 100 fold
        """
        from ..core.functions import align_waveform, get_half_width
        from ..core.parameters import sample_rate

        if align_wf:
            self.spk_wf = align_waveform(self.spk_wf)

        def _get_spk_profile(wf_ts, avg_wf, interpolate=interpolate):
            spk_height = np.abs(np.max(avg_wf) - np.min(avg_wf))  # in microseconds
            if interpolate:
                spk_width = (
                    abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1))
                    * ((1 / sample_rate[self.format]) / interp_factor)
                    * 1e6
                )  # in microseconds
            else:
                spk_width = (
                    abs(((np.argmax(avg_wf) - np.argmin(avg_wf)) + 1))
                    * (1 / sample_rate[self.format])
                    * 1e6
                )  # in microseconds
            deflection_range, half_width = get_half_width(
                wf_ts, avg_wf
            )  # get the half width from the peak deflection
            return spk_height, spk_width, half_width, deflection_range

        if not interp_factor:
            from ..core.parameters import interp_factor

            interp_factor = interp_factor

        self.avg_wf = np.nanmean(self.spk_wf, axis=0)
        self.wf_ts = (
            np.arange(0, self.avg_wf.shape[0]) / sample_rate[self.format] * 1e3
        )  # x-axis in ms

        if interpolate:  # interpolate the waveform to increase sampling frequency
            from scipy import interpolate

            f = interpolate.interp1d(self.wf_ts, self.avg_wf)
            wf_ts_interp = np.arange(
                0,
                self.wf_ts[-1],
                ((self.wf_ts[1] - self.wf_ts[0]) * (1 / interp_factor)),
            )
            assert (np.diff(wf_ts_interp)[0] * interp_factor) == np.diff(self.wf_ts)[0]
            avg_wf_interp = f(
                wf_ts_interp
            )  # use interpolation function returned by `interp1d`

            # Replace the original value with interpolated ones
            self.wf_ts_interp = wf_ts_interp
            self.avg_wf_interp = avg_wf_interp

            spk_height, spk_width, half_width, deflection_range = _get_spk_profile(
                wf_ts_interp, avg_wf_interp
            )
        else:
            spk_height, spk_width, half_width, deflection_range = _get_spk_profile(
                self.wf_ts, self.avg_wf
            )

        self.spk_height = round(spk_height, 3)  # in microvolts
        self.spk_width = round(spk_width, 3)  # in microseconds
        self.half_width = half_width
        self.deflection_range = (
            deflection_range  # the range where half width was calculated
        )

        # print("avg_wf, spk_height (uv), spk_width (us), wf_ts (ms) added")

    def get_conditional_spk(self) -> dict:
        """Get spike timestamps from different contexts"""
        conditional_spk = {}
        conditional_spk["U"] = [
            spk_ts
            for spk_ts, context in zip(self.spk_ts, self.contexts)
            if context == "U"
        ]
        conditional_spk["D"] = [
            spk_ts
            for spk_ts, context in zip(self.spk_ts, self.contexts)
            if context == "D"
        ]

        return conditional_spk

    def get_correlogram(self, ref_spk_list, target_spk_list, normalize=False) -> dict:
        """Get auto- or cross-correlogram"""
        import math

        from ..core.parameters import spk_corr_parm

        correlogram = {}

        for social_context in set(self.contexts):
            # Compute spk correlogram
            corr_temp = np.zeros(len(spk_corr_parm["time_bin"]))
            for ref_spks, target_spks, context in zip(
                ref_spk_list, target_spk_list, self.contexts
            ):

                if context == social_context:
                    for ref_spk in ref_spks:
                        for target_spk in target_spks:
                            diff = (
                                target_spk - ref_spk
                            )  # time difference between two spikes
                            if (diff) and (
                                diff <= spk_corr_parm["lag"]
                                and diff >= -spk_corr_parm["lag"]
                            ):
                                if diff < 0:
                                    ind = np.where(
                                        spk_corr_parm["time_bin"]
                                        <= -math.ceil(abs(diff))
                                    )[0][-1]
                                elif diff > 0:
                                    ind = np.where(
                                        spk_corr_parm["time_bin"] >= math.ceil(diff)
                                    )[0][0]
                                # print("diff = {}, bin index = {}".format(diff, spk_corr_parm['time_bin'][ind]))  # for debugging
                                corr_temp[ind] += 1

                    # Make sure the array is symmetrical
                    first_half = np.fliplr(
                        [
                            corr_temp[
                                : int(
                                    (spk_corr_parm["lag"] / spk_corr_parm["bin_size"])
                                )
                            ]
                        ]
                    )[0]
                    second_half = corr_temp[
                        int((spk_corr_parm["lag"] / spk_corr_parm["bin_size"])) + 1 :
                    ]
                    assert np.sum(first_half - second_half) == 0

                    # Normalize correlogram by the total sum (convert to probability density )
                    if normalize:
                        corr_temp /= np.sum(correlogram)

            correlogram[social_context] = corr_temp
        correlogram["parameter"] = spk_corr_parm  # store parameters in the dictionary

        return correlogram

    def jitter_spk_ts(self, shuffle_limit, reproducible=True):
        """
        Add a random temporal jitter to the spike

        Parameters
        ----------
        shuffle_limit : int
            shuffling limit (in ms)
            e.g., If set to 5, any integer values between -5 to 5 drawn from uniform distribution will be added to the spike timestamp
        reproducible : bool
            make the results reproducible by setting the seed as equal to index
        """

        spk_ts_jittered_list = []
        for ind, spk_ts in enumerate(self.spk_ts):
            np.random.seed()
            if reproducible:  # randomization seed
                seed = ind
                np.random.seed(seed)  # make random jitter reproducible
            else:
                seed = np.random.randint(len(self.spk_ts), size=1)
                np.random.seed(seed)  # make random jitter reproducible
            nb_spk = spk_ts.shape[0]
            jitter = np.random.uniform(-shuffle_limit, shuffle_limit, nb_spk)
            spk_ts_jittered_list.append(spk_ts + jitter)
        self.spk_ts_jittered = spk_ts_jittered_list

    def get_jittered_corr(self) -> dict:
        """Get spike correlogram from time-jittered spikes"""
        from collections import defaultdict

        from ..core.parameters import corr_shuffle

        correlogram_jitter = defaultdict(list)

        for iter in range(corr_shuffle["shuffle_iter"]):
            self.jitter_spk_ts(corr_shuffle["shuffle_limit"])
            corr_temp = self.get_correlogram(self.spk_ts_jittered, self.spk_ts_jittered)
            # Combine correlogram from two contexts
            for key, value in corr_temp.items():
                if key != "parameter":
                    try:
                        correlogram_jitter[key].append(value)
                    except:
                        correlogram_jitter[key] = value

        # Convert to array
        for key, value in correlogram_jitter.items():
            correlogram_jitter[key] = np.array(value)

        return correlogram_jitter

    def get_isi(self, add_premotor_spk=False):
        """
        Get inter-spike interval

        Parameters
        ----------
        add_premotor_spk : bool
            Add spikes from the premotor window for calculation
        """

        isi_dict = {}
        list_zip = zip(self.onsets, self.offsets, self.spk_ts)

        if not add_premotor_spk:
            # Include spikes from the pre-motif buffer for calculation
            # Pre-motor spikes are included in spk_list by default
            spk_list = []
            for onset, offset, spks in list_zip:
                onset = np.asarray(list(map(float, onset)))
                offset = np.asarray(list(map(float, offset)))
                spk_list.append(
                    spks[np.where((spks >= onset[0]) & (spks <= offset[-1]))]
                )

        for context1 in set(self.contexts):
            if not add_premotor_spk:
                spk_list_context = [
                    spk_ts
                    for spk_ts, context2 in zip(spk_list, self.contexts)
                    if context2 == context1
                ]
            else:
                spk_list_context = [
                    spk_ts
                    for spk_ts, context2 in zip(self.spk_ts, self.contexts)
                    if context2 == context1
                ]
            isi_dict[context1] = get_isi(spk_list_context)

        return isi_dict

    @property
    def nb_files(self) -> dict:
        """
        Return the number of files per context

        Returns
        -------
        nb_files : dict
            Number of files per context ('U', 'D', 'All')
        """
        nb_files = {}
        nb_files["U"] = len([context for context in self.contexts if context == "U"])
        nb_files["D"] = len([context for context in self.contexts if context == "D"])
        nb_files["All"] = nb_files["U"] + nb_files["D"]

        return nb_files

    def nb_bouts(self, song_note: str) -> dict:
        """
        Return the number of bouts per context

        Parameters
        ----------
        song_note : str
            song motif syllables

        Returns
        -------
        nb_bouts : dict
        """

        from ..core.functions import get_nb_bouts

        nb_bouts = {}
        syllable_list = [
            syllable
            for syllable, context in zip(self.syllables, self.contexts)
            if context == "U"
        ]
        syllables = "".join(syllable_list)
        nb_bouts["U"] = get_nb_bouts(song_note, syllables)

        syllable_list = [
            syllable
            for syllable, context in zip(self.syllables, self.contexts)
            if context == "D"
        ]
        syllables = "".join(syllable_list)
        nb_bouts["D"] = get_nb_bouts(song_note, syllables)
        nb_bouts["All"] = nb_bouts["U"] + nb_bouts["D"]

        return nb_bouts

    def nb_motifs(self, motif: str) -> dict:
        """
        Return the number of motifs per context

        Parameters
        ----------
        motf : str
            Song motif (e.g., 'abcd')

        Returns
        -------
        nb_motifs : dict
        """
        from ..utils.functions import find_str

        nb_motifs = {}
        syllable_list = [
            syllable
            for syllable, context in zip(self.syllables, self.contexts)
            if context == "U"
        ]
        syllables = "".join(syllable_list)
        nb_motifs["U"] = len(find_str(syllables, motif))

        syllable_list = [
            syllable
            for syllable, context in zip(self.syllables, self.contexts)
            if context == "D"
        ]
        syllables = "".join(syllable_list)
        nb_motifs["D"] = len(find_str(syllables, motif))
        nb_motifs["All"] = nb_motifs["U"] + nb_motifs["D"]

        return nb_motifs

    def get_note_info(self, target_note, pre_buffer=0, post_buffer=0):
        """
        Obtain a class object (NoteInfo) for individual note
        spikes will be collected from note onset (+- pre_buffer) to offset (+- post_buffer)

        Parameters
        ----------
        target_note : str
            Get information from this note
        pre_buffer : int
            Amount of time buffer relative to the event onset (e.g., syllable onset)
        post_buffer : int
            Amount of time buffer relative to the event offset (e.g., syllable onset)

        Returns
        -------
        NoteInfo : class object
        """
        from ..utils.functions import find_str

        syllables = "".join(self.syllables)
        onsets = np.hstack(self.onsets)
        offsets = np.hstack(self.offsets)
        durations = np.hstack(self.durations)
        contexts = ""

        for i in range(len(self.contexts)):  # concatenate contexts
            contexts += self.contexts[i] * len(self.syllables[i])

        ind = np.array(find_str(syllables, target_note))  # get note indices
        if not ind.any():  # skil if the note does not exist
            return

        note_onsets = np.asarray(list(map(float, onsets[ind])))
        note_offsets = np.asarray(list(map(float, offsets[ind])))
        note_durations = np.asarray(list(map(float, durations[ind])))
        note_contexts = "".join(np.asarray(list(contexts))[ind])

        # Get the note that immeidately follows
        next_notes = ""
        for i in ind:
            next_notes += syllables[i + 1]

        # Get spike info
        spk_ts = np.hstack(self.spk_ts)
        note_spk_ts_list = []
        for onset, offset in zip(note_onsets, note_offsets):
            note_spk_ts_list.append(
                spk_ts[
                    np.where(
                        (spk_ts >= onset - pre_buffer)
                        & (spk_ts <= offset + post_buffer)
                    )
                ]
            )

        # Organize data into a dictionary
        note_info = {
            "note": target_note,
            "next_notes": next_notes,
            "onsets": note_onsets,
            "offsets": note_offsets,
            "durations": note_durations,
            "contexts": note_contexts,
            "median_dur": np.median(note_durations, axis=0),
            "spk_ts": note_spk_ts_list,
            "path": self.path,  # directory where the data exists
            "pre_buffer": pre_buffer,
            "post_buffer": post_buffer,
        }

        return NoteInfo(note_info)  # return note info

    @property
    def open_folder(self):

        from ..utils.functions import open_folder as _open_folder

        return _open_folder(self.path)


class NoteInfo:
    """
    Class for storing information about a single note syllable and its associated spikes
    """

    def __init__(self, note_dict):

        # Set the dictionary values to class attributes
        for key in note_dict:
            setattr(self, key, note_dict[key])

        # Perform PLW (piecewise linear warping)
        self.spk_ts_warp = self._piecewise_linear_warping()

    def __repr__(self):
        return str([key for key in self.__dict__.keys()])

    def select_index(self, index) -> None:
        """
        Select only the notes with the matching index

        Parameters
        ----------
        index : np.array or list
            Note indices to keep
        """
        if isinstance(index, list):
            index = np.array(index)

        self.contexts = "".join(np.array(list(self.contexts))[index])
        self.onsets, self.offsets, self.durations, self.spk_ts, self.spk_ts_warp = (
            self.onsets[index],
            self.offsets[index],
            self.durations[index],
            self.spk_ts[index],
            self.spk_ts_warp[index],
        )

    def select_context(self, target_context: str, keep_median_duration=True) -> None:
        """
        Select one context

        Parameters
        ----------
        target_context : str
            'U' or 'D'
        keep_median_duration : bool
            Normally medial note duration is calculated using all syllables regardless of the context
            one may prefer to use this median to reduce variability when calculating pcc
            if set False, new median duration will be calculated using the selected notes
        """

        zipped_list = list(
            zip(
                self.contexts,
                self.next_notes,
                self.onsets,
                self.offsets,
                self.durations,
                self.spk_ts,
                self.spk_ts_warp,
            )
        )
        zipped_list = list(
            filter(lambda x: x[0] == target_context, zipped_list)
        )  # filter context
        unzipped_object = zip(*zipped_list)
        (
            self.contexts,
            self.next_notes,
            self.onsets,
            self.offsets,
            self.durations,
            self.spk_ts,
            self.spk_ts_warp,
        ) = list(unzipped_object)
        self.contexts = "".join(self.contexts)
        self.next_notes = "".join(self.next_notes)
        self.onsets = np.array(self.onsets)
        self.offsets = np.array(self.offsets)
        self.durations = np.array(self.durations)
        self.spk_ts = np.array(self.spk_ts)
        self.spk_ts_warp = np.array(self.spk_ts_warp)

        if not keep_median_duration:
            self.median_dur = np.median(self.median_dur, axis=0)

    def get_entropy(self, normalize=True, mode="spectral"):
        """
        Calculate syllable entropy from all renditions and get the average
        Two versions : spectro-temporal entropy & spectral entropy
        """
        from ..core.functions import get_spectral_entropy, get_spectrogram
        from ..core.parameters import nb_note_crit
        from ..utils.functions import find_str

        entropy_mean = {}
        entropy_var = {}
        audio = AudioData(self.path)

        for context in ["U", "D"]:

            se_mean_arr = np.array([], dtype=np.float32)
            se_var_arr = np.array([], dtype=np.float32)
            ind = np.array(find_str(self.contexts, context))

            if ind.shape[0] >= nb_note_crit:
                for (start, end) in zip(self.onsets[ind], self.offsets[ind]):
                    timestamp, data = audio.extract([start, end])  # audio object
                    _, spect, _ = get_spectrogram(timestamp, data, audio.sample_rate)
                    se = get_spectral_entropy(spect, normalize=normalize, mode=mode)
                    if isinstance(se, dict):
                        se_mean_arr = np.append(
                            se_mean_arr, se["mean"]
                        )  # spectral entropy averaged over time bins per rendition
                        se_var_arr = np.append(
                            se_var_arr, se["var"]
                        )  # spectral entropy variance per rendition
                    else:
                        se_mean_arr = np.append(
                            se_mean_arr, se
                        )  # spectral entropy time-resolved
                entropy_mean[context] = round(se_mean_arr.mean(), 3)
                entropy_var[context] = round(se_var_arr.mean(), 5)
        if mode == "spectro_temporal":
            return entropy_mean, entropy_var
        else:  # spectral entropy (does not have entropy variance)
            return entropy_mean

    def _piecewise_linear_warping(self):
        """Perform piecewise linear warping per note"""
        import copy

        note_spk_ts_warp_list = []

        for onset, duration, spk_ts in zip(self.onsets, self.durations, self.spk_ts):
            spk_ts_new = copy.deepcopy(spk_ts)
            ratio = self.median_dur / duration
            origin = 0

            spk_ts_temp, ind = spk_ts[spk_ts >= onset], np.where(spk_ts >= onset)

            spk_ts_temp = ((ratio * ((spk_ts_temp - onset))) + origin) + onset
            np.put(
                spk_ts_new, ind, spk_ts_temp
            )  # replace original spk timestamps with warped timestamps
            note_spk_ts_warp_list.append(spk_ts_new)

        return note_spk_ts_warp_list

    def get_note_peth(
        self,
        time_warp=True,
        shuffle=False,
        pre_evt_buffer=None,
        duration=None,
        bin_size=None,
        nb_bins=None,
    ):
        """
        Get peri-event time histograms for single syllable

        Parameters
        ----------
        time_warp : perform piecewise linear transform
        shuffle : add jitter to spike timestamps
        duration : duration of the peth
        bin_size : size of single bin (in ms) (take values from peth_parm by default)
        nb_bins : number of time bins (take values from peth_parm by default)

        Returns
        -------
        PethInfo : class object
        """
        peth_dict = {}

        if shuffle:
            peth, time_bin, peth_parm = get_peth(
                self.onsets,
                self.spk_ts_jittered,
                pre_evt_buffer=pre_evt_buffer,
                duration=duration,
                bin_size=bin_size,
                nb_bins=nb_bins,
            )
        else:
            if time_warp:  # peth calculated from time-warped spikes by default
                # peth, time_bin = get_note_peth(self.onsets, self.spk_ts_warp, self.median_durations.sum())  # truncated version to fit the motif duration
                peth, time_bin, peth_parm = get_peth(
                    self.onsets,
                    self.spk_ts_warp,
                    pre_evt_buffer=pre_evt_buffer,
                    duration=duration,
                    bin_size=bin_size,
                    nb_bins=nb_bins,
                )
            else:
                peth, time_bin, peth_parm = get_peth(
                    self.onsets,
                    self.spk_ts,
                    pre_evt_buffer=pre_evt_buffer,
                    duration=duration,
                    bin_size=bin_size,
                    nb_bins=nb_bins,
                )

        peth_dict["peth"] = peth
        peth_dict["time_bin"] = time_bin
        peth_dict["parameters"] = peth_parm
        peth_dict["contexts"] = self.contexts
        peth_dict["median_duration"] = self.median_dur
        return PethInfo(peth_dict)  # return peth class object for further analysis

    def jitter_spk_ts(self, shuffle_limit):
        """
        Add a random temporal jitter to the spike
        This version limit the jittered timestamp within the motif window
        """

        from ..core.parameters import pre_motor_win_size

        spk_ts_jittered_list = []
        list_zip = zip(self.onsets, self.offsets, self.spk_ts)
        for ind, (onset, offset, spk_ts) in enumerate(list_zip):

            # Find motif onset & offset
            onset = float(onset) - pre_motor_win_size  # start from the premotor window

            jittered_spk = np.array([], dtype=np.float32)

            for spk_ind, spk in enumerate(spk_ts):
                while True:
                    jitter = np.random.uniform(-shuffle_limit, shuffle_limit, 1)
                    new_spk = spk + jitter
                    if onset < new_spk < offset:
                        jittered_spk = np.append(jittered_spk, spk + jitter)
                        break

            spk_ts_jittered_list.append(jittered_spk)
        self.spk_ts_jittered = spk_ts_jittered_list

    @property
    def nb_note(self) -> dict:
        """Return number of notes per context"""
        from ..utils.functions import find_str

        nb_note = {}
        for context in ["U", "D"]:
            nb_note[context] = len(find_str(self.contexts, context))
        return nb_note

    @property
    def mean_fr(self) -> dict:
        """Return mean firing rates for the note (includes pre-motor window) per context"""
        from ..core.parameters import nb_note_crit, pre_motor_win_size
        from ..utils.functions import find_str

        note_spk = {}
        note_fr = {}
        for context1 in ["U", "D"]:
            if self.nb_note[context1] >= nb_note_crit:
                note_spk[context1] = sum(
                    [
                        len(spk)
                        for context2, spk in zip(self.contexts, self.spk_ts)
                        if context2 == context1
                    ]
                )
                note_fr[context1] = round(
                    note_spk[context1]
                    / (
                        (
                            self.durations[find_str(self.contexts, context1)]
                            + pre_motor_win_size
                        ).sum()
                        / 1e3
                    ),
                    3,
                )
            else:
                note_fr[context1] = np.nan
        return note_fr


class MotifInfo(ClusterInfo):
    """
    Class object for motif information
    child class of ClusterInfo
    """

    def __init__(
        self, path, channel_nb, unit_nb, motif, format="rhd", *name, update=False
    ):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        self.motif = motif
        if name:
            self.name = name[0]
        else:
            self.name = str(self.path)

        # Load motif info
        file_name = self.path / "MotifInfo_{}_Cluster{}.npy".format(
            self.channel_nb, self.unit_nb
        )
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file
            motif_info = self._load_motif()
            # Save info dict as a numpy object
            np.save(file_name, motif_info)
        else:
            motif_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])

        # Delete un-used attributes
        self._delete_attr()

    def _delete_attr(self):
        """Delete un-used attributes/methods inheritied from the parent class"""
        delattr(self, "spk_wf")
        delattr(self, "nb_spk")
        delattr(self, "file_start")
        delattr(self, "file_end")

    def _load_motif(self):
        """Load motif info"""
        from ..core.parameters import peth_parm
        from ..utils.functions import find_str

        # Store values here
        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
        context_list = []

        list_zip = zip(
            self.files,
            self.spk_ts,
            self.onsets,
            self.offsets,
            self.syllables,
            self.contexts,
        )

        for file, spks, onsets, offsets, syllables, context in list_zip:
            print("Loading... " + file)
            onsets = onsets.tolist()
            offsets = offsets.tolist()

            # Find motifs
            motif_ind = find_str(syllables, self.motif)

            # Get syllable, analysis time stamps
            for ind in motif_ind:
                # start (first syllable) and stop (last syllable) index of a motif
                start_ind = ind
                stop_ind = ind + len(self.motif) - 1

                motif_onset = float(onsets[start_ind])
                motif_offset = float(offsets[stop_ind])
                # Includes pre-motor spikes
                motif_spk = spks[
                    np.where(
                        (spks >= motif_onset - peth_parm["buffer"])
                        & (spks <= motif_offset)
                    )
                ]
                onsets_in_motif = onsets[
                    start_ind : stop_ind + 1
                ]  # list of motif onset timestamps
                offsets_in_motif = offsets[
                    start_ind : stop_ind + 1
                ]  # list of motif offset timestamps

                file_list.append(file)
                spk_list.append(motif_spk)
                duration_list.append(motif_offset - motif_onset)
                onset_list.append(onsets_in_motif)
                offset_list.append(offsets_in_motif)
                syllable_list.append(syllables[start_ind : stop_ind + 1])
                context_list.append(context)

        # Organize event-related info into a single dictionary object
        motif_info = {
            "files": file_list,
            "spk_ts": spk_list,
            "onsets": onset_list,
            "offsets": offset_list,
            "durations": duration_list,  # this is motif durations
            "syllables": syllable_list,
            "contexts": context_list,
            "parameter": peth_parm,
        }

        # Set the dictionary values to class attributes
        for key in motif_info:
            setattr(self, key, motif_info[key])

        # Get duration
        note_duration_list, median_duration_list = self.get_note_duration()
        self.note_durations = note_duration_list
        self.median_durations = median_duration_list
        motif_info["note_durations"] = note_duration_list
        motif_info["median_durations"] = median_duration_list

        # Get PLW (piecewise linear warping)
        spk_ts_warp_list = self.piecewise_linear_warping()
        # self.spk_ts_warp = spk_ts_warp_list
        motif_info["spk_ts_warp"] = spk_ts_warp_list

        return motif_info

    def select_context(self, target_context: str, keep_median_duration=True) -> None:
        """
        Select one context

        Parameters
        ----------
        target_context : str
            'U' or 'D'
        keep_median_duration : bool
            Normally medial note duration is calculated using all syllables regardless of the context.

            One may prefer to use this median to reduce variability when calculating pcc.

            IF set False, new median duration will be calculated using the selected notes.
        """

        zipped_list = list(
            zip(
                self.contexts,
                self.files,
                self.onsets,
                self.offsets,
                self.durations,
                self.spk_ts,
                self.spk_ts_warp,
                self.note_durations,
            )
        )

        zipped_list = list(
            filter(lambda x: x[0] == target_context, zipped_list)
        )  # filter context
        unzipped_object = zip(*zipped_list)
        (
            self.contexts,
            self.files,
            self.onsets,
            self.offsets,
            self.durations,
            self.spk_ts,
            self.spk_ts_warp,
            self.note_durations,
        ) = list(unzipped_object)

        if not keep_median_duration:
            _, self.median_durations = self.get_note_duration()

    def get_note_duration(self):
        """
        Calculate note & gap duration per motif
        """

        note_durations = np.empty((len(self), len(self.motif) * 2 - 1))

        list_zip = zip(self.onsets, self.offsets)

        for motif_ind, (onset, offset) in enumerate(list_zip):

            # Convert from string to array of floats
            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))

            # Calculate note & interval duration
            timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
            timestamp = sum(timestamp, [])

            for i in range(len(timestamp) - 1):
                note_durations[motif_ind, i] = timestamp[i + 1] - timestamp[i]

        # Get median duration
        median_durations = np.median(note_durations, axis=0)

        return note_durations, median_durations

    def piecewise_linear_warping(self):
        """
        Performs piecewise linear warping on raw analysis timestamps
        Based on each median note and gap durations
        """
        import copy

        from ..utils.functions import extract_ind

        spk_ts_warped_list = []
        list_zip = zip(self.note_durations, self.onsets, self.offsets, self.spk_ts)

        for motif_ind, (durations, onset, offset, spk_ts) in enumerate(
            list_zip
        ):  # per motif

            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))
            # Make a deep copy of spk_ts so as to make it modification won't affect the original
            spk_ts_new = copy.deepcopy(spk_ts)

            # Calculate note & interval duration
            timestamp = [[onset, offset] for onset, offset in zip(onset, offset)]
            timestamp = sum(timestamp, [])

            for i in range(0, len(self.median_durations)):
                ratio = self.median_durations[i] / durations[i]
                diff = timestamp[i] - timestamp[0]
                if i == 0:
                    origin = 0
                else:
                    origin = sum(self.median_durations[:i])

                # Add spikes from motif
                ind, spk_ts_temp = extract_ind(spk_ts, [timestamp[i], timestamp[i + 1]])
                spk_ts_temp = (
                    (ratio * ((spk_ts_temp - timestamp[0]) - diff)) + origin
                ) + timestamp[0]
                # spk_ts_new = np.append(spk_ts_new, spk_ts_temp)
                np.put(
                    spk_ts_new, ind, spk_ts_temp
                )  # replace original spk timestamps with warped timestamps

            spk_ts_warped_list.append(spk_ts_new)
        return spk_ts_warped_list

    def get_mean_fr(self, add_pre_motor=False):
        """
        Calculate mean firing rates during motif

        Parameters
        ----------
        add_pre_motor : bool
            Set True if you want to include spikes from the pre-motor window for calculating firing rates
            (False by default)

        """
        from ..core.parameters import peth_parm

        fr_dict = {}
        motif_spk_list = []
        list_zip = zip(self.onsets, self.offsets, self.spk_ts)

        # Make sure spikes from the pre-motif buffer is not included in calculation
        for onset, offset, spks in list_zip:

            onset = np.asarray(list(map(float, onset)))
            offset = np.asarray(list(map(float, offset)))
            if add_pre_motor:
                motif_spk_list.append(
                    spks[
                        np.where(
                            (spks >= (onset[0] - peth_parm["buffer"]))
                            & (spks <= offset[-1])
                        )
                    ]
                )
            else:
                motif_spk_list.append(
                    spks[np.where((spks >= onset[0]) & (spks <= offset[-1]))]
                )

        for context1 in set(self.contexts):
            nb_spk = sum(
                [
                    len(spk)
                    for spk, context2 in zip(motif_spk_list, self.contexts)
                    if context2 == context1
                ]
            )
            if add_pre_motor:
                total_duration = sum(
                    [
                        duration + peth_parm["buffer"]
                        for duration, context2 in zip(self.durations, self.contexts)
                        if context2 == context1
                    ]
                )
            else:
                total_duration = sum(
                    [
                        duration
                        for duration, context2 in zip(self.durations, self.contexts)
                        if context2 == context1
                    ]
                )

            mean_fr = nb_spk / (total_duration / 1e3)
            fr_dict[context1] = round(mean_fr, 3)
        # print("mean_fr added")
        self.mean_fr = fr_dict

    def jitter_spk_ts(self, shuffle_limit: int, **kwargs):
        """
        Add a random temporal jitter to the spike
        This version limit the jittered timestamp within the motif window
        """

        from ..core.parameters import pre_motor_win_size

        spk_ts_jittered_list = []
        list_zip = zip(self.onsets, self.offsets, self.spk_ts)
        for ind, (onset, offset, spk_ts) in enumerate(list_zip):

            # Find motif onset & offset
            onset = (
                float(onset[0]) - pre_motor_win_size
            )  # start from the premotor window
            offset = float(offset[-1])

            jittered_spk = np.array([], dtype=np.float32)

            for spk_ind, spk in enumerate(spk_ts):
                while True:
                    jitter = np.random.uniform(-shuffle_limit, shuffle_limit, 1)
                    new_spk = spk + jitter
                    if onset < new_spk < offset:
                        jittered_spk = np.append(jittered_spk, spk + jitter)
                        break

            spk_ts_jittered_list.append(jittered_spk)
        self.spk_ts_jittered = spk_ts_jittered_list

    def get_peth(self, time_warp=True, shuffle=False):
        """
        Get peri-event time histogram & raster during song motif

        Parameters
        ----------
        time_warp : bool
            perform piecewise linear transform
        shuffle : bool
            add jitter to spike timestamps

        Returns
        -------
        PethInfo : class object
        """
        peth_dict = {}

        if shuffle:  # Get peth with shuffled (jittered) spikes
            peth, time_bin, peth_parm = get_peth(self.onsets, self.spk_ts_jittered)
        else:
            if time_warp:  # peth calculated from time-warped spikes by default
                # peth, time_bin = get_note_peth(self.onsets, self.spk_ts_warp, self.median_durations.sum())  # truncated version to fit the motif duration
                peth, time_bin, peth_parm = get_peth(self.onsets, self.spk_ts_warp)
            else:
                peth, time_bin, peth_parm = get_peth(self.onsets, self.spk_ts)

        peth_parm.pop("time_bin")
        peth_parm.pop("nb_bins")
        peth_dict["peth"] = peth
        peth_dict["time_bin"] = time_bin
        peth_dict["parameters"] = peth_parm
        peth_dict["contexts"] = self.contexts
        peth_dict["median_duration"] = self.median_durations.sum()
        return PethInfo(peth_dict)  # return peth class object for further analysis

    def __len__(self):
        return len(self.files)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    @property
    def open_folder(self):
        """Open the data folder"""
        from ..utils.functions import open_folder

        open_folder(self.path)

    def _print_name(self):
        print("")
        print("Load motif {self.name}".format(self=self))


class PethInfo:
    def __init__(self, peth_dict: dict):
        """
        Class object for peri-event time histogram (PETH)

        Parameters
        ----------
        peth_dict : dict
            "peth" : array  (nb of trials (motifs) x time bins), numbers indicate analysis counts in that bin
            "contexts" : list of strings, social contexts
        """

        # Set the dictionary values to class attributes
        for key in peth_dict:
            setattr(self, key, peth_dict[key])

        # Get conditional peth, fr, spike counts
        peth_dict = {}
        peth_dict["All"] = self.peth
        for context in set(self.contexts):
            if type(self.contexts) == str:
                self.contexts = list(self.contexts)
            ind = np.array(self.contexts) == context
            peth_dict[context] = self.peth[ind, :]
        self.peth = peth_dict

    def get_fr(self, gaussian_std=None, smoothing=True):
        """
        Get trials-by-trial firing rates by default

        Parameters
        ----------
        gaussian_std : int
            gaussian smoothing parameter. If not specified, read from analysis.parameters
        smoothing : bool
            performs gaussian smoothing on the firing rates
        """

        # if duration:
        #     ind = (((0 - peth_parm['buffer']) <= time_bin) & (time_bin <= duration))
        #     peth = peth[:, ind]
        #     time_bin = time_bin[ind]

        from scipy.ndimage import gaussian_filter1d

        from ..core.parameters import gauss_std, nb_note_crit, peth_parm

        if (
            not gaussian_std
        ):  # if not specified, get the value fromm analysis.parameters
            gaussian_std = gauss_std

        # Get trial-by-trial firing rates
        fr_dict = {}
        for k, v in self.peth.items():  # loop through different conditions in peth dict
            if v.shape[0] >= nb_note_crit:
                fr = v / (peth_parm["bin_size"] / 1e3)  # in Hz

                if smoothing:  # Gaussian smoothing
                    fr = gaussian_filter1d(fr, gaussian_std)

                # Truncate values outside the range
                ind = ((0 - peth_parm["buffer"]) <= self.time_bin) & (
                    self.time_bin <= self.median_duration
                )
                fr = fr[:, ind]
                fr_dict[k] = fr
        self.fr = fr_dict
        self.time_bin = self.time_bin[ind]

        # Get mean firing rates
        mean_fr_dict = {}
        for context, fr in self.fr.items():
            fr = np.mean(fr, axis=0)
            mean_fr_dict[context] = fr
        if smoothing:
            mean_fr_dict["gauss_std"] = gauss_std
        self.mean_fr = mean_fr_dict

    def get_pcc(self):
        """Get pairwise cross-correlation"""
        from ..core.parameters import nb_note_crit

        pcc_dict = {}
        for k, v in self.fr.items():  # loop through different conditions in peth dict
            if k != "All":
                if v.shape[0] >= nb_note_crit:
                    pcc = get_pcc(v)
                    pcc_dict[k] = pcc
        self.pcc = pcc_dict

    def get_fr_cv(self):
        """Get coefficient of variation (CV) of firing rates"""
        if not self.mean_fr:
            self.get_fr()

        fr_cv = {}
        for (
            context,
            fr,
        ) in self.mean_fr.items():  # loop through different conditions in peth dict
            if context in ["U", "D"]:
                fr_cv[context] = round(fr.std(axis=0) / fr.mean(axis=0), 3)
        return fr_cv

    def get_sparseness(self, bin_size=None):
        """
        Get sparseness index

        Parameters
        ----------
        bin_size : int
            By default, it uses the same time bin size used in peth calculation (in ms)

        Returns
        -------
        sparseness : dict
        """

        import math

        from ..core.parameters import gauss_std, nb_note_crit

        mean_fr = dict()
        sparseness = dict()

        if bin_size != None and bin_size != self.parameters["bin_size"]:
            for context, peth in self.peth.items():
                if context == "All":
                    continue
                new_peth = np.empty([peth.shape[0], 0])
                nb_bins = math.ceil(peth.shape[1] / bin_size)
                bin_ind = 0
                start_ind = 0
                end_ind = 0 + bin_size

                while bin_ind < nb_bins:
                    if end_ind > peth.shape[1]:
                        end_ind = peth.shape[1]
                    # print(start_ind, end_ind)
                    peth_bin = (
                        peth[:, start_ind:end_ind].sum(axis=1).reshape(peth.shape[0], 1)
                    )
                    new_peth = np.append(new_peth, peth_bin, axis=1)
                    start_ind += bin_size
                    end_ind += bin_size
                    bin_ind += 1

                fr = new_peth / (bin_size / 1e3)  # in Hz
                mean_fr[context] = np.mean(fr, axis=0)

        else:
            mean_fr = self.mean_fr

        # Calculate sparseness
        for context, fr in mean_fr.items():
            if context not in ["U", "D"]:
                continue
            norm_fr = fr / np.sum(fr)
            sparseness[context] = round(
                1 + (np.nansum(norm_fr * np.log10(norm_fr)) / np.log10(len(norm_fr))), 3
            )

        return sparseness

    def get_spk_count(self):
        """
        Calculate the number of spikes within a specified time window
        """
        from ..core.parameters import peth_parm, spk_count_parm

        win_size = spk_count_parm["win_size"]
        spk_count_dict = {}
        fano_factor_dict = {}
        spk_count_cv_dict = {}

        for k, v in self.peth.items():  # loop through different conditions in peth dict
            spk_arr = np.empty((v.shape[0], 0), int)  # (renditions x time bins)
            if k != "All":  # skip all trials
                win_inc = 0
                for i in range(v.shape[1] - win_size):
                    count = v[:, i : win_size + win_inc].sum(axis=1)
                    # print(f"from {i} to {win_size + win_inc}, count = {count}")
                    spk_arr = np.append(spk_arr, np.array([count]).transpose(), axis=1)
                    win_inc += 1
                # Truncate values outside the range
                ind = ((0 - peth_parm["buffer"]) <= self.time_bin) & (
                    self.time_bin <= self.median_duration
                )
                spk_arr = spk_arr[:, : ind.shape[0]]

                spk_count = spk_arr.sum(axis=0)
                fano_factor = spk_arr.var(axis=0) / spk_arr.mean(
                    axis=0
                )  # per time window (across renditions) (renditions x time window)
                spk_count_cv = spk_count.std(axis=0) / spk_count.mean(
                    axis=0
                )  # cv across time (single value)

                # store values in a dictionary
                spk_count_dict[k] = spk_count
                fano_factor_dict[k] = fano_factor
                spk_count_cv_dict[k] = round(spk_count_cv, 3)

        self.spk_count = spk_count_dict
        self.fano_factor = fano_factor_dict
        self.spk_count_cv = spk_count_cv_dict

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class BoutInfo(ClusterInfo):
    """
    Get song & spike information for a song bout
    Child class of ClusterInfo
    """

    def __init__(
        self, path, channel_nb, unit_nb, song_note, format="rhd", *name, update=False
    ):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        self.song_note = song_note

        if name:
            self.name = name[0]
        else:
            self.name = str(self.path)

        # Load bout info
        file_name = self.path / "BoutInfo_{}_Cluster{}.npy".format(
            self.channel_nb, self.unit_nb
        )
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file
            bout_info = self._load_bouts()
            # Save info dict as a numpy object
            np.save(file_name, bout_info)
        else:
            bout_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in bout_info:
            setattr(self, key, bout_info[key])

    def _print_name(self):
        print("")
        print("Load bout {self.name}".format(self=self))

    def __len__(self):
        return len(self.files)

    def _load_bouts(self):
        # Store values here
        from ..utils.functions import find_str

        file_list = []
        spk_list = []
        onset_list = []
        offset_list = []
        syllable_list = []
        duration_list = []
        context_list = []

        list_zip = zip(
            self.files,
            self.spk_ts,
            self.onsets,
            self.offsets,
            self.syllables,
            self.contexts,
        )

        for file, spks, onsets, offsets, syllables, context in list_zip:

            bout_ind = find_str(syllables, "*")

            for ind in range(len(bout_ind)):
                if ind == 0:
                    start_ind = 0
                else:
                    start_ind = bout_ind[ind - 1] + 1
                stop_ind = bout_ind[ind] - 1
                # breakpoint()
                bout_onset = float(onsets[start_ind])
                bout_offset = float(offsets[stop_ind])

                bout_spk = spks[np.where((spks >= bout_onset) & (spks <= bout_offset))]
                onsets_in_bout = onsets[
                    start_ind : stop_ind + 1
                ]  # list of bout onset timestamps
                offsets_in_bout = offsets[
                    start_ind : stop_ind + 1
                ]  # list of bout offset timestamps

                file_list.append(file)
                spk_list.append(bout_spk)
                duration_list.append(bout_offset - bout_onset)
                onset_list.append(onsets_in_bout)
                offset_list.append(offsets_in_bout)
                syllable_list.append(syllables[start_ind : stop_ind + 1])
                context_list.append(context)

        # Organize event-related info into a single dictionary object
        bout_info = {
            "files": file_list,
            "spk_ts": spk_list,
            "onsets": onset_list,
            "offsets": offset_list,
            "durations": duration_list,  # this is bout durations
            "syllables": syllable_list,
            "contexts": context_list,
        }
        return bout_info

    def plot(self):
        # TODO: this function needs revision
        import warnings

        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        from ..core.parameters import bout_buffer, bout_color, freq_range
        from ..db.load import DBInfo, ProjectLoader
        from ..utils import save
        from ..utils.draw import remove_right_top

        warnings.filterwarnings("ignore")

        # Parameters
        save_fig = False
        update = False
        dir_name = "RasterBouts"
        fig_ext = ".png"  # .png or .pdf
        font_size = 12  # figure font size
        rec_yloc = 0.05
        rect_height = 0.2
        text_yloc = 1  # text height
        nb_row = 13
        nb_col = 1
        tick_length = 1
        tick_width = 1

        # Load database
        db = ProjectLoader().load_db()
        # SQL statementwa
        # query = "SELECT * FROM cluster"
        # query = "SELECT * FROM cluster WHERE ephysOK"
        query = "SELECT * FROM cluster WHERE id = 12"
        db.execute(query)

        # Loop through db
        for row in db.cur.fetchall():

            # Load cluster info from db
            cluster_db = DBInfo(row)
            name, path = cluster_db.load_cluster_db()
            unit_nb = int(cluster_db.unit[-2:])
            channel_nb = int(cluster_db.channel[-2:])
            format = cluster_db.format

            ci = ClusterInfo(
                path, channel_nb, unit_nb, format, name, update=update
            )  # cluster object
            bi = BoutInfo(
                path,
                channel_nb,
                unit_nb,
                cluster_db.songNote,
                format,
                name,
                update=update,
            )  # bout object

            list_zip = zip(
                bi.files, bi.spk_ts, bi.onsets, bi.offsets, bi.syllables, bi.contexts
            )

            for bout_ind, (
                file,
                spks,
                onsets,
                offsets,
                syllables,
                context,
            ) in enumerate(list_zip):

                # Convert from string to array of floats
                onsets = np.asarray(list(map(float, onsets)))
                offsets = np.asarray(list(map(float, offsets)))
                spks = spks - onsets[0]

                # bout start and end
                start = onsets[0] - bout_buffer
                end = offsets[-1] + bout_buffer
                duration = offsets[-1] - onsets[0]

                # Get spectrogram
                audio = AudioData(path, update=update).extract(
                    [start, end]
                )  # audio object
                audio.spectrogram()
                audio.spect_time = audio.spect_time - audio.spect_time[0] - bout_buffer

                # Plot figure
                fig = plt.figure(figsize=(8, 7))
                fig.tight_layout()
                fig_name = f"{file} - Bout # {bout_ind}"
                print("Processing... " + fig_name)
                fig.suptitle(fig_name, y=0.95)

                # Plot spectrogram
                ax_spect = plt.subplot2grid(
                    (nb_row, nb_col), (2, 0), rowspan=2, colspan=1
                )
                ax_spect.pcolormesh(
                    audio.spect_time,
                    audio.spect_freq,
                    audio.spect,  # data
                    cmap="hot_r",
                    norm=colors.SymLogNorm(
                        linthresh=0.05, linscale=0.03, vmin=0.5, vmax=100
                    ),
                )

                remove_right_top(ax_spect)
                ax_spect.set_ylim(freq_range[0], freq_range[1])
                ax_spect.set_ylabel("Frequency (Hz)", fontsize=font_size)
                plt.yticks(freq_range, [str(freq_range[0]), str(freq_range[1])])
                plt.setp(ax_spect.get_xticklabels(), visible=False)
                plt.xlim([audio.spect_time[0] - 100, audio.spect_time[-1] + 100])

                # Plot syllable duration
                ax_syl = plt.subplot2grid(
                    (nb_row, nb_col), (1, 0), rowspan=1, colspan=1, sharex=ax_spect
                )
                note_dur = offsets - onsets  # syllable duration
                onsets -= onsets[0]  # start from 0
                offsets = onsets + note_dur

                # Mark syllables
                for i, syl in enumerate(syllables):
                    rectangle = plt.Rectangle(
                        (onsets[i], rec_yloc),
                        note_dur[i],
                        rect_height,
                        linewidth=1,
                        alpha=0.5,
                        edgecolor="k",
                        facecolor=bout_color[syl],
                    )
                    ax_syl.add_patch(rectangle)
                    ax_syl.text(
                        (onsets[i] + (offsets[i] - onsets[i]) / 2),
                        text_yloc,
                        syl,
                        size=font_size,
                    )
                ax_syl.axis("off")

                # Plot song amplitude
                audio.data = stats.zscore(audio.data)
                audio.timestamp = audio.timestamp - audio.timestamp[0] - bout_buffer
                ax_amp = plt.subplot2grid(
                    (nb_row, nb_col), (4, 0), rowspan=2, colspan=1, sharex=ax_spect
                )
                ax_amp.plot(audio.timestamp, audio.data, "k", lw=0.1)
                ax_amp.axis("off")

                # Plot rasters
                ax_raster = plt.subplot2grid(
                    (nb_row, nb_col), (6, 0), rowspan=2, colspan=1, sharex=ax_spect
                )
                # spks2 = spks - start -peth_parm['buffer'] -peth_parm['buffer']
                ax_raster.eventplot(
                    spks,
                    colors="k",
                    lineoffsets=0.5,
                    linelengths=tick_length,
                    linewidths=tick_width,
                    orientation="horizontal",
                )
                ax_raster.axis("off")

                # Plot raw neural data
                nd = NeuralData(path, channel_nb, format, update=update).extract(
                    [start, end]
                )  # raw neural data
                nd.timestamp = nd.timestamp - nd.timestamp[0] - bout_buffer
                ax_nd = plt.subplot2grid(
                    (nb_row, nb_col), (8, 0), rowspan=2, colspan=1, sharex=ax_spect
                )
                ax_nd.plot(nd.timestamp, nd.data, "k", lw=0.5)

                # Add a scale bar
                plt.plot(
                    [ax_nd.get_xlim()[0] + 50, ax_nd.get_xlim()[0] + 50],
                    [-250, 250],
                    "k",
                    lw=3,
                )  # for amplitude
                plt.text(
                    ax_nd.get_xlim()[0] - (bout_buffer / 2), -200, "500 µV", rotation=90
                )
                plt.subplots_adjust(wspace=0, hspace=0)
                remove_right_top(ax_nd)
                ax_nd.spines["left"].set_visible(False)
                plt.yticks([], [])
                ax_nd.set_xlabel("Time (ms)")

                # Save results
                if save_fig:
                    save_path = save.make_dir(
                        ProjectLoader().path / "Analysis", "RasterBouts"
                    )
                    save.save_fig(fig, save_path, fig_name, fig_ext=fig_ext)
                else:
                    plt.show()

        print("Done!")


class BaselineInfo(ClusterInfo):
    def __init__(self, path, channel_nb, unit_nb, format="rhd", *name, update=False):
        super().__init__(path, channel_nb, unit_nb, format, *name, update=False)

        from ..core.parameters import baseline
        from ..utils.functions import find_str

        if name:
            self.name = name[0]
        else:
            self.name = str(self.path)

        # Load baseline info
        file_name = self.path / "BaselineInfo_{}_Cluster{}.npy".format(
            self.channel_nb, self.unit_nb
        )
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file

            # Store values in here
            file_list = []
            spk_list = []
            nb_spk_list = []
            duration_list = []
            context_list = []
            baseline_info = {}

            list_zip = zip(
                self.files,
                self.spk_ts,
                self.file_start,
                self.onsets,
                self.offsets,
                self.syllables,
                self.contexts,
            )

            for file, spks, file_start, onsets, offsets, syllables, context in list_zip:

                bout_ind_list = find_str(syllables, "*")
                bout_ind_list.insert(0, -1)  # start from the first index

                for bout_ind in bout_ind_list:
                    # print(bout_ind)
                    if (
                        bout_ind == len(syllables) - 1
                    ):  # skip if * indicates the end syllable
                        continue

                    baseline_onset = (
                        float(onsets[bout_ind + 1])
                        - baseline["time_buffer"]
                        - baseline["time_win"]
                    )

                    if bout_ind > 0 and baseline_onset < float(
                        offsets[bout_ind - 1]
                    ):  # skip if the baseline starts before the offset of the previous syllable
                        continue

                    if baseline_onset < file_start:
                        baseline_onset = file_start

                    baseline_offset = (
                        float(onsets[bout_ind + 1]) - baseline["time_buffer"]
                    )

                    if (
                        baseline_offset - baseline_onset < 0
                    ):  # skip if there's not enough baseline period at the start of a file
                        continue

                    if baseline_onset > baseline_offset:
                        print(
                            "start time ={} to end time = {}".format(
                                baseline_onset, baseline_offset
                            )
                        )

                    baseline_spk = spks[
                        np.where((spks >= baseline_onset) & (spks <= baseline_offset))
                    ]

                    file_list.append(file)
                    spk_list.append(baseline_spk)
                    nb_spk_list.append(len(baseline_spk))
                    duration_list.append(
                        (baseline_offset - baseline_onset)
                    )  # convert to seconds for calculating in Hz
                    context_list.append(context)

            baseline_info = {
                "files": file_list,
                "spk_ts": spk_list,
                "nb_spk": nb_spk_list,
                "durations": duration_list,
                "contexts": context_list,
                "parameter": baseline,
            }
            # Save baseline_info as a numpy object
            np.save(file_name, baseline_info)

        else:
            baseline_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in baseline_info:
            setattr(self, key, baseline_info[key])

    def _print_name(self):
        print("")
        print("Load baseline {self.name}".format(self=self))

    def get_correlogram(self, ref_spk_list, target_spk_list, normalize=False):
        """
        Override the parent method

        Combine correlogram from undir and dir since no contextual differentiation is needed in baseline
        """

        from ..core.parameters import spk_corr_parm

        correlogram_all = super().get_correlogram(
            ref_spk_list, target_spk_list, normalize=False
        )
        correlogram = np.zeros(len(spk_corr_parm["time_bin"]))

        # Combine correlogram from two contexts
        for key, value in correlogram_all.items():
            if key in ["U", "D"]:
                correlogram += value

        return correlogram  # return class object for further analysis

    def get_jittered_corr(self) -> np.ndarray:
        """Get spike correlogram from time-jittered spikes"""
        from ..core.parameters import corr_shuffle

        correlogram_jitter = []

        for iter in range(corr_shuffle["shuffle_iter"]):
            self.jitter_spk_ts(corr_shuffle["shuffle_limit"])
            corr_temp = self.get_correlogram(self.spk_ts_jittered, self.spk_ts_jittered)
            correlogram_jitter.append(corr_temp)

        return np.array(correlogram_jitter)

    def get_isi(self):
        """Get inter-spike interval"""
        return get_isi(self.spk_ts)

    @property
    def mean_fr(self):
        """Mean firing rates"""
        nb_spk = sum([len(spk_ts) for spk_ts in self.spk_ts])
        total_duration = sum(self.durations)
        mean_fr = nb_spk / (total_duration / 1e3)
        return round(mean_fr, 3)

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class AudioData:
    """
    Create an object that has concatenated audio signal and its timestamps

    Get all data by default; specify time range if needed
    """

    def __init__(self, path, format=".wav", update=False):
        from ..core.load import load_audio

        self.path = path
        self.format = format

        file_name = self.path / "AudioData.npy"
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file
            audio_info = load_audio(self.path, self.format)
        else:
            audio_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in audio_info:
            setattr(self, key, audio_info[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    @property
    def open_folder(self):

        from ..utils.functions import open_folder as _open_folder

        return _open_folder(self.path)

    def extract(self, time_range: list):
        """
        Extracts data from the specified range

        Parameters
        ----------
        time_range : list
        """

        start = time_range[0]
        end = time_range[-1]

        ind = np.where((self.timestamp >= start) & (self.timestamp <= end))
        return self.timestamp[ind], self.data[ind]

    def spectrogram(self, timestamp, data, freq_range=[300, 8000]):
        """Calculate spectrogram"""
        from ..utils.spect import spectrogram

        spect, spect_freq, _ = spectrogram(
            data, self.sample_rate, freq_range=freq_range
        )
        spect_time = np.linspace(
            timestamp[0], timestamp[-1], spect.shape[1]
        )  # timestamp for spectrogram
        return spect_time, spect, spect_freq

    def get_spectral_entropy(self, spect, normalize=True, mode=None):
        """
        Calculate spectral entropy

        Parameters
        ----------
        normalize : bool
            Get normalized spectral entropy
        mode : {'spectral', ''spectro_temporal'}

        Returns
        -------
        array of spectral entropy
        """
        from ..core.functions import get_spectral_entropy

        return get_spectral_entropy(spect, normalize=normalize, mode=mode)


class NeuralData:
    def __init__(self, path, channel_nb, format="rhd", update=False):

        self.path = path
        self.channel_nb = str(channel_nb).zfill(2)
        self.format = format  # format of the file (e.g., rhd), this info should be in the database

        file_name = self.path / f"NeuralData_Ch{self.channel_nb}.npy"
        if (
            update or not file_name.exists()
        ):  # if .npy doesn't exist or want to update the file
            data_info = self.load_neural_data()
            # Save event_info as a numpy object
        else:
            data_info = np.load(file_name, allow_pickle=True).item()

        # Set the dictionary values to class attributes
        for key in data_info:
            setattr(self, key, data_info[key])

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def load_neural_data(self):
        """
        Load and concatenate all neural data files (e.g., .rhd) in the input dir (path)
        """

        from ..core.load import read_rhd
        from ..core.parameters import sample_rate

        print("")
        print("Load neural data")
        # List .rhd files
        files = list(self.path.glob(f"*.{self.format}"))

        # Initialize
        timestamp_concat = np.array([], dtype=np.float64)
        amplifier_data_concat = np.array([], dtype=np.float64)

        # Store values in these lists
        file_list = []

        if self.format == "cbin":

            # if the neural data is in .cbin format, read from .mat files that has contains concatenated data
            # currently does not have files to extract data from .cbin files in python
            import scipy.io

            mat_file = list(self.path.glob(f"*Ch{self.channel_nb}(merged).mat"))[0]
            timestamp_concat = scipy.io.loadmat(mat_file)["t_amplifier"][0].astype(
                np.float64
            )
            amplifier_data_concat = scipy.io.loadmat(mat_file)["amplifier_data"][
                0
            ].astype(np.float64)

        else:
            # Loop through Intan .rhd files
            for file in files:

                # Load data file
                print("Loading... " + file.stem)
                file_list.append(file.name)
                intan = read_rhd(file)  # note that the timestamp is in second
                # Concatenate timestamps
                intan["t_amplifier"] -= intan["t_amplifier"][0]  # start from t = 0
                if timestamp_concat.size == 0:
                    timestamp_concat = np.append(timestamp_concat, intan["t_amplifier"])
                else:
                    intan["t_amplifier"] += timestamp_concat[-1] + (
                        1 / sample_rate[self.format]
                    )
                    timestamp_concat = np.append(timestamp_concat, intan["t_amplifier"])

                # Concatenate neural data
                for ind, ch in enumerate(intan["amplifier_channels"]):
                    if int(self.channel_nb) == int(ch["native_channel_name"][-2:]):
                        amplifier_data_concat = np.append(
                            amplifier_data_concat, intan["amplifier_data"][ind, :]
                        )

            timestamp_concat *= 1e3  # convert to microsecond

        # Organize data into a dictionary
        data_info = {
            "files": file_list,
            "timestamp": timestamp_concat,
            "data": amplifier_data_concat,
            "sample_rate": sample_rate[self.format],
        }
        file_name = self.path / f"NeuralData_Ch{self.channel_nb}.npy"
        np.save(file_name, data_info)

        return data_info

    def extract(self, time_range: list):
        """
        Extracts data from the specified range

        Parameters
        ----------
        time_range : list
            list of time stamps [start, end]

        Returns
        -------
        timestamp : arr
        data : arr
        """

        start = time_range[0]
        end = time_range[-1]

        ind = np.where((self.timestamp >= start) & (self.timestamp <= end))
        return self.timestamp[ind], self.data[ind]

    @property
    def open_folder(self):

        from ..utils.functions import open_folder as _open_folder

        return _open_folder(self.path)


class Correlogram:
    """
    Class for correlogram analysis
    """

    def __init__(self, correlogram):

        from ..core.parameters import burst_hz, spk_corr_parm

        corr_center = round(correlogram.shape[0] / 2) + 1  # center of the correlogram
        self.data = correlogram
        self.time_bin = np.arange(
            -spk_corr_parm["lag"],
            spk_corr_parm["lag"] + spk_corr_parm["bin_size"],
            spk_corr_parm["bin_size"],
        )
        if self.data.sum():
            self.peak_ind = (
                np.min(
                    np.abs(
                        np.argwhere(correlogram == np.amax(correlogram)) - corr_center
                    )
                )
                + corr_center
            )  # index of the peak
            self.peak_latency = self.time_bin[self.peak_ind] - 1
            self.peak_value = self.data[self.peak_ind]
            burst_range = np.arange(
                corr_center - (1000 / burst_hz) - 1,
                corr_center + (1000 / burst_hz),
                dtype="int",
            )  # burst range in the correlogram
            self.burst_index = round(self.data[burst_range].sum() / self.data.sum(), 3)
        else:
            self.peak_ind = (
                self.peak_latency
            ) = self.peak_value = self.burst_index = np.nan

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])

    def category(self, correlogram_jitter: np.ndarray) -> str:
        """
        Get bursting category of a neuron based on autocorrelogram

        Parameters
        ----------
        correlogram_jitter : np.ndarray
            Random time-jittered correlogram for baseline setting

        Returns
        -------
            Category of a neuron ('Bursting' or 'Nonbursting')
        """
        from ..core.parameters import corr_burst_crit

        corr_mean = correlogram_jitter.mean(axis=0)

        if corr_mean.sum():

            corr_std = correlogram_jitter.std(axis=0)
            upper_lim = corr_mean + (corr_std * 2)
            lower_lim = corr_mean - (corr_std * 2)

            self.baseline = upper_lim

            # Check peak significance
            if (
                self.peak_value > upper_lim[self.peak_ind]
                and self.peak_latency <= corr_burst_crit
            ):
                self.category = "Bursting"
            else:
                self.category = "NonBursting"

        else:
            self.baseline = self.category = np.array(np.nan)
        return self.category

    def plot_corr(
        self,
        ax,
        time_bin,
        correlogram,
        title,
        xlabel=None,
        ylabel=None,
        font_size=10,
        peak_line_width=0.8,
        normalize=False,
        peak_line=True,
        baseline=True,
    ):
        """
        Plot correlogram

        Parameters
        ----------
        ax : axis object
            axis to plot the figure
        time_bin : np.ndarray
        correlogram : np.ndarray
        title : str
        font_size : int
            title font size
        normalize : bool
            normalize the correlogram
        """
        import matplotlib.pyplot as plt

        from ..utils.draw import remove_right_top
        from ..utils.functions import myround

        if correlogram.sum():
            ax.bar(time_bin, correlogram, color="k", rasterized=True)
            ymax = max([self.baseline.max(), correlogram.max()])
            round(ymax / 10) * 10
            ax.set_ylim(0, ymax)
            plt.yticks([0, ax.get_ylim()[1]], [str(0), str(int(ymax))])
            ax.set_title(title, size=font_size)
            ax.set_xlabel(xlabel)
            if normalize:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(ylabel)
            remove_right_top(ax)

            if peak_line and not np.isnan(self.peak_ind):
                # peak_time_ind = np.where(self.time_bin == self.peak_latency)
                ax.axvline(
                    x=self.time_bin[self.peak_ind],
                    color="r",
                    linewidth=peak_line_width,
                    ls="--",
                )

            if baseline and not np.isnan(self.baseline.mean()):
                ax.plot(self.time_bin, self.baseline, "m", lw=0.5, ls="--")
        else:
            ax.axis("off")
            ax.set_title(title, size=font_size)


class BurstingInfo:
    def __init__(self, ClassInfo, *input_context):

        from ..core.parameters import burst_hz

        # ClassInfo can be BaselineInfo, MotifInfo etc
        if input_context:  # select data based on social context
            spk_list = [
                spk_ts
                for spk_ts, context in zip(ClassInfo.spk_ts, ClassInfo.contexts)
                if context == input_context[0]
            ]
            duration_list = [
                duration
                for duration, context in zip(ClassInfo.durations, ClassInfo.contexts)
                if context == input_context[0]
            ]
            self.context = input_context
        else:
            spk_list = ClassInfo.spk_ts
            duration_list = ClassInfo.durations

        # Bursting analysis
        burst_spk_list = []
        burst_duration_arr = []

        nb_bursts = []
        nb_burst_spk_list = []

        for ind, spks in enumerate(spk_list):

            # spk = bi.spk_ts[8]
            isi = np.diff(spks)  # inter-spike interval
            inst_fr = 1e3 / np.diff(spks)  # instantaneous firing rates (Hz)
            bursts = np.where(inst_fr >= burst_hz)[0]  # burst index

            # Skip if no bursting detected
            if not bursts.size:
                continue

            # Get the number of bursts
            temp = np.diff(bursts)[
                np.where(np.diff(bursts) == 1)
            ].size  # check if the spikes occur in bursting
            nb_bursts = np.append(nb_bursts, bursts.size - temp)

            # Get burst onset
            temp = np.where(np.diff(bursts) == 1)[0]
            spk_ind = temp + 1
            # Remove consecutive spikes in a burst and just get burst onset

            burst_onset_ind = bursts

            for i, ind in enumerate(temp):
                burst_spk_ind = spk_ind[spk_ind.size - 1 - i]
                burst_onset_ind = np.delete(burst_onset_ind, burst_spk_ind)

            # Get burst offset index
            burst_offset_ind = np.array([], dtype=np.int)

            for i in range(bursts.size - 1):
                if bursts[i + 1] - bursts[i] > 1:  # if not successive spikes
                    burst_offset_ind = np.append(burst_offset_ind, bursts[i] + 1)

            # Need to add the subsequent spike time stamp since it is not included (burst is the difference between successive spike time stamps)
            burst_offset_ind = np.append(burst_offset_ind, bursts[bursts.size - 1] + 1)
            burst_onset = spks[burst_onset_ind]
            burst_offset = spks[burst_offset_ind]
            burst_spk_list.append(spks[burst_onset_ind[0] : burst_offset_ind[0] + 1])
            burst_duration_arr = np.append(
                burst_duration_arr, burst_offset - burst_onset
            )

            # Get the number of burst spikes
            nb_burst_spks = 1  # note that it should always be greater than 1

            if nb_bursts.size:
                if bursts.size == 1:
                    nb_burst_spks = 2
                    nb_burst_spk_list.append(nb_burst_spks)

                elif bursts.size > 1:
                    for ind in range(bursts.size - 1):
                        if bursts[ind + 1] - bursts[ind] == 1:
                            nb_burst_spks += 1
                        else:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
                            nb_burst_spks = 1

                        if ind == bursts.size - 2:
                            nb_burst_spks += 1
                            nb_burst_spk_list.append(nb_burst_spks)
            # print(nb_burst_spk_list)
        if sum(nb_burst_spk_list):
            self.spk_list = burst_spk_list
            self.nb_burst_spk = sum(nb_burst_spk_list)
            self.fraction = (
                round(sum(nb_burst_spk_list) / sum([len(spks) for spks in spk_list]), 3)
            ) * 100
            self.duration = round((burst_duration_arr).sum(), 3)  # total duration
            self.freq = round(nb_bursts.sum() / (sum(duration_list) / 1e3), 3)
            self.mean_nb_spk = round(np.array(nb_burst_spk_list).mean(), 3)
            self.mean_duration = round(burst_duration_arr.mean(), 3)  # mean duration
        else:  # no burst spike detected
            self.spk_list = []
            self.nb_burst_spk = (
                self.fraction
            ) = (
                self.duration
            ) = self.freq = self.mean_nb_spk = self.mean_duration = np.nan

    def __repr__(self):  # print attributes
        return str([key for key in self.__dict__.keys()])


class ISI:
    """
    Class object for inter-spike interval analysis
    """

    def __init__(self, isi):
        """
        Parameters
        ----------

        isi : np.ndarray
            Inter-spike interval array
        """
        from ..core.parameters import isi_bin, isi_scale, isi_win

        self.data = isi
        self.hist, self.time_bin = np.histogram(np.log10(isi), bins=isi_bin)
        self.time_bin = self.time_bin[:-1]
        # Peak latency of the ISI distribution
        self.time_bin = 10**self.time_bin
        self.peak_latency = self.time_bin[
            np.min(np.where(self.hist == np.min(self.hist.max())))
        ]  # in ms
        # Proportion of within-refractory period spikes
        self.within_ref_prop = (np.sum(self.data < 1) / self.data.shape[0]) * 100
        # CV of ISI
        self.cv = round(self.hist.std(axis=0) / self.hist.mean(axis=0), 3)

    def plot(self, ax, *title, font_size=10):
        from ..utils.draw import remove_right_top

        ax.bar(self.time_bin, self.hist, color="k")
        # ax.set_ylim([0, myround(math.ceil(ax.get_ylim()[1]), base=5)])
        ax.axvline(1, color="k", linestyle="dashed", linewidth=1)
        ax.axvline(self.peak_latency, color="r", linestyle="dashed", linewidth=0.3)
        ax.set_ylabel("Count")
        ax.set_xlabel("Time (ms)")
        ax.set_xscale("log")
        if title:
            ax.set_title(title[0], size=font_size)
        remove_right_top(ax)
