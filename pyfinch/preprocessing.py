"""
Module for data preprocessing.

Prepare & visualize raw data for further analysis.
"""

from pathlib import Path


def label_context():
    """
    Add a tag to a file name (e.g., _Dir, _Undir) to differentiate between different contexts.

    Files for each context should be stored in /Dir or /Undir folder.
    """

    import os
    import shutil
    from tkinter import Tk
    from tkinter import messagebox as mb
    from .utils.functions import find_data_path

    data_path = find_data_path()
    os.chdir(data_path)

    list_dir = [dir for dir in os.listdir(data_path)]

    for tag in list_dir:

        new_path = os.path.join(data_path, tag)
        os.chdir(new_path)
        list_file = [file for file in os.listdir(new_path)]

        for file in list_file:

            new_file = ""
            print("Processing... {}".format(file))
            ext = os.path.splitext(file)[1]  # file extension
            if ext == ".txt":
                shutil.copyfile(file, os.path.join(data_path, file))
            elif "merged" in file:
                shutil.copyfile(file, os.path.join(data_path, file))
            elif "not" in file:  # .not.mat files
                new_file = (
                    "_".join([file.split(".")[0], tag.title()])
                    + "."
                    + ".".join(file.split(".")[-3:])
                )
                shutil.copyfile(file, os.path.join(data_path, new_file))
            elif "labeling" in file:
                new_file = (
                    "_".join([file[: str.find(file, "(labeling)")], tag.title()])
                    + file[str.find(file, "(labeling)") :]
                )
                shutil.copyfile(file, os.path.join(data_path, new_file))
            else:
                new_file = (
                    "_".join([file.split(".")[0], tag.title()]) + ext
                )  # e.g., b70r38_190321_122233_Undir.rhd
                shutil.copyfile(file, os.path.join(data_path, new_file))

    os.chdir(data_path)
    root = Tk()
    root.withdraw()

    def ask():
        """Ask if you wish to keep/delete the root folder"""
        resp = mb.askquestion("Question", "Do you want to delete the root folder?")
        if resp == "yes":
            shutil.rmtree(new_path)  # works even if the folder is not empty
            mb.showinfo("", "Check the files!")
            root.destroy()
        else:
            mb.showinfo("", "Done!")
            root.destroy()

    ask()


def downsample_wav():
    """Downsample .wav files"""

    import librosa
    from pathlib import Path
    import soundfile as sf
    from .utils.functions import find_data_path

    target_sr = 32000  # target sampling rate

    # Specify dir here or search for the dir manually
    data_dir = Path(
        r"H:\Box\Data\Deafening Project\o25w75\Predeafening\D01(20120208)\01\Songs"
    )
    try:
        data_dir
    except NotADirectoryError:
        data_dir = find_data_path()

    files = list(data_dir.glob("*.wav"))

    for file in files:
        print("Processing... " + file.stem)
        signal, _ = librosa.load(
            file, sr=target_sr
        )  # Downsample to the target sample rate
        # sf.write(file, signal, target_sr)
        sf.write(data_dir / file, signal, target_sr)


def convert_adbit2volts(spk_waveform):
    """Input the waveform matrix extracted from the cluster .txt output"""

    # Parameters on the Offline Sorter
    volt_range = 10  # in milivolts +- 5mV
    sampling_bits = 16
    volt_resolution = 0.000153  # volt_range/(2**sampling_bits)
    spk_waveform_new = spk_waveform / (0.000153 * 1e3)  # convert to microvolts
    return spk_waveform_new


"""
Convert files into differnt format / change names
"""

import numpy as np


def change_cbin_names(data_path=None):
    """
    Change the name of the .cbin & .rec files to fit the format (b14r74_190913_161407_Undir) used in the current analysis

    Parameters
    ----------
    data_path : str
    """

    import os
    from pathlib import Path
    from .utils.functions import find_data_path

    # Find data path
    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    # Get .cbin, .rec files to process
    cbin_files = [str(rhd) for rhd in data_path.rglob("*.cbin")]
    rec_files = [str(rhd) for rhd in data_path.rglob("*.rec")]

    if not cbin_files:
        print("No .cbin files in the directory!")
    if not rec_files:
        print("No .rec files in the directory!")
    else:

        for cbin, rec in zip(cbin_files, rec_files):

            cbin_file = Path(cbin)  # e.g.,  'k27o36.3-03302012.188'
            rec_file = Path(rec)

            if (
                not len(cbin_file.name.split(".")) == 2
            ):  # not the file format sent from Mimi
                bird_id = cbin_file.stem.split(".")[0]
                date = cbin_file.stem.split(".")[1].split("-")[1]
                date = date[-2:] + date[:4]
                file_ind = cbin_file.stem.split(".")[-1]

                new_file_name = "_".join(
                    [bird_id, date, file_ind]
                )  # e.g., k27o36_120330_188
                new_cbin_file = new_file_name + Path(cbin).suffix  # add .cbin extension
                new_rec_file = new_file_name + Path(rec).suffix  # add .rec extension

                new_cbin_file = cbin_file.parent / new_cbin_file  # path
                new_rec_file = rec_file.parent / new_rec_file  # path

                if not new_cbin_file.exists():
                    print(cbin_file.name)
                    os.rename(cbin_file, new_cbin_file)
                    os.rename(rec_file, new_rec_file)

            else:
                print("Not the expected file format")
                break

        print("Done!")


def convert2syllable(data_path=None):
    """
    Rename variables (labels -> syllables) to avoid the clash with the

    Reserved keyword (labels) in newer version of MATLAB

    Parameters
    ----------
    data_path : str
    """
    from pathlib import Path
    import scipy.io
    from .utils.functions import find_data_path

    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    notmat_files = [file for file in data_path.glob("*.not.mat")]

    for file in notmat_files:

        # Load the .not.mat file
        print("Loading... " + file.stem)

        # Rename the key
        notmat = scipy.io.loadmat(file)

        if "labels" in notmat:
            notmat["syllables"] = notmat.pop("labels")
            scipy.io.savemat(file, notmat)  # store above values to new .not.mat
        else:
            print("labels don't exist!")

    print("Done!")


def intan2wav(data_path=None, *args):
    """
    Convert all .rhd files into .wav and plot the raw data

    Parameters
    ----------
    data_path : str or path
            The folder that contains labeling .not.mat files.
    args : sample_rate, freq_range

    """
    from .utils.intan.load_intan_rhd_format import read_rhd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from math import ceil
    from .utils.functions import find_data_path
    from .utils.spect import spectrogram

    # Find data path
    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    rhd_files = [str(rhd) for rhd in data_path.rglob("*.rhd")]

    for rhd in rhd_files:

        file_name = Path(rhd).stem
        fig_name = Path(rhd).with_suffix(".png")
        intan = read_rhd(rhd)  # load the .rhd file

        nb_channels = len(intan["amplifier_data"])
        intan["t_amplifier"] -= intan["t_amplifier"][0]  # start from t = 0

        fig, ax = plt.subplots(
            nrows=nb_channels + 1, ncols=1, sharex=True, figsize=(8, 2 * nb_channels)
        )

        # Plot spectrogram for song
        if "sample_rate" not in args:
            from pyfinch.analysis.parameters import sample_rate

            sample_rate = sample_rate["rhd"]

        if "freq_range" not in args:
            from pyfinch.analysis.parameters import freq_range

            freq_range = freq_range

        freq_range = freq_range

        ax[0].set_title(file_name, fontsize=12)
        spect, spect_freq, _ = spectrogram(
            intan["board_adc_data"][0],
            samp_freq=sample_rate,
            freq_range=freq_range,
            transform_type="log_spect",
        )
        spect_time = np.linspace(
            intan["t_amplifier"][0], intan["t_amplifier"][-1], spect.shape[1]
        )  # timestamp for spectrogram

        ax[0].pcolormesh(
            spect_time,
            spect_freq,
            spect,
            cmap="gray_r",
            rasterized=True,
            vmin=np.mean(spect),
            vmax=np.max(spect),
        )
        # ax[0].specgram(intan['board_adc_data'][0], Fs=sample_rate, cmap='binary', scale_by_freq=True)
        ax[0].spines["right"].set_visible(False), ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_visible(False), ax[0].spines["bottom"].set_visible(
            False
        )
        ax[0].set_ylim(freq_range)
        ax[0].set_ylabel("Frequency (Hz)", fontsize=8)
        ax[0].set_yticks([freq_range[0], freq_range[1]])
        ax[0].set_yticklabels([str(freq_range[0]), str(freq_range[1])])

        # Set the range of the y-axis
        y_range = [
            abs(intan["amplifier_data"].min()),
            abs(intan["amplifier_data"].max()),
        ]
        y_range = ceil(max(y_range) / 1e2) * 1e2

        for i, ch in enumerate(intan["amplifier_data"]):
            ax[i + 1].plot(intan["t_amplifier"], ch, "k", linewidth=0.5, clip_on=False)
            ax[i + 1].spines["right"].set_visible(False)
            ax[i + 1].spines["top"].set_visible(False)
            ax[i + 1].set_ylabel(intan["amplifier_channels"][i]["native_channel_name"])
            ax[i + 1].set_ylim([-y_range, y_range])

            if i is nb_channels - 1:  # the bottom plot
                ax[i + 1].set_xlabel("Time (s)")

        plt.tight_layout()


def notestat(
    data_path=None,
    fig_name=None,
    save_path=None,
    save_fig=True,
    fig_ext=".png",
    view_folder=True,
    dot_size=3,
    fig_size=[6, 5],
):
    """
    Plot syllable durations to view distributions & detect outliers.

    This should done after each syllable segmentation with uisonganal.m

    Labeling files should exist in .not.mat format in the data path.

    Parameters
    ----------
    data_path : str or path
        The folder that contains labeling .not.mat files.
    fig_name : str, optional
        Name of the figure. If not specified, defaults to the data path name.
    save_path : path, optional
        Path to save the figure. If not specified, save to the data path.
    save_fig : bool, default=True
        If true, save result figure. If False, just display it.
    fig_ext : str, default='.png'
        Figure file extension
    view_folder : bool, default=True
        Open the save folder
    dot_size : int, default=3
        Size of the scatter dot
    fig_size : list
        Size of the figure [width, height]
    """

    import math

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from .analysis.load import read_not_mat
    from .utils import save
    from .utils.draw import remove_right_top
    from .utils.functions import myround, find_data_path

    # Find data path
    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    # Store results in the dataframe
    audio_files = list(data_path.glob("*.wav"))

    df = pd.DataFrame()

    # Loop over all the audio files
    for file in audio_files:
        # Load the .not.mat file
        # print('Loading... ' + file.stem)
        notmat_file = file.with_suffix(".wav.not.mat")

        if not notmat_file.exists():
            raise FileNotFoundError

        birdID = file.name.split("_")[0]
        onsets, offsets, intervals, durations, syllables, context = read_not_mat(
            notmat_file
        )

        nb_syllable = len(syllables)

        temp_df = pd.DataFrame(
            {
                "FileID": [notmat_file] * nb_syllable,
                "Syllable": list(syllables),
                "Duration": durations,
            }
        )
        df = df.append(temp_df, ignore_index=True)

    # Plot the results
    syllable_list = sorted(list(set(df["Syllable"].to_list())))

    fig, ax = plt.subplots(figsize=fig_size)
    plt.title(fig_name)
    sns.stripplot(
        ax=ax,
        data=df,
        x="Syllable",
        y="Duration",
        order=syllable_list,
        s=dot_size,
        palette=sns.color_palette(),
        # set color category to be consistent regardless of the number of syllables
        jitter=0.15,
    )

    for syllable, x_loc in zip(syllable_list, ax.get_xticks()):
        nb_syllable = df[df["Syllable"] == syllable]["Syllable"].count()
        max_dur = df[df["Syllable"] == syllable]["Duration"].max()
        text = "({})".format(nb_syllable)
        x_loc -= ax.get_xticks()[-1] * 0.03
        y_loc = max_dur + ax.get_ylim()[1] * 0.05
        plt.text(x_loc, y_loc, text)

    ax.set_ylim([0, myround(math.ceil(ax.get_ylim()[1]), base=50)])
    remove_right_top(ax)
    plt.ylabel("Duration (ms)")
    fig.tight_layout()

    # Save the figure
    if save_fig:
        if not save_path:
            save_path = data_path
        if not fig_name:
            fig_name = data_path.name
        save.save_fig(
            fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder
        )
    else:
        plt.show()

    # print("Done!")


def rhd(data_path=None, save_fig=True, fig_ext=".png", view_folder=True):
    """
    Plot .rhd files (intan)

    Returns
    -------

    """

    from .analysis.load import read_rhd
    import matplotlib.pyplot as plt
    from .utils.functions import find_data_path
    from .utils import save

    # Find data path
    if data_path:
        data_path = Path(data_path)
    else:  # Search for data dir manually
        data_path = find_data_path()

    rhd_files = list(data_path.glob("*.rhd"))

    for rhd in rhd_files:

        intan = read_rhd(rhd)  # load the .rhd file
        nb_channels = len(intan["amplifier_data"])

        fig, axes = plt.subplots(figsize=(6, 1.5 * nb_channels))
        for ch in intan["amplifier_data"]:
            plt.plot(intan["t_amplifier"], ch, "k")

        # Save the figure
        if save_fig:
            if not save_path:
                save_path = data_path
            if not fig_name:
                fig_name = data_path.name
            save.save_fig(
                fig, save_path, fig_name, fig_ext=fig_ext, view_folder=view_folder
            )
        else:
            plt.show()
