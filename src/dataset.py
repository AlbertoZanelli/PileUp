import numpy as np
from torch.utils.data import Dataset
import os
import glob
from dotenv import load_dotenv
import ast
import torch
import struct
from scipy.interpolate import interp1d
load_dotenv()
main_dir = os.getenv("DATA_PATH")


class NumpyDataset(Dataset):
    def __init__(self, np_array):
        self.data = torch.from_numpy(np_array)  # convert to torch.Tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BinaryDataset(Dataset):
    def __init__(self, file_path, win_length, positions = None, win_shift = 0, dtype = np.uint32):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the binary file.
            win_length (int): Length of each data window to extract.
            positions (list or np.array): List of starting positions in the binary file for windows.
            win_shift (int): Optional shift applied to window start positions (default: 0).
            dtype (numpy dtype): Data type for reading binary data (default: np.uint32).
        """
        self.file_path = file_path
        header = read_bin_header(file_path)
        self.samplingRate = header["samplingRate"]
        self.adcRange = header["adcRange"]
        self.win_length = int(win_length)
        self.dtype = dtype
        # Memory-map the binary file to allow efficient reading without loading the entire file into memory.
        # Skips the first 4 bytes of the file using offset=4 (often used to skip a header or metadata).
        self.file_map = np.memmap(self.file_path, dtype = dtype, mode = 'r', offset = 4)
        self.positions = positions if positions is not None else np.arange(0, len(self.file_map) - win_length,
                                                                           win_length)
        self.win_shift = int(win_shift)

    def load_and_process_window(self, start):
        """
        Loads a window of data from the memory-mapped file and applies processing.

        Args:
            start (int): Starting index for the window.

        Returns:
            np.array: Processed data window.
        """
        # Extract a slice of raw data of length `win_length` starting at `start`
        np_data = np.array(self.file_map[start:start + self.win_length], dtype = self.dtype)

        # Bit-shift each 32-bit value to the right by 8 bits, convert to float64,
        # normalize from raw values to a range between -10.24 and +10.24
        np_data = ((np.right_shift(np_data, 8).astype(np.float64) / (2 ** 23)) * self.adcRange - self.adcRange)

        return np_data

    def get_batch(self, indices):
        """
        Vectorized batch loading for arbitrary window positions.

        Args:
            indices (array-like): Dataset indices.

        Returns:
            np.ndarray: Shape (batch_size, win_length)
        """
        indices = np.asarray(indices)

        # Compute window start positions
        starts = self.positions[indices] - self.win_shift  # (B,)

        # Build index matrix: (B, win_length)
        idx = starts[:, None] + np.arange(self.win_length)[None, :]

        # Gather data in one shot
        batch = self.file_map[idx]

        # Vectorized processing
        batch = (
                (np.right_shift(batch, 8).astype(np.float64) / (2 ** 23))
                * self.adcRange
                - self.adcRange
        )

        return torch.tensor(batch)

    def set_win_length(self, win_length, win_shift = 0):
        """
        Adjusts the window length and shift for data extraction.

        Args:
            win_length (int): New window length.
            win_shift (int): New window shift.
        """
        if win_length == self.win_length and win_shift == self.win_shift:
            return
        win_length = int(win_length)
        start = self.win_shift - win_shift
        if start + win_length > self.win_length:
            raise ValueError(f"Requested window length {win_length} exceeds max length {self.win_length}")
        self.win_length = win_length
        self.win_shift = win_shift

    def __len__(self):
        """
        Returns the total number of data windows in the dataset.
        """
        return len(self.positions)

    def __getitem__(self, index):
        """
        Retrieves and processes a single data window based on the index.

        Args:
            index (int): Index into the list of positions.

        Returns:
            np.array: Processed data window.
        """
        # Calculate the actual start index for the window
        start = self.positions[index] - self.win_shift

        # Load and process the data window
        data = self.load_and_process_window(start)

        return data


class BinaryDatasetDenoised(BinaryDataset):
    def __init__(self, file_path, denoised_data, win_length, positions, win_shift = 0, dtype = np.uint32):
        super().__init__(file_path, win_length, positions, win_shift, dtype)
        self.denoised_data = denoised_data

    def __getitem__(self, index):
        start = self.positions[index] - self.win_shift
        raw_data = self.load_and_process_window(start) - self.denoised_data[index]
        return raw_data


class CachedBinaryDataset(Dataset):
    def __init__(self, file_path, win_length, positions, win_shift_start=0,
                 win_shift=0, device="cpu", dtype=np.uint32):
        """
        Initializes the CachedBinaryDataset.

        Args:
            file_path (str): Path to the binary file to be loaded.
            win_length (int): Length of each data window to extract.
            positions (array-like): List of starting positions in the binary file for windows.
            win_shift (int, optional): Shift applied to window start positions. Defaults to 0.
            device (str, optional): Device to store the data (e.g., "cpu" or "cuda"). Defaults to "cpu".
            dtype (numpy dtype, optional): Data type for reading binary data. Defaults to np.uint32.
        """
        self.win_length = int(win_length)
        self.win_shift = int(win_shift)
        self.n_windows = len(positions)
        self.positions = positions

        header = read_bin_header(file_path)
        self.samplingRate = header["samplingRate"]
        self.adcRange = header["adcRange"]

        # Load full file into memory (skip first 4 bytes)
        full_raw = np.memmap(file_path, dtype=dtype, mode='r')[1:]  # skip first entry

        # Build 2D indices for full windows
        starts = positions.astype(np.int64)
        idx = starts[:, None] + np.arange(win_length, dtype=np.int64)[None, :] + win_shift_start
        raw_windows = full_raw[idx]

        # Vectorized processing
        shifted = np.right_shift(raw_windows, 8).astype(np.float32)
        all_windows = (shifted / (2 ** 23)) * self.adcRange - self.adcRange

        # Convert to torch
        self.data = torch.from_numpy(all_windows)
        self.orignal_winlength = self.data.shape[1]
        if device == "cuda":
            self.data = self.data.cuda(non_blocking=True)

    def set_win_length(self, win_length):
        self.win_length = int(win_length)
        start = self.orignal_winlength // 2 - self.win_length // 2 + self.win_shift
        if start < 0 or start + self.win_length > self.orignal_winlength:
            raise ValueError(f"Requested window (start={start}, length={self.win_length}) exceeds "
                             f"max length {self.orignal_winlength}")

    def set_win_shift(self, win_shift):
        self.win_shift = int(win_shift)
        start = self.orignal_winlength // 2 - self.win_length // 2 + self.win_shift
        if start < 0 or start + self.win_length > self.orignal_winlength:
            raise ValueError(f"Requested window (start={start}, length={self.win_length}) exceeds "
                             f"max length {self.orignal_winlength}")

    def set_fracshift(self, fracshift):
        self.set_win_shift(int(self.win_length*fracshift))

    def get_batch(self, batch_indices):
        start = self.orignal_winlength // 2 - self.win_length // 2 + self.win_shift
        if start < 0 or start + self.win_length > self.orignal_winlength:
            raise ValueError(f"Requested window (start={start}, length={self.win_length}) exceeds "
                             f"max length {self.orignal_winlength}")
        end = start + self.win_length
        return self.data[batch_indices, start:end]

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = self.orignal_winlength // 2 - self.win_length // 2 + self.win_shift
        if start < 0 or start + self.win_length > self.orignal_winlength:
            raise ValueError(f"Requested window (start={start}, length={self.win_length}) exceeds "
                             f"max length {self.orignal_winlength}")
        end = start + self.win_length
        return self.data[idx, start:end]



class CachedBinaryDataset_withgenerated(CachedBinaryDataset):
    def __init__(self, file_path, position_file, win_length=2048, pulse = None, positions = None,
                 n_windows = 12288, win_shift_start=0, win_shift=0, injection_fraction = 0.5,
                 scaling_factor = 1.,
                 device = "cpu", dtype = np.uint32):
        """
            Initializes the CachedBinaryDataset_withgenerated class.

            Args:
                file_path (str): Path to the binary file to be loaded.
                position_file (str): Path to the file containing position data.
                win_length (int): Length of each data window to extract.
                pulse (callable or np.ndarray, optional): Pulse function or array for interpolation. Defaults to None.
                n_windows (int, optional): Number of windows to process. Defaults to 12288.
                win_shift (int, optional): Shift applied to window start positions. Defaults to 0.
                device (str, optional): Device to store the data (e.g., "cpu" or "cuda"). Defaults to "cpu".
                dtype (numpy dtype, optional): Data type for reading binary data. Defaults to np.uint32.
        """
        pos_file = np.loadtxt(position_file, ndmin = 2)[:n_windows]
        if positions is None:
            positions = (pos_file[:, 0] // 4).astype(int)
        else:
            positions = np.asarray(positions)
            n_windows = min(n_windows, len(positions))
            pos_file = pos_file[:n_windows]
        print(pos_file.shape)
        super().__init__(file_path, win_length, positions,
                         win_shift_start,win_shift, device, dtype)
        time = np.arange(0, self.win_length) - int(injection_fraction * self.win_length)
        if callable(pulse):
            pulse_inter = pulse
        elif pulse is not None:
            len_pulse = len(pulse)
            pulse_inter = interp1d(
                np.arange(- len_pulse // 2, len_pulse // 2),
                pulse, kind = 'quadratic', fill_value = 0., bounds_error = False)
        else:
            def pulse_inter(x):
                return 0 * x
        if pos_file.shape[1] > 3:
            smearing = pos_file[:, -2]
            dt = pos_file[:, -1]
            e1 = pos_file[:, 1]
            e2 = pos_file[:, 2]
            self.data += scaling_factor*(e1[:, None] * pulse_inter(time + smearing[:, None]) +
                          e2[:, None] * pulse_inter(time + smearing[:, None] + dt[:, None]))
        else:
            smearing = pos_file[:, -1]
            e1 = pos_file[:, 1]
            self.data += scaling_factor * e1[:, None] * pulse_inter(time + smearing[:, None])

class CachedBinaryDataset_withdenoise(CachedBinaryDataset):
    def __init__(self, file_path, position_file, win_length, denoised_data, pulse = None, positions = None,
                 n_windows = 12288, win_shift_start=0, win_shift=0, injection_fraction = 0.5,
                 device = "cpu", dtype = np.uint32):
        """
            Initializes the CachedBinaryDataset_withgenerated class.

            Args:
                file_path (str): Path to the binary file to be loaded.
                position_file (str): Path to the file containing position data.
                win_length (int): Length of each data window to extract.
                denoised_data (torch.Tensor): Denoised data to be subtracted from the raw data.
                pulse (callable or np.ndarray, optional): Pulse function or array for interpolation. Defaults to None.
                n_windows (int, optional): Number of windows to process. Defaults to 12288.
                win_shift (int, optional): Shift applied to window start positions. Defaults to 0.
                device (str, optional): Device to store the data (e.g., "cpu" or "cuda"). Defaults to "cpu".
                dtype (numpy dtype, optional): Data type for reading binary data. Defaults to np.uint32.
        """
        pos_file = np.loadtxt(position_file, ndmin = 2)[:n_windows]
        if positions is None:
            positions = (pos_file[:, 0] // 4).astype(int)
        super().__init__(file_path, win_length, positions,
                         win_shift_start,win_shift, device, dtype)
        time = np.arange(0, self.win_length) - int(injection_fraction * self.win_length)
        if callable(pulse):
            pulse_inter = pulse
        elif pulse is not None:
            len_pulse = len(pulse)
            pulse_inter = interp1d(
                np.arange(- len_pulse // 2, len_pulse // 2),
                pulse, kind = 'quadratic', fill_value = 0., bounds_error = False)
        else:
            def pulse_inter(x):
                return 0 * x
        if pos_file.shape[1] > 3:
            smearing = pos_file[:, -2]
            dt = pos_file[:, -1]
            e1 = pos_file[:, 1]
            e2 = pos_file[:, 2]
            self.data += (e1[:, None] * pulse_inter(time + smearing[:, None]) +
                          e2[:, None] * pulse_inter(time + smearing[:, None] + dt[:, None]))
        else:
            smearing = pos_file[:, -1]
            e1 = pos_file[:, 1]
            self.data += e1[:, None] * pulse_inter(time + smearing[:, None])
        self.data -= denoised_data

def batch_iterator(dataset, batch_size=4096, device="cpu", use_loader=True,
                   shuffle=False, num_workers=4, prefetch_factor=2):
    """
        Creates an iterator for batching data from a dataset, with options for using a DataLoader or a custom batch
        loader.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to iterate over.
            batch_size (int, optional): Number of samples per batch. Defaults to 4096.
            device (str, optional): Device to move the batches to (e.g., "cpu" or "cuda"). Defaults to "cpu".
            use_loader (bool, optional): Whether to use PyTorch's DataLoader for batching. Defaults to True.
            shuffle (bool, optional): Whether to shuffle the dataset before batching. Defaults to False.
            num_workers (int, optional): Number of worker threads for data loading (used if `use_loader` is True).
            Defaults to 4.
            prefetch_factor (int, optional): Number of batches to prefetch per worker (used if `use_loader` is True).
            Defaults to 2.

        Yields:
            torch.Tensor: A batch of data moved to the specified device.
    """
    if use_loader:
        # Standard DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor)
        for batch in loader:
            yield batch.to(device, dtype=torch.float32)
    else:
        # Custom get_batch loop (vectorized, fast)
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        for i in range(n_batches):
            indices = torch.arange(i*batch_size, min((i+1)*batch_size, len(dataset)))
            batch = dataset.get_batch(indices)
            yield batch.to(device, dtype=torch.float32)


def load_binary_file(file_path, dtype = np.uint32):
    """
    Loads a binary file into a numpy array.

    Args:
        file_path (str): Path to the binary file.
        dtype (numpy dtype): Data type for reading binary data (default: np.uint32).

    Returns:
        np.array: Numpy array containing the binary data.
    """
    data = np.fromfile(file_path, dtype = dtype, offset = 12)
    adc_range = read_bin_header(file_path)["adcRange"]
    data = ((np.right_shift(data, 8).astype(np.float64) / (2 ** 23)) * adc_range - adc_range)
    return data


def read_bin_header(path):
    """
        Reads the header of a binary file and extracts metadata.

        Args:
            path (str): Path to the binary file.

        Returns:
            dict: A dictionary containing the following keys:
                - "endian" (str): Endianness of the file ('<' for little-endian, '>' for big-endian).
                - "samplingRate" (float): Sampling rate extracted from the header.
                - "adcRange" (float): ADC range extracted from the header.

        Raises:
            ValueError: If the file is too small to contain the expected header structure.
    """
    header_struct = struct.Struct("<c3xff")

    with open(path, "rb") as f:
        data = f.read(header_struct.size)
        if len(data) != header_struct.size:
            raise ValueError("File too small to contain BinHeader")

        endian, samplingRate, adcRange = header_struct.unpack(data)

    # endian comes as a bytes object of length 1
    endian = endian.decode('ascii', errors = 'ignore')

    return {
        "endian": endian,
        "samplingRate": samplingRate,
        "adcRange": adcRange
    }


def find_file(target, specific_subdir = ""):
    """
        Searches for a file within a specified subdirectory or the main directory.

        Args:
            target (str): The name of the file to search for.
            specific_subdir (str, optional): A specific subdirectory within the main directory to narrow the search.
            Defaults to "".

        Returns:
            str: The full path to the first matching file.

        Raises:
            FileNotFoundError: If the file is not found in the specified subdirectory or the main directory.
    """
    file_paths = glob.glob(os.path.join(main_dir, specific_subdir, "**", target), recursive = True)
    if len(file_paths) == 0:
        file_paths = glob.glob(os.path.join(main_dir, "**", target), recursive = True)
        if len(file_paths) == 0:
            raise FileNotFoundError(f"File {target} not found in {os.path.join(main_dir, specific_subdir)}")
    return file_paths[0]


def find_files(data_name, pos_name, channel, pos_prefix = "_stdcut", differenciate = False, file_suffix = "",
               specific_subdir = ""):
    """
        Finds the file paths for data and position files based on the provided parameters.

        Args:
            data_name (str): Base name of the data file to search for.
            pos_name (str): Base name of the position file to search for.
            channel (int): Channel number to include in the file name pattern.
            pos_prefix (str, optional): Prefix for the position file. Defaults to "_stdcut".
            differenciate (bool, optional): Whether to use differentiated data paths. Defaults to False.
            file_suffix (str, optional): Suffix to append to the file name. Defaults to "".
            specific_subdir (str, optional): Specific subdirectory to narrow the search. Defaults to "".

        Returns:
            tuple: A tuple containing:
                - file_path (str): Full path to the data file.
                - path_pos (str): Full path to the position file.

        Raises:
            FileNotFoundError: If either the data file or the position file is not found.
    """
    modif = "_dif" if differenciate else ""
    target = data_name + f"_{channel:03}_???{modif}.bin{file_suffix}"
    file_path = find_file(target, specific_subdir)
    try:
        target_pos = pos_name + f"_{channel:03}_???{pos_prefix}.pos"
        path_pos = find_file(target_pos, specific_subdir)
    except FileNotFoundError:
        target_pos = pos_name + f".pos"
        path_pos = find_file(target_pos, specific_subdir)
    return file_path, path_pos


def create_data_sets(channel, dataset_dict = None, win_length = 2000, shift = 0,
                     pos_prefix = "", file_suffix = "",
                     differenciate = False, len_data = 15000, cached = False,
                     **kwargs):
    """
    Creates datasets for single and pileup data, along with additional processed information.

    Args:
        channel (int): Channel number to process.
        dataset_dict (dict, optional): Dictionary containing dataset configuration. Defaults to None.
            Expected keys:
                - "data_name" (list of str): Base names of the data files.
                - "pos_name" (list of str): Base names of the position files.
                - "specific_subdir" (list of str): Subdirectories for file search.
                - "full_info" (list of bool): Flags indicating whether to include additional information.
        win_length (int, optional): Length of each data window. Defaults to 2000.
        shift (int, optional): Shift applied to window start positions. Defaults to 0.
        pos_prefix (str, optional): Prefix for position files. Defaults to "".
        file_suffix (str, optional): Suffix for binary files. Defaults to "".
        differenciate (bool, optional): Whether to use differentiated data paths. Defaults to False.
        len_data (int, optional): Number of data entries to process. Defaults to 15000.
        cached (bool, optional): Whether to use cached datasets. Defaults to False.
        **kwargs: Additional arguments passed to the dataset class.

    Returns:
        tuple: A tuple containing:
            - data_single (BinaryDataset or CachedBinaryDataset): Dataset for single data.
            - data_pileup (BinaryDataset or CachedBinaryDataset): Dataset for pileup data.
            - parameters (list): Additional processed information for pileup data.
    """
    if dataset_dict is None:
        dataset_dict = {
            "data_name": ["pup_n1-d0_000813_20230628T161508", "pup_n1-d8_000813_20230628T161508"],
            "pos_name": ["pup_n1-d0_000813_20230628T161508", "pup_n1-d8_000813_20230628T161508"],
            "specific_subdir": ["RUN9_pulse_injected_new/", "RUN9_pulse_injected_new/"],
            "full_info": [False, True]
        }
    dataset_class = CachedBinaryDataset if cached else BinaryDataset
    datas = []
    parameters = []
    for data_name, pos_name, specific_subdir, full_info in zip(dataset_dict["data_name"],
                                                               dataset_dict["pos_name"],
                                                               dataset_dict["specific_subdir"],
                                                               dataset_dict["full_info"]):
        file_path, pos_path = find_files(data_name, pos_name, channel,
                                         pos_prefix = pos_prefix, differenciate = differenciate,
                                         file_suffix = file_suffix,
                                         specific_subdir = specific_subdir)
        print(pos_path)
        pos_file = np.loadtxt(pos_path, ndmin=2)[:len_data]
        positions = (pos_file[:, 0] // 4).astype(int)
        data = dataset_class(file_path, win_length, positions, win_shift = win_length // 2 - shift,
                             **kwargs)
        datas.append(data)
        if full_info:
            dt = pos_file[:, -1]
            r = pos_file[:, 2] / (pos_file[:, 2] + pos_file[:, 1])
            e = pos_file[:, 1:3]
            parameters.append({"dt": dt, "r": r, "e": e})

    return *datas, parameters


def get_channel_specs(channel, n_deriv = 1, window_fct = np.ones, nps_name = "raw_noise"):
    """
    Loads and processes filter data for a given channel.

    Args:
        channel (int): Channel number to process.
        n_deriv (int, optional): Number of derivatives to compute. Defaults to 1.
        window_fct (callable, optional): Window function to apply. Defaults to np.ones.
        nps_name (str, optional): Name of the noise power spectrum file. Defaults to "raw_noise".

    Returns:
        tuple: A tuple containing:
            - H_unit (numpy.ndarray): Normalized filter response.
            - S (numpy.ndarray): Fourier transform of the differentiated mean pulse.
            - nps (numpy.ndarray): Baseline spectrum.
            - w (numpy.ndarray): frequency array.
    """
    from src.analysis import compute_H
    # Load baseline spectrum
    meas_name = "000813_20230628T161508"

    nps = np.fromfile(find_file(f"{nps_name}_d{n_deriv}.bin_spec.bin",
                                specific_subdir = f"RUN9_pulse_injected_new/channel{channel}"))
    meanpulse = np.fromfile(find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
                                      specific_subdir = "RUN9_pulse_injected_new/"))
    meanpulse /= np.max(np.abs(meanpulse))
    # Compute the N-th derivative of the mean pulse
    dmeanpulse = np.diff(meanpulse, n = n_deriv, prepend = [0] * n_deriv)
    pulse_factor = np.max(np.abs(dmeanpulse))
    size = len(nps)
    dmeanpulse = dmeanpulse[np.argmax(dmeanpulse) - size // 2: np.argmax(dmeanpulse) + size // 2]
    if len(dmeanpulse) < size:
        dmeanpulse = np.pad(dmeanpulse, (0, size - len(dmeanpulse)), 'constant')
    dmeanpulse /= np.max(np.abs(dmeanpulse))
    # Perform Fourier transform and compute angular frequency
    S, w, H_unit = compute_H(dmeanpulse, nps, window_fct)
    return H_unit, S, nps, w, pulse_factor, dmeanpulse


def load_params_simu(filename):
    """
        Loads simulation parameters from a configuration file.

        Args:
            filename (str): Path to the configuration file.

        Returns:
            dict: A dictionary containing the parsed parameters. Keys are parameter names (str),
                  and values are the corresponding parsed values. Values are parsed as Python literals
                  (e.g., int, float, list, dict) if possible; otherwise, they are returned as strings.
    """
    params = {}
    with open(filename, "r") as f:
        for line in f:
            # Remove inline comments
            line = line.split("#", 1)[0].strip()
            # Skip empty lines
            if not line:
                continue
            # Parse key=value
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().rstrip(";")
                # Try to parse Python literal
                try:
                    parsed_value = ast.literal_eval(value)
                except ValueError:
                    parsed_value = value  # leave as string

                params[key] = parsed_value

    return params


def get_amp_Q_val(channel):
    """
        Computes the amplitude quality value (Q value) for a given channel.

        Args:
            channel (int): The channel number for which the Q value is to be computed.

        Returns:
            float: The computed amplitude Q value.
        """
    target = f"000813_20230628T161508_{channel:03}_???-pup_single.injconf"
    params = load_params_simu(find_file(target, specific_subdir = "RUN9_pulse_injected_new/"))
    return params["LY_close"] * 3.034 * params["ampl_Tl"] / 2.614511 / params["LY_far"] * params["area_factor"]
