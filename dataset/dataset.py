import os

import numpy as np
import pandas as pd
import torch


class BuildDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir: str,
            spectrum_size: int,
            load_mode: str,
            spectrum_dir_name: str,
            photometric_dir_name: str,
            label_dir_name: str,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.spectrum_size = spectrum_size
        self.load_mode = load_mode
        self.spectrum_dir_name = spectrum_dir_name
        self.photometric_dir_name = photometric_dir_name

        assert self.load_mode in ['train', 'val', 'test'], '[Error] load_mode must be in [train, val, test]'
        self.label_csv = pd.read_csv(
            os.path.join(self.dataset_dir, self.load_mode, label_dir_name, 'label.csv'),
            header=0,
            converters={
                'obsid': str,
                'SNR': float,
                'teff_K': float,
                'logg_cm_s2': float,
                'feh_dex': float,
                'mass_Msun': float,
                'age_Gyr': float,
            }
        )

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        data_row = self.label_csv.iloc[idx]
        data_id = data_row['obsid']
        snr = data_row['SNR']
        teff_k = data_row['teff_K']
        logg = data_row['logg_cm_s2']
        feh_dex = data_row['feh_dex']
        mass_m_sun = data_row['mass_Msun']
        age_gyr = data_row['age_Gyr']
        # check label, if nan, raise error
        assert not np.isnan(teff_k), '[Error] teff_k of {} is nan'.format(data_id)
        assert not np.isnan(logg), '[Error] logg of {} is nan'.format(data_id)
        assert not np.isnan(feh_dex), '[Error] feh_dex of {} is nan'.format(data_id)
        assert not np.isnan(mass_m_sun), '[Error] mass_m_sun of {} is nan'.format(data_id)
        assert not np.isnan(age_gyr), '[Error] age_gyr of {} is nan'.format(data_id)

        wavelength_spectrum = np.load(
            os.path.join(self.dataset_dir, self.load_mode, self.spectrum_dir_name, data_id + '.npy')
        )  # like (2, 3522)
        wavelength = wavelength_spectrum[0]  # wavelength
        spectrum = wavelength_spectrum[1]  # flux
        photometric = np.load(
            os.path.join(self.dataset_dir, self.load_mode, self.photometric_dir_name, data_id + '.npy')
        )
        assert spectrum.shape[-1] <= self.spectrum_size, '[Error] spectrum_size must be larger than {}'.format(
            spectrum.shape[-1]
        )
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
        # padding in both sides
        padding_half = (self.spectrum_size - spectrum.shape[-1]) // 2
        spectrum = torch.cat(
            (torch.zeros(padding_half),
             torch.from_numpy(spectrum).float(),
             torch.zeros(padding_half))).unsqueeze(0)
        if spectrum.shape[1] < self.spectrum_size:
            spectrum = torch.cat((spectrum, torch.zeros(1, self.spectrum_size - spectrum.shape[1])), dim=1)
        assert spectrum.shape == (1, self.spectrum_size), '[Error] spectrum shape must be (1, {})'.format(
            self.spectrum_size
        )
        # check nan
        assert not torch.isnan(spectrum).any(), '[Error] spectrum {} contains nan'.format(
            os.path.join(self.dataset_dir, self.load_mode, self.spectrum_dir_name, data_id + '.npy')
        )
        # photometric min-max normalization in each channel
        photometric = (photometric - photometric.min(axis=(1, 2), keepdims=True)) / (
                photometric.max(axis=(1, 2), keepdims=True) - photometric.min(axis=(1, 2), keepdims=True))
        photometric = torch.from_numpy(photometric).float()  # C x H x W
        # check nan
        assert not torch.isnan(photometric).any(), '[Error] photometric {} contains nan'.format(
            os.path.join(self.dataset_dir, self.load_mode, self.photometric_dir_name, data_id + '.npy')
        )
        # label
        label = torch.tensor([mass_m_sun, age_gyr]).float()
        return data_id, snr, padding_half, wavelength, spectrum, photometric, label


def custom_collate_fn(batch):
    data_ids = [item[0] for item in batch]
    snrs = [item[1] for item in batch]
    padding_halves = [item[2] for item in batch]
    wavelengths = [item[3] for item in batch]
    spectra = torch.stack([item[4] for item in batch])
    photometrics = torch.stack([item[5] for item in batch])
    labels = torch.stack([item[6] for item in batch])
    return data_ids, snrs, padding_halves, wavelengths, spectra, photometrics, labels


def build_dataloader(config: dict, mode: str, cross_val_name: str = '') -> torch.utils.data.DataLoader:
    """
    build dataloader
    :param config: contains keys: dataset_dir, spectrum_dir, label_dir, type_list, spectrum_size, batch_size, num_workers
    :param mode: dataset load mode, must be in ['train', 'val', 'test']
    :param cross_val_name: cross validation name
    :return: dataloader
    """
    assert mode in ['train', 'val', 'test'], '[Error] mode must be "train", "val" or "test"'
    # check config
    keys = [
        # dataset
        'dataset_dir',
        'spectrum_dir',
        'photometric_dir',
        'label_dir',
        'spectrum_size',
        # dataloader
        'batch_size',
        'num_workers',
    ]
    for key in keys:
        assert key in config, f'[Error] {key} must be in config'
    dataset_dir = config['dataset_dir']
    if cross_val_name is not None and cross_val_name != '':
        dataset_dir = os.path.join(config['dataset_dir'], cross_val_name)
    _dataset = BuildDataset(
        dataset_dir=dataset_dir,
        spectrum_size=config['spectrum_size'],
        load_mode=mode,
        spectrum_dir_name=config['spectrum_dir'],
        photometric_dir_name=config['photometric_dir'],
        label_dir_name=config['label_dir'],
    )
    _dataloader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=config['batch_size'],
        shuffle=True if mode == 'train' else False,
        collate_fn=custom_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    return _dataloader
