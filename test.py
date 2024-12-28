import math
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.rich import tqdm

from config.config import config
from dataset.dataset import build_dataloader
from model.lightning_model import BuildLightningModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        model = BuildLightningModel.load_from_checkpoint(
            os.path.join(model_path),
            map_location=device,
        ).model
        # remove model. prefix
        state_dict = {k.replace('model.', ''): v for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict)
        print('[Info] Load model from {}'.format(model_path))
    except Exception as e:
        print('[Error] Load model from {} failed'.format(model_path))
        raise e
    return model


def plot_weight_heatmap(weights: np.ndarray, save_path: str, norm: bool = True):
    if norm:
        # min-max normalization in each batch
        weights = (weights - weights.min(axis=1, keepdims=True)) / (
                weights.max(axis=1, keepdims=True) - weights.min(axis=1, keepdims=True))

    _heatmap = plt.figure(figsize=(6, 3), dpi=300)
    sns.heatmap(
        weights.squeeze(),
        cmap='viridis',
        square=True,
        vmax=1 if norm else np.max(weights),
        vmin=0 if norm else np.min(weights),
        # remove x, y axis
        xticklabels=False,
        yticklabels=False,
    )
    plt.axvline(x=32, color='#ffffffb3', linewidth=3, linestyle='--')
    plt.axvline(x=64, color='#ffffffb3', linewidth=3, linestyle='--')
    plt.xlabel('Channel', fontweight='bold', fontsize=14)
    plt.ylabel('Batch', fontweight='bold', fontsize=14)
    plt.tight_layout()
    _heatmap.savefig(save_path)
    plt.close()


def inference(
        config: dict,
        model_path: str,
        cross_validation_fold_name: str,
        device: torch.device,
        snr_step: int,
        snr_upper_limit: int = 100,
):
    dataloader = build_dataloader(config, mode='test', cross_val_name=cross_validation_fold_name)
    model = load_model(model_path, device)
    _pred_list = []
    _se_weights = []
    _spec_dataset = []
    total_time = 0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(dataloader), ncols=100, desc='Predicting...') as pbar:
            for batch in dataloader:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                data_id, snr, padding_half, wavelength, spectrum, photometric, label = batch
                spectrum = spectrum.to(device)
                photometric = photometric.to(device)
                label = label.to(device)
                start.record()
                output, attn_weights, se_weights = model(photometric, spectrum)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)
                _se_weights.append(
                    {
                        'data_id': data_id,
                        'weight': se_weights.cpu().numpy()
                    }
                )
                _spec_dataset.append(
                    {
                        'data_id': data_id,
                        'spectrum': spectrum.cpu().numpy(),
                        'wavelength': wavelength,
                        'padding_half': padding_half,
                    }
                )
                pred_mass_m_sun, pred_age_gyr = output[:, 0], output[:, 1]
                for i in range(photometric.shape[0]):
                    _pred_list.append(
                        [data_id[i],
                         snr[i].item(),
                         pred_mass_m_sun[i].item(),
                         pred_age_gyr[i].item(),
                         label[i][0].item(),
                         label[i][1].item()
                         ])
                pbar.update(1)
    # print fps
    print('[Info] FPS: {:.2f}'.format(len(dataloader) / (total_time / 1000)))
    # divide pred_list by snr, every 10 snr is a group
    _pred_list.sort(
        key=lambda x: x[1]
    )
    _pred_list = [x for x in _pred_list if
                  x[1] > 0]  # data_id, snr, pred_mass_m_sun, pred_age_gyr, label_mass_m_sun, label_age_gyr
    _without_id_pred_list = np.array([
        [x[1], x[2], x[3], x[4], x[5]] for x in _pred_list
    ])  # snr, pred_mass_m_sun, pred_age_gyr, label_mass_m_sun, label_age_gyr
    min_snr = math.floor(_pred_list[0][1] / 10) * 10
    max_snr = math.ceil(_pred_list[-1][1] / 10) * 10 if math.ceil(
        _pred_list[-1][1] / 10) * 10 < snr_upper_limit else snr_upper_limit
    _without_id_pred_list_by_snr = []
    # for min snr < snr_upper_limit
    for i in range(min_snr, max_snr, snr_step):
        _arr = np.array([[x[1], x[2], x[3], x[4], x[5]] for x in _pred_list if i <= x[1] < i + snr_step])
        _without_id_pred_list_by_snr.append(_arr)
    # for max snr > snr_upper_limit
    _arr = np.array([[x[1], x[2], x[3], x[4], x[5]] for x in _pred_list if x[1] >= snr_upper_limit])
    if _arr.shape[0] > 0:
        _without_id_pred_list_by_snr.append(_arr)
    return (
        _pred_list,
        min_snr,
        _without_id_pred_list,
        _without_id_pred_list_by_snr,
        _se_weights,
        _spec_dataset,
    )


def get_mse(pred: np.ndarray, label: np.ndarray) -> float:
    return metrics.mean_squared_error(label, pred)


def get_mae(pred: np.ndarray, label: np.ndarray) -> float:
    return metrics.mean_absolute_error(label, pred)


def get_mean(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(pred - label))


def get_std(pred: np.ndarray, label: np.ndarray) -> float:
    return float(np.std(pred - label))


def plot_distribution(
        pred_res: np.ndarray,
        label_res: np.ndarray,
        save_path: str,
        title: str,
        xlabel: str,
        ylabel: str,
        colormap: str = 'viridis',
):
    assert pred_res.shape == label_res.shape, '[Error] pred_res.shape must equal to label_res.shape'
    header = ['label', 'pred']
    df = []
    for i in range(pred_res.shape[0]):
        df.append([float(label_res[i]), float(pred_res[i])])
    df = pd.DataFrame(df, columns=header)
    min_label = df['label'].min()
    max_label = df['label'].max()
    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=300)
    plt.plot([min_label, max_label], [min_label, max_label], color='grey', linestyle='--', linewidth=1)
    df['dist'] = np.abs(df['label'] - df['pred'])
    dist_max = df['dist'].max()
    dist_min = df['dist'].min()
    # df['dist'] = (df['dist'] - dist_min) / (dist_max - dist_min)
    scatter = ax1.scatter(
        df['label'],
        df['pred'],
        c=df['dist'],
        marker='o',
        s=16,
        edgecolors='none',
        norm=colors.Normalize(vmin=dist_min, vmax=dist_max),
        cmap=colormap,
    )
    ax1.tick_params(axis='both', direction='in')
    ax1.set_xlabel(xlabel, fontweight='bold', fontsize=16)
    ax1.set_ylabel(ylabel, fontweight='bold', fontsize=16)
    ax1.set_title(title, fontweight='bold', fontsize=16)
    # add colorbar by cax
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=dist_min, vmax=dist_max))
    # sm._A = []
    fig.colorbar(scatter, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def metric(
        init_snr: float,
        pred_list_without_id: np.ndarray,
        pred_list_by_snr_without_id: list[np.ndarray],
        evaluation: callable,
        snr_step: int) -> (float, float, float, list[float, float], list[float, float], list[float, float]):
    # snr, pred_mass_m_sun, pred_age_gyr, label_mass_m_sun, label_age_gyr
    _total_mass_m_sun = evaluation(pred_list_without_id[:, 1], pred_list_without_id[:, -2])
    _total_age_gyr = evaluation(pred_list_without_id[:, 2], pred_list_without_id[:, -1])
    # by snr
    _by_snr_mass_m_sun = []
    _by_snr_age_gyr = []
    for i in range(len(pred_list_by_snr_without_id)):
        _by_snr_mass_m_sun.append([
            init_snr + i * snr_step,
            evaluation(pred_list_by_snr_without_id[i][:, 1],
                       pred_list_by_snr_without_id[i][:, -2])
            if len(pred_list_by_snr_without_id[i]) > 0 else 0
        ])
        _by_snr_age_gyr.append([
            init_snr + i * snr_step,
            evaluation(pred_list_by_snr_without_id[i][:, 2],
                       pred_list_by_snr_without_id[i][:, -1])
            if len(pred_list_by_snr_without_id[i]) > 0 else 0
        ])
    return _total_mass_m_sun, _total_age_gyr, _by_snr_mass_m_sun, _by_snr_age_gyr


def plot(evaluation_res: list, save_path: str, title: str, xlabel: str, ylabel: str):
    x_axis = [x[0] for x in evaluation_res]
    y_axis = [x[1] for x in evaluation_res]

    _line = plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x_axis, y_axis)
    _line.savefig(save_path)
    plt.close()


def save_to_npy(pred_res: list[float, float], save_path: str):
    x_axis = [x[0] for x in pred_res]
    y_axis = [x[1] for x in pred_res]
    np.save(os.path.join(save_path), np.array([x_axis, y_axis]))


def save_pred_res(pred: np.ndarray, save_path: str):
    np.save(save_path, pred)  # snr, pred_feh_dex, pred_mass_m_sun, label_feh_dex, label_mass_m_sun


def interpolation_ndarray_1d(
        input_ndarray: np.ndarray,
        input_idx: np.ndarray,
        output_size: int,
) -> np.ndarray:
    """
    Linear interpolation 1D ndarray
    :param input_ndarray: L ndarray
    :param input_idx: L ndarray: wavelength
    :param output_size: int
    :return:
    """
    output_idx = np.linspace(input_idx.min(), input_idx.max(), output_size)
    output_ndarray = np.interp(output_idx, input_idx, input_ndarray)
    return output_ndarray


if __name__ == '__main__':
    model_save_path = 'Your .ckpt weight path'
    cross_val = 'Your dataset cross validation fold name'
    device = torch.device('cuda:0')
    res_save_path = './result_snr10/{}'.format(
        os.path.basename(model_save_path).split('.ckpt')[0]
    )
    step_snr = 10
    attn_threshold = 0.5
    plot_heatmap = True  # if plot se attn heatmap, it cost a lot of time if set True

    pred_list, snr_init, without_id_pred_list, without_id_pred_list_by_snr, se_weights, spec_dataset = inference(
        config,
        model_save_path,
        cross_val,
        device,
        snr_step=step_snr,
        snr_upper_limit=100
    )
    if os.path.exists(os.path.join(res_save_path)):
        import shutil

        shutil.rmtree(res_save_path)
    os.makedirs(os.path.join(res_save_path))
    os.makedirs(os.path.join(res_save_path, 'heatmap', 'pdf'))
    os.makedirs(os.path.join(res_save_path, 'heatmap', 'png'))
    spec_attn_save_path = os.path.join(res_save_path, 'spec_attn')
    save_pred_res(without_id_pred_list, '{}/pred_res.npy'.format(res_save_path))

    # se attn heatmap
    if plot_heatmap:
        for i in tqdm(range(len(se_weights))):
            plot_weight_heatmap(
                se_weights[i]['weight'],
                '{}/se_attn_heatmap_{}.pdf'.format(os.path.join(res_save_path, 'heatmap', 'pdf'),
                                                   se_weights[i]['data_id'][0]),
                norm=True,
            )
            plot_weight_heatmap(
                se_weights[i]['weight'],
                '{}/se_attn_heatmap_{}.png'.format(os.path.join(res_save_path, 'heatmap', 'png'),
                                                   se_weights[i]['data_id'][0]),
                norm=True,
            )

    # mae
    total_mass_m_sun, total_age_gyr, by_snr_mass_m_sun, by_snr_age_gyr = metric(
        snr_init,
        without_id_pred_list,
        without_id_pred_list_by_snr,
        get_mae,
        step_snr
    )
    save_to_npy(by_snr_mass_m_sun, '{}/mass_m_sun_mae_by_snr.npy'.format(res_save_path))
    save_to_npy(by_snr_age_gyr, '{}/age_gyr_mae_by_snr.npy'.format(res_save_path))
    print('[Info] MAE:\ntotal_mass_m_sun: {:.5f}, total_age_gyr: {:.5f}'.format(
        total_mass_m_sun,
        total_age_gyr))
    plot(by_snr_mass_m_sun, os.path.join(res_save_path, 'mae_by_snr_mass_m_sun.pdf'), r'$Mass_\odot$ MAE by SNR', 'SNR',
         'MAE')
    plot(by_snr_age_gyr, os.path.join(res_save_path, 'mae_by_snr_age_gyr.pdf'), r'$Age_{Gyr}$ MAE by SNR', 'SNR', 'MAE')

    # mean
    total_mass_m_sun, total_age_gyr, by_snr_mass_m_sun, by_snr_age_gyr = metric(
        snr_init,
        without_id_pred_list,
        without_id_pred_list_by_snr,
        get_mean,
        step_snr
    )
    save_to_npy(by_snr_mass_m_sun, '{}/mass_m_sun_mean_by_snr.npy'.format(res_save_path))
    save_to_npy(by_snr_age_gyr, '{}/age_gyr_mean_by_snr.npy'.format(res_save_path))
    print('[Info] Mean:\ntotal_mass_m_sun: {:.5f}, total_age_gyr: {:.5f}'.format(
        total_mass_m_sun,
        total_age_gyr))
    plot(by_snr_mass_m_sun, os.path.join(res_save_path, 'mean_by_snr_mass_m_sun.pdf'), r'$Mass_\odot$ Mean by SNR',
         'SNR',
         'Mean')
    plot(by_snr_age_gyr, os.path.join(res_save_path, 'mean_by_snr_age_gyr.pdf'), r'$Age_{Gyr}$ Mean by SNR', 'SNR',
         'Mean')

    # std
    total_mass_m_sun, total_age_gyr, by_snr_mass_m_sun, by_snr_age_gyr = metric(
        snr_init,
        without_id_pred_list,
        without_id_pred_list_by_snr,
        get_std,
        step_snr
    )
    save_to_npy(by_snr_mass_m_sun, '{}/mass_m_sun_std_by_snr.npy'.format(res_save_path))
    save_to_npy(by_snr_age_gyr, '{}/age_gyr_std_by_snr.npy'.format(res_save_path))
    print('[Info] STD:\ntotal_mass_m_sun: {:.5f}, total_age_gyr: {:.5f}'.format(
        total_mass_m_sun,
        total_age_gyr))
    plot(by_snr_mass_m_sun, os.path.join(res_save_path, 'std_by_snr_mass_m_sun.pdf'), r'$Mass_\odot$ Std by SNR', 'SNR',
         'Std')
    plot(by_snr_age_gyr, os.path.join(res_save_path, 'std_by_snr_age_gyr.pdf'), r'$Age_{Gyr}$ Std by SNR', 'SNR', 'Std')

    # distribution .pdf
    plot_distribution(
        without_id_pred_list[:, 1],
        without_id_pred_list[:, -2],
        os.path.join(res_save_path, 'spf_mass_m_sun_distribution.pdf'),
        r'SPFusion Mass ($m_\odot$) Distribution',
        'True Value',
        'Predicted Value'
    )
    plot_distribution(
        without_id_pred_list[:, 2],
        without_id_pred_list[:, -1],
        os.path.join(res_save_path, 'spf_age_gyr_distribution.pdf'),
        r'SPFusion Age (Gyr) Distribution',
        'True Value',
        'Predicted Value'
    )
    # distribution .png
    plot_distribution(
        without_id_pred_list[:, 1],
        without_id_pred_list[:, -2],
        os.path.join(res_save_path, 'spf_mass_m_sun_distribution.png'),
        r'SPFusion Mass ($m_\odot$) Distribution',
        'True Value',
        'Predicted Value'
    )
    plot_distribution(
        without_id_pred_list[:, 2],
        without_id_pred_list[:, -1],
        os.path.join(res_save_path, 'spf_age_gyr_distribution.png'),
        r'SPFusion Age (Gyr) Distribution',
        'True Value',
        'Predicted Value'
    )
