import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError

from model.build_model import BuildModel
from model.loss import Loss


class BuildLightningModel(pl.LightningModule):
    def __init__(
            self,
            spec_in_channel: int,
            photo_in_channel: int,
            star_fusion_kernel_size: int,
            star_fusion_mlp_ratio: int,
            star_fusion_act_in_photo: bool,
            dropout: float,
            fusion_blk_num_heads: int,
            trans_blk_num_heads: int,
            num_trans_blk: int,
            softmax_one: bool,
            loss_fun_name: str,
            regression_losses_weight: list[float, float],
            poolformer: bool,

            learn_rate: float,
            cos_annealing_t_0: int,
            cos_annealing_t_mult: int,
            cos_annealing_eta_min: float,
            final_out_channel: int,
            optimizer: str,

            random_seed: int = 42,
            photo_timm_model_name: str = 'fastvit_t8',
            photo_timm_out_indices: list = None,
            photo_timm_model_out_channel: int = 192,
            photo_timm_model_out_feature: int = 8 * 8,
            spec_out_channel: int = 32,
            spec_out_dim: int = 28,
            enable_torch_2: bool = False,
            torch_2_compile_mode: str = 'default',
    ):
        super().__init__()
        print('[Info] Using Random Seed: ', random_seed)
        assert torch_2_compile_mode in ['default', 'reduce-overhead', 'max-autotune'], \
            '[Error] torch_2_compile_mode must be in [default, reduce-overhead, max-autotune]'
        self.model = BuildModel(
            spec_in_channel=spec_in_channel,
            photo_in_channel=photo_in_channel,
            star_fusion_kernel_size=star_fusion_kernel_size,
            star_fusion_mlp_ratio=star_fusion_mlp_ratio,
            star_fusion_act_in_photo=star_fusion_act_in_photo,
            dropout=dropout,
            fusion_blk_num_heads=fusion_blk_num_heads,
            trans_blk_num_heads=trans_blk_num_heads,
            num_trans_blk=num_trans_blk,
            final_out_channel=final_out_channel,
            poolformer=poolformer,

            softmax_one=softmax_one,
            photo_timm_model_name=photo_timm_model_name,
            photo_timm_out_indices=photo_timm_out_indices,
            photo_timm_model_out_channel=photo_timm_model_out_channel,
            photo_timm_model_out_feature=photo_timm_model_out_feature,
            spec_out_channel=spec_out_channel,
            spec_out_dim=spec_out_dim,
        )
        if enable_torch_2:
            print('[Info] Using PyTorch 2.0 compile')
        self.learn_rate = learn_rate
        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.criterion = Loss(
            loss_fun_name=loss_fun_name,
            out_channel=final_out_channel,
            loss_weight=regression_losses_weight,
        )
        self.optimizer = optimizer
        self.train_mass_m_sun_mae = MeanAbsoluteError()
        self.train_age_gyr_mae = MeanAbsoluteError()
        self.val_mass_m_sun_mae = MeanAbsoluteError()
        self.val_age_gyr_mae = MeanAbsoluteError()
        self.test_mass_m_sun_mae = MeanAbsoluteError()
        self.test_age_gyr_mae = MeanAbsoluteError()
        self.best_val_mass_m_sun_mae = 9e10
        self.best_val_age_gyr_mae = 9e10

        self.val_loss_epoch = 0
        self.step = 0
        self.epoch_index = 0
        self.best_val_loss = 9e10

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        data_id, _, _, _, spectrum, photometric, label = batch
        pred, _, _ = self.model(photometric, spectrum)
        loss, unweighted_losses, weighted_losses = self.criterion(pred, label)
        self.log('train_loss', loss, prog_bar=True, on_step=True, batch_size=photometric.shape[0])
        for _index, _unweighted_loss in enumerate(unweighted_losses):
            self.log('train_unweighted_loss_{}'.format(_index + 1), _unweighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])
        for _index, _weighted_loss in enumerate(weighted_losses):
            self.log('train_weighted_loss_{}'.format(_index + 1), _weighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])

        pred_mass_m_sun, pred_age_gyr = pred[:, 0], pred[:, 1]
        target_mass_m_sun, target_age_gyr = label[:, 0], label[:, 1]

        self.train_mass_m_sun_mae(pred_mass_m_sun, target_mass_m_sun)
        self.log('train_mass_m_sun_mae', self.train_mass_m_sun_mae, prog_bar=True, batch_size=photometric.size(0))

        self.train_age_gyr_mae(pred_age_gyr, target_age_gyr)
        self.log('train_age_gyr_mae', self.train_age_gyr_mae, prog_bar=True, batch_size=photometric.size(0))

        return loss

    def on_train_epoch_end(self):
        self.epoch_index += 1

    def on_validation_epoch_start(self):
        self.val_loss_epoch = 0
        self.step = 0

    def validation_step(self, batch, batch_idx):
        data_id, _, _, _, spectrum, photometric, label = batch
        pred, _, _ = self.model(photometric, spectrum)
        loss, unweighted_losses, weighted_losses = self.criterion(pred, label)
        self.val_loss_epoch += loss
        self.step += 1
        self.log('val_loss', loss, prog_bar=True, on_step=True, batch_size=photometric.shape[0])
        for _index, _unweighted_loss in enumerate(unweighted_losses):
            self.log('val_unweighted_loss_{}'.format(_index + 1), _unweighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])
        for _index, _weighted_loss in enumerate(weighted_losses):
            self.log('val_weighted_loss_{}'.format(_index + 1), _weighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])

        pred_mass_m_sun, pred_age_gyr = pred[:, 0], pred[:, 1]
        target_mass_m_sun, target_age_gyr = label[:, 0], label[:, 1]
        # check is nan in pred or label
        if torch.isnan(pred).any() or torch.isnan(label).any():
            print('[Error] pred pr label has nan\n')
            print('pred: ', pred)
            print('label: ', label)
            raise ValueError

        self.val_mass_m_sun_mae(pred_mass_m_sun, target_mass_m_sun)
        self.log('val_mass_m_sun_mae', self.val_mass_m_sun_mae, prog_bar=True, batch_size=photometric.size(0))

        self.val_age_gyr_mae(pred_age_gyr, target_age_gyr)
        self.log('val_age_gyr_mae', self.val_age_gyr_mae, prog_bar=True, batch_size=photometric.size(0))

        return loss

    def on_validation_epoch_end(self):
        _val_loss = self.val_loss_epoch / self.step
        self.log('val_loss_epoch', _val_loss, prog_bar=True, on_epoch=True)
        if _val_loss < self.best_val_loss:
            self.best_val_loss = _val_loss
            self.log('best_val_loss_epoch', self.best_val_loss, prog_bar=True, on_epoch=True)

        _val_mass_m_sun_mae = self.val_mass_m_sun_mae.compute()
        if _val_mass_m_sun_mae < self.best_val_mass_m_sun_mae:
            self.best_val_mass_m_sun_mae = _val_mass_m_sun_mae
            self.log('best_val_mass_m_sun_mae', self.best_val_mass_m_sun_mae, prog_bar=True, on_epoch=True)
        self.log('val_mass_m_sun_mae', _val_mass_m_sun_mae, prog_bar=True, on_epoch=True)
        self.val_mass_m_sun_mae.reset()

        _val_age_gyr_mae = self.val_age_gyr_mae.compute()
        if _val_age_gyr_mae < self.best_val_age_gyr_mae:
            self.best_val_age_gyr_mae = _val_age_gyr_mae
            self.log('best_val_age_gyr_mae', self.best_val_age_gyr_mae, prog_bar=True, on_epoch=True)
        self.log('val_age_gyr_mae', _val_age_gyr_mae, prog_bar=True, on_epoch=True)
        self.val_age_gyr_mae.reset()

    def test_step(self, batch, batch_idx):
        data_id, _, _, _, spectrum, photometric, label = batch
        pred, _, _ = self.model(photometric, spectrum)
        loss, unweighted_losses, weighted_losses = self.criterion(pred, label)
        self.log('test_loss', loss, prog_bar=True, on_step=True, batch_size=photometric.shape[0])
        for _index, _unweighted_loss in enumerate(unweighted_losses):
            self.log('test_unweighted_loss_{}'.format(_index + 1), _unweighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])
        for _index, _weighted_loss in enumerate(weighted_losses):
            self.log('test_weighted_loss_{}'.format(_index + 1), _weighted_loss, prog_bar=False, on_step=True,
                     batch_size=photometric.shape[0])

        pred_mass_m_sun, pred_age_gyr = pred[:, 0], pred[:, 1]
        target_mass_m_sun, target_age_gyr = label[:, 0], label[:, 1]

        self.test_mass_m_sun_mae(pred_mass_m_sun, target_mass_m_sun)
        self.test_age_gyr_mae(pred_age_gyr, target_age_gyr)

        return loss

    def on_test_epoch_end(self):
        _test_age_gyr_mae = self.test_age_gyr_mae.compute()
        self.log('test_age_gyr_mae', _test_age_gyr_mae, prog_bar=True, on_epoch=True)
        self.test_age_gyr_mae.reset()

    def configure_optimizers(self):
        optimizer = eval('torch.optim.{}'.format(self.optimizer))(
            self.parameters(),
            lr=self.learn_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cos_annealing_t_0,
            T_mult=self.cos_annealing_t_mult,
            eta_min=self.cos_annealing_eta_min
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'name': 'lr'
            },
        }
