import argparse
import os
import random
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config.config import config
from dataset.dataset import build_dataloader
from model.build_model import BuildModel
from model.lightning_model import BuildLightningModel
from utils.tools import predict_model_memory_usage, auto_find_memory_free_card, set_random_seed


def train(
        model: pl.LightningModule,
        cross_validation_fold_name: str = 'kfold_0',
):
    if config['run_test_when_training_end']:
        print('[Info] Run test when training end')
    verbose = config['verbose']
    # devices setting
    precision = config['precision']
    predicted_memory_usage = predict_model_memory_usage(
        model=BuildModel(
            spec_in_channel=config['spec_in_channel'],
            photo_in_channel=config['photo_in_channel'],
            star_fusion_kernel_size=config['star_fusion_kernel_size'],
            star_fusion_mlp_ratio=config['star_fusion_mlp_ratio'],
            star_fusion_act_in_photo=config['star_fusion_act_in_photo'],
            dropout=config['dropout'],
            fusion_blk_num_heads=config['fusion_blk_num_heads'],
            trans_blk_num_heads=config['trans_blk_num_heads'],
            num_trans_blk=config['num_trans_blk'],
            final_out_channel=config['final_out_channel'],
            poolformer=config['poolformer'],

            softmax_one=config['softmax_one'],
            photo_timm_model_name=config['photo_timm_model_name'],
            photo_timm_out_indices=config['photo_timm_out_indices'],
            photo_timm_model_out_channel=config['photo_timm_model_out_channel'],
            photo_timm_model_out_feature=config['photo_timm_model_out_feature'],
            spec_out_channel=config['spec_out_channel'],
            spec_out_dim=config['spec_out_dim'],
        ),
        input_shape=[
            (config['batch_size'], config['photo_in_channel'], config['photo_size'], config['photo_size']),
            (config['batch_size'], config['spec_in_channel'], config['spectrum_size'])
        ],
        verbose=verbose,
    )
    used_device = [auto_find_memory_free_card(
        config['used_device'],
        predicted_memory_usage,
        idle=True,
        idle_max_seconds=60 * 60 * 24,
        verbose=verbose,
    )]
    # load dataset
    train_dataloader = build_dataloader(config, mode='train', cross_val_name=cross_validation_fold_name)
    val_dataloader = build_dataloader(config, mode='val', cross_val_name=cross_validation_fold_name)
    # log settings
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print('[Info] Training start time: ', current_time)
    logger_list = []
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    tensorboard_logger = TensorBoardLogger(save_dir=config['log_dir'], name='{}'.format(current_time))
    tensorboard_logger.log_hyperparams(config)
    logger_list.append(tensorboard_logger)
    if not config['debug'] and config['enable_wandb']:
        wandb_logger = WandbLogger(project=config['wandb_project_name'], save_dir=config['log_dir'],
                                   name='{}'.format(current_time))
        logger_list.append(wandb_logger)
    # early stopping
    early_stop_callback = EarlyStopping(
        config['monitor'],
        mode=config['mode'],
        min_delta=config['min_delta'],
        patience=40,
        verbose=True
    )
    # make checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['checkpoint_dir'], '{}'.format(current_time), 'checkpoints'),
        filename='best-{epoch}-{' + config['monitor'] + ':.5f}',
        save_top_k=1,
        monitor=config['monitor'],
        mode=config['mode'],
        save_weights_only=False
    )
    # lr monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=used_device,
        precision=precision,
        logger=logger_list,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback, RichProgressBar()]
        if config['verbose']
        else [checkpoint_callback, lr_monitor, early_stop_callback],
        max_epochs=config['epochs'],
        log_every_n_steps=1,
        enable_progress_bar=config['verbose'],
        check_val_every_n_epoch=1,
        fast_dev_run=config['debug'],
        enable_model_summary=config['verbose'],
    )
    # train
    trainer.fit(model, train_dataloader, val_dataloader)
    if config['run_test_when_training_end']:
        # test
        print('[Info] Start test')
        best_model_path = checkpoint_callback.best_model_path
        print('[Info] best model path: ', best_model_path)
        best_model = BuildLightningModel.load_from_checkpoint(best_model_path)
        best_model.eval()
        test_dataloader = build_dataloader(config, mode='test', cross_val_name=cross_validation_fold_name)
        trainer.test(best_model, test_dataloader)


def set_model_by_config(
        random_seed: int = config['random_seed'],
) -> pl.LightningModule:
    return BuildLightningModel(
        spec_in_channel=config['spec_in_channel'],
        photo_in_channel=config['photo_in_channel'],
        star_fusion_kernel_size=config['star_fusion_kernel_size'],
        star_fusion_mlp_ratio=config['star_fusion_mlp_ratio'],
        star_fusion_act_in_photo=config['star_fusion_act_in_photo'],
        dropout=config['dropout'],
        fusion_blk_num_heads=config['fusion_blk_num_heads'],
        trans_blk_num_heads=config['trans_blk_num_heads'],
        num_trans_blk=config['num_trans_blk'],
        softmax_one=config['softmax_one'],
        loss_fun_name=config['loss_fun_name'],
        regression_losses_weight=config['regression_losses_weight'],
        poolformer=config['poolformer'],

        learn_rate=config['learn_rate'],
        cos_annealing_t_0=config['cos_annealing_t_0'],
        cos_annealing_t_mult=config['cos_annealing_t_mult'],
        cos_annealing_eta_min=config['cos_annealing_eta_min'],
        final_out_channel=config['final_out_channel'],
        optimizer=config['optimizer'],

        random_seed=random_seed,
        photo_timm_model_name=config['photo_timm_model_name'],
        photo_timm_out_indices=config['photo_timm_out_indices'],
        photo_timm_model_out_channel=config['photo_timm_model_out_channel'],
        photo_timm_model_out_feature=config['photo_timm_model_out_feature'],
        spec_out_channel=config['spec_out_channel'],
        spec_out_dim=config['spec_out_dim'],
        enable_torch_2=config['enable_torch_2'],
        torch_2_compile_mode=config['torch_2_compile_mode'],
    )


def train_with_params_search(
        learn_rate: float,
        cos_annealing_t_0: int,
        cos_annealing_t_mult: int,
        cos_annealing_eta_min: float,

        star_fusion_kernel_size: int,
        star_fusion_mlp_ratio: int,
        star_fusion_act_in_photo: bool,
        dropout: float,
        fusion_blk_num_heads: int,
        trans_blk_num_heads: int,
        num_trans_blk: int,
        softmax_one: bool,
        loss_fun_name: str,
        optimizer: str,
        poolformer: bool,

        cross_validation_fold_name: str = 'kfold_0',
):
    if config['wandb_params_search']:
        config['learn_rate'] = learn_rate
        config['cos_annealing_t_0'] = cos_annealing_t_0
        config['cos_annealing_t_mult'] = cos_annealing_t_mult
        config['cos_annealing_eta_min'] = cos_annealing_eta_min

        config['star_fusion_kernel_size'] = star_fusion_kernel_size
        config['star_fusion_mlp_ratio'] = star_fusion_mlp_ratio
        config['star_fusion_act_in_photo'] = star_fusion_act_in_photo
        config['dropout'] = dropout
        config['fusion_blk_num_heads'] = fusion_blk_num_heads
        config['trans_blk_num_heads'] = trans_blk_num_heads
        config['num_trans_blk'] = num_trans_blk
        config['softmax_one'] = softmax_one
        config['loss_fun_name'] = loss_fun_name
        config['optimizer'] = optimizer
        config['poolformer'] = poolformer

    torch.set_float32_matmul_precision('high')
    if config['verbose']:
        print('config:', config)
        # build model
        generated_random_seed = random.randint(0, 10000) \
            if config['enable_random_seed_search'] \
            else config['random_seed']
        set_random_seed(generated_random_seed)
        print('[Info] Random seed: ', generated_random_seed)
        model = set_model_by_config(
            random_seed=generated_random_seed
        )
        train(
            model=model,
            cross_validation_fold_name=cross_validation_fold_name
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_validation_fold_name', type=str, default='kfold_0')

    parser.add_argument('--learn_rate', type=float, default=config['learn_rate'])
    parser.add_argument('--cos_annealing_t_0', type=int, default=config['cos_annealing_t_0'])
    parser.add_argument('--cos_annealing_t_mult', type=int, default=config['cos_annealing_t_mult'])
    parser.add_argument('--cos_annealing_eta_min', type=float, default=config['cos_annealing_eta_min'])

    parser.add_argument('--star_fusion_kernel_size', type=int, default=config['star_fusion_kernel_size'])
    parser.add_argument('--star_fusion_mlp_ratio', type=int, default=config['star_fusion_mlp_ratio'])
    parser.add_argument('--star_fusion_act_in_photo', type=bool, default=config['star_fusion_act_in_photo'])
    parser.add_argument('--dropout', type=float, default=config['dropout'])
    parser.add_argument('--fusion_blk_num_heads', type=int, default=config['fusion_blk_num_heads'])
    parser.add_argument('--trans_blk_num_heads', type=int, default=config['trans_blk_num_heads'])
    parser.add_argument('--num_trans_blk', type=int, default=config['num_trans_blk'])
    parser.add_argument('--softmax_one', type=bool, default=config['softmax_one'])
    parser.add_argument('--loss_fun_name', type=str, default=config['loss_fun_name'])
    parser.add_argument('--optimizer', type=str, default=config['optimizer'])
    parser.add_argument('--poolformer', type=bool, default=config['poolformer'])

    args = parser.parse_args()
    train_with_params_search(**vars(args))
