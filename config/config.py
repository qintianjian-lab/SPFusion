config = {
    # dev mode
    'debug': False,  # pytorch lightning trainer fast_dev_run
    'wandb_project_name': 'Your Project',  # your Wandb project name, if you use wandb
    'enable_wandb': True,  # enable wandb
    'wandb_params_search': True,  # enable wandb hyperparameter search
    'enable_random_seed_search': False,  # enable random seed search
    'verbose': True,  # print log
    'run_test_when_training_end': True,  # run test when training end

    'random_seed': 42,
    'used_device': [0],  # the GPU list you want to use
    'precision': '32-true',
    'dataset_dir': 'Your Dataset Directory',  # where you put the dataset

    'spectrum_dir': 'spectrum',
    'photometric_dir': 'photometric',
    'label_dir': 'label',
    # torch 2.0
    'enable_torch_2': False,
    'torch_2_compile_mode': 'default',  # default, reduce-overhead, max-autotune
    # parameters
    'spectrum_size': 3584,
    'spec_in_channel': 1,
    'photo_in_channel': 3,
    'photo_size': 128,
    'star_fusion_kernel_size': 7,
    'star_fusion_mlp_ratio': 4,
    'star_fusion_act_in_photo': False,
    'fusion_blk_num_heads': 1,
    'trans_blk_num_heads': 4,
    'num_trans_blk': 1,
    'softmax_one': True,
    'loss_fun_name': 'regression',  # regression uncertainty
    'regression_losses_weight': [1e2, 1],
    'photo_timm_model_name': 'resnet10t',
    'photo_timm_out_indices': [1],
    'photo_timm_model_out_channel': 64,
    'photo_timm_model_out_feature': 32 * 32,
    'spec_out_channel': 32,
    'spec_out_dim': 28,
    'poolformer': True,
    'final_out_channel': 2,
    # others
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    'monitor': 'val_loss_epoch',
    'min_delta': 0.005,
    'mode': 'min',
    # model settings
    'batch_size': 32,
    'num_workers': 8,
    'epochs': 150,
    'learn_rate': 1e-3,
    'cos_annealing_t_0': 20,
    'cos_annealing_t_mult': 1,
    'cos_annealing_eta_min': 1e-8,
    'optimizer': 'Adam',
    'dropout': 0.3,
}
