import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='SparseTSF & other models for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--is_transfering', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='SparseTSF', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# MixLinear
parser.add_argument('--alpha', type=float, default=0.5, help='MixLinear factor')
parser.add_argument('--lpf', type=int, default=15, help='MixLinear factor')
parser.add_argument('--swt_init', type=str, default='random', help='MixLinear SWT init type: random/haar/db2')
parser.add_argument('--swt_levels', type=int, default=2, help='MixLinear SWT decomposition levels')
parser.add_argument('--ablation_mode', type=str, default='original',
                    choices=['original', 'trend_only', 'detail_only'],
                    help='MixDLinear ablation mode')
parser.add_argument('--segment_num', type=int, default=24, help='MixLinear segment count for trend branch')
parser.add_argument('--num_bands', type=int, default=3, help='AdaBandLinear number of frequency bands')
parser.add_argument('--band_rank', type=int, default=8, help='AdaBandLinear low-rank dimension for band heads')
parser.add_argument('--freq_top_k', type=int, default=8, help='WaveMix Track-A retained low-frequency bins in FFT')
parser.add_argument('--attn_top_k', type=int, default=3, help='WaveMix Track-B retained neighbors in myopic inverted attention')
parser.add_argument('--recent_len', type=int, default=96, help='WaveMix Track-B recent window length for inverted embedding')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--com_len', type=int, default=1, help='compression length')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # FITS with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--iter_max', type=int, default=10, help='max iterations for iterative refinement')
parser.add_argument('--iter_patience', type=int, default=3, help='early stopping patience for iterative refinement')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=4, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed_list = range(2023, 2033)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    log_file = os.path.join('./logs', f'iterative_refinement_{args.model}_{args.data}_{args.seq_len}_{args.pred_len}.log')
    with open(log_file, 'a') as f:
        f.write(f'=== Iterative Refinement Log ===\n')
        f.write(f'Model: {args.model}, Data: {args.data}, seq_len: {args.seq_len}, pred_len: {args.pred_len}\n')
        f.write(f'Params: alpha={args.alpha}, lpf={args.lpf}, swt_init={args.swt_init}, swt_levels={args.swt_levels}\n')
        f.write(f'iter_max={args.iter_max}, iter_patience={args.iter_patience}\n')
        f.write(f'================================\n\n')

    best_mse = float('inf')
    best_setting = ''
    best_iter = 0
    no_improve_count = 0

    for ii in range(args.iter_max):
        seed_idx = ii % len(fix_seed_list)
        random.seed(fix_seed_list[seed_idx])
        torch.manual_seed(fix_seed_list[seed_idx])
        np.random.seed(fix_seed_list[seed_idx])
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.des,
            args.alpha,
            args.lpf,
            ii,
            fix_seed_list[seed_idx])

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training [{}/{}] : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ii + 1, args.iter_max, setting))
        exp.train(setting)

        print('>>>>>>>testing [{}/{}] : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(ii + 1, args.iter_max, setting))
        mse = exp.test(setting)

        with open(log_file, 'a') as f:
            f.write(f'Iter {ii+1}: setting={setting}, MSE={mse:.7f}\n')

        if mse < best_mse:
            best_mse = mse
            best_setting = setting
            best_iter = ii + 1
            no_improve_count = 0
            print('*** New best MSE: {:.7f} (iter {}, setting: {})'.format(best_mse, ii + 1, setting))
        else:
            no_improve_count += 1
            print('*** No improvement ({}/{}). Best MSE: {:.7f}'.format(no_improve_count, args.iter_patience, best_mse))

        if no_improve_count >= args.iter_patience:
            print('*** Early stopping: no improvement for {} consecutive iterations.'.format(args.iter_patience))
            with open(log_file, 'a') as f:
                f.write('Early stopped at iter {}: no improvement for {} consecutive iterations.\n'.format(ii + 1, args.iter_patience))
            break

        torch.cuda.empty_cache()

    with open(log_file, 'a') as f:
        f.write('\n=== Best result ===\n')
        f.write('Best iter: {}, Best MSE: {:.7f}, Setting: {}\n'.format(best_iter, best_mse, best_setting))
        f.write('================================\n\n')

    print('=== Final best MSE: {:.7f} (iter {}, setting: {}) ==='.format(best_mse, best_iter, best_setting))
else:


       ii = 0
       setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_{}_seed{}'.format(
           args.model_id,
           args.model,
           args.data,
           args.features,
           args.seq_len,
           args.pred_len,
           args.des,
           args.alpha,
           args.lpf,
           ii,
           fix_seed_list[ii])

       exp = Exp(args)  # set experiments
       print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
       exp.test(setting, test=1)
       torch.cuda.empty_cache()

