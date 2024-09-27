import argparse
import os

import random
import numpy as np
import torch
from utils.save_data import data_initialization
import wandb
from exp.exp_main import Exp_Main, Exp_Basic



def main():
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='YakirFormer for time series predection')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--train_only', type=bool, default=False,
                        help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id, should be a number')
    parser.add_argument('--model', type=str, required=False, default='HighDimLearner',
                        help='model name, options: [FEDformer,Yakirformer,FreTS,YakirV2,YakirV3]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/traffic/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate (multi features to multi features), S:univariate predict univariate (one feature to one feature), MS:multivariate predict univariate (multi feature to one feature)')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='amount of overlap betwen src and tgt')
    parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')


    #HighDim
    parser.add_argument('--channel_independence', type=str, default='0', help='need to fill')
    parser.add_argument('--embed_size', type=int, default=8, help='embed size for frets')
    parser.add_argument('--hidden_size', type=int, default=256, help='fc hidden layer size')
    parser.add_argument('--n_fft', type=int, default=6, help='embed size for frets')
    parser.add_argument('--hop_length', type=int, default=3, help='embed size for frets')
    parser.add_argument('--TopM', type=int, default=0, help='Amount of Frequenes to choose')
    parser.add_argument('--hide', type=str, default="None",
                        help='Hide real/imagnariy sftf outputs or real/imaganary weights [None / Rinput / Iinput / RWeight / IWeight]')




    # model define Reformer
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')

    #Other Models
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=7, help='dimension of model')

    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')

    parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=False,type=bool, help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--modes', type=int, default=32, help='amount of frequencys to use')
    parser.add_argument('--mode_select', type=str, default="random", help='how to choose the frequences')

    #Fedformer
    parser.add_argument('--version', type=str, default='Wavelets', help='fed former type, Wavelets/Fourier')
    parser.add_argument('--L', type=int, default=1, help='L')
    parser.add_argument('--base', type=str, default='legendre', help='legendre/chvichiv')
    parser.add_argument('--cross_activation', type=str, default='tanh', help='cross activation function')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='after how many rounds of no improvment stop training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer starting learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description ???')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate while running')
    parser.add_argument('--use_amp', action='store_true', help='lower float precesion for faster training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=3, help='gpu number')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus in pararael', default=False)
    parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')

    #log stuff
    parser.add_argument('--run_version', type=str, default='with out res connection', help='runtype')
    parser.add_argument('--log_interval', type=int, default=100, help='log every ith step')
    parser.add_argument('--wandb', type=bool, default=True, help='use wandb for logging')


    args = parser.parse_args()

    if args.wandb:
        wandb.login(key="c539e6a7d6f8cab8c34d65af4fc680be5e3fe58d")
        wandb.init(project="thesis", config={
            "model": args.model,
            "dataset_type": args.data,
            "dataset": args.data_path.replace(".csv",""),
            "dataset_features": args.enc_in,
            "features": args.features,
            "epochs": args.train_epochs,
            "sequence_length": args.seq_len,
            "label_length": args.label_len,
            "prediction_length": args.pred_len,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gpu": args.gpu,
            "channel_independence": args.channel_independence,
            "embed_size": args.embed_size,
            "hidden_size": args.hidden_size,
            "n_fft": args.n_fft,
            "TopM": args.TopM,
            "hop_length": args.hop_length,
            "hide": args.hide,
            "run_version": args.run_version
            # channel
        })

    setting = '{}_{}_{}_{}_features{}_target{}_sl{}_ll{}_pl{}_encin{}_embedsize{}_hiddensize{}_nfft{}_hoplength{}_TopM{}_hide{}_trainepochs{}_batchsize{}_lr{}_gpu{}_runversion{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.data_path.replace(".csv", ""),
        args.features,
        args.target,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.embed_size,
        args.hidden_size,
        args.n_fft,
        args.hop_length,
        args.TopM,
        args.hide,
        args.train_epochs,
        args.batch_size,
        args.learning_rate,
        args.gpu,
        args.run_version,
    )

    #setup gpus
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]



    print('Args in experiment:')
    print(args)


    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            print(f"running on experemient number {ii}")

            # setting record of experiments
            setting = '{}_{}_{}_{}_features{}_target{}_sl{}_ll{}_pl{}_encin{}_embedsize{}_hiddensize{}_nfft{}_hoplength{}_TopM{}_hide{}_trainepochs{}_batchsize{}_lr{}_gpu{}_runversion{}_iter{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.data_path.replace(".csv", ""),
                args.features,
                args.target,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.enc_in,
                args.embed_size,
                args.hidden_size,
                args.n_fft,
                args.hop_length,
                args.TopM,
                args.hide,
                args.train_epochs,
                args.batch_size,
                args.learning_rate,
                args.gpu,
                args.run_version,
                ii
            )

            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting,save_location=False)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting,save_location=False)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
            print(f"done with itr number {ii}")
    else:
        ii = 0
        setting = '{}_{}_{}_{}_features{}_target{}_sl{}_ll{}_pl{}_encin{}_embedsize{}_hiddensize{}_nfft{}_hoplength{}_TopM{}_hide{}_trainepochs{}_batchsize{}_lr{}_gpu{}_runversion{}_iter{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.data_path.replace(".csv", ""),
            args.features,
            args.target,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.enc_in,
            args.embed_size,
            args.hidden_size,
            args.n_fft,
            args.hop_length,
            args.TopM,
            args.hide,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.gpu,
            args.run_version,
            ii
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

    print("finshed running")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
