import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import pandas as pd
import os
import datetime


plt.switch_backend('agg')

def caculate_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_memory_bytes = total_params * 4
    total_memory_gb = total_memory_bytes / (1024 ** 3)
    print(f'Total parameters: {total_params}')
    print(f'Total memory (GB): {total_memory_gb:.6f}')
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.savefig(name, bbox_inches='tight')

def add_logs(args):
    name = '{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_dt{distil}_{des}'.format(
        model_id=args.model_id,
        model=args.model,
        data=args.data,
        features=args.features,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        embed=args.embed,
        distil=args.distil,
        des=args.des)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    log_path = vars(args)['log_path'] + "_" + name
    logger.addHandler(logging.FileHandler(log_path + '.log', 'a'))
    return logger



def signal_to_csv(save, timesteps, prediction, name=None, data=12):
    # Sending a signal 'prediction' to a csv file for blackbox
    # prediction must be ordered by time and by instruments (0 to 3345...)
    # prediction is [ts x inst x 1]
    start, end = timesteps  # timesteps is a tuple of prediction timestep
    postfix = ""
    if data == 'Att2a':
        type = "A2"
        postfix = "ed"
    elif end == 5281:
        type = 'A'
    elif end == 5533:
        type = 'A1'
    else:
        type = 'A1'
    print("type", type)
    now = f'{datetime.datetime.now():%Y%m%d-%H%M%S}'
    name = 'test-{}'.format(now) if name is None else name + "-{}".format(now)
    csv_path = os.path.join(save, f'{name}.request.csv')
    if '11' in data:
        data = 11
    elif '12' in data:
        data = 12
    print("data", data)
    json_dict = {"model_name": "{NAME}".format(NAME=name),
                 "create_time": "{DATE}".format(DATE=now),
                 "BB_eval_type": "{TYPE}".format(TYPE=type),
                 "fdb": {"train": "CCver{DATA}_db_publish{POSTFIX}.mat".format(DATA=data, POSTFIX=postfix),
                         "evaluation": "CCver{DATA}_db_publish{POSTFIX}.mat".format(DATA=data, POSTFIX=postfix)}}

    with open(csv_path, "wt") as fp:
        fp.write("# {}\n".format(json_dict).replace("""'""", '"'))  # json dictionary as a comment
    records = []
    range_ts = range(prediction.shape[0]) if isinstance(prediction, np.ndarray) else range(prediction.size(0))
    range_inst = range(prediction.shape[1]) if isinstance(prediction, np.ndarray) else range(prediction.size(1))
    for ts, tmp_ts in zip(range(start, end + 1), range_ts):
        for inst in range_inst:
            # ts+1 is because of 0 timestep!
            # inst+1 is because of instrument are from 1 to 3345
            records += [[ts + 1, inst + 1, round(prediction[tmp_ts, inst, -1].item(), 6)]]

    df = pd.DataFrame(records, columns=["Timestep_idx", "Instrument_idx", "Prediction"])
    df.to_csv(csv_path, mode='a', index=False)

    print("CSV ready!")


def blackbox_signal_sender(target_out, args, save_file=None):
    timesteps = (0, 533 if 'a' in args.data_path else 281)

    if "11a" in args.data_path:
        d_type = '11a'
    elif "11b" in args.data_path:
        d_type = '11b'
    elif "11" in args.data_path:
        d_type = '11'
    elif "12a" in args.data_path:
        d_type = '12a'
    elif "12" in args.data_path:
        d_type = '12'
    elif "Att2a" in args.data_path:
        d_type = 'Att2a'
    elif "U1" in args.data_path:
        d_type = 'U1'

    torch.save(target_out, os.path.join(args.save, f'test_res_{d_type}.pt'))
    save_file = save_file if save_file is not None else args.save
    signal_to_csv(save_file, timesteps=(timesteps[0]+5000, timesteps[1]+5000), prediction=target_out,
                   name=f'test_{d_type}', data=d_type)

