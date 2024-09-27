import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py
import numpy as np
from numpy.lib.stride_tricks import as_strided
import gc
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Default is (0, 1)

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Default is (0, 1)

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Binatix_day(Dataset):
    _loaded_data = None

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data_sizes_dict = [5534, 225, 3345] #N,D,S = 3345
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        def matlab_to_numpy():
            if Dataset_Binatix_day._loaded_data is not None:
                return Dataset_Binatix_day._loaded_data
            with h5py.File(os.path.join(self.root_path, self.data_path), 'r') as file:
                print("Keys: %s" % list(file.keys()))

                # Access the 'features' and 'targets' datasets
                if 'features' in file:
                    features_data = file['features']
                if 'targets' in file:
                    targets_data = file['targets']
                print('converting data to numpy')
                x = features_data[:]
                y = targets_data[:]
            if Dataset_Binatix_day._loaded_data is None:
                Dataset_Binatix_day._loaded_data = (x,y)
            return x,y

        self.x, self.y = matlab_to_numpy()

        # {'train': 0, 'val': 1, 'test': 2}
        train_count = 4500
        val_count = 500-self.seq_len
        test_count = 300+self.seq_len-1
        start_location = [0, train_count, train_count + val_count]
        end_location = [train_count, train_count + val_count, train_count + test_count + val_count]




    def __getitem__(self, index):
        N = 4500 if self.set_type == 0 else 500-self.seq_len if self.set_type == 1 else 300+self.seq_len
        max_N = N - self.pred_len - self.seq_len +1
        stock = (index) //max_N
        index = index%max_N

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = np.nan_to_num(self.x[s_begin:s_end,:,stock], nan=0.0)
        seq_y = np.nan_to_num(self.y[r_begin:r_end,:,stock], nan=0.0)
        seq_x_mark = np.ones_like(seq_x)
        seq_y_mark = np.ones_like(seq_y)


        if seq_x.shape != (96,255):
            print(seq_x.shape)
            print(index,stock)
        return seq_x, seq_y, index,stock+1

    def __len__(self):
        N = 4500 if self.set_type == 0 else 500-self.seq_len if self.set_type == 1 else 300+self.seq_len
        max_N = N-self.pred_len-self.seq_len
        return max_N*self.data_sizes_dict[-1]

    def inverse_transform(self, data):
        return data



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Default is (0, 1)

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Default is (0, 1)

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





class Dataset_Binatix(Dataset):
    binatix_dict = {
        'ccver11a': "CCver11_db_published.mat",
        'ccverU1': "db_U1-20240713.npz"
    }

    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='12', target=None, scale=False,
                 timeenc=0, isrnn=False, isdecoder=False, freq='h',train_valid=False, drop_nan_targets=False, drop_train_samples_method='none',
                 drop_train_samples_threshold=1, loaded_data=None):
        # root_path = "../../data/binatix/"
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = None
        self.scale = scale
        self.timeenc = timeenc
        self.isrnn = isrnn
        self.isdecoder = isdecoder
        self.freq = freq
        self.train_valid = train_valid
        self.drop_nan_targets = drop_nan_targets if flag != 'test' else False
        self.drop_train_samples_method = drop_train_samples_method
        self.drop_train_samples_threshold = drop_train_samples_threshold

        self.root_path = root_path
        self.data_path = data_path
        if '11' in data_path:
            self.time_idx = [158, 159, 160, 161]
        elif 'U1' in data_path:
            self.time_idx = [134,135,136,137,138,139,140,141,142,143,144]
        else:
            self.time_idx = [99, 100, 101, 102]  # for '12' in data_path
        self.__read_data__(loaded_data)

    def __read_data__(self, loaded_data):
        self.scaler = StandardScaler()
        self.inst_weights = torch.tensor(
            hdf5storage.loadmat(os.path.join(self.root_path, 'instruments_weights.mat'))['weights']
        )
        self.inst_groups = torch.tensor(
            hdf5storage.loadmat(os.path.join(self.root_path, 'instruments_groups.mat'))['groups']
        )

        if loaded_data is None:
            # opening binatix's data
            try:
                print('Loading data from .hkl file :', f"dataset/binatix/ccver{self.data_path}.hkl")
                data_dict = hkl.load(os.path.join(self.root_path,self.data_path+".hkl"))
                # data_dict = pickle.load(open(f"dataset/binatix/ccver{self.data_path}.pkl", "rb"))
            except (OSError, IOError, ValueError) as e:
                print('Loading data from .mat file :', self.root_path, Dataset_Binatix.binatix_dict[self.data_path])
                path_url = os.path.join(self.root_path, Dataset_Binatix.binatix_dict[self.data_path])
                if 'npz' in path_url:
                    data_dict = np.load(path_url)
                    data_dict = [item for item in data_dict.values()]
                    data_dict = {'features': data_dict[0], 'targets': data_dict[1]}
                    data_dict = {'features': data_dict[0[:5]], 'targets': data_dict[1][:5]}
                else:
                    data_dict = mat73.loadmat(path_url)
                hkl.dump(data_dict, f'dataset/binatix/ccver{self.data_path}.hkl', mode='w')
                # pickle.dump(data_dict, open(f"dataset/binatix/ccver{self.data_path}.pkl", "wb"), protocol=4)
            if type(data_dict) != dict:
                data_dict['features'], data_dict['targets'] = [item[0] for item in data_dict.values()]

            if 'Att2a' in self.data_path:   # cut Attn2a data to length of 5534
                data_dict['features'] = data_dict['features'][..., :5534]
                data_dict['targets'] = data_dict['targets'][..., :5534]

            if 'npz' in Dataset_Binatix.binatix_dict[self.data_path]:
                data_time = data_dict['features'][..., self.time_idx]
                data_raw = np.delete(data_dict['features'], self.time_idx, axis=-1)
                data_tar = data_dict['targets']
            else:
                data_time = data_dict['features'].swapaxes(1, 2)[..., self.time_idx]
                data_raw = np.delete(data_dict['features'].swapaxes(1, 2), self.time_idx, axis=-1)
                data_tar = np.expand_dims(data_dict['targets'], 2)
            # TODO: ----------------------------------------------------------------
            # TODO: | CAUTION!!!! do not use y_{t-1} as an input to predict y_{t}  |
            # TODO: ----------------------------------------------------------------
            '''
            data_raw: bs X timesteps X ['date'(x4), ...(other features), target feature]
            '''
            # data = np.zeros((3410, 5534, 86))
            # for i in range(3410):
            #     for j in range(5534):
            #         for k in range(86):
            #             data[i][j][k] = i * 1e10 + j * 1e5 + k
            if not self.isrnn and not self.isdecoder:
                # settings the target as the next time-step:
                data_tar = np.roll(data_tar, 1, axis=1)
                data_tar[:, 0] = 0.

            self.data_raw = np.concatenate([data_time, data_raw, data_tar], axis=-1)
            # # dropping instruments which has less than 80% valid nonans on the training set ~ 1416 instruments
            # valid_80per_nonan = (np.isnan(data_tar[:, :4200, 0]).sum(axis=-1) / 4200 > 0.8)
            # valid_80per_nonan_len = valid_80per_nonan.sum()
            # self.data_raw = self.data_raw[valid_80per_nonan]
            data_raw = self.data_raw
        else:
            data_raw = loaded_data

        if self.train_valid:
            num_train = 5000
            num_vali = 0
        else:
            num_train = 4200
            num_vali = 800
        num_test = data_raw.shape[1] - num_train - num_vali

        if self.isrnn:
            border1s = [0, num_train, data_raw.shape[1] - num_test]
        elif self.isdecoder:
            border1s = [0, num_train - self.seq_len + 1, data_raw.shape[1] - num_test - self.seq_len + 1]
        else:
            border1s = [0, num_train - self.seq_len, data_raw.shape[1] - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, data_raw.shape[1]]

        # border1s = [0, 68, 168]     # TEST ONLY!
        # border2s = [100, 200, 300]  # TEST ONLY!
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.isrnn:
            start_index_for_features = 0
        else:
            start_index_for_features = 4        # without timestamps
        if self.features == 'MS':
            data = data_raw[..., start_index_for_features:]  # all but seasonal
        elif self.features == 'S':
            data = data_raw[..., -1:]  # only target
        elif self.features == 'M':
            data = data_raw[..., start_index_for_features:]  # all but seasonal

        stamp = data_raw[..., :4]

        self.data_x = data[:, border1:border2]
        self.data_y = data[:, border1:border2]
        self.data_stamp = stamp[:, border1:border2]
        self.data_nonan = ~np.isnan(self.data_y)
        # dropping the samples if the LAST time of the input has more than THRESHOLD nonans
        if self.drop_train_samples_method == 'last':
            # Compute the mean of each row along the time axis, excluding the last column
            means = self.data_nonan[:, :-1, :-1].mean(axis=2)
            # Create a boolean mask indicating where the mean is below the threshold
            mask = means < self.drop_train_samples_threshold
            # Shift the mask by one time-step and set the first column to False
            mask = np.concatenate([np.zeros((mask.shape[0], 1)), mask], axis=1)
            mask[:, 0] = False
            # Apply the mask to the target column
            self.data_nonan[:, :, -1] = np.where(mask, False, self.data_nonan[:, :, -1])

        # dropping the samples if the SEQ_LEN time-steps of the input has more than THRESHOLD nonans
        elif self.drop_train_samples_method == 'all':
            # Compute the mean of the last L time steps for each row
            means = self.data_nonan[:, :-1, :-1].mean(axis=2)
            means = np.apply_along_axis(lambda x: np.convolve(x, np.ones(self.seq_len), mode='valid') / self.seq_len,
                                        axis=1, arr=means)
            # Create a boolean mask indicating where the mean is below the threshold
            mask = means < self.drop_train_samples_threshold
            # Shift the mask by one time-step and set the first column to False
            mask = np.concatenate([np.zeros((mask.shape[0], self.seq_len)), mask], axis=1)
            mask[:, :self.seq_len] = False
            # Apply the mask to the target column
            self.data_nonan[:, :, -1] = np.where(mask, False, self.data_nonan[:, :, -1])

        elif self.drop_train_samples_method == 'weighted':
            # TODO: weighted drop_train_samples_threshold
            pass

        # replacing nan with 0:
        np.nan_to_num(self.data_y, copy=False)
        np.nan_to_num(self.data_x, copy=False)

    def __getitem__(self):
        pass

    def __len__(self):
        pass

    def inverse_transform(self, data):
        return data


class Dataset_Binatix_Rnn(Dataset_Binatix):

    def __init__(self, *args, **kwargs):
        super(Dataset_Binatix_Rnn, self).__init__(*args, **kwargs)
        self.sampler = self.create_sampler(0)
        self.label_len = 0
        self.seq_len = 0

    def __getitem__(self, index):
        # if self.set_type == 0:
        index = self.sampler[index]

        # fixing the item getter for binatix [bs x ts x feats] - instruments first
        n_inst = index % self.data_x.shape[0]
        s_begin = int(index / self.data_x.shape[0])

        s_end = s_begin + self.pred_len
        r_begin = s_begin
        r_end = r_begin + self.pred_len
        # s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[n_inst, s_begin:s_end]
        seq_y = self.data_y[n_inst, r_begin:r_end]
        seq_x_mark = self.data_stamp[n_inst, s_begin:s_end]
        seq_y_mark = self.data_stamp[n_inst, r_begin:r_end]
        seq_nonan = self.data_nonan[n_inst, r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_nonan

    def __len__(self):
        # return len(self.sampler)
        return int(np.ceil(self.data_x.shape[1] / self.pred_len)) * self.data_x.shape[0]

    def create_sampler(self, start_idx):
        assert start_idx >= 0 and start_idx < self.seq_len + self.pred_len, "Start index must be in [0, seq_len+pred_len-1]"
        # timesteps = np.arange(start_idx, self.data_x.shape[0], self.seq_len + self.pred_len)
        # indices = np.arange(0, self.data_x.shape[0] * (self.data_x.shape[1] - self.seq_len - self.pred_len + 1))\
        #     .reshape(-1, self.data_x.shape[0]).transpose()
        # self.sampler = indices[:, timesteps].transpose().reshape(-1)
        timesteps = np.arange(start_idx, self.data_x.shape[1], self.pred_len)
        indices = np.arange(0, self.data_x.shape[0] * self.data_x.shape[1]).reshape(-1, self.data_x.shape[0]).transpose()

        return indices[:, timesteps].transpose().reshape(-1)

class Dataset_Binatix_Atn(Dataset_Binatix):

    def __init__(self, *args, **kwargs):
        super(Dataset_Binatix_Atn, self).__init__(*args, **kwargs)
        self.sampler =self.create_sampler()

    def __getitem__(self, index):
        if self.drop_nan_targets:
            index = self.sampler[index]

        # fixing the item getter for binatix [bs x ts x feats] - instruments first
        n_inst = index % self.data_x.shape[0]
        s_begin = int(index / self.data_x.shape[0])

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len        # TODO: decoder problem input should be 1 index before output

        seq_x = self.data_x[n_inst, s_begin:s_end]
        seq_y = self.data_y[n_inst, r_begin:r_end]
        seq_x_mark = self.data_stamp[n_inst, s_begin:s_end]
        seq_y_mark = self.data_stamp[n_inst, r_begin:r_end]
        seq_nonan = self.data_nonan[n_inst, r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_nonan

    def __len__(self):
        if self.drop_nan_targets:   # Training only - dropping nan targets while training
            return len(self.sampler)
        else:
            return self.data_x.shape[0] * (self.data_x.shape[1] - self.seq_len - self.pred_len + 1)

    def create_sampler(self):
        # order indices that __getitem__ takes
        return np.argwhere(
            self.data_nonan[:, -(self.data_nonan.shape[1] - self.seq_len - self.pred_len + 1):, -1].transpose().reshape(
                -1)
        ).squeeze()
