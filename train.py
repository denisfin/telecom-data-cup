import pandas as pd
import os
import pickle

from utils import preprocessing as prc
from utils import train_utils
import timeit


def data_preprocess(base_folder, train_folder, test_folder, params):
    paths_train = [os.path.join(train_folder, 'subs_csi_train.csv'),
                   os.path.join(train_folder, 'subs_features_train.csv'),
                   os.path.join(train_folder, 'subs_bs_consumption_train.csv'),
                   os.path.join(train_folder, 'subs_bs_voice_session_train.csv'),
                   os.path.join(train_folder, 'subs_bs_data_session_train.csv'),
                   'data/avg_kpi.csv',
                   os.path.join(base_folder, 'bs_chnn_kpi.csv')]

    paths_test = [os.path.join(test_folder, 'subs_csi_test.csv'),
                  os.path.join(test_folder, 'subs_features_test.csv'),
                  os.path.join(test_folder, 'subs_bs_consumption_test.csv'),
                  os.path.join(test_folder, 'subs_bs_voice_session_test.csv'),
                  os.path.join(test_folder, 'subs_bs_data_session_test.csv'),
                  'data/avg_kpi.csv',
                  os.path.join(base_folder, 'bs_chnn_kpi.csv')]

    params['mode'] = 'train'
    data_train = prc.load_data(paths_train, params)
    params['mode'] = 'test'
    data_test = prc.load_data(paths_test, params)

    # preprocessing Merged Train and Test data
    data = pd.concat([data_train, data_test], sort=False)
    trn_len = data_train.shape[0]

    # categorical columns
    data = prc.preprocess_cat_columns(data, params['categorical_columns'])
    # delete columns with nan values
    for col in data.columns:
        if data[col].count() != data.shape[0] and col != 'CSI':
            print('delete {}'.format(col))
            del data[col]

    # Back split data
    data_train = data[:trn_len]
    data_test = data[trn_len:]

    return data_train, data_test


def main():
    start = timeit.default_timer()
    base_folder = 'C:/condition_csi/dataset'
    train_folder = os.path.join(base_folder, 'train')
    test_folder = os.path.join(base_folder, 'test')
    do_preprocessing = False

    # preprocessing
    if do_preprocessing:
        params = dict(my_random_state=255,
                      n_rows=None,
                      categorical_columns=[],
                      numerical_columns=[],
                      mode=None)
        train_df, test_df = data_preprocess(base_folder, train_folder, test_folder, params)
        train_df.to_csv('data/train_df.csv', index=False)
        test_df.to_csv('data/test_df.csv', index=False)

        # save params
        with open('data/params.pkl', 'wb') as file:
            pickle.dump(params, file, pickle.HIGHEST_PROTOCOL)

    else:
        # load params
        with open('data/params.pkl', 'rb') as file:
            params = pickle.load(file)

        train_df = pd.read_csv('data/train_df.csv', nrows=params['n_rows'])
        test_df = pd.read_csv('data/test_df.csv', nrows=params['n_rows'])


    # TRAIN MODELS
    print('training...')
    train_utils.train(train_df, test_df, params)
    stop = timeit.default_timer()
    print('Time: {0} sec.'.format(int(stop - start)))


if __name__ == "__main__":
    main()