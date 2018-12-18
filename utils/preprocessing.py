import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime as dt
import gc

from sklearn import preprocessing
from sklearn.utils import resample
import scipy


def preprocess_cat_columns(data, categorical_columns):
    print('preprocessing categorical columns...')
    for cat_col in categorical_columns:
        if cat_col not in data.columns:
            continue
        print('preprocessing categorical columns...{}'.format(cat_col))
        if data[cat_col].count() != data.shape[0]:
            data[cat_col] = data[cat_col].fillna(data[cat_col].median(axis=0), axis=0)

        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[cat_col].values.astype('str')))
        data.loc[:, cat_col] = lbl.transform(list(data[cat_col].values.astype('str')))

    print('preprocessing categorical columns...end')
    return data


def preprocess_num_columns(data, numerical_columns, nan_value=0, type=float, normolize=True):
    print('preprocessing numerical columns...')
    for num_col in numerical_columns:
        if num_col not in data.columns:
            continue
        print('preprocessing numerical columns...{}'.format(num_col))
        if data[num_col].count() != data.shape[0]:
            data[num_col] = data[num_col].fillna(nan_value)

        if data[num_col].dtypes == 'object' and type == float:
            data[num_col] = ('0' + data[num_col].str.replace(',', '.')).astype(type)
            data[num_col] = data[num_col].fillna(nan_value)
        elif type == int:
            data[num_col] = data[num_col].astype(type)

        if normolize == True:
            data[num_col] = (data[num_col] - data[num_col].mean()) / data[num_col].std()

    print('preprocessing numerical columns...end')
    return data


def preprocess_date_columns(data, date_col, fill_value=None, format_str='%Y%m%d'):
    if date_col not in data.columns:
        return data

    print('preprocessing date columns...{}'.format(date_col))
    if fill_value == None:
        counter = Counter(data[date_col].tolist())
        average_value = counter.most_common(1)[0][0]
        data[date_col] = data[date_col].fillna(average_value, axis=0)
    else:
        data[date_col] = data[date_col].fillna(fill_value, axis=0)

    date_col_formated = data[date_col].apply(lambda x: dt.strptime(str(x), format_str))
    data.loc[:, date_col + '_month'] = date_col_formated.apply(lambda x: x.month)
    data.loc[:, date_col + '_day'] = date_col_formated.apply(lambda x: x.day)
    # data.loc[:, date_col + '_weekday'] = date_col_formated.apply(lambda x: x.weekday())
    del date_col_formated
    del data[date_col]

    return data


def delete_noninformative_columns(data):
    for c in data.columns:
        num_unique = len(data[c].unique())
        print("Count unique '{0}': {1}".format(c, num_unique))
        if num_unique < 2:
            data = data.drop(columns=[c])
    return data


def aggregate_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.

    Parameters
    --------
        df (dataframe):
            the dataframe to calculate the statistics on
        group_var (string):
            the variable by which to group df
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated for
            all numeric columns. Each instance of the grouping variable will have
            the statistics (mean, min, max, sum; currently supported) calculated.
            The columns are also renamed to keep track of features created.

    """
    print('aggregate numeric columns...')
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(
        ['count', 'mean', 'max', 'min', 'sum', lambda x: scipy.stats.mode(x)[0]]).reset_index()

    # Need to create new column names
    columns = [group_var]
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = list(map(lambda x: x.replace('<lambda>', 'mode'), columns))
    return agg


def generate_features(data):
    data_cell_avail = data[['CELL_AVAILABILITY_4G_sum', 'CELL_AVAILABILITY_3G_sum', 'CELL_AVAILABILITY_2G_sum']]
    max_value_ind = np.argmax(data_cell_avail.values, axis=1)

    data['BAD_INTERNET'] = pd.Series(np.zeros(data.shape[0]))
    cell = data['INTERNET_TYPE_ID'] < max_value_ind + 1
    data.loc[cell, 'BAD_INTERNET'] = 1

    data['DATA_SPEED'] = pd.Series(np.zeros(data.shape[0]))
    cell = data['SUM_DATA_MIN'] != 0
    data.loc[cell, 'DATA_SPEED'] = data['SUM_DATA_MB'] / data['SUM_DATA_MIN']
    # data['DATA_SPEED'] = data['SUM_DATA_MB'] / data['SUM_DATA_MIN']
    data['COSTS'] = data['ITC'] + data['VAS'] + data['RENT_CHANNEL'] + data['ROAM']

    # Strength features
    # data['COM_CAT#20+COM_CAT#22'] = 0.5*data['COM_CAT#20'] + 0.5*data['COM_CAT#22']
    # data['REVENUE+COM_CAT#23'] = 0.5*data['REVENUE'] + 0.5*data['COM_CAT#23']
    # data['ITC+COM_CAT#28'] = 0.5*data['ITC'] + 0.5*data['COM_CAT#28']
    # data = data.drop(columns=['ITC', 'REVENUE', 'COM_CAT#20', 'COM_CAT#22', 'COM_CAT#23'], axis=1)
    return data


def separate_xy(data, dropNaN=False, info=False):
    # Удалить SK_ID для обучения
    X = data.drop(('CSI'), axis=1)
    if dropNaN != True:
        data['CSI'] = data['CSI'].fillna(0)  # ПОПРОБОВАТЬ УДАЛИТЬ Nan
    else:
        data.dropna(subset=['CSI'], inplace=True)
    y = data['CSI'].astype(int)

    if info:
        print('Labels: {}'.format(set(y)))
        print('Zero count: {}, One count: {}'.format(len(y) - sum(y), sum(y)))

    return X, y


def upsampling(data):
    df_majority = data[data['CSI'] == 0]
    df_minority = data[data['CSI'] == 1]

    df_minority_upsampling = resample(df_minority, replace=True,
                                      n_samples=df_majority.shape[0], random_state=123)

    df_upsampling = pd.concat([df_majority, df_minority_upsampling])
    print(df_upsampling['CSI'].value_counts())
    return df_upsampling


def merge_features(csi, features, params=None):
    categorical_columns = ['COM_CAT#1', 'COM_CAT#2', 'COM_CAT#3', 'BASE_TYPE', 'ACT', 'ARPU_GROUP',
                           'COM_CAT#7', 'DEVICE_TYPE_ID', 'INTERNET_TYPE_ID', 'COM_CAT#25', 'COM_CAT#26',
                           'COM_CAT#34']
    numerical_columns_float = ['REVENUE', 'ITC', 'VAS', 'RENT_CHANNEL', 'ROAM', 'COST', 'COM_CAT#8',
                               'COM_CAT#17', 'COM_CAT#18', 'COM_CAT#19', 'COM_CAT#20', 'COM_CAT#21',
                               'COM_CAT#22', 'COM_CAT#23', 'COM_CAT#24', 'COM_CAT#27', 'COM_CAT#28',
                               'COM_CAT#29', 'COM_CAT#30', 'COM_CAT#31', 'COM_CAT#32', 'COM_CAT#33']
    delete_columns = ['COM_CAT#3', 'COM_CAT#24']

    data = pd.merge(csi, features, left_on='SK_ID', right_on='SK_ID', how='left')  # data.loc[data['SK_ID'] == 3851]
    # del csi, features

    # data_sorted = data.sort_values(by=['SK_ID'])
    print(data.shape)
    print(data.info())

    # delete  non-informative columns
    # for del_col in delete_columns:
    #     del data[del_col]

    # categorical columns
    for cat in categorical_columns:
        if cat not in params['categorical_columns']:
            params['categorical_columns'].append(cat)

    # numerical_columns
    for num in numerical_columns_float:
        if num not in params['numerical_columns']:
            params['numerical_columns'].append(num)

    # numerical columns float (fillna, mean, std)
    data = preprocess_num_columns(data, numerical_columns_float, type=float, normolize=False)

    # date columns
    data = preprocess_date_columns(data, 'CONTACT_DATE', format_str='%d.%m')
    data = preprocess_date_columns(data, 'SNAP_DATE', format_str='%d.%m.%y')

    # delete non-informative columns count_unique = 1
    data = delete_noninformative_columns(data)

    print(data.info())
    # scatter_matrix(data, alpha=0.05)
    corr = data.corr()

    return data


def merge_consum(data, consum, params=None):
    print('Count SK_ID before merging: {}'.format(len(set(data['SK_ID']))))
    numerical_columns_float = ['SUM_MINUTES', 'SUM_DATA_MB', 'SUM_DATA_MIN']
    numerical_columns_int = ['CELL_LAC_ID']

    print(consum.shape)
    print(consum.info())

    # numerical_columns
    for num in numerical_columns_float:
        if num not in params['numerical_columns']:
            params['numerical_columns'].append(num)

    consum = preprocess_num_columns(consum, numerical_columns_float, type=float, normolize=False)
    consum['MON'] = consum['MON'].apply(lambda x: dt.strptime(str(x), '%d.%m').month)

    if params['mode'] == 'train':
        data = pd.merge(data, consum, left_on=['SK_ID', 'SNAP_DATE_month'], right_on=['SK_ID', 'MON'], how='inner')
    if params['mode'] == 'test':
        data = pd.merge(data, consum, left_on=['SK_ID', 'SNAP_DATE_month'], right_on=['SK_ID', 'MON'], how='left')
    del data['MON']
    # del consum

    print(data.info())
    # corr = data.corr()
    print('Count SK_ID after merging: {}'.format(len(set(data['SK_ID']))))
    # data.to_csv('data/data-after-consum_{}.csv'.format(params['mode']), index=False)

    return data


def merge_voice_session(data, voice_session, params=None):
    numerical_columns_float = ['VOICE_DUR_MIN']
    numerical_columns_int = ['CELL_LAC_ID']

    print(voice_session.shape)
    print(voice_session.info())

    # numerical_columns
    for num in numerical_columns_float:
        if num not in params['numerical_columns']:
            params['numerical_columns'].append(num)

    # numerical columns float (fillna, mean, std)
    voice_session = preprocess_num_columns(voice_session, numerical_columns_float, type=float, normolize=False)

    voice_session = preprocess_date_columns(voice_session, 'START_TIME', fill_value=1, format_str='%d.%m %H:%M:%S')

    # delete non-informative columns count_unique = 1
    voice_session = delete_noninformative_columns(voice_session)

    print(voice_session.info())
    corr = voice_session.corr()

    voice_session = voice_session.groupby(['START_TIME_day', 'START_TIME_month', 'SK_ID',
                                           'CELL_LAC_ID'], sort=False).sum().reset_index()
    voice_session = voice_session[voice_session['VOICE_DUR_MIN'] > 0]

    data = pd.merge(data, voice_session, left_on=['SK_ID', 'CELL_LAC_ID', 'SNAP_DATE_month'],
                    right_on=['SK_ID', 'CELL_LAC_ID', 'START_TIME_month'], how='left')
    del data['SNAP_DATE_month']
    # del voice_session

    return data


def merge_data_session(data, data_session, params=None):
    numerical_columns_float = ['DATA_VOL_MB']
    numerical_columns_int = ['CELL_LAC_ID']

    print(data_session.shape)
    print(data_session.info())

    # numerical_columns
    for num in numerical_columns_float:
        if num not in params['numerical_columns']:
            params['numerical_columns'].append(num)

    # numerical columns float (fillna, mean, std)
    data_session = preprocess_num_columns(data_session, numerical_columns_float, type=float, normolize=False)

    data_session = preprocess_date_columns(data_session, 'START_TIME', fill_value=1, format_str='%d.%m %H:%M:%S')

    # delete non-informative columns count_unique = 1
    data_session = delete_noninformative_columns(data_session)

    data_session = data_session.groupby(['START_TIME_day', 'START_TIME_month', 'SK_ID',
                                         'CELL_LAC_ID'], sort=False).sum().reset_index()
    data_session = data_session[data_session['DATA_VOL_MB'] > 0]

    data = pd.merge(data, data_session, left_on=['SK_ID', 'CELL_LAC_ID', 'START_TIME_day', 'START_TIME_month'],
                    right_on=['SK_ID', 'CELL_LAC_ID', 'START_TIME_day' 'START_TIME_month'], how='left')
    # del voice_session

    print(data_session.info())
    corr = data_session.corr()

    return data


def preprocess_avg_kpi(avg_kpi):
    date_column = 'T_DATE'
    numerical_columns_int = ['CELL_LAC_ID']
    numerical_columns_float = ['PART_CQI_QPSK_LTE', 'PART_MCS_QPSK_LTE', 'RBU_AVAIL_DL', 'RBU_AVAIL_UL', 'RBU_OTHER_DL',
                               'RBU_OTHER_UL', 'RBU_OWN_DL', 'RBU_OWN_UL', 'SHO_FACTOR', 'UTIL_CE_DL_3G',
                               'UL_VOLUME_LTE', 'DL_VOLUME_LTE', 'TOTAL_DL_VOLUME_3G', 'TOTAL_UL_VOLUME_3G']

    cell_availability = ['CELL_AVAILABILITY_2G', 'CELL_AVAILABILITY_3G', 'CELL_AVAILABILITY_4G', 'CSSR_2G', 'CSSR_3G',
                         'PSSR_2G', 'PSSR_3G', 'PSSR_LTE']

    blocking = ['ERAB_PS_BLOCKING_RATE_LTE', 'ERAB_PS_BLOCKING_RATE_PLMN_LTE', 'ERAB_PS_DROP_RATE_LTE',
                'RAB_CS_BLOCKING_RATE_3G', 'RAB_CS_DROP_RATE_3G', 'RAB_PS_BLOCKING_RATE_3G', 'RAB_PS_DROP_RATE_3G',
                'RRC_BLOCKING_RATE_3G', 'RRC_BLOCKING_RATE_LTE', 'TBF_DROP_RATE_2G', 'TCH_DROP_RATE_2G']

    delete_columns = ['HSPDSCH_CODE_UTIL_3G', 'NODEB_CNBAP_LOAD_HARDWARE', 'PROC_LOAD_3G', 'RBU_AVAIL_DL_LTE',
                      'RTWP_3G', 'UTIL_BRD_CPU_3G', 'UTIL_CE_HW_DL_3G', 'UTIL_CE_UL_3G', 'UTIL_SUBUNITS_3G']

    print(avg_kpi.shape)
    print(avg_kpi.info())

    # delete  non-informative columns
    avg_kpi = avg_kpi.drop(delete_columns, axis=1)
    gc.collect()

    # fill cell_availability
    nan_value = 0
    avg_kpi = preprocess_num_columns(avg_kpi, cell_availability, nan_value=nan_value, normolize=False)

    cell_2g = (avg_kpi['CELL_AVAILABILITY_2G'] == nan_value) & (
            (avg_kpi['CELL_AVAILABILITY_3G'] != nan_value) | (avg_kpi['CELL_AVAILABILITY_4G'] != nan_value))
    avg_kpi.loc[cell_2g, 'CELL_AVAILABILITY_2G'] = 1
    avg_kpi['CELL_AVAILABILITY_2G_sum'] = avg_kpi['CELL_AVAILABILITY_2G'] + avg_kpi['CSSR_2G'] + avg_kpi['PSSR_2G']
    column_nan_values = avg_kpi['CELL_AVAILABILITY_2G_sum'] == nan_value
    avg_kpi.loc[column_nan_values, 'CELL_AVAILABILITY_2G_sum'] = 0.5
    # cell_2g_intersect = (avg_kpi['PSSR_2G'] != nan_value) & (avg_kpi['CSSR_2G'] != nan_value)
    # avg_kpi.loc[cell_2g_intersect, 'CSSR_2G'] = (avg_kpi.loc[cell_2g_intersect, 'PSSR_2G'] +
    #                                              avg_kpi.loc[cell_2g_intersect, 'CSSR_2G']) / 2
    # cell_2g_intersect = (avg_kpi['CELL_AVAILABILITY_2G'] != nan_value) & (avg_kpi['CSSR_2G'] != nan_value)
    # avg_kpi.loc[cell_2g_intersect, 'CELL_AVAILABILITY_2G'] = (avg_kpi.loc[cell_2g_intersect, 'CELL_AVAILABILITY_2G'] +
    #                                                           avg_kpi.loc[cell_2g_intersect, 'CSSR_2G']) / 2

    cell_3g = (avg_kpi['CELL_AVAILABILITY_3G'] == nan_value) & (avg_kpi['CELL_AVAILABILITY_4G'] != nan_value)
    avg_kpi.loc[cell_3g, 'CELL_AVAILABILITY_3G'] = 1
    avg_kpi['CELL_AVAILABILITY_3G_sum'] = avg_kpi['CELL_AVAILABILITY_3G'] + avg_kpi['CSSR_3G'] + avg_kpi['PSSR_3G']
    column_nan_values = avg_kpi['CELL_AVAILABILITY_3G_sum'] == nan_value
    avg_kpi.loc[column_nan_values, 'CELL_AVAILABILITY_3G_sum'] = 0.5

    avg_kpi['CELL_AVAILABILITY_4G_sum'] = avg_kpi['CELL_AVAILABILITY_4G'] + avg_kpi['PSSR_LTE']
    column_nan_values = avg_kpi['CELL_AVAILABILITY_4G_sum'] == nan_value
    avg_kpi.loc[column_nan_values, 'CELL_AVAILABILITY_4G_sum'] = 0.5
    avg_kpi = avg_kpi.drop(cell_availability, axis=1)

    # fill blocking
    avg_kpi = preprocess_num_columns(avg_kpi, blocking, normolize=False)
    avg_kpi['DROP_RATE_2G_sum'] = avg_kpi['TBF_DROP_RATE_2G'] + avg_kpi['TCH_DROP_RATE_2G']
    avg_kpi['BLOCKING_RATE_3G_sum'] = avg_kpi['RAB_CS_BLOCKING_RATE_3G'] + avg_kpi['RAB_CS_DROP_RATE_3G'] + \
                                      avg_kpi['RAB_PS_BLOCKING_RATE_3G'] + avg_kpi['RAB_PS_DROP_RATE_3G'] + \
                                      avg_kpi['RRC_BLOCKING_RATE_3G']
    avg_kpi['BLOCKING_RATE_LTE_sum'] = avg_kpi['ERAB_PS_BLOCKING_RATE_LTE'] + avg_kpi['ERAB_PS_BLOCKING_RATE_PLMN_LTE'] \
                                       + avg_kpi['ERAB_PS_DROP_RATE_LTE'] + avg_kpi['RRC_BLOCKING_RATE_LTE']
    avg_kpi = avg_kpi.drop(blocking, axis=1)

    # fill other numerical
    avg_kpi = preprocess_num_columns(avg_kpi, numerical_columns_float, normolize=False)
    avg_kpi['PART_CQI_LTE'] = avg_kpi['PART_CQI_QPSK_LTE'] + avg_kpi['PART_MCS_QPSK_LTE']
    avg_kpi['RBU_AVAIL'] = avg_kpi['RBU_AVAIL_DL'] + avg_kpi['RBU_AVAIL_UL']
    avg_kpi['RBU_OTHER_LTE'] = avg_kpi['RBU_OTHER_DL'] + avg_kpi['RBU_OTHER_UL']
    avg_kpi['RBU_OWN_LTE'] = avg_kpi['RBU_OWN_DL'] + avg_kpi['RBU_OWN_UL']
    avg_kpi['VOLUME_LTE'] = avg_kpi['DL_VOLUME_LTE'] + avg_kpi['UL_VOLUME_LTE']
    avg_kpi['VOLUME_3G'] = avg_kpi['TOTAL_DL_VOLUME_3G'] + avg_kpi['TOTAL_UL_VOLUME_3G']
    delete_columns = ['PART_CQI_QPSK_LTE', 'PART_MCS_QPSK_LTE', 'RBU_AVAIL_DL', 'RBU_AVAIL_UL', 'RBU_OTHER_DL',
                      'RBU_OTHER_UL', 'RBU_OWN_DL', 'RBU_OWN_UL', 'UL_VOLUME_LTE', 'DL_VOLUME_LTE',
                      'TOTAL_DL_VOLUME_3G', 'TOTAL_UL_VOLUME_3G']
    avg_kpi = avg_kpi.drop(delete_columns, axis=1)

    # fill date
    avg_kpi = preprocess_date_columns(avg_kpi, date_column, format_str='%d.%m')

    avg_kpi = avg_kpi.groupby(['T_DATE_month', 'CELL_LAC_ID'], sort=False).sum().reset_index()
    del avg_kpi['T_DATE_day']

    avg_kpi.to_csv('data/avg_kpi.csv', index=False)


def merge_data_avg_kpi(data, avg_kpi, params=None):
    numerical_columns = ['VOLUME_3G', 'VOLUME_LTE', 'RBU_OWN_LTE', 'RBU_OTHER_LTE', 'RBU_AVAIL', 'PART_CQI_LTE',
                         'BLOCKING_RATE_LTE_sum', 'BLOCKING_RATE_3G_sum', 'DROP_RATE_2G_sum',
                         'CELL_AVAILABILITY_4G_sum', 'CELL_AVAILABILITY_3G_sum', 'CELL_AVAILABILITY_2G_sum']

    for num in numerical_columns:
        if num not in params['numerical_columns']:
            params['numerical_columns'].append(num)

    print('Count SK_ID before merging: {}'.format(len(set(data['SK_ID']))))
    if params['mode'] == 'train':
        data = pd.merge(data, avg_kpi, left_on=['CELL_LAC_ID', 'SNAP_DATE_month'],
                        right_on=['CELL_LAC_ID', 'T_DATE_month'], how='inner')
    if params['mode'] == 'test':
        data = pd.merge(data, avg_kpi, left_on=['CELL_LAC_ID', 'SNAP_DATE_month'],
                        right_on=['CELL_LAC_ID', 'T_DATE_month'], how='left')
    del data['T_DATE_month']

    print('Count SK_ID after merging: {}'.format(len(set(data['SK_ID']))))
    # data.to_csv('data/data-after-avg_kpi_{}.csv'.format(params['mode']), index=False)
    print(data.info())
    # corr = data.corr()

    return data


def load_data(paths_csv, params):
    n_rows = params['n_rows']
    mode = params['mode']

    # merged different csv
    csi = pd.read_csv(paths_csv[0], sep=';', nrows=n_rows)
    features = pd.read_csv(paths_csv[1], sep=';', nrows=n_rows)
    data = merge_features(csi, features, params)

    consum = pd.read_csv(paths_csv[2], sep=';', nrows=n_rows)
    data = merge_consum(data, consum, params)

    # voice_session = pd.read_csv(paths_csv[3], sep=';', nrows=n_rows)
    # data = merge_voice_session(data, voice_session, params)

    # data_session = pd.read_csv(paths_csv[4], sep=';', nrows=n_rows)
    # data = merge_data_session(data, data_session, params)

    avg_kpi = pd.read_csv(paths_csv[5], sep=',', nrows=n_rows)
    data = merge_data_avg_kpi(data, avg_kpi, params)

    # fill nan values
    for col in data.columns:
        if data[col].count() != data.shape[0] and (col not in params['categorical_columns']):
            data[col] = data[col].fillna(data[col].median(axis=0), axis=0)

    # generate new features
    data = generate_features(data)

    # groupby SK_ID
    data = data.drop('CELL_LAC_ID', axis=1)
    mean_columns = params['categorical_columns'].copy()
    mean_columns.extend(['SK_ID', 'CSI', 'CONTACT_DATE_day', 'SNAP_DATE_month'])
    data_mean = data.loc[:, mean_columns].groupby(['SK_ID', 'SNAP_DATE_month'], sort=False).mean().reset_index()

    sum_columns = []
    for col in data.columns:
        if col not in mean_columns:
            sum_columns.append(col)
    sum_columns.extend(['SK_ID', 'SNAP_DATE_month'])
    data_sum = data.loc[:, sum_columns].groupby(['SK_ID', 'SNAP_DATE_month'], sort=False).sum().reset_index()

    data = pd.merge(data_mean, data_sum, left_on=['SK_ID', 'SNAP_DATE_month'],
                    right_on=['SK_ID', 'SNAP_DATE_month'], how='inner')

    # aggregate numeric columns
    agg_num = aggregate_numeric(data[params['numerical_columns'] + ['SK_ID']], group_var='SK_ID', df_name='')
    data = pd.merge(data, agg_num, left_on='SK_ID', right_on='SK_ID', how='left')

    # chnn_kpi = pd.read_csv(paths_csv[6], sep=';', nrows=n_rows)

    if mode == 'train':
        corr = data.corr()
        corr_sort = corr.sort_values('CSI', ascending=False)['CSI']
        print("10 top linear corr with target")
        for i in range(1, 11):
            print('feature {}: {}'.format(corr_sort.index[i], corr_sort.iloc[i]))

    print(data.shape)

    return data
