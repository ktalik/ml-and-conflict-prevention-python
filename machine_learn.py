#!/usr/bin/env python2

'''
Python rewrite and extension of *Machine Learning and Conflict Prevention*
available at <https://github.com/ktalik/ml-and-conflict-prevention>

Copyright (C) Konrad Talik <konrad.talik@slimak.matinf.uj.edu.pl>

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this code.  If not, see <http://www.gnu.org/licenses/>.
'''


import numpy as np
import pandas as pd
import time

import matplotlib

from pandas import read_csv
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')


##
# Constants
##

COLUMNS_SORTED_ACC = [
    'battles.index',
    'land.conf.norm',
    'pop.sum',
    'pop.mean.lag',
    'gdp.mean.lag',
    'pop.mean',
    'flood.freq.mean',
    'uw.perc.mean',
    'gdp.mean.change.lag.2',
    'gdp.mean.change',
    'pop.mean.lag.2',
    'gdp.mean',
    'pop.sum.lag.2',
    'gdp.mean.sum.lag.2',
    'u5pop.mean',
    'gdp.mean.sum.lag',
    'pop.sum.lag',
    'gov.share.1',
    'gdp.mean.lag.2',
    'gdp.mean.change',
    'drought.freq.mean',
    'MEANfatalities.lagged.3',
    'imr.perc.mean',
    'gdp.mean.sum',
    'ethnic.comp',
    'SUMfatalities.lagged.3',
    'oppo.share.1',
    'gov.share.2',
    'oppo.share.3',
    'oppo.share.2'
]

COLUMNS_FROM_PAPER = [
    'ID',
    'YEAR',
    'ethnic.comp',
    'land.conf.norm',
    'flood.freq.mean',
    'drought.freq.mean',
    'lootable.diamonds',
    'petrol',
    'u5pop.mean',
    'uw.perc.mean',
    'imr.perc.mean',
    'gdp.mean',
    'gdp.mean.lag',
    'gdp.mean.lag.2',
    'gdp.mean.change',
    'gdp.mean.change.lag',
    'gdp.mean.change.lag.2',
    'gdp.mean.sum',
    'gdp.mean.sum.lag',
    'gdp.mean.sum.lag.2',
    'pop.mean',
    'pop.mean.lag',
    'pop.mean.lag.2',
    'pop.sum',
    'pop.sum.lag',
    'pop.sum.lag.2',
    'SUMfatalities.index',
    'SUMfatalities.lagged',
    'SUMfatalities.lagged.2',
    'MEANfatalities.index',
    'MEANfatalities.lagged',
    'MEANfatalities.lagged.3',
    'battles.index',
    # Whether head of government is from military
    'military',
    # The vote share of government and opposition coalitions in the legislative branch
    'gov.share',
    'gov.share.1',
    'gov.share.2',
    'gov.share.3',
    'oppo.share',
    'oppo.share.1',
    'oppo.share.2',
    'oppo.share.3'
]

#LEARNING_COLUMNS = COLUMNS_SORTED_ACC
LEARNING_COLUMNS = COLUMNS_FROM_PAPER
LEARN_ON_EQUAL_POWER = True
BATTLE_CLASS_POWER_MULTIPLIER = 1  # 5

ALL_COL_1997 = 9

START_YEAR = 1997
END_YEAR = 2012

TARGET_COLUMN = 'dummy.battles'  # 'dummy.battles'

PLOT_KDE = False
PLOT_LEARNING = True
PLOT_SCATTER_MATRIX = False
PLOT_ITER_COLUMNS = 3
PLOT_COLUMNS = LEARNING_COLUMNS[:PLOT_ITER_COLUMNS]
PLOT_COLORS = ['blue', 'red']
PLOT_DATA = 250*1000

PLOT_SAVE_DIR = 'figures/'

##
# Utils
##

def colored_scatter_matrix(data, colors, title, save=None):
    """ Scatter matrix with parametrized colors (e.g. classes) """
    print 'Plot scatter matrix...'
    fig, ax = plt.subplots(figsize=(12.0, 7.5))
    pd.scatter_matrix(
        data,
        diagonal='kde',
        figsize=(10, 10),
        ax=ax,
        c=colors,
        cmap=None
    )
    ax.set_title(title)
    if save:
        fig.savefig(save)
    else:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()


def plot_kde(data, colors=None, title=None, save=None):
    """ Plot KDE of input data DataFrame """
    print 'Plot kde...'
    fig, ax = plt.subplots(figsize=(12.0, 7.5))
    data.plot(kind='kde', ax=ax, c=colors, cmap=None, legend=False)
    ax.set_title(title)
    if save:
        fig.savefig(save)
    else:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

def init_learning_plot():
    """ Initialize plotting interactive mode """
    plt.ion()

def update_learning_plot(train_acc, valid_acc, valid_0_acc, valid_1_acc):
    """ Update plot that displays learning results """
    plt.clf()
    if not train_acc or not valid_0_acc or not valid_1_acc:
        return
    valid_0_line, = plt.plot(
        valid_0_acc,
        'g',
        label='Validation ACC of 0: {:.2f}%'.format(valid_0_acc[-1])
    )
    valid_1_line, = plt.plot(
        valid_1_acc,
        'r',
        label='Validation ACC of 1: {:.2f}%'.format(valid_1_acc[-1])
    )
    learn_line, = plt.plot(
        train_acc,
        'b',
        label='Learning ACC: {:.2f}%'.format(train_acc[-1])
    )
    valid_line, = plt.plot(
        valid_acc,
        'k',
        label='Validation ACC: {:.2f}%'.format(valid_acc[-1])
    )
    plt.legend(
        loc='lower left',
        handles=[learn_line, valid_line, valid_0_line, valid_1_line])
    plt.show()
    plt.pause(0.001)

def save_learning_plot(filename='current_learning.png'):
    plt.savefig(PLOT_SAVE_DIR + filename)

##
# Preprocessing
##

def preprocess_data():
    """ Load all available input data and return them as single DataFrame """

    ACLED = dict()

    # Read data by looping over yearly conflict data as well as suplementary data

    print 'Read data...'
    for i in range(START_YEAR, END_YEAR + 1):
        current_year = i
        ACLED[current_year] = read_csv('DATA/Conflict/{}.csv'.format(current_year))

    diamonds = read_csv('DATA/diamonds.csv')
    districts = read_csv('DATA/districts.csv')
    ethnic_comp = read_csv('DATA/ethnic.comp.csv')
    gdp_mean = read_csv('DATA/gdp.mean.csv')
    gdp_mean_change = read_csv('DATA/gdp.mean.change.csv')
    gdp_sum = read_csv('DATA/gdp.sum.csv')
    hazard = read_csv('DATA/hazard.csv')
    land = read_csv('DATA/land.csv')
    petrol = read_csv('DATA/petrol.csv')
    pop_mean = read_csv('DATA/pop.mean.csv')
    pop_sum = read_csv('DATA/pop.sum.csv')
    poverty = read_csv('DATA/u5-imr.csv')
    dpi = read_csv('DATA/dpi.csv')

    # Sort

    print 'Sort data...'
    for year in ACLED.keys():
        ACLED[year] = ACLED[year].sort(columns=['OBJECTID'])

    diamonds = diamonds.sort(columns=['OBJECTID'])
    districts = districts.sort(columns=['OBJECTID'])
    ethnic_comp = ethnic_comp.sort(columns=['OBJECTID'])
    hazard = hazard.sort(columns=['OBJECTID'])
    land = land.sort(columns=['OBJECTID'])
    petrol = petrol.sort(columns=['OBJECTID'])
    poverty = poverty.sort(columns=['OBJECTID'])

    # Merge dataframes column-wise

    print 'Merge dataframes column-wise...'
    for i, current_year in enumerate(ACLED.keys()):

        df = pd.concat([
            ACLED[current_year],
            diamonds, districts, ethnic_comp, hazard, land, petrol, poverty],
            axis=1
        )

        df2 = df[['OBJECTID', 'ISO']]
        df2['YEAR'] = current_year
        df2['ethnic.comp'] = df['COUNT']

        for col in [
            'land.conf.norm', 'flood.freq.mean', 'drought.freq.mean', 'petrol',
            'u5pop.mean', 'imr.perc.mean', 'uw.perc.mean'
        ]:
            df2[col] = df[col]

        df2['lootable.diamonds'] = df['Ldia']

        # Indexing and for-loop in R starts from 1, in Python they start from 0.
        # However as to for-loop iteration there is no difference.

        df2['gdp.mean']                 = gdp_mean.ix[:, 9 + i]
        df2['gdp.mean.lag']             = gdp_mean.ix[:, 8 + i]
        df2['gdp.mean.lag.2']           = gdp_mean.ix[:, 7 + i]
        df2['gdp.mean.change']          = gdp_mean_change.ix[:, 9 + i]
        df2['gdp.mean.change.lag']      = gdp_mean_change.ix[:, 8 + i]
        df2['gdp.mean.change.lag.2']    = gdp_mean_change.ix[:, 7 + i]
        df2['gdp.mean.sum']             = gdp_sum.ix[:, 9 + i]
        df2['gdp.mean.sum.lag']         = gdp_sum.ix[:, 8 + i]
        df2['gdp.mean.sum.lag.2']       = gdp_sum.ix[:, 7 + i]
        df2['pop.mean']                 = pop_mean.ix[:, 9 + i]
        df2['pop.mean.lag']             = pop_mean.ix[:, 8 + i]
        df2['pop.mean.lag.2']           = pop_mean.ix[:, 7 + i]
        df2['pop.sum']                  = pop_sum.ix[:, 9 + i]
        df2['pop.sum.lag']              = pop_sum.ix[:, 8 + i]
        df2['pop.sum.lag.2']            = pop_sum.ix[:, 7 + i]

        df2['SUMfatalities']    = df.ix[:, 5]
        df2['MEANfatalities']   = df.ix[:, 6]
        df2['battles']          = df.ix[:, 4]

        df2['SUMfatalities.lagged']     = np.nan
        df2['SUMfatalities.lagged.2']   = np.nan
        df2['SUMfatalities.lagged.3']   = np.nan
        df2['SUMfatalities.index']      = np.nan

        df2['MEANfatalities.lagged']    = np.nan
        df2['MEANfatalities.lagged.2']  = np.nan
        df2['MEANfatalities.lagged.3']  = np.nan
        df2['MEANfatalities.index']     = np.nan

        df2['battles.lagged']   = np.nan
        df2['battles.lagged.2'] = np.nan
        df2['battles.lagged.3'] = np.nan
        df2['battles.index']    = np.nan

        if (current_year < 1998):
            pass

        elif (current_year < 1999):
            df2['SUMfatalities.lagged'] = ACLED[current_year - 1].ix[:, 5]
            df2['SUMfatalities.index']  = ACLED[current_year - 1].ix[:, 5]

            df2['MEANfatalities.lagged']    = ACLED[current_year - 1].ix[:, 6]
            df2['MEANfatalities.index']     = ACLED[current_year - 1].ix[:, 6]

            df2['battles.lagged']   = ACLED[current_year - 1].ix[:, 4]
            df2['battles.index']    = ACLED[current_year - 1].ix[:, 4]

        elif (current_year < 2000):
            df2['SUMfatalities.lagged']     = ACLED[current_year - 1].ix[:, 5]
            df2['SUMfatalities.lagged.2']   = ACLED[current_year - 2].ix[:, 5]
            df2['SUMfatalities.index']      = \
                ACLED[current_year - 1].ix[:, 5] \
                + (ACLED[current_year - 2].ix[:, 5] * .5)

            df2['MEANfatalities.lagged']     = ACLED[current_year - 1].ix[:, 6]
            df2['MEANfatalities.lagged.2']   = ACLED[current_year - 2].ix[:, 6]
            df2['MEANfatalities.index']      = \
                ACLED[current_year - 1].ix[:, 6] \
                + (ACLED[current_year - 2].ix[:, 6] * .5)

            df2['battles.lagged']     = ACLED[current_year - 1].ix[:, 4]
            df2['battles.lagged.2']   = ACLED[current_year - 2].ix[:, 4]
            df2['battles.index']      = \
                ACLED[current_year - 1].ix[:, 4] \
                + (ACLED[current_year - 2].ix[:, 4] * .5)

        else:
            df2['SUMfatalities.lagged']     = ACLED[current_year - 1].ix[:, 5]
            df2['SUMfatalities.lagged.2']   = ACLED[current_year - 2].ix[:, 5]
            df2['SUMfatalities.lagged.3']   = ACLED[current_year - 3].ix[:, 5]
            df2['SUMfatalities.index']      = \
                ACLED[current_year - 1].ix[:, 5] \
                + (ACLED[current_year - 2].ix[:, 5] * .5) \
                + (ACLED[current_year - 3].ix[:, 5] * .25)

            df2['MEANfatalities.lagged']     = ACLED[current_year - 1].ix[:, 6]
            df2['MEANfatalities.lagged.2']   = ACLED[current_year - 2].ix[:, 6]
            df2['MEANfatalities.lagged.3']   = ACLED[current_year - 3].ix[:, 6]
            df2['MEANfatalities.index']      = \
                ACLED[current_year - 1].ix[:, 6] \
                + (ACLED[current_year - 2].ix[:, 6] * .5) \
                + (ACLED[current_year - 3].ix[:, 6] * .25)

            df2['battles.lagged']     = ACLED[current_year - 1].ix[:, 4]
            df2['battles.lagged.2']   = ACLED[current_year - 2].ix[:, 4]
            df2['battles.lagged.3']   = ACLED[current_year - 3].ix[:, 4]
            df2['battles.index']      = \
                ACLED[current_year - 1].ix[:, 4] \
                + (ACLED[current_year - 2].ix[:, 4] * .5) \
                + (ACLED[current_year - 3].ix[:, 4] * .25)

        # Move target classes to end and remove current fatality data
        df3 = df2['battles']
        df2.pop('battles')
        df2['battles'] = df3

        # Commit merged dataframes 'out'
        ACLED[current_year] = df2[df2['YEAR'] == current_year]


    # Merge data row wise,
    # remove current fatlity data and create alt dummy target class
    print 'Merge data row-wise...'
    data_full = ACLED[START_YEAR]
    for year in range(START_YEAR + 1, END_YEAR + 1):
        data_full = data_full.append(ACLED[year])
    data_full.pop('SUMfatalities')
    data_full.pop('MEANfatalities')
    data_full['dummy.battles'] = map(bool, data_full['battles'])

    # Create colors for classes for later plot
    data_colors = np.array(map(lambda x: PLOT_COLORS[x], data_full['dummy.battles']))

    # Remove duplicated colums
    def remdup(df, column):
        return pd.concat([
            df[column][[0]],
            df[[c for c in df.columns if c != column]]],
            axis=1
        )

    # Final merge of data frames column wise
    print 'Final merge data column-wise...'
    dpi = dpi.sort_values(by=['ID', 'YEAR'])
    data_full = remdup(data_full, 'OBJECTID')
    data_full.rename(columns={'OBJECTID': 'ID'}, inplace=True)
    data_full = remdup(data_full, 'ISO')
    data_full = data_full.sort_values(by=['ID', 'YEAR'])
    df = dpi.ix[:, 6:68]
    df['ID'] = dpi['ID']
    df['YEAR'] = dpi['YEAR']
    data_full = pd.merge(data_full, df, on=['ID', 'YEAR'])

    # Final typecast
    print 'Final typecast...'
    data_full.fillna(0, inplace=True)
    for c in [
        'ISO', 'poli.sys', 'exec.allign', 'exec.rel', 'leg.elective.rules',
        'exec.elective.rules', 'muni.gov.elected', 'state.gov.elected'
        ]:
            # FIXME: Better representation
            data_full[c] = map(hash, data_full[c])
            data_full[c] = data_full[c].astype(np.float32)
    data_full.astype(np.float32)

    return data_full

def prepare_for_learning(data_full):
    """
    Prepare datasets for classification learning

    Method shuffles input data, normalizes its values and classes power.
    Returns tuple `(X_learn, y_learn, X_test, y_test)`.
    Learning dataset includes data from 1997 to 2011.
    Test dataset includes 2012 data.
    """
    print 'Shuffle data...'
    np.random.seed(2016)
    data_full = data_full.reindex(np.random.permutation(data_full.index))

    if PLOT_SCATTER_MATRIX:
        colored_scatter_matrix(
            data_full[PLOT_COLUMNS][:PLOT_DATA],
            data_colors[:PLOT_DATA],
            title='RAW data'
        )

    # Trim to relevant feature space and split 2012 off for final testing
    print 'Trim and split...'

    data_trimmed = data_full[LEARNING_COLUMNS]

    X_learn = np.array(
        data_trimmed[
            data_full['YEAR'] != 2012
        ])
    y_learn = np.array(
        data_full[
            data_full['YEAR'] != 2012
        ][TARGET_COLUMN])

    X_learn = X_learn.astype(np.float32)
    y_learn = y_learn.astype(np.int32)

    X_test = np.array(
        data_trimmed[
            data_full['YEAR'] == 2012
        ])
    del data_trimmed
    y_test = np.array(
        data_full[
            data_full['YEAR'] == 2012
        ][TARGET_COLUMN])
    print 'Delete data...'
    del data_full

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    if LEARN_ON_EQUAL_POWER:
        print 'Slice so that classes have equal power...'

        print '\nBefore slice:'
        print 'Negative examples (0):', y_learn[y_learn == 0].size
        print 'Positive examples (1):', y_learn[y_learn == 1].size

        X_learn_battles = X_learn[y_learn == 1]
        y_learn_battles = y_learn[y_learn == 1]

        # Include selected number of examples where there is no battles
        X_learn = np.concatenate((
            X_learn_battles,
            X_learn[y_learn == 0][
                :y_learn_battles.size * BATTLE_CLASS_POWER_MULTIPLIER]
        ))

        for _ in range(BATTLE_CLASS_POWER_MULTIPLIER - 1):
            # Duplicate battle class examples so that powers are equal
            X_learn = np.concatenate((X_learn, X_learn_battles))
        del X_learn_battles

        y_learn =  np.concatenate((
            y_learn_battles,
            y_learn[y_learn == 0][
                :y_learn_battles.size * BATTLE_CLASS_POWER_MULTIPLIER]
        ))

        for _ in range(BATTLE_CLASS_POWER_MULTIPLIER - 1):
            # Duplicate battle class examples so that powers are equal
            y_learn = np.concatenate((y_learn, y_learn_battles))
        del y_learn_battles

        print '\nAfter slice:'
        print 'Negative examples (0):', y_learn[y_learn == 0].size
        print 'Positive examples (1):', y_learn[y_learn == 1].size

    print ''
    print 'X_learn.shape', X_learn.shape
    print 'y_learn.shape', y_learn.shape

    print 'Normalize...'

    X_max = np.max([
        np.max(np.abs(X_learn), axis=0),
        np.max(np.abs(X_test), axis=0)
    ]) * 2
    y_max = np.max([
        np.max(np.abs(y_learn), axis=0),
        np.max(np.abs(y_test), axis=0)
    ]) * 2

    X_learn /= X_max
    X_test /= X_max

    print 'Data normalized so that:'
    print '* X_learn has:'
    print 'min:', X_learn.min(), 'max:', X_learn.max()
    print '* X_test has:'
    print 'min:', X_test.min(), 'max:', X_test.max()


    if PLOT_KDE:
        print 'Plot KDE for normalized learning data...'
        for c in range(X_learn.shape[1]):
            plot_kde(
               data=pd.DataFrame(X_learn[:, c], columns=[LEARNING_COLUMNS[c]]),
               title=str(LEARNING_COLUMNS[c]),
               save=PLOT_SAVE_DIR + 'kde/' + str(LEARNING_COLUMNS[c]) + '.png'
            )

    return X_learn, y_learn, X_test, y_test


def experiment(
    X_learn, y_learn, X_test, y_test,
    algo='rmsprop',
    learning_rate=0.0001,
    momentum=0,
    neurons=10**2,
    patience=100*1000,
    # Sparse hidden activations
    # have shown much promise in computational neural networks.
    hidden_l1 = 0.01,
    weight_l2 = 0.0001,
    ):

    import climate

    climate.enable_default_logging()

    TRAIN_PART = 0.7
    data_threshold = int(TRAIN_PART * len(X_learn))

    print 'Divide sets...'

    np.random.seed(2016)
    np.random.shuffle(X_learn)
    np.random.seed(2016)
    np.random.shuffle(y_learn)
    X_train, y_train = X_learn[:data_threshold], y_learn[:data_threshold]
    X_valid, y_valid = X_learn[data_threshold:], y_learn[data_threshold:]

    datasets = {
        'training':     (X_train, y_train),
        'validation':   (X_valid, y_valid),
    }

    layers=(X_learn.shape[1], (neurons, 'relu'), 2)

    import theanets

    exp = theanets.Experiment(
        theanets.Classifier,
        layers=layers,
    )

    train_acc_history = []
    valid_acc_history = []
    valid_0_acc_history = []
    valid_1_acc_history = []

    init_learning_plot()

    from sklearn.metrics import accuracy_score

    print 'Start learning...'

    iteration = 0

    for tm, vm in exp.itertrain(
            datasets['training'],
            datasets['validation'],
            algo=algo,
            learning_rate=learning_rate,
            momentum=momentum,
            hidden_l1=hidden_l1,
            weight_l2=weight_l2,
            patience=patience,
            ):
        iteration += 1

        # Validate every class separately
        y_pred_0 = exp.network.classify(X_valid[y_valid == 0])
        y_pred_1 = exp.network.classify(X_valid[y_valid == 1])
        valid_0_acc_history.append(
            accuracy_score(y_valid[y_valid == 0], y_pred_0) * 100)
        valid_1_acc_history.append(
            accuracy_score(y_valid[y_valid == 1], y_pred_1) * 100)

        train_acc_history.append(tm['acc'] * 100)
        valid_acc_history.append(vm['acc'] * 100)

        # FIXME: First validation ACC is 1.0
        update_learning_plot(
            train_acc_history[10:],
            valid_acc_history[10:],
            valid_0_acc_history[10:],
            valid_1_acc_history[10:])

        if iteration == 490:
            save_learning_plot()

    save_learning_plot('current_learning_end.png')

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = exp.network.classify(X_test)

    print 'classification_report:\n', classification_report(y_test, y_pred)
    print 'confusion_matrix:\n', confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    data_full = preprocess_data()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(data_full)
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
    }
    experiment(**kwargs)

