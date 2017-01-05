#!/usr/bin/env python2
# coding: utf-8

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
import sklearn.cluster
import sklearn.preprocessing
import scipy as sp
import time

import matplotlib

from pandas import read_csv
from copy import deepcopy
from matplotlib import pyplot as plt


matplotlib.style.use('ggplot')


##
# Constants
##

COLUMNS_SORTED_ACC = [
    # Number of battles in previous three years
    'battles.index',
    # Land conflict index (risk?)
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
    # Immature mortality rate percent mean
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

# These are slices,
# so two slices [a, b, c] mean 3 buckets (-inf, a), [a, b), [b, inf)
REPRESENTATION_SLICES = {
    'ID': [50000, 150000],
    'land.conf.norm': [0.5],
    'flood.freq.mean': [2.5, 5],
    'drought.freq.mean': [5, 7.5],
    'lootable.diamonds': [0.5],
    'uw.perc.mean': [0.1, 0.4],
    'imr.perc.mean': [0.1],
    'gdp.mean': [50],
    'gdp.mean.lag': [50],
    'gdp.mean.lag.2': [50],
    'gdp.mean.change': [0.1],
    'gdp.mean.change.lag': [0.1],
    'gdp.mean.change.lag.2': [0.1],
    'gdp.mean.sum': [50],
    'gdp.mean.sum.lag': [50],
    'gdp.mean.sum.lag.2': [50],
    'pop.mean': [5 * 10**4],
    'pop.mean.lag': [5 * 10**4],
    'pop.mean.lag.2': [5 * 10**4],
    'pop.sum': [0.1 * 10**7],
    'pop.sum.lag': [0.1 * 10**7],
    'pop.sum.lag.2': [0.1 * 10**7],
    'battles.index': [10**5, 2.5 * 10**5],
    'military': [0.5],
    'gov.share': [0.05],
    'gov.share.1': [0.5],
    'gov.share.2': [0.1, 0.3],
    'gov.share.3': [0.05, 0.15],
    'oppo.share': [0.05, 0.15],
    'oppo.share.1': [0.1, 0.4],
    'oppo.share.2': [0.05, 0.2],
    'oppo.share.3': [0.05, 0.2],
}

NAMED_FEATURES = [
    'ISO', 'poli.sys', 'exec.allign', 'exec.rel', 'leg.elective.rules',
    'exec.elective.rules', 'muni.gov.elected', 'state.gov.elected'
]

# >>> sorted(list(set(df['ISO'])))
COUNTRY_ISO = ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR', 'COD',
'COG', 'DJI', 'DZA', 'EGY', 'ERI', 'ESH', 'ESP', 'ETH', 'GAB', 'GHA', 'GIN',
'GMB', 'GNB', 'GNQ', 'ISR', 'JOR', 'KEN', 'LBR', 'LBY', 'LSO', 'MAR', 'MLI',
'MOZ', 'MRT', 'MWI', 'NAM', 'NER', 'NGA', 'PSE', 'RWA', 'SDN', 'SEN', 'SLE',
'SOM', 'SSD', 'SWZ', 'TCD', 'TGO', 'TUN', 'TZA', 'UGA', 'ZAF', 'ZMB', 'ZWE']

COLUMNS_NEW_REPRESENTATION = REPRESENTATION_SLICES.keys() + ['YEAR']

COLUMNS_WITH_GEO = COLUMNS_NEW_REPRESENTATION + ['lat_dec', 'lon_dec']

###
# Input data selection and preprocessing
###

#LEARNING_COLUMNS = COLUMNS_SORTED_ACC
LEARNING_COLUMNS = COLUMNS_FROM_PAPER
#LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION
#LEARNING_COLUMNS = COLUMNS_WITH_GEO

# Use REPRESENTATION_SLICES rules to create a new KDE-based data representation
CHANGE_DATA_REPRESENTATION = False

# NOTE: Below are active with CHANGE_DATA_REPRESENTATION
# Remove base feature from which a new representation is derived
REMOVE_BASE_FEATURE = False # True
# Generate automatic slices
AUTOMATIC_SLICES = False # True

###
# Input data parameters
###

LEARN_ON_EQUAL_POWER = True
BATTLE_CLASS_POWER_MULTIPLIER = 1 # 5

###
# Input data constants
###

ALL_COL_1997 = 9

START_YEAR = 1997
END_YEAR = 2012

TARGET_COLUMN = 'dummy.battles'  # 'dummy.battles'

###
# Plot
###

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


def update_learning_plot(iters, plot_history):
    #train_acc, valid_acc, valid_0_acc, valid_1_acc, test_acc):
    """ Update plot that displays learning results """

    plt.clf()
    legend_handles = []

    # Colors are for metrics
    colors = {
        'accuracy_score' : 'green',
        'matthews_corrcoef' : 'blue',
        'roc_auc_score' : 'red',
        'f1_score' : 'gold',
        'mse' : 'green',
        'r2_score' : 'red',
        '0 validation' : 'blue',
        '1 validation' : 'red',
        '0 test' : 'blue',
        '1 test' : 'red',
    }

    # Line styles are for datasets
    styles = {
        'training' : ':',
        'validation' : '--',
        '0 validation' : '--',
        '1 validation' : '--',
        '0 test' : '-',
        '1 test' : '-',
        'test' : '-',
    }

    # Name shortcut
    names = {
        'accuracy_score' : 'ACC',
        'matthews_corrcoef' : 'MCC',
        'roc_auc_score' : 'AUC',
        'f1_score' : 'F1',
        'mse': 'MSE',
        'r2_score': 'R2 Score',
    }

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for metric in plot_history:
        for plot_set in plot_history[metric]:

            color = colors.get(plot_set, colors.get(metric, None))
            style = styles.get(plot_set, None)
            history = plot_history[metric][plot_set]
            name = names.get(metric, metric)

            if not history:
                continue

            print 'hist', history

            if len(history) < 2:
                continue

            label = None

            if not (metric[:7] == 'hid1_w_' \
                and history[-1] > -5 and history[-1] < 5):
                # Ommit low weight values for picture readability

                    label = '{} {}: {:.2f}%'.format(
                        plot_set[0].upper() + plot_set[1:],
                        name,
                        history[-1]
                    )

                    legend_line, = plt.plot(
                        iters,
                        history,
                        color=color,
                        linestyle=style,
                        label=label,
                    )

                    legend_handles.append(legend_line)

                    ann_xy = (iters[-1], history[-1])
                    ax.annotate(
                        '{}: {:.2f}%'.format(name, history[-1]),
                        xy=ann_xy, xytext=ann_xy, size='xx-small')

    plt.legend(
        loc='lower left',
        handles=legend_handles,
        framealpha=0.5
    )
    plt.show()
    plt.pause(0.001)


def save_learning_plot(filename='current_learning.png'):
    figure = plt.gcf()
    figure.set_size_inches(17.07, 8.22)
    plt.savefig(PLOT_SAVE_DIR + filename)

##
# Load raw data
##

def raw_data():
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
    data_full.drop_duplicates(inplace=True)

    # Final typecast
    print 'Final typecast...'
    data_full.fillna(0, inplace=True)

    for c in NAMED_FEATURES:
            # Eliminate duplicate zeros...
            data_full[c].replace(to_replace='0', value=0, inplace=True)

    return data_full


def data_expanded():
    '''
    Expand base data with new features
    '''
    df = raw_data()

    if 'lat_dec' in LEARNING_COLUMNS or 'lon_dec' in LEARNING_COLUMNS:
        from data import scrap_wikipedia_cities as wiki_countries

        alpha3_name = wiki_countries.alpha3_name
        countries = wiki_countries.countries_df

        df['country'] = [alpha3_name[iso] for iso in df['ISO']]

        if 'lat_dec' in LEARNING_COLUMNS:
            df['lat_dec'] = [
                countries[country]['latitude_dec'] for country in df['country']]
        if 'lon_dec' in LEARNING_COLUMNS:
            df['lon_dec'] = [
                countries[country]['longitude_dec'] for country in df['country']]

        df.pop('country')

    # FIXME: Remove NaNs?
    df.fillna(0, inplace=True)

    return df


def data_numeric():
    data_full = data_expanded()

    for c in NAMED_FEATURES:
        c_values = list(set(data_full[c]))
        data_full[c] = [c_values.index(x) for x in data_full[c]]

        if AUTOMATIC_SLICES:
            global REPRESENTATION_SLICES
            REPRESENTATION_SLICES[c] = range(len(set(data_full[c])))

    return data_full


##
# Preprocessing
##


def preprocess_data():
    data_full = data_numeric()
    data_full.astype(np.float32)
    c = data_full.columns
    assert(len(list(c)) == len(set(c)))

    return data_full


def data_encoded():
    """
    Prepare datasets for classification learning

    Most importantly, additional data columns are introduced in order to
    provide a new data representation for better learning results.
    Method shuffles input data, normalizes its values and classes power.
    """
    data_full = preprocess_data()

    print 'Shuffle data...'
    np.random.seed(2016)
    data_full = data_full.reindex(np.random.permutation(data_full.index))

    if PLOT_SCATTER_MATRIX:
        colored_scatter_matrix(
            data_full[PLOT_COLUMNS][:PLOT_DATA],
            data_colors[:PLOT_DATA],
            title='RAW data'
        )

    data_prepared = data_full[LEARNING_COLUMNS]

    if CHANGE_DATA_REPRESENTATION:
        print 'Change data representation...'

        # Generate binary representations
        for feature in REPRESENTATION_SLICES:
            # Process only selected columns
            if not feature in LEARNING_COLUMNS:
                continue

            slices = REPRESENTATION_SLICES[feature]
            for i, value in enumerate(slices):
                if len(slices) > 1 and i > 0:
                    data_prepared[feature + '.f' + str(i)] = \
                        (data_prepared[feature] >= slices[i-1]) \
                      & (data_prepared[feature] < value)
                else:
                    data_prepared[feature + '.f' + str(i)] = \
                        data_prepared[feature] < value
            data_prepared[feature + '.f' + str(len(slices))] = \
                data_prepared[feature] >= slices[len(slices)-1]

            if REMOVE_BASE_FEATURE:
                data_prepared.pop(feature)

            c = data_prepared.columns
            assert(len(list(c)) == len(set(c)))

    if PLOT_KDE:
        print 'Plot KDE for data...'
        for c in data_prepared.columns:
            plot_kde(
               data=data_prepared[c][:4000],
               title=c,
               save=PLOT_SAVE_DIR + 'kde/' + c + '.png'
            )
        exit(0)

    c = data_prepared.columns
    assert(len(list(c)) == len(set(c)))

    return data_full, data_prepared


def prepare_for_learning(data_full, data_prepared):
    '''
    data_full - raw data DataFrame
    data_prepared - data_encoded() DataFrame
    Returns tuple `(X_learn, y_learn, X_test, y_test)`.
    Learning dataset includes data from 1997 to 2011.
    Test dataset includes 2012 data.
    '''
    # Trim to relevant feature space and split 2012 off for final testing
    print 'Trim and split...'

    X_learn = np.array(
        data_prepared[
            data_full['YEAR'] != 2012
        ])
    y_learn = np.array(
        data_full[
            data_full['YEAR'] != 2012
        ][TARGET_COLUMN])

    X_learn = X_learn.astype(np.float32)
    y_learn = y_learn.astype(np.int32)

    X_test = np.array(
        data_prepared[
            data_full['YEAR'] == 2012
        ])
    del data_prepared
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

    # Normalize to interval
    norm_a, norm_b = 0.1, 0.9

    X_min = np.min([
        np.min(X_learn, axis=0),
        np.min(X_test, axis=0)
    ])
    X_max = np.max([
        np.max(X_learn, axis=0),
        np.max(X_test, axis=0)
    ])

    X_learn = (((norm_b - norm_a) * (X_learn - X_min)) / (X_max - X_min)) + norm_a
    X_test = (((norm_b - norm_a) * (X_test - X_min)) / (X_max - X_min)) + norm_a

    print 'Data normalized so that:'
    print '* X_learn has:'
    print 'min:', X_learn.min(), 'max:', X_learn.max()
    print '* X_test has:'
    print 'min:', X_test.min(), 'max:', X_test.max()

    return X_learn, y_learn, X_test, y_test


def experiment(
    X_learn, y_learn, X_test, y_test,
    neurons,
    theanets_kwargs,
    column_names=None,
    name='my_exp',
    max_iters=float('inf'),
    gather_metrics=[
        'accuracy_score',
        'matthews_corrcoef',
        'roc_auc_score',
        'f1_score',
        ],
    plot_every=50,
    regression=False,
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
        '0 validation': (X_valid[y_valid == 0], y_valid[y_valid == 0]),
        '1 validation': (X_valid[y_valid == 1], y_valid[y_valid == 1]),
        '0 test':       (X_test[y_test == 0], y_test[y_test == 0]),
        '1 test':       (X_test[y_test == 1], y_test[y_test == 1]),
        'test':         (X_test, y_test)
    }

    for key in datasets.keys():
        datasets[key] = (datasets[key][0], datasets[key][1])

    layers=(X_learn.shape[1],) + neurons + (2,)

    if regression:
        layers=(X_learn.shape[1],) + neurons + (1,)

    import theanets

    exp = theanets.Experiment(
        theanets.Classifier,
        layers=layers,
    )

    if regression:
        exp = theanets.Experiment(
            theanets.Regressor,
            layers=layers
        )

    network = exp.network

    print 'Start learning...'

    iteration = 0

    # Metrics that we want to measure while observing results
    metrics = []
    # ACC
    if 'accuracy_score' in gather_metrics:
        metrics.append({
            'name': 'accuracy_score',
            'function': sklearn.metrics.accuracy_score
        })
    # MCC
    if 'matthews_corrcoef' in gather_metrics:
        metrics.append({
            'name': 'matthews_corrcoef',
            'function': sklearn.metrics.matthews_corrcoef
        })
    # AUC
    if 'roc_auc_score' in gather_metrics:
        metrics.append({
            'name': 'roc_auc_score',
            'function': sklearn.metrics.roc_auc_score
        })
    # F1
    if 'f1_score' in gather_metrics:
        metrics.append({
            'name': 'f1_score',
            'function': sklearn.metrics.f1_score
        })

    # Regression

    # MSE
    if 'mse' in gather_metrics:
        metrics.append({
            'name': 'mse',
            'function': sklearn.metrics.mean_squared_error
        })

    if 'r2_score' in gather_metrics:
        metrics.append({
            'name': 'r2_score',
            'function': sklearn.metrics.r2_score
        })

    if column_names is not None:
        for w_i in xrange(X_learn.shape[1]):
            metrics.append({
                'name': 'hid1_w_' + column_names[w_i],
                'inspect': ('hid1', 'w', w_i)
            })

    # Sets that we want to test
    plot_sets = [
        'training',
        'validation',
        '0 test',
        '1 test',
        'test'
    ]

    init_learning_plot()

    # Init histories
    plot_history = dict()
    for metric in metrics:
        plot_history[metric['name']] = dict()
        for plot_set in plot_sets:
            if metric['name'] == 'roc_auc_score' \
                    and plot_set[0] == '0':
                continue
            if metric['name'] == 'roc_auc_score' \
                    and plot_set[0] == '1':
                continue
            if metric['name'] == 'f1_score' \
                    and plot_set[0] == '0':
                continue
            if metric['name'] == 'f1_score' \
                    and plot_set[0] == '1':
                continue
            if metric['name'] == 'matthews_corrcoef' \
                    and plot_set[0] == '0':
                continue
            if metric['name'] == 'matthews_corrcoef' \
                    and plot_set[0] == '1':
                continue
            plot_history[metric['name']][plot_set] = []
    iters = []

    # Automatic metrics from theanets
    #plot_history['error'] = dict()
    #plot_history['error']['validation'] = []
    #plot_history['loss'] = dict()
    #plot_history['loss']['validation'] = []

    # XXX
    #plot_history['accuracy_score']['training'] = []

    print 'Collecting following metrics:'
    print gather_metrics

    for tm, vm in exp.itertrain(
            datasets['training'],
            datasets['validation'],
            **theanets_kwargs
            ):
        iteration += 1

        if iteration > max_iters:
            break

        if iteration % plot_every:
            continue

        for metric in metrics:

            if not metric['name'] in gather_metrics:
                continue

            for plot_set in plot_sets:

                X_set, y_set = datasets[plot_set]

                if metric['name'] == 'roc_auc_score' \
                        and plot_set[0] == '0':
                    continue
                if metric['name'] == 'roc_auc_score' \
                        and plot_set[0] == '1':
                    continue
                if metric['name'] == 'f1_score' \
                        and plot_set[0] == '0':
                    continue
                if metric['name'] == 'f1_score' \
                        and plot_set[0] == '1':
                    continue
                if metric['name'] == 'matthews_corrcoef' \
                        and plot_set[0] == '0':
                    continue
                if metric['name'] == 'matthews_corrcoef' \
                        and plot_set[0] == '1':
                    continue

                if metric['name'] == 'accuracy_score' \
                        and plot_set == 'validation':

                    # Get it directly from theanets
                    plot_history[metric['name']][plot_set].append(
                        vm['acc'] * 100)

                    continue

                if metric['name'] == 'accuracy_score' \
                        and plot_set == 'training':

                    # Get it directly from theanets
                    plot_history[metric['name']]['training'].append(
                        tm['acc'] * 100)

                    continue

                if not regression:
                    y_pred = exp.network.classify(X_set)
                else:
                    y_pred = exp.network.predict(X_set)

                if 'function' in metric:
                    scaling = 100.
                    mini = 0.
                    if metric['name'] == 'mse' \
                            or metric['name'] == 'r2_score':
                        scaling = 1.
                        mini = float('-inf')
                    plot_history[metric['name']][plot_set].append(
                        max(
                            mini,
                            metric['function'](y_set, y_pred) * scaling)
                    )

            if 'inspect' in metric:
                layer, param_name, param_x = metric['inspect']
                param = network.find(layer, param_name)
                values = param.get_value()
                mean_value = np.mean(values[param_x]) * 100.
                plot_history[metric['name']][plot_set].append(mean_value)

        iters.append(iteration)

        print 'iteration', iteration, {
            m + '_' + s: plot_history[m][s][-1:] for m in plot_history
            for s in plot_history[m]
        }
        update_learning_plot(iters, plot_history)

        if iteration in [500, 1000, 2000, 5000]:
            save_learning_plot(
                'current_learning_' + str(iteration)
                + '_' + name
                + '.png'
            )

    save_learning_plot('current_learning_end.png')

    from sklearn.metrics import classification_report, confusion_matrix

    if not regression:
        y_pred = exp.network.classify(X_test)
    else:
        y_pred = exp.network.predict(X_test)


    if not regression:
        print 'classification_report:\n', \
            classification_report(y_test, y_pred)
        print 'confusion_matrix:\n', \
            confusion_matrix(y_test, y_pred)

    for metric in metrics:
        plot_history[metric['name']]['test_max'] = max(
            plot_history[metric['name']]['test'])

    return plot_history


def exp_mlcp1_basic_t1_100():
    """MLCP1-Basic with T1-100

    Basic dataset from MLCP with one layer 100 hidden neurons architecture.

    There was no representation change introduced in this dataset.
    The normalization operation was the only operation performed.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_FROM_PAPER

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((100, 'relu'),),
        'max_iters': 50000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp2_limited_t1_100():
    """MLCP2-Limited with T1-100

    MLCP dataset limited to 20 features with higher importance,
    based on variable importance measurement of random forest metric used in MLCP.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_SORTED_ACC

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T1-100
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((100, 'relu'),),
        'max_iters': 50000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp3_extended_t1_100():
    """MLCP3-Extended with T1-100

    Extended representation of MLCP2-Limited, where a KDE
    based one-hot encodings were used instead of base dimensions. This way, base
    features were replaced with two or three bit encodings, resulting in 45 features.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = True

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T1-100
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((100, 'relu'),),
        'max_iters': 50000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp4_combined_t1_100():
    """MLCP4-Combined with T1-100

    MLCP1-Basic and MLCP3-Extended combined together,
    resulting in 87 features.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T1-100
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((100, 'relu'),),
        'max_iters': 50000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_regularizers_grid_search():
    """Regularizers grid search"""
    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    regularizers = [
        {'hidden_l1': 0.1}, {'weight_l1': 0.0001}, {'weight_l2': 0.0001},
        {'input_noise': 0.1}, {'hidden_noise': 0.1},
        {'input_dropout': 0.3}, {'hidden_dropout': 0.3}]

    test_no = 0

    metric_history = list()

    for reg in regularizers:
        for algo in ['rmsprop', 'rprop', 'sgd', 'adam']:

            test_no += 1

            theanets_kwargs = dict(
                algo='rmsprop',
                learning_rate=0.0001,
                momentum=0,
                patience=100*1000,
            )

            theanets_kwargs.update(reg)

            kwargs = {
                'X_learn': X_learn,
                'y_learn': y_learn,
                'X_test': X_test,
                'y_test': y_test,
                'neurons': ((10, 'relu'),),
                'name': str(test_no).zfill(2) + '_' + algo + '_' + reg.keys()[0],
                'max_iters': 1000,
                'theanets_kwargs': theanets_kwargs
            }

            metric_history.append(((reg, algo), experiment(**kwargs)))

    print metric_history


def exp_mlcp3_extended_t2_10():
    """MLCP3-Extended with T2-10

    T2-10 Selected as a result of grid search on different regularization
    methods and optimizer algorithms for hidden layer of 10 neurons. 
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = True

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T2-10
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((10, 'relu'),),
        'name': '03_5x5_l1l2_',
        'max_iters': 50000,
        'gather_metrics': [
            'accuracy_score',
            #'matthews_corrcoef',
            #'roc_auc_score',
            #'f1_score',
            ],
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_dropout=0.3,
            #hidden_l1 = 0.01,
            #weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp4_combined_t3_10():
    """MLCP4-Combined with T3-10"""
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T3-10
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((10, 'relu'),),
        'name': '03_5x5_l1l2_',
        'max_iters': 50000,
        'gather_metrics': [
            'accuracy_score',
            #'matthews_corrcoef',
            #'roc_auc_score',
            #'f1_score',
            ],
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            #hidden_dropout=0.3,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp5_countries_t4_10_10():
    """MLCP5-Countries with T4-10-10
    
    MLCP4-Combined expanded with one-hot encoding features
    for every separate country, resulting in 100 features.

    T4-10-10 - Deeper architecture with two layers
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T4-10-10
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((10, 'relu'), (10, 'relu'),),
        'name': label,
        'max_iters': 1000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_dropout=0.3,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp6_autoslices_t4_10_10():
    """MLCP6-Autoslices with T4-10-10

    MLCP6-Autoslices is MLCP5-Countries expanded with data of named (or text)
    values. Features were replaced with one-hot encoding representation
    of every possible value.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    global AUTOMATIC_SLICES
    AUTOMATIC_SLICES = True

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T4-10-10
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((10, 'relu'),(10, 'relu')),
        'name': 'slices',
        'max_iters': 1100,
        'gather_metrics': [
            'accuracy_score',
            #'matthews_corrcoef',
            #'roc_auc_score',
            #'f1_score',
            ],
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_dropout=0.3,
            #input_noise=0.1,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp6_autoslices_t5_100():
    """MLCP6-Autoslices with T5-100"""
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    global AUTOMATIC_SLICES
    AUTOMATIC_SLICES = True

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    # T5-100
    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'neurons': ((100, 'relu')),
        'name': 'mlcp6_autoslices_t5_100',
        'max_iters': 5000,
        'gather_metrics': [
            'accuracy_score',
            #'matthews_corrcoef',
            #'roc_auc_score',
            #'f1_score',
            ],
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_dropout=0.3,
            #input_noise=0.1,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        )
    }
    experiment(**kwargs)


def exp_mlcp7_geo_t5_100():
    """MLCP7-Geo with T5-100
    
    MLCP7-Geo is MLCP6-Autoslices extended with latitude and longitude data of
    corresponding countries.

    This experiment compares results on data with and without geo columns.
    """
    global LEARNING_COLUMNS
    LEARNING_COLUMNS = COLUMNS_NEW_REPRESENTATION

    global CHANGE_DATA_REPRESENTATION
    CHANGE_DATA_REPRESENTATION = True

    global REMOVE_BASE_FEATURE
    REMOVE_BASE_FEATURE = False

    global AUTOMATIC_SLICES
    AUTOMATIC_SLICES = True

    learning_columns = deepcopy(LEARNING_COLUMNS)

    for add_geo, label in [
        (True, 'with_geo'),
        (False, 'without_geo'),
    ]:
        if add_geo:
            LEARNING_COLUMNS = learning_columns + ['lat_dec', 'lon_dec']
        else:
            LEARNING_COLUMNS = learning_columns

        data_full, data_prepared = data_encoded()
        X_learn, y_learn, X_test, y_test = prepare_for_learning(
            data_full, data_prepared)

        # T5-100
        kwargs = {
            'X_learn': X_learn,
            'y_learn': y_learn,
            'X_test': X_test,
            'y_test': y_test,
            'neurons': ((100, 'relu'),),
            'name': label,
            'max_iters': 5000,
            'gather_metrics': [
                'accuracy_score',
                #'matthews_corrcoef',
                #'roc_auc_score',
                #'f1_score',
                ],
            'theanets_kwargs': dict(
                algo='rmsprop',
                learning_rate=0.0001,
                momentum=0,
                patience=100*1000,
                hidden_dropout=0.3,
                #input_noise=0.1,
                hidden_l1 = 0.01,
                weight_l2 = 0.0001,
            )
        }
        experiment(**kwargs)


def exp_feature_weights():
    """Feature weights analysis"""

    data_full, data_prepared = data_encoded()
    X_learn, y_learn, X_test, y_test = prepare_for_learning(
        data_full, data_prepared)

    kwargs = {
        'X_learn': X_learn,
        'y_learn': y_learn,
        'X_test': X_test,
        'y_test': y_test,
        'column_names': data_prepared.columns,
        'neurons': ((10, 'relu'),),
        'name': 'feature_weights',
        #'max_iters': 5000,
        'theanets_kwargs': dict(
            algo='rmsprop',
            learning_rate=0.0001,
            momentum=0,
            patience=100*1000,
            hidden_dropout=0.3,
            #input_noise=0.1,
            hidden_l1 = 0.01,
            weight_l2 = 0.0001,
        ),
        'gather_metrics': [
            'accuracy_score',
            'matthews_corrcoef',
        ] + [
            'hid1_w_' + data_prepared.columns[i]
            for i in xrange(len(data_prepared.columns))
        ],
        'plot_every': 200,
    }
    experiment(**kwargs)


if __name__ == '__main__':
    exp_mlcp6_autoslices_t4_10_10()
