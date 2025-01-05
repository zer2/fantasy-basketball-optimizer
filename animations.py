from manim import *

import os
os.chdir('..') 

import pandas as pd
import numpy as np
from collections import Counter

def make_weekly_df(season_df : pd.DataFrame):
    """Prepares a stat dataframe and a position series for a season"""

    cols = ['FTM','FTA','FGM','FGA'
        ,'PTS','REB','AST','STL','BLK','FG3M','TO']

    season_df['DATE'] = pd.to_datetime(season_df['date'])
    season_df['WEEK'] = season_df['DATE'].dt.isocalendar()['week']

    #make sure we aren't missing any weeks when a player didn't play
    weekly_df_index = pd.MultiIndex.from_product([pd.unique(season_df['PLAYER_NAME'])
                                                 ,pd.unique(season_df['WEEK'])]
                                                 ,names = ['PLAYER_NAME','WEEK'])
    weekly_df = season_df.groupby(['PLAYER_NAME','WEEK'])[cols].sum()
    season_df = pd.DataFrame(weekly_df, index = weekly_df_index ).fillna(0)

    #experimental!!!
    #season_df = season_df[season_df.sum(axis = 1) > 0]
    
    return season_df

#from manim import config
#config.tex_compiler = "C:/Program Files/MiKTeX/miktex/bin/x64/latex"

weekly_df = make_weekly_df(pd.read_csv('C:/Users/zacha/Projects/FBBO/data_for_testing/2023-24_complete.csv'))

representative_players = list(weekly_df.groupby('PLAYER_NAME')['PTS'].mean().sort_values(ascending = False).index[0:156])

block_averages = weekly_df.groupby('PLAYER_NAME')['BLK'].mean().loc[representative_players].values
block_values = weekly_df.loc[representative_players]['BLK'].values

print(weekly_df.groupby('PLAYER_NAME')['BLK'].mean().loc[representative_players].sort_values())


def get_histogram(x,values):
    '''get values in a series bucketed for a histogram display

    Args:
        x: values for buckets 
        values: actual values in the dataset 
    '''
    rounded = [np.round(v*2)/2 for v in values]
    rounded_counter = Counter(rounded)

    y = [rounded_counter.get(z) if z in rounded_counter.keys() else 0 for z in x]

    return y

block_averages_rounded = np.array([np.round(x*2)/2 for x in block_averages])
block_values_rounded = np.array([np.round(x*2)/2 for x in block_values])

block_averages_rounded_mean = block_averages_rounded.mean()
block_values_rounded_mean = block_values_rounded.mean()
block_averages_rounded_std = block_averages_rounded.std()
block_values_rounded_std = block_values_rounded.std()

block_averages_mean = block_averages.mean()
block_values_mean = block_values.mean()
block_averages_std = block_averages.std()
block_values_std = block_values.std()

x_averages = [z/2 for z in range(-8,22)]
x_values = [z for z in range(0,20)]

y_averages_value_basis = get_histogram(x_values, block_averages_rounded)
y_averages_0 = get_histogram(x_averages, block_averages_rounded)
y_averages_1 = get_histogram(x_averages, block_averages_rounded - block_averages_rounded_mean)
y_averages_2 = get_histogram(x_averages, (block_averages_rounded - block_averages_rounded_mean)/block_averages_rounded_std)

y_values = get_histogram(x_values, block_values_rounded)

y_averages_value_basis = get_histogram(x_values, block_averages_rounded)
y_averages_0 = get_histogram(x_averages, block_averages_rounded)
y_averages_1 = get_histogram(x_averages, block_averages_rounded - block_averages_rounded_mean)
y_averages_2 = get_histogram(x_averages, (block_averages_rounded - block_averages_rounded_mean)/block_averages_rounded_std)

y_values = get_histogram(x_values, block_values_rounded)

class BlockAveragesVis(Scene):

    def construct(self):

        title = Text('Weekly block rates of top 156 players in 2023').shift(UP * 3)
        label_0 = Tex('Weekly averages $(m_p)$').shift(UP * 2)

        chart = BarChart(values= y_averages_0
                            ,bar_names = x_averages
                            ,y_range = [0,80,10]
                                 ).shift(DOWN + LEFT).scale(0.8)

        axis_titles = chart.get_axis_labels(
            Tex(""), Tex("\# of players")
        )

        label_1 = Tex('$m_p - m_\mu (= $' + str(np.round(block_averages_mean,2)) + ')').shift(UP * 2)

        label_2 = Tex(r"$\frac{m_p - m_\mu (= " +  str(np.round(block_averages_mean,2)) + \
                                            ")}{m_\sigma (= " + str(np.round(block_averages_std,2)) + r')}$').shift(UP * 2)

        self.play(Write(title))

        self.play(Write(label_0)
                ,Create(chart)
                ,Write(axis_titles))

        self.wait(2)

        self.play(ReplacementTransform(label_0,label_1)
                  ,chart.animate.change_bar_values(y_averages_1)
                    )

        self.wait(3)

        self.play(ReplacementTransform(label_1,label_2)
                  ,chart.animate.change_bar_values(y_averages_2)
                  )

        self.wait(3)