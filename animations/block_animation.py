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

def get_histogram(x,values, interval_size = 0.5, round_normal = True):
    '''get values in a series bucketed for a histogram display

    Args:
        x: values for buckets 
        values: actual values in the dataset 
    '''

    #use the random uniform to round up or down randomly 
    rounded = [(np.round(v/interval_size) if round_normal else int(v/interval_size)) * interval_size for v in values]
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

x_averages = [z/2 for z in range(-5,21)]
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

class BlockZScoreVis(Scene):

    def construct(self):

        title = Text('Weekly block rates of top 156 players in 2023-24').shift(UP * 3)
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

        self.wait(10)


class BlockZScoreSimulation(Scene):

    def construct(self):
        # Simulate the Block differential using player averages
        # Randomly select 13 players for Team A
        n_iterations = 10000
        teams = np.random.choice(representative_players, size=(n_iterations, 25), replace=True)

        # Split into Team A (13 players) and Team B (12 players + 1 remaining)
        team_a = teams[:, :13]
        team_b = teams[:, 13:]

        # Get block levels for each player
        block_levels = weekly_df.groupby('PLAYER_NAME')['BLK'].mean().loc[representative_players]

        results = []

        chart = BarChart(values= [0 for z in range(-30,30)]
                    ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                    ,y_range = [0,10,1]
                    ,y_length = 10
                            ).shift(RIGHT * 3.5).scale(0.5)

        axis_titles = chart.get_axis_labels(
            Tex(""), Tex("\# of occurences")
        )

        manual_axis_title = Text('Difference in totals').scale(0.6).next_to(chart, DOWN)

        self.add(chart, axis_titles, manual_axis_title)

        for i in range(n_iterations):

            display_teams = (i <10) or (i % 200 == 199) 
            wait_time = 1 if i<10 else 0.01

            team_a_players = [player for player in team_a[i, :]]
            team_a_values = [block_levels[player] for player in team_a[i, :]]

            team_b_players = [player for player in team_b[i, :]]
            team_b_values = [block_levels[player] for player in team_b[i, :]]

            results = results + [sum(team_a_values) - sum(team_b_values)]

            if display_teams:
            # Animate table and labels

                # Generate uniform tables
                team_a_table = self.create_uniform_table(
                    [(player, str(np.round(block_value, 2))) for player, block_value in zip(team_a_players, team_a_values)]
                )

                team_b_table = self.create_uniform_table(
                    [(player, str(np.round(block_value, 2))) for player, block_value in zip(team_b_players, team_b_values)]
                    + [('Remaining Player', '')]  # Add extra row to balance height
                )

                # Positioning tables
                team_a_table.to_edge(LEFT, buff=1)
                team_b_table.to_edge(LEFT, buff=4)

                # Team labels
                team_a_label = Text("Team A", color=BLUE).scale(0.6).next_to(team_a_table, UP)
                team_b_label = Text("Team B", color=RED).scale(0.6).next_to(team_b_table, UP)

                self.add(team_a_label, team_a_table)
                self.add(team_b_label, team_b_table)
                # Clear previous tables before the next iteration
                hist = get_histogram([z for z in range(-30,30)], results, 1)

                iteration_counter = Text('Iteration number: ' + str(i + 1)).scale(0.6).next_to(team_b_table, DOWN)
                self.add(iteration_counter)

                self.play(chart.animate.change_bar_values(hist))

                self.wait(wait_time)

                self.clear_mobjects_except([team_a_label, team_b_label, chart, axis_titles, manual_axis_title])

                if i == 9:
                    self.remove(chart)
                    chart = BarChart(values= hist
                            ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                            ,y_range = [0,800,100]
                            ,y_length = 10
                            ,x_axis_config = {'label_direction' : DOWN}
                                    ).shift(RIGHT * 3.5).scale(0.5)

                    self.add(chart)

        results_df = pd.DataFrame({'Result' : results})
        results_df.to_csv('blocks_z.csv', index = False)   
        self.wait(5) 

    def create_uniform_table(self, data):
        """Creates a table with uniform cell sizes."""
        cell_widths = [1.9,0.5]
        cell_height = 0.4

        table_rows = VGroup()
        for row_data in data:
            row_group = VGroup()
            for cell_text, cell_width in zip(row_data, cell_widths):
                rect = Rectangle(width=cell_width, height=cell_height)
                text = Text(str(cell_text)).scale(0.2)
                text.move_to(rect.get_center())
                row_group.add(VGroup(rect, text))
            row_group.arrange(RIGHT, buff=0)  # Align columns in a row
            table_rows.add(row_group)

        table_rows.arrange(DOWN, buff=0)  # Align rows vertically
        return table_rows

    def clear_mobjects_except(self, keep_objects):
        """Remove all objects from the scene except the ones in keep_objects."""
        for mob in self.mobjects:
            if mob not in keep_objects:
                self.remove(mob)

class BlockZScoreSimulationAnalysis(Scene):

    def construct(self):
        results = list([x[0] for x in pd.read_csv('blocks_z.csv').astype(float).values])

        hist = get_histogram([z for z in range(-30,30)], results, 1)

        chart = BarChart(values= hist
                ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                ,y_range = [0,800,100]
                ,y_length = 5
                        ).shift(UP * 0.5)
        self.add(chart)

        #placeholder text: We'll switch it out later
        text = Text("").next_to(chart, DOWN)

        self.wait(14)

        text, chart, line = self.set_middle_point(text, chart, None, results, hist, 0)
        self.wait(8)

        text, chart, line = self.set_middle_point(text, chart, line, results, hist, 1.77, r"$\approx \frac{1}{2} + \frac{x - \mu}{\sigma}$")
        self.wait(20)

        text, chart, line = self.set_middle_point(text, chart, line, results, hist, 1.77,  r"$\approx \frac{1}{2} + \mathord{?} * (x - \mu)$")
        self.wait(50) 

        text, chart, line = self.set_middle_point(text, chart, line, results, hist, 1.77,  r"$\approx \mathord{?} * (x - \mu)$")
        self.wait(50) 

        text, chart, line = self.set_middle_point(text, chart, line, results, hist, 1.77,  r"$\approx \frac{x - \mu}{\sigma}$")


    def set_middle_point(self, text, chart, line, values, hist, middle_point, special_string = None):
        fraction_winning = sum([x < middle_point  + (x == middle_point)/2 for x in values])/len(values)

        bar_colors = ['GREEN' if z < middle_point else ('GREEN_A' if z == middle_point else 'WHITE') for z in range(-30,30,2)]

        new_chart = BarChart(values= hist
                ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                ,y_range = [0,800,100]
                ,y_length = 5
                , bar_colors = bar_colors
                        ).shift(UP * 0.5)
        
        new_line = Line(start=DOWN * 2.2, end=UP * 2.2, color=YELLOW, stroke_width=8).shift((0.2 + middle_point * 0.16) * RIGHT)

        middle_point_str = str(middle_point) if not special_string else (r"$\mu = $ " + str(middle_point)) 
        new_text = Tex("Blocks scored by remaining player: " + middle_point_str + r"\\" \
                        + 'Probability of victory: ' + str(np.round(fraction_winning * 100,2)) + '\%') \
                            .next_to(new_chart, DOWN).scale(0.55)
        self.play(ReplacementTransform(line, new_line) if line is not None else Create(new_line)
                  ,ReplacementTransform(chart, new_chart)
                  ,ReplacementTransform(text, new_text))
        
        if special_string:
            self.wait(15)

            super_new_text = Tex("Blocks scored by remaining player: " + str(middle_point) + r"\\" \
                + 'Probability of victory: ' + str(np.round(fraction_winning * 100,2)) + "\% " + special_string) \
                    .next_to(new_chart, DOWN).scale(0.55)   
        
            self.play(ReplacementTransform(chart, new_chart)
                  ,ReplacementTransform(new_text, super_new_text))
            return super_new_text, new_chart, new_line
        else:
            return new_text, new_chart, new_line


class SuccessRateComparison(Scene):
    def construct(self):
        # Define colors
        CONTAINER_COLOR = WHITE
        DASHED_LINE_COLOR = YELLOW
        TEXT_COLOR = WHITE
        BASELINE_COLOR = RED
        TRANSLUCENT_COLOR = WHITE

        # Define positions
        left_container_pos = LEFT * 3
        right_container_pos = RIGHT * 3

        # Container height
        container_height = 4

        # Create left and right containers
        left_container = Rectangle(height=container_height, width=2, color=CONTAINER_COLOR)
        right_container = Rectangle(height=container_height, width=2, color=CONTAINER_COLOR)

        left_container.move_to(left_container_pos)
        right_container.move_to(right_container_pos)

        # Draw a translucent line connecting the tops of the two bars
        top_line = Line(
            start=left_container.get_corner(UR),
            end=right_container.get_corner(UL),
            color=TRANSLUCENT_COLOR,
            stroke_opacity=0.4  # Translucency effect
        )

        # Label for the height of the bars
        attempts_label = Text("Number of Attempts", color=WHITE).scale(0.4)
        attempts_label.next_to(top_line, UP, buff=0.2)

        # Dashed lines for "Number of Successes" (~77% height)
        success_y = container_height * 0.77
        success_line_left = DashedLine(
            start=left_container.get_corner(UL) + DOWN * (container_height - success_y),
            end=left_container.get_corner(UR) + DOWN * (container_height - success_y),
            color=DASHED_LINE_COLOR
        )

        success_line_right = DashedLine(
            start=right_container.get_corner(UL) + DOWN * (container_height - success_y),
            end=right_container.get_corner(UR) + DOWN * (container_height - success_y),
            color=DASHED_LINE_COLOR
        )

        mid_line = Line(
            start=success_line_left.get_corner(UR),
            end=success_line_right.get_corner(UL),
            color=TRANSLUCENT_COLOR,
            stroke_opacity=0.4  # Translucency effect
        )

        success_label = Text("Baseline # of Successes", color=WHITE).scale(0.4)
        success_label.next_to(mid_line, UP , buff=0.2)

        second_success_label = Text("(based on the composite \n average success rate)", color=WHITE).scale(0.4)
        second_success_label.next_to(mid_line, DOWN , buff=0.2)

        # Draw first set of elements
        self.play(Create(left_container), Create(right_container))
        self.play(Create(top_line), Write(attempts_label))
        self.wait(5)
        self.play(Create(success_line_left), Create(success_line_right), Create(mid_line), Write(success_label), Write(second_success_label))
        self.wait(4)

        # Second set of dashed lines (~80% left, ~78% right)
        extra_success_left_y = container_height * 0.80
        extra_success_right_y = container_height * 0.78

        extra_line_left = DashedLine(
            start=left_container.get_corner(UL) + DOWN * (container_height - extra_success_left_y),
            end=left_container.get_corner(UR) + DOWN * (container_height - extra_success_left_y),
            color=BASELINE_COLOR
        )

        extra_line_right = DashedLine(
            start=right_container.get_corner(UL) + DOWN * (container_height - extra_success_right_y),
            end=right_container.get_corner(UR) + DOWN * (container_height - extra_success_right_y),
            color=BASELINE_COLOR
        )

        self.play(Create(extra_line_left), Create(extra_line_right))

        # Labels for "successes above baseline rate"
        extra_success_label_left = Text("15 additional \nsuccesses", color=BASELINE_COLOR).scale(0.4)
        extra_success_label_left.next_to(extra_line_left, LEFT, buff=0.2)

        extra_success_label_right = Text("5 additional \nsuccesses", color=BASELINE_COLOR).scale(0.4)
        extra_success_label_right.next_to(extra_line_right, RIGHT, buff=0.2)

        self.play(Write(extra_success_label_left), Write(extra_success_label_right))
        self.wait(5.5)

        equation_text = Tex(r"Successes over baseline $\approx$ $(x - \mu)$ $*$ Volume").shift(DOWN * 2.5).scale(0.6)
        self.play(Write(equation_text))
        self.wait(10)

#Animations 7A/7B/
class BlockGScoreSimulation(Scene):

    def construct(self):
        # Simulate the Block differential using player averages
        # Randomly select 13 players for Team A
        n_iterations = 10000
        teams = np.random.choice(representative_players, size=(n_iterations, 25), replace=True)

        # Split into Team A (13 players) and Team B (12 players + 1 remaining)
        team_a = teams[:, :13]
        team_b = teams[:, 13:]

        # Get block levels for each player
        block_levels = weekly_df.groupby('PLAYER_NAME')['BLK'].mean().loc[representative_players]

        block_actual_values = weekly_df.groupby('PLAYER_NAME')['BLK']
        block_actual_values_map = {player : block_actual_values.get_group(player) for player in representative_players}

        results = []

        chart = BarChart(values= [0 for z in range(-30,30)]
                    ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                    ,y_range = [0,10,1]
                    ,y_length = 10
                            ).shift(RIGHT * 3.5).scale(0.5)

        axis_titles = chart.get_axis_labels(
            Tex(""), Tex("\# of occurences")
        )

        manual_axis_title = Text('Difference in totals').scale(0.6).next_to(chart, DOWN)

        self.add(chart, axis_titles, manual_axis_title)

        for i in range(n_iterations):

            display_teams = (i <10) or (i % 200 == 199) 
            wait_time = 1 if i<10 else 0.01

            team_a_players = [player for player in team_a[i, :]]
            team_a_values = [block_levels[player] for player in team_a[i, :]]
            team_a_actual_values = [block_actual_values_map[player].sample(1)[0] for player in team_a[i, :]]

            team_b_players = [player for player in team_b[i, :]]
            team_b_values = [block_levels[player] for player in team_b[i, :]]
            team_b_actual_values = [block_actual_values_map[player].sample(1)[0] for player in team_b[i, :]]

            results = results + [sum(team_a_actual_values) - sum(team_b_actual_values)]

            if display_teams:
            # Animate table and labels

                # Generate uniform tables
                team_a_table = self.create_uniform_table(
                    [(player, str(np.round(block_value, 2)), str(actual_value)) for player, block_value, actual_value in zip(team_a_players, team_a_values, team_a_actual_values)]
                )

                team_b_table = self.create_uniform_table(
                    [(player, str(np.round(block_value, 2)), str(actual_value)) for player, block_value, actual_value in zip(team_b_players, team_b_values, team_b_actual_values)]
                    + [('Remaining Player', '', '')]  # Add extra row to balance height
                )

                # Positioning tables
                team_a_table.to_edge(LEFT, buff=0.2)
                team_b_table.to_edge(LEFT, buff=3.5)

                # Team labels
                team_a_label = Text("Team A", color=BLUE).scale(0.6).next_to(team_a_table, UP)
                team_b_label = Text("Team B", color=RED).scale(0.6).next_to(team_b_table, UP)

                self.add(team_a_label, team_a_table)
                self.add(team_b_label, team_b_table)
                # Clear previous tables before the next iteration
                hist = get_histogram([z for z in range(-30,30)], results, 1, round_normal = False)

                iteration_counter = Text('Iteration number: ' + str(i + 1)).scale(0.6).next_to(team_b_table, DOWN)
                self.add(iteration_counter)

                self.play(chart.animate.change_bar_values(hist))

                self.wait(wait_time)

                self.clear_mobjects_except([team_a_label, team_b_label, chart, axis_titles, manual_axis_title])

                if i == 9:
                    self.remove(chart)
                    chart = BarChart(values= hist
                            ,bar_names = [z if z%2 == 0 else '' for z in range(-30,30)]
                            ,y_range = [0,800,100]
                            ,y_length = 10
                            ,x_axis_config = {'label_direction' : DOWN}
                                    ).shift(RIGHT * 3.5).scale(0.5)

                    self.add(chart)

        results_df = pd.DataFrame({'Result' : results})
        results_df.to_csv('blocks_g.csv', index = False)   
        self.wait(5) 

    def create_uniform_table(self, data):
        """Creates a table with uniform cell sizes."""
        cell_widths = [1.9,0.5,0.5]
        cell_height = 0.4

        table_rows = VGroup()
        for row_data in data:
            row_group = VGroup()
            for cell_text, cell_width in zip(row_data, cell_widths):
                rect = Rectangle(width=cell_width, height=cell_height)
                text = Text(str(cell_text)).scale(0.2)
                text.move_to(rect.get_center())
                row_group.add(VGroup(rect, text))
            row_group.arrange(RIGHT, buff=0)  # Align columns in a row
            table_rows.add(row_group)

        table_rows.arrange(DOWN, buff=0)  # Align rows vertically
        return table_rows

    def clear_mobjects_except(self, keep_objects):
        """Remove all objects from the scene except the ones in keep_objects."""
        for mob in self.mobjects:
            if mob not in keep_objects:
                self.remove(mob)   