from manim import *
import random
import pandas as pd

season_df = pd.read_csv('C:/Users/zacha/Projects/FBBO/data_for_testing/2023-24_complete.csv')
representative_players = list(season_df.groupby('PLAYER_NAME')['PTS'].mean().sort_values(ascending = False).index[0:156])
representative_players_text = ['\n'.join(x.split(' ')[0:2]) for x in representative_players]  
representative_players_text = [x for x in representative_players_text if (len(x.split('\n')[0]) < 10) & (len(x.split('\n')[1]) < 10)  ]  

class FantasyBasketballMatchup(Scene):
    def construct(self):
        # Create grids for Team A and Team B


        grid_a = self.create_grid("Team A")
        grid_b = self.create_grid("Team B")

        grid_a.shift(LEFT * 3.5)
        grid_b.shift(RIGHT * 3.5)

        team_a_title = Text("Team A", color=BLUE).next_to(grid_a, UP, buff=0.5)
        team_b_title = Text("Team B", color=GREEN).next_to(grid_b, UP, buff=0.5)

        # Add titles and grids to the scene
        self.play(FadeIn(team_a_title), FadeIn(team_b_title))
        self.play(FadeIn(grid_a), FadeIn(grid_b))

        # Player data for Team A and Team B
        team_a_players = [
            ("Victor Wembanyama", "88 pts, 31 reb, 15 ast, ..."),
            ("Lauri Markannen", "55 pts, 20 reb, 7 ast, ..."),
            ("Kevin Durant", "74 pts, 24 reb, 7 ast, ..."),
        ]

        team_b_players = [
            ("Nikola Jokic", "80 pts, 31 reb, 25 ast, ..."),
            ("Damian Lillard", "62 pts, 8 reb, 22 ast, ..."),
            ("Tyrese Maxey", "67 pts, 96 reb, 16 ast, ..."),
        ]

        # Dynamically fill in stats row by row
        for i in range(3):
            self.fill_grid_row(grid_a, i, team_a_players[i])
            self.fill_grid_row(grid_b, i, team_b_players[i])

        actions_a = self.add_summary_row(grid_a, "(ten more players)", "...", 3)
        actions_b = self.add_summary_row(grid_b, "(ten more players)", "...", 3)
        self.play(actions_a[0], actions_a[1], actions_b[0], actions_b[1])

        # Add a fourth row with grey styling
        actions_a = self.add_summary_row(grid_a, "Total", "793 pts, 273 reb, 132 ast, ...", 4, t2c={'793 pts':YELLOW, '273 reb' : YELLOW})
        actions_b = self.add_summary_row(grid_b, "Total", "724 pts, 240 reb. 159 ast, ...", 4, t2c={'159 ast':YELLOW})
        self.play(actions_a[0], actions_a[1], actions_b[0], actions_b[1])

        self.wait(30)

    def create_grid(self, team_name):
        # Create a grid with four rows and two columns
        grid = VGroup()
        for i in range(5):
            row = VGroup()
            for j in range(2):
                box = Rectangle(height=1, width=3, color=WHITE)
                if i == 3:  # Grey row
                    box.set_fill(GREY, opacity=0.5)
                else:
                    box.set_fill(WHITE, opacity=0.2)
                row.add(box)
            row.arrange(RIGHT, buff=0.1)
            grid.add(row)
        grid.arrange(DOWN, buff=0.1)
        return grid

    def fill_grid_row(self, grid, row_idx, player_data):
        """Fill a specific row in the grid with player data."""
        player_name, player_stats = player_data
        name_box = grid[row_idx][0]
        stats_box = grid[row_idx][1]

        name_text = Text(player_name, font_size=16).move_to(name_box.get_center())
        stats_text = Text(player_stats, font_size=16).move_to(stats_box.get_center())

        self.play(FadeIn(name_text), FadeIn(stats_text))

    def add_summary_row(self, grid, left_text, summary_text, row_idx, t2c = {}):
        """Add a summary or total row in the grid."""
        name_box = grid[row_idx][0]
        stats_box = grid[row_idx][1]

        name_text = Text(left_text, font_size=16).move_to(name_box.get_center())
        stats_text = Text(summary_text, t2c = t2c, font_size=14, disable_ligatures=True).move_to(stats_box.get_center())

        return (FadeIn(name_text), FadeIn(stats_text))

class SlotMachineTwoTables(Scene):
    def construct(self):
        # Define player names
        random.shuffle(representative_players_text)  # Shuffle for randomness

        # Slot machine rectangle
        slot_machine = RoundedRectangle(corner_radius=0.2, width=2, height=2, color=YELLOW)
        slot_machine.to_edge(LEFT, buff=2)

        # Slot machine lever
        lever = Line(UP * 0.5, DOWN * 0.5, color=GRAY).next_to(slot_machine, UP, buff=0.2)
        lever_pivot = Dot(lever.get_start(), color=GRAY)

        # Placeholder for player name
        slot_text = Text("???", font_size=36).move_to(slot_machine)

        # Create two tables (each with 13 spots)
        top_table = VGroup(*[Square(side_length=0.7, color=WHITE) for _ in range(13)])
        bottom_table = VGroup(*[Square(side_length=0.7, color=WHITE) for _ in range(12)] + [Square(side_length=0.7, color=RED)])  # 12 white spots

        # Arrange tables
        top_table.arrange_in_grid(rows = 2, buff=0).next_to(slot_machine, RIGHT, buff=1).shift(UP * 1.5)
        bottom_table.arrange_in_grid(rows = 2, buff=0).next_to(slot_machine, RIGHT, buff=1).shift(DOWN * 1.5)

        team_a_label = Text('Team A', color = BLUE).next_to(top_table, UP)
        team_b_label = Text('Team B', color = GREEN).next_to(bottom_table, UP)

        end_text = Text('Goal: maximize the expected \n value of categories won \n (equivalent to maximizing sum \n of probabilities of winning \n each category)'
                        ,font_size = 12).next_to(bottom_table, RIGHT)

        # Placeholder for selected player names
        selected_names = []

        # Add elements to scene
        self.play(Create(slot_machine), Write(slot_text))
        self.play(Create(lever), Create(lever_pivot))
        self.play(Write(team_a_label), Write(team_b_label))
        self.play(Create(top_table), Create(bottom_table))
        self.play(Write(end_text))

        # Animation for selecting players
        for i in range(25): #26
            # Simulate pulling lever
            self.play(lever.animate.rotate(-PI/6, about_point=lever_pivot.get_center()), run_time=0.2)
            self.play(lever.animate.rotate(PI/6, about_point=lever_pivot.get_center()), run_time=0.2)


            # Simulate slot spinning with random names
            for _ in range(10):  # Quick shuffle effect
                random_name = random.choice(representative_players_text)
                self.play(Transform(slot_text, Text(random_name, font_size=12).move_to(slot_machine)), run_time=0.1)

            # Select final name
            selected_player = representative_players_text.pop(0)  # Get player name
            final_text = Text(selected_player, font_size=12).move_to(slot_machine)
            self.play(Transform(slot_text, final_text), run_time=0.5)

            # Determine which table to place the player in
            if i % 2 == 0:
                # Assign to top table
                name_text = Text(selected_player, font_size=10).move_to(top_table[int(i/2)])
            else:
                j = int(i/2) - 13
                name_text = Text(selected_player, font_size=10).move_to(bottom_table[j])
      

            selected_names.append(name_text)
            self.play(Write(name_text))

        self.wait(2)

