from manim import *

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
