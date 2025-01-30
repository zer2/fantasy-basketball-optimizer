from manim import *


class FantasyBasketballDraft(Scene):
    def construct(self):

        # Define managers and players
        managers = ["Team A", "Team B", "Team C", "Team D","Team E","Team F","Team G","Team H","Team I","Team J"]
        players = [
            "Victor\nWemban-\nyama", "Nikola\nJokić", "Luka\nDončić", "Shai\nGilgeous-\nAlexander", 
            "Anthony\nDavis", "Giannis\nAnteto-\nkounmpo", "Tyrese\nHaliburton", "Jayson\nTatum"
            ,"Steph\nCurry","Chet\nHolmgren","Trae\nYoung","Scottie\nBarnes","James\nHarden"
            ,"Anthony\nEdwards","Donovan\nMitchell","Jalen\nBrunson","Joel\nEmbiid","Domantas\nSabonis"
            ,"Damian\nLillard","Lauri\nMarkkanen","Kevin\nDurant","Tyrese\nMaxey","..."
        ]

        num_rounds = 3
        grid_width = len(managers)
        grid_height = num_rounds

        # Create grid
        grid = VGroup()
        cell_width = 1.1
        cell_height = 1

        for row in range(grid_height):
            for col in range(grid_width):
                cell = Rectangle(width=cell_width, height=cell_height)
                cell.move_to(
                    np.array([
                        col * cell_width - (grid_width - 1) * cell_width / 2,
                        -(row * cell_height) + (grid_height - 1) * cell_height / 2,
                        0
                    ])
                )
                grid.add(cell)

        # Add grid to the scene
        self.play(Create(grid), run_time = 0.1)

        additional_grid = VGroup()
        additional_rounds = 1  # Example for one extra round
        for row in range(additional_rounds):
            for col in range(grid_width):
                cell = Rectangle(width=cell_width, height=cell_height, stroke_opacity=0.3)
                cell.move_to(
                    np.array([
                        col * cell_width - (grid_width - 1) * cell_width / 2,
                        -(grid_height + row) * cell_height + (grid_height - 1) * cell_height / 2,
                        0
                    ])
                )
                additional_grid.add(cell)

        self.play(Create(additional_grid), run_time = 0.1)

        # Add round labels on the left
        for round_number in range(num_rounds):
            round_label = Text(f"Round {round_number + 1}", font_size=12, stroke_opacity=0.3)
            round_label.next_to(grid[round_number * grid_width], LEFT, buff=0.2)
            self.play(Write(round_label), run_time = 0.1)


        # Add "Round N+1" label for the extra row
        next_round_label = Text(f"Rounds\n4 - 13", font_size=12)
        next_round_label.next_to(additional_grid[0], LEFT, buff=0.2)
        self.play(Write(next_round_label), run_time = 0.1)

        # Add manager names as column headers
        for i, manager in enumerate(managers):
            header = Text(manager, font_size=12)
            header.next_to(grid[i], UP, buff=0.2)
            self.play(Write(header), run_time = 0.1)


        # Add players in snake order
        player_index = 0
        for round_number in range(num_rounds):
            if round_number % 2 == 0:  # Left-to-right
                columns = range(len(managers))
            else:  # Right-to-left
                columns = reversed(range(len(managers)))

            for col in columns:
                if player_index >= len(players):
                    break

                player_name = players[player_index]
                row_position = round_number
                position = col + (row_position * grid_width)
                player_cell = grid[position]

                player_text = Text(player_name, font_size=12, color = 'blue')

                player_text.move_to(player_cell.get_center())

                self.play(FadeIn(player_text), run_time = 0.6)
                player_text.set_color('white')

                player_index += 1

        # Add indication of more rounds

        self.wait(5)

