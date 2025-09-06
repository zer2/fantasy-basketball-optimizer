from manim import *

class TableScene(Scene):
    def construct(self):
        # Define the table data
        table_data = [
            ["", "\\sigma^2", "\\tau^2", "Z-score denominator", "G-score denominator", r"G-score as fraction\\of Z-score"],
            ["Assists", "41.57", "31.55", "6.45", "8.55", "75\\%"],
            ["Blocks", "2.35", "2.70", "1.53", "2.25", "68\\%"],
            ["Field Goal \\%", "0.003", "0.007", "0.06", "0.10", "56\\%"],
            ["Free Throw \\%", "0.009", "0.018", "0.10", "0.16", "58\\%"],
            ["Points", "325.52", "448.32", "18.04", "27.82", "65\\%"],
            ["Rebounds", "52.01", "57.55", "7.21", "10.47", "69\\%"],
            ["Steals", "1.01", "4.20", "1.00", "2.28", "44\\%"],
            ["Threes", "9.52", "9.04", "3.09", "4.31", "72\\%"],
            ["Turnovers", "5.45", "8.85", "2.34", "3.78", "62\\%"]
        ]
        
        # Create the table using MobjectTable
        table = MobjectTable([
            [MathTex(cell) if (('sigma' in cell) or ('tau' in cell)) else Tex(cell) for cell in row] for row in table_data
        ], include_outer_lines=True).scale(0.5)
        
        
        # Display the table
        self.play(FadeIn(table))
        self.wait(50)




class TableScene2(Scene):
    def construct(self):
        # Define the table data
        table_data = [
            ["", "Z vs. G", "", "G vs. Z", ""],
            ["", "Most Categories", "Each Category", "Most Categories", "Each Category"],
            ["Aggregate", "0.4\\%", "0.5\\%", "32.5\\%", "21.4\\%"],
            ["Seat 0", "0.5\\%", "0.8\\%", "41.6\\%", "30.7\\%"],
            ["Seat 1", "0.2\\%", "0.3\\%", "21.1\\%", "14.5\\%"],
            ["Seat 2", "0.3\\%", "0.8\\%", "33.7\\%", "29.2\\%"],
            ["Seat 3", "0.5\\%", "0.7\\%", "30.7\\%", "19.1\\%"],
            ["Seat 4", "0.3\\%", "0.4\\%", "22.7\\%", "14.1\\%"],
            ["Seat 5", "0.0\\%", "0.1\\%", "25.6\\%", "14.5\\%"],
            ["Seat 6", "0.1\\%", "0.4\\%", "30.6\\%", "17.2\\%"],
            ["Seat 7", "0.4\\%", "0.1\\%", "53.0\\%", "33.3\\%"],
            ["Seat 8", "0.2\\%", "0.0\\%", "48.8\\%", "38.6\\%"],
            ["Seat 9", "0.5\\%", "0.0\\%", "33.4\\%", "16.9\\%"],
            ["Seat 10", "1.1\\%", "0.8\\%", "22.2\\%", "11.4\\%"],
            ["Seat 11", "0.8\\%", "1.4\\%", "26.8\\%", "17.1\\%"]
        ]
        
        # Create the table
        table = MobjectTable([
            [MathTex(cell) if (('sigma' in cell) or ('tau' in cell)) else Tex(cell) for cell in row] for row in table_data
        ], include_outer_lines=True).scale(0.4)        
        # Title for the table
        title = Text("Simulation results from the 2023 season", font_size=36)
        title.next_to(table, UP, buff=0.5)
        
        # Display the table
        self.play(FadeIn(table), FadeIn(title))
        self.wait(50)

class StatsGrid(Scene):
    def construct(self):
        # Define text labels for the boxes
        labels = [
            "Points", "Threes", "Rebounds",
            "Assists", "Steals", "Blocks",
            "Turnovers", "Field Goal %", "Free Throw %"
        ]

        colors = [GREEN] * 6 + [RED] + [YELLOW] * 2

        # Create corresponding text objects
        text_labels = VGroup(*[Text(label, font_size=24, color = color) for label, color in zip(labels, colors)])
        text_labels.arrange_in_grid(rows=3, cols=3, buff=2)  # Create 3x3 grid

        # Animation: Create each box one by one
        for text in text_labels:
            self.play(Write(text))

        self.wait(50)