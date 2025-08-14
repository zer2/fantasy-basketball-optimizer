from manim import *
import pandas as pd 
import numpy as np

class ZScoretoGScore(Scene):

    def construct(self):

        left_title = Text("Z-Scores").shift(LEFT * 2 + UP * 2)
        right_title = Text("G-scores").shift(RIGHT * 2 + UP * 2)
        left_equation = Tex(r"$\frac{x - \mu}{\sigma}$").next_to(left_title, DOWN * 4).scale(1.5)
        left_caption = Tex(r"$ \sigma$" + r" is the standard deviation \\ across relevant players" ) \
                                .scale(0.4).next_to(left_equation, DOWN)
        right_equation = Tex(r"$\frac{x - \mu}{\sqrt{\sigma^2 + \tau^2}}$").next_to(right_title, DOWN * 4).scale(1.5)
        right_caption = Tex(r"$ \tau$" + r" is the average period-to-period \\ standard deviation for players. \\" + \
                            r"In head-to-head, a period is a week. \\In Rotisserie, a period is a season ") \
                                .scale(0.4).next_to(right_equation, DOWN)

        middle_arrow = Arrow(start = LEFT, end = RIGHT).next_to(left_equation, RIGHT * 2)

        self.play(Write(left_title), Write(left_equation), Write(left_caption))
        self.wait(3)
        self.play(Create(middle_arrow))
        self.play(Write(right_title), Write(right_equation), Write(right_caption))
        self.wait(30)

class TooManyRebounders(Scene):
    def construct(self):
        # Define categories
        categories = [
            "Field Goal %", "Free Throw %", "Threes",
            "Points", "Rebounds", "Assists",
            "Steals", "Blocks", "Turnovers"
        ]
        
        # Create rectangles with labels and bars
        boxes = []
        labels = []
        bars = []
        
        for i, category in enumerate(categories):
            rect = Square(side_length=2).set_stroke(WHITE, width=2)
            label = Text(category, font_size=24).next_to(rect, DOWN * 0.3)
            bar = Rectangle(height=1, width=1.5, color=BLUE, fill_opacity=1).move_to(rect.get_bottom() + UP * 0.5, aligned_edge=DOWN)
            
            boxes.append(rect)
            labels.append(label)
            bars.append(bar)
        
        # Arrange only boxes in a 3x3 grid
        box_group = VGroup(*boxes).arrange_in_grid(rows=3, cols=3, buff=0.5)
        
        # Position bars inside boxes
        for box, bar in zip(boxes, bars):
            bar.move_to(box.get_bottom() + UP * 0.5, aligned_edge=DOWN)
        
        # Position labels below each box
        for box, label in zip(boxes, labels):
            label.next_to(box, DOWN * 0.5)
        
        # Animate
        self.play(Create(VGroup(*boxes)))
        self.play(Write(VGroup(*labels)))
        self.play(GrowFromEdge(VGroup(*bars), edge=DOWN))
        self.wait(1)
        
        # Animate bar adjustments
        for _ in range(13):
            self.play(
                *[bars[i].animate.stretch_to_fit_height(bars[i].height - 0.05, about_edge=DOWN) for i in range(9) if categories[i] != "Rebounds"],
                bars[categories.index("Rebounds")].animate.stretch_to_fit_height(bars[categories.index("Rebounds")].height + (0.05 - i/200), about_edge=DOWN)
            )
        
        self.wait(2)


from manim import *

class ZScoreDefinition(Scene):
    def construct(self):
        # Define colors
        box_color = BLUE
        sum_color = RED
        arrow_color = BLUE
        bracket_color = YELLOW
        # Scaling factor to fit everything properly
        scale_factor = 0.9

        title = Title('Formula for a Z-score')

        # Create number boxes (without numbers inside)
        boxes = VGroup(*[Square(side_length=0.8, color=box_color) for _ in range(9)])
        boxes.arrange(RIGHT, buff=0.6).scale(scale_factor).shift(DOWN * 0.7)  # Increased spacing

        categories = ['Points','Threes', 'Rebounds','Assists','Steals','Blocks','Turnovers','Field Goal \%','Free Throw \%']
        category_text = VGroup(*[Tex(t, font_size = 16).next_to(box, DOWN) for t, box in zip(categories, boxes)])

        # Create plus signs between numbers
        plus_signs = VGroup(*[
            MathTex("+", font_size=36).next_to(boxes[i], RIGHT, buff=0.15) for i in range(8)
        ])

        # Create equals sign before sum box
        equals_sign = MathTex("=", font_size=40).next_to(plus_signs, DOWN * 7, buff=0.2)

        # Create sum box
        sum_box = Square(side_length=0.8, color=sum_color).scale(scale_factor)
        sum_box.next_to(equals_sign, RIGHT, buff=0.4)
        sum_text = Tex('Total', font_size = 16).next_to(sum_box, RIGHT)

        # Create formula texts
        formula1 = MathTex(r"\frac{x - \mu}{\sigma}", font_size=30).next_to(boxes[:6], UP, buff=1.2)
        formula2 = MathTex(r"\frac{\mu - x}{\sigma}", font_size=30).next_to(boxes[6], UP, buff=1.2)
        formula3 = MathTex(r"\frac{x_V}{\mu_V} \frac{x_r - \mu_r}{\sigma}", font_size=30).next_to(boxes[7:], UP, buff=1.2)

        # Create arrows
        arrows = VGroup(
            *[Arrow(start=boxes[i].get_top(), end=formula1.get_bottom(), color=arrow_color, buff=0.2) for i in range(6)],
            Arrow(start=boxes[6].get_top(), end=formula2.get_bottom(), color=arrow_color, buff=0.2),
            Arrow(start=boxes[7].get_top(), end=formula3.get_bottom(), color=arrow_color, buff=0.2),
            Arrow(start=boxes[8].get_top(), end=formula3.get_bottom(), color=arrow_color, buff=0.2)
        )

        # Create brackets
        bracket_1 = Brace(boxes[:7], UP , buff = 2, color=bracket_color)
        bracket_2 = Brace(boxes[7:], UP, buff = 2, color=bracket_color)
        bracket_1_text = Tex('Counting statistics', font_size = 22).next_to(bracket_1, UP)
        bracket_2_text = Tex('Percentage statistics', font_size = 22).next_to(bracket_2, UP)

        bottom_strings = VGroup(Tex(r"$x$ is a player's \\ average for a category", font_size = 16)
                        ,Tex(r"$\mu$ is the mean across relevant players \\ $\sigma$ is the standard deviation", font_size = 16)
                        ,Tex(r"$x_V$ is a player's average shot volume \\$x_r$ is their success rate", font_size = 16)
                        ,Tex(r"$\mu_r$ is the aggregate mean success rate across relevant players. \\$\sigma$ is the standard deviation of $\frac{x_V}{x_\mu}$ * $x_r - \mu_r$ "
                             , font_size = 16)
                                ).arrange(RIGHT)
        bottom_strings.to_edge(DOWN)

        # Animate the scene
        self.play(Write(title))
        self.play(Create(boxes), Write(category_text))
        self.wait(3)
        self.play(Write(formula1))
        self.play(*[Write(plus) for plus in plus_signs])
        self.play(Write(equals_sign))
        self.play(Create(sum_box), Write(sum_text))
        self.wait(2)
        self.play(Write(formula2), Write(formula3))
        self.play(Create(arrows), )
        self.play(Create(bracket_1), Create(bracket_2), Write(bracket_1_text),Write(bracket_2_text) )
        self.play(Write(bottom_strings))
        self.wait(120)

class ShiftedMultiModalPDF(Scene):
    def construct(self):
        # Shift the mean to a nonzero value
        mu_value = 2.2  # Changed mean value for visualization

        # Axes, shifted to ensure the mean isn't at zero
        axes = Axes(
            x_range=[-2, 6, 1],  # Shifted right
            y_range=[0, 1, 0.2],
            axis_config={"color": BLUE},
            x_length=8,
            y_length=4,
            tips=False,
        ).shift(UP)

        labels = axes.get_axis_labels(x_label="x", y_label="PDF(x)")

        # Define a multi-modal probability density function (shifted right)
        def pdf(x):
            return 0.4 * np.exp(-(x - 2)**2) + 0.3 * np.exp(-0.5 * (x - 4)**2) + 0.25 * np.exp(-0.5 * (x - 0.5)**2)

        # Graph of the PDF
        pdf_graph = axes.plot(pdf, color=YELLOW)

        # Mean (mu) - marked as a point on the x-axis
        mu_dot = Dot(axes.c2p(mu_value, 0), color=RED)
        mu_label = MathTex(r"\mu").next_to(mu_dot, DOWN, buff=0.3).set_color(RED)

        # Standard deviation (sigma) - positioned correctly
        sigma_value = 1  # Arbitrary visualization
        sigma_left = axes.c2p(mu_value - sigma_value, 0)
        sigma_right = axes.c2p(mu_value + sigma_value, 0)
        sigma_line = Line(sigma_left, sigma_right, color=GREEN)

        # Move sigma label up and away from mu
        sigma_label = MathTex(r"\sigma").set_color(GREEN).next_to(sigma_line, UP, buff=0.4)  # Now positioned *above* the sigma line

        # Mark a specific x value
        x_value = 3.5  # Arbitrary x value
        x_dot = Dot(axes.c2p(x_value, 0), color=BLUE)
        x_label = MathTex(r"x").next_to(x_dot, DOWN, buff=0.3).set_color(BLUE)

        # Z-score formula below the chart for redundancy
        z_score_bottom = MathTex(r"Z = \frac{x - \mu}{\sigma}").scale(1.2)
        z_score_bottom.next_to(axes, DOWN, buff=1)

        # Animations
        self.play(Create(axes), Write(labels))
        self.play(Create(pdf_graph), run_time=2)
        self.play(Create(mu_dot), Write(mu_label))
        self.play(Create(sigma_line), Write(sigma_label))
        self.play(Create(x_dot), Write(x_label))
        self.play(Write(z_score_bottom))  # Display equation below the chart
        self.wait(50)


class Conclusions(Scene):
    def construct(self):
        conclusion_1 = Text('1. There is an elegant mathematical argument justifying the use \n of Z-scores', color = WHITE, font_size = 30)
        conclusion_2 = Text('2. The argument can be improved to make a more appropriate \n version of Z-scores', color = WHITE, font_size = 30)
        conclusion_3 = Text('3. There is still no perfect way to make a ranking system in a vacuum \n because value depends on context', color = WHITE, font_size = 30)

        conclusions = VGroup(conclusion_1, conclusion_2, conclusion_3)
        conclusions.arrange(DOWN, center = True, aligned_edge = LEFT, buff = 1).shift(LEFT * 0.25)

        title = Title('Conclusions', color = WHITE)
        self.play(Write(title))
        self.wait(50)

        for c in conclusions:

            self.play(Write(c))
            self.wait(50)

        for c in conclusions:

            self.play(c.animate.set_color(YELLOW))
            self.wait(50)
            self.play(c.animate.set_color(WHITE))

class NBAGameLogScroll(Scene):
    def construct(self):
        # Create dummy game log data.
        header = ["Game", "Points", "Rebounds", "Assists"]
        data = []
        # Generate 30 rows of sample data.
        for i in range(1, 31):
            game = str(i)
            points = str(5 + np.random.randint(0,15))
            rebounds = str(4 + np.random.randint(0,5))
            assists = str(3 + np.random.randint(0,4))
            data.append([game, points, rebounds, assists])
        
        # Create a table from the header and data.
        game_log_table = Table(
            [header] + data,
            include_outer_lines=True,
            h_buff=0.8,
            v_buff=0.5,
            element_to_mobject=Text,
        ).scale(0.6)
        
        # ---------------------------------------------------------------------
        # For the scrolling effect we assume that only a portion of the table
        # is visible at a time (a “viewport”). We create a yellow rectangle to
        # indicate that viewport (this rectangle does not actually clip the table).
        # ---------------------------------------------------------------------
        viewport_height = 4
        viewport = Rectangle(width=config.frame_width - 1, height=viewport_height, color=YELLOW).set_opacity(0)
        viewport.move_to(ORIGIN)
        self.add(viewport)
        
        # Position the table so that its top edge aligns with the top edge of the viewport.
        table_top_y = game_log_table.get_top()[1]
        viewport_top_y = viewport.get_top()[1]
        game_log_table.shift(np.array([0, viewport_top_y - table_top_y, 0]))
        
        # Add the table (it will extend beyond the viewport).
        self.add(game_log_table)
        self.wait(1)
        
        # ---------------------------------------------------------------------
        # Now we compute the scroll distance.
        # We want to scroll until the table’s bottom aligns with the viewport’s bottom.
        # ---------------------------------------------------------------------
        table_bottom_y = game_log_table.get_bottom()[1]
        viewport_bottom_y = viewport.get_bottom()[1]
        scroll_distance = viewport_bottom_y - table_bottom_y
        
        # Animate the table shifting upward (which gives the appearance of scrolling down).
        self.play(game_log_table.animate.shift(UP * scroll_distance), run_time=10)
        self.wait(2)