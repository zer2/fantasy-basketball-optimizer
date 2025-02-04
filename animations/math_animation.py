from manim import *

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
        arrow_color = WHITE

        # Scaling factor to fit everything properly
        scale_factor = 0.9

        # Create number boxes (without numbers inside)
        boxes = VGroup(*[Square(side_length=0.8, color=box_color) for _ in range(9)])
        boxes.arrange(RIGHT, buff=0.6).scale(scale_factor)  # Increased spacing

        categories = VGroup(*[Text(t).next_to(box, DOWN) for t, box in zip(['A','B','C','D','E','F','G','H','I'], boxes)])

        # Create plus signs between numbers
        plus_signs = VGroup(*[
            MathTex("+", font_size=36).next_to(boxes[i], RIGHT, buff=0.15) for i in range(8)
        ])

        # Create equals sign before sum box
        equals_sign = MathTex("=", font_size=40).next_to(plus_signs[-1], DOWN * 10, buff=0.2)

        # Create sum box
        sum_box = Square(side_length=0.8, color=sum_color).scale(scale_factor)
        sum_box.next_to(equals_sign, RIGHT, buff=0.4)


        # Create formula texts
        formula1 = MathTex("F_1 = a_1 + a_2 + \\dots + a_6", font_size=30).next_to(boxes[:6], UP, buff=1.2)
        formula2 = MathTex("F_2 = a_7", font_size=30).next_to(boxes[6], UP, buff=1.2)
        formula3 = MathTex("F_3 = a_8 + a_9", font_size=30).next_to(boxes[7:], UP, buff=1.2)

        # Create arrows
        arrows = VGroup(
            *[Arrow(start=boxes[i].get_top(), end=formula1.get_bottom(), color=arrow_color, buff=0.1) for i in range(6)],
            Arrow(start=boxes[6].get_top(), end=formula2.get_bottom(), color=arrow_color, buff=0.1),
            Arrow(start=boxes[7].get_top(), end=formula3.get_bottom(), color=arrow_color, buff=0.1),
            Arrow(start=boxes[8].get_top(), end=formula3.get_bottom(), color=arrow_color, buff=0.1)
        )

        # Animate the scene
        self.play(Create(boxes), Write(categories))
        self.play(*[Write(plus) for plus in plus_signs])
        self.play(Write(equals_sign))
        self.play(Create(sum_box))
        self.play(Write(formula1), Write(formula2), Write(formula3))
        self.play(Create(arrows))
        self.wait(2)