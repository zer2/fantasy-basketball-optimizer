"""What the Most Categories animation says out loud.

WIREFRAME. Read by gTTS, not Alistair, while the script is still moving -- see
shared/draft_voice.py. Lines here are drafts and are expected to change.

This is the only file to edit for narration. Each key is one beat of the scene, and that beat's
animation plays INSIDE its line: rewriting a line retimes its own beat rather than
desynchronising everything after it.

The voice reads what is written here literally, so spell out anything a reader would say
differently from how it is written ('sigma', 'two to the ninth').
"""

NARRATION = {
    # -- A week, and who took it -------------------------------------------------------
    'a_week':
        'With Most Categories scoring, what matters is winning a majority of categories. In the case '
        'of 9-cat, that is five out of nine categories. Winning any more than five gives no additional '
        'benefit, and losing more than four incurs no additional cost.',
    'all_or_nothing':
        'So when we think about optimizing for a matchup, we should be aiming to win that majority as often as possible. '
        'That is not the same thing as optimizing for the expected value of categories won, so we need a new '
        'objective function for it, not just adding up category-level win probabilities.',
    # -- What has to be added up -------------------------------------------------------
    'the_table':
        'There are many different ways to get to the majority of categories, and each has its own probability. '
        'Theoretically, we could just calculate the probability of each of these scenarios and add them up.',
    'how_many_rows':
        'There are two to the ninth of those rows. Two to the eighth if we only consider the winning scenarios. '
        'That is only 256, which is not so bad, except that this needs to run for every candidate player for every '
        'iteration of the algorithm. Plus, if we add a single additional category, that doubles the number of operations. '
        'So it would be best to find a better way to handle this calculation.',
    # -- The walk ----------------------------------------------------------------------
    'dynamic':
        'We can be smarter about how we traverse various scenarios. Within any group of categories, it only matters '
        'how many of them we won, not which ones we won. So if we assume that the categories are independent '
        'from each other, we can walk through the categories one by one, keeping track of the distribution '
        'of how many we have won. For example, if the first two categories are both 50-50, we know that by the end, we will '
        'have a 50 percent chance of being even, and 25 percent chances of being 2-0 or 0-2. That\'s all we need to remember; '
        'there is no need to keep track of the precise likelihood of winning the first category and losing the second.',
    'collapse':
        'Extending the walking analogy, winning a category is like walking one level upwards, and losing '
        'a category is walking one level down. We can just keep track of the probability of being at any particular level '
        'at any particular time. It does not matter how we got to that level; just the probability that we got there. And '
        'what matters is the total probability that we end up above the middle line. So we can walk forward through all '
        'nine categories, then add up the probabilities above the middle line at the end.',
    # -- What a category is worth ------------------------------------------------------
    'tipping':
        'This is elegant, but there is also a complication. For gradient descent, we need to calculate the slope, or gradient of all of the '
        'functions we use. How do we do that with such a complex function? Fortunately, there is a relatively easy way to calculate the '
        'gradient of this function with only a small extension to the machinery we just built.',
    'tipping_point_probability':
        'Let\'s pick a category. The slope in the direction of winning that category more is the responsiveness of the function: by increasing '
        'the probability of winning this category, how much do we increase the probability of winning a majority? Well, there is an intuitive '
        'way to think about that responsiveness. If the other eight categories are precisely even, then the probability of winning a majority is exactly '
        'the probability of winning this category. If the other eight categories are not even, then this category does not matter at all; '
        'the matchup has already been won or lost. That means, the responsiveness is just the probability the other categories are even- '
        'we call this a tipping point probability. ',
    'backward':
        'To calculate the tipping point probability, all we need to do is run the walking process again, this time backwards. '
        'We start at the last category and move to the first category. This will give us the information we need.',
    'the_cut':
        'Take any of the categories. Since our two walks went in opposite directions, the categories that lead up to the category '
        'on one walk are the opposite of the categories that lead up to it on the other walk. The two partial walks give us complete '
        'information on the scenarios for all of the other categories.',
    'convolution':
        'All we have to do is check the convolutions; if the left side was at plus some level and the right side was at minus that level, '
        'they offset and we are at a tipping point. So we multiply the opposing numbers to each other and add up the products.',
    'slide':
        'We can easily do this for any category. Each cut preserves all the information we need on the left and right side to know '
        'exactly what happens to the other categories. ',
    'punting':
        'This gives us a mathematical way of thinking about why punting is so good in Most Categories. If we are winning five categories '
        'decisively and losing four decisively, the tipping point probability is very high for the contested categories, since after '
        'excluding one of them, we are at a four-four tie with the others. For the punted categories, the tipping point probability is low, '
        'since we will usually be winning the others 5-3.',
}
