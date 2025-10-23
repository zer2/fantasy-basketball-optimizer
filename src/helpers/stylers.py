import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

from src.helpers.helper_functions import get_selected_categories, get_styler

class Styler(ABC):
    #Define the functions needed for styling. 

    @abstractmethod
    def styler_a(self, value : float) -> str:
        #used for overall H-scores and G-scores
        pass

    @abstractmethod
    def styler_b(self, value : float) -> str:
        #used for totals of totals, e.g. total G-scores
        pass

    @abstractmethod
    def styler_c(self, value : float) -> str:
        #used for the G-score totals of slates of players to be traded
        pass

    @abstractmethod
    def style_rosters(self, x: str, my_players : list[str]) -> str:
        #apply styling for the rosters row, which shows position allocations for the candidate player and previously chosen players
        pass

    @abstractmethod
    def color_blue(self, label : str, target : str) -> str:
        #highlights one of the index values blue to make it obvious that the row is for the player in question
        pass

    @abstractmethod
    def stat_styler_primary(self, value : float, multiplier : float, middle : float) -> str:
        """style category-level characteristics 

        Args:
            value: value of the cell
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to the default color, in the middle of the positive and negative color scales
        Returns:
            String describing format for a pandas styler object
        """
        pass

    @abstractmethod
    def stat_styler_secondary(self, value : float, multiplier : float, middle : float) -> str:
        """Style overall totals or differences 

        Args:
            value: value of the cell
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to the default color, in the middle of the positive and negative color scales
        Returns:
            String describing format for a pandas styler object
        """
        pass

    @abstractmethod
    def stat_styler_tertiary(self, value : float, multiplier : float, middle : float) -> str:
        """style algorithm decisions, like category-level weights 

        Args:
            value: value of the cell
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to the default color. 
                    in dark mode, values below this threshold have a smaller marginal effect on color 
        Returns:
            String describing format for a pandas styler object
        """
        pass

class DarkStyler(Styler):
    #for dark mode
         
    def styler_a(self, value : float) -> str:
        return "background-color:#2a2a33;color:white;" 

    def styler_b(self, value : float) -> str:
        return "background-color:#38384A;color:white;" 
    
    def styler_c(self, value : float) -> str:
        return "background-color:#252536;color:darkgrey;" 

    def style_rosters(self, x, my_players):
        if len(x) ==0:
            return 'background-color:#888899'
        elif x in my_players:
            rgb = (70,70 ,150)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:white;'
        else:
            rgb = (90,90 ,240)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:white;'
    
    def color_blue(self, label, target):
        return "background-color: #444466; color:white" if label == target else None
    
    def stat_styler_primary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For dark mode, the default is for the color to be dark, and light increases with more extrme values 
        #color scheme is blue for positive and magenta for negative 
    
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        raw_intensity = (value-middle)*multiplier
        intensity = min(int(abs(raw_intensity)), 165)

        if raw_intensity > 0:
            rgb = (90 ,90 + intensity, 90 + intensity)
        else:
            rgb = (90  + intensity,90,90 + intensity)

        return final_formatter(rgb)

    def stat_styler_secondary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For dark mode, the default is for the color to be dark, and light increases with more extrme values 
        #color scheme is yellow for positive and orange for negative 
                
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        raw_intensity = (value-middle)*multiplier
        intensity = min(int(abs(raw_intensity)), 255)

        if raw_intensity > 0:
            rgb = (130 + int(2 * intensity/3), 130 + int(2 * intensity/3),130)

        else:
            rgb = (130 + int(intensity),130 + int(intensity/3),130)

        return final_formatter(rgb)
    
    def stat_styler_tertiary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For dark mode, the default is for the color to be dark, and light increases with more extrme values 
        #color scheme is varying shades of blue. 

        #Values below the 'middle' have their intensities minimized, so that proper contrast can be drawn 
        #from values at and above the middle while unusually lower values are still visually distinct
                
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        raw_intensity = int(abs((value-middle)*multiplier))
        intensity = min(raw_intensity, 185)

        if raw_intensity > 0:
            rgb = (60, 60, 70 + intensity)
        else:
            rgb = (60, 60, 70 -int(intensity/20))

        return final_formatter(rgb)
    
class LightStyler(Styler):
    def styler_a(self, value : float) -> str:
        return "background-color: grey; color:white;" 
    
    def styler_b(self, value : float) -> str:
        return "background-color: lightgrey; color:black;" 
    
    def styler_c(self, value : float) -> str:
        return "background-color: darkgrey; color:black;" 
    
    def style_rosters(self, x, my_players):
        if len(x) ==0:
            bgc = "#F8F8F8"
            return 'background-color:' + bgc
        elif x in my_players:
            rgb = (220,220 ,255)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:black;'
        else:
            rgb = (175,175 ,255)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:black;'
        
    def color_blue(self, label, target):
        return "background-color: blue; color:white" if label == target else None
        
    def stat_styler_primary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For light mode, the default is pure white and color is generated with subtraction
        #positive values are green and negative values are red
                
        if value == -999:
            return 'background-color:#F6F6F6;color:#F6F6F6;'
        
        raw_intensity = (value-middle)*multiplier
        intensity = min(int(abs(raw_intensity)), 255)

        if raw_intensity > 0:
            rgb = (255 -  intensity,255 , 255 -  intensity)
        else:
            rgb = (255, 255 - intensity, 255 - intensity)

        return final_formatter(rgb)
    
    def stat_styler_secondary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For light mode, the default is pure white and color is generated with subtraction
        #positive values are yellow and negative values are purple 
            
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        raw_intensity = (value-middle)*multiplier
        intensity = min(int(abs(raw_intensity)), 255)

        if raw_intensity > 0:
            rgb = (255,255 , 255 - intensity)
        else:
            rgb = (255,255 - intensity,255)
            
        return final_formatter(rgb)


    def stat_styler_tertiary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        #For light mode, the default is for the color to be light, and color created by subtraction for extreme values
        #color scheme is varying shades of blue. 
            
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        intensity = int(np.clip(abs((value-middle)*multiplier),0,100))
        rgb = (255 - intensity, 255 - intensity , 255)

        return final_formatter(rgb)
        
def final_formatter(rgb : tuple) ->str:
  #takes an rgb code and returns formatting for it 
      
  bgc = '#%02x%02x%02x' % rgb

  #formula adapted from
  #https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  darkness_value = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
  tc = 'black' if darkness_value > 150 else 'white'
  return f"background-color: " + str(bgc) + ";color:" + tc + ";" 


  
#ZR: Ideally this should merge with the h percentage styler, so it can handle anything
def static_score_styler(df : pd.DataFrame, multiplier : float, total_multiplier : float = None) -> pd.DataFrame:
  """Helper function for styling tables of Z or G scores

  Args:
    df: DataFrame with columns per category and total. Additional columns optional
    
  Returns:
    Styled dataframe
  """

  agg_columns = [col for col in ['H-score','Gnrc. $','Orig. $','Total'] if col in df.columns]
  index_columns = [col for col in ['Rank','Player'] if col in df.columns]
  perc_columns = ['H-score'] if 'H-score' in df.columns else []
  total_rows = [r for r in ['Total diff','Total'] if r in df.index]

  colored_total_column = ['Total'] if (('H-score' in df.columns) and ('Total' in df.columns)) else []

  if colored_total_column:
    total_middle = df[colored_total_column[0]].nlargest(10).iloc[-1]
  else:
    total_middle = 0

  df = df[index_columns + agg_columns + get_selected_categories()]

  styler = get_styler()

  df_styled = df.style.format("{:.2f}"
                              , subset = pd.IndexSlice[:,agg_columns + get_selected_categories()]) \
                              .format("{:.1%}"
                                , subset = pd.IndexSlice[:,perc_columns] ) \
                            .map(styler.styler_a
                                ,subset = pd.IndexSlice[:,agg_columns]) \
                            .map(styler.styler_b
                                ,subset = pd.IndexSlice[total_rows,agg_columns]) \
                            .map(styler.stat_styler_primary
                              , subset = pd.IndexSlice[:,get_selected_categories()]
                              , multiplier = multiplier) \
                            .map(styler.stat_styler_secondary
                                 , subset = pd.IndexSlice[:,colored_total_column]
                                 , multiplier = total_multiplier
                                 , middle = total_middle
                                 )
  return df_styled

def h_percentage_styler(df : pd.DataFrame
                        , middle : float = 0.5
                        , multiplier : float = 300
                        , drop_player = None) -> pd.DataFrame:
  """Helper function for styling tables of H-score results

  Args:
    df: DataFrame with columns per category and overall H-score. Additional columns optional
        Values are ratios between 0 and 1
  Returns:
    Styled dataframe
  """
  perc_column = 'H-score' if 'H-score' in df.columns else 'Overall'

  styler = get_styler()

  df_styled = df.style.format("{:.2%}"
                                , subset = pd.IndexSlice[:,[perc_column]] ) \
                          .format("{:.1%}"
                                , subset = pd.IndexSlice[:,get_selected_categories()]) \
                          .map(styler.styler_a
                                , subset = pd.IndexSlice[:,[perc_column]]) \
                          .map(styler.stat_styler_primary
                              , middle = middle
                              , multiplier = multiplier
                              , subset = get_selected_categories())
  
  if drop_player is not None:
     df_styled = df_styled.map(styler.color_blue
                               , subset = pd.IndexSlice[:,['Player']]
                               , target = drop_player)
  return df_styled