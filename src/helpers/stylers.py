import numpy as np
from abc import ABC, abstractmethod

class Styler(ABC):
    #Define the functions needed for styling. 

    @abstractmethod
    def styler_a(self, value : float) -> str:
        pass

    @abstractmethod
    def styler_b(self, value : float) -> str:
        pass

    @abstractmethod
    def styler_c(self, value : float) -> str:
        pass

    @abstractmethod
    def style_rosters(self, x: str, my_players : list[str]) -> str:
        pass

    @abstractmethod
    def color_blue(self, label : str, target : str) -> str:
        pass

    @abstractmethod
    def stat_styler_primary(self, value : float, multiplier : float, middle : float) -> str:
        pass

    @abstractmethod
    def stat_styler_secondary(self, value : float, multiplier : float, middle : float) -> str:
        pass

    @abstractmethod
    def stat_styler_secondary(self, value : float, multiplier : float, middle : float) -> str:
        pass

class DarkStyler(Styler):
         
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
            rgb = (90,90 ,150)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:white;'
        else:
            rgb = (100,100 ,240)
            bgc = '#%02x%02x%02x' % rgb
            return 'background-color:' + bgc + '; color:white;'
    
    def color_blue(self, label, target):
        return "background-color: #444466; color:white" if label == target else None
    
    def stat_styler_primary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        """Styler function used for coloring stat values blue/magenta with varying intensities
        For dark mode, the default is for the color to be dark, and light increases with more extrme values 

        Args:
        value: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 
        multiplier: degree to which intensity of color scales relative to input value 
        middle: value that should map to pure white 
        Returns:
        String describing format for a pandas styler object
        """
            
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        intensity = min(int(abs((value-middle)*multiplier)* 0.8), 165)

        if (value - middle)*multiplier > 0:
            rgb = (90 ,90 + intensity, 90 + intensity)
        else:
            rgb = (90  + intensity,90,90 + intensity)

        return final_formatter(rgb)

    def stat_styler_secondary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        """Styler function used for coloring stat values yellow/orange with varying intensities 

        Args:
            value: any value 
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to pure white 
        Returns:
            String describing format for a pandas styler object
        """
                
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        intensity = min(int(abs((value-middle)*multiplier)), 150)

        if (value - middle)*multiplier > 0:
            rgb = (130 + int(2 * intensity/3), 130 + int(2 * intensity/3),130)

        else:
            rgb = (130 + int(intensity),130 + int(intensity/3),130)

        return final_formatter(rgb)
    

        
    def stat_styler_tertiary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        """Styler function used for coloring stat values blue with varying intensities 

        Args:
            value: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to pure white 
        Returns:
            String describing format for a pandas styler object
        """
                
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        intensity = min(int(abs((value-middle)*multiplier)), 100)

        if (value - middle)*multiplier > 0:
            rgb = (100, 100, 110 + intensity)
        else:
            rgb = (100, 100, 110 -int(intensity/10))

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
        """Styler function used for coloring stat values blue/magenta with varying intensities
        For dark mode, the default is for the color to be dark, and light increases with more extrme values 

        Args:
            value: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 
            multiplier: degree to which intensity of color scales relative to input value 
            middle: value that should map to pure white 
        Returns:
            String describing format for a pandas styler object
        """
                
        if value == -999:
            return 'background-color:#F6F6F6;color:#F6F6F6;'
        
        intensity = min(int(abs((value-middle)*multiplier)), 255)

        if (value - middle)*multiplier > 0:
            rgb = (255 -  intensity,255 , 255 -  intensity)
        else:
            rgb = (255, 255 - intensity, 255 - intensity)

        return final_formatter(rgb)
    
    def stat_styler_secondary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        """Styler function used for coloring stat values yellow/orange with varying intensities 

        Args:
        value: any value 
        multiplier: degree to which intensity of color scales relative to input value 
        middle: value that should map to pure white 
        Returns:
        String describing format for a pandas styler object
        """
            
        if value == -999:
            return 'background-color:#8D8D9E;color:#8D8D9E;'
        
        intensity = min(int(abs((value-middle)*multiplier)), 255)

        if (value - middle)*multiplier > 0:
            rgb = (255,255 , 255 - intensity)
        else:
            rgb = (255,255 - intensity,255)
            
        return final_formatter(rgb)


    def stat_styler_tertiary(self, value : float, multiplier : float = 50, middle : float = 0) -> str:
        """Styler function used for coloring stat values red/green with varying intensities 

        Args:
        value: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 
        multiplier: degree to which intensity of color scales relative to input value 
        middle: value that should map to pure white 
        Returns:
        String describing format for a pandas styler object
        """
            
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


  