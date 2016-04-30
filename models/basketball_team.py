import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import fuzzylite as fuzzy
from fuzzylite import FuzzyVariable, FuzzySystem, Rule

def construct_system():
    # Define linguistic variable 'height' on universe [170, 236] and its terms
    height = FuzzyVariable(np.arange(170, 236 + 1, 1), 'height')
    height['very_tall'] = fuzzy.trapmf(height.universe, [217, 222, 236, 236])
    height['tall']      = fuzzy.trapmf(height.universe, [203, 206, 217, 222])
    height['average']   = fuzzy.trapmf(height.universe, [189, 194, 204, 209])
    height['short']     = fuzzy.trapmf(height.universe, [170, 170, 189, 194])

    # Define linguistic variable 'skill' on universe [0, 100] and its terms
    skill = FuzzyVariable(np.arange(0, 100 + 1, 1), 'skill')
    skill['excellent'] = fuzzy.trapmf(skill.universe, [85, 90, 100, 100])
    skill['very_good'] = fuzzy.trapmf(skill.universe, [60, 65,  85,  90])
    skill['good']      = fuzzy.trapmf(skill.universe, [45, 50,  65,  70])
    skill['average']   = fuzzy.trapmf(skill.universe, [25, 30,  45,  50])
    skill['poor']      = fuzzy.trapmf(skill.universe, [10, 15,  30,  35])
    
    # Define linguistic variable 'confidence' on universe [0, 100] and its terms
    confidence = FuzzyVariable(np.arange(0, 100 + 1, 1), 'confidence')
    confidence['full']    = fuzzy.trapmf(confidence.universe, [80, 85, 100, 100])
    confidence['average'] = fuzzy.trapmf(confidence.universe, [60, 65,  80,  85])
    confidence['low']     = fuzzy.trapmf(confidence.universe, [35, 40,  60,  65])
    confidence['no']      = fuzzy.trapmf(confidence.universe, [ 0,  0,  35,  40])

    # Define rules base
    rules = [
        Rule([ skill['excellent'], height['very_tall'] ], confidence['full']),
        Rule([ skill['excellent'], height['tall']      ], confidence['full']),
        Rule([ skill['excellent'], height['average']   ], confidence['average']),
        Rule([ skill['excellent'], height['short']     ], confidence['average']),
    
        Rule([ skill['very_good'], height['very_tall'] ], confidence['full']),
        Rule([ skill['very_good'], height['tall']      ], confidence['full']),
        Rule([ skill['very_good'], height['average']   ], confidence['average']),
        Rule([ skill['very_good'], height['short']     ], confidence['average']),
    
        Rule([ skill['good'], height['very_tall'] ], confidence['full']),
        Rule([ skill['good'], height['tall']      ], confidence['full']),
        Rule([ skill['good'], height['average']   ], confidence['average']),
        Rule([ skill['good'], height['short']     ], confidence['low']),
    
        Rule([ skill['average'], height['very_tall'] ], confidence['average']),
        Rule([ skill['average'], height['tall']      ], confidence['average']),
        Rule([ skill['average'], height['average']   ], confidence['low']),
        Rule([ skill['average'], height['short']     ], confidence['no']),
    
        Rule([ skill['poor'], height['very_tall'] ], confidence['average']),
        Rule([ skill['poor'], height['tall']      ], confidence['average']),
        Rule([ skill['poor'], height['average']   ], confidence['low']),
        Rule([ skill['poor'], height['short']     ], confidence['no'])
    ]

    # Construct system using the set of rules and chose system's operators
    system = FuzzySystem(rules)
    system.agregation_operator = 'min'
    system.deffuzification_operator = 'mom'

    return system


def main():
    # Construct fuzzy production system
    system = construct_system()

    # Set input values to the system and run
    system.input = { 'height': 190, 'skill': 54 }
    system.produce(save_stages=True)
    
    # Print intermediate system stages
    # print(system.stages.fuzzification)
    # print(system.stages.agregation)
    # print(system.stages.activation)
    # print(system.stages.accumulation)

    # Get system output and print result
    print('confidence = ', system.output['confidence'])


if __name__ == '__main__':
    main()