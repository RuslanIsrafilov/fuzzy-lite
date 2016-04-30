import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import fuzzylite as fuzzy
from fuzzylite import FuzzyVariable, FuzzySystem, Rule

def construct_system():
    quality = FuzzyVariable(np.arange(0, 10 + 1, 1), 'quality')
    quality['poor']    = fuzzy.trimf(quality.universe, [0,  0,  5])
    quality['average'] = fuzzy.trimf(quality.universe, [0,  5, 10])
    quality['good']    = fuzzy.trimf(quality.universe, [5, 10, 10])

    service = FuzzyVariable(np.arange(0, 10 + 1, 1), 'service')
    service['poor']    = fuzzy.trimf(quality.universe, [0,  0,  5])
    service['average'] = fuzzy.trimf(quality.universe, [0,  5, 10])
    service['good']    = fuzzy.trimf(quality.universe, [5, 10, 10])

    tip = FuzzyVariable(np.arange(0, 25 + 1, 1), 'tip')
    tip['low']    = fuzzy.trimf(tip.universe, [0,   0, 13])
    tip['medium'] = fuzzy.trimf(tip.universe, [0,  13, 25])
    tip['high']   = fuzzy.trimf(tip.universe, [13, 25, 25])

    rules = [
        Rule([quality['poor'], service['poor']], tip['low']),
        Rule([service['average']], tip['medium']),
        Rule([service['good'], quality['good']], tip['high'])
    ]

    return FuzzySystem(rules)

def main():
    system = construct_system()
    system.input = { 'quality': 6.5, 'service': 9.8 }
    system.produce()
    print(system.output)

if __name__ == '__main__':
    main()
