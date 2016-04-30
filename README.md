# fuzzy-lite
Lightweight and simple fuzzy logic system implementation

## Features
* scikit-fuzzy like API
* Custom operators in production system
* Access to all intermediate system outputs

## Quick start
```python
import numpy as np
import fuzzylite as fuzzy

quality = fuzzy.FuzzyVariable(np.arange(0, 10 + 1, 1), 'quality')
quality['poor']    = fuzzy.trimf(quality.universe, [0,  0,  5])
quality['average'] = fuzzy.trimf(quality.universe, [0,  5, 10])
quality['good']    = fuzzy.trimf(quality.universe, [5, 10, 10])

service = fuzzy.FuzzyVariable(np.arange(0, 10 + 1, 1), 'service')
quality['poor']    = fuzzy.trimf(quality.universe, [0,  0,  5])
quality['average'] = fuzzy.trimf(quality.universe, [0,  5, 10])
quality['good']    = fuzzy.trimf(quality.universe, [5, 10, 10])

tip = fuzzy.FuzzyVariable(np.arange(0, 25 + 1, 1), 'tip')
tip['low']    = fuzzy.trimf(tip.universe, [0,   0, 13])
tip['medium'] = fuzzy.trimf(tip.universe, [0,  13, 25])
tip['high']   = fuzzy.trimf(tip.universe, [13, 25, 25])

rule1 = fuzz.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = fuzz.Rule(service['average'], tip['medium'])
rule3 = fuzz.Rule(service['good'] | quality['good'], tip['high'])

rules = [
  fuzzy.Rule([quality['poor'], service['poor']], tip['low']),
  fuzzy.Rule([service['average']], tip['medium']),
  fuzzy.Rule([service['good'], quality['good']], tip['high'])
]

system = fuzzy.FuzzySystem(rules)
system.input = { 'quality': 6.5, 'service': 9.8 }
system.produce()
print(system.output)
```

## Requirements
fuzzy-lite depends from NumPy package
