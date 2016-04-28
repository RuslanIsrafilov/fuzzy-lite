import numpy as np
from . import primitives as prim

class Term(object):
    def __init__(self, universe, xmf):
        self.xmf = xmf
        self.universe = universe
        self.variable = None

    def membership_value(self, value):
        return prim.interp_membership(self.universe, self.xmf, value)

    def varname(self):
        if self.variable == None:
            raise(Exception('Term is not binded to variable'))
        return self.variable.name

    def varuniverse(self):
        if self.variable == None:
            raise(Exception('Term is not binded to variable'))
        return self.variable.universe


class FuzzyVariable(object):
    def __init__(self, universe, name):
        self.universe = universe
        self.name = name
        self._terms = { }

    def __getitem__(self, name):
        return self._terms[name]
    
    def __setitem__(self, name, xmf):
        t = Term(self.universe, xmf)
        t.variable = self
        self._terms[name] = t


class Rule(object):
    def __init__(self, antecedents, consequent):
        self.antecedents_dict = self._init_terms_dict(antecedents)
        self.consequent = consequent

    def antecedent_term_by_var_name(self, name):
        return self.antecedents_dict[name]

    def _init_terms_dict(self, terms):
        kv = { }
        for t in terms:
            kv[t.varname()] = t
        return kv


class OperatorFactory(object):
    def __init__(self, opp):
        if hasattr(opp, '__call__'):
            self._opp = opp
        elif isinstance(opp, str):
            self._opp = self._operator_from_string(opp)

    def operator(self):
        return self._opp

    def _operator_from_string(self, name):
        switch = {
            'min': prim.operator_min,
            'max': prim.operator_max,
            'prod': prim.operator_prod,
            'centroid': prim.operator_centroid
        }
        return switch[name]


class FuzzySystem(object):
    def __init__(self, rules, agg='min', act='min', acc='max', deffuz='centroid'):
        self.rules = rules
        self.input = None
        self.output = None
        self.agregation_operator = agg
        self.activation_operator = act
        self.accumulation_operator = acc
        self.deffuzification_operator = deffuz

    def produce(self):
        if self.input == None:
            raise(Exception('Input of the system is None')) 
        
        agregation_operator = OperatorFactory(self.agregation_operator).operator()
        activation_operator = OperatorFactory(self.activation_operator).operator()
        accumulation_operator = OperatorFactory(self.accumulation_operator).operator()
        deffuzification_operator = OperatorFactory(self.deffuzification_operator).operator()

        accumulationInputs = { }
        consequentDict = { }
        for rule in self.rules:
            fuzzyvalues = self._fuzzification(rule, self.input)
            agregated = self._agregation(agregation_operator, fuzzyvalues)
            activated = self._activation(activation_operator, rule.consequent, agregated)
            var = rule.consequent.varname()
            if var not in accumulationInputs:
                accumulationInputs[var] = []
            accumulationInputs[var].append(activated)
            consequentDict[var] = rule.consequent
        
        output = { }
        for key, activated in accumulationInputs.items():
            accumulated = self._accumulation(accumulation_operator, activated)
            output[key] = self._deffuzification(deffuzification_operator, 
                                                consequentDict[key], accumulated)
        self.output = output

    def _fuzzification(self, rule, valuesdict):
        result = []
        for key, value in valuesdict.items():
            t = rule.antecedent_term_by_var_name(key)
            memvalue = t.membership_value(value)
            result.append(memvalue)
        return result

    def _agregation(self, operator, valueslist):
        current = valueslist[0]
        for i in range(1, len(valueslist)):
            current = operator(current, valueslist[i])
        return current

    def _activation(self, operator, term, truthDegree):
        xmf = term.xmf
        result = np.zeros((len(xmf)))
        for i in range(0, len(result)):
            result[i] = operator(xmf[i], truthDegree)
        return result

    def _accumulation(self, operator, arrayslist):
        result = np.zeros((len(arrayslist[0])))
        for j in range(0, len(result)):
            r = arrayslist[0][j]
            for i in range(1, len(arrayslist)):
                r = operator(r, arrayslist[i][j])
            result[j] = r
        return result

    def _deffuzification(self, operator, term, accumulated):
        return operator(term.varuniverse(), accumulated)

