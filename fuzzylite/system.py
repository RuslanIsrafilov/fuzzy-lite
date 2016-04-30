import numpy as np
from collections import defaultdict
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

    def terms(self):
        return self._terms


class Rule(object):
    def __init__(self, antecedents, consequent):
        self.antecedents_dict = self._init_terms_dict(antecedents)
        self.consequent = consequent

    def antecedent_term_by_var_name(self, name):
        if name in self.antecedents_dict:
            return self.antecedents_dict[name]
        return None

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
            'sum': prim.operator_sum,
            't_lukas': prim.operator_lukas_t_norm,
            's_lukas': prim.operator_lukas_s_norm,
            'act_lukas': prim.operator_lukas_act,
            'centroid': prim.operator_centroid,
            'mom': prim.operator_mom
        }
        return switch[name]


class FuzzySystemStages(object):
    def __init__(self):
        self.fuzzification = []
        self.agregation    = []
        self.activation    = []
        self.accumulation  = {}

    def push_fuzzification_stage(self, fuzzyvalues, varnames):
        self.fuzzification.append(dict(zip(varnames, fuzzyvalues)))

    def push_agregation_stage(self, value):
        self.agregation.append(value)    

    def push_activation_stage(self, values):
        self.activation.append(values)

    def push_accumulation_stage(self, varname, values):
        self.accumulation[varname] = values


class FuzzySystem(object):
    def __init__(self, rules, agg='min', act='min', acc='max', deffuz='centroid'):
        self.rules = rules
        self.input = None
        self.output = None
        self.agregation_operator = agg
        self.activation_operator = act
        self.accumulation_operator = acc
        self.deffuzification_operator = deffuz
        self.stages = None

    def produce(self, save_stages=False):
        if self.input == None:
            raise(Exception('Input of the system is None')) 
        
        self.stages = FuzzySystemStages()
        agregation_operator = OperatorFactory(self.agregation_operator).operator()
        activation_operator = OperatorFactory(self.activation_operator).operator()
        accumulation_operator = OperatorFactory(self.accumulation_operator).operator()
        deffuzification_operator = OperatorFactory(self.deffuzification_operator).operator()

        accumulationInputs = defaultdict(list)
        consequentVariableUniverses = {}

        for rule in self.rules:
            # fuzzification
            fuzzyvalues, varnames = self._fuzzification(rule, self.input)
            if save_stages:
                self.stages.push_fuzzification_stage(fuzzyvalues, varnames)

            # agregation
            agregated = self._agregation(agregation_operator, fuzzyvalues)
            if save_stages:
                self.stages.push_agregation_stage(agregated)

            # activation
            activated = self._activation(activation_operator, rule.consequent, agregated)
            if save_stages:
                self.stages.push_activation_stage(activated)

            var = rule.consequent.varname()
            accumulationInputs[var].append(activated)
            consequentVariableUniverses[var] = rule.consequent.varuniverse()
        
        output = {}
        for key, activated in accumulationInputs.items():
            accumulated = self._accumulation(accumulation_operator, activated)
            if save_stages:
                self.stages.push_accumulation_stage(key, accumulated)

            varuniverse = consequentVariableUniverses[key]
            output[key] = self._deffuzification(deffuzification_operator, varuniverse, accumulated)
        self.output = output

    def _fuzzification(self, rule, valuesdict):
        result = []
        varnames = []
        for key, value in valuesdict.items():
            t = rule.antecedent_term_by_var_name(key)
            if not t == None: 
                memvalue = t.membership_value(value)
                result.append(memvalue)
                varnames.append(key)
        return (np.array(result), varnames)

    def _agregation(self, operator, valueslist):
        current = valueslist[0]
        for i in range(1, len(valueslist)):
            current = operator(current, valueslist[i])
        return current

    def _activation(self, operator, term, truthdegree):
        xmf = term.xmf
        result = np.zeros((len(xmf)))
        for i in range(0, len(result)):
            result[i] = operator(truthdegree, xmf[i])
        return result

    def _accumulation(self, operator, arrayslist):
        result = np.zeros((len(arrayslist[0])))
        for j in range(0, len(result)):
            r = arrayslist[0][j]
            for i in range(1, len(arrayslist)):
                r = operator(r, arrayslist[i][j])
            result[j] = r
        return result

    def _deffuzification(self, operator, universe, accumulated):
        return operator(universe, accumulated)

