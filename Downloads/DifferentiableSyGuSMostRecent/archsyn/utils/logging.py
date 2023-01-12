import logging
import os
import dsl
import z3

# STUFF I ADDED
from functools import reduce
from itertools import product
import operator
import numpy as np
import math

def lcm(denominators):
    return reduce(lambda a,b: a*b // math.gcd(a,b), denominators)

# this one below seems to be sufficiently general
def numericalInvariant_to_str(parameters, env, type_of_function): #parameters should come in np array format, bias can be in tensor format
    weights = [str(parameters[i]) + "*" + env[i] + " + " for i in range(len(parameters) - 1)]
    weights_str = reduce(operator.concat, weights, "")
    if type_of_function == "affine":
        return weights_str + str(parameters[-1:][0]) + " >= 0"
    elif type_of_function == "equality":
        return weights_str + str(parameters[-1:][0]) + " = 0"
    else:
        assert False # Passed in something which was not a proper function


# Looks like I had already made this one sufficiently general
def smoothed_numerical_invariant(params):
    weights = (params["weights"][0].detach()).numpy()
    biggest_weight =  abs(np.max(weights)) 
    assert biggest_weight != 0 
    bias = float(params["bias"][0].detach())/biggest_weight
    new_weights = [weight/biggest_weight for weight in weights]
    approximations = []
    N = 5 # this is how precise we want to try for the approximation
    for new_weight in new_weights:
        closest_approx_values = (-100, -100)
        closest_approx = 1000
        for i in range(-5, N+1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append(closest_approx_values)
    # Problem which the following code addresses: (5,5) = (1,1) as an approximation but the latter one will be chosen. We want the approximation denominators to be small as possible.
    new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]
    least_common_multiple = lcm([frac[1] for frac in new_approximations])
    #print("Least common multiple becomes (pls dont be negative)", least_common_multiple)
    return [least_common_multiple * approximations[i][0]/approximations[i][1] for i in range(len(weights))] + [ math.ceil(least_common_multiple * bias)] # TODO: work on locating better bias term.

# This one is also sufficiently general
def print_program2(program, env, smoothed = False):
    if program.name == "affine" or program.name == "equality":
        if smoothed:
            weights = smoothed_numerical_invariant(program.parameters)
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
        else:
            weights = list((program.parameters["weights"][0].detach()).numpy())
            bias = float(program.parameters["bias"][0].detach())
            weights.append(bias) #converting to proper form
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
    else:
        print("(" + program.name)
        for submodule, function in program.submodules.items():
            print_program2(function, env, smoothed)
        print(" )")

# This one I've adjusted to be more general
def invariant_from_program(program, env):
    # assume vars are x, y
    if program.name == "affine":
        weights = smoothed_numerical_invariant(program.parameters)
        z3_ineq = 0
        # TODO: make it more general, look at line 402 from cln_training.py
        #z3_ineq = weights[0] * z3.Real(env[0]) + weights[1] * z3.Real(env[1]) + weights[2] * 1 >= 0.0 
        z3_ineq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] >= 0.0
        return z3_ineq
    elif program.name == "equality":
        print("The parameters are ", program.parameters)
        weights = smoothed_numerical_invariant(program.parameters)
        print(" And the weights are ", weights)
        z3_eq = 0
        # TODO: make it more general according to line 402 from cln_training.py
        #z3_eq = weights[0] * z3.Real(env[0]) + weights[1] * z3.Real(env[1]) + weights[2] * 1 == 0.0 # double equals for defining the equality
        z3_eq = sum(weight*z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] == 0.0
        return z3_eq
    elif program.name == "and":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.And(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    elif program.name == "or":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.Or(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    # let's see if it actually works like this  
def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

def log_and_print(line):
    print(line)
    logging.info(line)

def print_program(program, ignore_constants=True):
    if not isinstance(program, dsl.LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))
