from __future__ import print_function
import math
from functools import reduce
import os
import sys
import numpy as np
import torch
import pandas as pd
import random
import torch.optim as optim
from itertools import chain
from tqdm import tqdm
import z3
import ast

import argparse
import pickle
import time
import heapq

# import program_learning
from dataset import Dataset
from algorithms import NAS
from program_graph import ProgramGraph
from utils.data_loader import CustomLoader, IOExampleLoader
from utils.evaluation import label_correctness, value_correctness
from utils.logging import init_logging, log_and_print, print_program, print_program2, lcm, numericalInvariant_to_str, smoothed_numerical_invariant
from utils.logging import invariant_from_program
from utils.loss import SoftF1LossWithLogits
from dsl_inv import DSL_DICT, CUSTOM_EDGE_COSTS


from dsl.library_functions import LibraryFunction

from metal.common.utils import CEHolder
from metal.common.constants import CE_KEYS
from metal.parser.sygus_parser import SyExp

# import cln2inv stuff

from cln2inv_stuff.invariant_checking import InvariantChecker

import pdb



def parse_args():
    parser = argparse.ArgumentParser()
    # cmd_args for experiment setup
    # parser.add_argument('-t', '--trial', type=int, required=True,
    #                     help="trial ID")
    # parser.add_argument('--exp_name', type=str, required=True,
    #                     help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # cmd_args for data
    # parser.add_argument('--train_data', type=str, required=True,
    #                     help="path to train data")
    # parser.add_argument('--test_data', type=str, required=True,
    #                     help="path to test data")
    # parser.add_argument('--valid_data', type=str, required=False, default=None,
    #                     help="path to val data. if this is not provided, we sample val from train.")
    # parser.add_argument('--train_labels', type=str, required=True,
    #                     help="path to train labels")
    # parser.add_argument('--test_labels', type=str, required=True,
    #                     help="path to test labels")
    # parser.add_argument('--valid_labels', type=str, required=False, default=None,
    #                     help="path to val labels. if this is not provided, we sample val from train.")
    # parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
    #                     help="input type of data")
    # parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
    #                     help="output type of data")
    # parser.add_argument('--input_size', type=int, required=True,
    #                     help="dimenion of features of each frame")
    # parser.add_argument('--output_size', type=int, required=True,
    #                     help="dimension of output of each frame (usually equal to num_labels")
    # parser.add_argument('--num_labels', type=int, required=True,
    #                     help="number of class labels")

    # cmd_args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.0,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")
    parser.add_argument('--sem', type=str, required=False, choices=["arith","minmax"], default="arith",
                        help="discrete semantics approximation")

    # cmd_args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=1.0, #TODO: changed this to 1 no need for validation here
                        help="split training set for validation."+\
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('-search_lr', '--search_learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    # parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits", "softf1"],
    #                     help="loss function for training")
    parser.add_argument('--f1double', action='store_true', required=False, default=False,
                        help='whether use double for soft f1 loss')
    parser.add_argument('--class_weights', type=str, required=False, default = None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--topN_select', type=int, required=False, default=2,
                        help="number of candidates remain in each search")
    parser.add_argument('--resume_graph', type=str, required=False, default=None,
                        help="resume graph from certain path if necessary")
    parser.add_argument('--sec_order', action='store_true', required=False, default=False,
                        help='whether use second order for architecture search')
    parser.add_argument('--spec_design', action='store_true', required=False, default=False,
                        help='if specific, train process is defined manually')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help="manual seed")
    parser.add_argument('--finetune_epoch', type=int, required=False, default=12, #CHANGED --finetune_epoch to --finetune_epochs
                        help='Epoch for finetuning the result graph.')
    parser.add_argument('--finetune_lr', type=float, required=False, default=0.01,
                        help='Epoch for finetuning the result graph.')

    # cmd_args for algorithms
    # parser.add_argument('--algorithm', type=str, required=True,
    #                     choices=["mc-sampling", "mcts", "enumeration", "genetic", "astar-near", "iddfs-near", "rnn", 'nas'],
    #                     help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR"+
                        " (>1.0 prunes aggressively in this case)")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")
    parser.add_argument('--cell_depth', type=int, required=False, default=3,
                        help="max depth for each cell for nas algorithm")


    parser.add_argument('-data_root', default=None, help='root of dataset')
    parser.add_argument('-file_list', default=None, help='list of programs')
    parser.add_argument('-single_sample', default=None, type=str, help='tune single program')
    parser.add_argument('-use_interpolation', default=0, type=int, help='whether use interpolation')
    parser.add_argument('-top_left', type=bool, default=False, help="set to true to use top-left partition")
    parser.add_argument('-GM', type=bool, default=False, help="set to true to use Gradient Matching")

    parser.add_argument('--problem_num', type = int, default = 1, help = "The problem number from the cln2inv benchmarks")
    return parser.parse_args()

#Print program in Sygus format. (Not used anywhere currently, don't need SyExp format to evaluate the program anymore)
def convert_to_sygus(program):
    if not isinstance(program, LibraryFunction):
        return SyExp(program.name, [])
    else:
        if program.has_params:
            Q = SyExp(program.name, [program.parameters])
            return SyExp(program.name, [program.parameters])
        else:
            collected_names = []
            for submodule, functionclass in program.submodules.items():
                collected_names.append(convert_to_sygus(functionclass))
            return SyExp(program.name, collected_names)


def reward_w_interpolation(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # check if it passes
    status, key, ce = lambda_new_ce()
    if status > 0:
        return 1.0

    # interpolate ce and add neary ones into the buffer
    holder.interpolate_ce(ce)

    #harmonic mean
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0

    return -2.0 + hm_t


def reward_1(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # print("\n\nsample_index:", sample_index)
    # holder.show_stats()
    # ct = 0
    # s = 0
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        # print("key:", key,  "score: ", score, "ce_per_key:", holder.ce_per_key)
        # if key in holder.ce_per_key:
        #     ct += len(holder.ce_per_key[key].ce_list)
        #     s += 0.99
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0
    # print("ct=",ct, "t=", t, "s=",s)

    return -2.0 + hm_t

variable_dictionary = {
    99: ["n", "x", "y"],
    98: ["i", "j", "x", "y"],
    97: ["i", "j", "x", "y"],
    96: ["i", "j", "x", "y"],
    95: ["i", "j", "x", "y"],
    94: ["i", "j", "k", "n"],
    93: ["i", "n", "x", "y"],
    92: ["x", "y", "z1", "z2", "z3"],
    91: ["x", "y"],
    90:  ["lock", "x", "y", "v1", "v2", "v3"],
    9: ["x", "y"],
    89: ["lock", "x", "y", "v1", "v2", "v3"],
    88: ["lock", "x", "y"],
    87: ["lock", "x", "y"],
    86: ["x", "y", "z1", "z2", "z3"],
    85: ["x", "y", "z1", "z2", "z3"],
    84: ["x", "y"],
    83: ["x", "y"],
    82:["i", "x", "y", "z1", "z2", "z3"],
    81:["i", "x", "y", "z1", "z2", "z3"],
    80: ["i", "x", "y", "z1", "z2", "z3"],
    8: ["x", "y"],
    79: ["i", "x", "y"],
    78: ["i", "x", "y"],
    77: ["i", "x", "y"],
    76: ["c", "y", "z", "x1", "x2", "x3"],
    75: ["c", "y", "z", "x1", "x2", "x3"],
    74: ["c", "y", "z", "x1", "x2", "x3"],
    73: ["c", "y", "z"],
    72: ["c", "y", "z"],
    71: ["c", "y", "z"],
    70: ["x", "y"],
    7:  ["x", "y"],
    69: ["n", "v1", "v2", "v3", "x", "y"],
    68: ["n", "y", "x"],
    67: ["n", "y", "x"],
    66: ["x", "y"],
    65: ["x", "y"],
    64: ["x", "y"],
    63: ["x", "y"],
    62: ["c", "n", "v1", "v2", "v3"],
    61: ["c", "n", "v1", "v2", "v3"],
    60: ["c", "n", "v1", "v2", "v3"],
    6: ["v1", "v2", "v3", "x", "size", "y", "z"], # like in other cases, some of these variables are predeclared to be a specific value. It is worth checking to see if we can reduce the variable load by looking at the interaction with z3
    59:["c", "n", "v1", "v2", "v3"],
    58:["c", "n", "v1", "v2", "v3"],
    57:["c", "n", "v1", "v2", "v3"],
    56:["c", "n", "v1", "v2", "v3"],
    55:["c", "n", "v1", "v2", "v3"],
    54:["c", "n", "v1", "v2", "v3"],
    53: ["c", "n", "v1", "v2", "v3"],
    52: ["c"],
    51: ["c"],
    50: ["c"],
    5: ["x", "size", "y", "z"],
    49:["c", "n"],
    48:["c", "n"],
    47:["c", "n"],
    46:["c", "n"],
    45:["c", "n"],
    44:["c", "n"],
    43:["c", "n"],
    42:["c", "n"],
    41:["c", "n"],
    40: ["c", "n"],
    4: ["x", "y", "z"],
    39:["n", "c"],
    38:["n", "c"],
    37:["c"],
    36:["c"],
    35:["c"],
    34:["n", "v1", "v2", "v3", "x"],
    33:["n", "v1", "v2", "v3", "x"],
    32:["n", "v1", "v2", "v3", "x"],
    31: ["n", "v1", "v2", "v3", "x"],
    30: ["x"],
    3: ["x", "y", "z"],
    29:["n", "x"],
    28:["n", "x"],
    27:["n", "x"],
    26: ["n", "x"],
    25: ["x"],
    24:["i", "j"],
    23: ["i", "j"],
    22: ["x", "m", "n", "z1", "z2", "z3"],
    21: ["x", "m", "n", "z1", "z2", "z3"],
    20:  ["x", "m", "n", "z1", "z2", "z3"],
    2: ["x", "y"],
    19: ["x", "m", "n", "z1", "z2", "z3"],
    18:["x", "m", "n"],
    17:["x", "m", "n"],
    16: ["x", "m", "n"],
    15: ["x", "m", "n"],
    14:["x", "y", "z1", "z2", "z3"],
    133: ["n", "x"],
    132: ["i", "j", "c", "t"],
    131: ["d1", "d2", "d3", "x1", "x2", "x3"],
    130: ["d1", "d2", "d3", "x1", "x2", "x3"],
    13: ["x", "y", "z1", "z2", "z3"],
    129: ["x", "y", "z1", "z2", "z3"],
    128: ["x", "y"],
    127: ["i", "j", "x", "y", "z1", "z2", "z3"],
    126: ["i", "j", "x", "y", "z1", "z2", "z3"],
    125: ["i", "j", "x", "y"],
    124: ["i", "j", "x", "y"],
    123: ["i", "size", "sn", "v1", "v2", "v3"],
    122: ["i", "size", "sn", "v1", "v2", "v3"],
    121: ["i", "sn"],
    120: ["i", "sn"],
    12: ["x", "y", "z1", "z2", "z3"],
    119: ["i", "size", "sn"],
    118: ["i", "size", "sn"],
    117: ["sn", "v1", "v2", "v3", "x"],
    116: ["sn", "v1", "v2", "v3", "x"],
    115: ["sn", "x"],
    114: ["sn", "x"],
    113: ["i", "n", "sn", "v1", "v2", "v3"],
    112: ["i", "n", "sn", "v1", "v2", "v3"],
    111: ["i", "n", "sn"],
    110: ["i", "n", "sn"],
    1: ["x", "y"],
    10: ["x", "y"],
    100: ["n", "x", "y"],
    101: ["n", "x"],
    102: ["n", "x"],
    103: ["x"],
    104: ["n", "v1", "v2", "v3", "x"],
    105: ["n", "v1", "v2", "v3", "x"],
    106: ["a", "m", "j", "k"],
    107: ["a", "m", "j", "k"],
    108: ["a","c", "m", "j", "k"],
    109: ["a","c", "m", "j", "k"],
    11: ["x", "y", "z1", "z2", "z3"]

}

def evaluate(algorithm, graph, train_loader, train_config, device):
    validset = train_loader.validset
    with torch.no_grad():
        metric = algorithm.eval_graph(graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
    return metric

if __name__ == '__main__':
    cmd_args = parse_args()
    # added stuff for cln2inv
    problem_num = cmd_args.problem_num
    fname = str(problem_num) + '.c'
    csvname = str(problem_num) + '.csv'
    src_path = 'benchmarks-cln2inv/code2inv/c/'
    check_path = 'benchmarks-cln2inv/code2inv/smt2'
    trace_path = 'benchmarks-cln2inv/code2inv/csv/'
    #with open(src_path + fname, 'r') as f:
    #    code = f.read()
    #    tree = ast.parse(code)
    #    variables = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    #print(variables)
    #def load_trace(csv_name):
    #    df = pd.read_csv(csv_name)
    #    df_data = df.drop(columns=['init', 'final'])
    #    df_data['1'] = 1
    #    return df_data
    #df = load_trace(trace_path + csvname)
    #var_names = list(df.columns) # check what format this comes in
    #print(var_names)
    #assert False

    env = variable_dictionary[problem_num] # replace with some method of getting the program variables (maybe use something from metal)
    #assert len(env) == 2 SHould be generalized for arbitrary variable count

    #env = ("sn", "x") # replace with some method of getting the program variables (this has been done, in a terrible way)
    invariantChecker = InvariantChecker(fname, check_path)
    # manual seed all random for debug
    log_and_print('random seed {}'.format(cmd_args.random_seed))
    torch.random.manual_seed(cmd_args.random_seed)
    #torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.random_seed)
    random.seed(cmd_args.random_seed)
    wait = False
    ##dataset = Dataset(cmd_args)
    #print(dataset)
    ##specsample_ls = dataset.sample_minibatch(1, replacement=True)
    #print (f'spec: {specsample_ls[0].spectree.spec}')
    #print (f'grammar: {specsample_ls[0].spectree.grammar}')
    #print (f'vars: {specsample_ls[0].spectree.vars}')

    #print (f'node_seq: {specsample_ls[0].spectree.node_seq}')
    #print (f'node_type_seq: {specsample_ls[0].spectree.node_type_seq}')
    #print (f'numOf_nodes: {specsample_ls[0].spectree.numOf_nodes}')
    #print (f'nodename2ind: {specsample_ls[0].spectree.nodename2ind}')

    #print (f'all_tests: {specsample_ls[0].spectree.all_tests}')

    # Context free grammar for synthesis
    ##cfg = specsample_ls[0].spectree.grammar
    ##root_symbol = cfg.start
    #print (f'Grammar root: {root_symbol}')
    # IO examples holder.
    ##g = specsample_ls[0]
    ##holder = CEHolder(g)
    ##print("Spec sample is ", specsample_ls)
    ##print("Holder is ", holder)

    # Variables of the to-synthesize program.
    ##vars = specsample_ls[0].spectree.vars
    ##var_ids = {}
    ##for id, var in enumerate(vars):
    ##    var_ids[var] = id
    ##print("var_ids are", var_ids)
    full_exp_name = 'Test'
    save_path = os.path.join(cmd_args.save_dir, full_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init log
    init_logging(save_path)
    log_and_print("Starting experiment {}\n".format(full_exp_name))

    #///////////////////////////////
    #///////////////////////////////
    #///////////////////////////////

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    Testing = True 
    def lossfxn(out, labels):
        augmented_out = [(torch.ones(1) - out[l][1]) if labels[l] == 1 else (out[l][0] if labels[l] == 2 else (torch.ones(1) - (torch.ones(1) - out[l][0])*out[l][1])) for l in range(len(out))] 
        return torch.mean(torch.hstack(augmented_out))  #torch.mean(torch.log(torch.ones(len(augmented_out)) - torch.hstack(augmented_out))) 
        #return torch.nn.MSELoss()(out, labels)


    if device != 'cpu':
        lossfxn = lossfxn.cuda()

    max_depth = 3 #This is a tweakable parameter. The higher the depth, the longer the search.
    input_size = len(env) 
    output_size = 1
    num_labels = 1
    input_type = output_type = "atom"

    train_config = {
        'arch_lr' : cmd_args.search_learning_rate,
        'model_lr' : cmd_args.search_learning_rate,
        'train_lr' : cmd_args.learning_rate,
        'search_epoches' : cmd_args.neural_epochs,
        'finetune_epoches' : cmd_args.symbolic_epochs,
        'arch_optim' : optim.Adam,
        'model_optim' : optim.Adam,
        'lossfxn' : lossfxn,
        'evalfxn' : value_correctness,
        'num_labels' : num_labels,
        'save_path' : save_path,
        'topN' : cmd_args.topN_select,
        'arch_weight_decay' : 0,
        'model_weight_decay' : 0,
        'penalty' : cmd_args.penalty,
        'secorder' : cmd_args.sec_order,
        'specific' : [#[None, 2, 0.01, 5], [4, 2, 0.01, 5], [3, 2, 0.01, 5], [2, 2, 0.01, 5], \
                [None, max_depth, 0.01, 30]]#, ["astar", max_depth, 0.1, 5]]#, [4, 4, 0.01, 500], [3, 4, 0.01, 500], [2, 4, 0.01, 500]]#, ["astar", 4, 0.01, cmd_args.neural_epochs]] todo: here is where the epochs are defined for the main training session
    }

    # Initialize program graph
    if cmd_args.resume_graph is None:
        #program_graph = ProgramGraph(None, input_type, output_type, input_size, output_size,
        #                            cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
        #                            device, ite_beta=cmd_args.ite_beta, cfg=cfg, var_ids=var_ids, root_symbol=root_symbol)
        program_graph = ProgramGraph(DSL_DICT, input_type, output_type, input_size, output_size,
                                    cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
                                    device, ite_beta = cmd_args.ite_beta, cfg = None, var_ids = None, root_symbol = None)
        start_depth = 0
    else:
        assert os.path.isfile(cmd_args.resume_graph)
        program_graph = pickle.load(open(cmd_args.resume_graph, "rb"))
        program_graph.max_depth = max_depth
        start_depth = program_graph.get_current_depth()
        # start_depth = 3

    # Initialize algorithm
    algorithm = NAS(frontier_capacity=cmd_args.frontier_capacity)

    #///////////////////////////////
    #///////////////////////////////
    #///////////////////////////////

    # testing functions here:
    # train_data must be sorted by var_ids
    if False: # just so I can close all these, may need them for testing later
        pass
        #train_data, train_labels = [[[-1000., 1000.], [1,0]]], [[1]]
        #expr = (lambda x,y : ((x - 4*y - 3> 0) and (2*x - y - 3 > 0)) or (((-1)*x + y - 1 > 0) and (x + 2*y + 2 > 0)))
        #expr = (lambda x, y : ((-2)*x - 2*y + 4 > 0) and ((-4)*x + 3*y +1> 0)) #((-4)*x - 2*y + 4 > 0) and ((-4)*x + 3*y +1> 0):
        #expr = (lambda x, y : ((((x - 4*y - 4> 0) and (2*x - y - 4 > 0)) or (((-1)*x + y - 1 > 0) and (x + 2*y + 2 > 0)))) and (5*x + 2*y + 3 > 0))
        #expr = (lambda x,y: (2*x + (-3)*y + 1 > 0))
        #expr = (lambda x,y: (x > 0) and (y > 0))
        #K = 15
        #if False: # This was intended for testing
        #    for x in range(-1*K,K):#TODO: changed from 25
        #        for y in range(-1*K,K):
        #            if (expr(x, y)): 
        #                train_data.append([[-1000.,-1000.], [float(x),float(y)]])
        #                train_labels.append([1.])
        #            else:
        #                train_data.append([  [float(x),float(y)] ,  [-1000., -1000.] ])
        #                train_labels.append([2.])
                    # in future, for implication examples, we also add implciation example, assuming it is in P -> Q form, label 3 for 11 as there is something on both left and right.}}
        #arr1 = [[[-1000,-1000], [x, x]] for x in range(19)]
        #arr2 = [[1.] for x in range(19)]
        #arr1 = [[[-1000, -1000], [1, 1]]]
        #arr2 = [[1.]]
        #arr5 = [[[x,x], [x+1, x+1]] for x in range(19)]
        #arr6 = [[3.] for x in range(19)]
        #arr3 = [[[i+1,i], [-1000, -1000]] for i in range(19)]
        #arr4 = [[2.] for i in range(19)]
        #arr7 = [[[i-1,i], [-1000, -1000]] for i in range(19)]
        #arr8 = [[2.] for i in range(19)]
        #arr5 = [[[i,j], [-1000, -1000]] for i in range(19) for j in range(i+1, 19)]
        #arr6 = [[2.] for i in range(19) for j in range(i+1, 19)]
        #arr3 = [[[x, y], [x -1,  y + 1]] for x in range(0, 10) for y in range(1, 10)]
        #arr4 = [[3.] for x in range(0,10) for y in range(1,10)]
        #arr5 = [[[-1000, -1000], [10, 0]] for x in range(1)] + [[[-1000, -1000],[0, 20]] for x in range(1)]
        #arr6 = [[1.] for x in range(2)]
        #train_data, train_labels = arr1+arr3+arr5, arr2 + arr4 + arr6
        train_data, train_labels = [[[-1, 0], [-1000,-1000]]], [[2.]]
    train_data = []
    train_labels = []
    # assuming env has already been found
    non_loop_invariant = 1.0 * z3.Real(env[0]) >= 0.0 # a non-loop invariant (presumably). this is to get the first data point, note cln2inv warm starts with data from a spreadsheet, I think
    result = invariantChecker.check_cln([non_loop_invariant], env)
    if result[0]:
        assert False # the bad invaraint worked???
    else:
        if result[2] == "loop":
            train_data.append(result[3])
            train_labels.append([3.])
        else:
            if result[3][0] >= 0: # the 
                # then false
                train_data.append([[float(result_element) for result_element in result[3]], [-1000. for i in env]])
                train_labels.append([2.])
            else: # then the datapoint needs to be true, actually
                train_data.append([[-1000. for i in env], [float(result_element) for result_element in result[3]]])
                train_labels.append([1.])
    verification_iter = 0
    while verification_iter < 500:
        #print("TRAINING DATA IS !!!!", train_data)
        if wait:
            time.sleep(2)
        # Initialize program graph
        if cmd_args.resume_graph is None:
            program_graph = ProgramGraph(DSL_DICT, input_type, output_type, input_size, output_size,
                                        cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
                                        device, ite_beta = cmd_args.ite_beta, cfg = None, var_ids = None, root_symbol = None)
            start_depth = 0
        else:
            assert os.path.isfile(cmd_args.resume_graph)
            program_graph = pickle.load(open(cmd_args.resume_graph, "rb"))
            program_graph.max_depth = max_depth
            start_depth = program_graph.get_current_depth()
            # start_depth = 3

        # Initialize algorithm
        algorithm = NAS(frontier_capacity=cmd_args.frontier_capacity)
        iteri = 0
        partition_num = 0
        all_graphs = [[0, program_graph]]
        while(True):
            _, program_graph = heapq.heappop(all_graphs)
            search_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
            batched_trainset = search_loader.get_batch_trainset()
            batched_validset = search_loader.get_batch_validset()

            #log_and_print('data for architecture search')
            #log_and_print('batch num of train: {}'.format(len(batched_trainset)))
            #log_and_print('batch num of valid: {}'.format(len(batched_validset)))

            # for program train
            train_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
            batched_prog_trainset = train_loader.get_batch_trainset()
            prog_validset = train_loader.get_batch_validset()
            testset = train_loader.testset

            log_and_print('data for architecture search')
            log_and_print('batch num of train: {}'.format(len(batched_prog_trainset)))
            log_and_print('batch num of valid: {}'.format(len(prog_validset)))
            log_and_print('total num of test: {}'.format(len(testset)))

            # Run program learning algorithm
            best_graph, program_graph = algorithm.run_specific(program_graph,\
                                        search_loader, train_loader,
                                        train_config, device, start_depth=start_depth, warmup=False, cegis=(iteri>0), sem=cmd_args.sem)


            best_program = best_graph.extract_program()
            program_graph.show_graph()
            # print program
            log_and_print("Best Program Found:")
            program_str = print_program(best_program)
            log_and_print(program_str)

            # Save best program
            pickle.dump(best_graph, open(os.path.join(save_path, "graph.p"), "wb"))
            # Finetune
            if (not Testing) and cmd_args.finetune_epoch is not None:
                train_config = {
                    'train_lr' : cmd_args.finetune_lr,
                    'search_epoches' : cmd_args.neural_epochs,
                    'finetune_epoches' : cmd_args.finetune_epoch, # changed from cmd_args.finetune_epochs as this could not be found
                    'model_optim' : optim.Adam,
                    'lossfxn' : lossfxn,
                    'evalfxn' : label_correctness,
                    'num_labels' : num_labels,
                    'save_path' : save_path,
                    'topN' : cmd_args.topN_select,
                    'arch_weight_decay' : 0,
                    'model_weight_decay' : 0,
                    'secorder' : cmd_args.sec_order
                }
                log_and_print('Finetune')
                # start time
                start = time.time()
                best_graph = algorithm.train_graph_model(best_graph, train_loader, train_config, device, lr_decay=1.0)
                # calculate time
                total_spend = time.time() - start
                log_and_print('finetune time spend: {} \n'.format(total_spend))
                # store
                pickle.dump(best_graph, open(os.path.join(save_path, "finetune_graph.p"), "wb"))

                # debug
                testset = train_loader.testset
                best_program = best_graph.extract_program()

            best_program = best_program.submodules["program"]
            if True:
                print_program2(best_program, env)
                print(" and the smoothed version of the program is ")
                print_program2(best_program, env, True)
                def lambda_program_generator(program, Smoothed = False):
                    if program.name == "affine":
                        if Smoothed:
                            vals = smoothed_numerical_invariant(program.parameters)
                            return lambda *args: sum(val*arg for val,arg in zip(vals, args)) + vals[-1] >= 0 # I think if vals is too long, the zip will ignore it (I checked indeed it does)
                            #return lambda x, y : (vals[0]*x + vals[1]*y + vals[2] >= 0) 
                        else:
                            # do a preprocessing step to remove the gradient attachment
                            new_weights = [float(x.detach()) for x in program.parameters["weights"][0]] + [float(program.parameters["bias"][0].detach())]
                            # have confirmed this looks like it should
                            return lambda *args: sum(val*arg for val,arg in zip(new_weights, args)) + new_weights[-1] >= 0
                            #return lambda x, y: (float(program.parameters["weights"][0][0].detach())*x + float(program.parameters["weights"][0][1].detach())*y + float(program.parameters["bias"][0].detach()) >= 0)
                    elif program.name == "equality":
                        if Smoothed:
                            vals = smoothed_numerical_invariant(program.parameters)
                            return lambda *args: sum(val*arg for val,arg in zip(vals, args)) + vals[-1] >= 0
                            #return lambda x, y : (vals[0]*x + vals[1]*y + vals[2] == 0) #TODO: generalize
                        else:
                            new_weights = [float(x.detach()) for x in program.parameters["weights"][0]] + [float(program.parameters["bias"][0].detach())]
                            return lambda *args: sum(val*arg for val,arg in zip(new_weights, args)) + new_weights[-1] >= 0
                            #return lambda x, y: (float(program.parameters["weights"][0][0].detach())*x + float(program.parameters["weights"][0][1].detach())*y + float(program.parameters["bias"][0].detach()) == 0)
                    elif program.name == "and": #and
                        return lambda *args : (lambda_program_generator(list(program.submodules.items())[0][1], Smoothed)(*args) and lambda_program_generator(list(program.submodules.items())[1][1], Smoothed)(*args))
                    else: # or
                        return lambda *args : (lambda_program_generator(list(program.submodules.items())[0][1], Smoothed)(*args) or lambda_program_generator(list(program.submodules.items())[1][1], Smoothed)(*args))
                
                func = lambda_program_generator(best_program, False)
                func_smoothed = lambda_program_generator(best_program, True)
                Missed = []
                for datum in zip(train_data,train_labels): # this is for checking that the output function actually works before smoothing
                    #print(datum)
                    if datum[1][0] == 2.0: # false
                        if func(*(datum[0][0])):
                            Missed.append(list(datum[0]))
                    elif datum[1][0] == 1.0: # true
                        if not func(*(datum[0][1])):
                            Missed.append(list(datum[0]))
                    elif datum[1][0] == 3.0: #implication_example
                        if not ((not func(*(datum[0][0]))) or (func(*(datum[0][1])))):
                            Missed.append(list(datum[0])) 
                Missed_Smooth = []
                for datum in zip(train_data,train_labels):  # form: ([[x,y],[w,z]], [label])
                    #print(datum)
                    if datum[1][0] == 2.0: # false
                        if func_smoothed(*(datum[0][0])):
                        #if func_smoothed(datum[0][0][0], datum[0][0][1]):
                            Missed_Smooth.append(list(datum[0]))
                    elif datum[1][0] == 1.0:
                        if not func_smoothed(*(datum[0][1])):
                            Missed_Smooth.append(list(datum[0]))
                    elif datum[1][0] == 3.0:
                        if not ((not func_smoothed(*(datum[0][0]))) or (func_smoothed(*(datum[0][1])))):
                            Missed_Smooth.append(list(datum[0])) 
                print("Number missed w/o smooth is ", len(Missed), " with missed examples", Missed)
                print("Number missed w/ smooth is ", len(Missed_Smooth) , " with missed examples ", Missed_Smooth)
                if wait:
                    time.sleep(3) #TODO: for viewing
            #sygus_program = convert_to_sygus(best_program)
            if len(Missed_Smooth) == 0:
                log_and_print("Found a solution!!!!!:")
                print_program2(best_program, env, smoothed = True)
                break
            else:
                print("Missed smooth examples are", Missed_Smooth)
                train_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
                for pair in all_graphs:
                    pair[0] = evaluate(algorithm, pair[1], train_loader, train_config, device)
                splited_subgraph = program_graph.partition(cmd_args.top_left, cmd_args.GM)
                partition_num += 1
                if splited_subgraph is not None:
                    for subgraph in splited_subgraph:
                        all_graphs.append([evaluate(algorithm, subgraph, train_loader, train_config, device), subgraph])
                heapq.heapify(all_graphs)
            iteri += 1
            print("Length of training data is ", len(train_data), " TRAINING DATA IS !!!!", train_data)

            print("number of partitions: ", partition_num)
        #print("Beginning to check invariant")
        inv_smt = invariant_from_program(best_program, env)
        print("Invaraint smt is ", inv_smt)
        #print("Sexpred version of the invariant is ", invariant_from_program(best_program).sexpr())
        result = invariantChecker.check_cln([inv_smt], env)
        print("The result was", result)
        if wait:
            time.sleep(5)
        if result[0]:
            print(result[1]) # this is the invariant string
            assert False # have found a solution!
        elif result[2] == "loop": # we have an implication example, just add it into the training data
            train_data.append(result[3])
            train_labels.append([3.])
        else:
            # I have a sat example from pre or post, but I don't know which category it falls into, so I check its output from the generated function and add it to the opposite label
            #output = func_smoothed(result[3][0],result[3][1])
            output = func_smoothed(*result[3])
            if output: # then we actually need this datapoint to be false
                train_data.append([[float(result_element) for result_element in result[3]], [-1000. for i in env]])
                train_labels.append([2.])
            else: # then the datapoint needs to be true, actually
                train_data.append([[-1000. for i in env],[float(result_element) for result_element in result[3]]])
                train_labels.append([1.])
        verification_iter+=1

