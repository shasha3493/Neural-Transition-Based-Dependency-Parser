#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2021-2022: Homework 3
parser_transitions.py: Algorithms for completing partial parsess.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes.
        self.sentence = sentence
        
        '''
        initialize the following fields:
             self.stack: The current stack represented as a list with the top of the stack as the
                         last element of the list.
             self.buffer: The current buffer represented as a list with the first item on the
                          buffer as the first item of the list
             self.dependencies: The list of dependencies produced so far. Represented as a list of
                     tuples where each tuple is of the form (head, dependent).
                     Order for this list doesn't matter.
        
         Note: The root token is being represented with the string "ROOT"
        '''

        self.stack = ["ROOT"]
        self.buffer = sentence.copy()
        self.dependencies = []
        


    def parse_step(self, transition):
        """A single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. Assume the provided
                                transition is a legal transition.
        """

        if transition == "S":
            self.stack.append(self.buffer.pop(0)) # adding to the stack
        elif transition == "LA":
            self.dependencies.append((self.stack[-1], self.stack.pop(-2))) # adding the dependency and removing the dependant from the stack
        else:
            self.dependencies.append((self.stack[-2], self.stack.pop(-1))) # adding the dependency and removing the dependant from the stack
            

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        # perfrom transition action one by one from the list of transition actions
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []
    
    # Initialize partial_parses as a list of PartialParse() classes with each PartialParse() initialized to each sentence
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:] # initially unfinished_parses contains everything 
    
    while unfinished_parses:
        minibatch = unfinished_parses[:batch_size] # take a batch
        transitions = model.predict(minibatch)
        
        # For every sentence in a batch, 
        for i, transition in enumerate(transitions):
            # perform transition action, update the state of stack, buffer and dependencies
            minibatch[i].parse_step(transition)
            
            # if, for a senetence, len(buffer) == 0 and len(stack)==1, the sentence has been parsed completely - remove the sentence from the list of unfinished_parses 
            if len(minibatch[i].buffer)==0 and len(minibatch[i].stack)==1:
                unfinished_parses.remove(minibatch[i])
                
    # accummulate dependencies from all the sentences           
    dependencies = [partial_parse.dependencies for partial_parse in partial_parses]
    
    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    
    '''
    name: name of the transition step
    transition: transition step symbol
    stack: current state of stack
    buf: current state of buffer
    deps: current state of depedencies
    
    ex_stack: expected state of stack after the transition step
    ex_buffer: expected state of buffer after the transition step
    ex_deps: expected state of depedencies after the transition step
    '''
    
    pp = PartialParse([])
    
    # initializes the current state of stack, buffer and dependencies to the PartialParser class
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps
    
    # tranistion step
    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    
    # matching the results with expected stack, buffer and dependenscies state
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    # compares the state of stack, buffer and dependencies after taking the transition action with the expected state of stack, buffer and dependencies 
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    
    Takes in a sample sentence. Performs transition actions one by one.
    Compares the final state of stack, buffer and dependencies
    with expected state of stack, buffer and dependencies.
    
    """
    sentence = ["parse", "this", "sentence"] # a sample sentence
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"]) # initialize PartialParser() with the sample sentence and performs transition actions one by one
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    
    # compare for correctness
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    """
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """First shifts everything onto the stack and then does exclusively right arcs if the first word of
        the sentence is "right", "left" if otherwise.
        """
        return [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """First shifts everything onto the stack and then interleaves "right" and "left".
        """
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """

    # Unidirectional arcs test
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

    # Out-of-bound test
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

    # Mixed arcs test
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
