from language import Language
import nltk
from nltk import CFG
import random

class ContextFreeLanguage(Language):

    def __init__(self, name, chars, max_length, grammar_str: str):
        super().__init__(name, chars, max_length)
        self._grammar = CFG.fromstring(grammar_str)
        self._parser = nltk.RecursiveDescentParser(self._grammar)
        # then, some initializations for random generation.
        self._prod = dict()  # mapping: non-terminal symbols --> list of productions
        for production in self._grammar.productions():
            if production.lhs() in self._prod:
                self._prod[production.lhs()].append(production.rhs())
            else:
                self._prod[production.lhs()] = [production.rhs()]
    
    def verify(self, string):
        # returns whether this string is parsable with the grammar.
        try:
            return list(self._parser.parse(string.split())) != []
        except:
            return False
    
    def generate_example(self, truth_value):
        if truth_value == True:
            return self._generate_tree(self._grammar.start())
        else:
            return self._generate_negative_example_generic()
    
    def _generate_tree(self, root) -> str:
        if isinstance(root, str):  # isterminal
            return root
        else:
            children = random.choice(self._prod[root])
            tokens = []
            for child in children:
                tokens.append(self._generate_tree(child))
            return " ".join(tokens)
