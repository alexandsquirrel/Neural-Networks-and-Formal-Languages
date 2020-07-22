from language import Language
import re, exrex
import random

class RegularLanguage(Language):

    def __init__(self, name, chars, max_length, regex: str, neg_regex: str=None):
        super().__init__(name, chars, max_length)
        self._regex = regex
        self._pattern = re.compile(self._regex)
        self._neg_regex = neg_regex
        if neg_regex is not None:
            self._neg_pattern = re.compile(self._neg_regex)
    
    def verify(self, string):
        return self._pattern.fullmatch(string) is not None

    def generate_example(self, truth_value):
        if truth_value == True:
            return self._generate_match()
        else:
            return self._generate_no_match()

    def _generate_match(self) -> str:
        return exrex.getone(self._regex, self._max_length)
    
    # rejection sampling
    def _generate_no_match(self) -> str:
        if self._neg_regex != None:
            # if the user has supplied with the complement regex, then we can generate deterministically!
            return exrex.getone(self._neg_regex, self._max_length)
        else:
            return self._generate_negative_example_generic()