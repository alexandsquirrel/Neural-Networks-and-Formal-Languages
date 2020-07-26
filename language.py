import numpy as np
import random

class Language:

    '''
      Parameters:
      name: the name of the language.
      chars: legal characters.
    '''
    def __init__(self, name: str, chars: list, max_length: int):
        self._name = name
        self._chars = chars
        self._max_length = max_length
    
    '''
      Same as verify() below. Just another convenient way to write.
    '''
    def __call__(self, string: str) -> bool:
        return self.verify(string)

    '''
      Verifies whether a string is in this language.
    '''
    def verify(self, string: str) -> bool:
        pass
    
    '''
      Generates a random true/false example of this language.
    '''
    def generate_example(self, truth_value: bool) -> str:
        pass

    def _generate_negative_example_generic(self) -> str:
        # trail-and-error. could have a more sophisticated method!
        while True:
            string = random_string(self._max_length, self._chars)
            if not self.verify(string):
                return string
    
    def generate_batch_data(self, num_datapoints: int, truth_value: bool) -> set:
        '''
        dataset = set()
        num_iter = 0
        cutoff_iters = 1e6
        while len(dataset) < num_datapoints and num_iter < cutoff_iters:
            example = self.generate_example(truth_value)
            dataset.add(example)
        return dataset
        '''
        dataset = []
        while len(dataset) < num_datapoints:
            example = self.generate_example(truth_value)
            if example:
                dataset.append(example)
        return dataset


# TODO: optimize string operations.
def random_string(max_length: int, legal_chars: list) -> str:
    length = np.random.randint(1, max_length+1)
    string = ''
    for i in range(length):
        string += str(random.choice(legal_chars))
    return string


'''
1) generate 200K strings using _generate_tree
2) sample from those 200k for final dataset
'''