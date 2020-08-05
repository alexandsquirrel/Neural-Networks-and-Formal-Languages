from regular_language import RegularLanguage

quant_chars = ['0', '1', '2', '3']
quant_mapping = {'0':0, '1':1, '2':2, '3':3}

all_regex = '[023]*'
notall_regex = '[0-3]*1[0-3]*'

every = RegularLanguage(name="every", chars=quant_chars, max_length=20, 
                        regex=all_regex, neg_regex=notall_regex)

nall = RegularLanguage(name="not_all", chars=quant_chars, max_length=20, 
                        regex=notall_regex, neg_regex=all_regex)

exactly_three = RegularLanguage(name="exactly_three", chars=quant_chars, max_length=20,
                                regex='[1-3]*0[1-3]*0[1-3]*0[1-3]*')

def exactly_n(n):
    name = 'exactly_' + str(n)
    regex = '[1-3]*' + n * '0[1-3]*'
    return RegularLanguage(name=name, chars=quant_chars, max_length=20, regex=regex)

exactly_zero = exactly_n(0)
exactly_one = exactly_n(1)
exactly_two = exactly_n(2)
exactly_four = exactly_n(4)
exactly_five = exactly_n(5)
exactly_six = exactly_n(6)
exactly_eight = exactly_n(8)
exactly_twenty = exactly_n(20)