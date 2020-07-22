from regular_language import RegularLanguage

quant_chars = ['0', '1', '2', '3']

all_regex = '[023]*'
notall_regex = '[0-3]*1[0-3]*'

every = RegularLanguage(name="every", chars=quant_chars, max_length=20, 
                        regex=all_regex, neg_regex=notall_regex)

nall = RegularLanguage(name="not_all", chars=quant_chars, max_length=20, 
                        regex=notall_regex, neg_regex=all_regex)
