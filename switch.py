"""
A pythonic `switch` implementation.
ref: http://code.activestate.com/recipes/410692-readable-switch-construction-without-lambdas-or-di/
"""
class switch():
    def __init__(self, value, equal_rule=None, check_rule=None):
        self.value = value
        self.equal_rule = equal_rule
        self.check_rule = check_rule
        
    def __eq__(self, case_value):
        if self.equal_rule: 
            return self.equal_rule(self.value, case_value)
        else: 
            return self.value == case_value
    
    def __iter__(self):
        yield self.match
        raise StopIteration
        
    """ TODO: multiple conditions to check """
    def check(match_func):
        """ or: without `self` argument. """
        def wrapper(self, *args, **kws):
            if len(args) == 1 and self.check_rule:
                case_value = args[0]
                assert callable(self.check_rule) and self.check_rule(case_value)
#                 print('assertion completed.')
            return match_func(self, *args, **kws)
        
        """ another equivalent writing form of `wrapper`. """
#         def wrapper(*args, **kws):
#             if args:
#                 assert len(args) == 2
#                 self, case_value = args
#                 assert self.check_rule(case_value)
#                 print('assertion completed 2.')
#             return match_func(*args, **kws)
        
        return wrapper
    
    @check
    def match(self, case_value):
        return True if case_value == 'default' else self.__eq__(case_value) 
    
    
if __name__ == '__main__':    
    v = 'ten'
    for case in switch(v):
        if case('one'):
            print(1)
            break
        if case('two'):
            print(2)
            break
        if case('ten'):
            print(10)
            break
        if case('eleven'):
            print(11)
            break
        if case('default'):
            pass
