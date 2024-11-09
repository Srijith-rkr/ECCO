def get_execution_feedback(accept, pass_tests, errors, run_times, memory):
    if len(errors) == 0: # Passed all test cases 
        feedback = f'Your solution was functionally CORRECT across ALL test cases!\n'
    elif len(pass_tests) > 0: # Passed at least one test case
        feedback = f'Your solution was INCORRECT and passed {len(pass_tests)} test cases.\n'
    else: # Passed no test cases
        feedback = f'Your solution was FULLY INCORRECT and passed 0 test cases. This could either be a flaw in logic or a syntax error. Please see error logs.\n'
            
    if len(pass_tests) > 0:
        feedback += '\nHere are the run time and memory that your code utilized for each test case\n'
        for test_id in run_times.keys(): # Has run time for passed as well as failed
            time, mem = run_times[test_id], memory[test_id]
            pass_or_fail_str = 'PASSED' if int(test_id) in pass_tests else 'FAILED'
            feedback += f'-- Stats for test case {test_id} --\n'
            feedback += f'Correct: {pass_or_fail_str}\nRun time: {time} s\nMemory: {mem} KB\n'

    if len(errors) > 0:
        feedback += 'Here are the error logs for the failed test cases\n'
        for test_id in errors.keys():
            # TODO: Add the input for failed test case
            feedback += f'-- Error log for failed test case {test_id} --\n'
            if errors[test_id]:
                feedback += errors[test_id] + '\n' # Wrong Answer: {} Expected Answer: {}
            else: # TODO: Figure out why the error is None?
                feedback += '\n'

    return feedback
            


    