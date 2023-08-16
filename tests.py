import pandas as pd

def check_double_pair(final_pairs, double_paired):
    '''Check that only one person has been double paired and they are the person defined from the beginning'''
    
    final_pairs = pd.DataFrame(final_pairs, columns = ['EID 1', 'EID 2'])  # dataframe format for the rest of the script

    pairs_per_person = pd.Series(final_pairs['EID 1'].tolist() + final_pairs['EID 2'].tolist()).value_counts()
    multi_paired = pairs_per_person[pairs_per_person>1].index.tolist()

    if len(multi_paired) > 1:
        raise Exception('More than one person is double paired: '+ str(multi_paired))

    elif len(multi_paired) == 1:
        if multi_paired[0] != double_paired:
            raise Exception('The wrong person was double paired: '+  str(multi_paired[0]))
            
    return pairs_per_person, final_pairs

def check_all_paired(people_list, pairs_per_person, odd_number):
    '''Check that everyone in the people_list has been paired'''
    
    list_to_check = people_list[:-1] if odd_number else people_list
    left_out = set(list_to_check) - set(pairs_per_person.index.tolist()) 
    if len(left_out) > 0 :
        raise Exception('Not everyone has been paired. The following people have been left out: '+str(left_out))
