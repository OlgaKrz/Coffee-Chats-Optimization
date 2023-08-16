from pulp import *
from datetime import datetime, timedelta
import random 
import pandas as pd
import numpy as np
import calendar

def read_roster_input(roster_file_path, roster_generic_filename):
    ''' This function reads the most updated roster file '''

    past_month = calendar.month_name[(datetime.now() - timedelta(days = 15)).month]
    roster_input_file  = roster_file_path + (roster_generic_filename%past_month)
    roster_original = pd.read_excel(roster_input_file)

    return roster_original

def determine_output_files(roster_original, roster_file_path, roster_generic_filename, 
                           final_pairs_file_path, final_pairs_generic_filename):
    ''' This function determines the file names of the output files '''
   
    target_month        = calendar.month_name[(datetime.now() + timedelta(days = 15)).month] 
    old_pair_columns    = [x for x in roster_original.columns if 'Pair' in x]  # these are the columns that save the past pair assignments 
    target_pair_incr_id = int(old_pair_columns[-1].split()[1].split('.')[0]) + 1
    
    roster_output_file       = roster_file_path      + (roster_generic_filename%target_month)
    final_pairs_output_file  = final_pairs_file_path + (final_pairs_generic_filename%(str(target_pair_incr_id),target_month)) 
    
    return roster_output_file, final_pairs_output_file

def temp_updates(roster_original, eids_remove_temp, mentees_impacted, new_mentors):
    ''' This function updates the roster dataframe with any temporary removals '''
    
    roster = roster_original[~roster_original['EID'].isin(eids_remove_temp)]
    for mentee, new_mentor in zip(mentees_impacted, new_mentors):
        roster.loc[roster['EID'] == mentee,'People Lead'] = new_mentor
        roster.loc[roster['EID'] == mentee,'Buddy'] = mentee
        
    return roster

def add_temp_columns_to_roster(roster):
    ''' This function adds columns to the roster that will come handy later in the optimization formulation '''
    
    # Add binary column to indicate if e2e-er is level 9 / 10 and in Mordor -- will be used in an optimization constraint
    roster['Mordor_9_10'] = np.where((roster['Level'] >=9) & (roster['Office Address'] == 'Mordor'), 1, 0)

    # Explode the close contacts column (currently the column includes, in each row, names separated by commas but we want to split the series of names, in each row, into separate columns)
    n_new_columns = max(roster['Close Contacts'].replace(np.nan,'').apply(lambda x:len(x.split(','))))
    close_contact_columns = ['close_contact_'+str(i+1) for i in range(n_new_columns)]
    roster[close_contact_columns] = roster['Close Contacts'].str.split(',', expand = True)
    roster[close_contact_columns] = roster[close_contact_columns].fillna('')
    return roster, close_contact_columns
    
def odd_number_handling(roster):
    '''
    This function randomly selects a person to be double-paired if the total number of employees is odd.
    The selection is random but is done from a filtered pool of employees (Level 10 and not double-paired in the past)
    
    Returns
    -------
    roster: Pandas DataFrame
        The roster with one line appended if the number of employees was odd
    double_paired: str
        The EID of the employee to be double-paired if the roster has an odd number of employees, empty string otherwise
    odd_number: bool
        True if the roster has an odd number of employees, False otherwise
    '''
    
    my_seed = int("%d%02d" % (datetime.now().year, datetime.now().month))
    odd_number = roster.shape[0] % 2 == 1
    old_pair_columns    = [x for x in roster.columns if 'Pair' in x]
    
    if odd_number:
        random.seed(my_seed)

        # Get a list of all level 10s -- candidates for double pair
        double_pair_candidates = roster.loc[roster['Level'] == 10,'EID'].tolist()

        # Exclude the people who have been double paired in the past
        for col in [x for x in old_pair_columns if '.' in x]:
            eid = roster.loc[~roster[col].isna(),'EID'].values.tolist()
            if len(eid)>0:
                double_pair_candidates.remove(eid[0])

        # Pick one person from the double_pair_candidates
        idx = random.randrange(0, len(double_pair_candidates))
        double_paired = double_pair_candidates[idx]
        print('\n', double_paired , 'was randomly selected among all Senior Analysts to be double paired because we have an odd number of people in the roster.')

        # Re-append the respective row to the roster (to make it available for pairing)
        new_row = roster[roster['EID'] == double_paired]
        new_row['EID'] = new_row['EID']+'2'
        roster = roster.append(new_row).reset_index(drop = True)
    else: 
        double_paired = ''   
        
    return roster, double_paired, odd_number

def flag_project_overlap(x):
    ''' This function checks if there are duplicate project names in a string '''
    
    projects_list = x.strip().split(',')
    n_projects_unique = len(set(projects_list))
    n_projects        = len(projects_list)
    return int(n_projects_unique!=n_projects)

def prepare_optim_parameters(roster, close_contact_columns):
    '''
    This function prepares all the matrices that will be used later by the optimization solver.
    
    Returns
    -------
    roster: pandas DataFrame
        Roster with an updated index
    people_list: list
        The list of employees that will be paired (with the douple-paired EID, if any, duplicated)
    person_to_index: dict
        Dictionary with maps the people EIDs to an index 
    P: list
        Binary list that indicates if an employee is a partner
    B: list
        Binary list that indicates if an employee is a level 9 or 10
    M: list
        Binary list that indicates if an employee is an MIT MBAn
    R: numpy array
        Binary matrix that indicates if there is project overlap between two employees  
    L: numpy array
        Binary matrix that indicates if there is a close relationship (PL - mentee, buddy - mentee, family ties, past coffee chat pair) between two employees
    '''
    
    people_list  = roster['EID'].tolist()
    roster.index = roster['EID'].tolist()
    person_to_index = {person:i for person,i in zip(people_list, range(len(people_list)))}
    
    old_pair_columns    = [x for x in roster.columns if 'Pair' in x]

    # If an EID (in the past pairs or close contacts) does not exist in the roster anymore, replace it with the EID in the index
    # We do this because it doesn't change the optimization result and it prevents blank names and NAs from rendering errors 
    # when not matched to an index in the people_list
    for col in old_pair_columns+close_contact_columns:
        condition = ((~roster[col].isin(people_list)) | (roster[col].isna()))# & (roster[col]!='')
        roster.loc[condition,col] = roster.loc[condition].index

    ############################################ OPTIMIZATION ARRAYS & MATRICES ################################################ 
    roster.index = roster['EID'].apply(lambda x: people_list.index(x)).tolist()

    # Binary lists
    P                  = roster['e2e Partner'].replace({np.nan:0,'Y':1}).tolist()
    B                  = roster['Mordor_9_10'].tolist()
    M                  = roster['MIT MBAn'].tolist()

    # 1D Lists -- to be used when creating the L matrix
    people_leads       = roster['People Lead'].dropna().apply(lambda x: people_list.index(x))
    buddies            = roster['Buddy'      ].dropna().apply(lambda x: people_list.index(x))

    # 2D lists -- to be used when creating the L matrix
    close_contacts     = roster.loc[~roster['Close Contacts'].isna(),close_contact_columns].applymap(lambda x: people_list.index(x))
    previous_pairs     = roster[old_pair_columns].applymap(lambda x: people_list.index(x))
    
    # Multi-value arrays -- to be used when creating the R matrix
    projects           = roster['Project'].replace({np.nan:''}).tolist()

    ####################################### CREATE L MATRIX (people links) ##########################################################
    # Create a matrix of zeros and replace with ones when there is a relationship
    
    n = roster.shape[0]
    L = np.zeros((n, n))
    
    rows_pp = sorted(previous_pairs.index.tolist() * len(previous_pairs.columns))   # list of sorted indices with size n * n_previous_pairs
    rows_cc = sorted(close_contacts.index.tolist() * len(close_contacts.columns))   # list of sorted indices with size n * max(close_contacts_by_person)

    L[people_leads.index.tolist(), people_leads  ] = 1; L[people_leads  , people_leads.index.tolist()] = 1    # People lead connections
    L[buddies.index.tolist()     , buddies       ] = 1; L[buddies       , buddies.index.tolist()     ] = 1    # Buddy connections
    L[rows_pp, np.array(previous_pairs).flatten()] = 1; L[np.array(previous_pairs).flatten(), rows_pp] = 1    # Past coffee chats
    L[rows_cc, np.array(close_contacts).flatten()] = 1; L[np.array(close_contacts).flatten(), rows_cc] = 1    # Close contacts


    ####################################### CREATE R MATRIX (project assignments) ###################################################

    matrix_1 = pd.DataFrame([projects]*n)       # repeat the project list across n columns of a dataframe (columns are identical to each other) 
    matrix_2 = pd.DataFrame([projects]*n).T     # matrix_1 transposed (rows are identical to each other)
    combined = matrix_1+','+matrix_2
    R = np.array(combined.applymap(lambda x: flag_project_overlap(x)))
    
    return roster, people_list, person_to_index, P, B, M, R, L

def run_optim_model(people_list, person_to_index, P, B, M, R, L):
    '''
    This function:
    1. formulates the optimization problem with pulp, as defined in the README
    2. solves the optimization problem
    3. converts its result to a list of tuples (each tuple is a coffee chat pair) 
    
    Retuns
    ------
    final_pairs: list of tuples
    '''
    
    # Variables
    x = LpVariable.dicts("x",((i, j) for i in people_list for j in people_list), cat='Binary')

    # Objective -- it is not changing anything but we need it for the code to work
    objective = lpSum(x[i,j] for i in people_list for j in people_list)

    # Model
    chats = LpProblem("cc_pairing",LpMinimize)   # cc_pairing is just the name of the output file, it can be anything 
    chats += objective

    # Constraints
    for i in people_list:

        # 1. Assign every person to exactly one pair (except for the double paired person)
        c1 = lpSum([x[i,j] for j in people_list]) == 1
        chats+=c1

        for j in people_list:
            c2 = x[i,j] == x[j,i]

            # 3. Partners not paired up
            c3 = x[i,j] + P[person_to_index[i]] + P[person_to_index[j]] <= 2

            # 4. Levels 9-10 in Mordor not paired up (they already hang out)
            c4 = x[i,j] + B[person_to_index[i]] + B[person_to_index[j]] <= 2

            # 5. People from the same project not paired up
            c5 = x[i,j] + R[person_to_index[i], person_to_index[j]] <= 1

            # 6. People not paired up with buddies, people leads, past pairs
            c6 = x[i,j] + L[person_to_index[i], person_to_index[j]] <= 1

            # 7. MIT MBAns not paired up (they already hang out)
            c7 = x[i,j] + M[person_to_index[i]] + M[person_to_index[j]] <= 2

            # Add all constraints to the chats model
            chats+=c2; chats+= c3; chats+= c4; chats+= c5; chats+= c6; chats+= c7;
            
    # Solve
    chats.solve(solver = PULP_CBC_CMD(msg = 1));
    if LpStatus[chats.status] != 'Optimal':
        raise Exception('Could not find an optimal solution. Optimization result:' + LpStatus[chats.status])

    # Convert the optimal solution (x) to a list of tuples (every pair will be a tuple)
    final_pairs = []
    for i in people_list:
        i = i.replace('2','')
        if len([x for x in final_pairs if i in x]) == 0:
            pair = (i,)
            for j in people_list:
                if x[(i,j)].varValue == 1:
                    pair = pair + (j.replace('2',''),)
            final_pairs.append(pair)
            
    return final_pairs

def save_new_pairs_to_file(final_pairs, roster, final_pairs_output_file):
    '''
    This function prepares the file that is shared with the company on a monthly basis.
    It has 5 columns: Pair index, EID (person 1), Full Name (person 1), EID (person 2), Full Name (person 2)
    
    As the coffee chats evolved, this shared file is not only listing the pairs but it is also assigning the responsibility of reaching out to one person per pair based on a couple of (arbitrary) rules:
    1. Seniority: the most junior person of the pair is responsible for reaching out
    2. Alphabetical order: if same level, the first person in alphabetical order is responsible for reaching out
    
    Therefore, this function is also swapping people within pairs between "person 1" and "person 2" based on these rules. 
    '''
    
    roster.fillna('',inplace= True)

    # Find who is responsible for outreach based on levels (person on the leftmost column should be responsible to reach out)
    # 1. Bring in the level 
    # 1a. person 1
    final_pairs['EID'] = final_pairs['EID 1']
    final_pairs = pd.merge(final_pairs, roster[['EID','Level']], on = 'EID', how = 'left')
    final_pairs.rename(columns = {'Level':'Level 1'}, inplace = True)

    # 1b. person 2
    final_pairs['EID'] = final_pairs['EID 2']
    final_pairs = pd.merge(final_pairs, roster[['EID','Level']], on = 'EID', how = 'left')
    final_pairs.rename(columns = {'Level':'Level 2'}, inplace = True)

    final_pairs.drop(columns = ['EID'], inplace = True)
    final_pairs['Person 1 [reach out] - EID'] = final_pairs['EID 1']
    final_pairs['Person 2 [just reply :)] - EID'] = final_pairs['EID 2']

    # 2. Swap names if necessary: goal is to have the most junior person (or, if same level, the first one alphabetically) on the leftmost column
    change_order_1 = final_pairs['Level 1']<final_pairs['Level 2']
    change_order_2 = (final_pairs['Level 1'] == final_pairs['Level 2']) & (final_pairs['EID 1'] > final_pairs['EID 2'])
    final_pairs.loc[change_order_1 | change_order_2, 'Person 1 [reach out] - EID']     = final_pairs['EID 2']
    final_pairs.loc[change_order_1 | change_order_2, 'Person 2 [just reply :)] - EID'] = final_pairs['EID 1']

    final_pairs = final_pairs[['Person 1 [reach out] - EID', 'Person 2 [just reply :)] - EID']]
    
    ## get first and last names
    # person 1
    final_pairs = pd.merge(final_pairs, 
                           roster[['EID','First Name','Last Name']].rename(columns = {'EID':'Person 1 [reach out] - EID'}),
                           on = 'Person 1 [reach out] - EID',
                           how='left')
    final_pairs['Person 1 - Full Name'] = final_pairs['First Name'] + ' ' + final_pairs['Last Name']
    final_pairs.drop(columns = ['First Name', 'Last Name'], inplace = True)
    # person 2
    final_pairs = pd.merge(final_pairs, 
                           roster[['EID','First Name','Last Name']].rename(columns = {'EID':'Person 2 [just reply :)] - EID'}),
                           on = 'Person 2 [just reply :)] - EID',
                           how='left')
    final_pairs['Person 2 - Full Name'] = final_pairs['First Name'] + ' ' + final_pairs['Last Name']
    final_pairs.drop(columns = ['First Name', 'Last Name'], inplace = True)

    ## final grooming
    final_pairs['Pair'] = final_pairs.index + 1
    final_pairs = final_pairs[['Pair',
                               'Person 1 [reach out] - EID'    ,'Person 1 - Full Name',
                               'Person 2 [just reply :)] - EID','Person 2 - Full Name']]
    
    final_pairs.to_excel(final_pairs_output_file, index = False)

def update_roster_with_new_pairs(roster_original, final_pairs, odd_number, roster_output_file, eids_remove_temp):
    ''' 
    This function is updating the existing roster file (the one we read as input at the beginning of the process) with a column (or two columns if the number of people was odd). 
    These new columns record the most recent pair assignment.
    The output file is used in the next pair assignment.
    '''
    
    roster_original.fillna('',inplace= True)

    # Convert the final_pairs list to a dictionary that has a key for every E2E-er 
    pairing_dict = {}
    for pair in final_pairs:
        for p1,p2 in zip([0,1],[1,0]):  # going through the loop twice - once for the first person (p1) and once for the second person (p2) in the pair
            if pair[p1] not in pairing_dict.keys():
                pairing_dict[pair[p1]] = pair[p2]
            else:
                # we get in this else statement if we encounter a name twice in the final_pairs list, so if the name is double-paired
                # in that case, we add the second pair of the double-paired individual to the pairing dictionary
                pairing_dict[pair[p1]+'_SECOND'] = pair[p2]
                person_with_two_chats = pair[p1]
                print(person_with_two_chats, 'is double paired')
                
    for eid in eids_remove_temp:
        pairing_dict[eid] = ''
        
    old_pair_columns    = [x for x in roster_original.columns if 'Pair' in x]
    target_pair_incr_id = int(old_pair_columns[-1].split()[1].split('.')[0]) + 1
    
    roster_original['Pair '+str(target_pair_incr_id)] = roster_original['EID'].apply(lambda x:pairing_dict[x])

    # Add column.2 if the number was odd and we have one person with two pairs
    if odd_number:
        roster_original['Pair '+str(target_pair_incr_id + 0.2)] = np.nan
        roster_original.loc[roster_original['EID'] == person_with_two_chats,'Pair '+str(target_pair_incr_id + 0.2)] = pairing_dict[person_with_two_chats+'_SECOND']

    # Save in file
    roster_original.to_excel(roster_output_file, index= False)