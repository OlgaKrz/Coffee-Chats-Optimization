# Coffee Chats - Optimization Code

## A Drop of History
The coffee chats initiative started in March 2022 as an effort to keep the End-to-End (E2E) Analytics community engaged in the Work-From-Home environment. The idea is to randomly pair people up on a monthly basis. However, random assignments are not necessarily optimal so, we added a few constraints (e.g.: avoid pairing up the company partners) to maximize the impact of the chats and to increase the likelihood of the E2E community embracing them.  

In order to speed up the launch of the coffee chats, the semi-random monthly assignment was initially done in Excel using the RAND() function. Checking that constraints are met, though, was time consuming. Combined with E2E's reputation as an analytics-driven consultancy, my innate urge for efficiency and encouragement from a couple of colleagues, I found myself optimizing the coffee chat pair assignments using an Integer Programming (IP) formulation. 

**Confidentiality Guarantee**:
The original E2E roster has been modified to remove any confidential information. The names, addresses and client names in the existing roster were generated by ChatGPT and are all fictional, inspired by movies. 

**Note**: If, in a given month, the number of employees is odd, we first pick one person to be assigned to two chats and then run the optimization. How do we pick the double-paired individual? We first restrict our sample to the most junior E2E-ers who have not been double-paired in the past and then we randomly pick one of them. 

## Code Overview
#### Input
The code reads a roster file in which every row corresponds to an employee with a unique EID (Employee ID). 

Context about some columns of the input file:
- **e2e Partner**: Populated with "Y" if the employee is a partner at E2E, blank otherwise.
- **Level**: Determines the seniority level of the employee (L1 being the most senior and L11 being the least senior).
- **Project**: Lists the client names of the projects that the employee is involved in (column value is blank when unknown). 
- **People Lead**: Every employee at E2E has a "People Lead" who is their mentor. People Lead and mentee have frequent career-related discussions (column value is blank when unknown) so the column was added to avoid pairing them up for coffee chats.
- **Buddy**: Every employee at E2E also has a "Buddy" who is a more informal mentor than the People Lead. Buddies make themselves available for work-related concerns / questions and might also meet often so, this column was added for the same reason as the "People Lead" column.
- **MIT MBAn**: Determines whether the employee is an alumni of the Master of Business Analytics at MIT. There are quite a few employees with that degree at E2E and they already interact a lot with each other so this column was added to minimize the coffee chat pair assignments among them. 
- **Close Contacts**: If employee A is listed in that column, it means that they have a close relationship with the employee in that row (e.g.: family relationship or friendship). This column was added to minimize coffee chat pair assignment among the people who are already interacting outside work. 
- **Pair Columns**: These columns list the past coffee chat assignments of a given employee (e.g.: first month's coffee chat assignment in the "Pair 1" column, second month's coffee chat assignment in the "Pair 2" column, etc.). If, for employee C, the value is blank in one of those columns, there are two possibilities: (1) employee C was paired up with employee D who is not with E2E anymore or (2) employee C had not joined E2E yet and, hence, was not paired up with anyone.  
- **Pair Columns ending in ".2"**: These columns are added if the total number of employees in a given month was odd, which led to one person being assigned to two coffee chat pairs. In these columns, there are blank values for all the employees but the one that was double-paired.  

#### Outputs
1. **Final Assignment**: This is the list of the coffee chats pairs for a given month. It has 5 columns:
    - Pair number
    - EID of Employee 1
    - Full name of Employee 1
    - EID of Employee 2
    - Full name of Employee 2
   
   The column names are such because we invented some rules to facilitate the realization of the coffee chats: the person who is responsible to reach out and set up the chat is the most junior in the pair and if the paired up employees are at the same level, then the responsible one is the one whose first name is first alphabetically. Therefore, "Employee 1" is the responsible to reach out in this file.

2. **Roster File**: This is the roster file (that the code read as input) ***updated*** with the latest coffee chat pair. It will inform next month's assignment.
