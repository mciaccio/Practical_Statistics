
'''
@author:  - menfi

Created on - Dec 1, 2017

Menfi Systems Incorporated
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# % matplotlib inline
print()
np.random.seed(42)

DATADIR = '/myData/PracticalStatistics/'  
DATAFILE = 'classroom_actions.csv'
 
classroom_actions_csv  = os.path.join(DATADIR, DATAFILE)

df = pd.read_csv(classroom_actions_csv) 
# df = pd.read_csv('/myData/PracticalStatistics/classroom_actions.csv') 
print('df.head()')
print(df.head())
original_dataframe_row_count = df.shape[0]
print("original_dataframe_row_count - {}\n".format(original_dataframe_row_count))

control_group = df.query("group == 'control'")
control_count = control_group.id.nunique()
print("control_count - {}".format(control_count))

control_group_completed = control_group.query("completed == True")
control_completed_count = control_group_completed.id.nunique()
print("control_completed_count - {}".format(control_completed_count)) 

proportion_control_completed =  control_completed_count / control_count
print("proportion_control_completed - {}\n".format(proportion_control_completed))

experiment_group = df.query("group == 'experiment'")
experiment_group_count = experiment_group.id.nunique()
print("experiment_group_count - {}".format(experiment_group_count))

experiment_group_completed = experiment_group.query("completed == True")
experiment_completed_count = experiment_group_completed.id.nunique()
print("experiment_completed_count - {}".format(experiment_completed_count))

proportion_experiment_completed = experiment_completed_count / experiment_group_count
print("proportion_experiment_completed - {}\n".format(proportion_experiment_completed))

observed_difference_click_through_rate = proportion_experiment_completed - proportion_control_completed
print("observed_difference_click_through_rate - {}\n".format(observed_difference_click_through_rate))

differences = []
for _ in range(10000): # ** fix 10000 **
    sample_of_df = df.sample(original_dataframe_row_count, replace = 'True')
#     print(sample_of_df.shape[0])
    
    control_group1 = sample_of_df.query("group == 'control'")
    control1_group_row_count = control_group1.id.nunique() 
    # print("control1_group_row_count - {}".format(control1_group_row_count))

    control_group_completed1 = control_group1.query("completed == True")
    control_group_completed1_row_count = control_group_completed1.id.nunique()
    # print("control_group_completed1_row_count - {}".format(control_group_completed1_row_count))
    
    control1_proportion = control_group_completed1_row_count / control1_group_row_count
#     print("control1_proportion - {}".format(control1_proportion))

    experiment_group1 = sample_of_df.query("group == 'experiment'")
    experiment_group1_row_count = experiment_group1.id.nunique()
#     print("experiment_group1_row_count - {}".format(experiment_group1_row_count))

    experiment_group1_completed = experiment_group1.query("completed == True")
    experiment_group1_completed_row_count = experiment_group1_completed.id.nunique()
#     print("experiment_group1_completed_row_count - {}".format(experiment_group1_completed_row_count))
        
    experiment1_proportion = experiment_group1_completed_row_count / experiment_group1_row_count
#     print("experiment1_proportion - {}".format(experiment1_proportion))
    
    proportion_difference = experiment1_proportion - control1_proportion
    # print("proportion_difference - {}\n".format(proportion_difference))
    differences.append(proportion_difference)
    
# print("differences - {}".format(differences))
print("len(differences) - {}".format(len(differences)))

plt.hist(differences)
plt.show()

# find p value
# simulate the distribution under the null
# find probability initial observed statistic came from the simulated distribution under the null centered on zero  
null_vals = np.random.normal(0, np.array(differences).std(), np.array(differences).size )
# Plot the null distribution
# Plot observed statistic with the null distribution
plt.hist(null_vals)
plt.axvline(x=observed_difference_click_through_rate, color='red')
plt.show()

# Compute p-value
# What is the probability our statistic, original DataFrame wide analysis, came from the simulated distribution under the null hypothesis?
# null_vals greater than, more extreme than statistic, to the right of the red line on the second histogram plot, null values in favor of H1 - alternative Hypothesis 
# statistic - observed_difference_statistic - 1.3026031488719099 
p_value = (null_vals > observed_difference_click_through_rate).mean()
print("p_value - {}\n".format(p_value))


''' 
print(" - {}".format())
print(".shape[0] - {}".format(.shape[0]))
print("type() - {}".format(type()))
print("len() - {}".format(len()))

print(type())
print(len())

print('')
print()
print()

print('.head()')
print(.head())
print()
'''