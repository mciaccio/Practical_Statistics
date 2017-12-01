
'''
@author:  - menfi

Created on - Nov 30, 2017

Menfi Systems Incorporated
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# % matplotlib inline

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

control_group_df = df.query("group == 'control'") # DataFrame
print('control_group_df.head()')
print(control_group_df.head())
print()

control_group_df_total_days = control_group_df.total_days # Series
print('control_group_df_total_days.head()')
print(control_group_df_total_days.head())
print()
# print(type(control_group_df_total_days))

control_group_df_total_days_mean = control_group_df_total_days.mean() # float
print("control_group_df_total_days_mean - {}".format(control_group_df_total_days_mean))


experiment_group_df = df.query("group == 'experiment'") # DataFrame
experiment_group_df_total_days_mean =  experiment_group_df.total_days.mean() 
print("experiment_group_df_total_days_mean - {}".format(experiment_group_df_total_days_mean))

observed_difference_statistic = experiment_group_df_total_days_mean - control_group_df_total_days_mean
print("observed_difference_statistic - {}\n".format(observed_difference_statistic))  

# simulate sampling distribution with bootstrapping 
save_differences = []
for _ in range(10000): # fix ** 10000 **
    sample_of_initial_dataframe = df.sample(original_dataframe_row_count, replace = True)
    
    control_group_df = sample_of_initial_dataframe.query("group == 'control'")
    control_mean = control_group_df.total_days.mean()
    # print("control_mean - {}".format(control_mean))
    
    experiment_group_df = sample_of_initial_dataframe.query("group == 'experiment'")
    experiment_mean = experiment_group_df.total_days.mean()
    # print("experiment_mean - {}\n".format(experiment_mean))
    
    save_differences.append(experiment_mean - control_mean)
        
# print("save_differences - {}".format(save_differences))
print("len(save_differences) - {}".format(len(save_differences)))

plt.hist(save_differences)
plt.show()

# find p value
# simulate the distribution under the null
# find probability initial observed statistic came from the simulated distribution under the null centered on zero  
null_vals = np.random.normal(0, np.array(save_differences).std(), np.array(save_differences).size )
# Plot the null distribution
# Plot observed statistic with the null distribution
plt.hist(null_vals)
plt.axvline(x=observed_difference_statistic, color='red')
plt.show()

# Compute p-value
# What is the probability our statistic, original DataFrame wide analysis, came from the simulated distribution under the null hypothesis?
# null_vals greater than, more extreme than statistic, to the right of the red line on the second histogram plot, null values in favor of H1 - alternative Hypothesis 
# statistic - observed_difference_statistic - 1.3026031488719099 
p_value = (null_vals > observed_difference_statistic).mean()
print("p_value - {}".format(p_value))

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