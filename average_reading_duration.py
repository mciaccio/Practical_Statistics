
'''
@author:  - menfi

Created on - Nov 30, 2017

Menfi Systems Incorporated
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print()

# %matplotlib inline

np.random.seed(42) 

DATADIR = '/myData/practical_statistics/case_study_AB_tests'  
DATAFILE = 'course_page_actions.csv'
 
course_page_actions_csv  = os.path.join(DATADIR, DATAFILE)

df = pd.read_csv(course_page_actions_csv)

print('df.head(1)')
print(df.head(1))
original_dataframe_row_count = df.shape[0]
print("original_dataframe_row_count - {}\n".format(original_dataframe_row_count))

action_view_df = df.query("action == 'view'") # DataFrame

print('action_view_df.head(2)')
print(action_view_df.head(2))
print()

group_control_df = df.query("group == 'control'") # DataFrame
print('group_control_df.head(2)')
print(group_control_df.head(2))

group_control_df['duration']
# print(group_control_df['duration'])
# print(group_control_df.duration.mean())

group_control_duration_mean = group_control_df.duration.mean() # float
print("group_control_duration_mean - {}\n".format(group_control_duration_mean))

group_experiment_df = df.query("group == 'experiment'") # DataFrame
print('group_experiment_df.head(2)')
print(group_experiment_df.head(2))
# print(type(group_experiment_df))

group_experiment_duration_mean = group_experiment_df.duration.mean() #float
# group_experiment_duration_mean = group_experiment_df['duration'].mean() #float
print("group_experiment_duration_mean - {}\n".format(group_experiment_duration_mean))

duration_observed_difference = group_experiment_duration_mean - group_control_duration_mean # float

# experiment group duration average, mean - 15 seconds more on course overview page  
# duration_observed_difference - 15.525098619574393

# now we have our observed statistic 
# next simulate sampling distribution with bootstrapping 
print("duration_observed_difference - {}\n".format(duration_observed_difference))

# simulate sampling distribution with bootstrapping 

sampled_click_through_rate_difference_list = []

save_differences = []
for _ in range(10000): # fix ** 10000 **
    sample_of_initial_dataframe = df.sample(original_dataframe_row_count, replace = True)
    
    group_control_df = sample_of_initial_dataframe.query("group == 'control'")
    group_control_duration_mean = group_control_df.duration.mean()
    # print("group_control_duration_mean - {}".format(group_control_duration_mean))


    group_experiment_df = sample_of_initial_dataframe.query("group == 'experiment'")
    group_experiment_duration_mean = group_experiment_df.duration.mean()
    # print("group_experiment_duration_mean - {}\n".format(group_experiment_duration_mean))
    
    experiment_duration_mean_minus_control_duration_mean = group_experiment_duration_mean - group_control_duration_mean
    save_differences.append(experiment_duration_mean_minus_control_duration_mean) 

# print("save_differences - {}".format(save_differences))
print("len(save_differences) - {}".format(len(save_differences)))

plt.hist(save_differences)
plt.show()

# find p value
# simulate the distribution under the null
# find probability initial observed statistic came from the simulated distribution under the null centered on zero  

null_vals = np.random.normal(0, np.array(save_differences).std(), np.array(save_differences).size )

plt.hist(null_vals)
plt.axvline(x=duration_observed_difference, color='red')
plt.show()

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
