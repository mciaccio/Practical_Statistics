
'''
@author:  - menfi

Created on - Nov 27, 2017

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
 
homepage_actions_csv = os.path.join(DATADIR, DATAFILE)

df = pd.read_csv(homepage_actions_csv)

print('df.head(1)')
print(df.head(1))
dataFrame_row_count = df.shape[0]
print("dataFrame_row_count - {}\n".format(dataFrame_row_count))

# Get dataframe with all records from control group
control_group_df = df.query('group == "control"')
# print('control_group_df.head(1)')
# print(control_group_df.head(1))
control_group_df_row_count = control_group_df.shape[0]
# print("control_group_df_row_count - {}\n".format(control_group_df_row_count))
# nunique_control_group_ids = control_group_df.id.nunique()
# print("nunique_control_group_ids - {}\n".format(nunique_control_group_ids))
 
control_group_enroll_action_df = control_group_df.query('action == "enroll"') # DataFrame
# print('control_group_enroll_action_df.head(1)')
# print(control_group_enroll_action_df.head(1))
control_group_enroll_action_unique_id_count = control_group_enroll_action_df.id.nunique()
print("control_group_enroll_action_unique_id_count - {}".format(control_group_enroll_action_unique_id_count))

control_group_view_action_df = control_group_df.query('action == "view"') # DataFrame
# print('control_group_view_action_df.head(1)')
# print(control_group_view_action_df.head(1))
control_group_view_action_unique_id_count = control_group_view_action_df.id.nunique()
print("control_group_view_action_unique_id_count - {}".format(control_group_view_action_unique_id_count))
 
# Compute click through rate for control group
# control_ctr = control_df.query('action == "enroll"').id.nunique() / control_df.query('action == "view"').id.nunique()
# control_click_through_rate = control_group_enroll_action_unique_id_count / control_group_view_action_unique_id_count
control_click_through_rate = control_group_enroll_action_unique_id_count / control_group_view_action_unique_id_count
print("control_click_through_rate - {}\n".format(control_click_through_rate))
#      control_click_through_rate - 0.2364438839848676

# Get dataframe with all records from experiment group
# experiment_df = 
experiment_group_df = df.query('group == "experiment"')
experiment_group_df_nunique_id_count = experiment_group_df.id.nunique()
print("experiment_group_df_nunique_id_count - {}".format(experiment_group_df_nunique_id_count))

experiment_group_view_action_df = experiment_group_df.query('action == "view"') # DataFrame
# print('experiment_group_view_action_df.head(1)')
# print(experiment_group_view_action_df.head(1))
experiment_group_view_action_unique_id_count = experiment_group_view_action_df.id.nunique()
# print("experiment_group_view_action_unique_id_count - {}".format(experiment_group_view_action_unique_id_count))

experiment_group_enroll_action_df = experiment_group_df.query('action == "enroll"') # DataFrame
# print('experiment_group_enroll_action_df.head(1)')
# print(experiment_group_enroll_action_df.head(1))
experiment_group_enroll_action_unique_id_count = experiment_group_enroll_action_df.id.nunique()
print("experiment_group_enroll_action_unique_id_count - {}".format(experiment_group_enroll_action_unique_id_count))

experiment_click_through_rate = experiment_group_enroll_action_unique_id_count / experiment_group_view_action_unique_id_count
print("experiment_click_through_rate - {}\n".format(experiment_click_through_rate))

# Compute the observed difference in click through rates
# Display observed difference
# ** also called observed statistic ** plotted below **
observed_click_through_rate_difference = experiment_click_through_rate - control_click_through_rate
print("observed_click_through_rate_difference - {}\n".format(observed_click_through_rate_difference))

# Create a sampling distribution of the difference in proportions
# standard deviation and size used to calculate ** simulate distribution under the null hypothesis *** below   
# with bootstrapping
# diffs = []
sampled_click_through_rate_difference_list = []
for _ in range(10000):
    sample_of_dataframe = df.sample(dataFrame_row_count, replace=True) # DataFrame 
    
    sample_control_group_df = sample_of_dataframe.query('group == "control"')
    
    sample_control_group_view_action_df = sample_control_group_df.query('action == "view"') # DataFrame
    sample_control_group_view_action_unique_id_count = sample_control_group_view_action_df.id.nunique()
    # print("sample_control_group_view_action_unique_id_count - {}".format(sample_control_group_view_action_unique_id_count))

    sample_control_group_enroll_action_df = sample_control_group_df.query('action == "enroll"') # DataFrame
    sample_control_group_enroll_action_unique_id_count = sample_control_group_enroll_action_df.id.nunique()
    # print("sample_control_group_enroll_action_unique_id_count - {}".format(sample_control_group_enroll_action_unique_id_count))
    
    sampling_control_click_through_rate =  sample_control_group_enroll_action_unique_id_count / sample_control_group_view_action_unique_id_count
    # print("sampling_control_click_through_rate - {}\n".format(sampling_control_click_through_rate))


    sample_experiment_group_df = sample_of_dataframe.query('group == "experiment"')
    
    sample_experiment_group_view_action_df = sample_experiment_group_df.query('action == "view"') # DataFrame
    sample_experiment_group_view_action_unique_id_count = sample_experiment_group_view_action_df.id.nunique()
    # print("sample_experiment_group_view_action_unique_id_count - {}".format(sample_experiment_group_view_action_unique_id_count))    
 
    sample_experiment_group_enroll_action_df = sample_experiment_group_df.query('action == "enroll"') # DataFrame
    sample_experiment_group_enroll_action_unique_id_count = sample_experiment_group_enroll_action_df.id.nunique()
    # print("sample_experiment_group_enroll_action_unique_id_count - {}".format(sample_experiment_group_enroll_action_unique_id_count))
    
    sampling_experiment_click_through_rate =  sample_experiment_group_enroll_action_unique_id_count / sample_experiment_group_view_action_unique_id_count
    # print("sampling_experiment_click_through_rate - {}\n".format(sampling_experiment_click_through_rate))

    sampled_click_through_rate_difference_list.append(sampling_experiment_click_through_rate - sampling_control_click_through_rate) 

# print("sampled_click_through_rate_difference_list - {}".format(sampled_click_through_rate_difference_list))
print("len(sampled_click_through_rate_difference_list) - {}".format(len(sampled_click_through_rate_difference_list)))

# plt.hist(sampled_click_through_rate_difference_list)
# convert python list to numpy array
plt.hist(np.array(sampled_click_through_rate_difference_list))
plt.show()

# Simulate distribution under the null hypothesis
# null_vals = 

# simulate distribution under the null hypothesis *** correct words ***
# **np.random.seed(42) ** - above important 
# null_vals is the simulated, created distribution under the null hypothesis   
# normal distribution centered at 0 zero
# use the standard deviation from the, 10,000, sampling distribution
# use the size from the, 10,000, sampling distribution 
# sampled_click_through_rate_difference_list
# 
null_vals = np.random.normal(0, np.array(sampled_click_through_rate_difference_list).std(), np.array(sampled_click_through_rate_difference_list).size) # class 'numpy.ndarray

# Plot the null distribution
# Plot observed statistic with the null distibution
plt.hist(null_vals)
plt.axvline(x=observed_click_through_rate_difference, color='red')
plt.show()

# Compute p-value
# What is the probability our statistic, original DataFrame wide analysis, came from the simulated distribution under the null hypothesis?
# null_vals greater than, more extreme than statistic, to the right of the red line on the second histogram plot, null values in favor of H1 - alternative Hypothesis 
# statistic - observed_click_through_rate_difference - 0.030034443684015644
p_value = (null_vals > observed_click_through_rate_difference).mean()
print("p_value - {}".format(p_value))

# p_value - 0.0044
# p value about 0.44%
# p value < 0.01 or 1% 
# difference between control and experiment click through rates does appear to be significant 
# reject null hypothesis in favor of the alternative hypothesis 
# unlikely statistic is from the null, red line is off to the right, very few values more extreme, more to the right 

 
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
