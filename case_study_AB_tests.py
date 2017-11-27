
'''
@author:  - menfi

Created on - Nov 26, 2017

Menfi Systems Incorporated
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print()

# %matplotlib inline 

DATADIR = '/myData/practical_statistics/case_study_AB_tests'  
DATAFILE = 'homepage_actions.csv'
 
homepage_actions_csv = os.path.join(DATADIR, DATAFILE)

df = pd.read_csv(homepage_actions_csv)

print('df.head()')
print(df.head())
print()
  
# print('df.tail()')
# print(df.tail())
# print()
  
print("df.action.unique()")
# print(df['action'].unique())
print(df.action.unique())
print()

df_row_count = df.shape[0]
print("df_row_count - {}".format(df_row_count))

total_number_of_actions = df_row_count
print("total_number_of_actions - {}\n".format(total_number_of_actions))

unique_users = df.id.unique() # numpy.ndarray # 6328
unique_users_count = len(unique_users)
print("unique_users_count - {}\n".format(unique_users_count))
  
group_control = df.query("group == 'control'") 
group_control_count = len(group_control) 
print("group_control_count - {}".format(group_control_count))

control_viewers = df.query("group == 'control' and action == 'view'")
control_viewers_count = len(control_viewers) 
print("control_viewers_count - {}".format(control_viewers_count))

control_clickers = df.query("group == 'control' and action == 'click'") 
control_clickers_count = len(control_clickers) 
print("control_clickers_count - {}".format(control_clickers_count))

control_click_through_rate = control_clickers_count / control_viewers_count
print("control_click_through_rate - {}\n".format(control_click_through_rate))

group_experiment = df.query("group == 'experiment'") 
group_experiment_count = len(group_experiment) 
print("group_experiment_count - {}".format(group_experiment_count))

experiment_viewers = df.query("group == 'experiment' and action == 'view'")
experiment_viewers_count = len(experiment_viewers) 
print("experiment_viewers_count - {}".format(experiment_viewers_count))

experiment_clickers = df.query("group == 'experiment' and action == 'click'")
experiment_clickers_count = len(experiment_clickers) 
print("experiment_clickers_count - {}".format(experiment_clickers_count))  
experiment_click_through_rate = experiment_clickers_count / experiment_viewers_count
print("experiment_click_through_rate - {}\n".format(experiment_click_through_rate))

# hang on to this original DataFrame wide analysis, this will be a data point, red line on our final plot 
# this is our statistic - observed difference in proportions
# later we will calculate the p value for this statistic - the observed differences in proportions   
click_through_rate_observed_difference = experiment_click_through_rate - control_click_through_rate
print("click_through_rate_observed_difference - {}".format(click_through_rate_observed_difference))

# At this point we looked at the entire DataFrame and did math on the entire DataFrame
#
# Additional analysis is needed to determine if the ** click_through_rate_observed_difference ** is statistically significant  
# Is the click_through_rate_observed_difference statistically significant or due to chance? 
# 
# The required additional analysis is the bootstrap process,
#   
# do the analysis 10,000 times 
#
# get a random sample from the original DataFrame do this 10,000 times
# do the original analysis on each of the 10,000 samples
# save the 10,000 results in a list
# plot - histogram the 10,000 click through rate differences 
# 

df_row_count = df.shape[0]
print("df_row_count - {}\n".format(df_row_count))

experiment_CTR_minus_control_CTR_list = []
for _ in range (10000):
    sample_of_dataframe = df.sample(df_row_count, replace=True)
    # print("len(sample_of_dataframe) - {}\n".format(len(sample_of_dataframe)))
    
    # control
    sample_control_dataframe = sample_of_dataframe.query("group == 'control'") # DataFrame 
  
    sample_control_click_id_nunique = sample_control_dataframe.query("action == 'click'").id.nunique() # class 'numpy.ndarray
    # print("sample_control_click_id_nunique - {}".format(sample_control_click_id_nunique))
    
    sample_control_view_id_nunique = sample_control_dataframe.query("action == 'view'").id.nunique() # class 'numpy.ndarray
    # print("sample_control_view_id_nunique - {}".format(sample_control_view_id_nunique))
     
    control_click_through_rate = sample_control_click_id_nunique / sample_control_view_id_nunique
    # print("control_click_through_rate - {}\n".format(control_click_through_rate))
    
    # experiment
    sample_experiment_dataframe = sample_of_dataframe.query("group == 'experiment'") # DataFrame
     
    sample_experiment_click_id_nunique = sample_experiment_dataframe.query("action == 'click'").id.nunique() # class 'numpy.ndarray
    # print("sample_experiment_click_id_nunique - {}".format(sample_experiment_click_id_nunique))
    
    sample_experiment_view_id_nunique = sample_experiment_dataframe.query("action == 'view'").id.nunique() # class 'numpy.ndarray
    # print("sample_experiment_view_id_nunique - {}".format(sample_experiment_view_id_nunique))
    
    experiment_click_through_rate = sample_experiment_click_id_nunique / sample_experiment_view_id_nunique
    # print("experiment_click_through_rate - {}\n".format(experiment_click_through_rate))
    
    experiment_CTR_minus_control_CTR_list.append(experiment_click_through_rate - control_click_through_rate)
    
# print('experiment_CTR_minus_control_CTR_list')
# print(experiment_CTR_minus_control_CTR_list)
# print() 

plt.hist(experiment_CTR_minus_control_CTR_list)
plt.show()

# ** Here is where we simulate, create the distribution under the null hypothesis ** 
# experiment_CTR_minus_control_CTR_list - original DataFrame wide analysis 
print("np.array(experiment_CTR_minus_control_CTR_list).std() - {}".format(np.array(experiment_CTR_minus_control_CTR_list).std()))
# np.array(experiment_CTR_minus_control_CTR_list).std() - 0.010300505167071384

print("np.array(experiment_CTR_minus_control_CTR_list).size - {}\n ".format(np.array(experiment_CTR_minus_control_CTR_list).size))
# np.array(experiment_CTR_minus_control_CTR_list).size - 10

# null_vals is the simulated, created distribution under the null hypothesis   
# normal distribution centered at 0 zero
# same standard deviation as original DataFrame wide analysis
null_vals = np.random.normal(0, np.array(experiment_CTR_minus_control_CTR_list).std(), np.array(experiment_CTR_minus_control_CTR_list).size) # class 'numpy.ndarray

# print('null_vals')
# print(null_vals)
# print() 

# plot null distribution and line at our observed difference
plt.hist(null_vals)
# add red line, our original DataFrame wide analysis
# click_through_rate_observed_difference = experiment_click_through_rate - control_click_through_rate
# click_through_rate_observed_difference - 0.030034443684015644
# our observed statistic

plt.axvline(x=click_through_rate_observed_difference, color='red')
plt.show()

# What is the probability our statistic, original DataFrame wide analysis, came from the simulated distribution under the null hypothesis?
# compute p-value
# null_vals greater than, more extreme than statistic, to the right of the red line on the second histogram plot, null values in favor of H1 - alternative Hypothesis 
# statistic - click_through_rate_observed_difference - 0.030034443684015644
p_value = (null_vals > click_through_rate_observed_difference).mean()
print("p_value - {}".format(p_value))

# p_value - 0.0044
# p value about 0.44%
# p value < 0.01 or 1% 
# difference between control and experiment click through rates does appear to be significant 
# reject null hypothesis in favor of the alternative hypothesis 
# unlikely statistic is from the null, red line is off to the right, very few values more extreme, more to the right 


 
 
''' 
print(" - {}".format())
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



