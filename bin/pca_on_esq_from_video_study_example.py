"""
Created on Wed Mar 31 12:16:08 2021

@author: Bronte Mckeown & Will Strawson

Created to apply PCAs to different combinations of ESQ dataframes.

In this example, there are two independent samples:
    1) N = 70 PS
    2) N = 49 PS
    
This example script runs a PCA with varimax rotation on each sample separately.

Input: Per-observation ESQ dataframe (.csv)
Output: Dataframe with PCA scores and plots per sample all in one PDF as well
as wordclouds for each PCA for each sample.

"""


from ThoughtSpace import pca_plots
import os
import pandas as pd
from collections import OrderedDict
import numpy as np

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import Rotator

## FOR YOUR OWN ANALYSIS, EDIT A NEW COPY OF ME :) ##

# TODO: make it so if it crashes during saving it closes the file
# TODO: potentially move saving csv file out of merging function
# TODO: potentially migrate results out of scratch folder
# TODO: improve generalizability of projection section (implement function)
# TODO: figure out why factor column moves when adding scores to datasheet
# TODO: make function for applying PCA and rotation

#%% load data & basic cleaning/prep

## Set user variables:

# if you know how many components you want to extract, insert True, otherwise false
# If True,  n components can be defined for each sample in n_component_dict below
n_components = True  

# set to true if you want to extract components based on eigenvalue >1
# otherwise, set to False
ev_extraction = False  

# set to true if you want to apply rotation
rotation_on = True
# set rotation you want to apply to each sample's PCA
# in this example, we want varimax applied to both
rotation = ['varimax', 'varimax']

# if you want to z-score ESQ by sample (e.g., N70 only or N49 only), set to True
by_sample = True  
sample_col = "sample"  # input sample column name if z-scoring by sample

by_condition = False  # if you want to z-score by condition (e.g.,control, action, suspense)
condition_col = "condition"  # input condition column name if z-scoring by condition

by_person = False  # if you want to z-score by person
person_col = "idno"  # input id number column if z-scoring by id


# set path of ESQ data
# commented out code below can be used to direct to data stored in current repo
# however, for this example, the data path is hard coded

# current path should = parent repo directory
# current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# data_name = "0.1_combined_esq_N119_74_cols.csv"
# data_path = "scratch/data/" + data_name
# join git repo parent to relative path to data
# data_path = os.path.join(current_path, data_path)
# print("Input data path: ", data_path, "\n")

data_path = '//mnt//c//Users//bront//Documents//PhD//Projects//lab_to_realworld//data//lab//temp_data//temp_data_N119//0.1_combined_esq_N119_74_cols.csv'

# read in ESQ data as a csv or tsv file
# Remember to specify the separtor if not a csv file (e.g. '\t' if not .csv)
df = pd.read_csv(data_path)

# To create join key for merging PCA results at the end- 'observation_id',
# provide name of participant identifier column
df = pca_plots.create_observation_id(df, "idno")

# To find index of the start and end of ESQ columns in master dataframe (df),
# provide strings of start and end ESQ column names
col_start, col_end = pca_plots.esq_cols(df, "focus", "source")

# This calls the z_score function to zscore ESQ columns identified above 
# using esq_cols function
# Nothing should need editing here! It's all set up above
df = pca_plots.z_score(
    df,
    col_start,
    col_end,
    by_sample=by_sample,
    sample_col=sample_col,
    by_condition=by_condition,
    condition_col=condition_col,
    by_person=by_person,
    person_col=person_col,
)

# Use function again to find index of Z scored ESQ columns by providing strings 
# of s-scored start and end cols
col_start, col_end = pca_plots.esq_cols(df, "Zfocus", "Zsource")

# Here, user should enter esq labels for display on plots & word clouds
# Labels need to be written in the same order as they appear in the columns
display = [
    "Task",
    "Future",
    "Past",
    "Self",
    "Person",
    "Emotion",
    "Words",
    "Detail",
    "Deliberate",
    "Problem",
    "Diverse",
    "Intrusive",
    "Memory"
]

#%% select groups for applying PCA

# In this example, we are applying PCA with varimax rotation to two samples
# separately (n = 70 & n = 49)

df_N70 = df.loc[df["sample"] == "N70"].copy()  # just select N70 sample
df_N49 = df.loc[df["sample"] == "N49"].copy()  # just select N49 sample


# create ordered dict of dataframes which you want to apply PCA to
# key: name of the dataframe, value: dataframe
# If no group splits (i.e. one PCA on all data), simply create one entry in this dictionary!

df_dict = OrderedDict(
    [
        ("df_N70", df_N70),
        ("df_N49", df_N49)
    ]
)

# dictionary for number of components for each solution - must use same key as df_dict
n_components_dict = OrderedDict(
    [
        ("df_N70", 3),
        ("df_N49", 3)
       
    ]
)

# Add any info you want about specific analysis run for result file names
# in this example, I've included the samples and how I've z-scored them
results_id = "N70_N49_zscored_by_sample"

# create empty dict to store esq columns from each dataframe stored in df_dict
esq_dict = OrderedDict()
# create empty dict to store display labels from each dataframe stored in df_dict
display_dict = OrderedDict()
# create empty OrderedDict for storing rotation type
rotation_dict = OrderedDict()

# loop over df_dict to:
    # assign esq cols to esq_dict
    # assign display labels to display_dict
    # assign rotation dict
for index, (key, i)  in enumerate(df_dict.items()):
    i = i.iloc[:,col_start:col_end] # should be z-scored ESQ questions
    i = i.apply(pd.to_numeric, errors="coerce") # make sure numeric
    i = i.dropna() # drop rows with nan values
    esq_dict[key] = i
    display_dict[key] = display # dictionary where values are display (list above)
    # if different solutions have different display items, here is where they can be specified
    # remove Z from z scored column names for display
    # display_dict[key] = [x.replace("Z", "") for x in i.columns.tolist()]
    rotation_dict[key] = rotation[index] # set rotation dict

#%% run naive PCA on all dataframes stored in esq_dict

# first check KMO and Bartlett's test of sphericiity
# TODO: this needs wrapping into a function
for k, v in esq_dict.items():
    chi_square_value,p_value=calculate_bartlett_sphericity(v)
    print ("Chi-square value:", chi_square_value, "P-value:", p_value)
    kmo_all,kmo_model=calculate_kmo(v)
    print ("KMO:", kmo_model)

# svd = full, meaning it calculates as many components as there are items
pca_dict = pca_plots.naive_pca(esq_dict)

#%% extract pca scores and component loadings from pca_dict

# TODO: this needs wrapping into a function

# create empty ordered dictionaries for storing un-rotated & rotated scores, loadings and n components
unrotated_scores = OrderedDict()
unrotated_loadings = OrderedDict()
unrotated_n_components = OrderedDict()
unrotated_percent_variance = OrderedDict()
unrotated_cum_percent_variance = OrderedDict()

rotated_scores = OrderedDict()
rotated_loadings = OrderedDict()
rotated_n_components = OrderedDict()
rotated_percent_variance = OrderedDict()
rotated_cum_percent_variance = OrderedDict()

# loop over dataframes sroted in 'esq_dict'
# in this example, this is a df for N = 70 & a df for N = 49
for k in esq_dict:
    # If you have set n_components to True:
    if n_components:
        # Generate per-observation scores for each factor
        scores = pca_dict[k].transform(esq_dict[k])[:, : n_components_dict[k]]
        # Generate per-item factor loadings
        pc = pca_dict[k].components_[: n_components_dict[k], :]
        
        # Generate % variance explained for each factor
        explained = pca_dict[k].explained_variance_ratio_
        print  ("Variance explained by each factor:", explained)
        # Generate cumalitive variance explained
        cum_explained = pca_dict[k].explained_variance_ratio_.cumsum()
        print ("Cumalitive variance explained:", cum_explained)

        # Store factor scores
        unrotated_scores[k] = scores
        # Store factor loadings
        unrotated_loadings[k] = pc
        # Store number of components
        unrotated_n_components[k] = n_components_dict[k]
        # Store % variance explained
        unrotated_percent_variance[k] = explained
        # Store cumaltitive % variance explained
        unrotated_cum_percent_variance[k] = cum_explained
        print("Shape of 'scores' dataframe (i.e. n_components):", scores.shape)
    # if you have set extraction based on eigevalues to be True:
    elif ev_extraction:
        evs = [i for i in pca_dict[k].explained_variance_ if i > 1]
        n_components = len(evs)
        # Generate per-observation scores for each factor
        scores = pca_dict[k].transform(esq_dict[k])[:, :n_components]
        # Generate per-item factor loadings
        pc = pca_dict[k].components_[:n_components, :]
        # Store factor scores
        unrotated_scores[k] = scores
        # Store factor loadings
        unrotated_loadings[k] = pc
        # Store number of components
        unrotated_n_components[k] = n_components
        # reset n_components to None
        #n_components = None

    else:
        scores = pca_dict[k].transform(esq_dict[k])
        pc = pca_dict[k].components_
        unrotated_scores[k] = scores
        unrotated_loadings[k] = pc
        unrotated_n_components[k] = scores.shape[-1]
        print("Shape of 'scores' dataframe (i.e. n_components):", scores.shape)
        
    if rotation_on: 
    
        ## Apply rotation
            
        # print type of rotation for sanity checking
        print ('Rotation:', rotation_dict[k])
        
        # set up rotator object with method selected from rotation dictionary
        rotator = Rotator(method = rotation_dict[k])
        
        # Generate per-item component loadings (i.e., component loadings table)
        # rotator function wants component loadings table to be transposed
        # then, we want to re-transpose it when saving as variable
        rotated_pc = rotator.fit_transform(pc.T).T
        # add component loadings to dictionary
        rotated_loadings[k] = rotated_pc
        
        # print out shape of rotated component loading table
        print (rotated_pc.shape)
        
        # Generate per-observation scores for each component
        # need to transpose rotated_pc
        rotated_score = np.dot(esq_dict[k], rotated_pc.T)
        # add scores to dictionary
        rotated_scores[k] = rotated_score
        
        # print shape of scores dataframe
        rotated_n_components[k] = rotated_scores[k].shape[-1]
        print( "[{} rotated] shape of 'scores' dataframe (i.e., n_components".format(rotation_dict[k]), rotated_scores[k].shape)

#%% Add PCA scores to master df

# adds pca scores to esq_dict dataframes for merging
if rotation_on:
    esq_dict_with_scores = pca_plots.append_scores(rotated_scores, esq_dict)
    print("Added rotated scores")
else:
    esq_dict_with_scores = pca_plots.append_scores(unrotated_scores, esq_dict)
    print("Added non-rotated scores")
    
# merge all and save
output_df = pca_plots.merge_dataframes(
    esq_dict_with_scores,
    df,
    data_path,
    results_id,
    rotation_on,
    n_components,
    n_components_dict,
)

#%% Plots & word clouds for PCA solutions for each group (saves out PDF)
# set mask threshold for heatmaps
mask_threshold = 0
if rotation_on:
    pca_plots.page_of_plots(
        pca_dict,
        rotated_loadings,
        mask_threshold,
        results_id,
        rotation_on,
        n_components,
        n_components_dict,
        display_dict,
    )
    pca_plots.wordclouder(rotated_loadings, display_dict, savefile=True)
else:
    pca_plots.page_of_plots(
        pca_dict,
        unrotated_loadings,
        mask_threshold,
        results_id,
        rotation_on,
        n_components,
        n_components_dict,
        display_dict,
    )
    pca_plots.wordclouder(unrotated_loadings, display_dict, savefile=True)

# %% Project patterns between samples

# TODO: include example using projection function!

# if not needed for your analysis, just comment out this section!

# create empty ordered dictionary to store projected scores
lab_projected_scores_dict = OrderedDict()

# loop over keys and values of varimax rotated component loadings
for key, loadings in rotated_loadings.items():
    # loop over each pattern in loadings
    idx = 0
    for pattern in loadings:
        # select columns to project on in lab data
        if "N70" in key:
            idx = idx +1
            cols_to_project_on = output_df.loc[output_df["sample"] == "N49", "Zfocus":"Zsource"]
            # compute dot product 
            projected_pattern = cols_to_project_on.dot(pattern.T)
            # add to projected scores dict
            lab_projected_scores_dict["projected_N70_to_N49_fac{}".format(idx)] = projected_pattern
        elif "N49" in key:
            idx = idx +1
            cols_to_project_on = output_df.loc[output_df["sample"] == "N70", "Zfocus":"Zsource"]
            # compute dot product 
            projected_pattern = cols_to_project_on.dot(pattern.T)
            # add to projected scores dict
            lab_projected_scores_dict["projected_N49_to_N70_fac{}".format(idx)] = projected_pattern

# covert dictionary to dataframe
lab_projected_df = pd.DataFrame.from_dict(lab_projected_scores_dict)

# add projected columns to output_df
output_df_with_projection = pd.concat([output_df, lab_projected_df], axis=1)

# save output_df_with_projection as csv to data folder
output_df_with_projection.to_csv("//mnt//c//Users//bront//Documents//PhD//Projects//lab_to_realworld//data//lab//with_pca//0.1_combined_esq_N119_74_cols_N70_N50_all_vid_rotation-on_ncomponents=3_with_projected.csv", index = False)

