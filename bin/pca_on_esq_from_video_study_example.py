"""
Created on Wed Mar 31 12:16:08 2021

@author: Bronte Mckeown & Will Strawson

Created to apply PCAs to different combinations of ESQ dataframes.
Input: Per-observation ESQ dataframe (.csv)
Output: Dataframe with PCA scores and plots per group all in one PDF.
"""

from pca_and_projection import pca_plots
import os
import pandas as pd
from collections import OrderedDict
import numpy as np

## EDIT A NEW COPY OF ME :) ##

# TODO: make it so if it crashes during saving it closes the file
# TODO: potentially move saving csv file out of merging function
# TODO: potentially migrate results out of scratch folder
# TODO: improve generalizability of projection section (functions?)
# TODO: figure out why factor column moves when adding scores to datasheet

#%% load data & basic cleaning/prep
# user variables
n_components = True  # if you know how many components you want to extract, insert True, otherwise false
# Specific number components can be defined for each solution in n_component_dict
ev_extraction = (
    False  # set to true if you want to extract components based on eigenvalue >1
)
varimax_on = True  # set to true if you want to apply varimax rotation

by_sample = True  # if you want to z-score ESQ by sample (e.g., N70 only or N49 only)
sample_col = "sample"  # set sample column name if z-scoring by sample

by_condition = False  # if you want to z-score by condition (e.g.,control, action, suspense)
condition_col = "condition"  # condition column name if z-scoring by condition

by_person = False  # if you want to z-score by person
person_col = "idno"  # set id number column


# set path of ESQ data
# current path should = parent repo directory
current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_name = "0.1_combined_esq_N119_74_cols.csv"
data_path = "scratch/data/" + data_name
# join git repo parent to relative path to data
data_path = os.path.join(current_path, data_path)
print("Input data path: ", data_path, "\n")

# read in ESQ data
df = pd.read_csv(data_path)  # Remember to specify the separtor (e.g. '\t' if not .csv)

# create join key for merging PCA results at the end- 'observation_id'
df = pca_plots.create_observation_id(df, "idno")

# find index of start and end of ESQ columns in master dataframe (df)
# provide strings of start and end ESQ column names
col_start, col_end = pca_plots.esq_cols(df, "focus", "source")

# zscore ESQ columns identified above using esq_cols function
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

# find index of Z scored ESQ columns by providing strings of start and end cols
col_start, col_end = pca_plots.esq_cols(df, "Zfocus", "Zsource")

# user should enter esq labels for display on plots & word clouds
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

df_N70 = df.loc[df["sample"] == "N70"].copy()  # just select N70 sample
df_N49 = df.loc[df["sample"] == "N49"].copy()  # just select N70 sample

# add info to file name outputs about specific analysis run
results_id = "N70_N49_zscored_by_sample"

# create ordered dict of dataframes which you want to apply PCA to
# key: name of the dataframe, value: dataframe
# If no group splits (i.e. one PCA on all data), simply create one entry in this dictionary

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
# create empty dict to store esq columns from each dataframe stored in df_dict
esq_dict = OrderedDict()
# create empty dict to store display labels from each dataframe stored in df_dict
display_dict = OrderedDict()

# loop over df_dict to assign esq cols to esq_dict & labels to labels_dict
for key, i in df_dict.items():
    i = i.iloc[:, col_start:col_end]  # should be z-scored ESQ questions
    i = i.apply(pd.to_numeric, errors="coerce") # make sure numeric
    i = i.dropna()  # drop rows with nan values
    esq_dict[key] = i
    # let display be a dictionary where values are display (list above)
    display_dict[key] = display
    # if different solutions have different display items, here is where they can be specified
    # remove Z from z scored column names for display
    # display_dict[key] = [x.replace("Z", "") for x in i.columns.tolist()]

#%% run naive PCA on all dataframes stored in esq_dict

# svd = full, meaning it calculates as many components as there are items
pca_dict = pca_plots.naive_pca(esq_dict)

#%% extract pca scores and component loadings from pca_dict

# create empty ordered dictionaries for storing un-rotated & rotated scores, loadings and n components
pca_scores = OrderedDict()
pca_component_loadings = OrderedDict()
pca_n_components = OrderedDict()

vari_pca_scores = OrderedDict()
vari_component_loadings = OrderedDict()
vari_n_components = OrderedDict()

for k in esq_dict:
    # if number of components is set to TRUE at start of script
    if n_components:
        # Generate per-observation scores for each factor
        scores = pca_dict[k].transform(esq_dict[k])[:, : n_components_dict[k]]
        # Generate per-item factor loadings
        pc = pca_dict[k].components_[: n_components_dict[k], :]

        # Store factor scores
        pca_scores[k] = scores
        # Store factor loadings
        pca_component_loadings[k] = pc
        # Store number of components
        pca_n_components[k] = n_components_dict[k]
        print("Shape of 'scores' dataframe (i.e. n_components):", scores.shape)
    
    # if ev_extraction is set to TRUE at start of script
    elif ev_extraction:
        evs = [i for i in pca_dict[k].explained_variance_ if i > 1]
        n_components = len(evs)
        # Generate per-observation scores for each factor
        scores = pca_dict[k].transform(esq_dict[k])[:, :n_components]
        # Generate per-item factor loadings
        pc = pca_dict[k].components_[:n_components, :]
        # Store factor scores
        pca_scores[k] = scores
        # Store factor loadings
        pca_component_loadings[k] = pc
        # Store number of components
        pca_n_components[k] = n_components
        print("Shape of 'scores' dataframe (i.e. n_components):", scores.shape)
        # reset n_components to None
        n_components = None

    else:
        scores = pca_dict[k].transform(esq_dict[k])
        pc = pca_dict[k].components_
        pca_scores[k] = scores
        pca_component_loadings[k] = pc
        pca_n_components[k] = scores.shape[-1]
        print("Shape of 'scores' dataframe (i.e. n_components):", scores.shape)
    
    # if varimax rotation is set to TRUE at start of script
    if varimax_on:
        # Generate per-item factor loadings
        vari_pc = pca_plots.varimax(pc.T).T  # what does pc =?
        # Generate per-observation scores for each factor
        vari_scores = np.dot(esq_dict[k], vari_pc.T)
        pca_varimax = "; varimax"
        vari_pca_scores[k] = vari_scores
        vari_component_loadings[k] = vari_pc
        vari_n_components[k] = scores.shape[-1]
        print(
            "[varimax] Shape of 'scores' dataframe (i.e. n_components):", scores.shape
        )

#%% Add PCA scores to master df

# adds pca scores to esq_dict dataframes for merging
if varimax_on:
    esq_dict_with_scores = pca_plots.append_scores(vari_pca_scores, esq_dict)
    print("Added varimax scores")
else:
    esq_dict_with_scores = pca_plots.append_scores(pca_scores, esq_dict)
    print("Added non-rotated scores")

# merge all and save
output_df = pca_plots.merge_dataframes(
    esq_dict_with_scores,
    df,
    data_path,
    results_id,
    varimax_on,
    n_components,
    n_components_dict,
)  

#%% Plots & word clouds for PCA solutions for each group (saves out PDF)
# set mask threshold for heatmaps
mask_threshold = 0
if varimax_on:
    pca_plots.page_of_plots(
        pca_dict,
        vari_component_loadings,
        mask_threshold,
        results_id,
        varimax_on,
        n_components,
        n_components_dict,
        display_dict,
    )
    pca_plots.wordclouder(vari_component_loadings, display_dict, savefile=False)
else:
    pca_plots.page_of_plots(
        pca_dict,
        pca_component_loadings,
        mask_threshold,
        results_id,
        varimax_on,
        n_components,
        n_components_dict,
        display_dict,
    )
    pca_plots.wordclouder(pca_component_loadings, display_dict, savefile=False)

# %% Project patterns between samples

# if not needed for your analysis, just comment out this section!

# create empty ordered dictionary to store projected scores
lab_projected_scores_dict = OrderedDict()

# loop over keys and values of varimax rotated component loadings
for key, loadings in vari_component_loadings.items():
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
output_df_with_projection.to_csv("//mnt//c//Users//bront//Documents//PhD//Projects//lab_to_realworld//pca_baby//scratch//data//lab//0.1_combined_esq_N119_74_cols_N70_N50_all_vid_varimax-on_ncomponents=3_with_projected.csv", index = False)

