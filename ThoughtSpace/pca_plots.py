#%%  Import libraries 

import os 

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from collections import OrderedDict
from adjustText import adjust_text

from wordcloud import WordCloud
import matplotlib.cm as cm
import matplotlib.colors as mcolor

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

# TO DO:
    # Add KMO + Bartletts as function
    # Add PCA extraction and rotation as function
    # Add stuff for different rotation types


def create_observation_id(df, idno_col):
    """
    Function to create join key (observation ID) and set as index
    """
    df['probenum'] = df.groupby(idno_col).cumcount()+1; df
    df['probenum'] = df['probenum'].astype(str)
    # also make idno col a string to the two columns can be added 
    df[idno_col] = df[idno_col].astype(str)
    # move probenum to start of dataframe
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df['observation_id'] = df[idno_col].apply(str) + '_' + df['probenum'].apply(str)
    return df.set_index('observation_id')

def esq_cols(df, start_col, end_col):
    """
    Function to find index of first and last ESQ columns.

    Parameters
    ----------
    df : dataframe
        User should provide dataframe. 
    start_col : string
        User should provide first ESQ column name.
    end_col : string
        User should provide last ESQ column name.

    Returns
    -------
    Index of firt and last ESQ columns.

    """
    col_start = df.columns.get_loc(start_col)
    col_end = df.columns.get_loc(end_col)+1
    return col_start, col_end

def z_score(df, col_start, col_end, by_sample = False, sample_col = None, 
            by_condition = False, condition_col = None,
            by_person = False, person_col = None):
    """
    Function to Z score given columns
    """
    # store column names which you want Z scoring
    cols = df.columns.tolist()[col_start:col_end]
    
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors = 'coerce')
        
        # if by_condition and by_sample set to true, apply z-score by condition and sample 
        if by_condition and by_sample:
            df['Z{}'.format(c)] = df.groupby([condition_col, sample_col])[c].transform(lambda x: zscore(x, ddof=1, nan_policy='omit'))
        # if by_condition set to true and by_sample set to false, apply z-score by condition only to all samples
        elif by_condition and not by_sample:
            df['Z{}'.format(c)] = df.groupby(condition_col)[c].transform(lambda x: zscore(x, ddof=1, nan_policy='omit'))
        # if by_sample set to true and by_condition set to false, apply z-score by sample only to all conditions
        elif by_sample and not by_condition:
            df['Z{}'.format(c)] = df.groupby(sample_col)[c].transform(lambda x: zscore(x, ddof=1, nan_policy='omit'))
        # if both set to false, apply Z-score to all 
        elif by_person:
            df['Z{}'.format(c)] = df.groupby([person_col])[c].transform(lambda x: zscore(x, ddof=1, nan_policy='omit'))
        else:
            df['Z{}'.format(c)] = zscore(df[c], nan_policy='omit')
    return df 

def naive_pca(input_dict):
    """
    Function to run naive PCA on input dictionary full of dataframes (Z-scored!)
    """
    pca_dict = OrderedDict() # create empty dict to store PCAs
    for k, v in input_dict.items():
        pca = PCA(svd_solver='full').fit(v)
        pca_dict[k] = pca
    return pca_dict 


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    """ 
    Applies varimax rotation (taken from wikipedia.)
    """
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape # gives the total number of rows and total columns of the matrix Phi
    R = eye(k) # Given a k*k identity matrix (gives 1 on diagonal and 0 elsewhere)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R) # Matrix multiplication
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda)))))) # Singular value decomposition svd
        R = dot(u,vh) # construct orthogonal matrix R
        d = sum(s) #Singular value sum
        if d/d_old < tol: break
        #if d_old!=0 and d/d_old < 1 + tol: break # https://www.programmersought.com/article/16054790260/
    return dot(Phi, R) # Return the rotation matrix Phi*R


def append_scores(score_dict, source_dict):
    """
    Function to add per-observation factor scores to original dataframes.
    This returns a new dataframe. 
    """
    new_dict = OrderedDict() 
    # Loop through keys (group/condition splits)
    for k in source_dict:
        # Get default indexing for concat to work
        source_dict[k] = source_dict[k].reset_index() 
        # Convert source dict from np array to pd dataframe
        score_dict[k] = pd.DataFrame(score_dict[k])
        # rename columns from '0','1' etc to Factor number and group/condition name
        for idx, i in enumerate(score_dict[k].columns.tolist()):
            score_dict[k] = score_dict[k].rename(columns={i:'{}_fac{}'.format(k ,idx+1)})
        # Only join dataframes if number of rows (observations) are the same
        if len(score_dict) == len(source_dict):
            df_conc = pd.concat([source_dict[k], score_dict[k]], axis=1)
            new_dict[k] = df_conc
        else:
            print ("Number of rows are not the same.")
    return new_dict

def merge_dataframes(input_dict, master_df, data_path, results_id, rotation_on, n_components, n_components_dict):

    """
    Function to add ALL the dataframes with PCA scores to OG DF
    """
    idx = 0 
    # loop through dicionary, ignoring first entry 
    for k,v in input_dict.items():
        idx += 1
        if idx == 1: 
            print ('idx:', idx)
            # Merge
            print ('Merging:',k)
            cols_to_use = input_dict[k].columns.difference(master_df.columns)
            merged_df = master_df.merge(input_dict[k][cols_to_use], on='observation_id', how='outer')
            print ('Shape of merged df:', merged_df.shape)
        else:
            print ('idx:', idx)
            print ('Merging:',k)


            # get only those columns who don't appearin both dataframes (otherwise '_x' and '_y' is added)
            cols_to_use = input_dict[k].columns.difference(merged_df.columns)

            # add observation_id back to this list as it was removed
            cols_to_use = cols_to_use.union(['observation_id'])

            #print ('INPUT COLS TO USE:', cols_to_use)
            merged_df = merged_df.merge(input_dict[k][cols_to_use], on='observation_id', how='outer')
            print ('Shape of merged df:', merged_df.shape)

    outputdf = merged_df 
    # TODO move saving data to out of function
    # TODO This fucks up if extension isn't .csv 
    # TODO this shouldn't output to the parent directory 
    # TODO want to save WHICH rotation has been applied in filename
    output_name = data_path.split('.csv')[0] + ('_{results_id}_{rotation}_ncomponents={ncomp}'.format(
        results_id = results_id,
        rotation=('rotation-on' if rotation_on else 'rotation-off'),
        # For number components, create list of all values in n_compoentns dictionary to show all possible number of compoennets enetered nby user  
        ncomp=(list(n_components_dict.values()) if n_components else 'EV>1'))) + '.csv'

    # TODO make this save line more robust - breaks for Will due to paths
    outputdf.to_csv(output_name, index = False)
    return outputdf


def page_of_plots(pca_dict,loadings_dict, masking_num, results_id, rotation_on, n_components, n_components_dict, display):
    """
    Function to create PDF of all figures for all groups
    """
    out_pdf = 'scratch/results/figures_{results_id}_{rotation}_ncomponents={ncomp}.pdf'.format(
        results_id = results_id,
        rotation=('rotation-on' if rotation_on else 'rotation-off'),
        ncomp=(list(n_components_dict.values()) if n_components else 'EV>1')
    )

    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)

    for k in pca_dict:
        # ONE PAGE: 
        fig, axes = plt.subplots(2,2, figsize=(8,8))
        fig.suptitle(k)

        # add subplot for scree plot
        screeplt = sns.lineplot(ax=axes[0,0],data = pca_dict[k].explained_variance_ratio_* 100, marker="o") 
        screeplt.set_xticks(range( len(pca_dict[k].explained_variance_ratio_)))
        screeplt.set_xticklabels(range(1, len(pca_dict[k].explained_variance_ratio_)+1))
        screeplt.set_title("Scree plot")
        screeplt.set_ylabel("explained variance (%)")

        # add subplot for cumloading
        cumplt = sns.lineplot(ax=axes[0,1],data = np.cumsum(pca_dict[k].explained_variance_ratio_) * 100,  marker="o")
        cumplt.set_xticks(range( len(pca_dict[k].explained_variance_ratio_)))
        cumplt.set_xticklabels(range(1, len(pca_dict[k].explained_variance_ratio_)+1))
        cumplt.set_title("Cumulated variance")
        cumplt.set_ylabel("explained variance (%)")

        # add subplot for loading plot
        loadplt = sns.scatterplot(ax=axes[1,0],x = loadings_dict[k][0, :], 
                            y = loadings_dict[k][1, :],
                            marker = ".")
        loadplt.set_xlabel("PC 1")
        loadplt.set_ylabel("PC 2")
        loadplt.set_title("Loading plot")
        
        x = loadings_dict[k][0, :]
        y = loadings_dict[k][1, :]

        texts = [loadplt.text(x[i],y[i], display[k][i], ha='center', va='center') for i in range(len(display[k]))] 
        adjust_text(texts,ax=axes[1,0])

        # add subplot for heatmap 
        mask = (loadings_dict[k].T < masking_num) & (loadings_dict[k].T > -masking_num)
        heatmp= sns.heatmap(ax=axes[1,1],data=loadings_dict[k].T, cmap="RdBu_r", vmax=0.7, vmin=-0.7, mask = mask)
        heatmp.set_xticks(np.arange(n_components_dict[k])+0.5)
        heatmp.set_xticklabels(range(1, n_components_dict[k] + 1))
        heatmp.set_yticks(np.arange(len(display[k]))+0.5)
        heatmp.set_yticklabels(display[k], rotation = 360)
        heatmp.set_title("Principle components")

        fig.tight_layout()
        # save PDF
        pdf.savefig(fig)
    pdf.close()

def project_patterns(component_loadings_dict, master_df_with_scores, component_loadings_dict_key, \
condition_projecting, condition_to_project_on, sample_to_project_on ,n_components, cond_col_str, sample_col_str,
esq_Zstart, esq_Zend):
    """
    Project patterns (dot product) from one PCA solution to a given set of obsrvations (rows)
    """
    # pattern to be projected 
    pattern = component_loadings_dict[component_loadings_dict_key] 
    # select rows 
    observations_to_project_on = \
    master_df_with_scores.loc[master_df_with_scores[cond_col_str] == condition_to_project_on]

    observations_to_project_on = \
    observations_to_project_on.loc[master_df_with_scores[sample_col_str] == sample_to_project_on] 
    # select columns
    loc = master_df_with_scores.columns.get_loc
    observations_to_project_on = \
    observations_to_project_on.iloc[:, np.r_[loc('observation_id'), loc(esq_Zstart):loc(esq_Zend)+1]]
    # compute dot product 
    observations_to_project_on.set_index('observation_id', inplace=True)
    projected_scores = observations_to_project_on.dot(pattern.T)
    
    
    projected_scores.columns = \
    [f"{condition_projecting}_pattern_on_{condition_to_project_on}-{sample_to_project_on}_fac{x}" for x  in range(1, n_components+1)]
    projected_scores.reset_index(inplace=True)
    print (projected_scores.columns.tolist())

    return projected_scores

#%%  Saving info for wordcloud script (temporary solution)
# TO DO: need to make saving out more flexible (add as argument?)
def wordclouder(component_loading_dict, display, savefile=False):
    """
    Function to return 1) wordclouds.pngs (saved by default) 2) .csvs containg colour codes & weightings used to make wordclouds 
    """
    for key, value in component_loading_dict.items(): # Loop over loading dictionaries - 1 dataframe per iteration
        df = pd.DataFrame(value.T) # transpose
        output_name = "scratch/results/{}_component_loadings_for_wordclouds.csv".format(key) 
        if savefile:
            df.to_csv(output_name, index = False, header = False)
        else:
            pass
        # could easily put the following into a function:
        principle_vector = np.array(df, dtype =float) # turn df into array
        pv_in_hex= []
        vmax = np.abs(principle_vector).max() #get the maximum absolute value in array
        vmin = -vmax #minimu 
        for i in range(principle_vector.shape[1]): # loop through each column (compoenent)
            rescale = (principle_vector  [:,i] - vmin) / (vmax - vmin) # rescale scores 
            colors_hex = []
            for c in cm.RdBu_r(rescale): 
                colors_hex.append(mcolor.to_hex(c)) # adds colour codes (hex) to list
            pv_in_hex.append(colors_hex) # add all colour codes for each item on all components 
        colors_hex = np.array(pv_in_hex ).T 
        df_v_color = pd.DataFrame(colors_hex)
        if savefile:
            df_v_color.to_csv("scratch/results/{}_colour_codes_for_wordclouds.csv".format(key), index = False, header = False)
        else:
            pass
        # loops over compoentn loadings
        for col_index in df:
            absolute = df[col_index].abs() # make absolute 
            integer = 100 * absolute # make interger 
            integer = integer.astype(int) 
            concat = pd.concat([integer, df_v_color[col_index]], axis=1) # concatanate loadings and colours 
            concat.columns = ['freq', 'colours']
            concat.insert(1, 'labels', display[key]) # add labels (items) from display list 
            if savefile:
                concat.to_csv("scratch/results/{}_loadings_and_colour_codes_factor_{}.csv".format(key, col_index+1), index = False, header = True)
            else:
                pass

            freq_dict = dict(zip(concat.labels, concat.freq)) # where key: item and value: weighting
            colour_dict = dict(zip(concat.labels, concat.colours))# where key: itemm and value: colour
            def color_func(word, *args, **kwargs): #colour function to supply to wordcloud function.. don't ask !
                try:
                    color = colour_dict[word]
                except KeyError:
                    color = '#000000' # black
                return color
            # create wordcloud object
            wc = WordCloud(background_color="white", color_func=color_func, 
                        width=400, height=400, prefer_horizontal=1, 
                        min_font_size=8, max_font_size=200
                        )
            # generate wordcloud from loadings in frequency dict
            wc = wc.generate_from_frequencies(freq_dict)
            wc.to_file('scratch/results/{}_wordcloud_factor_{}.png'.format(key, col_index+1))


def kmo_bartlett(esq_dict):
    """
    Input: 
    Return Chi squared and p value (bartlett) and KMO for each pca solution in a dictionary 
    """
    kmo_bartlett_dict = {} # store values here

    for k, v in esq_dict.items():
        chi_square_value,p_value=calculate_bartlett_sphericity(v)
        #print ("Chi-square value:", chi_square_value, "P-value:", p_value)
        kmo_all,kmo_model=calculate_kmo(v)
        #print ("KMO:", kmo_model)
        kmo_bartlett_dict[k] = {
            'chi_square_value':chi_square_value,
            'chi_squared_p':p_value,
            'KMO':kmo_model}
    return kmo_bartlett_dict
