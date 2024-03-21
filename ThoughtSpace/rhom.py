from sklearn.base import BaseEstimator
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from itertools import combinations
from itertools import permutations
from itertools import product

from _base import basePCA

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import copy
from factor_analyzer.rotator import Rotator

from tensorly.metrics.factors import congruence_coefficient
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes

import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
from scipy.stats import t, norm


def crazyshuffle(arr):
    arr = arr.loc[:, 'focus':].values
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    out = arr[rows, cols]
    return out

def bootstrap(estimator, X, y, group=None, cv=None, omnibus = False, splithalf = False, pro_cong = False, bypc = False, shuffle = False, fit_params={}):
    if shuffle:
        firstcols_X = X.loc[:, group]

        Xdata = crazyshuffle(X)
        Xdata = pd.DataFrame(Xdata)
        X = pd.concat([firstcols_X, Xdata], axis = 1)

    avg_score = []
    exp_var = []

    if pro_cong:
        avg_phi = []
        
    if not omnibus:
        if bypc:
            complist = []
            for i in range(estimator.n_comp):
                complist.append([])
            if pro_cong:
                philist = []
                for i in range(estimator.n_comp):
                    philist.append([])
            for x1, x2 in cv.bypc_split(X,y):
                
                ests = gen_ests(estimator, x1, x2, pro_cong = pro_cong)
                if pro_cong:
                    for i in range(len(philist)):
                        complist[i].append(ests[0][i])
                        philist[i].append(ests[1][i])
                else:
                    for i in range(len(complist)):
                        complist[i].append(ests[i])

        elif splithalf:
        
            for x1, x2 in cv.redists(df=X, subset=y): 
                ests = gen_ests(estimator, x1, x2, pro_cong=pro_cong)
                if pro_cong:
                    avg_score.append(ests[0])
                    avg_phi.append(ests[1])
                else:
                    avg_score.append(ests)              

        else:
            for x1, x2 in cv.split(X,y):

                ests = gen_ests(estimator, x1, x2, pro_cong = pro_cong)
                if pro_cong:
                    avg_score.append(ests[0])
                    avg_phi.append(ests[1])
                else:
                    avg_score.append(ests)
    
    else:

        for x1, x2 in cv.redists(df=X, subset=y):

            ests = gen_ests(estimator, x1, x2, pro_cong=pro_cong)
            if pro_cong:
                avg_score.append(ests[0])
                avg_phi.append(ests[1])
            else:
                avg_score.append(ests)

    if bypc:
        if pro_cong:
            return [complist, philist]
        
        return complist
    
    if pro_cong:
        return [avg_score, avg_phi, exp_var]

    return avg_score

def gen_ests(estimator, x1, x2, pro_cong = False, fit_params={}):

    estimator.fit(x1,x2,**fit_params)
    preds = estimator.predict()
    corrs = np.corrcoef(preds[0], preds[1], rowvar=False)
    rhm = estimator.hom_pairs(corrs)
    if pro_cong:
        phi = estimator.pro_cong()
        return [rhm, phi]

    return rhm

def tcc(fac1=None, fac2=None):

    #Lovik et al.(2020): Using the absolute value of the numerator is more suitable for factor matching
    numerator = np.sum(abs(fac1*fac2))
    denominator = np.sqrt(np.sum(fac1**2) * np.sum(fac2**2))
    return numerator / denominator

class rhom(BaseEstimator):
    
    def __init__(self, rd = None, n_comp=4, rotate = True, method="varimax", bypc = False):
        self.rd = rd
        self.n_comp = n_comp
        self.rotate = rotate
        self.method = method
        self.bypc = bypc
    
    def get_params(self,deep=True):
        return (copy.deepcopy(
            {"rd":self.rd})
            if deep
            else {"rd":self.rd})
        
    def fit(self, x, y=None,xidx=None,yidx=None, ncv=0):
        # self.model_x = PCA(n_components=self.n_comp)
        # self.model_x.fit(x)

        if self.n_comp >= 2:
            self.model_x = basePCA(n_components=self.n_comp, rot_method=self.method)
            self.model_x.fit(pd.DataFrame(x))
        else:
            self.model_x = basePCA(n_components=self.n_comp, rot_method="none")

        # self.model_x = basePCA(n_components=self.n_comp, rot_method='none')
        # self.model_x.fit(x)

        # self.model_x2 = PCA(n_components=self.n_comp)
        # self.model_x2.fit(y)

        self.model_x2 = basePCA(n_components=self.n_comp, rot_method='none')
        self.model_x2.fit(pd.DataFrame(y))

    def predict(self, y=None):
        y = self.rd
        if self.rotate :
            results = []

            # for model in [self.model_x,self.model_x2]:
            #     rot = Rotator(method=method)
            #     loadings = rot.fit_transform(model.components_.T)
                
            #     self.results = np.dot(y, loadings)
            #     results.append(self.results)
            # if self.n_comp >= 2:
            #     rot = Rotator(method=self.method)
            #     # loadings = rot.fit_transform(self.model_x.components_.T)
            #     loadings = rot.fit_transform(self.model_x.loadings.to_numpy())
            # else:
            #     # loadings = self.model_x.components_.T
            #     loadings = self.model_x.loadings.to_numpy()

            # R, s = orthogonal_procrustes(loadings, self.model_x2.components_.T)
            # loadings_x2 = np.dot(self.model_x2.components_.T, R.T) * s

            loadings = self.model_x.loadings.to_numpy()

            R, s = orthogonal_procrustes(loadings, self.model_x2.loadings.to_numpy())
            loadings_x2 = np.dot(self.model_x2.loadings.to_numpy(), R.T) * s

            for loads in [loadings, loadings_x2]:
                self.results = np.dot(y, loads)
                results.append(self.results)

        else:
            results = []
            for model in [self.model_x,self.model_x2]:
                preds = model.transform(self.rd)
                results.append(preds)
    
        return results

    def score(self,x,y):
        return self.model.score(x,y)

    def hom_pairs(self,cor_matrix):
        cor_matrix = cor_matrix[-self.n_comp:,:self.n_comp]
        cor_matrix = np.abs(cor_matrix)

        x = np.argmax(cor_matrix, axis = 0)
        x = [[en,a] for en,a in enumerate(x)]
        x2 = np.argmax(cor_matrix, axis = 1)
        x2 = [[a,en] for en,a in enumerate(x2)]
        x2 = sorted(x2, key = lambda x: x[0])

        if x != x2:
            idx = list(range(self.n_comp))
            sols = list(permutations(idx, r=self.n_comp))
            soldict = {}
            for perm in sols:
                plis = []
                outls = []
                for e, cell in enumerate(perm):
                    plis.append([e,cell])

                    outs = cor_matrix[e,cell]
                    outls.append(outs)
                out = np.mean(outls)
                soldict[out] = plis
            
            bestval = max(list(soldict.keys()))

            if self.bypc:
                bestorg = soldict[bestval]
                bestorg = sorted(bestorg, key = lambda x: x[0])    
                rhoms = [cor_matrix.T[z[0],z[1]] for z in bestorg]              
                return rhoms
            else:
                return bestval

        else:
            rhoms = [cor_matrix.T[z[0],z[1]] for z in x]
            if self.bypc:
                return rhoms
            else:
                return np.mean(rhoms)

    def pro_cong(self):
        loadings_X = self.model_x.loadings.to_numpy()
        loadings_x2 = self.model_x2.loadings.to_numpy()
        
        R, s = orthogonal_procrustes(loadings_X, loadings_x2)
        loadings_x2 = np.dot(loadings_x2, R.T) * s

        tcc_matrix = []
        tcclist = []
        for i in range(len(loadings_X.T)):
            for j in range(len(loadings_x2.T)):
                phi = tcc(loadings_X.T[i], loadings_x2.T[j])
                tcclist.append(phi)
            tcc_matrix.append(tcclist)
            tcclist =[]
        
        tcc_matrix = np.asarray(tcc_matrix)
        phi = self.hom_pairs(tcc_matrix)
        return phi

class pair_cv():

    def __init__(self, k=5, n=1000, boot=False, omnibus=False, group=None) :
        self.n_splits = k
        self.n_redists = n
        self.boot = boot
        self.omnibus = omnibus
        self.group = group

    def divide_chunks(self, l, c) :
        n = len(l)//c
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def assignModel(self, df=None) :
        rows = df.shape[0]
        df['o/s'] = "omnibus"
        df['o/s'][0:int(rows/2)] = "sample"
        df['o/s'] = np.random.permutation(df['o/s'].values)
        return df

    def standardize(self, df=None) :
        scaler = StandardScaler()
        dft = scaler.fit_transform(df)
        df = pd.DataFrame(dft,index=df.index,columns=df.columns)
        return df

    def omni_prep(self, df=None) :
        samples = df[self.group].unique()
        subsamps = {}
        for sample in samples:
            subsamps[str(sample)] = df[df[self.group] == sample]

        models = {"omnibus":pd.DataFrame()}
        for subsamp in subsamps:
            model = subsamps[subsamp]
            model = self.assignModel(model)

            models[subsamp] = model[model["o/s"] == "sample"]
            models[subsamp] = models[subsamp].drop(labels = [self.group, "o/s"], axis = 1)
            models[subsamp] = self.standardize(models[subsamp])

            models['omnibus'] = models['omnibus'].append(model[model["o/s"] == "omnibus"])

        models['omnibus'] = models['omnibus'].drop(labels = [self.group, "o/s"], axis = 1)
        models['omnibus'] = self.standardize(models['omnibus'])
        
        return models

    def omni_prep_mini(self, df=None, subsamps=None, subset=None) :
        # samples = df[idcol].unique()
        # subsamps = {}
        # for sample in samples:
        #     if sample != subset :
        #         subsamps[sample] = df[df[idcol] == sample]

        model = df[df[self.group] == subset]
        model = self.assignModel(model)

        models = {"omnibus":model[model["o/s"] == "omnibus"]}

        models[subset] = model[model["o/s"] == "sample"]
        models[subset] = models[subset].drop(labels = [self.group, "o/s"], axis = 1)
        models[subset] = self.standardize(models[subset])

        for subsamp in subsamps:
            model = subsamps[subsamp]
            model = self.assignModel(model)

            models['omnibus'] = models['omnibus'].append(model[model["o/s"] == "omnibus"])

        models['omnibus'] = models['omnibus'].drop(labels = [self.group, "o/s"], axis = 1)
        models['omnibus'] = self.standardize(models['omnibus'])

        return models

    def split(self, X, y, groups=None) :
        foldidx = list(range(self.n_splits))
        try:
            X = X.values
            y = y.values

        except:
            pass

        np.random.shuffle(X)
        np.random.shuffle(y)
        x1_c = list(self.divide_chunks(X, self.n_splits))
        x2_c = list(self.divide_chunks(y, self.n_splits))

        folds = []
        if not self.boot:
            for fold in product(foldidx, repeat=2):
                folds.append(fold)
            folds = [[x1_c[z],x2_c[q]] for z,q in folds]

        elif self.boot:
            for z in range(1,self.n_splits+1):
                for fold in combinations(foldidx, r=z):
                    folds.append(fold)
                
            boot_folds = []
            for fold in product(folds, repeat=2):
                boot_folds.append(fold)

            Xs1 = [list(x[0]) for x in boot_folds]
            Xs2 = [list(x[1]) for x in boot_folds]

            fold_chks = []
            folds1 = []
            for z in Xs1:
                fold_chks.append([x1_c[v] for v in z])
                folds1.append(np.concatenate((fold_chks[0]), axis=0).squeeze())
                fold_chks =[]

            fold_chks = []
            folds2 = []
            for z in Xs2:
                fold_chks.append([x2_c[v] for v in z])
                folds2.append(np.concatenate((fold_chks[0]), axis=0).squeeze())
                fold_chks =[]    
            folds = [list(z) for z in zip(folds1,folds2)]
        
        for x1,x2 in folds:
            yield x1,x2

    def bypc_split(self, X, y, groups=None):
        foldidx = list(range(self.n_splits))
        try:
            X = X.values
            y = y.values

        except:
            pass

        np.random.shuffle(y)
        x_c = list(self.divide_chunks(y, self.n_splits))
        folds = []
        if not self.boot:
            for fold in product(foldidx, repeat=2):
                folds.append(fold)
                folds = [[X,x_c[z]] for z in folds]

        elif self.boot:
            for z in range(1,self.n_splits+1):
                for fold in combinations(foldidx, r=z):
                    folds.append(fold)
                
            boot_folds = []
            for fold in product(folds, repeat=2):
                boot_folds.append(fold)

            Xs = [list(x[1]) for x in boot_folds]

            fold_chks = []
            folds = []
            for z in Xs:
                fold_chks.append([x_c[v] for v in z])
                folds.append([X, np.concatenate((fold_chks[0]), axis=0).squeeze()])
                fold_chks =[]
        
        for x1,x2 in folds:
            yield x1,x2
  
    def split_half(self, df=None) :
        rows = df.shape[0]
        df['subset'] = 2
        df['subset'][0:int(rows/2)] = 1

        return df
        
    def split_frame(self, df) :

        models = {'1':pd.DataFrame(), '2':pd.DataFrame()}

        for model in models:
            models[model] = df[df["subset"] == int(model)]
            models[model] = models[model].drop(labels = ["subset"], axis = 1)
            models[model] = self.standardize(models[model])

        df.drop(labels = ["subset"], axis = 1)
        
        return models
    
    def redists(self, df=None, subset=None) :
        redists = []
        if self.omnibus:
            samples = df[self.group].unique()
            subsamps = {}
            for sample in samples:
                if sample != subset :
                    subsamps[sample] = df[df[self.group] == sample]

            for i in range(self.n_redists):
                redist = self.omni_prep_mini(df=df, subset=subset, subsamps=subsamps)
                redistv = list(redist.values())
                redists.append(redistv)
        
        else:
            if self.group != None:
                splitdf = df[df[self.group] == subset].copy()
                splitdf = splitdf.drop(labels="dataset", axis=1)
                splitdf = self.split_half(splitdf.copy())
            else:
                splitdf = self.split_half(df.copy())

            for i in range(self.n_redists):
                splitdf['subset'] = np.random.permutation(splitdf['subset'].values)
                redist = self.split_frame(splitdf)
                redistv = list(redist.values())
                redists.append(redistv)  

        return redists    

def splithalf(df=None, group=None, npc=None, rotation='varimax', save=True, file_prefix=randint(10000,99999), display=False, shuffle=False) :
    '''
    Split-Half Reliability
    ----------------------
    This function conducts a bootstrapped split-half reliability analysis
    on your dataframe. It can do so on a full dataset, or at each level of a
    grouping variable. It simply bootstrap reassigns random halves of the data
    into two subsets and computes their component similarity based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

     Parameters:
    -----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.
        
        save: bool, default=True
            Save outputted split-half reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns:
    --------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    if group != None :
        df_t = df.drop(labels = [group], axis = 1)
        samples = df[group].unique()

    else :
        df_t = df
        samples = ['fulldata']

    if rotation != rotation :
        boot_model = rhom(rd = copy.deepcopy(df_t.values), n_comp = npc, rotate=False)
    else :
        boot_model = rhom(rd = copy.deepcopy(df_t.values), n_comp = npc, method=rotation)
        
    cv = pair_cv(group=group)

    split_df = pd.DataFrame(columns=['n_comp', group, 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

    for i in range(len(samples)):
        resultslist = bootstrap(boot_model, df, samples[i], cv=cv, splithalf = True, pro_cong=True)

        print('Running ' + str(samples[i]))
        #scaler = StandardScaler()
        #s_results = scaler.fit_transform(np.array(results).reshape(-1, 1)).squeeze()
        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])
 
        stats_dict = {}
        stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
        stats_dict[group] = str(samples[i])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        split_df = split_df.append(stats_dict, ignore_index = True)

        if display:

            print("Split-Half Reliability for " + str(samples[i]) + ": ")
            print("*"*20)
            for results in resultslist:
                if results == resultslist[0]:
                    print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
                else:
                    print(f"Mean Factor Congruence: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")

            print("*"*40)

    if save:

        split_df.to_csv('results/' + str(file_prefix) + '_' + str(len(df.columns)) + 'D_' + str(boot_model.n_comp) + 'PC.csv', index = False)
        print('dataframe saved')

    return split_df  

def dir_proj(df=None, group=None, npc=None, rotation="varimax", save=True, plot=True, display=False, shuffle = False, file_prefix=randint(10000,99999)):    
    '''
    Direct-Projection Reproducibility
    ---------------------------------
    This function conducts a bootstrapped direct-projection analysis
    on your data based on some inputted grouping variable. This involves
    dividing each group into its own dataframe, and assessing the similarity
    of the components generated by each group to each other group based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)
    
    Parameters:
    -----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.
        
        save: bool, default=True
            Save outputted reproducibility results to .csv.

        plot: bool, default=True
            Visualise results with heatmaps.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns:
    --------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
    
        .csv:
            If save=True, will save a .csv file to /results.

        .png:
            If plot=True, will save heatmaps for loading similarity and component-score similarity.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    groups = df[group].unique()

    #create a data frame dictionary to store your data frames
    maindict = {elem : pd.DataFrame() for elem in groups}

    for key in maindict.keys():
        maindict[key] = df[:][df[group] == key]
        maindict[key] = maindict[key].drop(labels=group, axis=1)

    dalist = groups
    dagoodlist = list(combinations(dalist, 2))

    scaler = StandardScaler()
    df = df.drop(labels=group, axis=1)
    dft = scaler.fit_transform(df)
    df = pd.DataFrame(dft,index=df.index,columns=df.columns)

    dirproj_df = pd.DataFrame(columns=['n_comp', 'referent', 'comparator', 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    if plot:
        dirproj_mtx = pd.DataFrame(columns=dalist, index=dalist)
        dirproj_phi = pd.DataFrame(columns=dalist, index=dalist)

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

    boot_model = rhom(rd = copy.deepcopy(df.values), n_comp = npc, method=rotation)
    cv = pair_cv(boot=True)
    for currentset in dagoodlist:
        print("Running " + str(currentset[0]) + " x " + str(currentset[1]))
        resultslist = bootstrap(boot_model, maindict[currentset[0]],maindict[currentset[1]], cv=cv, pro_cong = True)
        #scaler = StandardScaler()
        #s_results = scaler.fit_transform(np.array(results).reshape(-1, 1)).squeeze()
        print("Saving " + str(currentset[0]) + " x " + str(currentset[1]))
        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])
    
            
        stats_dict = {}

        stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
        stats_dict['referent'] = str(currentset[0])
        stats_dict['comparator'] = str(currentset[1])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        dirproj_df = dirproj_df.append(stats_dict, ignore_index = True)

        if plot:

            dirproj_mtx.loc[str(currentset[0]), str(currentset[1])] = rhm_x
            dirproj_mtx.loc[str(currentset[1]), str(currentset[0])] = rhm_x

            dirproj_phi.loc[str(currentset[0]), str(currentset[1])] = phi_x
            dirproj_phi.loc[str(currentset[1]), str(currentset[0])] = phi_x
        
        if display:

            print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
            print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")

            print("*"*40)

    if plot:
        dirproj_mtx = dirproj_mtx.fillna(1)
        dirproj_phi = dirproj_phi.fillna(1)

        plt.close()

        sns.heatmap(dirproj_mtx,
                    vmin = dirproj_mtx.values.min(),
                    annot = True,
                    annot_kws={"fontsize": 35 / np.sqrt(len(dirproj_mtx))},
                    cmap = "flare")
        plt.suptitle('Mean Homologue Similarity', fontsize=16)
        plt.savefig('results/' + str(file_prefix) + '_heatmap' + str(len(df.columns)) + "D(alt)_" + str(boot_model.n_comp) + 'PC_rhm.png')
        plt.show()
        plt.close()


        sns.heatmap(dirproj_phi,
                    vmin = dirproj_phi.values.min(),
                    annot = True,
                    annot_kws={"size": 35 / np.sqrt(len(dirproj_phi))},
                    cmap = "flare")
        plt.suptitle('Mean Factor Congruence', fontsize=16)
        plt.savefig('results/'  + str(file_prefix) + '_heatmap' + str(len(df.columns)) + "D(alt)_" + str(boot_model.n_comp) + 'PC_phi.png')
        plt.show()
        plt.close()

    if save:
        dirproj_df.to_csv('results/' + str(file_prefix) + '_dj' + str(len(df.columns)) + "D(alt)_" + str(boot_model.n_comp) + 'PC.csv', index = False)

    return dirproj_df

def omni_sample(df=None, group=None, npc=None, rotation="varimax", save=True, display=False, shuffle = False, file_prefix=randint(10000,99999)):
    '''
    Omnibus-Sample Reproducibility
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data.
    It randomly bootstrap reassigns halves of each level of an inputted grouping variable 
    to be used in either a 'sample' or 'omnibus' subset. The 'sample' subsets generate
    components representative of that level of the grouping variable, while the 'omnibus'
    subsets are aggregated with other groups to produce 'common' components. The analysis
    assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters:
    -----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns:
    --------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    samples = df[group].unique()

    df_t = df.drop(labels = [group], axis = 1)

    omsamp_df = pd.DataFrame(columns=['n_comp', group, 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

    if rotation != rotation :
        boot_model = rhom(rd = copy.deepcopy(df_t.values), n_comp = npc, rotate = False)
    else :
        boot_model = rhom(rd = copy.deepcopy(df_t.values), n_comp = npc, method=rotation)
        
    cv = pair_cv(omnibus = True, group=group)
    totalRhm = []
    totalPhi = []
    for i in range(len(samples)):
        print("Running omnibus x " + str(samples[i]))
        resultslist = bootstrap(boot_model, df, samples[i], cv=cv, omnibus=True, pro_cong=True, shuffle = shuffle)

        print("Saving omnibus x " + str(samples[i]))
        totalRhm.append(resultslist[0])
        totalPhi.append(resultslist[1])

        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])

        stats_dict = {}

        stats_dict['n_comp'] = int(boot_model.n_comp)
        stats_dict[group] = str(samples[i])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        omsamp_df = omsamp_df.append(stats_dict, ignore_index = True)

        if display:
            print('Omnibus x ' + str(samples[i]) + ':')
            print("*"*20)
            print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
            print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")
            print("*"*40)

    print('Saving overall results')
    rhm_ci, rhm_x, rhm_sd = getstats(totalRhm)
    phi_ci, phi_x, phi_sd = getstats(totalPhi)

    if display:
        print("*"*20)
        print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
        print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")
        print("*"*40)

    stats_dict = {}

    stats_dict['n_comp'] = int(boot_model.n_comp)
    stats_dict['dataset'] = "Total"

    stats_dict['rhm_x'] = rhm_x
    stats_dict['rhm_sd'] = rhm_sd
    stats_dict['rhm_LCI'] = rhm_ci[0]
    stats_dict['rhm_UCI'] = rhm_ci[1]

    stats_dict['phi_x'] = phi_x
    stats_dict['phi_sd'] = phi_sd
    stats_dict['phi_LCI'] = phi_ci[0]
    stats_dict['phi_UCI'] = phi_ci[1]

    omsamp_df = omsamp_df.append(stats_dict, ignore_index = True)

    if save:

        omsamp_df.to_csv('results/' + str(file_prefix) + '_omsamp_' + str(len(df_t.columns)) + "D_" + str(boot_model.n_comp) + 'PC.csv', index = False)

    return omsamp_df

def bypc(df=None, group=None, npc=None, rotation="varimax", save=True, plot=True, display=False, shuffle = False, file_prefix=randint(10000,99999)):
    '''
    Omnibus-Sample Reproducibility: By-Component
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data,
    modified to assess the correspondence between each component of an omnibus solution and its
    corresponding components in each subset. It randomly reassigns halves of each level 
    of an inputted grouping variable to be stably used in either a 'sample' or 'omnibus' subset. 
    The 'sample' subsets are folded to generate cross-validated components representative of that level of
    the grouping variable, while the 'omnibus' subset is aggregated with other groups to produce 'common' components.
    The analysis assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters:
    -----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        plot: bool, default=True
            Save wordclouds, a Scree plot, and .csv files for the specified omnibus set.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns:
    --------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        thoughtspace PCA results:
            If plot=True, will save the results, including wordclouds and Scree plot, for the omnibus set in a thoughtspace folder.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    df_t = df.drop(labels=group, axis=1)
    
    boot_model = rhom(rd = copy.deepcopy(df_t.values), bypc=True, n_comp=npc, method=rotation)
    cv = pair_cv(boot = True, group=group)

    maindict = cv.omni_prep(df = df)

    samples = df[group].unique()
    dagoodlist=[]
    for i in range(len(samples)):
        sampset = ['omnibus', samples[i]]
        dagoodlist.append(sampset)

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

    stats_bypc = pd.DataFrame(columns=['n_comp', 'dataset', 'comp', 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    for currentset in dagoodlist:
        comps = bootstrap(boot_model, maindict[currentset[0]], maindict[currentset[1]], cv=cv, bypc = True, pro_cong = True)

        print("Running " + str(currentset[0]) + " x " + str(currentset[1]))
        # print("*" * 20)
        for i in range(len(comps[0])):
            stats_dict = {}
            rhm_ci, rhm_x, rhm_sd = getstats(comps[0][i])
            phi_ci, phi_x, phi_sd = getstats(comps[1][i])

            stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
            stats_dict['dataset'] = str(currentset[1])
            stats_dict['comp'] = i + 1

            stats_dict['rhm_x'] = rhm_x
            stats_dict['rhm_sd'] = rhm_sd
            stats_dict['rhm_LCI'] = rhm_ci[0]
            stats_dict['rhm_UCI'] = rhm_ci[1]

            stats_dict['phi_x'] = phi_x
            stats_dict['phi_sd'] = phi_sd
            stats_dict['phi_LCI'] = phi_ci[0]
            stats_dict['phi_UCI'] = phi_ci[1]

            stats_bypc = stats_bypc.append(stats_dict, ignore_index=True)

            if display:
                print(f"Component {i + 1}:")
                print(f"Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g},{rhm_ci[1]:.3g}]")
                print(f"Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g},{phi_ci[1]:.3g}]")
                print("*"*40)

    if plot:
        model = basePCA(n_components=npc, rot_method=rotation)
        model.fit(maindict['omnibus'])
        model.fit_project(maindict['omnibus'])
        model.save(path="results", pathprefix=str(file_prefix)+"_os")

    if save:
        stats_bypc.to_csv('results/' + str(file_prefix) + '_bypc_' + str(len(df_t.columns)) + 'D(alt)_' + str(boot_model.n_comp) + 'PC.csv', index = False)

    return stats_bypc