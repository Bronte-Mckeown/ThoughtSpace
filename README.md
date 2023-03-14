# ThoughtSpace

Python-based pipeline for flexible Principal Components Analysis (PCA) and projecting between datasets.

## Prerequisite

There are a couple of ways you can implement ThoughtSpace in your own analysis. Pick the one that makes the most sense to you.

Option 1. You can simply add the following to your requirements.txt file in the directory where you'll be running the analysis: git+https://github.com/Bronte-Mckeown/ThoughtSpace

- If you do not know what a requirements.txt file is, see this link for help: https://www.javatpoint.com/how-to-create-requirements-txt-file-in-python#:~:text=In%20Python%20requirement.,Typically%20this%20file%20%22requirement.

Option 2. Fork this ThoughtSpace repository, and then 2) clone the forked repository.

- Forking creates your very own remote version of this repository and cloning creates a local version of that repository on your computer. If unsure, please google 'how to clone github repository'.

- Once you've cloned your forked version of this repository, you'll want to add it to the requirements.txt file in the directory where you'll be running the analysis.
You can do this by either adding an absolute path to your local copy of ThoughtSpace to requirements.txt, or by adding a link to your own forked version of this github repo.

## Set-up your virtual environment

First, you'll need to set up a virtual environment. This can be done using the terminal/command-prompt.
We will be using the language Bash. This is the default language in a terminal shell for both Mac and Linux. To use on Windows, use the Ubuntu app (if this is confusing, google 'how to use Bash on windows').

In your terminal, enter the following commands (line-by-line):

```

cd path/to/repo #change directory to the directory where you want to run analysis from 
virtualenv env -p python3 #create virtual environment using python 3
source env/bin/activate #activate this virtual environment
pip install -r requirements.txt #install packages in this text file (including ThoughtSpace)

```

## Repository structure

* Please note that ThoughtSpace is currently undergoing significant developments which will make it much more user-friendly so sit tight! *

### `ThoughtSpace/`

- In here are the necessary functions that make up the package 'ThoughtSpace'.
- These are called by your scripts when you 'import ThoughtSpace' in your scripts.
- No need to edit anything in here - unless, of course, you want to contribute a function. In which case, submit a Pull Request.

### `bin/`

- Here is the example script that calls functions in ThoughtSpace to perform the analysis.
- Save a copy of the '...example.py' script from the main branch into your own analysis-specific directory and edit accordingly for your own analysis.

### `scratch/`

- This is a place to keep stuff you're not too precious about
- You can create a `data/` directory, in which you can store the raw data you want PCA to be performed on and save out datasets with PCA scores.
- It also has a `results/` directory where, you guessed it, your results (e.g., figures) will go (as outputted by the script in `bin/`)
- In future versions of ThoughtSpace, there will be more flexibility in where your results are saved but for now they will always be in scratch/results

---

## How to run the PCA analysis (using `bin/...example.py`)

#### Aim: 1) To caluculate per-observation PCA for each component of each PCA across different group splits (if you have group splits) 2) To produce screeplots, cumulative variance plots, heatmaps and wordclouds for each

_Open the example.py script in bin/, and save a copy. Edit this copy only. Keep the example.py script for reference._
_This example script is specific to a particular dataset so variables will need editing_
_Note: Not every line in the example.py script is needed to run this analysis - If your data has no group/condition splits, for example, you can ignore some of the script_

1. Change the variables at the top of the script to suit your needs

- e.g., How many components do you want extracted? Which rotation do you want to apply? Do you have separate groups? Separate conditions?

2. Input your dataframe (.csv, .tsv)
   - Data should be structured so that there is one row per observation
3. Create observation ID (unique to every row)
4. Select which columns contain thought data
   - Z scoring is only applied to these columns
5. Z-score data in these columns
   - Can be applied to whole dataframe, or split by condition/sample/subject (set this in step one, Optional).
6. Select which columns contain z-scored thought data

- PCA will only be applied to these columns

7. Input labels for figures in list
8. Select groups for applying PCA and create separate dataframes for each split (Optional)
   - These are stored in a dictionary
9. Set number of components wanted for each group
10. Set 'results id' string for file outputs
11. Calculate KMO, Bartletts and naive PCA (no input needed)

- This inital step results in as many components as there are thought columns.

12. Data extraction & rotation
    - Nothing needs editing here - all parameters should have already been set at the top of the script in step 1.
11. Append factor scores back onto the original dataframe (No editing necessary)
    - Now, added to your dataframe, you will have one PCA score per component per row.
12. Save out dataframe and associated results
    - In scratch/ you should have 1) word clouds 2) loadings of each thought item on each component (if `savefile==True` in `wordclouder` function) 3) a PDF summary and 4) a new version of your original dataframe with PCA scores appended
13. Option projection of patterns
    - Currrently, this is just done using a for loop in the example script but there is also a function available in ThoughtSpace/pca_plots that does this more flexibly
    - Will update soon as improvements are coming!
