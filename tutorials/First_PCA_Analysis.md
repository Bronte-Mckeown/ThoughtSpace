## Running your first PCA analysis

We are now ready to run your first ThoughtSpace analysis!

**1. Open visual studio code**

**2. Create new python file in your analysis directory (see above for details on how to do this)**
    
**3. Type the following and save the file:**

In the example pasted below, change the file path to your own csv file stored in your own data directory or alternatively, use the URL method pasted below to use the example data available on ThoughtSpace.

Using URL method to read in example data (daily life experience sampling data before and during lockdown in the UK): 

```python
import pandas as pd # for reading in the csv file containg the data
from ThoughtSpace.pca import basePCA # this imports the basePCA class from ThoughtSpace

# read in data
url = 'https://github.com/Bronte-Mckeown/ThoughtSpace/tree/master/scratch/data/example_data.csv'
df = pd.read_csv(url)

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and transform data to create PCA scores
# pca_scores now contains dataframe with original data plus PCA columns
pca_scores = model.fit_transform(df)

# save results to results folder using .save
model.save(path="results",pathprefix="PCA_results")
```

Using your own data:

Things to note about the data format required:
- PCA will be applied to any numerical columns provided.
- So make sure any columns you don't want included, such as an 'ID number' column or an 'age' column, are set to string variables, and that variables you do want included (i.e., experience sampling items) are set to numerical variables.

```python
import pandas as pd # for reading in your csv
from ThoughtSpace.pca import basePCA # to use ThoughtSpace

# read in data
df = pd.read_csv("data/lockdown_data.csv")

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and transform data to create PCA scores
# pca_scores now contains dataframe with original data plus PCA columns
pca_scores = model.fit_transform(df)

# save results to results folder using .save
model.save(path="results",pathprefix="PCA_results")
```

**4. Before running this file, select the conda environment you have created which has ThoughtSpace installed in.**

You now need to select the correct Python interpreter by opening the Command Palette (Ctrl+Shift+P), typing "Python: Select Interpreter," and choosing the interpreter from the list of detected Python interpreters.

It will be called whatever you called the conda environment created above (e.g., "covid").

**5. Open up your Github repository/ analysis folder in the VS code explorer**

- Click on "File" in the top menu.
- Select "Open Folder..." from the dropdown.
- Browse your file system and select the folder you want to open. This will be your analysis project folder.
- This step means that relative paths will work (i.e., you can just type "results" to access results folder instead of needing to include "Users..." etc in the file path)

**6. Run Python script using arrow in top right corner.**
    
The print out in the terminal will tell you about the KMO and Bartlett's Test of Sphericity:
    
- *Kaiser-Meyer-Olkin (KMO) Measure*: KMO is a statistic that measures the adequacy of the sample size for conducting factor analysis. It assesses the proportion of variance among variables that might be common variance. In simpler terms, KMO helps you determine if your data is suitable for PCA. A high KMO value (usually above 0.6) suggests that the data is appropriate for factor analysis, indicating that the variables are related enough for PCA to yield meaningful results.

- *Bartlett's Test of Sphericity*: Bartlett's test checks whether the variables in your dataset are correlated, indicating that they might be suitable for PCA. It tests the null hypothesis that the correlation matrix is an identity matrix (which would mean no correlation between variables). In the context of PCA, you want to reject this null hypothesis because PCA is based on the idea that variables are correlated. If Bartlett's test is statistically significant (i.e., the p-value is below a chosen significance level, such as 0.05), it suggests that your data is appropriate for PCA.

**7. Check your results!**

They will be stored in your results folder.

In the results folder, there will be the following sub-directories:
    - csvdata
    - descriptives
    - screeplots
    - wordclouds

In the csvdata directory, you will find:

- pca_loadings.csv: rows = experience sampling items, columns = pca components, values = component loadings
- fitted_pca_scores.csv: data PCA was trained on + PCA scores
- projected_pca_scores.csv: data PCA was applied to + PCA scores
- full_pca_scores.csv: data PCA was trained on plus data PCA was applied to + PCA scores
- pca_scores_original_format.csv: original dataframe PCA was trained on (including string columns not included in PCA) + PCA scores

If you have trained the PCA data on the same data you apply the PCA to, fitted_pca_scores, projected_pca_scores, and full_pca_scores will all be identical.
    
If you train the PCA on different data to the one you apply the PCA to, 'fitted' will contain data that the PCA was trained on, 'projected' will contain data that the PCA was applied to, and 'full' will contain both datasets.
    
In the descriptives directory are bar graphs showing the average responses to each experience sampling item, separated in the same way as the csv files described above.
    
The screeplot directory contains scree plots showing the 1) eigenvalues of each PCA component and 2) the explained variance of each component.
    
Finally, the wordclouds directory contains png images of wordclouds representing each of the PCA components identified.
    
- They are first named numerically (e.g., PC1).
- The three terms are the terms with the highest loading, separated by direction of loading:
    - Positive indicates the highest positive terms
    - Negative indicates the highest negative

- In these images, each word = experience sampling item, size = magnitude of loading, and color = direction of loading.
    
## Push your changes to remote
    
You have now run your first ThoughtSpace analysis in your own (local) Github repository.
    
It is now a good idea to 'push' your changes.

What do I mean when I say "push changes"?

Put simply:

1. Making Changes:

    - Imagine you're working on a school essay. You write, edit, and add new content to your document. These changes are like the modifications you make in your project files.

2. Saving Changes Locally (Commit):

    - Before you leave your computer, you save your essay to not lose your work. Similarly, in coding, you save your changes locally. This is called a "commit." It's like saving your progress in a game.

3. Sending Changes Online (Push):

    - Now, if you want to share your essay with your teacher or friends, you need to give them a copy. In coding, you need to send your saved changes online. This is called a "push." It's like uploading a file to the internet.

4. Using GitHub Desktop:

    - GitHub Desktop is like a magical tool that helps you do these steps easily. It shows you what changes you made (like highlighting the edited parts of your essay). When you click "commit," it's like saving your work. And when you click "push," it's like sharing your essay with others online.

Now here are the instructions to do these steps:

1. Open GitHub Desktop

2. View Changes:

    - GitHub Desktop will show you the changes you've made in the "Changes" tab.
    - You'll see a list of files that have been modified or created.
        - This will be the analysis script file and the results.

3. Commit Changes:

    - Write a brief summary of the changes you made in the "Summary" box (e.g., first analysis)
    - Click on the "Commit to main" button. 
    - Your changes are now committed to your local repository.

4. Push Changes to Remote:

    - After committing, you'll see a button labeled "Push origin."
    - Click on this button.
    - GitHub Desktop will upload your committed changes to the remote repository on GitHub.com.
    - Depending on your settings, you might be asked to log in to your GitHub account and confirm the push.

5. Verification:

    - Once the push is completed, your changes are now on the remote repository.
    - You can go to your GitHub repository on GitHub.com to see the changes reflected there.