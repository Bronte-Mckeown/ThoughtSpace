# ThoughtSpace

Welcome to ThoughtSpace. 

ThoughtSpace is a Python-based toolbox for analysing experience sampling data via Principal Components Analysis (PCA) to identify common "patterns of thought".

## A Beginners Guide

This readme is a Beginners guide to using ThoughtSpace that assumes little-to-no prior knowledge of coding and GitHub. 

It does assume you already have a GitHub account set up. So, if you don't, create one first!

The guide will take you through installation and your first analysis, with examples.

### Installing GitHub Desktop

GitHub Desktop is a an application provided by GitHub. 

It allows users to interact with Git repositories visually (rather than from the oh-so-scary command line) and provides an easier way for users, especially beginners, to manage their version-controlled projects.

I recommend using GitHub desktop for all git-related steps described below.

So... if you haven't already install GitHub Desktop (https://desktop.github.com/) and sign into your account.

### Installing Visual Studio Code

We are going to start with making sure you have visual studio code (VS Code) installed. If you already have it installed, skip to the next section.

#### What is VS code?

*What it is:* Visual Studio Code is like a digital workspace on your computer. It's a text editor, but it's much more than that. It's a powerful tool that helps you write, edit, and debug code easily in various programming languages, including Python and R.

*Imagine It Like:* Think of Visual Studio Code as a blank notebook where you write down your thoughts and ideas. In this case, your "thoughts" are the code you write for different programming languages.

If you have never used it before, here are the steps to install, assuming you have a windows machine (the steps may vary slightly for mac users):

**1. Download Visual Studio Code:**

- Open your web browser and go to the official Visual Studio Code website: https://code.visualstudio.com/.
- Click on the "Download for Windows" button. This will download the installer file to your computer.

**2. Run the Installer:**

- Once the download is complete, locate the installer file (it's usually named something like VSCodeSetup-x64-<version>.exe) and double-click on it to run the installer.

**3. License agreement:**

- The installer will launch a setup wizard and present the license agreement. Click "I Agree" and then "Next" to proceed.

**4. Select Destination Location:**

- Choose the destination location where you want to install Visual Studio Code. 
- On Windows, it's common to install software applications in the "Program Files" directory (or "Program Files (x86)" on 64-bit systems). 
    - This is usually the default installation location suggested by the VS Code installer.
    - Installing in the Program Files directory helps keep your system organized and ensures that the software is accessible to all users on the computer.
- The default location is usually fine for most users. Click "Next" to continue.

**5. Select Start Menu Folder:**

- Choose the folder where you want shortcuts for Visual Studio Code to appear in the Start menu. Click "Next."

**6. Select Additional Tasks:**

- Optionally, you can choose to create desktop and Quick Launch icons, and associate file types with Visual Studio Code. 
- Keep the default options and additionally choose the options according to your preference and click "Next."

**7. Install:**

- Click the "Install" button to begin the installation process. The installer will copy the necessary files to your computer.

**8. Completion:**

- Once the installation is complete, click "Finish" to exit the installer. You can optionally chose to launch visual studio code at this point too.

**9. Launch Visual Studio Code:**

- After installation, you can launch Visual Studio Code by finding it in the Start menu or by double-clicking the desktop icon if you created one during the installation process.

### Setting up to code in Python

You have now sucessfully installed VS Code to your machine.

In order to code in Python within VS Code, there are a few extra steps you need to complete.
    
The first step is to make sure you have Python installed on your machine.
    
*What It Is:* Python is a programming language. It's a way for people to communicate their instructions to a computer. You can write programs in Python to perform all sorts of tasks, from simple calculations to complex data analysis and web applications.
    
*Imagine It Like:* Python is like a special language you speak with your computer. You tell the computer what to do, step by step, using Python instructions. For example, you might tell the computer to add two numbers together, and Python helps you do that.
    
ThoughtSpace is written using the Python programming language.
    
If you already have Python installed, you can skip this step.
    
If you are unsure whether it is already installed, open up the "Command Prompt" from the start menu and type:

```
python --version
```

OR

```
python3 --version
```

Press enter.
    
If Python is installed, this command will return the installed Python version number. 
    
If Python is not installed, you will likely see an error message indicating that the command is not recognized. In that case, you would need to install Python on your machine using the steps described below.

**1. Install Python**

- If it is not on your machine, download and install the latest version of Python for Windows  from the official website (yellow box): https://www.python.org/downloads/

- You can opt for a custom installation, but the default "Install Now" option is fine.

- **IMPORTANT:** Make sure to check the box that says "Add Python to PATH." This option is important for Visual Studio Code to detect the Python installation.

- Selecting "Use admin privileges when installing py.exe" is also a good idea.

- Select "Yes" if it asks you whether you want to allow this app to make changes to your device.

- Select "Close" when it is finished.
    
You now have a Python installation on your machine.

**2. Install Python Extension in Visual Studio Code.**
    
You now need to install the Python extension in VS code to be able to use Python from within VS code.

- In Visual Studio Code, go to the Extensions view by clicking on the square icon in the sidebar on the left or by pressing Ctrl+Shift+X.

- Search for "Python" in the Extensions marketplace search bar.

- Install the one published by Microsoft (it should be the first result and the one with the blue tick). Click "Install" to install the extension.

- Note: if you have windows 11 and run into "Error while fetching extensions. XHR failed", the following link discusses a range of common fixes (https://stackoverflow.com/questions/70177216/visual-studio-code-error-while-fetching-extensions-xhr-failed)
and the following steps are a good place to start:


    - Open Windows Security
    - Click on Firewall & Network Security
    - Click on Allow an app through the firewall
    - Click on Change Settings
    - Click on Allow Another App
    - Browse to where VS Code is installed and click on Code.exe
    - Make sure that Code shows in the list of the allowable apps and that 'private' and 'public' networks are selected.


- Another easy fix to try: navigate to User settings using "Ctr+Shift+P", search "proxy", scroll down to "Proxy Support". The default is "override". Try switching to "off", restart visual studio and try again.

- If these suggestions don't fix it and for any other issues while setting up, stackoverflow often has good fixes and Chat GPT can also be useful here. 
    - If in doubt, chat to someone with more experience before making any big changes to your machine. This is particulary important if it is a uni-managed device.

**3. Select Python Interpreter**
    
You now need to select a Python Interpreter.
    
 *What It Is:* The Python interpreter is like a translator between human-readable Python code and the language that your computer understands. When you write Python code in VS Code, the Python interpreter takes that code and executes it, making your instructions come to life on your computer.

*Imagine It Like:* The Python interpreter is like a magical friend who understands both your language (Python) and your computer's language. You tell your magical friend what you want to do in Python, and they make sure your computer understands and follows your instructions.

- You can select the interpreter by opening the Command Palette (Ctrl+Shift+P), typing "Python: Select Interpreter," and choosing the interpreter from the list of detected Python interpreters.

**4. Verify it's all worked**

- Select "File", then "New File". You can then select "Python File" (ending with .py)

- Type the following in the first line: 

```
print ("Hello World")
```

- "Save as" this file to your computer (anywhere is fine as this is just a test).

- Then, in the top right corner select the arrow button, which says "Run Python File" when you hover over it.

- Make sure you have selected "Terminal" in the bottom window, and you should be able to see the print out "Hello World". 
    
- It should look something like this:
![image.png](https://hackmd.io/_uploads/rJoXGMeQ6.png)

- If you run into issues running this script, stackoverflow often has good fixes (be specific and concise in your search) and Chat GPT can also be useful here. 
- If in doubt, chat to someone with more experience before making any big changes to your machine.
    
### Recap of what you have done so far...

- Installed Visual studio code:
  - What It Is: Visual Studio Code is like a digital workspace on your computer. It's a text editor, but it's much more than that. It's a powerful tool that helps you write and edit code for various programming languages, including Python.
  - Imagine It Like: Think of Visual Studio Code as a blank notebook where you write down your thoughts and ideas. In this case, your "thoughts" are the code you write for different programming languages.

- Installed Python (and Python extension within visual studio code):
  - What It Is: Python is a programming language. It's a way for people to communicate their instructions to a computer. You can write programs in Python to perform all sorts of tasks, from simple calculations to complex data analysis and web applications.
  - Imagine It Like: Python is like a special language you speak with your computer. You tell the computer what to do, step by step, using Python instructions. For example, you might tell the computer to add two numbers together, and Python helps you do that.

- Selected Python Interpreter: 
  - What It Is: The Python interpreter is like a translator between human-readable Python code and the language that your computer understands. When you write Python code in VS Code, the Python interpreter takes that code and executes it, making your instructions come to life on your computer.
  - Imagine It Like: The Python interpreter is like a magical friend who understands both your language (Python) and your computer's language. You tell your magical friend what you want to do in Python, and they make sure your computer understands and follows your instructions.

#### Putting It All Together:

- So when you created your program that says "Hello, World!" on your computer screen, you wrote the instructions for this in Python (your language) inside Visual Studio Code (your notebook). When you ran your program, Visual Studio Code used the Python interpreter to translate your Python instructions into a language your computer understands. As a result, your computer displayed "Hello, World!" on the screen.
  
- In summary, Visual Studio Code is your workspace, Python is the language you use to communicate with your computer, and the Python interpreter is the translator that makes sure your computer understands and carries out your Python instructions.

### Fork and Clone ThoughtSpace.
    
Now it's time to get ThoughtSpace installed onto your machine so you can use it in your own analyses.

**1. Fork ThoughtSpace**
    
- You need to first 'fork' ThoughtSpace. This step creates your own remote copy of this repository.
    
- On the top right corner of ThoughtSpace's main repository page, there will be a button that says "Fork". Click on this button.
    
- GitHub will now create a copy of the original ThoughtSpace repository and place it in your GitHub account.
    
- You can skip this step and go straight to 'cloning' below, however, forking is useful if you want to make any contributions to ThoughtSpace via a "pull request".

- For more information on how all this works, see here: https://docs.github.com/en/get-started/quickstart/fork-a-repo


**4. Clone ThoughtSpace to your machine**
    
If you have forked ThoughtSpace first:


- Go to the forked GitHub repository using your web browser.
    - Copy the URL of the repository. It should look something like this (with your username): https://github.com/username/ThoughtSpace
    
If you have skipped straight to cloning:

- Copy the URL of ThoughtSpace's main page: https://github.com/Bronte-Mckeown/ThoughtSpace 

Now, you have the URL copied:

- Open GitHub Desktop
- In GitHub Desktop, click on the "File" menu in the upper-left corner.
- Select "Clone Repository..." from the dropdown menu.

- GitHub Desktop will open a dialog box. In this box:
    - URL: Paste the URL of the repository you copied from GitHub.
    - Local Path: Choose the local directory on your computer where you want to clone the repository. Click "Choose..." to select the folder. This could be in a folder called something like "repos".
    - Click the "Clone" button to start the cloning process.

- GitHub Desktop will download the repository files to your computer. Wait for the process to finish. Once it's done, you'll see the repository listed in your GitHub Desktop application.

- You now have a local copy of ThoughtSpace on your computer which you can see foryourself if you navigate to the folder you selected when cloning using the file explorer.
    
- If you forked first, this will be a local copy of *your* forked version of ThoughtSpace. Otherwise, this is a local copy of ThoughtSpace.

### Create a new GitHub repository on your local machine to run your analysis.
    
You now are set up for coding in Python and have ThoughtSpace installed on your machine.
    
Now, it's time to start your analysis. If you already have a GitHub repository (or folders) set up for your own analysis, you can skip these steps, but this guide assumes you have never done anything like this before.

**1. In Github Desktop, select "File" and then "New Repository".**

- Input your repository name of choice (e.g., If you are analyzing the COVID-19 dataset available in ThoughtSpace's example_data, you might call it something like "covid_analysis").

- You can optionally add a description (e.g., "This repository uses ThoughtSpace to analyze COVID-19 lockdown data.").

- Select a folder to store the repository. It's up to you where you keep it but Github Desktop will default to a folder it creates on installation: "GitHub". This is a good option.

- Select "Create Repository". 

- You have now created a GitHub repository on your local PC. It is has not yet been "pushed" to the remote but don't worry about what that means for now (we will get to that later).

**2. Create 'data' and 'results' folders in repository.**

- Now navigate to where your repository is stored using the file explorer.

- Create a directory called 'data' and a directory called 'results'.

**3. Add your data (in csv format) to the data folder.**

- If you don't have your own data yet, you can use the example data available in ThoughtSpace (see below for more information).

### Final steps 

You now have all the necessary software installed and have set up your own GitHub repository for running your ThoughtSpace analysis.


In order to be able to use ThoughtSpace within your own GitHub analysis repository, there are a few final steps:

**1. Open up the Command Prompt terminal.**

**2. Create a new anaconda environment with Python 3.8.13**

- An Anaconda environment is like a self-contained workspace for your Python projects.
- Imagine it as a virtual box where you can install specific versions of Python and libraries (like ThoughtSpace) tailored for a particular project, without affecting other projects.
- It helps you manage different project requirements independently.
- This way, you can work on one project using one set of libraries and another project using a different set, ensuring they don't interfere with each other.
- You are going to create one of these environments and install ThoughtSpace into this environment.
- Name it according your specific analysis (e.g., if analysing the example data, you might call it something like "covid").

- Specifically, type the following into your command prompt (changing name to your preference) and press enter: 

```
conda create -n <name_of_environment> python=3.8.13
```

- Example with 'covid' as name:

```
conda create -n covid python=3.8.13
```
    
- When you run this command, you are using Conda, a popular package and environment management system in Python, to create a new virtual environment with a specific Python version.
    
- Here's what happens step by step:

    - conda create: This part of the command tells Conda that you want to create a new environment.

    - -n <name_of_environment>: This option specifies the name you want to give to your new environment. Replace <name_of_environment> with the desired name, for example, myenv.

    - python=3.8.13: This part of the command specifies that you want to install Python version 3.8.13 in your new environment. Conda will download and install Python 3.8.13 along with essential packages required for Python development.

- Type 'y' if prompted to install associated packages.

**3. Activate environment**

You have now created the anaconda environment on your machine.
    
But now you need to 'activate' it in order to install ThoughtSpace. 

- Type the following into your command prompt and press enter:

```
conda activate <name_of_environment>
```

- Example with 'covid' as name:

```
conda activate covid
```

**4. Navigate to ThoughtSpace directory**

- Type the following (replacing with the file path to your local copy of ThoughtSpace) into your command prompt and press enter:

```
cd <path_to_thoughtspace>
```

- It might look something like this:
```
cd Documents\repos\ThoughtSpace
```

**5. pip install ThoughtSpace requirements**

We are now in a position where 1) the anaconda environment has been created and activated and 2) we are within the ThoughtSpace directory via the command line.

In order to install ThoughtSpace into this activated environment, type the following into your command prompt and press enter:

```
pip install .
```


You have now installed ThougthSpace inside the anaconda environment you created.
    
#### Doing this all again following updates to ThoughtSpace...
    
As ThoughtSpace is still in development, you may want to update your version of ThoughtSpace as changes are made.
    
If you have forked and then cloned, do the following first:

1. Go to your forked repository on GitHub.
2. Click on the "Fetch upstream" button
    - This button is usually found on the top right side of the repository page, near the "Code" button.
3. Click on the "Create pull request" button. This will take you to a new pull request page.
4. GitHub will automatically set the base repository and base branch to your fork and your default branch (e.g., main or master). It will also set the head repository and branch to the original repository and the branch from which you want to merge changes.
5. Review the changes and click on the "Create pull request" button to create the pull request. Add a title and description if necessary, and then click "Create pull request" again.
6. Finally, click on "Merge pull request" to merge the changes from the original repository into your forked repository.

Your forked repository is now synced with the changes from the original repository.

If you skipped forking and just simply cloned ThoughtSpace directly, do the following first:
    
1. Open GitHub desktop
2. Select the ThoughtSpace repository
3. Select "Fetch origin". This will fetch the changes from the remote ThoughtSpace repository and update your local copy of this repository.

Now that your local copy of ThoughtSpace is up-to-date, do the following:
    1. Open the Command Prompt
    2. Activate your conda environment as before (see above)
    3. Navigate to ThoughtSpace directory using 'cd' as before (see above)
    4. Type "pip install .", press Enter.

This step will uninstall the old version of ThoughtSpace and re-install the new version.

### Running your first PCA analysis

We are now ready to run your first ThoughtSpace analysis!

**1. Open visual studio code**

**2. Create new python file in your analysis directory (see above for details on how to do this)**
    
**3. Type the following and save the file:**

In the example pasted below, change the file path to your own csv file stored in your own data directory or alternatively, use the URL method pasted below to use the example data available on ThoughtSpace.

Using URL method to read in example data (daily life experience sampling data before and during lockdown in the UK): 

```
import pandas as pd # for reading in the csv file containg the data
from ThoughtSpace._base import basePCA # this imports the basePCA class from ThoughtSpace

# read in data
url = 'https://github.com/Bronte-Mckeown/ThoughtSpace/tree/master/scratch/data/example_data.csv'
df = pd.read_csv(url)

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and transform data to create PCA scores
# pca_scores now contains dataframe with original data plus PCA columns
pca_scores = model.fit_transform("data/lockdown_data.csv")

# save results to results folder using .save
model.save(path="results",pathprefix="PCA_results")
```

Using your own data:

Things to note about the data format required:
- PCA will be applied to any numerical columns provided.
- So make sure any columns you don't want included, such as an 'ID number' column or an 'age' column, are set to string variables, and that variables you do want included (i.e., experience sampling items) are set to numerical variables.

```
import pandas as pd # for reading in your csv
from ThoughtSpace._base import basePCA # to use ThoughtSpace

# read in data
df = pd.read_csv("data/lockdown_data.csv")

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and transform data to create PCA scores
# pca_scores now contains dataframe with original data plus PCA columns
pca_scores = model.fit_transform("data/lockdown_data.csv")

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
    
- Kaiser-Meyer-Olkin (KMO) Measure: KMO is a statistic that measures the adequacy of the sample size for conducting factor analysis. It assesses the proportion of variance among variables that might be common variance. In simpler terms, KMO helps you determine if your data is suitable for PCA. A high KMO value (usually above 0.6) suggests that the data is appropriate for factor analysis, indicating that the variables are related enough for PCA to yield meaningful results.

- Bartlett's Test of Sphericity: Bartlett's test checks whether the variables in your dataset are correlated, indicating that they might be suitable for PCA. It tests the null hypothesis that the correlation matrix is an identity matrix (which would mean no correlation between variables). In the context of PCA, you want to reject this null hypothesis because PCA is based on the idea that variables are correlated. If Bartlett's test is statistically significant (i.e., the p-value is below a chosen significance level, such as 0.05), it suggests that your data is appropriate for PCA.

**5. Check your results!**

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
    
  ### Push your changes to remote
    
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

- GitHub Desktop is like a magical tool that helps you do these steps easily. It shows you what changes you made (like highlighting the edited parts of your essay).
When you click "commit," it's like saving your work. And when you click "push," it's like sharing your essay with others online.

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
