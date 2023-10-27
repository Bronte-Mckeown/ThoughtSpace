# ThoughtSpace

Python-based pipeline for flexible Principal Components Analysis (PCA) and projecting between datasets.

## A Beginners Guide

This is a Beginners guide that assumes little-to-no prior knowledge of coding and GitHub. It does assume you already have a GitHub account set up. So, if you don't, create one first.

### Installing Visual Studio Code
VS Code, is a free, open-source code editor developed by Microsoft. It's designed for developers and programmers to write, edit, and debug code easily and efficiently.
It can be used to code in a range of programming languages including Python and R.

If you have never used it before, here are the steps to install, assuming you have a windows machine:
1. Download Visual Studio Code:

- Open your web browser and go to the official Visual Studio Code website: https://code.visualstudio.com/.
- Click on the "Download for Windows" button. This will download the installer file to your computer.

2. Run the Installer:

- Once the download is complete, locate the installer file (it's usually named something like VSCodeSetup-x64-<version>.exe) and double-click on it to run the installer.

3. Setup Wizard:

- The installer will launch a setup wizard and present the license agreement. Click "I Agree" and then "Next" to proceed.

4. Select Destination Location:

- Choose the destination location where you want to install Visual Studio Code. 

- On Windows, it's common to install software applications in the "Program Files" directory (or "Program Files (x86)" on 64-bit systems). 

- This is usually the default installation location suggested by the VS Code installer.

- Installing in the Program Files directory helps keep your system organized and ensures that the software is accessible to all users on the computer.

- The default location is usually fine for most users. Click "Next" to continue.

5. Select Start Menu Folder:

- Choose the folder where you want shortcuts for Visual Studio Code to appear in the Start menu. Click "Next."

6. Select Additional Tasks:

- Optionally, you can choose to create desktop and Quick Launch icons, and associate file types with Visual Studio Code. Keep the default options and additionally choose the options according to your preference and click "Next."

7. Install:

- Click the "Install" button to begin the installation process. The installer will copy the necessary files to your computer.

8. Completion:

- Once the installation is complete, click "Finish" to exit the installer. You can optionally chose to launch visual studio code at this point too.

9. Launch Visual Studio Code:

- After installation, you can launch Visual Studio Code by finding it in the Start menu or by double-clicking the desktop icon if you created one during the installation process.

### Setting up to code in Python

- In order to code in Python within visual studio code, there are a few extra steps.

1. Install Python

- Before you start coding in Python, you need to have Python installed on your system.
- If you are unsure whether it is already installed, open up the "Command Prompt" and type:

```
python --version
```

OR

```
python3 --version
```

- If Python is installed, this command will return the installed Python version number. If Python is not installed, you will likely see an error message 
indicating that the command is not recognized. In that case, you would need to install Python on your machine.

- If it is not on your machine, download and install the latest version of Python for Windows (yellow box) from the official website: https://www.python.org/downloads/

- You can opt for a custom installation, but the default "Install Now" option is fine.

- IMPORTANT: Just make sure to check the box that says "Add Python to PATH." This option is important for Visual Studio Code to detect the Python installation.

- Selecting "Use admin privileges when installing py.exe" is also a good idea.

- Select "Yes" if it asks you whether you want to allow this app to make changes to your device.

- Select "Close" when it is finished.

2. Install Python Extension in Visual Studio Code.

- In Visual Studio Code, go to the Extensions view by clicking on the square icon in the sidebar on the left or by pressing Ctrl+Shift+X.

- Search for "Python" in the Extensions marketplace search bar.

- Install the one published by Microsoft (it should be the first result and the one with the blue tick). Click "Install" to install the extension.

- Note: if you have windows 11 and run into "Error while fetching extensions. XHR failed", the following link discusses fixes (https://stackoverflow.com/questions/70177216/visual-studio-code-error-while-fetching-extensions-xhr-failed)
and the following steps are a good place to start with:

```
Open Windows Security
Click on Firewall & Network Security
Click on Allow an app through the firewall
Click on Change Settings
Click on Allow Another App
Browse to where VS Code is installed and click on Code.exe
Make sure that Code shows in the list of the allowable apps and that 'private' and 'public' networks are selected.
```

- Another easy fix to try: navigate to User settings using "Ctr+Shift+P", search "proxy", scroll down to "Proxy Support". The default is "override". Try switching to "off", restart visual studio and try again.

- If these suggestions don't fix it and for any other issues while installing, stackoverflow often has good fixes and Chat GPT can also be useful here. If in doubt, chat to someone with more experience before making any big changes to your machine. This is particulary important if it is a uni-managed device.

3. Select Python Interpreter

- You can select the interpreter by opening the Command Palette (Ctrl+Shift+P), typing "Python: Select Interpreter," and choosing the interpreter from the list of detected Python interpreters.

4. Verify it's worked

- Select "File", then "New File". You can then select "Python File".

- Type the following: 

```
print ("Hello World")
```

- "Save as" this file to your computer.

- Then, in the top right corner select the arrow button, which will "Run Python File".

- Make sure you have selected "Terminal" in the bottom window, and you should be able to see the print out "Hello World". 

- If you run into issues, stackoverflow often has good fixes and Chat GPT can also be useful here. If in doubt, chat to someone with more experience before making
any big changes to your machine.

#### Recap of what you have done so far...

- Installed Visual studio code:
  - What It Is: Visual Studio Code is like a digital workspace on your computer. It's a text editor, but it's much more than that. It's a powerful tool that helps you write and edit code for various programming languages, including Python.
  - Imagine It Like: Think of Visual Studio Code as a blank notebook where you write down your thoughts and ideas. In this case, your "thoughts" are the code you write for different programming languages.

- Installed Python (and Python extension within visual studio code):
  - What It Is: Python is a programming language. It's a way for people to communicate their instructions to a computer. You can write programs in Python to perform all sorts of tasks, from simple calculations to complex data analysis and web applications.
  - Imagine It Like: Python is like a special language you speak with your computer. You tell the computer what to do, step by step, using Python instructions. For example, you might tell the computer to add two numbers together, and Python helps you do that.

- Selected Python Interpreter: 
  - What It Is: The Python interpreter is like a translator between human-readable Python code and the language that your computer understands. When you write Python code in VS Code, the Python interpreter takes that code and executes it, making your instructions come to life on your computer.
  - Imagine It Like: The Python interpreter is like a magical friend who understands both your language (Python) and your computer's language. You tell your magical friend what you want to do in Python, and they make sure your computer understands and follows your instructions.

##### Putting It All Together:

- Imagine you want to create a program that says "Hello, World!" on your computer screen. You write the instructions for this in Python inside Visual Studio Code (your notebook). When you run your program, Visual Studio Code uses the Python interpreter to translate your Python instructions into a language your computer understands. As a result, your computer displays "Hello, World!" on the screen.
  
- In summary, Visual Studio Code is your workspace, Python is the language you use to communicate with your computer, and the Python interpreter is the translator that makes sure your computer understands and carries out your Python instructions.

### Fork and Clone ThoughtSpace.

1. Fork ThoughtSpace

- Please see instructions here: https://docs.github.com/en/get-started/quickstart/fork-a-repo

- This step creates your own remote copy of this repository.

4. Clone ThoughtSpace to your local PC

- Launch GitHub Desktop on your computer.

- Find the Repository to Clone:
    - Go to the forked GitHub repository using your web browser.
    - Copy the URL of the repository. It looks like https://github.com/username/ThoughtSpace.git.

- Clone the Repository:
    - In GitHub Desktop, click on the "File" menu in the upper-left corner.
    - Select "Clone Repository..." from the dropdown menu.

- GitHub Desktop will open a dialog box. In this box:
    - URL: Paste the URL of the repository you copied from GitHub.
    - Local Path: Choose the local directory on your computer where you want to clone the repository. Click "Choose..." to select the folder.
    - Click the "Clone" button to start the cloning process.

- GitHub Desktop will download the repository files to your computer. Wait for the process to finish. Once it's done, you'll see the repository listed in your GitHub Desktop application.

- You now have a local copy of your forked copy of ThoughtSpace.

### Create a new GitHub repository on your local machine.

The easiest way to do this is: 

1. Install GitHub Desktop (https://desktop.github.com/) and sign into your account.

2. In Github Desktop, select "File" and then "New Repository".

- Input your repository name of choice (e.g., If you were analyzing the COVID-19 dataset available in example_data, you might call it something like "covid_analysis").

- You can optionally add a description (e.g., "This repository uses ThoughtSpace to analyze COVID-19 lockdown data.").

- Select a folder to store the repository. It's up to you where you keep it but Github Desktop will default to a folder it creates on installation: "GitHub". This is not a good option.

- Select "Create Repository". 

- You have now created a GitHub repository on your local PC. It is has not yet been "pushed" to the remote but don't worry about what that means for now (we will get to that later).

3. Create data and results folders in repository.

- Now navigate to where your repository is stored using the file explorer.

- Create a directory called 'data' and a directory called 'results'.

3. Add your data as a csv file to the data folder.

- Alternatively, for your first analysis, you can use the example data available in ThoughtSpace (see below).

#### Final steps to being able to use ThoughtSpace

You now have 1) a local copy of ThoughtSpace on your machine and 2) a GitHub repository in which you will run your analysis. 

In order to be able to use ThoughtSpace within your own GitHub analysis repository, you need to do the following:

1. Open up command prompt

2. Create a new anaconda environment with Python 3.8.13 and name it according your analysis (e.g., covid)

- An Anaconda environment is like a self-contained workspace for your Python projects.
- Imagine it as a virtual box where you can install specific versions of Python and libraries tailored for a particular project, without affecting other projects.
- It helps you manage different project requirements independently.
- This way, you can work on one project using one set of libraries and another project using a different set, ensuring they don't interfere with each other.
- You are going to create one of these environments and install ThoughtSpace into this environment.

Type the following into your command prompt and press enter: 

```
conda create -n <name_of_environment> python=3.8.13
```

e.g., 

```
conda create -n covid python=3.8.13
```

Type 'y' if prompted to install associated packages.

3. Activate environment

You have now created the environment on your machine. But now you need to 'activate' it in order to install ThoughtSpace. 

Type the following into your command prompt and press enter:

```
conda activate <name_of_environment>
```

e.g., 

```
conda activate covid
```

4. Navigate to ThoughtSpace directory

Type the following (replacing with your own path) into your command prompt and press enter:

```
cd <path_to_thoughtspace>
```

5. pip install ThoughtSpace requirements

From within ThoughtSpace, type the following into your command prompt and press enter:

```
pip install .
```

You have now installed ThougthSpace inside the anaconda environment you created.

##### Run your first PCA analysis

1. Open visual studio code

2. Create new python file in your analysis directory (see above for details on how to do this)

In the example below, change the file path to your own csv stored in your own data directory or alternatively, use the URL method to use the example data available on ThoughtSpace.

Using example data (daily life experience sampling data before and during lockdown in the UK): 

```
import pandas as pd # for reading in your csv
from ThoughtSpace._base import basePCA # to use ThoughtSpace

# read in data
url = 'https://github.com/Bronte-Mckeown/ThoughtSpace/tree/master/scratch/data/example_data.csv'
df = pd.read_csv(url)

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and project to create PCA scores
projected_results = model.fit_project("data/lockdown_data.csv")

# save results to results folder
model.save(path="results",pathprefix="PCA_results")
```

Using your own data:

Things to note about the data format required:
- PCA will be applied to any numerical columns provided.
- So make sure any columns you don't want included, such as an ID column, are set to string variables,
and that variables you do want included are set to numerical variables.

```
import pandas as pd # for reading in your csv
from ThoughtSpace._base import basePCA # to use ThoughtSpace

# read in data
df = pd.read_csv("data/lockdown_data.csv")

# sets up PCA object, asking for 4 components with varimax rotation
model = basePCA(n_components=4,rotation="varimax")

# train PCA on data and project to create PCA scores
projected_results = model.fit_project("data/lockdown_data.csv")

# save results to results folder
model.save(path="results",pathprefix="PCA_results")
```

3. Before running this script, select the conda environment you have created.

You can select the interpreter by opening the Command Palette (Ctrl+Shift+P), typing "Python: Select Interpreter," and choosing the interpreter from the list of detected Python interpreters.

It will be called whatever you called the conda environment created above (e.g., covid).

4. Run Python script using arrow in top right corner.

5. Check your results!

They will be stored in your results folder.

In the csv directory, you will find:

- pca_loadings: rows = items, columns = pca components, values = loadings
- fitted_pca_scores: data PCA was trained/fitted on
- projected_pca_scores: data PCA was applied to
- full_pca_scores: data PCA was trained/fitted on plus data PCA was applied to
- pca_scores_original_format: PCA scores + rest of original dataframe PCA was trained on (including string columns not included in PCA)
