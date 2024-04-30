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
