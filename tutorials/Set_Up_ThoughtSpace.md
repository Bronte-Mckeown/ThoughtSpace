## Final steps for setting up ThoughtSpace

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