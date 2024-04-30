# ThoughtSpace

Welcome to ThoughtSpace. 

ThoughtSpace is a Python-based toolbox for analysing experience sampling data via Principal Components Analysis (PCA) to identify common "patterns of thought".

## A Beginners Guide

This readme is a Beginners guide to using ThoughtSpace that assumes little-to-no prior knowledge of coding and GitHub. 

It does assume you already have a GitHub account set up. So, if you don't, create one first!

The guide will take you through installation and your first analysis, with examples:

- [Installing Github Desktop](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/Installing_GitHub_Desktop.md)
- [Setting Up Visual-Studio Code](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/Installing_VS_Code.md)
- [Setting Up Python in VSCode](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/Setting_Up_Python.md)

## Setting Up and Running ThoughtSpace

If you have the necessary interpreter and programming steps set up, you can get into preparing and running ThoughtSpace.

If you're new to GitHub, here's a guide to forking and cloning the ThoughtSpace repository:

- [Fork and Clone ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/Fork_and_Clone_ThoughtSpace.md)

Once you've set up a local (and forked) ThoughtSpace repository on your computer, you can set up a virtual environment for using ThoughtSpace:

- [Setting Up ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/Set_Up_ThoughtSpace.md)
    - [Updating ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/updating_thoughtspace.md)

- [Running Your First PCA Analysis](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/First_PCA_Analysis.md)

## The rhom Module: Testing the Robustness of Your Components

After you've generated your components, it's good to get a sense of how robustly they represent your data and how well they might generalize across different levels of some grouping variable of interest (e.g., different sampling environments).

The rhom module allows you to do just that! Usage of this module comprises mainly of four functions representing a range of ways to assess both component reliability and generalizability. Take a look at the guides below as to how to use them and be sure to look at the example script provided in the /examples folder!
    - [Split-Half Reliability]()
    - [Direct-Projection Reproducibility]()
    - [Omnibus-Sample Reproducibility]()
    - [By-Component Omnibus-Sample Reproducibility]()

