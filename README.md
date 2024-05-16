# ThoughtSpace

Welcome to ThoughtSpace. 

ThoughtSpace is a Python-based toolbox for analysing experience sampling data via Principal Components Analysis (PCA) to identify common "patterns of thought".

## A Beginners Guide

This readme is a Beginners guide to using ThoughtSpace that assumes little-to-no prior knowledge of coding and GitHub. 

It does assume you already have a GitHub account set up. So, if you don't, create one first!

The guide will take you through installation and your first analysis, with examples:

- [Installing Github Desktop](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/Installing_GitHub_Desktop.md)
- [Setting Up Visual-Studio Code](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/Installing_VS_Code.md)
- [Setting Up Python in VSCode](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/Setting_Up_Python.md)

## Setting Up and Running ThoughtSpace

If you have the necessary interpreter and programming steps set up, you can get into preparing and running ThoughtSpace.

If you're new to GitHub, here's a guide to forking and cloning the ThoughtSpace repository:

- [Fork and Clone ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/Fork_and_Clone_ThoughtSpace.md)

Once you've set up a local (and forked) ThoughtSpace repository on your computer, you can set up a virtual environment for using ThoughtSpace:

- [Setting Up ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/Set_Up_ThoughtSpace.md)
    - [Updating ThoughtSpace](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/updating_thoughtspace.md)

- [Running Your First PCA Analysis](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/First_PCA_Analysis.md)

## The rhom Module: Testing the Robustness of Your Components

After you've generated your components, it's important to get a sense of how robustly they represent your data and how well they  generalize across types of situations (e.g., different sampling environments, different participant populations, etc.).

Usage of rhom primarily involves four functions that assess both component reliability and generalizability in a few different ways. Take a look at the guides below, organized by the questions each function targets, and be sure to look at the example script provided in the /examples folder!

*How robustly do my components represent my data?*
- [Split-Half Reliability](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/split-half.md)

*How similar are the components produced by different situations?*
- [Direct-Projection Reproducibility](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/direct-project.md)

*How representative are the components I get when I combine data from different situations?*
- [Omnibus-Sample Reproducibility](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/omni-sample.md)

Answering these questions requires a metric that provides some indication of the similarity between two components (e.g., generated from different halves of the same dataset, generated from separate datasets with the same measure, etc.). To do so, the rhom module leverages two metrics of component similarity:

- [Tucker's Congruence Coefficient (TCC): Comparing Components by Their Loadings]()
- [R-Homologue: Comparing Components by the Way They Organize Observations]()

