## **Tucker's Congruence Coefficient**
### *Comparing Components by Their Loadings*

In PCA, each component is some linear combination of the variables being decomposed that accounts for as much variance in those variables as possible. In other words, a component essentially summarizes broad relationships between groups of variables. You can kind of think of PCA as correlating all of the inputted variables with each other at once and ranking which groupings (i.e., 'components') show the strongest correlations/patterns in the data. Each variable will contribute to each pattern in a different way (e.g., some variables will be more/less involved in some patterns than others, some variables will positively associate with the pattern while others negatively associate with it, etc.). The way each variable contributes to the pattern across the variables (i.e., the component) is called a 'loading'.

One way to assess the similarity between two components is by examining the similarity of each of their loadings. Since loadings describe the particular pattern the component is capturing in the data, if two components have similar loadings they can be thought of as identifying similar largescale relationships among the variables. 

The metric typically used to measure the similarity between loadings is called *Tucker's Congruence Coefficient (TCC)* also known as *Tucker's Phi ($`\phi`$)* (Lorenzo-Seva & ten Berge, 2006; Tucker, 1951). The formula below is a slightly modified version of the original shown to be more effective for identifying which components are the most similar across two datasets (Lovik et al., 2020).

**Tucker's Congruence Coefficient ($\phi$)**

$\phi(x,y)=\frac{\sum |x_i y_i|}{\sqrt{\sum x_i^2\sum y_i^2}}$

- $\sum |x_i y_i|$ is the sum of the absolute products of each loading of the two components (i.e., $x$ and $y$).
- $\sqrt{\sum x_i^2\sum y_i^2}$ is the total variance in the loadings between the two components with 0 held as the absolute mean value of any set of loadings.

The TCC is essentially a slightly modified Pearson's correlation coefficient: it is the proportion of the total variability among the loadings captured by the covariability between the loadings. However, instead of calculating covariance and variance by subtracting each loading from the mean loading value for each component, it maintains an absolute mean loading value of 0.

To interpret the TCC, Lorenzo-Seva & ten Berge (2006) recommend interpreting $\phi\ge.85$ as indicating fair similarity and $\phi\ge.95$ as indicating exact similarity. 

## References

Lorenzo-Seva, U., & ten Berge, J. M. F. (2006). Tucker's congruence coefficient as a meaningful index of factor similarity. Methodology: European Journal of Research Methods for the Behavioral and Social Sciences, 2(2), 57-64. https://doi.org/10.1027/1614-2241.2.2.57

Lovik, A., Nassiri, V., Verbeke, G., & Molenberghs, G. (2020). A modified tucker’s congruence coefficient for factor matching. Methodology: European Journal of Research Methods for the Behavioral and Social Sciences, 16(1), 59-74. https://doi.org/10.5964/meth.2813 

Tucker, L. R. (1951). A method for synthesis of factor analysis studies (PRS-984). Personnel Research Section Reports. Armed Services Technical Research Agency.

