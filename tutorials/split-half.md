## **Split-Half Reliability**
### *How robustly do my components represent my data?*

Dimensionality reduction techniques such as Principal Components Analysis identify multivariate patterns in large dataframes to effectively 'create' a set of summary variables that capture the meaningful variance in a measure in a more compact fashion. This technique can be very powerful for making large datasets more analytically accessible and interpretable. Still, it's important not to merely take components at face value but to check whether the patterns PCA identifies effectively represent the data that they are decomposing.

While techniques such as Exploratory Factor Analysis have 'goodness-of-fit' metrics that measure how effectively a set of latent factors represent underlying data (e.g., RMSEA, RMSR, TLI, etc.), these measures are inappropriate for PCA (Widaman, 2018). However, because there are many scenarios where PCA could be argued to be the preferable method of dimensionality reduction (e.g., there is no theoretical basis for a given phenomenon to be a result of a set of discrete latent 'factors', etc.), techniques for assessing the robustness of emergent components rather than latent factors are valuable (Alavi et al., 2020; Fabrigar & Wegener, 2011). 

One simple way to test component 'goodness-of-fit' is with a technique called 'split-half reliability'. This method involves repeatedly randomly dividing your dataframe into halves, extracting components from each half and then testing if the components from each half are similar. If the components were capturing patterns weakly representative of your data structure, then the patterns would likely not stably appear across subsets of the data. However, if random halves of the data repeatedly produce similar components, that provides some evidence that the patterns the components are capturing are robust enough to consistently appear in random subsamples of your data.

This measure of robustness is helpful for measuring component reliability but can also be useful in its own right, say, for: 1) helping determine the number of components to extract from a given dataset, and 2) whether certain items in a questionnaire introduce more noise to a measure that damages component reliability (e.g., see Chitiz et al., 2024; Everett, 1983; Mulholland et al., 2023).

## References

Alavi, M., Visentin, D. C., Thapa, D. K., Hunt, G. E., Watson, R., & Cleary, M. (2020). Exploratory factor analysis and principal component analysis in clinical studies: Which one should you use? Journal of Advanced Nursing, 76(8), 1886-1889. https://doi.org/10.1111/jan.14377 

Chitiz, L., McKeown, B., Mulholland, B., Wallace, R., Goodall-Halliwell, I., Ho, N. S.-P., Konu, D., Poerio, G. L., Wammes, J., Milham, M. P., Klein, A., Jefferies, E., Leech, R., & Smallwood, J. (2024). Mapping cognition across lab and daily life using experience-sampling. PsyArXiv.

Everett, J. E. (1983). Factor Comparability as a Means of Determining the Number of Factors and Their Rotation. Multivariate Behavioral Research, 18(2), 197-218. https://doi.org/10.1207/s15327906mbr1802_5 

Fabrigar, L. R., & Wegener, D. T. (2011). Exploratory Factor Analysis. Oxford University Press. https://doi.org/10.1093/acprof:osobl/9780199734177.001.0001 

Mulholland, B., Goodall-Halliwell, I., Wallace, R., Chitiz, L., McKeown, B., Rastan, A., Poerio, G. L., Leech, R., Turnbull, A., Klein, A., Milham, M., Wammes, J. D., Jefferies, E., & Smallwood, J. (2023). Patterns of ongoing thought in the real world. Consciousness and Cognition, 114, 103530. https://doi.org/10.1016/j.concog.2023.103530

Widaman, K. F. (2018). On Common Factor and Principal Component Representations of Data: Implications for Theory and for Confirmatory Replications. Structural Equation Modeling: A Multidisciplinary Journal, 25(6), 829-847. https://doi.org/10.1080/10705511.2018.1478730 


