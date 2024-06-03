## **Omnibus-Sample Reproducibility**
### *How representative are the components I get when I combine data from different situations?*

This question relates to a slightly different type of analysis than what direct-projection might answer. In this case, the user's goal is not to compare components across different samples but to *synthesize* across studies from which components were derived. Specifically, this technique tests the viability of a set of components for robust synthesis across literature. One example of an analysis that might include a test of omnibus-sample reproducibility is if a researcher conducted experience-sampling across a variety of different contexts and wanted to determine how individuals' thought-patterns varied across many different situations (see Chitiz et al., 2024; Ho et al., 2020).

Rather than generating components from different datasets as in direct-projection, this method (the **omni_sample()** function) takes two or more datasets and compares how the components derived from their combined data ("omnibus components") relate to the components each dataset might produce on its own ("sample components") (See our pages on *[Tucker's Congruence Coefficient](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/tcc.md)* and *[R-Homologue](https://github.com/Bronte-Mckeown/ThoughtSpace/blob/Rhom/tutorials/RHom.md)* to learn about how we measure component similarity). To ensure the omnibus and sample components are derived from independent data, this analysis randomly assigns half of each dataset to either be included in the combined data or be kept on its own to produce the sample components representative of the dataset on its own (Chitiz et al., 2024; Ho et al., 2020). To generate more robust reproducibility estimates, this process of randomly generating 'omnibus' and 'sample' datasets is conducted for *n* bootstrap resamples. See below for a graphical representation of this technique (adapted from Chitiz et al., 2024).

**Analyzing how well components derived from combined data represent the constituent datasets they synthesize**

<img src=figures/omni-sample.png width = 100%>

*A visualisation of the ‘omnibus-sample’ method used to determine whether the ‘common’ components derived from aggregated data reproduce the components seen in each dataset individually, O = ‘omnibus’, S = ‘sample’, 𝛬 = principal components, F = component scores, TCC = Tucker’s Congruence Coefficient.*

Interpretation of this analysis is straightforward much like the others, the more similar the components from aggregated data to components from each dataset, the more viable synthesis is across datasets. Additionally, in cases where there doesn't seem to be a clear number of components to extract based on parallel analysis and variance accounted for, analyses such as direct-projection and omnibus-sample reproducibility can be useful for taking a generalizability-driven approach for determining an optimal PCA solution.

### **By-Component Omnibus-Sample Reproducibility**

Another handy function from the rhom module is the **bypc()** function (see the rhom example script), which can break down how the components from combined data relate to the datasets they're synthesizing on a by-component level.

**Note!** To do this, the bypc() function has to maintain a stable separation of the datasets into omnibus and sample divisions. Thus, **before using it make sure that your data demonstrates at least moderate split-half reliability (i.e., RHom. $\ge$ .80, TCC $\ge$ .85)** (Chitiz et al., 2024).

## References

Chitiz, L., McKeown, B., Mulholland, B., Wallace, R., Goodall-Halliwell, I., Ho, N. S.-P., Konu, D., Poerio, G. L., Wammes, J., Milham, M. P., Klein, A., Jefferies, E., Leech, R., & Smallwood, J. (2024). Mapping cognition across lab and daily life using experience-sampling. PsyArXiv.

Ho, N. S.-P., Poerio, G. L., Konu, D., Turnbull, A., Sormaz, M., Leech, R., Bernhardt, B., Jefferies, E., & Smallwood, J. (2020). Facing up to the wandering mind: Patterns of off-task laboratory thought are associated with stronger neural recruitment of right fusiform cortex while processing facial stimuli. Neuroimage, 214, 116765. https://doi.org/10.1016/j.neuroimage.2020.116765 

