## **R-Homologue (RHom.)**
### *Comparing Components by the Way They Organize Observations*

Recall that when PCA captures patterns in your data, the resultant components are a new set of variables, capturing much of the meaningful variance in your data in a smaller number of dimensions. In other words, each component is effectively a new measure representing a certain combination of the original vairables on which each observation can be scored. For example, in the case of experience-sampling, a "Detailed Task-Focus" component capturing the positive associations between thoughts being detailed, deliberate, and on-task, can score each of the original thought probes based on how much they align with the "Detailed Task-Focus" pattern the component is capturing (i.e., how much they are  detailed, deliberate and task-focused). The scores of each observation on a given component are called "component scores".

These component scores offer another valuable way of testing component reliability, much more similar to the way researchers test the reliability of typical measures. You simply score the same set of observations according to two different components. If the components are similar measures, they should score the same observations in the same way.

To measure this we use a metric called *R-Homologue (RHom.)* which essentially represents the Pearson's correlation between the component scores generated between two components. The reason for the 'homologue' is that two datasets might produce the same components but in a different order of variance accounted for, so the first component in one solution may correspond to, say, the 3rd component in another solution. We call corresponding components across two solutions a 'homologous pair', with each component being the 'homologue' of the other. We usually define a homologue as the component in solution B whose component scores are most highly correlated with a given component in solution A (see Formula 1.1) (Chitiz et al., 2024; Mulholland et al., 2023). When estimating the average correlation across components between two solutions, we pair components according to what achieves the highest average RHom. between the two solutions (see Formula 1.2) (Chitiz et al., 2024).



**R-Homologue (RHom.)**

**(Formula 1.1)**

$R_{Hom.} = \arg{\max_{j}} \, R(\mathbf{S}_i^{(A)}, \mathbf{S}_j^{(B)})$

- $\mathbf{S}_i^{(A)}$: The component scores generated from the $i$-th principal component from solution $A$.
- $\mathbf{S}_j^{(B)}$: The component scores generated from the $j$-th principal component from solution $B$.
- $R(\mathbf{S}_i^{(A)}, \mathbf{S}_j^{(B)})$: The Pearson correlation coefficient between $\mathbf{S}_i^{(A)}$ and $\mathbf{S}_j^{(B)}$.


**(Formula 1.2)**

$`\overline{X}_{{R_{Hom.}}} = \max_{\pi} [\frac{1}{n} \sum R(\mathbf{S}_i^{(A)}, \mathbf{S}_{\pi(j)}^{(B)})]`$

- $\mathbf{S}_i^{(A)}$: The component scores generated from the $i$-th principal component from solution $A$.
- $\mathbf{S}_{\pi(j)}^{(B)}$: The component scores generated from the $j$-th principal component from solution $B$ in a given permutation ($\pi$).
- $`R(\mathbf{S}_i^{(A)}, \mathbf{S}_{\pi(j)}^{(B)})`$: The Pearson correlation coefficient between the component scores of $\mathbf{S}_i^{(A)}$ and $\mathbf{S}_j^{(B)}$.
- $\pi$ is a permutation of the indices $\{1, 2, \ldots, n\}$ representing the pairing of components from solution $A$ to components from solution $B$.
- $n$: The number of principal components in each solution.

To interpret RHom., we borrowed from guidelines regarding problematic multicollinearity in multiple regression that a correlation of $R_{Hom.} \ge .80$ as indicative of two components measuring a redundant phenomenon and thus achieving good similarity (Berry & Feldman, 1985; Chitiz et al., 2024).

## References

Berry, W. D., & Feldman, S. (1985). Multiple regression in practice. Sage Publications. https://doi.org/10.4135/9781412985208 

Chitiz, L., McKeown, B., Mulholland, B., Wallace, R., Goodall-Halliwell, I., Ho, N. S.-P., Konu, D., Poerio, G. L., Wammes, J., Milham, M. P., Klein, A., Jefferies, E., Leech, R., & Smallwood, J. (2024). Mapping cognition across lab and daily life using experience-sampling. PsyArXiv. https://doi.org/10.31234/osf.io/yjqv2

Mulholland, B., Goodall-Halliwell, I., Wallace, R., Chitiz, L., McKeown, B., Rastan, A., Poerio, G. L., Leech, R., Turnbull, A., Klein, A., Milham, M., Wammes, J. D., Jefferies, E., & Smallwood, J. (2023). Patterns of ongoing thought in the real world. Consciousness and Cognition, 114, 103530. https://doi.org/10.1016/j.concog.2023.103530 






