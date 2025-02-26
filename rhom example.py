import pandas as pd
from ThoughtSpace.rhom import splithalf, omni_sample, dir_proj, bypc

df = pd.read_csv('output.csv')

# If not specifying a grouping variable, remember to specify only the data to be decomposed
splithalf_df = df.iloc[:, 2:11]

split_results = splithalf(df = splithalf_df,
                          npc = 4,
                          method = "promax",
                          boot = 1000,
                          file_prefix = "example_splithalf",
                          save = False)

# When conducting a direct-projection reproducibility analysis remember to specify the grouping variable whose levels you're comparing
dirproj_df = df.iloc[:,2:11]
dirproj_df['group'] = df['grouping variable']

dirproj_results = dir_proj(df = dirproj_df,
                           group = "group",
                           npc = 4,
                           method = "varimax",
                           folds = 5,
                           file_prefix = "example_directproject")

# An omnibus-sample reproducibility analysis can provide an alternative way of determining how robustly disparately sampled data can be blended
omsamp_results = omni_sample(df = dirproj_df,
                             group = 'group',
                             npc = 4,
                             method = "varimax",
                             boot = 1000,
                             file_prefix = "example_omsamp")

# If split-half reliability is strong enough, you can examine omnibus-sample reproducibility on a by-component level.
bypc_results = bypc(df = df,
                    group = 'group',
                    npc = 4,
                    method = "varimax",
                    file_prefix = "example_byPC")

