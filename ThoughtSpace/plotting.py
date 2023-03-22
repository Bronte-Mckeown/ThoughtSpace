from wordcloud import WordCloud
import os

def save_wordclouds(df,path):
    def color_func(font_size):
        print('e')
   
    arrmax = df.max().max()
    arrmin = -arrmax
    for col in df.columns:
        subdf = df[col].apply(lambda x: (x - arrmin) / (arrmax - arrmin))
        wc = WordCloud(background_color="white", colormap="RdBu_r", 
                            width=400, height=400, prefer_horizontal=1, 
                            min_font_size=8, max_font_size=200
                            )
        df_dict = subdf.to_dict()

        wc = wc.generate_from_frequencies(frequencies=df_dict)
        wc.to_file(os.path.join(path, col + ".png"))
