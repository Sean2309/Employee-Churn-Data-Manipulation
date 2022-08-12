from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import os
import pandas as pd
import matplotlib.pyplot as plt
from eda_backend_seaborn import str_replacing

sw = set(STOPWORDS)

def wordcloud_plot(viz_df: pd.DataFrame, dest_path: str, text_label: str = []):
    print("\ntext label iteration: " + str(text_label) + "\nwordcloud")
    text = " ".join(map(str, viz_df[text_label]))
    wordcloud = WordCloud(stopwords=sw, background_color="white", width=1600, height=800).generate(text)
    text_label = str_replacing(text_label)
    file_name = f"WordCloud_of_{text_label}.png"
    plt.figure( figsize=(19.1,10), facecolor="k")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(os.path.join(dest_path, file_name), facecolor="k", bbox_inches="tight")
    plt.close()