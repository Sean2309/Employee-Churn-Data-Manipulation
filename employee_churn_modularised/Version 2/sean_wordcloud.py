import nltk
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords 

text = " ".join(i for i in df.Description)
sw = stopwords.words("A")
additional_sw = ["^", "<", ">", "_"]
df["Processed_Status"] = df["Status"].apply(lambda x: preprocess_text(x.lower(), sw, additional_sw)) 
# word_cloud = WordCloud(stopwords=stopwords,collocations = False, background_color = "white").generate(text)
# plt.figure( figsize=(15,10))
# plt.imshow(word_cloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()