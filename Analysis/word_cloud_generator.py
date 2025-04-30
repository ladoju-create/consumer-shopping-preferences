import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")

#text column is named 'text'
#text = " ".join(df["What improvements would make your online shopping experience better?"].dropna())  
text = " ".join(df["What improvements would make your in-store shopping experience better?"].dropna())  

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

text_cleaned = " ".join([word for word in text.lower().split() if word not in stop_words and word.isalpha()])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_cleaned)

#word cloud display
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  
plt.show()

