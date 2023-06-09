from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('articles.csv', encoding='latin1')
df = df[df['soup'].notna()]
count = CountVectorizer(stop_words='english')
count_metrics = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_metrics, count_metrics)

df = df.reset_index()
indices = pd.Series(df.index, index=df['title'])
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)

    sim_scores = sim_scores[1:11]
    article_indices = [i[0] for i in sim_scores]
    return df[['title', 'lang', 'total_events', 'timestamp']].iloc[article_indices].values.tolist()