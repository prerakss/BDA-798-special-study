import nltk
from feature_engineering import df_temp
from data_transformation import genome_scores_df, genome_tags_df
import textblob
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer


# Based on some threshold on relevance score, trying to find polarity of tags which are most relevant
genome_sorted = genome_scores_df.sort_values(['movieId','relevance'],ascending=False)
test = genome_sorted.merge(genome_tags_df, how='inner', left_on = 'tagId', right_on = 'tagId')

# testing for 1 movie_id having relevance tag score higher than 0.75
test1 = test[(test['movieId']==1) & (test['relevance'] > 0.75)]
test2 = test1['tag'].to_numpy()

for i in test1['tag']:
  polarity = textblob.TextBlob(i).polarity
  print(i,polarity)

# RESULT: Since tags are not sentences, textblob polarity returns 0.0 for a lot of tags

'''For every movie the genome_scores_df has a relevance score for every tag.
The idea is to extract top 30 most relevant tags for every movie, and then using KmeansClusterer,
group tags that are similar into same cluster. For instance, if there are 8 clusters, potentially
there could be 8 new predictors which could help us predict the lifetime_gross (response variable)
'''
# testing for 1 movie_id and their 30 most relevant tag
movie_tags_temp = genome_sorted.groupby('movieId').head(30)
movie_tags_30 = movie_tags_temp.merge(genome_tags_df, how='inner', left_on='tagId', right_on='tagId')
movieId_tags = movie_tags_30.groupby('movieId')['tag'].apply(list).reset_index()
words = movieId_tags['tag']

model = Word2Vec(words, min_count=1)

X = model[model.wv.vocab]
kclusterer = KMeansClusterer(8, distance=nltk.cluster.util.cosine_distance)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

words = list(model.wv.vocab)
for i, word in enumerate(words):
    print (word + ":" + str(assigned_clusters[i]))

'''RESULT: Not sure if I should include these 8 clusters as predictors, since for instance the KmeansClusterer
somehow puts tags like 'oscar (best foreign language film)', 'china', 'colourful' in the same cluster!
'''

