import streamlit as st
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from retry import retry
import torch
from datasets import load_dataset
from sentence_transformers.util import semantic_search
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


## Lyrical Similarity Stuff

df = pd.read_excel('Taylor_Swift_Genius_Data.xlsx')

# Remove stopwords
sw_nltk = stopwords.words('english')
lyrics_no_stopwords = []
sw_nltk.remove('not')
for i in range(len(df['Lyrics'])):
    text = str(df['Lyrics'][i])
    words = [word for word in text.split() if word.lower() not in sw_nltk]
    new_text = " ".join(words)
    lyrics_no_stopwords.append(new_text)
df['Lyrics'] = lyrics_no_stopwords

model_id = "sentence-transformers/all-mpnet-base-v22"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

# Get the song lyrics as list
texts = list(df['Lyrics'])

@retry(tries=3, delay=10)
def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
      raise RuntimeError(
          "The model is currently loading, please re-run the query."
          )

# Convert lyrics to embeddings using library, then create pandas dataframe out of that
output = query(texts)
embeddings = pd.DataFrame(output)


# Saving embeddings to host
embeddings.to_csv("embeddings.csv", index=False)

faqs_embeddings = load_dataset('cardo14/Taylor_Swift_Embeddings_2')
dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

### End lyrical similarities

### Acoustic Similarity stuff

df_spotify = pd.read_excel('Taylor_Swift_Spotify_Data.xlsx')

# clean song names
clean_songs = []
for song in df_spotify['Song Name']:
  song = str(song)
  song = song.replace("‚Äô", "\'")  
  song = song.replace("‚Äò", "\'")
  clean_songs.append(song)
df_spotify['Song Name'] = clean_songs

X = df_spotify.iloc[:, 3:] # take only the numeric columns
# Scaling the data to have the same magnitue
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# Reduce down to two dimensions
pca = PCA(2)
transform_X = pca.fit_transform(data_scaled)

# Add points back to original dataframe
pca_X = pd.DataFrame(transform_X)
df_spotify["PCA_X"] = pca_X.iloc[:, 0]
df_spotify["PCA_Y"] = pca_X.iloc[:, 1]


### End acoustic similarities

# Streamlit stuff


# Identify songs that are in both dataframes
common_df = pd.merge(df_spotify, df, on = 'Song Name') # Merge into common dataframe
# This lane filters valid song options based on user input
valid_songs = common_df['Song Name'].tolist()

selected_song = st.selectbox("Select a song:", options= [''] + valid_songs)



col1, col2 = st.columns(2) # so the buttons can be side by side
# Place buttons in columns
button1 = col1.button('Generate Acoustic Similarities')
button2 = col2.button('Generate Lyrical Similarities')

# Lyrical Similarities
if button2:
  song = texts[df.index[df['Song Name'] == selected_song].tolist()[0]]
  output = query(song)

  query_embeddings = torch.FloatTensor(output)
  print(f"The size of our embedded dataset is {dataset_embeddings.shape} and of our embedded query is {query_embeddings.shape}.")

  hits = semantic_search(query_embeddings, dataset_embeddings, top_k=len(valid_songs))

  #Print songs that are most similar
  top_songs = [] # List to store top 5 songs
  top_scores = [] # list to store top 5 scores
  for i in range(1, len(hits[0])): # Skip the first one since that's the exact match
    lyric = texts[hits[0][i]['corpus_id']]
    top_songs.append(df[df["Lyrics"] == lyric]['Song Name'].values[0]) # store the songs in list
    top_scores.append(round(hits[0][i]['score'], ndigits= 2)) # store the corresponding scores in list
  # Create dataframe out of it
  top_5_df = pd.DataFrame(
    {
        "Song Names": top_songs,
        "Similarity Score": top_scores
    }
  )

  # Function to apply color mapping using seaborn's viridis palette
  def color_map(val, vmin, vmax, cmap, alpha = 0.7):
      normed_val = (val - vmin) / (vmax - vmin)
      #rounded_val = round(normed_val, 2)
      rgba = [int(255 * x) for x in cmap(normed_val)[:3]] + [alpha]
      return f'background-color: rgba{tuple(rgba)};'

  # Get min and max values for the 'Similarity Score' column
  min_val = top_5_df['Similarity Score'].min() #- (top_5_df['Similarity_Score'].min() - top_5_df['Similarity_Score'].max())/4
  max_val = top_5_df['Similarity Score'].max() #+ (top_5_df['Similarity_Score'].min() - top_5_df['Similarity_Score'].max())/4

  # Create a colormap using seaborn's viridis palette
  cmap = sns.color_palette("viridis", as_cmap=True)

  # Apply color mapping to the 'Similarity Score' column
  styled_df = top_5_df.style.applymap(lambda x: color_map(x, min_val, max_val, cmap), subset=['Similarity Score'])
  #styled_df = styled_df.format({'Similarity Score': '{:.2%}'}) # only two decimals

  # Display the styled DataFrame in Streamlit
  st.dataframe(styled_df)




# Acoustic Similarities
if button1:
  temp_df = df_spotify[df_spotify["Song Name"] == selected_song]
  song_array = np.array(temp_df[['PCA_X', 'PCA_Y']])
  # Create empty lists for songs and distances
  song_names = []
  song_distances= []

  # Add the songnames and distances to dictionary
  for i in range(len(df_spotify)):
    temp_array = np.array(df_spotify[df_spotify["Song Name"] == df_spotify.iloc[i]['Song Name']][['PCA_X', 'PCA_Y']])
    distance = euclidean_distances(song_array.reshape(1, -1), temp_array.reshape(1, -1))[0, 0]
    song_names.append(df_spotify.iloc[i]['Song Name'])
    song_distances.append(1 / (1 + distance)) # Convert distances to similarity score on scale from 0 to 1
  
  d = {'Song Names': song_names, 'Similarity Score': song_distances} # Make dictionary
  comparison_df = pd.DataFrame(d)
  comparison_df = comparison_df.sort_values('Similarity Score', ascending=False)  # sort by similarity score
  comparison_df = comparison_df.iloc[1:, :]


  # Function to apply color mapping using seaborn's viridis palette
  def color_map(val, vmin, vmax, cmap, alpha = 0.7):
      normed_val = (val - vmin) / (vmax - vmin)
      #rounded_val = round(normed_val, 2)
      rgba = [int(255 * x) for x in cmap(normed_val)[:3]] + [alpha]
      return f'background-color: rgba{tuple(rgba)};'

  # Get min and max values for the 'Similarity Score' column
  min_val = comparison_df['Similarity Score'].min() #- (top_5_df['Similarity_Score'].min() - top_5_df['Similarity_Score'].max())/4
  max_val = comparison_df['Similarity Score'].max() #+ (top_5_df['Similarity_Score'].min() - top_5_df['Similarity_Score'].max())/4

  # Create a colormap using seaborn's viridis palette
  cmap = sns.color_palette("viridis", as_cmap=True)

  # Apply color mapping to the 'Similarity Score' column
  styled_df = comparison_df.style.applymap(lambda x: color_map(x, min_val, max_val, cmap), subset=['Similarity Score'])
  #styled_df = styled_df.format({'Similarity Score': '{:.2%}'}) # only two decimals

  # Display the styled DataFrame in Streamlit
  st.dataframe(styled_df)
    

