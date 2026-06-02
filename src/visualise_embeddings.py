import chromadb
import umap
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE



client = chromadb.PersistentClient(path="chroma_db_2")
collection = client.get_collection("langchain")

data = collection.get(include=["embeddings", "documents", "metadatas"])

embeddings = np.array(data["embeddings"])
documents = data["documents"]
sources = [m.get("source", "unknown").split("/")[-1] for m in data["metadatas"]]

print(f"Loaded {len(embeddings)} chunks, each {embeddings.shape[1]} dimensions")
print("Reducing to 2D with TSNE...")

tsne = TSNE(n_components=2, perplexity=2, random_state=42)
coords = tsne.fit_transform(embeddings)

# Build dataframe for plotting
df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "source": sources,
    "preview": [doc[:80] + "..." for doc in documents]
})

# Plot
fig = px.scatter(
    df, x="x", y="y",
    color="source",
    hover_data=["preview"],
    title="Feynman Knowledge Base - Embedding Space",
    width=1000, height=700
)

fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.show()