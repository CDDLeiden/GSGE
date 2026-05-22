# Clustering

Chemical space visualization and fragment clustering utilities.

## GSGE_clustering

::: GSGE.clustering.GSGE_clustering
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### t-SNE Visualization

```python
from GSGE import GSGE
from GSGE.clustering import GSGE_clustering

# Load GSGE with embeddings
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Create clustering analyzer
clustering = GSGE_clustering(gsge)

# Generate fragment embeddings
embeddings, graph_data = clustering._embed_fragments(
    frag_smiles=gsge.get_fragments_smiles(),
    device='cuda',
    batch_size=64
)

# 2D t-SNE plot
fig = clustering.plot_2D_TSNE(
    random_state=42,
    scatter_2d_args={'height': 800, 'width': 1600}
)
fig.show()

# 3D t-SNE plot
fig = clustering.plot_3D_TSNE(
    random_state=42,
    scatter_3d_args={'height': 800, 'width': 1600}
)
fig.show()
```

### UMAP Visualization

```python
from GSGE import GSGE
from GSGE.clustering import GSGE_clustering 

# Load GSGE with embeddings
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Create clustering analyzer
clustering = GSGE_clustering(gsge)

# 2D UMAP
fig = clustering.plot_2D_UMAP(
    random_state=42,
    scatter_2d_args={'height': 800, 'width': 1600}
)
fig.show()

# 3D UMAP
fig = clustering.plot_3D_UMAP(
    random_state=42,
    scatter_3d_args={'height': 800, 'width': 1600}
)
fig.show()
```

### MCS Clustering

```python
from GSGE import GSGE
from GSGE.clustering import GSGE_clustering 

# Load GSGE with embeddings
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Create clustering analyzer
clustering = GSGE_clustering(gsge)

# Cluster fragments by Maximum Common Substructure
clustering.plot_mcs_clusters(
    frag_smiles=gsge.get_fragments_smiles()[:100],
    threshold=0.5,
    save_path='mcs_clusters.png'
)
```

### Chemical Space Analysis

```python
import numpy as np
from sklearn.manifold import TSNE
from GSGE import GSGE
# Load GSGE with embeddings
gsge = GSGE(GSGE_load_path='gsge_save.pkl')

# Get embeddings
embeddings = gsge.get_fragment_embeddings()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# Analyze clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(tsne_result)

print(f"Found {len(np.unique(clusters))} clusters")
```
