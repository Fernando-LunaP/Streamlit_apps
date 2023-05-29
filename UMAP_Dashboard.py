import plotly.graph_objects as go
import numpy as np
import streamlit as st
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np

# Set seed
default_seed_value = 10
default_noise_value = 0.40

st.sidebar.markdown('## Dataset config.')
seed_value = st.sidebar.slider('Dataset', 1, 20, default_seed_value)
noise_level = st.sidebar.slider('Noise level', 0.1, 0.8, default_noise_value)


np.random.seed(seed_value)

def generate_clusters(num_clusters, num_points_per_cluster, noise_level):
    clusters = []
    labels = []
    for i in range(num_clusters):
        center = np.random.randn(3)
        cluster_points = center + np.random.randn(num_points_per_cluster, 3) * noise_level
        clusters.append(cluster_points)
        labels.extend([i] * num_points_per_cluster)
    return np.vstack(clusters), np.array(labels)

def plot_clusters(clusters, labels):
    fig = go.Figure()

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    colors = [
    '#E53935',  # Rojo intenso
    '#8E24AA',  # Morado intenso
    '#43A047',  # Verde intenso
    '#F06292',  # Rosa intenso
    '#FF8F00',  # Naranja intenso
    '#039BE5',  # Azul intenso
    '#9E9E9E'   # Gris intenso
]


    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_points = clusters[cluster_indices]
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            name= str(i+1),
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i % len(colors)],
                opacity=0.8
            )
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title="Point clound in a 3-dimensional space",
        width=800, height=800,
        legend=dict(title='Clusters',
        font=dict(size=18)  # Tamaño de fuente de la leyenda
    ),
        showlegend=True,
    )

    return fig

# Generate clusters
num_clusters = 7
num_points_per_cluster = 200
clusters, labels = generate_clusters(num_clusters, num_points_per_cluster, noise_level)

# Plot clusters
graf = plot_clusters(clusters, labels)

# UMAP parameters
default_n_neighbors = 20
default_min_dist = 0.6

# Sidebar controls for UMAP parameters
st.sidebar.markdown('## Umap Parameters.')
n_neighbors = st.sidebar.slider('n_neighbors', 5, 50, default_n_neighbors)
min_dist = st.sidebar.slider('min_dist', 0.1, 1.0, default_min_dist, step=0.1)

# UMAP dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
umap_data = reducer.fit_transform(clusters)

# Scatterplot with Seaborn (3D figure)
fig_scatter = plt.figure()
ax_scatter = fig_scatter.add_subplot(111, projection='3d')
scatterplot_scatter = ax_scatter.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], c=labels, cmap='Set1')
ax_scatter.set_xlabel('Dimension 1')
ax_scatter.set_ylabel('Dimension 2')
ax_scatter.set_zlabel('Dimension 3')
ax_scatter.set_title('3D Figure Scatterplot')

colors = [
    '#E53935',  # Rojo intenso
    '#8E24AA',  # Morado intenso
    '#43A047',  # Verde intenso
    '#F06292',  # Rosa intenso
    '#FF8F00',  # Naranja intenso
    '#039BE5',  # Azul intenso
    '#9E9E9E'   # Gris intenso
]


# Scatterplot with Seaborn (UMAP)
fig_umap, ax_umap = plt.subplots()
scatterplot_umap = sns.scatterplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=labels, palette=colors, ax=ax_umap, alpha=0.9)
scatterplot_umap.set_xlabel('UMAP Dimension 1')
scatterplot_umap.set_ylabel('UMAP Dimension 2')
scatterplot_umap.set_title('UMAP Dimensionality Reduction')
scatterplot_umap.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
                        labels=['1', '2', '3', '4', '5', '6', '7'])

# Streamlit app
st.title('Visualizing how UMAP works')
st.markdown('This dashboard contains a 3D scatterplot of random clusters and a visualization of UMAP dimensionality reduction.')
st.sidebar.markdown(f'Dataset: {seed_value}')
st.sidebar.markdown(f'Noise Level: {noise_level}')
st.sidebar.markdown(f'n_neighbors: {n_neighbors}')
st.sidebar.markdown(f'min_dist: {min_dist}')

# Display the scatterplots in Streamlit
#st.pyplot(fig_scatter)
st.plotly_chart(graf, use_container_width=True)
st.pyplot(fig_umap)