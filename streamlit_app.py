import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import base64
from sklearn.manifold import TSNE
import plotly.express as px

# Set the title of the app
st.title('PCA and t-SNE Visualization')

# Sidebar for parameter inputs
st.sidebar.header('Parameters')

# File uploader for the dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    num_rows = df.shape[0]

    # Ensure the required columns are present
    required_columns = [f'vector_{i}' for i in range(384)]

    tsne_dimension_1 = df.loc[0:num_rows - 1, 'tsne_dimension_1'].values
    tsne_dimension_2 = df.loc[0:num_rows - 1, 'tsne_dimension_2'].values
    clusters = df.loc[0:num_rows - 1, 'Cluster'].values
    headlines = df.loc[0:num_rows - 1, 'Headlines'].values

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        tsne_dimension_1,
        tsne_dimension_2,
        c=clusters,
        s=10,  # Reduced marker size for performance
        cmap='viridis',
        alpha=0.7
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
    ax.set_title('2D t-SNE Visualization with Clustering', fontsize=20)
    fig.colorbar(scatter, ax=ax, label='Cluster Label')

    st.pyplot(fig)

    fig = px.scatter(
        df,
        x='tsne_dimension_1',
        y='tsne_dimension_2',
        color='Cluster',
        hover_data=['Headlines'],  # Include 'Headlines' for hover information
        title='2D t-SNE Visualization with Clustering'
    )
    # Customize axis labels, title, and colorbar
    fig.update_layout(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        title_font_size=20,
        hoverlabel=dict(font=dict(size=18)),
        coloraxis_colorbar=dict(
            title='Cluster Label'
        )
    )
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
else:
    st.warning("Please upload a CSV file.")
    st.stop()