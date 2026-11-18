import numpy as np
import matplotlib.pyplot as plt

import umap.umap_ as umap
import umap.plot

def plot_projection(data, labels, model_name, random_state=None):
    """
    Project data using UMAP and plot
    """
    mapper = umap.UMAP( 
        n_neighbors=15,
        n_components=2, 
        metric='cosine', 
        min_dist=1,
        n_jobs=1, 
        random_state=random_state).fit(data)
    proj_data = mapper.transform(data)
 
    
    # replace labels with indices
    labels_dict = {label: i for i,label in enumerate(labels)}
    indices = [labels_dict[label] for label in labels]
    # plot data with samples colored according to labels
    ax = umap.plot.points(mapper, labels=np.array(indices))
    ax.get_figure().savefig(f"umap_projection_labelled_dat_{model_name}")