#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import umap


def umap_grid(hyperparam_dict, features_scaled, random_state=69):
    n_neighbors_list = hyperparam_dict["n_neighbors"]
    min_dist_list = hyperparam_dict["min_dist"]
    x_dim = len(n_neighbors_list)
    y_dim = len(min_dist_list)
    embedding_array = np.zeros(shape=(x_dim, y_dim), dtype="object")

    for x_index, n_neighbors in enumerate(n_neighbors_list):
        for y_index, min_dist in enumerate(min_dist_list):
            reducer = umap.UMAP(
                random_state=random_state,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
            )
            mapper = reducer.fit(features_scaled)
            embedding = mapper.embedding_
            embedding_array[x_index, y_index] = embedding

    return embedding_array
