"""Dataset loading."""

import os
import pickle
import numpy as np

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]

def get_simi_file(data_path):
    file_fr, file_ext = os.path.splitext(data_path)
    simi_file = file_fr + "simi.npy"
    return simi_file


def load_data(dataset, normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y, simi_file = load_uci_data(dataset)
    else:
        x, y, simi_file = load_data_path(dataset)
        # raise NotImplementedError("Unknown dataset {}.".format(dataset))
        
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    print("Simi_file:", simi_file)
    if os.path.exists(simi_file):
        similarities = np.load(simi_file, allow_pickle=True)
        print("Finish loading!")
    else:
        x0 = x[None, :, :]
        x1 = x[:, None, :]
        
        print("Calculating cos...")
        cos = np.sum(x0 * x1, axis=-1)
        print("Finish cos!")
        
        similarities = 0.5 * (1 + cos)
        similarities = np.triu(similarities) + np.triu(similarities).T
        similarities[np.diag_indices_from(similarities)] = 1.0
        similarities[similarities > 1.0] = 1.0
        similarities = similarities.astype(np.double)
        
        np.save(simi_file, similarities)
    
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    
    return x, y, get_simi_file(data_path)


def load_data_path(data_path):
    # data_dict = np.load(data_path, allow_pickle=True)
    print("Loading", data_path, "...")
    data_dict = pickle.load(open(data_path, "rb"))
    print(data_dict, type(data_dict))
    z, label = data_dict["representations"], data_dict["label"]
    print(z.shape, label.shape)
    
    weight = np.array([100 ** i for i in range(label.shape[1])])[None, :]
    new_label = np.sum(label * weight, axis=1)
    
    return z, new_label, get_simi_file(data_path)