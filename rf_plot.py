import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def get_feature(f, total_features):
    return (f + total_features) % total_features


def get_all_paths(prefix, i, left, right, classes, features, features_names):
    if left[i] == -1:
        return [prefix + ["_Term_"]], [classes[i]]
    l = [get_all_paths(prefix + [features_names[features[i]]], mas[i],
                       left, right, classes, features, features_names)
         for mas in [left, right]]
    return l[0][0] + l[1][0], l[0][1] + l[1][1]


def pad_paths(paths, max_depth):
    for i in range(len(paths)):
        paths[i] = paths[i] + ["_Term_" for _ in range(1 + max_depth - len(paths[i]))]


def get_tree_paths(tree, features_names, max_depth):
    features = [get_feature(f, len(features_names)) for f in tree.feature]
    left = tree.children_left
    right = tree.children_right
    classes = [np.argmax(elem[0]) for elem in tree.value]
    paths, cls = get_all_paths([], 0, left, right, classes, features, features_names)
    pad_paths(paths, max_depth)
    return paths, cls


def get_forest_paths(forest, features_names, max_depth):
    paths, cls = [], []
    for tree in forest.estimators_:
        new_paths, new_cls = get_tree_paths(tree.tree_, features_names, max_depth)
        paths = paths + new_paths
        cls = cls + new_cls
    return paths, cls


def plot_paths(paths, cls, max_depth, colors, alpha):
    _paths = [['_Term_' for _ in range(max_depth + 1)]] + paths
    _cls = ['_1'] + cls
    pd_paths = pd.DataFrame(_paths)
    pd_paths['Class'] = _cls
    plt.figure(figsize=(10, 10))
    parallel_coordinates(pd_paths, 'Class', color=['white'] + colors, alpha=alpha)
    plt.show()


def plot_random_forest(random_forest, features_names, colors=None, alpha=0.2):
    """
    Draw all paths in random forest trees on parallel coordinates plot.
    :param random_forest: fitted sklearn.ensemple.RandomForestClassifier.
    :param features_names: names of features in data, fitted in random_forest.
    :param colors: list of colors for classes in data. If not specified, uniform grey will be used.
    :param alpha: transparency of lines in plot.
    """
    paths, cls = get_forest_paths(random_forest, features_names, random_forest.max_depth)
    if colors is None:
        colors = ['grey' for _ in range(len(features_names))]
    plot_paths(paths, cls, random_forest.max_depth, colors, alpha=alpha)
