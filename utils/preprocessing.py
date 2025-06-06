# compute the odds ratio for each feature of X_train

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif
from copy import deepcopy
import numpy as np
import pandas as pd

def preprocess(X_train,y_train,scaling,selection_type,quantile,to_keep=200):

    if selection_type == 'correlation':
        X_train = preprocess_correlation(X_train, y_train, quantile)

    elif selection_type == 'mutual_info':
        X_train = preprocess_mutual_info(X_train, y_train, quantile)
    
    if scaling == 'standardization':
        X_train = preprocess_standardization(X_train)
    elif scaling == 'normalization':
        X_train = preprocess_normalization(X_train)

    return preprocess_PCA(X_train,to_keep)

def preprocess_odd_ratios(X_train, y_train, to_keep=200):

    # create a pipeline with a standard scaler and a logistic regression
    pipe = make_pipeline(StandardScaler(), LogisticRegression())

    X_train_std = deepcopy(X_train)
    y_train_std = deepcopy(y_train)

    # remove the last 6 features of X_train_std
    X_train_std = X_train_std.iloc[:, :-6]

    # fit the pipeline on X_train and y_train
    pipe.fit(X_train_std, y_train_std)

    # get the coefficients of the logistic regression
    coefs = pipe.named_steps['logisticregression'].coef_[0]

    # compute the odds ratio
    odds_ratio = pd.Series(index=X_train_std[:5000].columns, data=coefs).apply(lambda x: 2 ** x)

    # order the odds ratio
    odds_ratio = odds_ratio.sort_values(ascending=False)

    # keep the 1000 features with the highest odds ratio
    X_train_selected = X_train[odds_ratio.index[:1000]]

    # sort the features by their name
    X_train_selected = X_train_selected.sort_index(axis=1)

    # take the std of the 1000 first features
    stds_selected = X_train_selected.std().sort_values(ascending=False)

    # take the 200 features with the highest std
    threshold = stds_selected.iloc[to_keep-1]

    # keep the features with a std higher than the threshold
    X_train_selected = X_train_selected.loc[:, stds_selected >= threshold]

    X_train_selected = StandardScaler().fit_transform(X_train_selected)

    return pd.DataFrame(data=X_train_selected, index=X_train.index)

def preprocess_correlation(X_train, y_train, quantile=0.75):

    genes_expressions = X_train.iloc[:, :5000].columns
    others_expressions = X_train.iloc[:, 5000:].columns

    correlations = []
    for gene in genes_expressions:
        correlation = np.abs(X_train[gene].corr(y_train))
        correlations.append(correlation)

    corr_df = pd.DataFrame({'correlations': correlations, 'genes': genes_expressions})

    corr_df.head()
    threshold = np.quantile(corr_df['correlations'], quantile)

    # keep the genes with a correlation greater than 0.3
    genes = corr_df[corr_df['correlations'] > threshold]['genes']

    print(f'Number of genes with a correlation greater than {threshold}: {len(genes)}')

    X_train_copy = deepcopy(X_train)
    X_train_selected = X_train[genes].join(X_train_copy[others_expressions])

    return X_train_selected

def preprocess_normalization(X_train):

    numerical_features = X_train.select_dtypes(exclude=['category']).columns

    X_train_normalized = normalize(X_train[numerical_features])

    return pd.DataFrame(data=X_train_normalized, index=X_train.index)

def preprocess_standardization(X_train):

    numerical_features = X_train.select_dtypes(exclude=['category']).columns

    X_train_standardized = StandardScaler().fit_transform(X_train[numerical_features])

    return pd.DataFrame(data=X_train_standardized, index=X_train.index)

def preprocess_mutual_info(X_train, y_train, quantile=0.75):

    genes_expressions = X_train.iloc[:, :5000].columns
    others_expressions = X_train.iloc[:, 5000:].columns

    mutual_infos = []
    for gene in genes_expressions:
        mutual_info = np.abs(mutual_info_classif(X_train[gene].values.reshape(-1, 1), y_train))
        mutual_infos.append(mutual_info[0])

    mi_df = pd.DataFrame({'mutual_infos': mutual_infos, 'genes': genes_expressions})

    mi_df.head()
    threshold = np.quantile(mi_df['mutual_infos'], quantile)

    # keep the genes with a mutual info greater than 0.3
    genes = mi_df[mi_df['mutual_infos'] > threshold]['genes']

    print(f'Number of genes with a mutual info greater than {threshold}: {len(genes)}')

    X_train_selected = X_train[genes].join(X_train[others_expressions])

    return X_train_selected

def preprocess_PCA(X_train,to_keep=200):

    X_train_std = deepcopy(X_train)

    if len(X_train_std) < to_keep:
        to_keep = len(X_train_std)

    # fit a PCA on X_train_std
    pca = PCA(n_components=to_keep)

    X_train_pca = pca.fit_transform(X_train_std)

    # create a dataframe with the PCA components
    X_train_pca = pd.DataFrame(data=X_train_pca, index=X_train.index)

    return X_train_pca
