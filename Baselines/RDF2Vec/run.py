import pandas as pd

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker, WLWalker
import numpy as np

# from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
import argparse


def load_labels(path, e2id, t2id):
    labels = np.zeros((len(e2id), len(t2id)))
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    for i, row in df.iterrows():
        e_id, t_id = e2id[row['entity']], t2id[row['label_class']]
        labels[e_id, t_id] = 1
    return labels


def main(path, args):
    # Load the knowledge graph.
    print("Loading the knowledge graph...")
    if args.dataset == 'FB15k':
        kg = KG(path+'FB15k.nt',
            fmt='nt')
    elif args.dataset == 'YAGO43k':
        kg = KG(path+'YAGO43k.ttl',
            fmt='ttl')
    else:
        raise ValueError("Invalid dataset name!")

    data = pd.read_csv(path+"completeDataset.tsv", sep="\t", encoding="utf-8")
    entities = [entity for entity in data["entity"].unique()]
    t2id = {t:i for i,t in enumerate(data["label_class"].unique())}
    e2id = {entity: i for i, entity in enumerate(entities)}
    all_true = load_labels(path+"completeDataset.tsv", e2id, t2id) # shape: (n_entities, n_classes)
    # id2ent = {i: entity for i, entity in enumerate(entities)}

    print("Number of entities to embed: {}".format(len(entities)))

    # Create our transformer, setting the embedding & walking strategy.
    # # Option1: Random-Walker
    # transformer = RDF2VecTransformer(
    # Word2Vec(epochs=10, vector_size=500, window=10, alpha=0.025, negative=25, sg=1),
    # # walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=10)],
    # walkers=[RandomWalker(args.max_depth, args.n_walks, with_reverse=False, n_jobs=-1)],
    # verbose=2
    # )

    # Option2: WL-Walker
    transformer = RDF2VecTransformer(
    Word2Vec(epochs=10, vector_size=500, window=10, alpha=0.025, negative=25, sg=1),
    walkers=[WLWalker(args.max_depth, args.n_walks, with_reverse=False, n_jobs=-1, wl_iterations=4)],
    verbose=2
    )

    # Get our embeddings.
    print("Training the RDF2Vec transformer...")
    embeddings, _ = transformer.fit_transform(kg, entities)

    # After getting embeddings, we need to get train, valid, test datasets for SVM classifier
    df_train = pd.read_csv(path+"trainingSet.tsv", sep="\t", encoding="utf-8", index_col='entity')
    df_val = pd.read_csv(path+"validSet.tsv", sep="\t", encoding="utf-8", index_col='entity')
    df_test = pd.read_csv(path+"testSet.tsv", sep="\t", encoding="utf-8", index_col='entity')

    Entities_Groups_train = df_train.groupby('entity').agg(lambda x: list(x)) 
    Entities_Groups_val = df_val.groupby('entity').agg(lambda x: list(x))
    Entities_Groups_test = df_test.groupby('entity').agg(lambda x: list(x))

    X_train, X_test, X_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for ent in Entities_Groups_train.index:
        X_train.append(embeddings[e2id[ent]])
        y = np.zeros(len(t2id))
        for label in Entities_Groups_train.loc[ent]['label_class']:
            y[t2id[label]] = 1
        y_train.append(y)

    for ent in Entities_Groups_val.index:
        X_valid.append(embeddings[e2id[ent]])
        y = np.zeros(len(t2id))
        for label in Entities_Groups_val.loc[ent]['label_class']:
            y[t2id[label]] = 1
        y_valid.append(y)

    idx_test = []
    for ent in Entities_Groups_test.index:
        X_test.append(embeddings[e2id[ent]])
        idx_test.append(e2id[ent])
        y = np.zeros(len(t2id))
        for label in Entities_Groups_test.loc[ent]['label_class']:
            y[t2id[label]] = 1
        y_test.append(y)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_valid, y_valid = np.array(X_valid), np.array(y_valid)
    X_test, y_test = np.array(X_test), np.array(y_test)


    # Fit a Support Vector Machine on train embeddings and pick the best
    # C-parameters (regularization strength).
    C = [0.001, 0.01, 0.1, 1, 10]
    # Ensure the determinism of this script by initializing a pseudo-random number.
    print("Fitting the SVM classifier on train embeddings...")
    for coef in C:
        print("C = {}".format(coef))
        base_clf = LinearSVC(random_state=42, penalty='l2', loss='squared_hinge', C=args.C, dual=True, verbose=0, max_iter=100000)
        clf = OneVsRestClassifier(base_clf).fit(X_train, y_train)
        # calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid')
        # clf = MultiOutputClassifier(calibrated_clf).fit(X_train, y_train)

        # Evaluate the Support Vector Machine on test embeddings.
        print("Evaluating the SVM classifier on test embeddings...")
        # y_pred = np.hstack([arr[:, 1].reshape(-1, 1) for arr in clf.predict_proba(X_test)])
        y_pred = clf.decision_function(X_test)
        mrr, hit1, hit3, hit10 = evaluate(y_pred, y_test, all_true[np.array(idx_test)])
        print("MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}".format(mrr, hit1, hit3, hit10))


def evaluate(y_pred, y_true, all_true):
    '''
    @param: all_true: all true labels for input predicted entities, size: (num_input_entities, num_types)
            y_pred: predicted probabilities for input entities, size: (num_input_entities, num_types)
            y_true: true labels for input entities, size: (num_input_entities, num_types)
    '''
    mrr, hit1, hit3, hit10 = [], [], [], []
    row_ixs, col_ixs = y_true.nonzero()
    for i, ent_id in enumerate(tqdm(row_ixs)):
        scale = np.max(y_pred[ent_id]) - np.min(y_pred[ent_id])
        tmp = y_pred[ent_id] - all_true[ent_id] * scale
        tmp[col_ixs[i]] = y_pred[ent_id, col_ixs[i]]
        argsort = np.argsort(tmp)[::-1]
        ranking = (argsort == col_ixs[i]).nonzero()[0][0]
        # assert ranking.size(0) == 1
        ranking = ranking + 1
        mrr.append(1.0 / ranking)
        hit1.append(1.0 if ranking <= 1 else 0.0)
        hit3.append(1.0 if ranking <= 3 else 0.0)
        hit10.append(1.0 if ranking <= 10 else 0.0)
    return np.mean(mrr), np.mean(hit1), np.mean(hit3), np.mean(hit10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15k', help='dataset name')
    parser.add_argument('--max_depth', type=int, default=4, help='max depth of random walk')
    parser.add_argument('--n_walks', type=int, default=1, help='length of random walk')
    # parser.add_argument('--C', type=float, default=1.0, help='regularization strength')
    args=parser.parse_args()

    if args.dataset == 'FB15k':
        data_path = "data/FB15k/"
    elif args.dataset == 'YAGO43k':
        data_path = "data/YAGO43k/"
    else:
        raise ValueError("Invalid dataset name!")

    main(data_path, args)