import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import os

class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2, hidden_dim=128):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class TeacherModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Useful functions
# For loading ConnectE embedding model For FB15k/YAGO43k Dataset
def load_embedding_model(Embedding_Path):
    ConnectE_embedding = {}
    
    with open(Embedding_Path,'r') as inf:
        for line in inf:
            lisplit=line.split('\t')
            ConnectE_embedding[lisplit[0]]=eval(lisplit[1])
    return ConnectE_embedding


def load_labels(paths, e2id, t2id):
    labels = np.zeros((len(e2id), len(t2id)))
    for path in paths:
        with open(path, encoding='utf-8') as r:
            for line in r:
                e, t = line.strip().split('\t')
                e_id, t_id = e2id[e], t2id[t]
                labels[e_id, t_id] = 1
    return labels


def load_dataset(PATH, Embedding_Path):
    # load pretrained embeddding, YAGO43k embedding.    
    entity_embeddings = load_embedding_model(Embedding_Path)

    # Get ground-truth labels for YAGO43k entities
    freebase_ttrain=pd.read_csv(PATH+'/Entity_Type_train.txt', sep='\t', header=None)
    freebase_ttest=pd.read_csv(PATH+'/Entity_Type_test.txt', sep='\t', header=None)
    freebase_tvalid=pd.read_csv(PATH+'/Entity_Type_valid.txt', sep='\t', header=None)
    fb_df= pd.concat([freebase_ttrain, freebase_tvalid, freebase_ttest])

    # entities dict and types dict
    e2id = {e: i for i, e in enumerate(fb_df[0].unique())}
    id2e = {i: e for e, i in e2id.items()}
    t2id = {t: i for i, t in enumerate(fb_df[1].unique())}

    # load labels
    all_true = load_labels([
        os.path.join(PATH, 'Entity_Type_train.txt'), 
        os.path.join(PATH, 'Entity_Type_valid.txt'), 
        os.path.join(PATH, 'Entity_Type_test.txt')
        ], e2id, t2id) # shape: (num_entities, num_types)
    
    y_train = load_labels([os.path.join(PATH, 'Entity_Type_train.txt')], e2id, t2id)
    train_id = y_train.sum(axis=1).nonzero()[0]
    y_train = y_train[train_id] # shape: (num_train, num_types)

    y_valid = load_labels([os.path.join(PATH, 'Entity_Type_valid.txt')], e2id, t2id)
    valid_id = y_valid.sum(axis=1).nonzero()[0]
    y_valid = y_valid[valid_id]

    y_test = load_labels([os.path.join(PATH, 'Entity_Type_test.txt')], e2id, t2id)
    test_id = y_test.sum(axis=1).nonzero()[0]
    y_test = y_test[test_id]
    
    # load embeddings
    X_train = np.array([entity_embeddings[id2e[eid]] for eid in train_id]) # shape: (num_train, dim)
    X_valid = np.array([entity_embeddings[id2e[eid]] for eid in valid_id])
    X_test = np.array([entity_embeddings[id2e[eid]] for eid in test_id])

    return X_train, y_train, X_valid, y_valid, X_test, y_test, all_true, train_id, valid_id, test_id


def train_teacher_model(X_l, y_l, B_test, y_test, B_valid, y_valid, all_true, dim=128, batch_size=128, n_epochs=100, n_patience=3, lr=0.001):

    input_dim = X_l.shape[1]
    output_dim = y_l.shape[1]
    hidden_dim = dim

    model = TeacherModel(input_dim, output_dim, hidden_dim)
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_data = TensorDataset(X_l, y_l)
    valid_data = TensorDataset(B_valid, y_valid)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(n_epochs):
        print('Epoch:', epoch)
        # Training loop
        model.train()
        for batch_X, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                valid_losses.append(loss.item())

        avg_valid_loss = sum(valid_losses) / len(valid_losses)

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= n_patience:
            break

    model.load_state_dict(best_model_state)

    # Assuming evaluate is already defined
    if torch.cuda.is_available():
        B_test = B_test.cuda()
    DNN_pred = model(B_test).cpu()
    mrr, hit1, hit3, hit10 = evaluate(DNN_pred, y_test, all_true)
    print('teacher model results:', mrr.item(), hit1.item(), hit3.item(), hit10.item())

    return model


def train_student_model(x_l, x_u, x_valid, y_l, y_teacher, y_valid, dropout=0.2, batch_size=128, lr=0.001):

    BATCH_SIZE = batch_size
    EPOCHS = 100 # Adjust if necessary
    n_patience = 10 # Adjust if necessary
    n_iterations = 5000

    student_model = StudentModel(x_l.size(1), y_l.size(1), dropout_rate=dropout, hidden_dim=128)
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        student_model = student_model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    teacher_pred = y_teacher  # Initialize with initial teacher's predictions
    x_train = torch.cat((x_l, x_u), 0)
    y_train = torch.cat((y_l, teacher_pred), 0)

    valid_data = TensorDataset(x_valid, y_valid)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    max_valid_f1 = 0
    stop_count = 0 # early stopping counter for number of iterations

    for i in range(n_iterations):

        train_data = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        if stop_count > 10: # here to change the stop count
            break
        
        best_loss = float('inf')
        for epoch in range(EPOCHS):
            # Training loop
            student_model.train()
            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = student_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation loop
            student_model.eval()
            valid_losses = []
            with torch.no_grad():
                for batch_X, batch_y in valid_loader:
                    if torch.cuda.is_available():
                        batch_X = batch_X.cuda()
                        batch_y = batch_y.cuda()
                    outputs = student_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    valid_losses.append(loss.item())
            
            avg_valid_loss = sum(valid_losses) / len(valid_losses)
            if avg_valid_loss <= best_loss:
                best_loss = avg_valid_loss
                best_model_state = student_model.state_dict()
                patience_counter = 0 # early stopping counter for student model in each iteration
            else:
                patience_counter += 1
            
            if patience_counter >= n_patience:
                print('Finishing at epoch', epoch)
                break
        
        # load the best model of current iteration
        student_model.load_state_dict(best_model_state)
        # evaluate the current model
        student_model.eval()
        with torch.no_grad():
            predict, y_true = [], []
            for batch_X, batch_y in valid_loader:
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                outputs = student_model(batch_X).cpu()
                predict.append(outputs)
                y_true.append(batch_y)
            # concat the predictions and labels
            predict = torch.cat(predict, dim=0)
            y_true = torch.cat(y_true, dim=0)
            predict = torch.where(predict <= 0.5, torch.tensor(0, dtype=torch.int).to(predict.device), torch.tensor(1, dtype=torch.int).to(predict.device))
            f1 = f1_score(y_true.cpu().numpy(), predict.cpu().numpy(), average='samples')

        print(i, 'Validation F1:', f1)

        if f1 >= max_valid_f1:
            max_valid_f1 = f1
            stop_count = 0
            # Save the model
            best_model = student_model.state_dict()
        else:
            stop_count += 1
            print('Validation F1 is decreasing, start early stopping', 'count:', stop_count)

        # Update teacher predictions and x_train, y_train
        student_model.eval()
        x_u_loader = DataLoader(x_u, batch_size=batch_size, shuffle=False)
        teacher_pred_list = []
        x_u_list = []
        with torch.no_grad():
            for batch_x_u in x_u_loader:
                if torch.cuda.is_available():
                    batch_x_u = batch_x_u.cuda()
                batch_teacher_pred = student_model(batch_x_u)
                x_u_list.append(batch_x_u.cpu())
                teacher_pred_list.append(batch_teacher_pred.cpu())
        teacher_pred = torch.cat(teacher_pred_list, dim=0)
        x_u = torch.cat(x_u_list, dim=0)
        x_train = torch.cat((x_l, x_u), 0)
        y_train = torch.cat((y_l, teacher_pred), 0)

    # Load the best model
    student_model.load_state_dict(best_model)

    return student_model


# ranking-based metrics
def evaluate(y_pred, y_true, all_true):
    '''
    Calculates the MRR/ Hits@1 / Hits@3 / Hits@10 scores for the given predictions and targets
    Args:
        y_pred: torch.LongTensor. size[num_test_entities, num_classes]
        y_true: torch.LongTensor. size[num_test_entities, num_classes], only classes in test set are labeled as 1
        all_true: torch.LongTensor. size[num_test_entities, num_classes], all true labels for input predicted entities
    '''
    mrr, hit1, hit3, hit10 = [], [], [], []
    row_ixs, col_ixs = torch.nonzero(y_true, as_tuple=True)
    for i, ent_id in enumerate(tqdm(row_ixs)):
        scale = y_pred[ent_id].max() - y_pred[ent_id].min()
        tmp = y_pred[ent_id] - all_true[ent_id] * scale
        tmp[col_ixs[i]] = y_pred[ent_id, col_ixs[i]]
        argsort = torch.argsort(tmp, descending=True)
        ranking = (argsort == col_ixs[i]).nonzero()
        assert ranking.size(0) == 1
        ranking = ranking.item() + 1
        mrr.append(1.0 / ranking)
        hit1.append(1.0 if ranking <= 1 else 0.0)
        hit3.append(1.0 if ranking <= 3 else 0.0)
        hit10.append(1.0 if ranking <= 10 else 0.0)
    return torch.tensor(mrr).mean(), torch.tensor(hit1).mean(), torch.tensor(hit3).mean(), torch.tensor(hit10).mean()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', help='Dataset FB15kET or YAGO43kET.', type=str, default='FB15kET')
    parser.add_argument('--dim', help='Dimension of the hidden units.', type=int, default=128)
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=128)
    parser.add_argument('--labeled_size', help='Size of labeled data.', type=float, default=0.01)
    parser.add_argument('--seed', help='Random seed.', type=int, default=47)
    parser.add_argument('--epochs', help='Number of epochs.', type=int, default=100)
    parser.add_argument('--patience', help='Patience for early stopping.', type=int, default=3)
    parser.add_argument('--dropout', help='Dropout rate.', type=float, default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float, default=0.001)
    args=parser.parse_args()
    
    # hyperparameters
    batch_size = args.batch_size
    labeled_size = args.labeled_size
    dim = args.dim
    seed = args.seed
    n_epochs = args.epochs
    n_patience = args.patience
    drop = args.dropout
    lr = args.learning_rate

    # load the dataset
    print('Loading dataset...')
    if args.Dataset == 'FB15kET':
        path = 'data/FB15kET'
        embedding_path = 'data/FB15kET/FB15K-ConnectE.txt'
    elif args.Dataset == 'YAGO43kET':
        path = 'data/YAGO43kET'
        embedding_path = 'data/YAGO43kET/YAGO-ConnectE.txt'
    else:
        ValueError('Dataset is incorrect, please choose datasets from [FB15kET, YAGO43kET].')
    X_train, y_train, X_valid, y_valid, X_test, y_test, all_true, train_id, valid_id, test_id = load_dataset(path, embedding_path)

    # train-valid-test dataset split into label-unlabelled (x_,x_u), (y_l, y_u)
    X_l, X_u, y_l, y_u = train_test_split(X_train, y_train, train_size=labeled_size, random_state=seed)
    
    # change numpy to tensor
    X_l = torch.tensor(X_l, dtype=torch.float)
    X_u = torch.tensor(X_u, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_valid = torch.tensor(X_valid, dtype=torch.float)
    y_l = torch.tensor(y_l, dtype=torch.float)
    y_u = torch.tensor(y_u, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    y_valid = torch.tensor(y_valid, dtype=torch.float)
    all_true = torch.tensor(all_true, dtype=torch.float)

    # train our semi-supervisied model, and print the evaluation results.
    print('Training teacher model...')
    dnn_model = train_teacher_model(X_l, y_l, X_test, y_test, X_valid, y_valid, all_true[test_id], dim=dim, batch_size=batch_size, n_epochs=n_epochs, n_patience=n_patience, lr=lr)
    
    # load data within certain batch size
    x_u_loader = DataLoader(X_u, batch_size=batch_size, shuffle=False)
    dnn_model.eval()
    y_teacher = []
    with torch.no_grad():
        for batch_x_u in x_u_loader:
            if torch.cuda.is_available():
                batch_x_u = batch_x_u.cuda()
            y_teacher.append(dnn_model(batch_x_u).cpu())
    y_teacher = torch.cat(y_teacher, dim=0)
    # train student model
    print('Training student model...')
    final_model = train_student_model(X_l, X_u, X_valid, y_l, y_teacher, y_valid, dropout=drop, batch_size=batch_size, lr=lr)
    
    print ('# Training is done, now evaluation.#')
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    final_model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_X in test_loader:
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
            y_pred.append(final_model(batch_X).cpu())
    y_pred = torch.cat(y_pred, dim=0)
    mrr, hit1, hit3, hit10 = evaluate(y_pred, y_test, all_true[test_id])
    print('Final results:', mrr, hit1, hit3, hit10)
