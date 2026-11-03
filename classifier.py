"""Define a linear downstream classifier and its evaluation routine."""

import torch
from torch import nn
from torch.optim import Adam
from evaluation import BaseEvaluator
from sklearn.metrics import roc_auc_score,roc_curve,f1_score
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import copy



class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            Linear(num_features, num_classes),
            #Linear(num_features, 2048),
            #nn.ReLU(),
            #nn.Dropout(0.15),
            #Linear(2048, num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 200, learning_rate: float = 0.0005, test_interval: int = 1):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_interval = test_interval        

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, num_classes, device):
        val_aucs = []
        val_losses = []
        test_aucs = []
        test_losses = []
        train_losses = []
        train_aucs = []
        train_f1s = []
        val_f1s = []
        test_f1s = []
        best_val_auc = 0
        best_val_loss = 100000
        input_dim = X_train.shape[1]
        classifier = MLP(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate)
        if num_classes == 2:
            criterion = nn.BCELoss()
        else:
            class_counts = torch.tensor([(y_train == i).sum() for i in range(num_classes)], dtype=torch.float)
            weights = 1.0 / class_counts
            weights = weights / weights.sum()
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        
        for epoch in range(1,self.num_epochs):            
            classifier.train()
            optimizer.zero_grad()
            output = classifier(X_train)
            class_loss = criterion(output, F.one_hot(y_train, num_classes=num_classes).float())
            class_loss.backward()
            optimizer.step()
            if epoch % self.test_interval == 0:
                classifier.eval()
                with torch.no_grad():
                    output_train = classifier(X_train)
                    output_val = classifier(X_val)
                    output_test = classifier(X_test)
                    class_loss_train = class_loss.item()
                    class_loss_val = criterion(output_val, F.one_hot(y_val, num_classes=num_classes).float()).item()
                    class_loss_test = criterion(output_test, F.one_hot(y_test, num_classes=num_classes).float()).item()
                if num_classes == 2:
                    train_prob = output_train[:,-1].cpu().numpy()
                    val_prob = output_val[:,-1].cpu().numpy()
                    test_prob = output_test[:,-1].cpu().numpy()
                    train_auc = roc_auc_score(y_train.cpu().numpy(), train_prob)
                    val_auc = roc_auc_score(y_val.cpu().numpy(), val_prob)
                    test_auc = roc_auc_score(y_test.cpu().numpy(), test_prob)
                    fpr, tpr, thresholds = roc_curve(y_train.cpu().numpy(), train_prob)
                    youden_j = tpr - fpr
                    best_index = np.argmax(youden_j)
                    best_threshold = thresholds[best_index]
                    train_prob[train_prob<best_threshold] = 0
                    train_prob[train_prob>=best_threshold] = 1
                    train_f1 = f1_score(y_train.cpu().numpy(), train_prob)
                    fpr, tpr, thresholds = roc_curve(y_val.cpu().numpy(), val_prob)
                    youden_j = tpr - fpr
                    best_index = np.argmax(youden_j)
                    best_threshold = thresholds[best_index]
                    val_prob[val_prob<best_threshold] = 0
                    val_prob[val_prob>=best_threshold] = 1
                    val_f1 = f1_score(y_val.cpu().numpy(), val_prob)
                    fpr, tpr, thresholds = roc_curve(y_test.cpu().numpy(), test_prob)
                    youden_j = tpr - fpr
                    best_index = np.argmax(youden_j)
                    best_threshold = thresholds[best_index]
                    test_prob[test_prob<best_threshold] = 0
                    test_prob[test_prob>=best_threshold] = 1
                    test_f1 = f1_score(y_test.cpu().numpy(), test_prob)
                    
                    val_f1s.append(val_f1)
                    test_f1s.append(test_f1)
                    train_f1s.append(train_f1)
                else:
                    train_auc = roc_auc_score(y_train.cpu().numpy(), output_train.cpu().numpy(),
                                              multi_class='ovr', average='macro')
                    val_auc = roc_auc_score(y_val.cpu().numpy(), output_val.cpu().numpy(),
                                              multi_class='ovr', average='macro')
                    test_auc = roc_auc_score(y_test.cpu().numpy(), output_test.cpu().numpy(),
                                              multi_class='ovr', average='macro')
                val_aucs.append(val_auc)
                val_losses.append(class_loss_val)
                test_aucs.append(test_auc)
                test_losses.append(class_loss_test)
                train_aucs.append(train_auc)
                train_losses.append(class_loss_train)
               

                if class_loss_val < best_val_loss: 
                    best_train_auc = train_auc
                    best_val_auc = val_auc
                    best_test_auc = test_auc
                    best_train_loss = class_loss_train
                    best_val_loss = class_loss_val
                    best_test_loss = class_loss_test
                    if num_classes == 2:
                        best_train_f1 = train_f1
                        best_val_f1 = val_f1
                        best_test_f1 = test_f1
                    linear_state_dict = copy.deepcopy(classifier.state_dict())
                    
        print('')
        print(f'train_auc: {best_train_auc:.2f} | train_loss: {best_train_loss:.2f}')
        print(f'val_auc  : {best_val_auc:.2f} | val_loss  : {best_val_loss:.3f}')
        print(f'test_auc : {best_test_auc:.2f} | test_loss : {best_test_loss:.2f}')
        
        if num_classes == 2:
            return {
                'best_train_auc': best_train_auc,
                'best_val_auc': best_val_auc,
                'best_test_auc': best_test_auc,
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                'best_test_loss': best_test_loss,
                'best_train_f1': best_train_f1,
                'best_val_f1': best_val_f1,
                'best_test_f1': best_test_f1,
                'y_train': y_train.cpu().numpy(),
                'y_val': y_val.cpu().numpy(),
                'y_test': y_test.cpu().numpy(),
            }, linear_state_dict
        else:
            return {
                'best_train_auc': best_train_auc,
                'best_val_auc': best_val_auc,
                'best_test_auc': best_test_auc,
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                'best_test_loss': best_test_loss,
                'train_probs': output_train.cpu().numpy(),
                'val_probs': output_val.cpu().numpy(),
                'test_probs': output_test.cpu().numpy(),
                'y_train': y_train.cpu().numpy(),
                'y_val': y_val.cpu().numpy(),
                'y_test': y_test.cpu().numpy(),
            }, linear_state_dict



