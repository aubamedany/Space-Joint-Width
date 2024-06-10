import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim 
from torchmetrics import Recall
from torchmetrics import Accuracy
import time
from utils import *
best_model_params_path = "best_model.pth"


class GDModel(nn.Module):
    def __init__(self, pretrained_model_name="resnet18", num_classes=3, freeze=True):
        super(GDModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability
        self.base_model = models.resnet18(pretrained=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.accuracy = Accuracy(task="multiclass", num_classes=3)
        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False

            num_features_in = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(nn.Linear(num_features_in, num_classes),nn.Softmax(dim=1))

    def forward(self, x):
        x = x.to(self.device)
        x = self.base_model(x)
        return x
    def predict(self,x):
        self.train(False)
        x = x.to(self.device)
        outputs = self(x)
        y_preds = torch.argmax(outputs,dim=1)
        return y_preds
    def fitting(self,Xtrain,Ytrain, Xval,Yval,Xtest,Ytest ,num_epochs = 20):
        best_acc = 0.0
        t1 = time.time()
        Xtrain, Ytrain = shuffle(Xtrain,Ytrain)
        Xval, Yval = shuffle(Xval,Yval)
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            self.train(True)
            epoch_loss = 0.0
            for num,(batchX,batchY) in enumerate(minibatch(Xtrain,Ytrain,batch_size=32)):
                batchX = batchX.to(self.device)  
                batchY = batchY.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(batchX)
                loss = self.loss_func(outputs, batchY)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                y_preds = torch.argmax(outputs,dim=1)
            epoch_acc = self.accuracy(y_preds, Ytrain)
            t2 = time.time()
            epoch_train_time = t2 - t1
            # valid
            acc = self.evaluate(Xval,Yval)
            if  epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.state_dict(), best_model_params_path)
            print(f'Epoch{epoch} Training time:{epoch_train_time} Epoch_loss: {epoch_loss:.4f} Acc_train: {epoch_acc:.4f} Acc_valid: {acc:.4f}')
        acc_test = self.evaluate_test(Xtest,Ytest)
        print(f'Test Accuracy: {acc_test:.4f}')

    def evaluate(self,Xval,Yval):
        y_preds = self.predict(Xval)
        acc = self.accuracy(y_preds,Yval)
        return acc
    def evaluate_test(self,Xtest,Ytest):
        self.load_state_dict(torch.load(best_model_params_path))
        y_preds = self.predict(Xtest)
        acc = self.accuracy(y_preds,Ytest)
        return acc

        

