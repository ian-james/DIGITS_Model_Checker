import os
import torch
import torch

import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
import torchvision
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def train_MNIST_dataset():

    # Load the MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='./data',  
                                            train=True,  
                                            transform=torchvision.transforms.ToTensor(),  
                                            download=True) 
    test_dataset = torchvision.datasets.MNIST(root='./data',  
                                            train=False,  
                                            transform=torchvision.transforms.ToTensor(),  
                                            download=True)
    return train_dataset, test_dataset


class Classifier(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.dropout1 = nn.Dropout2d(0.25) 
        self.dropout2 = nn.Dropout2d(0.5) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10) 
  
    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.dropout1(x) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.dropout2(x) 
        x = x.view(-1, 64 * 7 * 7) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x
    
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):

    batch_size=100
    num_epochs=10
    # Split the training set into training and validation sets 
    val_percent = 0.2 # percentage of the data used for validation 
    val_size    = int(val_percent * len(train_dataset)) 
    train_size  = len(train_dataset) - val_size 
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,  
                                                            [train_size,  
                                                                val_size]) 
    
    # Create DataLoaders for the training and validation sets 
    train_loader = torch.utils.data.DataLoader(train_dataset,  
                                            batch_size=batch_size,  
                                            shuffle=True, 
                                            pin_memory=True) 
    val_loader = torch.utils.data.DataLoader(val_dataset,  
                                            batch_size=batch_size,  
                                            shuffle=False, 
                                            pin_memory=True) 
    losses = [] 
    accuracies = [] 
    val_losses = [] 
    val_accuracies = [] 
    # Train the model 
    for epoch in range(num_epochs): 
        for i, (images, labels) in enumerate(train_loader): 
            # Forward pass 
            images=images.to(device) 
            labels=labels.to(device) 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
            
            # Backward pass and optimization 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
    
            _, predicted = torch.max(outputs.data, 1) 
        acc = (predicted == labels).sum().item() / labels.size(0) 
        accuracies.append(acc) 
        losses.append(loss.item())   
            
        # Evaluate the model on the validation set 
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad(): 
            for images, labels in val_loader: 
                labels=labels.to(device) 
                images=images.to(device) 
                outputs = model(images) 
                loss = criterion(outputs, labels) 
                val_loss += loss.item() 
                
                _, predicted = torch.max(outputs.data, 1) 
            total = labels.size(0) 
            correct = (predicted == labels).sum().item() 
            val_acc += correct / total 
            val_accuracies.append(acc) 
            val_losses.append(loss.item())   
        
                
        print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
            epoch+1, num_epochs, loss.item(), val_loss, acc ,val_acc))


def plot_training_validiation_loss(losses, val_losses, accuracies, val_accuracies, num_epochs):
    
    # Plot the training and validation loss over time 
    plt.plot(range(num_epochs),  
            losses, color='red',  
            label='Training Loss', 
            marker='o') 
    plt.plot(range(num_epochs),  
            val_losses, 
            color='blue',  
            linestyle='--',  
            label='Validation Loss',  
            marker='x') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss') 
    plt.legend() 
    plt.show() 
    
    # Plot the training and validation accuracy over time 
    plt.plot(range(num_epochs),  
            accuracies,  
            label='Training Accuracy',  
            color='red',  
            marker='o') 
    plt.plot(range(num_epochs),  
            val_accuracies,  
            label='Validation Accuracy',  
            color='blue',  
            linestyle=':',  
            marker='x') 
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.title('Training and Validation Accuracy') 
    plt.legend() 
    plt.show()

def evaluation(model, test_loader, device, test_dataset, batch_size=100):

# Create a DataLoader for the test dataset 
    test_loader = torch.utils.data.DataLoader(test_dataset,  
                                            batch_size=batch_size,  
                                            shuffle=False) 
    
    # Evaluate the model on the test dataset 
    model.eval() 
    
    with torch.no_grad(): 
        correct = 0
        total = 0
        y_true = [] 
        y_pred = [] 
        for images, labels in test_loader: 
            images = images.to(device) 
            labels = labels.to(device) 
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
            predicted=predicted.to('cpu') 
            labels=labels.to('cpu') 
            y_true.extend(labels) 
            y_pred.extend(predicted) 
    
    print('Test Accuracy: {}%'.format(100 * correct / total)) 
    
    # Generate a classification report 
  

    print(classification_report(y_true, y_pred))



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    model = Classifier().to(device)
    print(model)
    summary(model, (1, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



if __name__ == "__main__":
    main()