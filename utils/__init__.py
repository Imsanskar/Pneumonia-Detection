import torch
from .evaulation import *

def checkpoint(model, file_name):
    torch.save(model.state_dict(), file_name)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
    return model


def train(model, optim, loss_fn, train_dataloader, val_dataloader, epochs = 10, early_stop_threshold = None, model_filename = "model.pth"):
    Losses = []
    Val_f1 = []
    Val_accuracy = []

    if early_stop_threshold == None:
        early_stop_threshold = 1000000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(epochs):
        train_loss = []

        model.train()
        running_loss = 0
        for idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            
            optim.step()
            train_loss.append(loss.item())
            running_loss += torch.linalg.norm(loss).item()
            if idx % 20 == 0:
                print(f"Epoch: {i}, idx: {idx}, loss: {running_loss / (idx + 1)}")
        model.eval()
        f1_score = None
        accuracy = None
        with torch.no_grad():
            f1_score = evaluate_validation_set(model, val_dataloader)
            accuracy = calculate_accuracy(model, val_dataloader)
        
        Losses.append(np.sum(train_loss)/len(train_loss))
        Val_f1.append(f1_score)
        Val_accuracy.append(accuracy)
        if early_stop_threshold != None:
            if f1_score > best_accuracy:
                best_accuracy = f1_score
                best_epoch = i
                checkpoint(model, "best_model.pt")
                print(f"Epoch: {i}, Validation F1 score: {f1_score}")
            elif (i - best_epoch) > early_stop_threshold: 
                print(f"Epoch: {i}, Early stopped training at epoch {i}")
                break
            else:
                print(f"Epoch: {i}, Loss does not improve from: {best_accuracy}")  
        
        checkpoint(model, model_filename)
        print(f"Epoch {i}: Training Loss: {np.sum(train_loss)/len(train_loss)}, Validation F1: {f1_score}, Validation Accuracy: {accuracy}")

    
    return Losses, Val_f1, Val_accuracy


def infer(model, data):
    y_pred = []
    for i in range(len(data)):
        img = data[i].view(1, 3, 224, 224)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        y_pred.append(pred_label)

    return torch.tensor(y_pred)