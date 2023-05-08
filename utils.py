import torch
import cv2
import numpy as np

def checkpoint(model, file_name):
    torch.save(model.state_dict(), file_name)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))
    return model


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1, precision, recall

def evaluate_validation_set(model, dataloader):
    model.eval()
    y_pred = []
    y_true = []
    for images,labels in dataloader:
        for i in range(len(labels)):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            img = images[i].view(1, 3, 224, 224)
            with torch.no_grad():
                logps = model(img)


            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            y_pred.append(pred_label)
            y_true.append(labels.cpu()[i])
    f1_score, _, _ = f1_loss(torch.tensor(y_true), torch.tensor(y_pred))
    return f1_score

def calculate_accuracy(model, dataloader):
    model.eval()
    correct_count, all_count = 0, 0
    for images,labels in dataloader:
        for i in range(len(labels)):
            img = images[i].view(1, 3, 224, 224)
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu()[i]
            all_count += 1
            if(true_label == pred_label):
                correct_count += 1

        
    return correct_count / all_count

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

def return_CAM(feature_conv, weight, class_idx, reshape_size):
    size_upsample = (224, 224)
    _, nc, h, w = feature_conv.shape
    output_cams = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w)).detach().cpu().numpy()
        cam = np.matmul(weight[idx].view(-1, 48).detach().cpu().numpy(), beforeDot)
        cam = cam.reshape(reshape_size, reshape_size)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(cam_img * 255)
        output_cams.append(cv2.resize(cam_img, size_upsample))
    return output_cams

def train(model, optim, loss_fn, train_dataloader, val_dataloader, epochs = 10, early_stop_threshold = 5, model_file_name = "model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_accuracy = -1
    best_epoch = -1 
    Losses = []
    Val_f1 = []
    Val_accuracy = []
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
        if f1_score > best_accuracy:
            best_accuracy = f1_score
            best_epoch = i
            checkpoint(model, model_file_name)
            print(f"Epoch: {i}, Validation F1 score: {f1_score}")
        elif (i - best_epoch) > early_stop_threshold: 
            print(f"Epoch: {i}, Early stopped training at epoch {i}")
            break
        else:
            print(f"Epoch: {i}, Loss does not improve from: {best_accuracy}")  
        
        print(f"Epoch {i}: Training Loss: {np.sum(train_loss)/len(train_loss)}, Validation F1: {f1_score}, Validation Accuracy: {accuracy}")


    return Losses, Val_f1, Val_accuracy

def return_CAM_VGG(feature_conv, weight, class_idx):
    size_upsample = (224, 224)
    _, nc, h, w = feature_conv.shape
    output_cams = []
    for idx in class_idx:
        beforeDot =  feature_conv.reshape((nc, h*w)).detach().cpu().numpy()
        cam = np.matmul(weight[idx].view(49, -1).detach().cpu().numpy(), beforeDot)
        cam = cam.reshape(7, 7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(cam_img * 255)
        output_cams.append(cv2.resize(cam_img, size_upsample))
    return output_cams