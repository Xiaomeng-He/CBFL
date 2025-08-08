import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score 
from torcheval.metrics import MulticlassAUPRC
from loss_function import loss_function

def calculate_metrics(act_labels, act_tgt, device, num_valid_class):

    Accuracy = MulticlassAccuracy(num_classes=num_valid_class, average='micro', ignore_index=0).to(device)
    accuracy = Accuracy(act_labels, act_tgt)

    Precision = MulticlassPrecision(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    precision_macro = Precision(act_labels, act_tgt)

    Recall = MulticlassRecall(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    recall_macro = Recall(act_labels, act_tgt)

    F1 = MulticlassF1Score(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    f1_macro = F1(act_labels, act_tgt)

    return accuracy, precision_macro, recall_macro, f1_macro

def train(model, 
          dataloader,
          optimizer,
          device,
          loss_mode,
          beta,
          gamma,
          class_freq):

    model.train() 

    train_epoch_loss = 0.0

    for batch in dataloader:

        # 1. load data
        batch = [tensor.to(device) for tensor in batch]
        train_src_act, train_src_time, train_tgt = batch
        # train_src_act shape: (batch_size, seq_len)
        # train_src_time shape: (batch_size, seq_len, 2)
        # train_tgt shape: (batch_size, seq_len)

        # 2. set the gradient to zero
        optimizer.zero_grad()

        # 3. run a forward pass and obtain predictions
        act_predictions = model(train_src_act, train_src_time) # shape: (batch_size, seq_len, num_act)

        # 4. calculate loss
        loss = loss_function(act_predictions, train_tgt, loss_mode, beta, gamma, class_freq)

        # 5. backpropagation
        loss.backward()
        optimizer.step()

        # 6. sum up losses from all batches
        train_epoch_loss += loss.item()

    # 7. divide total loss by number of batches to obain average loss
    avg_train_loss = train_epoch_loss / len(dataloader)

    return avg_train_loss

def validate(model, 
          dataloader,
          device,
          num_valid_class,
          loss_mode,
          beta=0.999,
          gamma=2,
          class_freq=None):

    model.eval()

    val_epoch_loss = 0.0
    all_act_labels = []
    all_act_tgt = []

    with torch.no_grad():

        for batch in dataloader:

            # 1. load data
            batch = [tensor.to(device) for tensor in batch]
            src_act, src_time, tgt = batch
            # src_act shape: (batch_size, seq_len)
            # src_time shape: (batch_size, seq_len, 2)
            # tgt shape: (batch_size, seq_len)

            # 2. run a forward pass and obtain predictions
            act_predictions = model(src_act, src_time) # shape: (batch_size, seq_len, num_act)
            act_labels = act_predictions.argmax(2) # shape: (batch_size, seq_len)

            # 3. calculate loss
            loss = loss_function(act_predictions, tgt, loss_mode, beta, gamma, class_freq)

            # 4. sum up losses from all batches
            val_epoch_loss += loss.item()

            # 5. accumulate predictions and targets
            all_act_labels.append(act_labels.view(-1).detach())
            all_act_tgt.append(tgt.view(-1).detach())

    # 6. divide total loss by number of batches to obain average loss
    avg_val_loss = val_epoch_loss / len(dataloader)

    # 7. calculate metrics
    all_preds = torch.cat(all_act_labels) # 1D tensor
    all_targets = torch.cat(all_act_tgt) # 1D tensor
    accuracy, precision_macro, recall_macro, f1_macro = calculate_metrics(all_preds, all_targets, device, num_valid_class)

    return avg_val_loss, accuracy, precision_macro, recall_macro, f1_macro

def evaluate(model, 
          dataloader,
          device,
          num_valid_class):

    model.eval()

    all_act_labels = []
    all_act_tgt = []
    all_pred_probs = []

    with torch.no_grad():

        for batch in dataloader:

            # 1. load data
            batch = [tensor.to(device) for tensor in batch]
            src_act, src_time, tgt = batch
            # src_act shape: (batch_size, seq_len)
            # src_time shape: (batch_size, seq_len, 2)
            # tgt shape: (batch_size,)

            # 2. run a forward pass and obtain predictions
            act_predictions = model(src_act, src_time) # shape: (batch_size, seq_len, num_act)

            act_labels = act_predictions.argmax(2) # shape: (batch_size, seq_len)

            # 3. get the predictions
            prefix_lens = (src_act != 0).sum(dim=1)
            assert (prefix_lens > 0).all(), "Found at least one row in 'src_act' with all zeros."
            last_indices = prefix_lens - 1 # shape: (batch_size,)

            last_pred_label = act_labels.gather(1, last_indices.unsqueeze(1)).squeeze(1) # shape: (batch_size, )

            last_indices = last_indices.view(-1, 1, 1).expand(-1, 1, act_predictions.shape[2]) # shape: (batch_size, 1, num_act)
            pred_probs = act_predictions.gather(1, last_indices).squeeze(1) # (batch_size, num_act)

            # 3. accumulate predictions and targets
            all_act_labels.append(last_pred_label.detach())
            all_pred_probs.append(pred_probs.detach().cpu())
            all_act_tgt.append(tgt)

    # 4. calculate metrics
    all_preds = torch.cat(all_act_labels)
    all_targets = torch.cat(all_act_tgt)
    all_pred_probs = torch.cat(all_pred_probs)  # shape: (num_samples, num_act)

    accuracy, precision_macro, recall_macro, f1_macro = calculate_metrics(all_preds, all_targets, device, num_valid_class)

    metric = MulticlassAUPRC(num_classes=num_valid_class)
    metric.update(all_pred_probs, all_targets)
    auprc_tensor = metric.compute()
    auprc = auprc_tensor.item()

    return  accuracy, precision_macro, recall_macro, f1_macro, auprc,\
        all_preds, all_targets, all_pred_probs