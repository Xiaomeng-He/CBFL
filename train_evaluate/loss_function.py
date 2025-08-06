import torch
import torch.nn as nn

def loss_function(act_predictions, 
                  act_tgt,
                  loss_mode,
                  beta=0.999,
                  gamma=2,
                  class_freq=None):

    if loss_mode == 'base':
        act_criterion = nn.CrossEntropyLoss(ignore_index=0)
    elif loss_mode == 'CBFL':
        act_criterion = CBFLoss(beta=beta, gamma=gamma, class_freq=class_freq)
    elif loss_mode == 'weighted':
        act_criterion = WeightedFocalLoss(gamma=gamma, class_freq=class_freq)
    
    act_predictions = act_predictions.view(-1, act_predictions.size(-1)) # shape: (batch_size * seq_length, num_act)
    act_tgt = act_tgt.view(-1) # shape: (batch_size * seq_length)

    loss = act_criterion(act_predictions, act_tgt)

    return loss

class CBFLoss(nn.Module):
    """
    Implement Class-Balanced Focal Loss.

    """
    def __init__(self, beta, gamma, class_freq=None):
        super().__init__()
        self.beta = beta 
        self.gamma = gamma 
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        if class_freq is not None:
            self.register_buffer('alpha', self.get_alpha(class_freq))
        else:
            self.register_buffer('alpha', None)

    def get_alpha(self, class_freq):
        

        effective_num = 1.0 - torch.pow(self.beta, class_freq.float())
        effective_num = torch.where(class_freq == 0, torch.ones_like(effective_num), effective_num)  # avoid zero division
        alpha = (1.0 - self.beta) / effective_num
        alpha = torch.where(class_freq == 0, torch.zeros_like(alpha), alpha) # alpha is 0 for class with frequency of 0
        alpha = alpha / alpha.sum() * (class_freq > 0).sum().float() # shape: (num_classes,)

        return alpha

    def forward(self, inputs, targets):

        ce_loss = self.ce(inputs, targets) 

        if ce_loss.dim() > 1:
            ce_loss = ce_loss.flatten() # shape: (batch_size * seq_len)

        if targets.dim() > 1:
            targets = targets.flatten()  # shape: (batch_size * seq_len)
        
        p_t = torch.exp(-ce_loss)  # shape: (batch_size * seq_len)
        
        focal_term = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        mask = targets != 0
        loss = loss[mask]

        return loss.mean()

class WeightedFocalLoss(nn.Module):
    """
    Implements Focal Loss combined with inverse frequency weighting.

    """
    def __init__(self, gamma, class_freq):
        super().__init__()
        self.gamma = gamma 
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.register_buffer('alpha', self.get_alpha(class_freq))

    def get_alpha(self, class_freq):

        original_zero_mask = class_freq == 0

        total_freq = class_freq.sum()
        valid_class = (class_freq > 0).sum().float()

        class_freq = torch.where(original_zero_mask, torch.ones_like(class_freq), class_freq)  # avoid zero division
        inv_class_freq = total_freq / class_freq
        
        inv_class_freq = inv_class_freq / inv_class_freq.sum() * valid_class
        inv_class_freq = torch.where(original_zero_mask, torch.zeros_like(inv_class_freq), inv_class_freq)

        return inv_class_freq

    def forward(self, inputs, targets):

        ce_loss = self.ce(inputs, targets) 

        if ce_loss.dim() > 1:
            ce_loss = ce_loss.flatten() # shape: (batch_size * seq_len)

        if targets.dim() > 1:
            targets = targets.flatten()  # shape: (batch_size * seq_len)
        
        p_t = torch.exp(-ce_loss)  # shape: (batch_size * seq_len)
        
        focal_term = (1 - p_t) ** self.gamma
        
        alpha = self.alpha.to(targets.device)
        alpha_t = alpha.gather(0, targets)
        loss = alpha_t * focal_term * ce_loss

        mask = targets != 0
        loss = loss[mask]

        return loss.mean()

