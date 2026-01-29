import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ctc_loss(predictions, targets, blank_label=0):


    batch_size, num_classes, prediction_length = predictions.shape
    _, target_length = targets.shape


    log_probs = F.log_softmax(predictions, dim=1)  


    log_probs = log_probs.permute(2, 0, 1) 


    input_lengths = torch.full(size=(batch_size,), fill_value=prediction_length, dtype=torch.long)


    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long) # 

    ctc_loss = torch.nn.CTCLoss(blank=blank_label, reduction='mean') # 


    concatenated_targets = torch.cat(list(targets)) #

    loss = ctc_loss(log_probs, concatenated_targets, input_lengths, target_lengths)

    return loss

def sort_loss(predictions, targets):

    loss = compute_ctc_loss(predictions, targets, blank_label=predictions.shape[1]-1)
    return loss


def image_classification_loss(predictions, certainties, targets, use_most_certain=True):

    targets_expanded = torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)

    losses = nn.CrossEntropyLoss(reduction='none')(predictions, targets_expanded)

    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:  # 
        loss_index_2[:] = -1
    

    
    B, _, internal_ticks = predictions.size()
    num_segments =4
    segment_length = internal_ticks // num_segments
    total_loss = 0
    
    for i in range(num_segments): #
        start_index = i * segment_length
        end_index = (i + 1) * segment_length
        segment_losses = losses[:, start_index:end_index] # 
        segment_certainties = certainties[:, 1, start_index:end_index] # 
        

        segment_min_loss_indices = segment_losses.argmin(dim=1) # [B]
        batch_indexer = torch.arange(B, device=predictions.device)
        segment_loss_minimum_ce = segment_losses[batch_indexer, segment_min_loss_indices].mean()


        if use_most_certain:
            segment_most_certain_indices = segment_certainties.argmax(dim=1) # [B]
            segment_loss_selected = segment_losses[batch_indexer, segment_most_certain_indices].mean()
        else:
            segment_loss_selected = segment_losses[:, -1].mean() 
            
        total_loss += (segment_loss_minimum_ce + segment_loss_selected) / 2
        
    final_loss = total_loss / num_segments
        
    return final_loss, loss_index_2
##############################################################################################################################



def maze_loss(predictions, certainties, targets, cirriculum_lookahead=5, use_most_certain=True):


    predictions_reshaped = predictions.flatten(0,1)

    targets_reshaped = torch.repeat_interleave(targets.unsqueeze(-1), 
                                               predictions.size(-1), -1).flatten(0,1).long()
    

    losses = nn.CrossEntropyLoss(reduction='none')(predictions_reshaped, targets_reshaped)
    losses = losses.reshape(predictions[:,:,0].shape)

    iscorrects = (predictions.argmax(2) == targets.unsqueeze(-1)).cumsum(1)
    correct_mask = (iscorrects == torch.arange(1, iscorrects.size(1)+1, device=iscorrects.device).reshape(1, -1, 1))
    correct_mask[:,0,:] = 1
    upto_where = correct_mask.cumsum(1).argmax(1).max(-1)[0]+cirriculum_lookahead
    loss_mask = torch.zeros_like(losses)
    for bi in range(predictions.size(0)):
        loss_mask[bi, :upto_where[bi]] = 1


    losses = (losses * loss_mask).sum(1)/(loss_mask.sum(1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1]
    loss_selected = losses[batch_indexer, loss_index_2]

    loss = ((loss_minimum_ce + loss_selected)/2).mean()
    return loss, loss_index_2, upto_where.detach().cpu().numpy()

def parity_loss(predictions, certainties, targets, use_most_certain=True):

    losses = nn.CrossEntropyLoss(reduction='none')(predictions.flatten(0,1), 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), 
                                                                           predictions.size(-1), -1).flatten(0,1).long()).reshape(predictions[:,:,0].shape)


    losses = losses.mean(1)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2


def qamnist_loss(predictions, certainties, targets, use_most_certain=True):

    losses = nn.CrossEntropyLoss(reduction='none')(predictions, 
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))
        
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    
    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2