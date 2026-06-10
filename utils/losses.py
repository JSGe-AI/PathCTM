import torch
import torch.nn as nn


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
