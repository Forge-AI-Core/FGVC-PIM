import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.
    Calculates the triplet loss for the hardest positive and hardest negative in a batch.
    """
    def __init__(self, margin: float = 0.3):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D]
        # labels: [B]
        
        orig_dtype = embeddings.dtype
        # Cast to float32 for numerical stability in distance calculations and sqrt
        embeddings = embeddings.float()
        
        # L2 normalize the embeddings to stabilize training and bound pairwise distances to [0, 2]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 1. Compute pairwise distance matrix (B x B)
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        
        # Prevent self-distance (diagonal elements) from being 0.0, which causes NaN gradients in sqrt.
        # Since self-distances are masked out in mask_ap and mask_an anyway, changing them to a non-zero value
        # does not affect the final loss but prevents NaN gradients.
        eye_mask = torch.eye(distances.size(0), device=distances.device)
        distances = distances + eye_mask * 1.0
        
        distances = torch.sqrt(distances + 1e-8) # numerical stability

        # 2. Get mask for positive and negative pairs
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # Mask for anchor-positive: same label, different indices
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
        mask_ap = labels_equal & ~indices_equal
        
        # Mask for anchor-negative: different labels
        mask_an = ~labels_equal
 
        # 3. For each anchor, find the hardest positive (maximum distance)
        ap_distances = distances * mask_ap.float()
        hardest_positive_dist, _ = torch.max(ap_distances, dim=1)
 
        # 4. For each anchor, find the hardest negative (minimum distance)
        max_dist = torch.max(distances)
        an_distances = distances + max_dist * (~mask_an).float()
        hardest_negative_dist, _ = torch.min(an_distances, dim=1)
 
        # 5. Compute triplet loss for hard triplets
        losses = hardest_positive_dist - hardest_negative_dist + self.margin
        losses = torch.clamp(losses, min=0.0)
        
        # Only average over triplets where we have valid positive and negative samples
        valid_triplets = (mask_ap.sum(dim=1) > 0) & (mask_an.sum(dim=1) > 0)
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, dtype=orig_dtype, requires_grad=True)
            
        return losses[valid_triplets].mean().to(orig_dtype)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Proposed in: Supervised Contrastive Learning (Khosla et al., 2020).
    Adapted for single-view feature embeddings.
    """
    def __init__(self, temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D]
        # labels: [B]
        orig_dtype = embeddings.dtype
        embeddings = embeddings.float()
        
        # 1. L2 normalize the embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 2. Compute similarity matrix (B x B)
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # For numerical stability, subtract the max logit from each row
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Mask out diagonal (self-contrast)
        batch_size = embeddings.size(0)
        logits_mask = torch.ones_like(logits) - torch.eye(batch_size, device=embeddings.device)
        
        # Positive mask: same labels, different indices
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask_pos = labels_equal & logits_mask.bool()
        
        # We need at least one positive pair for an anchor to compute contrastive loss
        valid_anchors = mask_pos.sum(dim=1) > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, dtype=orig_dtype, requires_grad=True)
            
        # Denominator: sum over all different elements (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True) + 1e-8
        
        # Log probability of all pairs
        log_prob = logits - torch.log(sum_exp_logits)
        
        # Compute mean log likelihood for positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        
        # Average loss over all valid anchors
        loss = -mean_log_prob_pos[valid_anchors].mean()
        
        return loss.to(orig_dtype)

