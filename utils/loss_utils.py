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
        
        # L2 normalize the embeddings to stabilize training and bound pairwise distances to [0, 2]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 1. Compute pairwise distance matrix (B x B)
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-16) # numerical stability

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
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return losses[valid_triplets].mean()
