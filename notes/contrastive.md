# Contrastive Learning Notes

**Contrastive Loss**
- One of the earliest training objectives for deep metric learning
- Given list of input samples ${x_i}$ with labels $y_i$ -learn function that encodes $x_i$ into embed vector so similar embedding clustered together while others far apart
- Contrastive loss - minimizes embedding distance of similar labels and maximizes embedding distance of different labels

$$L_{cont}(x_i, x_j, \theta) = \mathbb{1}[y_i = y_j] ||f_{\theta}(x_i) - f_{\theta}(x_j)|| + \mathbb{1}[y_i \neq y_j]\max(0, \epsilon - ||f_{\theta}(x_i) - f_{\theta}(x_j)||)$$

- $\epsilon$ is hyperparameter to determine distance between contrasting pairs


## Triplet Loss
- Samples an anchor point, a point from the same class as the anchor, and a point from a different class from the anchor
- $L_{triplet}$ maximizes distance of anchor point from different class and minimizes distance to same class point

## Lifted Structured Loss
- Works on every positive pair in the batch
  - takes the maximum of each positive node's norm from all negative points
- average negative distance per positive pair

## N Pair Loss
- triplet loss generalization that uses multiple samples (similar to lifted structured)

## Noise Contrastive Embedding
- 