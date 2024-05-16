# Contrastive Learning Notes

**Contrastive Loss**
- One of the earliest training objectives for deep metric learning
- Given list of input samples ${x_i}$ with labels $y_i$ -learn function that encodes $x_i$ into embed vector so similar embedding clustered together while others far apart
- Contrastive loss - minimizes embedding distance of similar labels and maximizes embedding distance of different labels

$$L_{cont}(x_i, x_j, \theta) = \mathbb{1}[y_i = y_j] ||f_{\theta}(x_i) - f_{\theta}(x_j)|| + \mathbb{1}[y_i \neq y_j]||||$$