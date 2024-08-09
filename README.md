## GrokAdamW: A PyTorch Optimizer for Accelerated Grokking

by Eric Hartford

**GrokAdamW** is a novel optimizer designed to enhance AI training by combining the strengths of Grokfast (a technique for accelerating "grokking" in deep learning models) with the robustness and efficiency of the AdamW optimizer. It's particularly useful for models exhibiting delayed generalization, where performance on validation data improves significantly after a period of overfitting to the training data.

**Update:** This optimizer was used to train the awesome tiny model [nisten/Biggie-SmoLlm-0.15B-Base](https://huggingface.co/nisten/Biggie-SmoLlm-0.15B-Base)

This implementation was inspired by the following papers:

- **Grokfast: Accelerated Grokking by Amplifying Slow Gradients**  
  Lee, J., Kang, B. G., Kim, K., & Lee, K. M. (2024).  
  *arXiv:2405.20233 [cs.LG]*.  
  [https://doi.org/10.48550/arXiv.2405.20233](https://doi.org/10.48550/arXiv.2405.20233)

- **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**  
  Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).  
  *arXiv:2201.02177 [cs.LG]*.  
  [https://doi.org/10.48550/arXiv.2201.02177](https://doi.org/10.48550/arXiv.2201.02177)

- **Decoupled Weight Decay Regularization**  
  Loshchilov, I., & Hutter, F. (2019).  
  *arXiv:1711.05101 [cs.LG]*.  
  [https://doi.org/10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)


## Table of Contents
1. [Overview](#overview)
2. [Theory](#theory)
3. [Mathematical Explanation](#mathematical-explanation)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Common Pitfalls and Debugging Tips](#common-pitfalls-and-debugging-tips)
8. [Contribution](#contribution)
9. [License](#license)

## Overview

**Grokking** is a phenomenon where deep learning models achieve sudden generalization after a long period of overfitting. Research suggests that this delayed generalization is related to the slow-varying components of gradients during training. **Grokfast**, inspired by this research, accelerates grokking by amplifying these slow-varying gradients.

**GrokAdamW** builds upon this concept by integrating Grokfast's adaptive frequency amplification into the AdamW optimization algorithm. It introduces several key innovations:

1. **Adaptive Momentum:** The momentum of the Grokfast component (which controls the emphasis on slow-varying gradients) dynamically adjusts based on a "Grokking Signal" that reflects the model's generalization progress.
2. **Layer-Wise Momentum Decay:** Recognizing that different layers learn at different rates, GrokAdamW implements a gradual decay of the AdamW momentum parameter (β1) from earlier to later layers, promoting faster generalization in early layers while preventing overfitting in later layers.
3. **Multiple Grokking Signals:** Allows for flexibility in defining the Grokking Signal by supporting multiple signal functions, which can be combined to capture different aspects of generalization performance.
4. **Optional Gradient Clipping:** Provides the option to clip gradients, enhancing training stability and preventing exploding gradients, a common issue in deep learning.

## Theory:

### Mathematical Explanation:

**Core AdamW Updates:**
For each layer *l*, parameter *p*, and training step *t*:

* First Moment Estimate:  
   * m_t[l, p] = β1_l * m_(t-1)[l, p] + (1 - β1_l) * ĝ_t[l, p] 
   * Where β1_l = β1_init * (1 - γ)^l (layer-wise momentum decay)
* Second Moment Estimate: 
   * v_t[l, p] = β2 * v_(t-1)[l, p] + (1 - β2) * ĝ_t[l, p]²
* Bias Correction: 
   * m̂_t[l, p] = m_t[l, p] / (1 - β1^t)
   * v̂_t[l, p] = v_t[l, p] / (1 - β2^t)
* Parameter Update: 
   * θ_t[l, p] = θ_(t-1)[l, p] - η * (m̂_t[l, p] / (sqrt(v̂_t[l, p]) + ε) + wd * θ_(t-1)[l, p])

**Grokfast Integration:**

* Grokking Signal:
    * GS_t =  Combine(signal_1(t), signal_2(t), ..., signal_n(t))  (using the provided `grokking_signal_fns`)
* EMA Filter Momentum:
    * α_t = α_init * exp(-κ * GS_t) 
* EMA Filter Update:
    * μ_t[l, p] = α_t * μ_(t-1)[l, p] + (1 - α_t) * g_t[l, p]
* Grokfast-Amplified Gradient:
    * ĝ_t[l, p] = g_t[l, p] + λ * μ_t[l, p]

**Optional Gradient Clipping:**

* If `gradient_clipping` > 0:
   * `torch.nn.utils.clip_grad_norm_(parameters, gradient_clipping)` 

## Installation

You can easily install GrokAdamW using pip:

```bash
pip install grokadamw
```

## Usage:

```python
import torch
import torch.nn as nn
from grokadamw import GrokAdamW

# Define your model
model = nn.Linear(10, 1)

# Define your grokking signal function(s)
def grokking_signal_fn(training_loss: float, validation_loss: float) -> float:
    if training_loss == 0:
        return 0.0  # Avoid division by zero
    return (validation_loss - training_loss) / training_loss

# Initialize GrokAdamW optimizer
optimizer = GrokAdamW(model.parameters(), lr=1e-3, grokking_signal_fn=grokking_signal_fn)

# Training loop
for epoch in range(num_epochs):
    # ... [Your training code] ...

    # Calculate validation loss (val_loss)

    # Perform optimization step
    loss = optimizer.step(closure=lambda: your_loss_function(model, data)) 
```

## Configuration:

GrokAdamW supports standard AdamW parameters (`lr`, `betas`, `eps`, `weight_decay`) and additional parameters for Grokfast:

* `alpha_init`: Initial momentum for the EMA filter (default: 0.98)
* `lamb`: Amplification factor for the filtered gradients (default: 2.0)
* `gamma`: Layer-wise momentum decay rate (default: 0.1)
* `grokking_signal_fns`: A list of functions that each return a scalar grokking signal (optional)
* `grokking_signal_decay_rate`: Decay rate for adjusting alpha based on the grokking signal (default: 0.1)
* `gradient_clipping`: Maximum norm for gradient clipping (default: 1.0, set to 0 to disable)

## Common Pitfalls and Debugging Tips

1. **Grokking Signal Functions Not Providing Useful Signals:** 
   - Ensure that the functions return meaningful values, reflecting aspects like validation vs. training loss differences.
   - Consider normalizing the output of signal functions.

2. **Issues with Gradient Clipping:**
   - If gradients are frequently being clipped, it may indicate a need to adjust the learning rate or other hyperparameters.

3. **Unexpected Behavior with Layer-wise Momentum Decay:**
   - Monitor the learning dynamics for different layers. If some layers are learning too slowly or too quickly, adjust `gamma` or individual layer hyperparameters accordingly.

4. **Monitoring Grokking Signal and Alpha Values:**
   - Use tools like TensorBoard or custom logging to track the grokking signal, alpha values, and gradient norms. This can help in understanding the optimizer's behavior and making necessary adjustments.

## Contribution

GrokAdamW is an ongoing research project. Your feedback and contributions are welcome! Please feel free to submit issues, feature requests, or pull requests. For more details, see our [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

GrokAdamW is licensed under the Apache 2.0 License. See the LICENSE file for more details.
