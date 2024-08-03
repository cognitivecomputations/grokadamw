## GrokAdamW: A PyTorch Optimizer for Accelerated Grokking

**GrokAdamW** is a novel optimizer designed to enhance AI training by combining the strengths of Grokfast (a technique for accelerating "grokking" in deep learning models) with the robustness and efficiency of the AdamW optimizer. It's particularly useful for models exhibiting delayed generalization, where performance on validation data improves significantly after a period of overfitting to the training data.

### Theory:

**Grokking** is a phenomenon where deep learning models achieve sudden generalization after a long period of overfitting. Research suggests that this delayed generalization is related to the slow-varying components of gradients during training.  **Grokfast**, inspired by this research, accelerates grokking by amplifying these slow-varying gradients.

**GrokAdamW** builds upon this concept by integrating Grokfast's adaptive frequency amplification into the AdamW optimization algorithm. It introduces several key innovations:

1. **Adaptive Momentum:** The momentum of the Grokfast component (which controls the emphasis on slow-varying gradients) dynamically adjusts based on a "Grokking Signal" that reflects the model's generalization progress.
2. **Layer-Wise Momentum Decay:** Recognizing that different layers learn at different rates, GrokAdamW implements a gradual decay of the AdamW momentum parameter (β1) from earlier to later layers, promoting faster generalization in early layers while preventing overfitting in later layers.
3. **Multiple Grokking Signals:**  Allows for flexibility in defining the Grokking Signal by supporting multiple signal functions, which can be combined to capture different aspects of generalization performance. 
4. **Optional Gradient Clipping:** Provides the option to clip gradients, enhancing training stability and preventing exploding gradients, a common issue in deep learning. 

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

### Usage:

**1. Installation:**

```bash
pip install grokadamw  # Assuming you've packaged it 
```

**2. Example:**

```python
import torch
import torch.nn as nn
from grokadamw import GrokAdamW

# Define your model
model = nn.Linear(10, 1)

# Define your grokking signal function(s)
def grokking_signal_fn():
    return (val_loss - train_loss) / train_loss

# Initialize GrokAdamW optimizer
optimizer = GrokAdamW(model.parameters(), lr=1e-3, grokking_signal_fn=grokking_signal_fn)

# Training loop
for epoch in range(num_epochs):
    # ... [Your training code] ...

    # Calculate validation loss (val_loss)

    # Perform optimization step
    loss = optimizer.step(closure=lambda: your_loss_function(model, data)) 
```

**Configuration:**

GrokAdamW supports standard AdamW parameters (`lr`, `betas`, `eps`, `weight_decay`) and additional parameters for Grokfast:

* `alpha_init`: Initial momentum for the EMA filter (default: 0.98)
* `lamb`: Amplification factor for the filtered gradients (default: 2.0)
* `gamma`: Layer-wise momentum decay rate (default: 0.1)
* `grokking_signal_fns`: A list of functions that each return a scalar grokking signal (optional)
* `grokking_signal_decay_rate`: Decay rate for adjusting alpha based on the grokking signal (default: 0.1)
* `gradient_clipping`: Maximum norm for gradient clipping (default: 1.0, set to 0 to disable)

**Key Points:**

* Define your `grokking_signal_fn(s)` to capture the difference between training and validation performance, or other metrics relevant to generalization.
* Carefully tune the Grokfast-specific hyperparameters (`alpha_init`, `lamb`, `gamma`) to achieve optimal performance for your model and task.
* Monitor the Grokking Signal, alpha values, and gradient clipping during training to understand the optimizer's behavior and make adjustments as needed. 

**Contribution:**

GrokAdamW is an ongoing research project. Your feedback and contributions are welcome!
