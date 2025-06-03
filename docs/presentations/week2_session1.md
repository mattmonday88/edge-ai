---
marp: true
theme: usa-theme
paginate: true
---

<!-- _class: title comfortable -->

# Edge-Optimized Model Architecture
## Week 2 - Session 1
### Deploying Edge AI
#### University of South Alabama

<!-- 
Speaker Notes:
- Welcome to Week 2 of Deploying Edge AI
- Last week we learned how to convert models to ONNX and establish performance baselines
- Today we begin the critical work of optimization
- We'll explore architectural approaches and techniques to make models suitable for edge deployment
- By the end of this session, you'll understand multiple strategies for reducing model complexity while maintaining accuracy
-->

---

<!-- _class: section -->

# From Baseline to Optimized Models

<!-- 
Speaker Notes:
- In Week 1, we established baseline performance metrics for our converted models
- Now we'll learn how to systematically improve those metrics
- This section bridges our previous work with the optimization techniques we'll explore today
- We'll review where we are and map out the optimization journey ahead
-->

---

<!-- _class: dense -->

# Building on Week 1 Foundations

<div class="columns">
<div>

## What We've Accomplished
- ✅ Model conversion to ONNX
- ✅ Performance profiling
- ✅ Baseline metrics established
- ✅ Resource requirements identified
- ✅ Deployment constraints understood

## Our Baseline Metrics
- Inference time (ms)
- Memory usage (MB)
- Model size (MB)
- Accuracy metrics
- Power consumption estimates

</div>
<div>

## Today's Optimization Goals
- 🎯 Reduce model size by 50-90%
- 🎯 Improve inference speed 2-10x
- 🎯 Lower memory requirements
- 🎯 Maintain acceptable accuracy
- 🎯 Enable deployment on target devices

## Optimization Strategies
- Architecture modification
- Knowledge distillation
- Network pruning
- Quantization
- Operation optimization

</div>
</div>

<!-- 
Speaker Notes:
- We've established a solid foundation with model conversion and profiling
- Your baseline metrics from Week 1 are the reference point for measuring optimization success
- Today's ambitious goals are achievable through systematic application of optimization techniques
- The specific targets depend on your model and deployment constraints:
  - Some models can achieve 90% size reduction with minimal accuracy loss
  - Speed improvements vary based on architecture and target hardware
  - Memory reduction is critical for embedded deployment
- We'll explore multiple complementary strategies:
  - Some techniques work better for certain model types
  - Often, combining approaches yields the best results
  - The key is understanding tradeoffs for your specific use case
- Remember: optimization is iterative - measure, optimize, validate, repeat
-->

---

# The Edge Optimization Mindset

<div class="highlight-box">

## Key Principles for Edge AI Optimization

1. **Accuracy vs. Efficiency Tradeoff**
   - Small accuracy losses often enable massive efficiency gains
   - Define acceptable accuracy thresholds upfront

2. **Hardware-Aware Design**
   - Optimize for your specific deployment hardware
   - Consider cache sizes, SIMD instructions, memory bandwidth

3. **Holistic Optimization**
   - Consider the entire inference pipeline, not just the model
   - Pre/post-processing can be significant bottlenecks

4. **Iterative Refinement**
   - Apply techniques incrementally
   - Measure impact at each step
   - Some optimizations interact synergistically

</div>

<!-- 
Speaker Notes:
- Successful edge optimization requires a different mindset than training models
- The accuracy vs. efficiency tradeoff is fundamental:
  - A 1-2% accuracy drop might enable 10x speed improvement
  - The "best" model isn't always the most accurate one
  - Consider your application's actual accuracy requirements
- Hardware-aware design is crucial:
  - Different optimizations work better on different hardware
  - CPU-specific optimizations differ from GPU optimizations
  - Memory access patterns matter as much as computation
- Holistic thinking prevents suboptimal solutions:
  - A highly optimized model with slow preprocessing is still slow
  - Consider data movement, not just computation
  - System-level optimization often yields surprising gains
- Iterative refinement allows controlled optimization:
  - Start with low-hanging fruit (often quantization)
  - Measure carefully after each change
  - Some combinations work better than others
- This mindset will guide all our optimization decisions today
-->

---

<!-- _class: section -->

# Model Architectures for Edge Deployment

<!-- 
Speaker Notes:
- Let's begin by examining model architectures specifically designed for edge deployment
- These architectures incorporate efficiency principles from the ground up
- Understanding these designs helps us modify existing models for edge use
- We'll look at successful patterns across different domains
-->

---

<!-- _class: dense -->

# Efficient Architecture Families

<div class="columns">
<div>

## MobileNet Family
- **MobileNetV1**: Depthwise separable convolutions
- **MobileNetV2**: Inverted residuals + linear bottlenecks
- **MobileNetV3**: Neural architecture search + SE blocks
- **Key Innovation**: Factorized convolutions reduce computation

## EfficientNet Family
- **Compound scaling**: Balanced depth/width/resolution
- **Mobile-friendly variants**: EfficientNet-Lite
- **Optimized building blocks**: MBConv layers
- **Key Innovation**: Systematic scaling approach

</div>
<div>

## ShuffleNet Family
- **Channel shuffle**: Efficient group convolutions
- **Pointwise group convolutions**: Reduced parameters
- **ShuffleNetV2**: Hardware-friendly design rules
- **Key Innovation**: Information flow across groups

## SqueezeNet & Others
- **Fire modules**: Squeeze and expand layers
- **GhostNet**: Cheap linear transformations
- **MicroNet**: Extreme low-resource design
- **Key Innovation**: Parameter efficiency focus

</div>
</div>

<!-- 
Speaker Notes:
- These architecture families represent different approaches to efficiency
- MobileNet pioneered many concepts now standard in edge AI:
  - Depthwise separable convolutions split spatial and channel operations
  - V2 added inverted residuals for better gradient flow
  - V3 used automated search to find optimal configurations
  - Widely adopted due to good accuracy/efficiency balance
- EfficientNet took a scientific approach to model scaling:
  - Instead of arbitrary scaling, uses compound coefficient
  - Scales depth, width, and resolution together
  - EfficientNet-Lite variants specifically for edge deployment
  - Often achieves better accuracy than MobileNet at similar speeds
- ShuffleNet focuses on reducing computational bottlenecks:
  - Channel shuffle enables cross-group information flow
  - V2 principles: equal channel widths, aware of memory access cost
  - Particularly effective on mobile CPUs
- Other notable architectures show diverse approaches:
  - SqueezeNet achieves AlexNet accuracy with 50x fewer parameters
  - GhostNet generates features with cheap operations
  - MicroNet pushes boundaries of how small models can be
- Key lesson: efficient architectures share common patterns we can apply
-->

---

<!-- _class: dense -->

# Architectural Design Patterns

<div class="alert-box">

## Common Efficiency Patterns

### Depthwise Separable Convolutions
```python
# Standard convolution: H×W×C_in → H×W×C_out
# Operations: H×W×K×K×C_in×C_out

# Depthwise separable = Depthwise + Pointwise
# Depthwise: H×W×C → H×W×C (spatial filtering)
# Operations: H×W×K×K×C
# Pointwise: H×W×C → H×W×C_out (channel mixing)
# Operations: H×W×C×C_out

# Reduction ratio: K²×C_out/(K²+C_out) ≈ 8-9x for 3×3 kernels
```

### Inverted Residuals
<div class="text-small">

- Expand → Depthwise → Project (opposite of ResNet)
- Keeps low-dimensional data in memory between blocks
- Critical for maintaining accuracy with fewer parameters

</div>
</div>

<!-- 
Speaker Notes:
- Understanding these patterns helps both in selecting and modifying architectures
- Depthwise separable convolutions are foundational:
  - Split expensive convolution into two cheaper operations
  - Depthwise handles spatial relationships within each channel
  - Pointwise (1×1 conv) handles cross-channel interactions
  - Typically 8-9x reduction in computation for 3×3 kernels
  - Small accuracy loss for massive efficiency gain
- Inverted residuals (used in MobileNetV2+):
  - Traditional ResNet: wide→narrow→wide with residual connection
  - Inverted: narrow→wide→narrow with residual on narrow side
  - Expansion happens in cheap depthwise layer
  - Reduces memory bandwidth requirements significantly
  - Linear bottlenecks (no activation) preserve information
- Other important patterns include:
  - Group convolutions: Process channel groups independently
  - Squeeze-and-excitation: Lightweight attention mechanism
  - Ghost modules: Generate features with linear operations
- These patterns can often be retrofitted to existing architectures
- The key is understanding which patterns work for your use case
-->

---

<!-- _class: dense -->

# Architectural Modifications for Edge

```python
# Example: Converting standard ResNet block to mobile-friendly version
import torch.nn as nn

# Original ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
# Mobile-Optimized Version
class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        # Inverted residual structure
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1)  # Expand
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                              groups=hidden_dim)  # Depthwise
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1)  # Project
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # No activation after final conv (linear bottleneck)
```

<div class="text-tiny" style="margin-top: 10px;">
Note: The mobile version uses depthwise separable convolutions and inverted residual structure for efficiency
</div>

<!-- 
Speaker Notes:
- This example shows how to transform a standard architecture for edge deployment
- The original ResNet block uses standard convolutions:
  - Two 3×3 convolutions with full channel mixing
  - High computational cost: O(C²×K²×H×W)
  - Designed for GPUs with high compute throughput
- The mobile-optimized version uses inverted residual structure:
  - 1×1 expansion increases channels cheaply
  - 3×3 depthwise convolution (groups=channels) for spatial processing
  - 1×1 projection reduces channels back
  - Total computation much lower while maintaining representational power
- Key modifications demonstrated:
  - Replace standard convolutions with depthwise separable
  - Use expansion factor to control capacity
  - Linear bottleneck (no final activation) preserves information
  - Careful attention to where residual connections go
- This pattern can be applied to many existing architectures:
  - Identify computational bottlenecks (usually 3×3 or larger convolutions)
  - Replace with efficient alternatives
  - Adjust capacity through width multipliers or expansion ratios
  - Validate that accuracy remains acceptable
- Practical tip: Start with pretrained efficient architectures when possible
-->

---

# Architecture Selection Strategy

<div class="highlight-box">

## Decision Framework for Edge Models

### 1. Deployment Constraints Assessment
- Target device capabilities (CPU cores, RAM, etc.)
- Latency requirements (real-time? batch processing?)
- Power budget and thermal limits
- Model size constraints (storage, download)

### 2. Task-Specific Considerations
- **Classification**: MobileNet, EfficientNet-Lite
- **Detection**: YOLO variants, MobileNet-SSD, NanoDet
- **Segmentation**: MobileNetV3-LiteSeg, ESPNet
- **NLP**: DistilBERT, TinyBERT, MobileBERT

### 3. Starting Point Selection
- Pretrained models when available
- Architecture complexity matched to task difficulty
- Consider ensemble vs. single model tradeoffs

</div>

<!-- 
Speaker Notes:
- Selecting the right architecture is crucial for successful edge deployment
- Start with a thorough constraints assessment:
  - Be realistic about hardware capabilities
  - Include safety margins for real-world conditions
  - Consider peak vs. sustained performance
  - Remember that thermal throttling can dramatically impact speed
- Task-specific architectures have been optimized for particular problems:
  - Classification models are most mature with many options
  - Detection requires balancing speed and localization accuracy
  - Segmentation is particularly challenging due to high-resolution outputs
  - NLP models have seen dramatic improvements recently
- For starting points:
  - Always prefer pretrained models to save training time
  - Match architecture complexity to your problem difficulty
  - Simple problems don't need complex models
  - Sometimes multiple small models outperform one large model
- Common selection mistakes:
  - Choosing models too large for the hardware
  - Ignoring preprocessing/postprocessing costs
  - Not considering model update requirements
  - Focusing only on accuracy without speed constraints
- Remember: the best edge model balances all requirements, not just accuracy
-->

---

<!-- _class: section -->

# Knowledge Distillation Techniques

<!-- 
Speaker Notes:
- Knowledge distillation is one of the most powerful techniques for creating efficient edge models
- The core idea: transfer knowledge from a large "teacher" model to a smaller "student" model
- This often produces better results than training the small model directly
- We'll explore the theory and practical implementation
-->

---

# Knowledge Distillation Fundamentals

<div class="columns">
<div>

## Core Concept
- **Teacher Model**: Large, accurate model
- **Student Model**: Smaller, efficient model
- **Knowledge Transfer**: Student learns from teacher's outputs
- **Soft Targets**: Probability distributions contain rich information

## Why It Works
- Teacher's outputs encode learned relationships
- Soft probabilities reveal similarity between classes
- Smoother targets are easier to learn
- Regularization effect improves generalization

</div>
<div>

## Distillation Process
```python
# Temperature-scaled softmax
def distillation_loss(student_logits, 
                     teacher_logits, 
                     true_labels,
                     temperature=3.0,
                     alpha=0.7):
    # Soft targets from teacher
    soft_targets = F.softmax(
        teacher_logits / temperature, 
        dim=1
    )
    soft_predictions = F.log_softmax(
        student_logits / temperature, 
        dim=1
    )
    
    # Distillation loss
    distill_loss = F.kl_div(
        soft_predictions,
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Combined with standard loss
    hard_loss = F.cross_entropy(
        student_logits, true_labels
    )
    
    return alpha * distill_loss + \
           (1 - alpha) * hard_loss
```

</div>
</div>

<!-- 
Speaker Notes:
- Knowledge distillation is inspired by how humans transfer knowledge
- The teacher model has learned rich representations we want to preserve
- Key insights that make distillation effective:
  - A teacher predicting [0.7, 0.2, 0.1] is more informative than hard label [1, 0, 0]
  - The 0.2 and 0.1 probabilities indicate learned similarities
  - This "dark knowledge" helps the student learn faster and better
- The temperature parameter is crucial:
  - Higher temperature creates softer probability distributions
  - Reveals more information about inter-class relationships
  - Typical values range from 3 to 10
  - Must use same temperature for both teacher and student
- The loss function combines two objectives:
  - Matching teacher's soft predictions (distillation loss)
  - Predicting correct hard labels (standard loss)
  - Alpha parameter balances these objectives
  - Temperature squared scaling maintains gradient magnitudes
- Practical benefits:
  - Student often exceeds performance of directly trained small model
  - Can achieve 90%+ of teacher accuracy with 10x fewer parameters
  - Works across many domains: vision, NLP, speech
-->

---

# Advanced Distillation Strategies

<div class="alert-box">

## Beyond Basic Distillation

### Feature-Based Distillation
- Match intermediate layer representations
- Useful when architectures differ significantly
- Requires careful layer alignment

### Progressive Distillation
- Gradually reduce model size through multiple steps
- Teacher → Medium → Small → Tiny
- Each step loses less information

### Self-Distillation
- Model teaches itself through iterative training
- No separate teacher needed
- Particularly effective for edge models

### Online Distillation
- Teacher and student train simultaneously
- Mutual learning can improve both models
- Reduces training time and resources

</div>

<!-- 
Speaker Notes:
- Basic distillation using only final outputs is just the beginning
- Feature-based distillation adds intermediate supervision:
  - Select corresponding layers from teacher and student
  - Add MSE loss between feature maps
  - Helps student learn better representations
  - Challenge: architectural differences make alignment difficult
  - Solution: use adaptation layers or select compatible layers
- Progressive distillation prevents information loss:
  - Instead of huge → tiny in one step
  - Go through intermediate sizes: 100M → 50M → 10M → 3M
  - Each step retains more knowledge
  - Final model often significantly better than direct distillation
- Self-distillation is surprisingly effective:
  - Train model normally, then use it as its own teacher
  - Can be repeated multiple times
  - Each iteration often improves performance
  - No need for separate large teacher model
- Online distillation enables efficient training:
  - Teacher and student share some layers
  - Both models improve during training
  - Particularly useful when teacher isn't pretrained
  - Can create ensemble effects
- Choosing the right strategy depends on your constraints and models
-->

---

<!-- _class: dense -->

# Implementing Efficient Distillation

```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, 
                 temperature=4.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()  # Teacher in inference mode
        
    def train_step(self, inputs, labels, optimizer):
        # Get teacher predictions (no gradients needed)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            
        # Student forward pass
        student_logits = self.student(inputs)
        
        # Calculate combined loss
        loss = self.distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        # Standard training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        # Soft targets loss
        T = self.temperature
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
```

<div class="text-tiny" style="margin-top: 10px;">
Implementation tip: Monitor both soft and hard losses separately during training to ensure proper balance
</div>

<!-- 
Speaker Notes:
- This implementation shows a practical distillation training setup
- Key implementation details:
  - Teacher model is in eval mode (no dropout, fixed batch norm)
  - Teacher predictions computed without gradients for efficiency
  - Combined loss balances soft and hard targets
  - Standard optimization workflow otherwise unchanged
- Important practical considerations:
  - Memory usage: Need both models in memory during training
  - Batch size may need reduction to fit both models
  - Teacher can be on CPU if GPU memory is tight (slower but works)
  - Can checkpoint teacher activations for very large models
- Training tips for best results:
  - Start with higher temperature (5-10) and reduce gradually
  - Alpha around 0.7 often works well but task-dependent
  - Train for more epochs than standard training
  - Learning rate scheduling is crucial
  - Monitor both soft and hard losses separately
- Common pitfalls to avoid:
  - Forgetting to set teacher to eval mode
  - Using different preprocessing for teacher and student
  - Temperature too low (no knowledge transfer) or too high (too uniform)
  - Not warming up the learning rate
- Validation approach:
  - Always compare to directly trained student (baseline)
  - Monitor student accuracy on validation set
  - Check if student predictions become more teacher-like
-->

---

<!-- _class: section -->

# Neural Network Pruning Strategies

<!-- 
Speaker Notes:
- Pruning removes unnecessary parameters from neural networks
- Based on the observation that many networks are heavily overparameterized
- Can achieve 90%+ sparsity with minimal accuracy loss in some cases
- We'll explore different pruning strategies and their tradeoffs
-->

---

# Understanding Network Pruning

<div class="columns">
<div>

## Types of Pruning

### Unstructured Pruning
- Removes individual weights
- Maximum flexibility
- Requires sparse matrix support
- Best compression ratios

### Structured Pruning
- Removes entire channels/filters
- Hardware-friendly
- Direct speedup benefits
- Lower compression ratios

### Dynamic Pruning
- Pruning decisions during inference
- Adaptive to input
- Higher complexity

</div>
<div>

## Pruning Workflow
```python
# Basic magnitude-based pruning
import torch.nn.utils.prune as prune

def prune_model(model, 
                sparsity=0.8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune 80% of connections
            prune.l1_unstructured(
                module, 
                name='weight', 
                amount=sparsity
            )
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(
                module, 
                name='weight', 
                amount=sparsity
            )
    
    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')
    
    return model
```

</div>
</div>

<!-- 
Speaker Notes:
- Network pruning exploits the fact that many weights contribute little to output
- Unstructured pruning offers maximum flexibility:
  - Can remove any weight regardless of position
  - Achieves highest sparsity levels (95%+ possible)
  - But sparse matrices are difficult to accelerate on standard hardware
  - Mainly useful with specialized sparse computation libraries
- Structured pruning is more practical for edge deployment:
  - Removes entire channels, filters, or neurons
  - Results in smaller dense networks
  - Direct speedup on any hardware
  - But less flexible, typically 50-80% reduction
- Dynamic pruning is an emerging approach:
  - Different weights activated for different inputs
  - Can be very efficient but complex to implement
  - Not widely supported in deployment frameworks
- The basic workflow shown:
  - Identify weights to prune (magnitude-based is simplest)
  - Apply pruning masks to zero out weights
  - Fine-tune to recover accuracy
  - Make pruning permanent by removing mask infrastructure
- Key insight: not all pruning gives actual speedup on edge devices
-->

---

<!-- _class: dense -->

# Structured Pruning Techniques

<div class="highlight-box">

## Channel Pruning Strategy

```python
def evaluate_channel_importance(model, dataloader, layer_name):
    """Calculate importance scores for channels"""
    importances = []
    
    def hook_fn(module, input, output):
        # Calculate channel-wise statistics
        channel_means = output.mean(dim=[0, 2, 3])  # Average activation per channel
        importances.append(channel_means.detach())
    
    # Register hook on target layer
    target_layer = dict(model.named_modules())[layer_name]
    hook = target_layer.register_forward_hook(hook_fn)
    
    # Run evaluation
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            model(inputs)
    
    hook.remove()
    
    # Aggregate importance scores
    importance_scores = torch.cat(importances).mean(dim=0)
    return importance_scores

def prune_channels(conv_layer, bn_layer, keep_indices):
    """Remove channels from conv and corresponding BN layer"""
    # Prune convolution output channels
    conv_layer.weight.data = conv_layer.weight.data[keep_indices]
    conv_layer.out_channels = len(keep_indices)
    
    # Prune batch norm accordingly
    bn_layer.weight.data = bn_layer.weight.data[keep_indices]
    bn_layer.bias.data = bn_layer.bias.data[keep_indices]
    bn_layer.running_mean = bn_layer.running_mean[keep_indices]
    bn_layer.running_var = bn_layer.running_var[keep_indices]
```

</div>

<!-- 
Speaker Notes:
- Structured pruning is most practical for edge deployment
- Channel pruning removes entire convolutional filters:
  - Maintains dense computation structure
  - Direct reduction in FLOPs and memory
  - No special hardware support needed
  - Typical approach for CNN optimization
- Channel importance evaluation methods:
  - Magnitude-based: L1/L2 norm of filter weights
  - Activation-based: Average activation strength
  - Taylor expansion: First-order approximation of loss impact
  - Gradient-based: Accumulated gradients during training
- The code shows activation-based importance:
  - Hooks capture intermediate activations
  - Calculate statistics across spatial dimensions
  - Average over multiple batches for stability
  - Lower scores indicate less important channels
- Pruning process requires careful coordination:
  - Must update both convolution and batch normalization
  - Adjacent layers need dimension matching
  - Some architectures have skip connections to handle
- Best practices:
  - Prune gradually (10-20% at a time)
  - Fine-tune between pruning iterations
  - Some layers are more sensitive than others
  - First and last layers often most important
  - Maintain architecture balance
-->

---

<!-- _class: dense -->

# Iterative Pruning and Fine-tuning

```python
class IterativePruner:
    def __init__(self, model, trainloader, valloader, 
                 initial_sparsity=0.1, target_sparsity=0.8):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.current_sparsity = 0
        self.target_sparsity = target_sparsity
        self.sparsity_increment = initial_sparsity
        
    def prune_iteration(self):
        """Single iteration of pruning and fine-tuning"""
        # 1. Evaluate importance scores
        importance_scores = self.evaluate_all_channels()
        
        # 2. Determine channels to prune
        num_channels = len(importance_scores)
        num_prune = int(num_channels * self.sparsity_increment)
        _, indices_to_remove = torch.topk(
            importance_scores, num_prune, largest=False
        )
        
        # 3. Apply pruning
        self.apply_channel_pruning(indices_to_remove)
        self.current_sparsity += self.sparsity_increment
        
        # 4. Fine-tune the pruned model
        self.fine_tune(epochs=10)
        
        # 5. Evaluate accuracy
        accuracy = self.evaluate()
        
        return accuracy
    
    def prune_to_target(self, accuracy_threshold=0.95):
        """Iteratively prune until target sparsity or accuracy limit"""
        original_accuracy = self.evaluate()
        
        while self.current_sparsity < self.target_sparsity:
            accuracy = self.prune_iteration()
            
            if accuracy < original_accuracy * accuracy_threshold:
                print(f"Accuracy dropped below threshold at {self.current_sparsity:.1%} sparsity")
                break
                
            print(f"Sparsity: {self.current_sparsity:.1%}, Accuracy: {accuracy:.3f}")
```

<!-- 
Speaker Notes:
- Iterative pruning is more effective than one-shot pruning
- The gradual approach allows the network to adapt:
  - Remaining weights compensate for removed ones
  - Fine-tuning recovers most accuracy loss
  - Can achieve much higher sparsity levels
  - More stable and predictable results
- Key components of iterative pruning:
  - Start with small pruning ratio (10-20%)
  - Evaluate importance after each iteration
  - Fine-tune to recover accuracy
  - Monitor accuracy vs. sparsity tradeoff
  - Stop when accuracy drops below threshold
- Implementation considerations:
  - Fine-tuning epochs depend on pruning amount
  - Learning rate scheduling is important
  - May need to reset optimizer state
  - Some layers may need protection from pruning
- Advanced techniques:
  - Adaptive sparsity per layer (some layers more sensitive)
  - Gradual sparsity increase during training
  - Knowledge distillation during fine-tuning
  - Pruning-aware training from the start
- Practical tips:
  - Save checkpoints at each iteration
  - Plot accuracy vs. sparsity curve
  - Try different importance metrics
  - Consider hardware-aware pruning patterns
-->

---

<!-- _class: section -->

# Quantization Fundamentals

<!-- 
Speaker Notes:
- Quantization reduces numerical precision to save memory and computation
- One of the most effective techniques for edge deployment
- Can provide 4x memory reduction and 2-4x speedup with minimal accuracy loss
- We'll explore both the theory and practical implementation
-->

---

# Understanding Quantization

<div class="columns">
<div>

## Quantization Basics
- **Purpose**: Reduce numerical precision
- **Typical**: FP32 → INT8 (4x reduction)
- **Benefits**: 
  - Smaller model size
  - Faster computation
  - Lower power consumption
  - Better cache utilization

## Types of Quantization
- **Post-Training**: Quantize trained model
- **Quantization-Aware**: Train with quantization
- **Dynamic**: Quantize activations at runtime
- **Static**: Fixed quantization parameters

</div>
<div>

## Quantization Mathematics
<div class="text-small">

```python
# Basic quantization formula
def quantize(x, scale, zero_point, 
             num_bits=8):
    qmin = 0
    qmax = 2**num_bits - 1
    
    # Scale and shift
    q = round(x / scale + zero_point)
    
    # Clamp to valid range
    q = np.clip(q, qmin, qmax)
    
    return q

def dequantize(q, scale, zero_point):
    return scale * (q - zero_point)

# Example: Quantizing weights
weights = np.array([0.1, -0.5, 0.8, -0.2])
scale = (weights.max() - weights.min()) / 255
zero_point = round(-weights.min() / scale)

quantized = quantize(weights, scale, zero_point)
recovered = dequantize(quantized, scale, zero_point)

print(f"Original: {weights}")
print(f"Quantized: {quantized}")
print(f"Recovered: {recovered}")
print(f"Error: {np.abs(weights - recovered).mean()}")
```

</div>
</div>
</div>

<!-- 
Speaker Notes:
- Quantization is fundamental to edge AI deployment
- The basic idea: use fewer bits to represent numbers
  - FP32 uses 32 bits per number
  - INT8 uses only 8 bits
  - 4x memory reduction and faster computation
  - Modern CPUs have optimized INT8 instructions
- Quantization involves mapping continuous values to discrete ones:
  - Calculate scale factor to map range to integers
  - Zero point handles asymmetric distributions
  - Round to nearest integer
  - Clamp to valid range (0-255 for 8-bit)
- Types of quantization offer different tradeoffs:
  - Post-training: Simple but may lose accuracy
  - Quantization-aware: Better accuracy but requires retraining
  - Dynamic: Flexible but overhead at runtime
  - Static: Fastest but requires calibration data
- The mathematics shown illustrates symmetric quantization
- Key challenge: maintaining accuracy with reduced precision
  - Some layers are more sensitive than others
  - Activations often harder to quantize than weights
  - Proper calibration is crucial
-->

---

<!-- _class: dense -->

# Post-Training Quantization

<div class="highlight-box">

## PyTorch Quantization Example

```python
import torch
import torch.quantization as quantization

def quantize_model(model, calibration_loader):
    # 1. Prepare model for quantization
    model.eval()
    
    # 2. Fuse operations (Conv + BN + ReLU)
    model_fused = torch.quantization.fuse_modules(
        model, 
        [['conv1', 'bn1', 'relu1'],
         ['conv2', 'bn2', 'relu2']]
    )
    
    # 3. Specify quantization configuration
    model_fused.qconfig = quantization.get_default_qconfig('fbgemm')  # x86 CPU
    # model_fused.qconfig = quantization.get_default_qconfig('qnnpack')  # ARM CPU
    
    # 4. Prepare model (insert fake quantization modules)
    model_prepared = quantization.prepare(model_fused)
    
    # 5. Calibrate with representative data
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model_prepared(inputs)
    
    # 6. Convert to quantized model
    model_quantized = quantization.convert(model_prepared)
    
    return model_quantized

# Usage
quantized_model = quantize_model(original_model, calibration_dataloader)
torch.jit.save(torch.jit.script(quantized_model), "model_quantized.pt")
```

<div class="text-tiny" style="margin-top: 10px;">
Note: Use 'fbgemm' for x86 processors and 'qnnpack' for ARM processors
</div>

</div>

<!-- 
Speaker Notes:
- Post-training quantization is the easiest path to model compression
- PyTorch provides comprehensive quantization support:
  - Works with existing trained models
  - No retraining required
  - Can achieve good results with proper calibration
- The process involves several steps:
  1. Model preparation: Set to eval mode for consistent behavior
  2. Operation fusion: Combine ops for efficiency (Conv+BN+ReLU)
  3. Backend selection: Choose appropriate config for target hardware
     - fbgemm: Optimized for x86 servers/desktops
     - qnnpack: Optimized for ARM mobile processors
  4. Fake quantization insertion: Simulates quantization during calibration
  5. Calibration: Crucial step that determines quantization parameters
     - Run representative data through the model
     - Collects activation statistics
     - More calibration data generally improves results
  6. Conversion: Replace FP32 ops with INT8 equivalents
- Best practices for calibration:
  - Use representative data covering expected input distribution
  - Include edge cases and difficult examples
  - Typically 100-1000 samples sufficient
  - Monitor per-layer quantization statistics
- The quantized model can be further optimized with TorchScript
-->

---

<!-- _class: dense -->

# Quantization-Aware Training

```python
class QuantizationAwareTraining:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def prepare_qat(self):
        # Fuse modules
        self.model.train()
        self.model.fuse_model()  # Model-specific fusion
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)
        
    def train_qat(self, epochs=10, lr=1e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                # Standard training loop with fake quantization
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validate
            if epoch % 5 == 0:
                accuracy = self.validate()
                print(f"Epoch {epoch}, Accuracy: {accuracy:.3f}")
        
        # Convert to quantized model
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model)
        return quantized_model
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        return correct / total
```

<!-- 
Speaker Notes:
- Quantization-aware training (QAT) yields better accuracy than post-training
- Key idea: simulate quantization during training
  - Model learns to be robust to quantization effects
  - Weights and activations adapt to work well in INT8
  - Can often match FP32 accuracy
- The training process includes fake quantization:
  - Forward pass quantizes and dequantizes
  - Backward pass uses straight-through estimator
  - Gradients flow through quantization boundaries
  - Model gradually adapts to quantization constraints
- Implementation considerations:
  - Start with a pre-trained FP32 model
  - Use lower learning rate than initial training
  - May need longer training for convergence
  - Monitor both FP32 and INT8 accuracy
- QAT is particularly important for:
  - Models with aggressive quantization (INT4, INT2)
  - Architectures sensitive to quantization
  - Applications requiring highest accuracy
  - Custom quantization schemes
- Tips for successful QAT:
  - Begin with conservative quantization (INT8)
  - Fine-tune quantization parameters per layer
  - Some layers may need to remain in FP32
  - Use knowledge distillation from FP32 teacher
- Trade-off: Better accuracy but requires retraining time
-->

---

<!-- _class: dense -->

# Practical Quantization Strategies

<div class="alert-box">

## Choosing the Right Quantization Approach

### Layer-wise Sensitivity Analysis
<div class="text-small">

```python
def analyze_quantization_sensitivity(model, val_loader):
    """Determine which layers are sensitive to quantization"""
    baseline_accuracy = evaluate_model(model, val_loader)
    
    sensitivity_results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Temporarily quantize just this layer
            original_forward = module.forward
            
            # Apply fake quantization
            module.forward = lambda x: fake_quantize(original_forward(x))
            
            # Measure accuracy drop
            accuracy = evaluate_model(model, val_loader)
            sensitivity = baseline_accuracy - accuracy
            
            # Restore original
            module.forward = original_forward
            
            sensitivity_results[name] = sensitivity
    
    return sensitivity_results

# Mixed precision based on sensitivity
def apply_mixed_precision(model, sensitivity_results, threshold=0.01):
    """Keep sensitive layers in FP32"""
    for name, module in model.named_modules():
        if name in sensitivity_results:
            if sensitivity_results[name] > threshold:
                # Keep in FP32
                module.qconfig = None
            else:
                # Quantize to INT8
                module.qconfig = torch.quantization.default_qconfig
```

</div>
</div>

<!-- 
Speaker Notes:
- Not all layers respond equally to quantization
- Sensitivity analysis identifies critical layers:
  - First and last layers often most sensitive
  - Layers with small value ranges difficult to quantize
  - Skip connections may need special handling
  - Attention mechanisms in transformers particularly sensitive
- The code demonstrates per-layer sensitivity testing:
  - Quantize one layer at a time
  - Measure accuracy impact
  - Build sensitivity map
  - Use results to guide quantization strategy
- Mixed precision strategies:
  - Keep sensitive layers in higher precision
  - Quantize robust layers more aggressively
  - Balance accuracy and efficiency
  - Can achieve better results than uniform quantization
- Practical guidelines:
  - Start with INT8 for weights and activations
  - Consider INT16 for sensitive layers
  - First/last layers often need higher precision
  - Batch normalization parameters usually stay FP32
- Advanced techniques:
  - Per-channel quantization for weights
  - Learned quantization parameters
  - Outlier-aware quantization
  - Custom quantization schemes for specific ops
- Remember: 90% of benefit often comes from quantizing 90% of operations
-->

---

<!-- _class: section -->

# Measuring Optimization Impact

<!-- 
Speaker Notes:
- Now that we've explored various optimization techniques, we need to measure their impact
- It's crucial to systematically evaluate the tradeoffs between efficiency and accuracy
- We'll look at comprehensive evaluation strategies and metrics
-->

---

<!-- _class: dense -->

# Comprehensive Performance Evaluation

<div class="columns">
<div>

## Multi-Metric Assessment
<div class="text-small">

```python
class OptimizationEvaluator:
    def __init__(self, original_model, 
                 optimized_model, 
                 test_loader):
        self.original = original_model
        self.optimized = optimized_model
        self.test_loader = test_loader
        
    def evaluate_all_metrics(self):
        return {
            'accuracy': self.compare_accuracy(),
            'speed': self.compare_speed(),
            'size': self.compare_size(),
            'memory': self.compare_memory(),
            'energy': self.estimate_energy()
        }
    
    def compare_accuracy(self):
        orig_acc = self.get_accuracy(
            self.original
        )
        opt_acc = self.get_accuracy(
            self.optimized
        )
        return {
            'original': orig_acc,
            'optimized': opt_acc,
            'drop': orig_acc - opt_acc,
            'relative': opt_acc / orig_acc
        }
```

</div>
</div>
<div>

## Optimization Report
<div class="text-small">

```python
def generate_optimization_report(results):
    """Create comprehensive report"""
    report = f"""
    Optimization Impact Report
    =========================
    
    Accuracy:
    - Original: {results['accuracy']['original']:.3f}
    - Optimized: {results['accuracy']['optimized']:.3f}
    - Drop: {results['accuracy']['drop']:.3f}
    
    Speed:
    - Speedup: {results['speed']['speedup']:.2f}x
    - Original: {results['speed']['original']:.2f}ms
    - Optimized: {results['speed']['optimized']:.2f}ms
    
    Model Size:
    - Reduction: {results['size']['reduction']:.1f}%
    - Original: {results['size']['original']:.1f}MB
    - Optimized: {results['size']['optimized']:.1f}MB
    
    Memory Usage:
    - Peak reduction: {results['memory']['reduction']:.1f}%
    - Original: {results['memory']['original']:.1f}MB
    - Optimized: {results['memory']['optimized']:.1f}MB
    
    Estimated Energy:
    - Savings: {results['energy']['savings']:.1f}%
    """
    return report
```

</div>
</div>
</div>

<!-- 
Speaker Notes:
- Comprehensive evaluation is essential for optimization decisions
- Multiple metrics must be considered together:
  - Accuracy: Task performance impact
  - Speed: Inference latency improvement
  - Size: Storage and download requirements
  - Memory: Runtime memory usage
  - Energy: Battery life implications
- The evaluation framework should be systematic:
  - Consistent test conditions
  - Representative test data
  - Multiple runs for statistical validity
  - Hardware-specific measurements
- Key metrics to track:
  - Absolute values (ms, MB, mJ)
  - Relative improvements (speedup, reduction %)
  - Statistical measures (mean, std, percentiles)
  - Task-specific metrics (mAP, BLEU, etc.)
- Reporting should be clear and actionable:
  - Executive summary with key findings
  - Detailed breakdowns for technical review
  - Visualizations of tradeoffs
  - Recommendations for deployment
- Common evaluation mistakes:
  - Testing on non-representative data
  - Ignoring variance in measurements
  - Focusing on single metrics
  - Not considering deployment constraints
- Remember: The best optimization balances all requirements
-->

---

<!-- _class: dense -->

# Accuracy-Efficiency Tradeoff Analysis

<div class="highlight-box">

## Pareto Frontier Visualization

<div class="text-small">

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier(optimization_results):
    """Visualize accuracy vs efficiency tradeoffs"""
    
    # Extract data points
    methods = []
    accuracies = []
    latencies = []
    sizes = []
    
    for method, results in optimization_results.items():
        methods.append(method)
        accuracies.append(results['accuracy'])
        latencies.append(results['latency_ms'])
        sizes.append(results['model_size_mb'])
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Latency
    axes[0].scatter(latencies, accuracies, s=100)
    for i, method in enumerate(methods):
        axes[0].annotate(method, (latencies[i], accuracies[i]))
    axes[0].set_xlabel('Latency (ms)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Speed Tradeoff')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy vs Model Size
    axes[1].scatter(sizes, accuracies, s=100)
    for i, method in enumerate(methods):
        axes[1].annotate(method, (sizes[i], accuracies[i]))
    axes[1].set_xlabel('Model Size (MB)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Size Tradeoff')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Example usage
results = {
    'Original': {'accuracy': 0.95, 'latency_ms': 50, 'model_size_mb': 100},
    'Pruned': {'accuracy': 0.94, 'latency_ms': 35, 'model_size_mb': 60},
    'Quantized': {'accuracy': 0.93, 'latency_ms': 20, 'model_size_mb': 25},
    'Distilled': {'accuracy': 0.92, 'latency_ms': 15, 'model_size_mb': 20},
    'Combined': {'accuracy': 0.91, 'latency_ms': 12, 'model_size_mb': 15}
}
```

</div>
</div>

<!-- 
Speaker Notes:
- Visualizing tradeoffs helps make informed optimization decisions
- The Pareto frontier shows optimal points:
  - No other point is better in all metrics
  - Each point represents a valid tradeoff
  - Choose based on deployment requirements
- Key visualization principles:
  - Plot multiple metric pairs
  - Annotate with method names
  - Include error bars if available
  - Highlight constraint boundaries
- Common tradeoff patterns:
  - Linear: Steady degradation with optimization
  - Knee curve: Sweet spot before rapid degradation
  - Stepped: Discrete optimization levels
  - Synergistic: Combined methods outperform individual
- Decision framework:
  - Define minimum acceptable accuracy
  - Identify maximum latency/size constraints
  - Find optimal point within constraints
  - Consider safety margins
- Real-world considerations:
  - User experience thresholds
  - Hardware capability limits
  - Battery life requirements
  - Update size constraints
- Remember: There's no universally "best" point
  - Application requirements drive selection
  - May need different models for different devices
  - Consider ensemble approaches
-->

---

# Week 2 Session 1 Project Milestone

<div class="alert-box">

## Project Milestone - Phase 2: Model Architecture Optimization

### Required Deliverables:
1. **Optimization Implementation** (Choose at least 2):
   - Knowledge distillation from larger model
   - Structured pruning (channel/filter removal)
   - Quantization (post-training or QAT)
   - Architecture modification for efficiency

2. **Performance Analysis**:
   - Comprehensive metrics comparison (before/after)
   - Accuracy-efficiency tradeoff visualization
   - Per-layer optimization impact analysis
   - Hardware-specific performance measurements

3. **GitHub Documentation Updates**:
   - Optimization methodology documentation
   - Reproducible optimization scripts
   - Detailed results with visualizations
   - Lessons learned and recommendations

</div>

<!-- 
Speaker Notes:
- This week's milestone focuses on applying optimization techniques
- You should implement at least two different optimization approaches:
  - Choose based on your model type and deployment constraints
  - Consider combining techniques for better results
  - Document why you chose specific approaches
- Performance analysis should be thorough:
  - Compare all metrics, not just accuracy
  - Create clear visualizations of tradeoffs
  - Analyze which layers benefit most from optimization
  - Test on hardware similar to deployment target
- GitHub documentation is crucial:
  - Others should be able to reproduce your optimizations
  - Include both successes and failures
  - Explain your decision-making process
  - Provide clear recommendations
- Common milestone challenges:
  - Optimization takes time - start early
  - Some techniques may not work for your model
  - Accuracy drops may be larger than expected
  - Hardware testing reveals unexpected issues
- Tips for success:
  - Start with established techniques
  - Keep detailed logs of experiments
  - Save checkpoints frequently
  - Test incrementally
- Remember: Document everything, even failed attempts provide valuable insights
-->

---

<!-- _class: comfortable -->

# Summary and Next Steps

<div class="columns">
<div>

## Today We Learned
- ✅ Efficient model architectures
- ✅ Knowledge distillation
- ✅ Network pruning strategies
- ✅ Quantization fundamentals
- ✅ Performance evaluation methods
- ✅ Optimization tradeoff analysis

## Key Takeaways
- Multiple optimization techniques available
- Combining approaches often best
- Systematic evaluation is crucial
- Hardware awareness matters
- Acceptable tradeoffs are application-specific

</div>
<div>

## Next Session Preview
**Advanced ONNX Optimization Techniques**
- ONNX Runtime optimization levels
- Graph-level optimizations
- Operator fusion strategies
- Custom operators for edge
- Memory optimization techniques
- Hardware-specific tuning

## Before Next Class
- Implement optimization techniques
- Analyze performance impacts
- Document your experiments
- Prepare questions about challenges

</div>
</div>

<!-- 
Speaker Notes:
- Today we covered fundamental optimization techniques for edge AI
- Key concepts to remember:
  - Efficient architectures designed for edge from the ground up
  - Knowledge distillation transfers knowledge to smaller models
  - Pruning removes unnecessary parameters
  - Quantization reduces numerical precision
  - All techniques have accuracy-efficiency tradeoffs
- For your project work:
  - Apply at least two optimization techniques
  - Measure impact comprehensively
  - Document your process thoroughly
  - Don't be discouraged by initial results
- In our next session:
  - We'll dive into ONNX-specific optimizations
  - Learn about graph-level transformations
  - Explore hardware-specific tuning
  - Advanced memory optimization techniques
- Questions to consider:
  - Which optimization techniques work best for your model?
  - What accuracy loss is acceptable for your application?
  - How do optimizations interact with each other?
  - What deployment constraints are most challenging?
- Remember: Optimization is iterative - keep experimenting and measuring
- Great work today! Looking forward to seeing your optimization results
-->

---

<!-- _class: title comfortable -->

# Questions?
## Week 2 - Session 1
### Edge-Optimized Model Architecture

<!-- 
Speaker Notes:
- We've covered a lot of optimization techniques today
- Each technique has its own strengths and appropriate use cases
- The key is understanding how to apply them to your specific models and constraints
- Please ask questions about any optimization challenges you're facing
- I'm excited to see how you'll apply these techniques to your projects
- Remember that optimization is as much art as science - experimentation is key
- For additional support, check the course GitHub repository for example code
- Office hours are available for one-on-one optimization debugging
- Good luck with your optimization experiments this week!
-->