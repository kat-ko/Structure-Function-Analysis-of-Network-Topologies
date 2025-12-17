# Learning Algorithm Analysis: Interference Task vs Dynspec

## Executive Summary

This document analyzes the learning algorithms used in the original interference task implementation and compares them with the dynspec framework to identify differences and propose adaptations.

---

## 1. Original Implementation (interference_task.py)

### 1.1 Learning Algorithm Components

**Optimizer:**
- **Type**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: Fixed, specified in config (`settings['learning_rate']`)
- **Implementation**: `torch.optim.SGD(network.parameters(), lr=lr)`
- **No momentum, weight decay, or other advanced features**

**Loss Function:**
- **Type**: Mean Squared Error (MSE)
- **Implementation**: `nn.MSELoss()`
- **Usage**: Regression task - predicts continuous values (cos/sin of feature angles)
- **Feature-specific loss**: Loss computed on different output dimensions based on `feature_probe` (0 or 1)

**Learning Rate Scheduling:**
- **None** - Fixed learning rate throughout training

**Training Loop Structure:**
```python
for epoch in range(n_epochs):
    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        # Forward pass
        out, hid = network(input)
        # Loss computation (feature-specific)
        loss = loss_function(out[:, :2] or out[:, 2:4], joined_label)
        # Backward pass
        loss.backward()
        optimizer.step()
        # Store metrics
```

**Key Characteristics:**
1. **Simple gradient descent**: Standard backpropagation with SGD
2. **Conditional updates**: Updates depend on `do_update` flag and `test_stim` flag
3. **No test set evaluation**: Only training metrics are tracked
4. **No model checkpointing**: No best model state saving
5. **No early stopping**: Training runs for fixed number of epochs
6. **Per-batch metrics**: Metrics stored for every batch

---

## 2. Dynspec Implementation (dynspec/training.py)

### 2.1 Learning Algorithm Components

**Optimizer:**
- **Type**: AdamW (Adam with decoupled weight decay)
- **Configuration**: Parameters from `config["optim"]` dictionary
- **Implementation**: `torch.optim.AdamW(model.parameters(), **config["optim"])`
- **Typical parameters**: `lr`, `weight_decay`, `betas`, etc.

**Loss Function:**
- **Type**: Cross Entropy Loss
- **Implementation**: `F.cross_entropy(output, t_target, reduction="none")`
- **Usage**: Classification task - predicts class probabilities
- **Multi-task support**: Handles nested loss computation for multiple tasks/modules

**Learning Rate Scheduling:**
- **Type**: ExponentialLR scheduler
- **Conditional**: Only if `gamma` parameter provided in `config["optim"]`
- **Implementation**: `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)`

**Training Loop Structure:**
```python
for epoch in range(n_epochs + 1):  # +1 for initial test
    if training and epoch > 0:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output, _ = model(data)
            # Loss computation (nested for multi-task)
            complete_loss = get_loss(output, t_target, use_both=both)
            loss = nested_mean(complete_loss)
            # Backward pass
            loss.backward()
            if config["training"].get("check_grad", False):
                check_grad(model)
            optimizer.step()
            # Store training metrics
    
    if testing:
        # Test set evaluation
        test_results = test_community(model, device, test_loader, config, ...)
        # Track best model
        if test_results["test_loss"] < best_loss:
            best_loss = test_results["test_loss"]
            best_state = copy.deepcopy(model.state_dict())
    
    if scheduler is not None:
        scheduler.step()
```

**Key Characteristics:**
1. **Advanced optimizer**: AdamW with adaptive learning rates
2. **Learning rate scheduling**: Exponential decay support
3. **Test set evaluation**: Regular evaluation on test set during training
4. **Best model tracking**: Saves best model state based on test loss
5. **Early stopping**: Optional early stopping based on accuracy threshold
6. **Gradient checking**: Optional gradient validation
7. **Modular design**: Separate train/test phases, configurable behavior
8. **Multi-task support**: Handles nested outputs for multiple tasks/modules

---

## 3. Key Differences Summary

| Aspect | Interference Task | Dynspec |
|--------|------------------|---------|
| **Optimizer** | SGD (fixed LR) | AdamW (configurable) |
| **Loss Function** | MSE (regression) | Cross Entropy (classification) |
| **LR Scheduling** | None | ExponentialLR (optional) |
| **Test Evaluation** | None during training | After each epoch |
| **Model Checkpointing** | None | Best model state saved |
| **Early Stopping** | None | Accuracy-based (optional) |
| **Gradient Checking** | None | Optional |
| **Configuration** | Hard-coded params | Config dictionary |
| **Task Type** | Regression (circular features) | Classification |
| **Update Logic** | Conditional (do_update, test_stim) | Standard (all batches) |

---

## 4. Why These Differences Exist

### 4.1 Task-Specific Requirements

**Interference Task:**
- **Regression problem**: Predicting continuous circular features (angles)
- **Feature-specific learning**: Different output dimensions for different features
- **Conditional updates**: Test trials don't update weights (important for experimental design)
- **Simple architecture**: Single network, straightforward training

**Dynspec:**
- **Classification problem**: Predicting discrete classes
- **Multi-module architecture**: Complex modular networks with multiple agents
- **Multi-task learning**: Handles multiple tasks simultaneously
- **Research focus**: Studying modularity and specialization

### 4.2 Design Philosophy

**Interference Task:**
- **Minimalist approach**: Simple SGD sufficient for the task
- **Experimental control**: Conditional updates preserve experimental design
- **Focus on interference**: Catastrophic forgetting is the research question

**Dynspec:**
- **State-of-the-art training**: AdamW for better convergence
- **Robust evaluation**: Test set monitoring prevents overfitting
- **Flexibility**: Config-based for easy experimentation

---

## 5. Adaptation Strategy

### 5.1 What Should Be Adapted

**High Priority:**
1. **Optimizer flexibility**: Support both SGD and AdamW (configurable)
2. **Learning rate scheduling**: Add optional scheduler support
3. **Test set evaluation**: Add test set evaluation during training
4. **Best model tracking**: Save best model state based on test performance
5. **Configuration structure**: Move to config-based parameter management

**Medium Priority:**
6. **Early stopping**: Optional early stopping based on accuracy
7. **Gradient checking**: Optional gradient validation
8. **Modular training loop**: Separate train/test phases more clearly

**Low Priority (Task-Specific):**
9. **Loss function**: Keep MSE (task requires regression)
10. **Conditional updates**: Keep do_update/test_stim logic (experimental design)

### 5.2 What Should Stay the Same

1. **MSE Loss**: Task requires regression, not classification
2. **Feature-specific loss**: Core to the interference task design
3. **Conditional update logic**: Essential for experimental validity
4. **Circular feature representation**: Task-specific requirement

---

## 6. Implementation Considerations

### 6.1 Optimizer Selection

**Option 1: Make optimizer configurable**
- Add `optimizer_type` to config: "SGD" or "AdamW"
- Pass optimizer parameters from config
- Default to SGD for backward compatibility

**Option 2: Always use AdamW**
- More modern, typically better convergence
- May require learning rate tuning
- Breaks backward compatibility

**Recommendation**: Option 1 (configurable)

### 6.2 Learning Rate Scheduling

**Implementation:**
- Add `scheduler_type` and `scheduler_params` to config
- Support ExponentialLR (like dynspec) and others
- Optional - default to None for backward compatibility

### 6.3 Test Set Evaluation

**Challenge**: Interference task doesn't have separate test set
- Current: `test_trial` flag marks test trials within training data
- Solution: Create separate test DataLoader from test trials
- Evaluate after each epoch (like dynspec)

### 6.4 Configuration Structure

**Current structure:**
```python
settings = {
    'n_epochs': ...,
    'learning_rate': ...,
    'batch_size': ...,
    ...
}
```

**Proposed structure (dynspec-like):**
```python
config = {
    'training': {
        'n_epochs': ...,
        'batch_size': ...,
        ...
    },
    'optim': {
        'lr': ...,
        'optimizer': 'SGD' or 'AdamW',
        'weight_decay': ...,
        'gamma': ...  # for scheduler
    },
    ...
}
```

---

## 7. Compatibility Considerations

### 7.1 Backward Compatibility

- **Default behavior**: Should match current implementation
- **Config migration**: Provide helper to convert old config format
- **Parameter names**: Keep existing names as aliases

### 7.2 Integration with Existing Code

- **Network interface**: Already abstracted, should work seamlessly
- **Data loading**: No changes needed
- **Metrics tracking**: Extend to include test metrics

---

## 8. Benefits of Adaptation

1. **Better convergence**: AdamW often converges faster/better than SGD
2. **Overfitting prevention**: Test set evaluation helps detect overfitting
3. **Model selection**: Best model tracking ensures best performance
4. **Flexibility**: Config-based approach enables easier experimentation
5. **Consistency**: Aligns with dynspec framework for future integration
6. **Research value**: Enables fair comparison with dynspec networks

---

## 9. Potential Challenges

1. **Task mismatch**: Dynspec uses classification, interference uses regression
   - **Solution**: Keep MSE loss, adapt other components

2. **Conditional updates**: Interference has unique update logic
   - **Solution**: Preserve this logic, add test evaluation separately

3. **Learning rate tuning**: AdamW may need different LR than SGD
   - **Solution**: Make LR configurable, provide defaults

4. **Test set definition**: No clear test set separation
   - **Solution**: Use test_trial flag to create test DataLoader

---

## 10. Recommended Implementation Order

1. **Phase 1: Configuration Structure**
   - Refactor to config-based parameter management
   - Maintain backward compatibility

2. **Phase 2: Optimizer Flexibility**
   - Add support for AdamW (optional, default SGD)
   - Add optimizer parameters to config

3. **Phase 3: Test Set Evaluation**
   - Create test DataLoader from test trials
   - Add test evaluation after each epoch

4. **Phase 4: Learning Rate Scheduling**
   - Add scheduler support (optional)
   - Implement ExponentialLR

5. **Phase 5: Best Model Tracking**
   - Track best model based on test performance
   - Save best state

6. **Phase 6: Early Stopping (Optional)**
   - Add early stopping based on accuracy
   - Make it optional/configurable

---

## Conclusion

The interference task uses a simpler learning algorithm (SGD + MSE) compared to dynspec (AdamW + Cross Entropy + scheduling). While the core task requirements (regression, conditional updates) should be preserved, adapting the training infrastructure to be more similar to dynspec will provide:

- Better training practices (test evaluation, model checkpointing)
- More flexibility (configurable optimizer, scheduling)
- Easier integration with dynspec networks in the future
- Better research practices (overfitting detection, model selection)

The adaptation should be done incrementally, maintaining backward compatibility while adding new capabilities.

