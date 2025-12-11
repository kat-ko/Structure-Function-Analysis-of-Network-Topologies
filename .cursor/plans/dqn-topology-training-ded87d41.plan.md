---
name: Optimize DQN Training Performance
overview: ""
todos:
  - id: cd605c4b-07b0-4c17-a036-c8635f6b1d63
    content: Precompute predecessor lists and weight mappings in __init__ to avoid NetworkX calls in forward pass
    status: pending
  - id: 9ce950b0-1378-4f5c-93c6-aa5a03d3f367
    content: Replace ParameterDict with indexed ParameterList or tensor-based storage for faster weight access
    status: pending
  - id: bd4ccd5a-51ca-479b-a160-678a8ec5d4b0
    content: Replace dictionary activations with tensor array, use precomputed predecessors, batch operations
    status: pending
  - id: d817c57b-e762-49ca-9ccd-59e4f3ac00df
    content: Verify optimizations maintain correctness and measure performance improvement
    status: pending
---

# Optimize DQN Training Performance

## Performance Bottlenecks Identified

1. **String-based parameter access** - `self.weights[f"{neighbor}_to_{node}"]` is extremely slow (string formatting + dict lookup per edge)
2. **NetworkX operations in forward pass** - `topology.predecessors(node)` called repeatedly during training
3. **Dictionary-based activations** - Python dict lookups instead of tensor operations
4. **Sequential processing** - Nodes processed one-by-one, preventing vectorization
5. **Repeated tensor creation** - `torch.full()` called for each node

## Optimization Strategy

### 1. Precompute Graph Structure

**File**: `topology_playground.py` - `TopologyNetwork.__init__()`

- Precompute predecessor lists for all nodes and store as lists (not NetworkX calls)
- Create mapping from (source, target) to weight parameter index
- Store node-to-predecessor mapping as a list of lists

### 2. Replace String-Based Parameter Access

**File**: `topology_playground.py` - `TopologyNetwork.__init__()`

- Instead of `nn.ParameterDict` with string keys, use indexed `nn.ParameterList` or a single tensor
- Create mapping: `(source, target) -> parameter_index`
- Access weights directly by index instead of string formatting

### 3. Optimize Forward Pass

**File**: `topology_playground.py` - `TopologyNetwork.forward()`

- Replace dictionary activations with a pre-allocated tensor array
- Use precomputed predecessor lists instead of NetworkX calls
- Batch operations where possible
- Use `torch.zeros()` once instead of `torch.full()` per node

### 4. Cache Graph Operations

**File**: `topology_playground.py` - `TopologyNetwork.__init__()`

- Precompute all predecessor lists at initialization
- Store as `self.predecessors_list[node] = [list of predecessor nodes]`
- Store as `self.predecessor_weights[node] = [list of weight parameters]`

## Expected Performance Improvements

- **10-50x faster forward pass** by eliminating string operations and dict lookups
- **Better GPU utilization** with tensor-based operations
- **Reduced Python overhead** by minimizing graph operations in hot path

## Implementation Details

- Keep the topological ordering for interpretability
- Maintain the same network structure and behavior
- Only optimize the internal implementation, not the interface