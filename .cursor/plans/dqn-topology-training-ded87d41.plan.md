<!-- ded87d41-b4a9-4fe0-b3a7-1b060b886da4 5a97a37c-2a9c-4db1-81f5-d808bff0f039 -->
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

### To-dos

- [ ] Precompute predecessor lists and weight mappings in __init__ to avoid NetworkX calls in forward pass
- [ ] Replace ParameterDict with indexed ParameterList or tensor-based storage for faster weight access
- [ ] Replace dictionary activations with tensor array, use precomputed predecessors, batch operations
- [ ] Verify optimizations maintain correctness and measure performance improvement