---
id: core_graph
title: "Execution Graph"
---

# [DRAFT] Athena Execution Graph

## What is an execution graph?

To train a neural network one needs to find a minimum of a function (called loss
function). To represent the function a graph structure is used. Consider the
following example:

```
f(x, y, z) = sin(x * y) + z
```

The function can be represented with a graph like this:

TBD picture

Athena uses directed acyclic graphs to represent any computation inside
the framework.

## Graph Implementation

Athena Graph stores nodes, that may contain operations, loaders, tensors, etc.
Each graph belongs to one and only one Athena context, and each graph has an ID
and a name unique across the context. Graph manages nodes and their memory for
user, as well as connections between nodes.

The following node types are available in Athena:

* `Node` is a generic node that stores an `Operation` and its parameters.
* `InputNode` is a special node type that allows user to feed graph with data.
* `OutputNode` is a no-op node indicating that output of its predecessor will
be used after graph execution is finished. Any other node chain can be removed
by the backend to save up compute resources.

Nodes can be added to graph like this:

```cpp
Graph graph(context, "MainGraph");
auto nodeId = graph.create<Node>(/* constructor arguments */);
```

An internal graph Node ID is returned from this method. Nodes also have names,
that must be unique across the graph. Either of these values can be used
to find node inside the graph later:

```cpp
graph.lookup(nodeId);
graph.lookup("name");
```

Users also define node connections inside the graph:

```cpp
// node1, node2 -- node IDs
graph.connect(node1, node2, 0);
```

creates an oriented edge from node1 to node2, and the newly created edge will
be node2's first argument (indexation starts from 0).

Athena `Graph` class has no copy constructor, but user can still create a copy
of a graph:

```cpp
Graph graph(context, "MainGraph");
auto newGraph = graph.clone(context, "NewGraph");
```

Graph differentiation is available in Athena:

```cpp
// graph
auto [gradGraph, gradNodes] = graph.grad();
```

will return a new graph to compute gradient and a vector of final nodes that
contain gradient values (those are not `Output` nodes). Gradients are calculated
per-node: each node has rules to construct a piece of graph that computes its
gradient.

## Graph Traversal

Athena backends work with a graph traversal that is formed taking into account
node dependencies and possible execution parallelism.

TODO describe traversal algorithm.
