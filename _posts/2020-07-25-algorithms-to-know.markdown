---
layout: post
title: "Algorithms to know"
date: 2020-07-25 12:12
tags: math code
---

# Algorithms to know

## QuickSort

**Purpose**: Sort array

**Idea**:

> Recursive sorting of smaller arrays until array size is 1 and there's nothing to sort.
> Pick a pivot, create array of smaller elements, and array of larger elements
> run sorting on smaller arrays. If array size is 1 return the array.

**Time Complexity**: $O(n\log(n))$. <br> $\log(n)$ levels due to splitting the array, then for each level we go through the whole array hence $n$.

**Code**

```python
def quicksort(array: List[int]) -> List[int]:
    """ Sort array of numbers from smallest to largest"""
    # Base case of recursion
    # array of size, nothing to sort
    if len(array) < 2:
        return array

    # Pick pivot at the middle of the array
    pivot = array[len(array) // 2]

    # create array of smaller than pivot elements
    smaller_arr = [el for el in array if el < pivot]

    # create array of larger than pivot elements
    larger_arr = [el for el in array if el > pivot]

    # run sorting on the array of smaller and larger elements
    # and combine with the pivot element in the middle
    return quicksort(smaller_arr) + [pivot] + quicksort(larger_arr)
```

**Example**

```python
import random
# sample 10 random numbers between 1 to 100
array = random.sample(range(1, 100), 10)

print(f"Input array: {array}")
print(f"Sorted array: {quicksort(array)}")
```

## Binary Search

**Purpose**: Find element in a sorted array.

**Idea**:

> Compare element to middle element in the array
> Based on comparison reduce search to half the array
> continue until element is found or no elements remain in array

**Time complexity**: $O(\log n)$

**Code**

```python
def binarysearch(array: List[int], number: int) -> int:
    """
    Find element in a sorted array
    """

    # initialize start and end indices
    start = 0
    end = len(array)

    while start < end:
        middle = (start + end) // 2  # get middle index of array
        if number == array[middle]:  # element was found
            return middle
        if array[middle] < number:
            # middle element is smaller than number, update start to after middle index
            # keep end the same. This halves the search array to the right
            start = middle + 1
        else:
            # middle element is larger than number, update end to before middle index
            # keeps start the same. This halves the search array to the left.
            end = middle - 1
    # loop has finished with an empty array, number not found
    return -1
```

**Example**

```python
# sample 100 random numbers between 1 to 1000, and sort
array = sorted(random.sample(range(1, 100), 10))
print(f"Input array: {array}")

# sample random number from array
number = random.sample(array, 1)[0]
print(f"Number to find: {number}")

print(f"Index found: {binarysearch(array, number)}")
```

## Graph Search

**Purpose**:
Breadth-First (BF) search is used to find if a path to a node exists in a graph
and finds the shortest path.

**Idea**:

> Go through all neighbors of a node before moving to the next-neighbor nodes.
> Keep neighbors in a queue to go through them by order of insertion.

**Time Complexity**: $O(E+V)$, at most going through every edge ($E$) and adding each edge to a queue ($V$)

**Code**

```python
from collections import deque

def trace_path(parents, start, end):
    """ Trace path back from end to start using parents """
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def graphsearch(graph: Dict[str, List[str]], start: str, end: str) -> bool:
    """
    Search shortest path from start to end node in a graph using breadth-first search
    """

    queue = deque()  # create empty queue
    queue += [start]  # add start node to queue
    searched: List[str] = []  # initialise list of processed nodes
    # initialise dictionary fo parent of node for tracing back path
    parents: Dict[str, str] = {}

    while queue:
        node = queue.popleft()  # get node from queue
        if node not in searched:  # process node if not already processed
            if node == end:  # found end node
                print(f"path to {end} found")
                print(f"path: {trace_path(parents, start, end)}")
                return True
            else:
                queue += graph[node]  # add neighbors of node to the queue
                searched.append(node)  # add node to processed nodes to not repeat
                for n in graph[node]:  # add parent for each of node's neighbors
                    parents[n] = node

    # queue is empty, path not found
    print(f"path to {end} not found")
    return False
```

**Example**

```python
graph = {}
graph["Mark"] = ["Alice", "Bob"]
graph["Alice"] = ["David"]
graph["Bob"] = ["John"]
graph["John"] = ["David"]
graph["David"] = []
graph["Alex"] = ['Sven']
graph["Sven"] = []

found = graphsearch(graph, "Mark", "David")
print()
found = graphsearch(graph, "Mark", "John")
print()
found = graphsearch(graph, "Mark", "Sven")
```

```
path to David found
path: ['Mark', 'Alice', 'David']

path to John found
path: ['Mark', 'Bob', 'John']

path to Sven not found
```

## Dijkstra Algorithm

**Purpose**: Finds shortest path (weight-wise) to a node in a graph.

**Idea**:

> Find smallest weight node using a min-heap data structure,
> there is no cheaper way to get to that node.
> Check if there's a cheaper way to get to its neighbors, update distances, min-heap
> and parents. Repeat for all nodes in the graph, keeping track of distances to nodes.

**Complexity**: $O(E\log(V))$, the log comes from the [min-heap](https://en.wikipedia.org/wiki/Binary_heap) (A binary tree where each node's value is less or equal to its children).
An alternative implementation with an array instead of min-heap will have $O(V^2)$

```python
import heapq  # standard library min-heap

def trace_path(parents, start, end):
    """ Trace path back from end to start using parents """
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path


def dijkstra(
    graph: Dict[str, Dict[str, float]], start: str, end: str
) -> Tuple[List[str], float]:    """
    Find shortest weighted path between nodes in a non-negative weights
    directed and acyclic graph
    """

    # Initialize dictionary of distances and parents
    distances: Dict[str, float] = {}
    parents: Dict[str, str] = {}

    # Initialize a min-heap of weights
    weights: List = []
    heapq.heapify(weights)

    # Set all distances to nodes to infinity
    infinity = float("inf")
    for node in graph:
        parents[node] = ""
        distances[node] = infinity

    # add start node to min-heap with weight zero, and set distance to zero
    heapq.heappush(weights, (0, start))
    distances[start] = 0

    # Get lowest weight nodes to start going through its neighbors
    while weights:
        node = heapq.heappop(weights)  # get node from min-heap

        neighbors = graph[node[1]]  # current node's neighbors
        for n in neighbors.keys():  # loop over neighbors
            new_dist = distances[node[1]] + neighbors[n]
            if new_dist < distances[n]:
                # update the weights and parent only if the new weighted path weighs less
                # in that case a more efficient path to the neighbor has been found
                distances[n] = new_dist
                parents[n] = node[1]
                heapq.heappush(weights, (new_dist, n))

    # get path of smallest weight using parents dictionary
    path = trace_path(parents, start, end)
    # get final distance of optimal path
    distance = distances[end]
    print(
        f"Shortest weighted-path from {start} to {end}: {path}, with total distance: {distance}"
    )
    return path, distance
```

**Example**

```python
# define graph
graph: Dict[str, Dict[str, float]] = {}

graph["A"] = {}
graph["A"]["B"] = 2
graph["A"]["C"] = 5
graph["B"] = {}
graph["B"]["C"] = 1
graph["B"]["D"] = 6
graph["C"] = {}
graph["C"]["D"] = 2
graph["D"] = {}

# find path and weight
path, weight = dijkstra(graph, "A", "D")
path, weight = dijkstra(graph, "A", "C")
```

```
Shortest weighted-path from A to D: ['A', 'B', 'C', 'D'], with total distance: 5
Shortest weighted-path from A to C: ['A', 'B', 'C'], with total distance: 3
```

## Uniform d-sphere sampling

**Purpose**: Sample points from a d-sphere uniformly without rejection

**Idea**:
d-dimensional Gaussians are isotropic, sample and normalize to get uniform
samples on the boundary. Sample radii from $\pi(r) \sim r^d$, as the volume element has $\int d\Omega r^{d-1}dr$, with $d\Omega$ the solid angle element.

**Code**

```python
import numpy as np


def sample_sphere(N: int, d: int) -> List[List[float]]:
    np.random.seed(42)
    samples = np.random.normal(size=(N, d))
    # normalize
    samples = [s / np.sqrt((s ** 2).sum()) for s in samples]
    uniform_radii = np.random.uniform(size=N) ** (1 / d)
    samples = [samples[i] * uniform_radii[i] for i in range(len(samples))]
    return np.array(samples)
```

**Example**

```python
samples = sample_sphere(1000, 2)

import matplotlib.pyplot as plt

plt.scatter(samples[:, 0], samples[:, 1], alpha=0.75, edgecolor="k")
x = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(x), np.sin(x), color="k")
plt.axis("equal");
plt.axis("off");
```

## Finite Distribution sampling

**Purpose**: Sample a distribution with a finite number of states $k\in\{1,..,K\}$ without rejection

**Idea**: Create cummulative sums of the probabilities and draw unfiromly from
0 to the total sum. The first state with cummulative sum larger than the drawn number is chosen.

**Code**

```python
def finite_sample(probs: List[float]) -> int:
    """
    Sample from a finite distribution without rejection
    """
    # create cummulative sums
    cprobs = [sum(probs[:i]) for i in range(len(probs) + 1)]
    r = np.random.uniform(0, max(cprobs))
    # bisect the cummulative probabilities to find the index k
    kmin = 0
    kmax = len(cprobs)
    while kmin < kmax:
        k = (kmin + kmax) // 2
        if cprobs[k] < r:
            kmin = k
        elif cprobs[k - 1] > r:
            kmax = k
        else:
            return k
```

**Example**

```python
probs = [0.1, 0.1, 0.2, 0.6]
samples = []
for i in range(50000):
    samples.append(finite_sample(probs))

plt.hist(
    samples, bins=range(1, len(probs) + 2), edgecolor="k", align="left", density=True
)
plt.xticks(range(1, len(probs) + 1))
plt.xlabel("k");
plt.ylabel("density");
```

The extension to continuous distribution ($\{k,p(k)\} \rightarrow \{x, \pi(x)\})$ consists of
changing the cumulative summation to an integral $\Pi(x) = \int_{-\infty}^{x} dx'\pi(x')$.
To get $x$, we perform a sample transformation from uniform over $[0,1]$ to $\Pi(x)$, such that $x$ is sampled from $\Pi^{-1}([0,1])$.

Example: <br>
$\pi(x) \propto x^{\gamma}$ <br>
$\Pi(x) = x^{\gamma+1}$ <br>
$x = [0,1]^{1/(\gamma+1)}$ <br>

## Thompson Sampling

References: [paper](https://arxiv.org/abs/1111.1797), [tutorial](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)

**Purpose**: Strategy for maximizing reward in multi-arm bandit scenario, balancing explore-exploit.

**Idea**: Use the posterior of the actions' probabilities and use the maximum sampled probability to perform an action. Then update posterior with new observation. With time, posterior of favorable action will converge to true probability, while other actions will be sampled more and more rarely.

**Code**

```python
np.random.seed(0)

# Mean probability of sucess for arm i
unknown_means = [0.3, 0.5, 0.8]


def play_arm(arm: int) -> int:
    """
    Plays a random arm with a probability of reward
    """
    p = np.random.uniform()
    if p <= unknown_means[arm]:
        return 1
    return 0


def thompson_sampling(arms: int, time_periods: int):
    s = np.zeros(arms)
    f = np.zeros(arms)

    max_arm = np.max(unknown_means)  # unknown optimal arm, tracked for demonstration
    regrets = []  # store regrets

    # store posterior distributions
    prob_dists = {}
    for i in range(arms):
        prob_dists[i] = []
        prob_dists[i].append([0, 0])

    for t in range(time_periods):
        thetas = []
        # sample from posterior distributions
        for i in range(arms):
            thetas.append(np.random.beta(s[i] + 1, f[i] + 1))

        # select arm to play
        arm = np.argmax(thetas)

        # add regret, difference between max arm and selected arm
        regrets.append(max_arm - unknown_means[arm])

        # get reward
        if play_arm(arm):
            s[arm] += 1
        else:
            f[arm] += 1

        prob_dists[arm].append([s[arm], f[arm]])

    return prob_dists, regrets
```

**Example**

```python
rounds = 100
timesteps = 1000
all_regrets = []
for i in range(rounds):
    probs, regrets = thompson_sampling(3, timesteps)
    all_regrets.append(regrets)

plt.plot(np.array(all_regrets).mean(axis=0))
plt.xlabel("Timestep")
plt.ylabel("Average regret")
plt.title("Average regret at time period")
```

The regret converges to zero as the sampling converges to the most favorable arm.

```python
import gif
from scipy.stats import beta

@gif.frame
def plot_posterior(i):
    """
    Plot current posterior distributions of the arms
    """
    x = np.linspace(0, 1, 200)
    for j in range(len(probs.keys())):
        try:
            s, f = probs[j][i]
            plt.plot(x, beta.pdf(x, s + 1, f + 1), label=f"arm {j+1}")
            plt.xlim([0, 1])
        except:
            # the corresponding arm hasn't been used in subsequent timesteps
            s, f = probs[j][-1]
            plt.plot(x, beta.pdf(x, s + 1, f + 1), label=f"arm {j+1}")
            plt.xlim([0, 1])
    plt.legend(loc=2)
    plt.title(f"run {i}")


frames = [plot_posterior(i) for i in range(timesteps)]
```

The Bayesian updates of the posterior distribution of the arms based on the observed rewards,
result in the Thompson sampling to be more confident about the higher average-reward arm, as it gets
sampled more and more compared to the other arms for which there is close to zero posterior probability
that they have a higher average reward.
