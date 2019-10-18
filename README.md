<img src="logo.png" alt="Cartoon picture of a venus fly trap." width="200px" align="left" />

# seymour

*Feed me training data, Seymour!*

A genetic algorithm-based, non-standard neural network solver library. I'm using Seymour in my personal research in order to optimize weird semi-structured AI systems with nondifferentiable activation functions.

# Getting Started

Seymour is written in Rust, and may be invoked on a dataset like so:

```rust
use seymour::solve;
use seymour::net_evaluate;

fn main() {
    let model = solve(
        net_evaluate, // the genome evaluation function to use
        vec![ // the layer sizes. input and output sizes must match the data.
            2, 5, 1
        ],                                                           
        &vec![ // the dataset; list of tuples of input and output vectors.
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0])
        ],                                                           
        100 // the number of rounds to train for.
  	);
    println!("[result]: {:?}", model.apply(&vec![0.0, 0.0]))
}         
```
