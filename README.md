<img src="logo.png" alt="Cartoon picture of a venus fly trap." width="200px" align="left" />

# seymour

*Feed me training data, Seymour!*

A genetic algorithm-based, non-standard neural network solver library.


# Getting Started

Seymour is written in Rust, and may be invoked on a dataset like so:

```rust
use seymour::solve;

fn main() {
    let model = solve(
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

This `model`'s `.apply` method will apply the calculated model to some input data.

# Comparison to Backpropagation

## Traditional Weissman Score

From HBO's Silicon Valley, [the Weissman score is](https://spectrum.ieee.org/view-from-the-valley/computing/software/a-madefortv-compression-metric-moves-to-the-real-world):

$$
W=\alpha\frac{r}{\bar{r}}\frac{\log{\bar{T}}}{\log{T}}
$$

Where $W$ is the weissman score of a compression algorithm on a given file, $\alpha$ is a scaling metric, $r$ and $T$ are the compression ratio and time to compress, and $r$ and $T$ are the compression ratio and time to compress for a standard, "universal" compressor such as GZIP or PAQ8F.

## Weissman Score for Machine Learning 

Since machine learning has a similar tradeoff between time to train and accuracy, I've applied the Weissman score to machine learning algorithms on standardized datasets.

I've compared Seymour to backpropagation using the Weissman score

