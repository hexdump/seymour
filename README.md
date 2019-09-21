# seymour

*Feed me training data, Seymour!*

A genetic algorithm-based, non-standard neural network solver library.

## Getting Started

```rust

use seymour::solve;                                                   |
                                                                      |
fn main() {                                                           |
    let manifold = solve(                                             |
        vec![                                                         |
            2, 5, 1                                                   |
        ],                                                            |
        &vec![                                                        |
            (vec![0.0, 0.0], vec![0.0]),                              |
            (vec![1.0, 0.0], vec![1.0]),                              |
            (vec![0.0, 1.0], vec![1.0]),                              |
            (vec![1.0, 1.0], vec![0.0])                               |
        ],                                                            |
        100);                                                         |
    println!("[result]: {:?}", manifold.apply(&vec![0.0, 0.0]))       |
}         
use seymour::solve();

fn main() {
  ...
  solve(vec![2, 3, 4, 1], &mut dataset);
}

And seymour will work its magic!
```

