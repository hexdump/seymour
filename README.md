<img src="logo.png" alt="Cute cartoon picture of a venus fly trap." width="250px" align="left" />

# About

*Feed me training data, Seymour!*

Seymour is a library for quickly prototyping genetic algorithms to solve machine learning problems using models which don't have traditional optimization techniques. For example, Seymour can be used with nondifferentiable activation functions, graph generation, non fully-connected network architectures, and anything you can come up with. Seymour can be theoretically used to optimize any existing machine learning architecture, but without the power of GPU-specific code and ridiculously fast backpropogation subroutines, it'll be probably a couple of orders of magnitude slower. Please note that I'm a freshman at college, so I don't understand best practices for machine learning yet. Be nice.

# Model

Seymour makes a couple assumptions about the models you make with it:
- The model will be trained using a dataset of correct inputs/outputs, where training the model entails minimizing the distance from this dataset.
- The model takes an object for input, passes it through the model, and returns one object for output. Models can have multiple layers which can be composed together to form one model.

Seymour has some significant oversights:
- Given that combining genomes for complex data structures is significantly more complicated and arbitrary than mutating genomes, Seymour uses exclusively asexual reproduction.
- Seymour will almost always keep the best agents from the last round, meaning getting over a error peak can be difficult.

# Implementation

Seymour begins by taking in a model object mapping inputs to outputs given its instance variables, and and dataset of inputs and their desired outputs. Seymour then initializes a `population` array of`population_size` *agents*, each with a copy of the model with randomized instance variables. Seymour then, in a loop of `epochs` times, does:

- Each *agent* produces one child *agent*, with instance variables a function of the provided entropy and the parent agents instance variables. Store these agents in a new `children_population`.
- For each child agent in `children_population`, calculate the model error given the child agents instance variables.
- Append `children_population` to the end of `population`.
- Sort all agents according to their error, smallest error to largest error.
- Drop the second half of the array.

## Notes

It's very important to have as large a `population_size` as is computationally feasible. Smaller `population_size`s will result in proportionally faster epochs, but will reduce the explored sample space and potentially significantly increase training error.
