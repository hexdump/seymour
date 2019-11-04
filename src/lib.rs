use rand::prelude::*;
use rayon::join;
use rayon::prelude::*;
use fastapprox::fast::*;
use std::f64::consts::PI;
use std::cmp::Ordering;

// hyperteparameters
const GENE_INITIALIZATION_SD: 2.0;

const MAX_SAMPLE_SIZE: usize = 250;

struct OptimizerParameters {
    num_layers: usize,
    layer_sizes: Vec<usize>,
    input_size: usize,
    output_size: usize,
    population_size: usize,
    genome_size: usize,
    dataset: Vec<(Vec<f64>, Vec<f64>)>,
    test_dataset: Vec<(Vec<f64>, Vec<f64>)>,
    iterations: usize
}

pub struct Agent {
    pub genome: Vec<f64>,
    pub error: f64
}

pub struct Model {//<'a> {
    agent: Agent,
    op: OptimizerParameters, //<'a>
    evaluate: fn(&Agent, &Vec<f64>, &mut Vec<Vec<f64>>) -> Vec<f64>
}

impl Model {
    pub fn apply(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut layers: Vec<Vec<f64>> = Vec::new();
        
        for layer_size in self.op.layer_sizes.iter() {
            layers.push(vec![0.0; *layer_size]);
        }

        return (self.evaluate)(&self.agent, input, &mut layers);
    }

    pub fn dump(&self, filename: String) {
        let mut file = std::fs::File::create(filename).expect("create failed");
        file.write_all(format!("layers = {:?}\n", self.op.layer_sizes).as_bytes()).expect("write failed.");
        file.write_all(format!("genome = {:?}\n", self.agent.genome).as_bytes()).expect("write failed");
        println!("data written to file" );
    }
}

fn diff(x0: f64, x1: f64) -> f64 {
    return 2.0 * (x0 - x1).abs() / (x0 + x1).abs();
}

fn vec_diff(v0: &Vec<f64>, v1: &Vec<f64>) -> f64 {  
    let mut total: f64 = 0.0;
    
    for i in 0..v0.len().min(v1.len()) {
        total += diff(v0[i], v1[i]);
    }

    total *= 1.0 + (v0.len() as f64 - v1.len() as f64) / (v0.len().max(v1.len()) as f64);
    
    return total / (v0.len() as f64);
}

fn update_error(evaluate: fn(&Agent, &Vec<f64>, &mut Vec<Vec<f64>>) -> Vec<f64>, agent: &mut Agent, layers: &mut Vec<Vec<f64>>, op: &OptimizerParameters) {
    let mut total: f64 = 0.0;
    let mut sample_size: usize = op.dataset.len();
    if sample_size > MAX_SAMPLE_SIZE {
        sample_size = MAX_SAMPLE_SIZE;
    }
    for datum in op.dataset[..sample_size].iter() {
        let input = &datum.0;
        let output = &datum.1;
        let est = evaluate(&agent, input, layers);
        total += vec_diff(&est, output);
    }
    total = total / (op.dataset[..sample_size].len() as f64);
    agent.error = total;
}

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}


pub fn manifold(genes: &[f64], inputs: Vec<f64>) -> f64 {
    let mut total = 0.0;
    for i in 0..inputs.len() {
        let x = inputs[i];
        total += (genes[i] * x).sin() + (x * genes[i] + x).cos();
    }
    return total / inputs.len() as f64;
}

pub fn net_evaluate (agent: &Agent,
                     input: &Vec<f64>,
                     layers: &mut Vec<Vec<f64>>) -> Vec<f64>{

    let mut g = 0;
    let mut last: Vec<f64> = Vec::new();
    
    // copy input into the first space in the network
    for i in 0..layers[0].len() {
        layers[0][i] = input[i];
        last.push(input[i]);
    }

    for i in 1..layers.len() {
        // for each layer
        let current = &mut layers[i]; 
        for j in 0..current.len() {            
            let len_last = last.len();
            current[j] = manifold(&agent.genome[g..g+len_last], last.to_vec());
            g += len_last;
        }
        last = current.to_vec();
    }

    return last.to_vec(); //output;
}

fn compare_floats(a: f64, b: f64, decimal_places: u8) -> Ordering {
    let factor = 10.0f64.powi(decimal_places as i32);
    let a = (a * factor).trunc();
    let b = (b * factor).trunc();
    if a > b {
        return Ordering::Greater;
    }
    else if a < b {
        return Ordering::Less;
    }
    else {
        return Ordering::Equal;;
    }
}

fn breed_genomes(a: &mut Vec<f64>, b: &mut Vec<f64>) {
    for i in 0..a.len() {
        if i % 2 == 0 {
            let temp = a[i];
            b[i] = a[i];
            a[i] = temp;
        }
    }      
}

use rand::distributions::{Normal, Distribution};

fn mutate_genome(mut agent: &mut Agent, mut rng: ThreadRng) {
    for i in 0..agent.genome.len() {
        let r: f64 = rng.gen();
        if (r > 0.25) {
            if agent.error > 0.0 {
                let normal = Normal::new(agent.genome[i].into(), ((sigmoid(agent.error * agent.error)).into()));
                agent.genome[i] = normal.sample(&mut rng) as f64;
            }
        }
    }
}

use std::{
    fs::File,
    io::{prelude::*, BufReader},
    path::Path,
};

fn breed_population(population: &mut Vec<Agent>) {

//    let temp_genome = Vec<f64>;
    
    for i in 0..population.len() - 1 {
        let (left_slice, right_slice) = population.split_at_mut(i + 1);
        let left = left_slice.last_mut().unwrap();
        let right = right_slice.first_mut().unwrap();
//        let left = population[i].genome;
//        let right = population[i + 1].genome;
//        &mut population[i].genome = breed_genomes(&left, &right)[0];
//        &mut population[i + 1].genome = breed_genomes(&left, &right)[1];
        
        //        let mut population_slice = &mut population[i..i+1];
//        let left = population_slice.first_mut().unwrap();
//        let right = population_slice.last_mut().unwrap();
        breed_genomes(&mut left.genome,
                      &mut right.genome);
    }
}

 pub fn solve(evaluate: fn(&Agent, &Vec<f64>, &mut Vec<Vec<f64>>) -> Vec<f64>,
              layer_sizes: Vec<usize>,
              dataset: &Vec<(Vec<f64>, Vec<f64>)>,
              test_dataset: &Vec<(Vec<f64>, Vec<f64>)>,
              population_size: usize,
              iterations: usize) -> Model {
    
    let mut op = OptimizerParameters {
        num_layers: layer_sizes.len(),
        layer_sizes: layer_sizes.to_vec(),
        input_size: layer_sizes[0],
        output_size: layer_sizes[layer_sizes.len() - 1],
        population_size: population_size,
        genome_size: 0,
        dataset: dataset.to_vec(),
        test_dataset: test_dataset.to_vec(),
        iterations: iterations
    };

    println!("{:?}", layer_sizes);

    let mut best_genome: Vec<f64> = Vec::new();
    let mut best_error: f64 = std::f64::INFINITY;
    
    for i in 0..op.num_layers - 1 {
        op.genome_size += layer_sizes[i] * layer_sizes[i + 1];
    }
    
    // intialize the space for data processing for each core.
    let mut layers: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers.push(vec![0.0; *layer_size]);
    }  
    let mut layers0: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers0.push(vec![0.0; *layer_size]);
    }
    let mut layers1: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers1.push(vec![0.0; *layer_size]);
    }
    let mut layers2: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers2.push(vec![0.0; *layer_size]);
    }
    let mut layers3: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers3.push(vec![0.0; *layer_size]);
    }
    let mut layers4: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers4.push(vec![0.0; *layer_size]);
    }
    let mut layers5: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers5.push(vec![0.0; *layer_size]);
    }
    let mut layers6: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers6.push(vec![0.0; *layer_size]);
    }
    let mut layers7: Vec<Vec<f64>> = Vec::new();
    for layer_size in op.layer_sizes.iter() {
        layers7.push(vec![0.0; *layer_size]);
    }
     
    let mut rng: ThreadRng = thread_rng();

    // note that this will initialize these arrays INDEPENDENTLY
    // for each agent; they won't all be initialized with the same
    // genome.
    let mut population: Vec<Agent> = Vec::new();

    for i in 0..op.population_size{
        population.push(Agent { genome: vec![0.0; op.genome_size],
                                error: 0.0 });
        let agent = &mut population[i];
        for j in 0..op.genome_size {
            agent.genome[j] = rng.gen::<f64>() * GENE_INITIALIZATION_SD;
        }
    }

    for ip in 0..op.iterations {

        op.dataset.shuffle(&mut rng);
        
        let population_len = population.len();
        let (mut slice_left, mut slice_right) = population.split_at_mut(population_len / 2);

        // 
        
        let slice_left_len = slice_left.len();
        let (mut slice_left_left, mut slice_left_right) = slice_left.split_at_mut(slice_left_len / 2);

        let slice_right_len = slice_right.len();
        let (mut slice_right_left, mut slice_right_right) = slice_right.split_at_mut(slice_right_len / 2);

        // 
        
        let slice_left_left_len = slice_left_left.len();
        let (mut slice0, mut slice1) = slice_left_left.split_at_mut(slice_left_left_len / 2);

        let slice_left_right_len = slice_left_right.len();
        let (mut slice2, mut slice3) = slice_left_right.split_at_mut(slice_left_right_len / 2);

        let slice_right_left_len = slice_right_left.len();
        let (mut slice4, mut slice5) = slice_right_left.split_at_mut(slice_right_left_len / 2);

        let slice_right_right_len = slice_right_right.len();
        let (mut slice6, mut slice7) = slice_right_right.split_at_mut(slice_right_right_len / 2);
        

        rayon::scope(|s| {
            s.spawn(|s| {
                for mut agent in slice0 {
                    update_error(evaluate, &mut agent, &mut layers0, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice1 {
                    update_error(evaluate, &mut agent, &mut layers1, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice2 {
                    update_error(evaluate, &mut agent, &mut layers2, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice3 {
                    update_error(evaluate, &mut agent, &mut layers3, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice4 {
                    update_error(evaluate, &mut agent, &mut layers4, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice5 {
                    update_error(evaluate, &mut agent, &mut layers5, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice6 {
                    update_error(evaluate, &mut agent, &mut layers6, &op);
                }
            });
            s.spawn(|s| {
                for mut agent in slice7 {
                    update_error(evaluate, &mut agent, &mut layers7, &op);
                }
            });
        });

        population.sort_by(|a, b| compare_floats(a.error, b.error, 10));

        for i in 0..op.population_size/2 {
            // TODO: there might be a memory leak here? i hope not.
            population[op.population_size - i - 1].genome = population[i].genome.to_vec();
            population[op.population_size - i - 1].error = population[i].error;
        }


        if population[0].error < best_error {
            best_error = population[0].error;
            best_genome = population[0].genome.to_vec();
        }
           
        let mut total = 0.0;
        for datum in op.test_dataset.iter() {
            let input = &datum.0;
            let output = &datum.1;
            let est = evaluate(&population[0], input, &mut layers);
            println!("{:?}", est);
            total += vec_diff(&est, output);
        }
        total = total / (op.test_dataset.len() as f64);
        let best_agent = &population[0];
        println!("(min_e, test_e) = ({}, {})", total, population[0].error);

        for i in 0..op.population_size {
            mutate_genome(&mut population[i], rng);
        }
    }

    return Model { evaluate: evaluate,
                   agent: Agent { genome: best_genome,
                                  error: best_error },
                      op: op }
}
