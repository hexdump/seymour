use rand::prelude::*;
use pbr::ProgressBar;

use std::f64::consts::PI;
use std::cmp::Ordering;

// customize this
const NUM_LAYERS: usize                = 5;
const LAYER_SIZES: [usize; NUM_LAYERS] = [3, 90,  90,  10,  5];
const LAYER_SIZES_SUM: usize           =     90 + 90 + 10 + 5;
const SAMPLE_SIZE: usize               = 8;

//////////// don't touch! ////////////
const INPUT_SIZE: usize = LAYER_SIZES[0];
const OUTPUT_SIZE: usize = LAYER_SIZES[NUM_LAYERS - 1];
const GENOME_SIZE: usize = LAYER_SIZES_SUM;
//////////// don't touch! ////////////

// customize this
const POPULATION_SIZE: usize = 1000;
const data:[([f64; INPUT_SIZE], [f64; OUTPUT_SIZE]); SAMPLE_SIZE] = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]),
    ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]),
    ([0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]),
    ([0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]),
    ([1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]),
    ([1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]),
    ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]),
];

#[derive(Copy, Clone)]
struct Agent {
    genome: [f64; GENOME_SIZE],
    error: f64
}

fn rpd(x0: f64, x1: f64) -> f64{
    return (x0 - x1).abs();
}

fn vec_rpd(v0: [f64; OUTPUT_SIZE], v1: [f64; OUTPUT_SIZE]) -> f64 {
    
    let mut total: f64 = 0.0;
    for i in 0..OUTPUT_SIZE {
        total += rpd(v0[i], v1[i]);
    }
    return total / (OUTPUT_SIZE as f64);
}

fn error(agent: Agent, layers: &mut Vec<Vec<f64>>) -> f64 {
    let mut total: f64 = 0.0;
        for datum in data.iter() {
            let input = &datum.0;
            let output = &datum.1;
            
            let estimated = evaluate(agent, *input, layers);
            total += vec_rpd(estimated, *output);
        }
    return total;
    
}

fn manifold(gene: f64, inputs: Vec<f64>) -> f64 {
    let mut total = 0.0;
    for i in 0..inputs.len() {
        let x = inputs[i];
        total += (gene * x).sin() + (x % gene + x).cos();
    }

    return total / (inputs.len() as f64);
}


fn evaluate(agent: Agent, input: [f64; INPUT_SIZE], layers: &mut Vec<Vec<f64>>) -> [f64; OUTPUT_SIZE] {

    let mut g = 0;
    let mut last: Vec<f64> = Vec::new();
    
    // copy input into the first space in the network
    for i in 0..layers[0].len() {
        layers[0][i] = input[i];
        last.push(input[i]);
    }

    for i in 1..layers.len() {
        let current = &mut layers[i]; 
        let gene = agent.genome[g];
        
        for j in 0..current.len() {
            current[j] = manifold(gene, last.to_vec());
            last.insert(j, current[j]);
        }
        last.truncate(current.len());
        
        g += 1;
        
    }

    let output_vec = &layers[layers.len() - 1];
    let mut output: [f64; OUTPUT_SIZE] = [0.0; OUTPUT_SIZE];
    for i in 0..OUTPUT_SIZE {
        if output_vec[i] < 0.0 {
            output[i] = 0.0;
        }
        else {
            output[i] = 1.0;
        }
            
//        output[i] = output_vec[i].round().abs();
    }

    return output;
    
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

fn breed_genomes(mut a: [f64; GENOME_SIZE], mut b: [f64; GENOME_SIZE]) {
    for i in 0..GENOME_SIZE {
        if i % 2 == 0 {
            let temp = a[i];
            b[i] = a[i];
            a[i] = temp;
        }
    }      
}

use rand::distributions::{Normal, Distribution};

fn mutate_genome(mut agent: &mut Agent, mut rng: ThreadRng) {
    for i in 0..GENOME_SIZE {
        let r: f64 = rng.gen();
        if (r > 0.25) {
            let normal = Normal::new(agent.genome[i],  agent.error * 2.0);
            agent.genome[i] = normal.sample(&mut rng);
        }
    }
}

fn parse_label(label: usize) -> [f64; 10] {
    let mut vec = [0.0; 10];
    vec[label] = 1.0;
    return vec;
}

fn parse_pixels(pixels: Vec<usize>) -> [f64; 748] {
    let mut vec = [0.0; 748];
    for i in 0..748 {
        vec[i] = (pixels[i] as f64) / 255.0;
    }
//    vec[label.parse::<i32>().unwrap()] = 1.0;
    return vec;
}
//.parse::<i32>().unwrap(

use std::{
    fs::File,
    io::{prelude::*, BufReader},
    path::Path,
};

fn lines_from_file(filename: impl AsRef<Path>) -> Vec<String> {
    let file = File::open(filename).expect("no such file");
    let buf = BufReader::new(file);
    buf.lines()
        .map(|l| l.expect("Could not parse line"))
        .collect()
}

fn main() {
    let mut layers: Vec<Vec<f64>> = Vec::new();
    
    // intialize the space for data processing. currently, since
    // seymour is single-threaded, this space is shared by all
    // Agents.
    for layer_size in LAYER_SIZES.iter() {
        layers.push(vec![0.0; *layer_size]);
    }

    let mut rng: ThreadRng = thread_rng();

    // note that this will initialize these arrays INDEPENDENTLY
    // for each agent; they won't all be initialized with the same
    // genome.
    let mut population: Vec<Agent> = Vec::new();


    // this is the most intensive part, and will hopefully be
    // multithreaded soon.
    println!("generating genomes...");
    let mut pb = ProgressBar::new((POPULATION_SIZE) as u64);
    for i in 0..POPULATION_SIZE{
        population.push(Agent { genome: [0.0; GENOME_SIZE], 
                                error: 0.0 });
        let agent = &mut population[i];
        for j in 0..GENOME_SIZE {
            agent.genome[j] = rng.gen::<f64>();
        }
        pb.inc();
    }
    pb.finish();

    for ip in 0..POPULATION_SIZE {

        println!("evaluating agents...");
        let mut pb = ProgressBar::new((POPULATION_SIZE) as u64);
        for i in 0..POPULATION_SIZE {
            pb.inc();
            population[i].error = 100.0 * error(population[i], &mut layers) / (POPULATION_SIZE as f64);
        }
        pb.finish();

        population.sort_by(|a, b| compare_floats(a.error, b.error, 10));

        for i in 0..POPULATION_SIZE/2 {
            population[POPULATION_SIZE - i - 1] = population[i];
        }

        for i in 0..POPULATION_SIZE/2 {
            breed_genomes(population[2 * i].genome,
                          population[2 * i + 1].genome);
        }

        println!("min_e = {}", population[0].error);
        for datum in data.iter() {
            println!("example = {:?}", evaluate(population[0], datum.0, &mut layers));
        }
        
        for i in 0..POPULATION_SIZE {
            mutate_genome(&mut population[i], rng);
        }
    }
}
