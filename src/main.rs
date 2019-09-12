extern crate pbr;
extern crate rand;

use rand::prelude::*;
use pbr::ProgressBar;

use std::f64::consts::PI;
use std::cmp::Ordering;
 
//const LAYERS: [i32; 4] = [784, 300, 100, 10];
//const GENOME_SIZE: usize = 300 + 100 + 10;
const LAYERS_NUM: usize = 3;
const LAYER_SIZES: [usize; LAYERS_NUM] = [748, 300, 10];

const INPUT_SIZE: usize = LAYER_SIZES[0];
const OUTPUT_SIZE: usize = LAYER_SIZES[LAYERS_NUM - 1];

const LAYER_SIZES_SUM: usize = 300 + 10;
const GENOME_SIZE: usize = 300 + 10;
const POPULATION_SIZE: usize = 100;

#[derive(Copy, Clone)]
struct Agent {
    genome: [f64; GENOME_SIZE],
    error: f64
}

fn rpd(x0: f64, x1: f64) -> f64{
    return (x0 - x1).abs();
//    if (x0 == x1) {
//        return 0.0;
//    }
//    else {
//        return ((x0 - x1) / (x0.abs() - x1.abs())).abs();
//    }
}

fn vec_rpd(v0: [f64; OUTPUT_SIZE], v1: [f64; OUTPUT_SIZE]) -> f64 {
    
    let mut total: f64 = 0.0;
    for i in 0..OUTPUT_SIZE {
        total += rpd(v0[i], v1[i]);
    }
    return total / (OUTPUT_SIZE as f64);
}

fn error(agent: Agent, layers: &mut Vec<Vec<f64>>, data: &Vec<([f64; INPUT_SIZE], [f64; OUTPUT_SIZE])>) -> f64 {
//    let data = vec![([0.0, 0.0], [0.0]),
//                    ([0.5, 0.5], [0.5]),
//                    ([0.5, 0.5], [0.5]),
//                    ([1.0, 1.0], [1.0])];

    let mut total: f64 = 0.0;
    unsafe {
        for datum in &data[..10] {
            let input = &datum.0;
            let output = &datum.1;
            
            let estimated = evaluate(agent, *input, layers);
            total += vec_rpd(estimated, *output);
        }
    }
    return total;
    
}

fn manifold(gene: f64, inputs: Vec<f64>) -> f64 {
    
    let mut total = 0.0;
    for i in 0..inputs.len() {
        let x = inputs[i] * gene + gene;
        total += (x*8.0).cos();
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
        let mut gene = agent.genome[i];
        
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
        output[i] = output_vec[i];
    }
    return output
    
}

fn compare_floats(a: f64, b: f64, decimal_places: u8) -> Ordering {
    let factor = 10.0f64.powi(decimal_places as i32);
    let a = (a * factor).trunc();
    let b = (b * factor).trunc();
    if (a > b) {
        return Ordering::Greater;
    }
    else if (a < b) {
        return Ordering::Less;
    }
    else {
        return Ordering::Equal;;
    }
}

fn breed_genomes(mut a: [f64; GENOME_SIZE], mut b: [f64; GENOME_SIZE], mut rng: ThreadRng) {
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
        let normal = Normal::new(agent.genome[i], 0.02 * agent.error * agent.error);
        agent.genome[i] = normal.sample(&mut rng);
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
    //    let file = File::open("foo.txt")?;

    let mut data: Vec<([f64; 748], [f64; 10])> = Vec::new();

    
//    let mut output = File::open("data.csv")?;
//    write!(output, "Rust\n💖\nFun")?;

//    let input = File::open(path)?;
//    let buffered = BufReader::new(input);
    

//    let mut pb = ProgressBar::new((POPULATION_SIZE) as u64);
//    pb.inc();
//    pb.finish();

    for line in lines_from_file("data.csv") {
        let items: Vec<&str> = line.split(",").collect();
        if let Some((label, pixels)) = items.split_first() {
            let label_int = label.parse::<usize>().unwrap();
            let pixels_ints: Vec<usize> = pixels.iter().map(|x| x.parse::<usize>().unwrap()).collect();
            unsafe {
                data.push(((parse_pixels(pixels_ints), parse_label(label_int))));
            }
        }
//        data.pop();
    }

    let mut layers: Vec<Vec<f64>> = Vec::new();
    
    // intialize the space for data processing. currently, since
    // seymour is single-threaded, this space is shared by all
    // Agents.
    for layer_size in LAYER_SIZES.iter() {
        layers.push(vec![0.0; *layer_size]);
    }

    let mut rng: ThreadRng = thread_rng();
//    let mut rng = rand_pcg::Pcg32::seed_from_u64(123);

    
    // note that this will initialize these arrays INDEPENDENTLY
    // for each agent; they won't all be initialized with the same
    // genome.
    let mut population: Vec<Agent> = Vec::new();

    println!("generating genomes...");
    let mut pb = ProgressBar::new((POPULATION_SIZE) as u64);
    for i in 0..POPULATION_SIZE {
        population.push(Agent { genome: [0.0; GENOME_SIZE], 
                                error: 0.0 });
        let agent = &mut population[i];
        for j in 0..GENOME_SIZE {
            agent.genome[j] = rng.gen::<f64>();
        }
        pb.inc();
    }
    pb.finish();
    
    for i in 0..1000 {
        println!("evaluating agents...");
        let mut pb = ProgressBar::new((POPULATION_SIZE) as u64);
        for i in 0..POPULATION_SIZE {
            pb.inc();
            population[i].error = error(population[i], &mut layers, &data);
        }
        pb.finish();

        population.sort_by(|a, b| compare_floats(a.error, b.error, 10));
        println!("{}", error(population[0], &mut layers, &data));
//        println!("{}", error(population[0], &mut layers));


        for i in 0..POPULATION_SIZE/2 {
            population[POPULATION_SIZE - i - 1] = population[i];
        }
        
//       println!("{:?}", population[0].genome);

        data.shuffle(&mut rng);
        
        for i in 0..POPULATION_SIZE/2 {
            breed_genomes(population[2 * i].genome,
                          population[2 * i + 1].genome,
                          rng);
        }

        for i in 0..POPULATION_SIZE {
            mutate_genome(&mut population[i], rng);
        }
    }
}
