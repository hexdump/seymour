import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

class Main {    
    public static void main(String args[]) {
	Population p = new Population();

	for (int i = 0; i < %{NUM_ROUNDS}; i++) {
	    p.breed();
	    p.sort();
	    p.select();
	}
}

class Randomizer {
    private Random rand = new Random();
    
    public double sampleNormal(double mean, double sd) {
	// TODO: customize standard deviation to allow for convergence
	// calling format: .nextGaussian(mean, stddev)
	return rand.nextGaussian();
    }

    public double[] mutateGenome(double[] genome) {
	double[] newGenome = new double[genome.length];
	for (int i = 0; i < genome.length; i++) {
	    newGenome[i] = sampleNormal(genome[i], 0.1);
	}
	return newGenome;
    }
}

class Individual implements Comparable<Individual> {
    public double[] genome;
    private Randomizer rand = new Randomizer();

    public Individual(double[] genome) {
	this.genome = genome;
	this.genome = rand.mutateGenome(this.genome);
    }

    public Individual() {
	this(new double[] {0, 0, 0});
    }
    
    public Individual asexuallyReproduce() {
	return new Individual(rand.mutateGenome(this.genome));
    }

    public Individual breedingCompatability(Individual other) {
	double diff = 0;
	for (int i = 0; i < this.genome.length; 
	for (double gene : this.genome) {
	    
	}
    }
    
    public Individual sexuallyReproduce(Individual other) {
	
    }

    public double evaluate() {
	return %{EVALUATION};
    }
    
    @Override
    public int compareTo(Individual other) {
	double f1 = this.evaluate();
	double f2 = other.evaluate();
	if (f1 > f2) {
	    return 1;
	}
	else if (f2 > f1) {
	    return -1;
	}
	return 0;
    }    
}

class Population {
    public ArrayList<Individual> individuals;
    
    public Population() {
	this.individuals = new ArrayList<Individual>();
	for (int i = 0; i < %{NUM_INDIVIDUALS}; i++) {
	    this.individuals.add(new Individual());
	}
    }

    public void sort() {
	Collections.sort(this.individuals);
    }
    
    public void breed() {
	int oldIndividualsLength = individuals.size();
	for (int i = 0; i < oldIndividualsLength; i++) {
	    this.individuals.add(individuals.get(i).asexuallyReproduce());
	    this.individuals.add(individuals.get(i).asexuallyReproduce());
	}	
    }

    public void select() {
	int numIndividuals = this.individuals.size();
	for (int i = numIndividuals - 1; i >= numIndividuals / 2; i--) {
	    this.individuals.remove(i);
	}
    }
}
