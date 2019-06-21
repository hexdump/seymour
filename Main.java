import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

class Main {    
    public static void main(String args[]) {
	Population p = new Population();

	p.breed();
	p.sort();
	p.select();

	p.breed();
	p.sort();
	p.select();

	p.breed();
	p.sort();
	p.select();


	p.breed();
	p.sort();
	p.select();


	p.breed();
	p.sort();
	p.select();

	p.sort();
	
	System.out.println("helo!");
	System.out.println(p.individuals.get(0).genome[0]);
	System.out.println(p.individuals.get(0).genome[1]);
	System.out.println(p.individuals.get(0).genome[2]);
	System.out.println(p.individuals.get(0).genome[0] + p.individuals.get(0).genome[1] + p.individuals.get(0).genome[2]);

	
    }
}

class Randomizer {
    private Random rand = new Random();
    
    public double sampleNormal(double mean, double sd) {
	return rand.nextGaussian();
	//	return rand.nextGaussian(mean, 0.1);
    }

    public double[] mutateGenome(double[] genome) {
	double[] newGenome = new double[genome.length];
	for (int i = 0; i < genome.length; i++) {
	    newGenome[i] = sampleNormal(genome[i], 0.1);
	}
	return newGenome;
    }
}

class Individual implements Comparable<Individual>{
    public double[] genome;
    private Randomizer rand = new Randomizer();

    public Individual(double[] genome) {
	this.genome = genome;
	this.genome = rand.mutateGenome(this.genome);
    }

    public Individual() {
	this(new double[] {0, 0, 0});
    }

    // public toString() {
    // }
    
    
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
	return Math.abs(4 - (genome[0] + genome[1] + genome[2]));
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
	this.individuals.add(new Individual());
	this.individuals.add(new Individual());
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

    //public void select() {
    //}
}
