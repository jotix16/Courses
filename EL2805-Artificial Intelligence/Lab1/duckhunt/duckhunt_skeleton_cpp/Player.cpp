#include "Player.hpp"
#include <cstdlib>
#include <iostream>
#include <algorithm>

#define MAXTURNS 99

namespace ducks
{

constexpr double GUESS_THRESHOLD = 0.5; 
constexpr double SHOOT_THRESHOLD = 0.8; 
constexpr double VOTING_THRESHOLD = 0.6; 
constexpr double BLACK_TRHESHOLD = 1.0e-19; 
constexpr unsigned int nStates = 3; 

Player::Player()
{
   successShot = 0; // Number of successful shots (debbugging)
   totalShots = 0;  // Number of total shots (debbuging)
   successGuess = 0; // Number of successful guesses (debbuging)
   totalGuesses = 0;  // Number of total guesses (debbuging)
   birdModels.resize(COUNT_SPECIES); // Each species will have its own array of models
   revealedSpecies = std::vector<bool>(COUNT_SPECIES, false); // True for revealed species, false for unrevealed (only needed for guesses[smart way, to get all kind of species guessed])
}

Action Player::shoot(const GameState &pState, const Deadline &pDue)
{

  unsigned int nBirds = pState.getNumBirds();   // Number of birds

  std::vector<int> observations; // Movements of each bird

  int birdSpecies = SPECIES_UNKNOWN;  // Species of the observed bird
  double confidence = 0.0;            // Confidence on the bird species
  double blackprob = 0.0;             // Probability of the bird being a BLACK STORK

  int targetBird = -1;
  int targetMove = -1;
  int targetSpecies = -1;
  double targetProbability = 0.0;

  if(nBirds == 0) return cDontShoot;
  if(pState.getRound()== 0 ) return cDontShoot;
  if(pState.getBird(0).getSeqLength() < 70) return cDontShoot;


  for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
  {
    Bird bird = pState.getBird(birdIndex);
    if(!bird.isAlive()) continue;

    
    observations = getObservations(bird); // We get the movement of the bird
    birdSpecies = getSpeciesGuess(observations, confidence, blackprob);

    if( (birdSpecies == SPECIES_UNKNOWN) || (birdSpecies == SPECIES_BLACK_STORK) || (confidence < 0.8) ) 
    {
      continue; // We only try to shoot birds we know are not BLACK STORKS
    }

    // Create auxiliary vectors for statistics on the movement selection
    std::vector<double> moveProbabilites(COUNT_MOVE);
    std::vector<double> stateDistribution(5, 0.0);

    // We train a model with the current observations
    HMM model(3, COUNT_MOVE);
    model.estimateModel(observations);
    stateDistribution = model.estimateDistribution(observations); //get state distro
    stateDistribution = normalize(stateDistribution);
    
    moveProbabilites = model.probabilityDistributionOfNextObs(stateDistribution); // find emission distro with this state distro
    moveProbabilites = normalize(moveProbabilites);
    
    double xProbability = 0.0;
    int xMove =0;

    // We get the mean of the probabilies of each move given by the models
    // Then we normalize for all the possible moves
    for(unsigned int i = 0; i < COUNT_MOVE; i++)
    {
        if(moveProbabilites[i] > xProbability )
        {
            xProbability = moveProbabilites[i];
            xMove = i;
        }   
        
    }
    
    // shoot only the bird whose next move we know with highest probability compared to the other birds
    if(targetProbability < xProbability)
    {
        targetProbability = xProbability;
        targetMove = xMove;
        targetBird = birdIndex;
    }
  }

  if (targetProbability < 0.6)
  {
    return cDontShoot;
  }
  
  totalShots++;
  //std::cerr<<" targetpr:"<< targetProbability <<" blackpr: "<< blackprob << std::endl;
  return Action(targetBird, EMovement(targetMove));
  

}

std::vector<ESpecies> Player::guess(const GameState &pState, const Deadline &pDue)
{
    int nBirds = pState.getNumBirds();
    std::vector<ESpecies> lGuesses(nBirds, SPECIES_UNKNOWN);

    /* On the first round we are forced to guess at random in order to get
       information about the birds species */
  	if (pState.getRound() == 0)
    {
		  for (unsigned int i = 0; i < nBirds; i++)
			   lGuesses[i] = ESpecies(SPECIES_PIGEON);
    }
    /* After the first round we can guess normally */
    else
	  {
       Bird bird;
       int guess;
       std::vector<int> observations;
       double confidence = 0.0;
       double blackprob = 0.0;

      // For all the birds...
      for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
      {
  			bird = pState.getBird(birdIndex);
  			observations = getObservations(bird);
  			guess = getSpeciesGuess(observations, confidence, blackprob);
        
  			//Guess the species if the confidence of the guess is above the threshold
  			if(ESpecies(guess) != SPECIES_UNKNOWN && confidence > GUESS_THRESHOLD)
  				{
            lGuesses[birdIndex] = ESpecies(guess);
          }
  			else if(blackprob > BLACK_TRHESHOLD)
  				{
            lGuesses[birdIndex] = ESpecies(SPECIES_BLACK_STORK);
          }
        else
  			{
  				// Vote a random species from those that have not been revealed yet
  				unsigned int randIdx = std::rand() % COUNT_SPECIES; // random starting index
  				unsigned int stop = 0; // iteration counter
  				for (unsigned int i = randIdx; stop < COUNT_SPECIES; i = (i + 1) % COUNT_SPECIES)
  				{
  					if (!revealedSpecies[i] && (i != SPECIES_BLACK_STORK || getKnownSpecies() == COUNT_SPECIES - 1))
  					//if species has not been revealed before you can guess it
  					{
  						lGuesses[birdIndex] = ESpecies(i);
  						break;
  					}
  					stop++;
  				}
  			}
  		}
	  }
    guesses = lGuesses;

    return lGuesses;
}

void Player::hit(const GameState &pState, int pBird, const Deadline &pDue)
{
    /*
     * If you hit the bird you are trying to shoot, you will be notified through this function.
     */

  	 std::cerr << "HIT BIRD!!!" << std::endl;
  	 successShot++;
}

void Player::reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue)
{
    /*
     * If you made any guesses, you will find out the true species of those birds in this function.
     */

     int nBirds = pSpecies.size();

     /* To know if we are guessing right... */
     for(unsigned int i = 0; i < nBirds; i++)
     {
       if(pSpecies[i] == guesses[i])
         successGuess++;
       totalGuesses++;
     }

     /* Train the bird models knowing it's species, then store it for better guessing */
     std::vector<int> observations;
     for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
     {
		    if (pSpecies[birdIndex] == -1)
				continue;
    		HMM model(nStates, COUNT_MOVE);
    		observations = getObservations(pState.getBird(birdIndex));
    		model.estimateModel(observations);
    		birdModels[pSpecies[birdIndex]].push_back(model);
    		revealedSpecies[pSpecies[birdIndex]] = true;

     }


     // The below cerrs are just for debbuging
     std::cerr << "\nRound #" << pState.getRound();

     std::cerr << "\nNum of guesses " << totalGuesses;
     std::cerr << "\nNum of shots: " << totalShots;

     if(totalGuesses == 0)
       std::cerr << "\nGuess ratio: 0";
     else
       std::cerr << "\n\nGuess ratio: " << ((double) successGuess)/((double) totalGuesses);
     if(totalShots == 0)
       std::cerr << "\nShoot ratio: 0";
     else
       std::cerr << "\nShoot ratio: " << ((double) successShot)/((double) totalShots);
     std::cerr << "\n\n";

     successGuess = 0;
     totalGuesses = 0;
     successShot = 0;
     totalShots = 0;
}

std::vector<int> Player::getObservations(Bird b)
/* Returns the sequence of movements of a bird */
{
  unsigned int T = b.getSeqLength();
  std::vector<int> sequence(T);

  for(unsigned int t = 0; t < T; t++)
    if(b.wasAlive(t))
      sequence[t] = b.getObservation(t);

  return sequence;
}

int Player::getSpeciesGuess(std::vector<int> sequence, double &confidence, double &blackprob)
/* Returns a guess of the spice of a bird */
{
	int guess = SPECIES_UNKNOWN;
	double probability;
	double bestProbability = 0.0;
	std::vector<std::vector<double> > probabilities(COUNT_SPECIES, std::vector<double> (0)); //sum of probabilities for each model
	std::vector<double> speciesProbabilities(COUNT_SPECIES, 0.0);

	// For each species
	for(unsigned int i = 0; i < COUNT_SPECIES; i++)
	{
		// For each model trained for this species
		for(unsigned int j = 0; j < birdModels[i].size(); j++)
		{
			// get the observation probability given the model parameters, P(O|lamda)
			probability = birdModels[i][j].getObsSeqProb(sequence);





			probabilities[i].push_back(probability);
		}
	}

  // We take the mean of the probabilities given by the models
	for(unsigned int i = 0; i < COUNT_SPECIES; i++)
		speciesProbabilities[i] = getMean(probabilities[i]);

  // We normalize the probabilities (P(A|B) = alpha * P(A,B))
	speciesProbabilities = normalize(speciesProbabilities);

	bestProbability = 0.0;
	// Get the max probability from the distribution
	for(int i = 0; i < COUNT_SPECIES; i++)
	{
		probability = speciesProbabilities[i];
		if(probability > bestProbability)
		{
			guess = i;
			bestProbability = probability;
		}
	}

  // We make sure to put confidence to 0 if we don't know the species
	if(guess == SPECIES_UNKNOWN)
		confidence = 0.0;
  else
		confidence = speciesProbabilities[guess];

  blackprob = speciesProbabilities[SPECIES_BLACK_STORK];

	return ESpecies(guess);
}

int Player::getKnownSpecies()
{
	int sum = 0;
	for (int i = 0; i < revealedSpecies.size(); i++)
		if (revealedSpecies[i] == true)
			sum++;
	return sum;
}


// Additional useful functions
double getMean(std::vector<double> v)
{
  double mean = 0.0;
  unsigned int size = v.size();

  if(size == 0)
    return 0.0;

  for(unsigned int i = 0; i < size; i++)
    mean += v[i];

  mean /= size;

  return mean;
}


} /*namespace ducks*/