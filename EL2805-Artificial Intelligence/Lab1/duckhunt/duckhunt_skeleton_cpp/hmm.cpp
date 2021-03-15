#include "hmm.hpp"


// constructors and destructors
HMM::HMM(std::vector<std::vector<double> > _A, std::vector<std::vector<double> > _B,
    std::vector<double> _pi)
{
  A = _A;
  B = _B;
  pi = _pi;
  N = _A.size();
  K = _B[0].size();
}

HMM::HMM(unsigned int nStates, unsigned int nObservations)
{
  N = nStates;
  K = nObservations;

  A.resize(N);
  for(unsigned int n = 0; n < N; n++) A[n].resize(N);
  B.resize(N);
  for(unsigned int n = 0; n < N; n++) B[n].resize(K);
  pi.resize(N);

  // For creating similar random numbers to represent the probabilities
  std::random_device random;
  std::mt19937 generator(rand());
  std::normal_distribution<double> gauss(10,1);

  // For making sure that each row adds to 1.0
  double rowSum = 0.0;
  double normalizer = 0.0;

  // Random initialization of pi
  for(unsigned int i = 0; i < N; i++)
  {
    pi[i] = gauss(generator);
    rowSum += pi[i];
  }
  normalizer = rowSum;
  rowSum = 0.0;
  for(unsigned int i = 0; i < N - 1; i++)
  {
    pi[i] /= (normalizer + DELTA);
    rowSum += pi[i];
  }
  pi[N - 1] = 1.0 - rowSum;

  // Random initialization of A
  for(unsigned int i = 0; i < N; i++)
  {
    rowSum = 0.0;
    for(unsigned int j = 0; j < N; j++)
    {
      A[i][j] = gauss(generator);
      rowSum += A[i][j];
    }
    normalizer = rowSum;
    rowSum = 0.0;
    for(unsigned int j = 0; j < N - 1; j++)
    {
      A[i][j] /= (normalizer + DELTA);
      rowSum += A[i][j];
    }
    A[i][N - 1] = 1.0 - rowSum;
  }

  // Random initialization of B
  for(unsigned int i = 0; i < N; i++)
  {
    rowSum = 0.0;
    for(unsigned int k = 0; k < K; k++)
    {
      B[i][k] = gauss(generator);
      rowSum += B[i][K];
    }
    normalizer = rowSum;
    rowSum = 0.0;
    for(unsigned int k = 0; k < K - 1; k++)
    {
      B[i][k] /= (normalizer + DELTA);
      rowSum += B[i][k];
    }
    B[i][K - 1] = 1.0 - rowSum;
  }
}

HMM::HMM(const HMM &model)
{
  A = model.getA();
  B = model.getB();
  pi = model.getPi();
  N = model.getNStates();
  K = model.getNObservations();
}



// important functions of the HMM

void HMM::estimateModel(std::vector<int> O)   // estimates(trains) a model based on given observation sequence
{
    
    int T = O.size();
    int M = K;
    
    // init alfa beta gama
        // Matrix alfa(T,N,1);
        std::vector<std::vector<double> > alfa;
        alfa.resize(T);
        for (unsigned int i = 0; i < T; i++) alfa[i].resize(N);

        // Matrix beta(T,N,1);
        std::vector<std::vector<double> > beta;
        beta.resize(T);
        for (unsigned int i = 0; i < T; i++) beta[i].resize(N);

        //Matrix c(1,T,1);
        std::vector<double> c;
        c.resize(T);

        // vector <Matrix> gama_1(T-1, Matrix(N, N,0));
        std::vector<std::vector<std::vector<double> > > gama_1;
        gama_1.resize(T-1);
        for (unsigned int i = 0; i < T-1; i++)
        {
            gama_1[i].resize(N);
            for (unsigned int  j = 0; j < N; j++) gama_1[i][j].resize(N);
            
        }

        // Matrix gama_2( T, N, 1 );
        std::vector<std::vector<double> > gama_2;
        gama_2.resize(T);
        for (unsigned int i = 0; i < T; i++) gama_2[i].resize(N);



    double den = 0;
    double num = 0;
    double log_prob;
    double prevSequenceProb = -10e-50;
    int iters=0;
    
    computeGammaAndDiGamma(O);
    double sequenceProb = getObsSeqProb(O);
    do
    {
      // Estimation of pi
      for(unsigned int i = 0; i < N; i++)
        pi[i] = gamma[0][i];

      // Estimation of A
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          num = 0.0;
          den = 0.0;
          for(unsigned int t = 0; t < T - 1; t++)
          {
            num += diGamma[t][i][j];
            den += gamma[t][i];
          }
          A[i][j] = num / (den + DELTA);
        }

      // Estimation of B
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int k = 0; k < K; k++)
        {
          num = 0.0;
          den = 0.0;
          for(unsigned int t = 0; t < T; t++)
          {
            if(O[t] == k)
              num += gamma[t][i];
            den += gamma[t][i];
          }
          B[i][k] = num / (den + DELTA);
        }
    
    prevSequenceProb = sequenceProb;
    computeGammaAndDiGamma(O);
    sequenceProb = getObsSeqProb(O);
    iters++;

    }while((iters <  35) && (sequenceProb > prevSequenceProb ||
      std::isnan(prevSequenceProb) || !std::isnan(sequenceProb)));
}


std::vector<double> HMM::estimateDistribution(std::vector<int> observations)
// estimates distribution of state given the observations
{
  std::vector<double> toReturn(N);
  double temp;
      computeGammaAndDiGamma(observations);

      // last row of gamma * A
      for(unsigned int k = 0; k < N; k++)
      {
        temp = 0.0;
        for(unsigned int j = 0; j < N; j++)
        {
          temp += gamma.back()[j] * A[j][k];
        }
      toReturn[k] = temp;
      }
  
    // alphaPass(observations);
    // return alpha[observations.size()-1];
    return toReturn;
}

std::vector<double> HMM::probabilityDistributionOfNextObs(std::vector<double> stateDistribution)
// PI * A * B
{
    std::vector<double> probabilityOfMoves(K);

    std::vector<std::vector<double>> tempC;
    tempC.resize(N);
    double temp;

  
    for(unsigned int i = 0; i < N; i++)
    {
      tempC[i].resize(K,0.0);
      for(unsigned int k = 0; k < K; k++)
      {
        temp = 0.0;
        for(unsigned int j = 0; j < N; j++)
        {
          temp += A[j][i]*B[i][k];
        }
      tempC[i][k] = temp;
      }
    }

    for (unsigned int k = 0; k < K; k++)
    {
      temp = 0.0;
      for (unsigned int j= 0; j < N; j++)
      {
        temp += stateDistribution[j] * tempC[j][k];
      }
      probabilityOfMoves[k] = temp;
    }
    

    return probabilityOfMoves;
}

unsigned int HMM::getNextMostProbObs(std::vector<int> sequence, double &maxProb)
/* Calculates the most probable observation in the next time step along with
   its probability given the sequence of observations until now.
   It only works with a logarithmic scaling to prevent underflow problems. */
{
    std::vector<double> nextProbs(K);
    unsigned int T = sequence.size();

    computeGammaAndDiGamma(sequence);

    for(unsigned int k = 0; k < K; k++)
    {
        nextProbs[k] = 0.0;
        for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
            nextProbs[k] += gamma[T - 1][i] * A[i][j] * B[j][k];
    }

    // normalize probabilities
    nextProbs = normalize(nextProbs);

    maxProb = nextProbs[0];
    int maxIndex = 0;
    for(unsigned int k = 1; k < K; k++)
        if(nextProbs[k] > maxProb)
        {
        maxProb = nextProbs[k];
        maxIndex = k;
        }

    return maxIndex;
}



std::vector<double> HMM::getNextObsProbDist(std::vector<int> sequence)
	/* Calculates the probability distribution of the next observation given the
	observations so far.
	Currently only with constant scaling. */
{
	std::vector<double> nextObsProbs(K);
	unsigned int T = sequence.size();

	computeGammaAndDiGamma(sequence);

	for (unsigned int k = 0; k < K; k++)
	{
		nextObsProbs[k] = 0.0;
		for (unsigned int i = 0; i < N; i++)
			for (unsigned int j = 0; j < N; j++)
				nextObsProbs[k] += gamma.back()[i] * A[i][j] * B[j][k];
	}

	return nextObsProbs;
}




double HMM::getObsSeqProb(std::vector<int> O)
/* Returns the probability of a sequence of observations, i.e summ of last row of alfa, 
  but since we normalized it, it corresponds to 1/c(T-1) */
{

    unsigned int T = O.size();
    double probability;
    alpha.resize(T);
    for(unsigned int t = 0; t < T; t++) alpha[t].resize(N);
      for(unsigned int i = 0; i < N; i++)
      {
        alpha[0][i] = pi[i] * B[i][O[0]];
      }

      for(unsigned int t = 1; t < T; t++)
      {
        for(unsigned int i = 0; i < N; i++)
        {
          alpha[t][i] = 0.0;
          for(unsigned int j = 0; j < N; j++) alpha[t][i] += A[j][i] * alpha[t - 1][j];
          alpha[t][i] *= B[i][O[t]];
        }
      }

      for (unsigned int i = 0; i < N; i++)
      {
        probability += alpha[T-1][i];
      }
      return probability;
}


void HMM::alphaPass(std::vector<int> sequence)
/* Compute the forward algorithm or alpha pass */
{
  unsigned int T = sequence.size();
  alpha.resize(T);
  for(unsigned int t = 0; t < T; t++) alpha[t].resize(N);
    /* Computes constant scaled alpha using Stamp algorithm */

      c.resize(T);

      // Calculate the initial alpha
      c[0] = 0.0;
      for(unsigned int i = 0; i < N; i++)
      {
        alpha[0][i] = pi[i] * B[i][sequence[0]];
        c[0] += alpha[0][i];
      }

      // Scale the initial alpha
      c[0] = 1 / (c[0] + DELTA);
      for(unsigned int i = 0; i < N; i++)
        alpha[0][i] = c[0] * alpha[0][i];

      // Calculate alpha for each time step
      for(unsigned int t = 1; t < T; t++)
      {
        c[t] = 0.0;
        for(unsigned int i = 0; i < N; i++)
        {
          alpha[t][i] = 0.0;
          for(unsigned int j = 0; j < N; j++)
            alpha[t][i] += A[j][i] * alpha[t - 1][j];
          alpha[t][i] *= B[i][sequence[t]];
          c[t] += alpha[t][i];
        }

        // Scale alpha for each time step
        c[t] = 1 / (c[t] + DELTA);
        for(unsigned int i = 0; i < N; i++)
          alpha[t][i] *= c[t];
      }

}

void HMM::betaPass(std::vector<int> sequence)
/* Compute the backward algorithm or beta pass */
{
  int T = sequence.size();

  beta.resize(T);
  for(unsigned int t = 0; t < T; t++) beta[t].resize(N);

      // Set the initial (last) beta  with the scaling of last c
    for(unsigned int i = 0; i < N; i++)
    beta[T - 1][i] = c[T - 1];

      // Calculate beta for each time step
    for(int t = T - 2; t >= 0; t--)
        for(unsigned int i = 0; i < N; i++)
        {
          beta[t][i] = 0;
          for(unsigned int j = 0; j < N; j++)
            beta[t][i] += A[i][j] * B[j][sequence[t + 1]] * beta[t + 1][j];

          // Scale beta for each time step (with the same scaling factor than alpha)
          beta[t][i] *= c[t];
        }
}

void HMM::computeGammaAndDiGamma(std::vector<int> sequence)
/* Compute the gamma and di-gamma matrices */
{
  unsigned int T = sequence.size();
  double normalizer = 0.0;

  alphaPass(sequence);  // Compute alpha
  betaPass(sequence);   // Compute beta

  diGamma.resize(T - 1);
  for(unsigned int t = 0; t < T - 1; t++)
  {
    diGamma[t].resize(N);
    for(unsigned int i = 0; i < N; i++) diGamma[t][i].resize(N);
  }
  gamma.resize(T);
  for(unsigned int t = 0; t < T; t++) gamma[t].resize(N);

    for(unsigned int t = 0; t < T - 1; t++)
    {
      normalizer = 0.0;
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][sequence[t + 1]]
            * beta[t + 1][j];
          normalizer += diGamma[t][i][j];
        }

      for(unsigned int i = 0; i < N; i++)
      {
        gamma[t][i] = 0.0;
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] /= (normalizer + DELTA);
          gamma[t][i] += diGamma[t][i][j];
        }
      }
    }

  /* Special computation for last time step (gamma) */
  normalizer = 0.0;
  for(unsigned int i = 0; i < N; i++)
    normalizer += alpha[T - 1][i];
  for(unsigned int i = 0; i < N; i++)
    gamma[T - 1][i] = alpha[T - 1][i] / (normalizer + DELTA);

}




std::vector<double> normalize(std::vector<double> v){
  double normalizer = 0.0;
  double rowSum = 0.0;

  for(unsigned int i = 0; i < v.size(); i++)
    normalizer += v[i];

  if(normalizer == 0.0)
    return v;

  for(unsigned int i = 0; i < v.size() - 1; i++)
  {
    v[i] /= (normalizer);
    rowSum += v[i];
  }
  v[v.size() - 1] = 1.0 - rowSum;
    return v;

}