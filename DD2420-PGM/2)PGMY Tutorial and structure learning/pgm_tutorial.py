'''

This code template belongs to
"
PGM-TUTORIAL: EVALUATION OF THE 
PGMPY MODULE FOR PYTHON ON BAYESIAN PGMS
"

Created: Summer 2017
@author: miker@kth.se

Refer to https://github.com/pgmpy/pgmpy
for the installation of the pgmpy module

See http://pgmpy.org/models.html#module-pgmpy.models.BayesianModel
for examples on fitting data

See http://pgmpy.org/inference.html
for examples on inference

'''

def separator():
    input('Enter to continue')
    print('-'*70, '\n')
    
# Generally used stuff from pgmpy and others:
import math
import random
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score
from scipy.stats import entropy
# Specific imports for the tutorial
import pgm_tutorial_data
from pgm_tutorial_data import ratio, get_random_partition

RAW_DATA = pgm_tutorial_data.RAW_DATA
FEATURES = [f for f in RAW_DATA]

'''
# Task 1 ------------ Setting up and fitting a naive Bayes PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                    ('delay', 'gender'),
                    ('delay', 'avg_mat'),
                    ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('Task 1.1')
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

print('')
print(model.cpds[0])


# Task 1.2
print('')
print('')
print('Task 1.2')

a = model.cpds[3].values[2]
print("Fitted marginal for delay(>=2):",a)
print("Frequency of the data for delay(>=2):",ratio(data, lambda t: t['delay']=='>=2'))

# Task 1.3
print('')
print('')
print('Task 1.3')

STATE_NAMES = model.cpds[2].state_names
n= len(STATE_NAMES['avg_mat'])
m =len(STATE_NAMES['delay'])

res = [ratio(data,lambda t: t['avg_mat']==i,lambda t: t['delay']==j) for i in STATE_NAMES['avg_mat'] for j in STATE_NAMES['delay'] ]
res = np.array(res).reshape(n,m)
print(res) # prints the frequencies from the data
print('')
print(model.cpds[2]) # prints cpds from fitted model
separator()
# End of Task 1
'''

'''
# Task 2 ------------ Probability queries (inference)

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('delay', 'age'),
                    ('delay', 'gender'),
                    ('delay', 'avg_mat'),
                    ('delay', 'avg_cs')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

ve = VariableElimination(model)

# Task 2.1
print('')
print('')
print('Task 2.1')
q = ve.query(variables = ['delay'], evidence = {'age': '<=20'},show_progress=False)
print(q)

# Task 2.2
print('')
print('')
print('Task 2.2')
m =len(STATE_NAMES['age'])
print(STATE_NAMES['age'])
q = [ve.query(variables = ['delay'], evidence = {'age': i},show_progress=False).values[0] for i in STATE_NAMES['age']]
print(q)

# Task 2.3
print('')
print('')
print('Task 2.3')

n= len(STATE_NAMES['age'])
m =len(STATE_NAMES['delay'])

res = [ratio(data,lambda t: t['delay']=='0',lambda t: t['age']==j) for j in STATE_NAMES['age'] ]
print(res) # prints the frequencies from the data

# Task 2.4
print('')
print('')
print('Task 2.4')
q=ve.map_query(variables=['age','delay'],  show_progress=False)
print(q)
separator()

# End of Task 2
'''


'''
 # Task 3 ------------ Reversed PGM

data = pd.DataFrame(data=RAW_DATA)
model = BayesianModel([('age', 'delay'),
                       ('gender', 'delay'),
                       ('avg_mat', 'delay'),
                       ('avg_cs', 'delay')])
model.fit(data) # Uses the default ML-estimation

STATE_NAMES = model.cpds[0].state_names
print('State names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

# Task 3.2
print('')
print('')
print('Task 3.2')
model_delay = model.cpds[3].values
print("Shape of tables for delay: ",model_delay.shape)
print("Number of entries for delay: ",model_delay.size)

# Task 3.3
print('')
print('')
print('Task 3.3')
# print(model.cpds[3])  # uncomment to see all CPDs of delay

# Task 3.4
print('')
print('')
print('Task 3.4')
ve = VariableElimination(model)
q = ve.query(variables = ['delay'],evidence = {'age': '>23','gender': '1','avg_cs':'<2','avg_mat':'<2'},show_progress=False)
print(q)




# Task 3.5
print('')
print('')
print('Task 3.5')
delay_names= model.cpds[3].state_names['delay']
q = ve.query(variables = ['delay'],show_progress=False) #pgm based marginal
res = [ratio(data, lambda t: t['delay']==j) for j in delay_names ]#marginal from relative frequencies

print(delay_names)
print("Marginal for delay (PGM)")
print(q)
print("Marginal for delay (freq)")
print(res)
relative_error = [abs(i-j)/i for (i,j) in zip(res,q.values)]
print('')
print("names: ",delay_names)
print("relative_errors: ",relative_error)
separator()

# End of Task 3
'''



''' 
# Task 4 ------------ Comparing accuracy of PGM models

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model2.cpds[3].state_names   ##CHANGE MIKEL
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES

VARIABLES = list(S.keys())

def random_query(variables, target):
    # Helper function, generates random evidence query
    n = random.randrange(1, len(variables)+1) #number of times to sample variables list(how many evidences)
    evidence = {v: random.randrange(len(S[v])) for v in random.sample(variables, n)}
    if target in evidence: del evidence[target]
    return (target, evidence)

queries = []
for target in ['delay']:
    variables = [v for v in VARIABLES if v != target]
    queries.extend([random_query(variables, target) for i in range(200)])

divs = []
# divs will be filled with lists on the form
# [query, distr. in data, distr. model 1, div. model 1, distr. model 2, div. model 2]
for v, e in queries:
    try:
        # Relative frequencies, compared below
        # s: value delay can take, w:               given  all targets corresponding to 0,1,2,3 of e
        rf = [ratio(RAW_DATA, lambda t: t[v]==s, lambda t:all(t[w] == S[w][e[w]] for w in e)) for s in S[v]]
        # Special treatment for missing samples
        #### if sum(rf) == 0: rf = [1/len(rf)]*len(rf) # Commented out on purpose

        #print(len(divs), '-'*20)
        #print('Query:', v, 'given', e)
        #print('rf: ', rf)
         
        div = [(v, e), rf]
        for m in models:
            #print('\nModel:', m.edges())
            ve = VariableElimination(m)
            e_hat = {temp:S[temp][e[temp]]  for temp in e } ##mikel
            #q = ve.query(variables = [v], evidence = e)
            q = ve.query(variables = [v], evidence = e_hat,show_progress=False) ##mikel
            #div.extend([q[v].values, entropy(rf, q[v].values)])
            div.extend([q.values, entropy(rf, q.values)]) ##mikel
            #print('PGM:', q[v].values, ', Divergence:', div[-1]) 
            #print('PGM:', q.values, ', Divergence:', div[-1]) ##mikel
        divs.append(div)        #<-------------- divs: list of [(target,evidence), relative freq: [target|freq],  p_m1(target|freq), KL(rf||p_m1), p_m2(target|freq), KL(rf||p_m2) )]
    except:
        # Error occurs if variable is both target and evidence. We can ignore it.
        # (Also, this case should be avoided with current code)
        pass

divs2 = [r for r in divs if math.isfinite(r[3]) and math.isfinite(r[5])]
# What is the meaning of what is printed below?
n = 2

print([len([r for r in divs2 if len(r[0][1])==n]), # times where evidence has 2 random variables
       len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]), # times that model1 better than model2
       len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]), # times that model2 better than model1
       len([r for r in divs if len(r[0][1])==n and \
            not(math.isfinite(r[3]) and math.isfinite(r[5]))]), # times that divergence is NAN, i.e. no samples
       sum(r[3] for r in divs2 if len(r[0][1])==n),
       sum(r[5] for r in divs2 if len(r[0][1])==n)]) # sum of divergence of first model for all queries where evidence has 2 variables

for n in np.arange(4)+1:
    norm = len([r for r in divs2 if len(r[0][1])==n])
    print("N=",n,"-> ",[len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]])/norm, 
       len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]])/norm,
       sum(r[3] for r in divs2 if len(r[0][1])==n),
       sum(r[5] for r in divs2 if len(r[0][1])==n),
       len([r for r in divs if len(r[0][1])==n and \
            not(math.isfinite(r[3]) and math.isfinite(r[5]))]) ])

# The following is required for working with same data in next task:
import pickle
f = open('data.pickle', 'wb')
pickle.dump(divs2, f)
f.close()

separator()
# End of Task 4
'''



''' 
# Task 5 ------------ Checking for overfitting

from scipy.stats import entropy

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model1.cpds[0].state_names
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

S = STATE_NAMES

# Assumes you pickled data from previous task
import pickle
divs_in = pickle.load(open('data.pickle', 'rb'))

divs = []
k_fold = 5
for k in range(k_fold):
    # Dividing data into 75% training, 25% validation.
    # Change the seed to something of your choice:
    seed = 'your personal random seed string' + str(k)
    raw_data1, raw_data2 = get_random_partition(0.75, seed)
    training_data = pd.DataFrame(data=raw_data1)
    # refit with training data
    [m.remove_cpds(*m.cpds) for m in models] # Gets rid of warnings
    [m.fit(training_data) for m in models]
    for i, div in enumerate(divs_in):
        print(len(divs_in)*k + i,'/', len(divs_in)*k_fold)
        div = div[:] # Make a copy for technical reasons
        try:
            v, e = div[0]
            # Relative frequencies from validation data, compared below
            rf = [ratio(raw_data2, lambda t: t[v]==s,
                        lambda t:all(t[w] == S[w][e[w]] for w in e)) \
                  for s in S[v]]
            for m in models:
                #print('\nModel:', m.edges())
                ve = VariableElimination(m)
                q = ve.query(variables = [v], evidence = e)
                div.append(entropy(rf, q[v].values))
                #print('PGM:', q[v].values, ', Divergence:', div[-1])
            divs.append(div)
        except IndexError:
            print('fail')

# Filter out the failures
divs2 = [d for d in divs if len(d) == 8]

# Modify the following lines according to your needs.
# Perhaps turn it into a loop as well.
n = 1
print([len([r for r in divs2 if len(r[0][1])==n]),
       len([r for r in divs2 if len(r[0][1])==n and r[3] < r[5]]),
       len([r for r in divs2 if len(r[0][1])==n and r[3] > r[5]]),
       len([r for r in divs2 if len(r[0][1])==n and r[-2] < r[-1]]),
       len([r for r in divs2 if len(r[0][1])==n and r[-2] > r[-1]])])

separator()
# End of Task 5
'''



# ''' 
# Task 6 ------------ Finding a better structure

data = pd.DataFrame(data=RAW_DATA)

model1 = BayesianModel([('delay', 'age'),
                       ('delay', 'gender'),
                       ('delay', 'avg_mat'),
                       ('delay', 'avg_cs')])

model2 = BayesianModel([('age', 'delay'),
                        ('gender', 'delay'),
                        ('avg_mat', 'delay'),
                        ('avg_cs', 'delay')])

models = [model1, model2]

[m.fit(data) for m in models] # ML-fit

STATE_NAMES = model1.cpds[0].state_names
#print(model2.cpds[3])
print('\nState names:')
for s in STATE_NAMES:
    print(s, STATE_NAMES[s])

# Information for the curious:
# Structure-scores: http://pgmpy.org/estimators.html#structure-score
# K2-score: for instance http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
# Additive smoothing and pseudocount: https://en.wikipedia.org/wiki/Additive_smoothing
# Scoring functions: https://www.cs.helsinki.fi/u/bmmalone/probabilistic-models-spring-2014/ScoringFunctions.pdf
k2 = K2Score(data)
print('Structure scores:', [k2.score(m) for m in models])

separator()

print('\n\nExhaustive structure search based on structure scores:')

from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, BicScore

# Warning: Doing exhaustive search on a PGM with all 5 variables
# takes more time than you should have to wait. Hence
# re-fit the models to data where some variable(s) has been removed
# for this assignement.
raw_data2 = {'age': data['age'],
             'avg_cs': data['avg_cs'],
             'avg_mat': data['avg_mat'],
             'delay': data['delay'], # Don't comment out this one
             'gender': data['gender'],
             }

data2 = pd.DataFrame(data=raw_data2)

import time
t0 = time.time()
# Uncomment below to perform exhaustive search
#searcher = ExhaustiveSearch(data2, scoring_method=K2Score(data2))
#search = searcher.all_scores()
print('time:', time.time() - t0)
#print(len(search))
# Uncomment for printout:
# for score, model in search:
#     print("{0}        {1}".format(score, model.edges()))


# hillclimb search with K2score and Bicscore
separator()
print('\n\nHillClimb search based on structure scores:')
est = HillClimbSearch(data2, scoring_method=K2Score(data2))
best_model = est.estimate()
t0 = time.time()
print("Best model nodes:",sorted(best_model.nodes()))
print("Best model edges:",best_model.edges())
print('time:', time.time() - t0)

separator()
print('\n\nHillClimb search based on structure scores:')
est = HillClimbSearch(data2, scoring_method=BicScore(data2))
best_model = est.estimate()
t0 = time.time()
print("Best model nodes:",sorted(best_model.nodes()))
print("Best model edges:",best_model.edges())
print('time:', time.time() - t0)
# End of Task 6
# '''