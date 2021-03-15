import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

# Implemented methods
methods = ['DynProg', 'ValIter', 'PolIter'];

# Some colours
RED          = '#FF0000';
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100
    MINOUTAUR_REWARD = -10

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        states[s] = (i,j,k,l);
                        map[(i,j,k,l)] = s;
                        s += 1;
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        i = self.states[state][0];
        j = self.states[state][1];
        k = self.states[state][2];
        l = self.states[state][3];
        
        # Is the future position an impossible one ?
        row = i + self.actions[action][0];
        col = j + self.actions[action][1];
        
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) 
        
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state
        else:
            return self.map[(row, col, k, l)];
        
    def __move_police(self, state):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        
        # Compute the future position given current (state, action)
        i = self.states[state][0]
        j = self.states[state][1]
        k = self.states[state][2]
        l = self.states[state][3]
        # Random action
        action = round(np.random.rand()*3) + 1;
        
        # Is the future position an impossible one ?
        row = k + self.actions[action][0];
        col = l + self.actions[action][1];
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) 
        
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state
        else:
            return self.map[(i,j,row,col)];
        
        
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s,a);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities
    
    def sarsa_reward(self,s,a):
        r = 0
        next_s = self.__move(s, a)
        i = self.states[next_s][0]
        j = self.states[next_s][1]
        k = self.states[next_s][2]
        l = self.states[next_s][3]
        if s == next_s and a != self.STAY:
            r += self.IMPOSSIBLE_REWARD
        if i == k and j == l:
            r += self.MINOUTAUR_REWARD
        next_s = self.__move_police(next_s)
        
        i = self.states[next_s][0]
        j = self.states[next_s][1]
        k = self.states[next_s][2]
        l = self.states[next_s][3]
        if i == k and j == l:
            r = self.MINOUTAUR_REWARD
        if self.maze[i,j] == 2:
            r += self.GOAL_REWARD
        return r, next_s
            
    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s,a) # player takes deterministical action a         
                i = self.states[s][0]
                j = self.states[s][1]
                k = self.states[s][2]
                l = self.states[s][3]
                if i==k and j==l:
                    rewards[s,a] = self.MINOUTAUR_REWARD
                else:
                    i = self.states[next_s][0]
                    j = self.states[next_s][1]
                    k = self.states[next_s][2]
                    l = self.states[next_s][3]                

                    if s == next_s and a != self.STAY:
                        # Reward for hitting a wall
                        rewards[s,a] += self.IMPOSSIBLE_REWARD
                    # Reward for getting caught from police
                    elif (i == k) and (j == l):
                        rewards[s,a] += self.MINOUTAUR_REWARD
                    # Reward for being at the banks
                    elif self.maze[i,j] == 2:
                        rewards[s,a] += self.GOAL_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] += self.STEP_REWARD;
        return rewards;

    def simulate(self, start, policy, method,T):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state after action from policy
                path.append(self.states[next_s])
                # Add the position in the maze corresponding to the next state after random move of minotaur
                next_s = self.__move_police(next_s);
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter' or method == 'PolIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Add the position in the maze corresponding to the next state after random move of minotaur
            next_s = self.__move_police(next_s);
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while t < T:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Add the position in the maze corresponding to the next state after random move of minotaur
                next_s = self.__move_police(next_s);
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path

    def sample(self, start, policy, method, nr, T):
        # returns
        # 0 if eaten
        # 1 if escaped to the exit
        # 2 if only survived the T time steps
        samples = np.zeros(nr);
        for j in range(nr):
            path = self.simulate(start, policy, method);
            if path[-1][0:2] == path[-1][2:4]:
                samples[j] = 0
            elif self.maze[path[-1][0:2]] == 2:
                samples[j] = 1
            else:
                samples[j] = 2

        eaten = sum(samples==0) / nr;
        exited = sum(samples==1) / nr;
        survived = sum(samples==2) / nr;
        return (eaten, exited, survived, samples)

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def policy_iteration(env, gamma):
    """ Solves the shortest path problem using policy iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon    
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    policy = np.zeros(n_states, dtype = int)
    Bpolicy = np.ones(n_states, dtype = int)
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    
    # Iteration counter
    n   = 0;
 
    # Iterate until convergence
    while (not (policy == Bpolicy).all()):
        # Increment by one the numbers of iteration
        n += 1;
        policy = Bpolicy
        # Update the value function (policy evaluation)
        for s in range(n_states):
            V[s] = r[s, policy[s]] + gamma*np.dot(p[:,s,policy[s]],V);
        
        # Policy improvement
        for s in range(n_states):
            for a in range(n_actions):
                Q[s,a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        Bpolicy = np.argmax(Q,1);
        # Show error
        #print(n,": ",np.linalg.norm(policy - Bpolicy))
    
    # Return the obtained policy
    return V, policy;

def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    plt.draw()

def Q_learning(env, gamma, nr):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :param float alpha_t      : Step size
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    
    # Initialization
    Q = np.zeros([n_states, n_actions])
    n = np.zeros([n_states, n_actions])
    plot_s = list()
    plot_t = list()
    s_t = env.map[(0,0,3,3)]
    for t in tqdm(list(range(nr))):
        a_t = round(np.random.rand()*4)
        n[s_t, a_t] += 1 
        r_t, s_next = env.sarsa_reward(s_t,a_t)
        # Update Q(s_t, a_t)
        alpha = 1 / np.power(n[s_t,a_t],2/3)
        Q[s_t,a_t] = Q[s_t,a_t] + alpha * (r_t + gamma*np.max(Q[s_next]) - Q[s_t,a_t])
        s_t = s_next
        if t%100 == 0:
            plot_s.append(np.max(Q[env.map[(0,0,3,3)]]))
            plot_t.append(t)
    policy = np.argmax(Q,1)
    plt.style.use('seaborn-darkgrid')
    plt.plot(plot_t, plot_s)
    plt.xlabel("Number of iterations")
    plt.ylabel("Evolution of Value Function")
    plt.show()
    return Q, policy


def SARSA(env, gamma, epsilon, nr):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :param float alpha_t      : learning rate
        :return numpy.array Q     : Learned (state, action) function values corresponding to the behaviour policy
        :return numpy.array policy: learned behaviour policy
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    
    # Initialization
    Q = np.zeros([n_states, n_actions])
    n = np.zeros([n_states, n_actions])
    plot_s = list()
    plot_t = list()
    s_t = env.map[(0,0,3,3)]
    a_t = np.random.randint(5)
    for t in tqdm(list(range(nr))):
        n[s_t, a_t] += 1     
        r_t, s_next = env.sarsa_reward(s_t,a_t)
        # Select the next planned action
        if np.random.rand() <= epsilon:
            a_next = np.random.randint(5)
        else:
            a_next = np.argmax(Q[s_next])
        # Update Q(s_t, a_t)
        alpha = 1 / np.power(n[s_t,a_t],2/3)
        Q[s_t,a_t] = Q[s_t,a_t] + alpha * (r_t + gamma*Q[s_next,a_next] - Q[s_t,a_t])
        s_t = s_next
        a_t = a_next
        if t%100 == 0:
            plot_s.append(np.max(Q[env.map[(0,0,3,3)]]))
            plot_t.append(t)
    policy = np.argmax(Q,1)
    plt.style.use('seaborn-darkgrid')
    plt.plot(plot_t, plot_s)
    plt.xlabel("Number of iterations")
    plt.ylabel("Evolution of Value Function")
    plt.show()
    return Q, policy


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


                
def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    flagg = False
    for i in range(len(path)):
        player_now = path[i][0:2]
        player_before = path[i-2][0:2]
        
        minotaur_now = path[i][2:4]
        minotaur_before = path[i-2][2:4]
        
        if i == 0:
            grid.get_celld()[(minotaur_now)].set_facecolor(RED)
            grid.get_celld()[(minotaur_now)].get_text().set_text('Minotaur')
            continue
            
        # Player's turn
        if not i%2:
            if player_now == minotaur_now:
                # The player goes to the minotaur and gets eaten
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(player_now)].get_text().set_text('Player'+str(round(i/2)))
                grid.get_celld()[(player_now)].set_facecolor(RED)
                grid.get_celld()[(player_now)].get_text().set_text('EATEN')
            elif maze[player_now] == 2:
                # The player escapes
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_now)].get_text().set_text('FINISH')
                grid.get_celld()[(player_now)].get_text().set_text('Player'+str(round(i/2)))
            else:
                # The player does normal move
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(player_now)].get_text().set_text('Player'+str(round(i/2)))
            if not player_now == player_before:
                # Reset the color of last position
                grid.get_celld()[(player_before)].set_facecolor(col_map[maze[player_before]])
                grid.get_celld()[(player_before)].get_text().set_text('')  
        # Minotaur's turn
        else:
            if player_now == minotaur_now:
                # The minotaur eats the player
                grid.get_celld()[(player_now)].set_facecolor(RED)
                grid.get_celld()[(player_now)].get_text().set_text('EATEN')
            else:
                # The minotaur does a normal move
                grid.get_celld()[(minotaur_now)].set_facecolor(RED)
                grid.get_celld()[(minotaur_now)].get_text().set_text('Minotaur')
                
            if not minotaur_now == minotaur_before:
                # Reset the color of last position
                grid.get_celld()[(minotaur_before)].set_facecolor(col_map[maze[minotaur_before]])
                grid.get_celld()[(minotaur_before)].get_text().set_text('')
                
        display.display(fig)
        display.clear_output(wait=True)

        time.sleep(0.3)
