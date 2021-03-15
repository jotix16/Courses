#include "player.hpp"
#include <cstdlib>
#include <algorithm>
#include <math.h>

namespace TICTACTOE
{
#define INF 99999999

GameState Player::play(const GameState &pState,const Deadline &pDue)
{
    //std::cerr << "Processing " << pState.toMessage() << std::endl;
    //  std::cerr<<"NEW MOVE: "<<std::endl;
    std::vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);


    // if (lNextStates.size() == 0) return GameState(pState, Move());
    int alpha = -INF, beta = INF, index = 0, score =-INF, v =0;

    this->playa = pState.getNextPlayer();
    if (playa == CELL_X) this->nextplayer= CELL_O;
    else this->nextplayer= CELL_X;
    
    

    for (unsigned int  i = 0; i < lNextStates.size(); i++)
    {
        v = alphabeta(lNextStates[i],alpha, beta, this->nextplayer,3);
        if (score < v)
        {
            score = v;
            index = i;
        }
        
    }
    
    /*
     * Here you should write your clever algorithms to get the best next move, ie the best
     * next state. This skeleton returns a random move instead.
     */
    // return lNextStates[rand() % lNextStates.size()];
    // std::cerr<<"MIkel: "<< score<<std::endl;
    return lNextStates[index];
}

int Player::alphabeta(const GameState pstate, int alpha, int beta, unsigned int player , unsigned int depth)
{

    std::vector<GameState> lNextStates;
    pstate.findPossibleMoves(lNextStates);

    int v = -INF; // because we are the max player

    // check termination
    if (depth == 0 || lNextStates.empty())
    {
        v = eval(pstate, this->playa);
    }
    else if (player == this->playa)
    {
        v = -INF;
        for (unsigned int i = 0; i < lNextStates.size(); i++)
        {
            v = std::max(v, alphabeta(lNextStates[i], alpha, beta, this->nextplayer, depth - 1 ));
            alpha = std::max(alpha, v);
            if (beta <= alpha) break; // beta prune
        }
        // std::cerr<<std::endl<<"Player we(max)";
    }
    else  // player == opponent
    {
        v = INF;
         for (unsigned int i = 0; i < lNextStates.size(); i++)
        {
            v = std::min(v, alphabeta(lNextStates[i], alpha, beta, this->playa, depth - 1 ));
            beta = std::min(beta, v);
            if (beta <= alpha) break; // alpha prune
        }
        // std::cerr<<std::endl<<"Player opponent(min)";     
    }   
    // std::cerr<<v; 
    return v;
}

int Player::eval(const GameState pstate, unsigned int player)
{
    int final_score = 0;

    int base =10;
    int temp =0, temp2 =0;

    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {   
            // row score: i -> row, j -> column
            if (pstate.at(i,j) == this->playa)
            {
                final_score += pow(base,temp);
                temp++;
            }
            else if (pstate.at(i,j) == this->nextplayer)
            {
                final_score -= pow(base,temp2);
                temp2++;
            }
      
         }

    }   
    

    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(j,i) == this->playa)
            {
                final_score += pow(base,temp);
                temp++;
            }          
            else if (pstate.at(j,i) == this->nextplayer)
            {
                final_score -= pow(base,temp2);
                temp2++;
            }
        }

    }

    // diagonal score for the diagonal 0,0 -> 3,3
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {    
        if(pstate.at(i,i) == this->playa) 
        {
            final_score += pow(base,temp);
            temp++;
        }
        else if (pstate.at(i,i) == this->nextplayer)
        {
            final_score -= pow(base,temp2);
            temp2++;
        }        
    }


    // diagonal score for the diagonal 0,3 -> 3,0
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {
        if (pstate.at(i,4-i-1) == this->playa)
        {
            final_score += pow(base,temp);
            temp++;
        }
        else if (pstate.at(i,4-i-1) == this->nextplayer)
        {
            final_score -= pow(base,temp2);
            temp2++;
        }
    }


    return final_score;
}

/*namespace TICTACTOE*/ }
