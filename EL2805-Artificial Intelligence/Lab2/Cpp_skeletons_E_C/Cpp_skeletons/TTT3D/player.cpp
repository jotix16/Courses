#include "player.hpp"
#include <cstdlib>
#include <algorithm>
#include <math.h>

namespace TICTACTOE3D
{
#define INF 99999999
GameState Player::play(const GameState &pState,const Deadline &pDue)
{
    // std::cerr << "Processing " << pDue.getSeconds() << std::endl;

    std::vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);
    int index = 0, score =-INF, v =0;
    
    this->playa = pState.getNextPlayer();
    if (playa == CELL_X) this->nextplayer= CELL_O;
    else this->nextplayer= CELL_X;
    
    int depth = 1;
    if(lNextStates.size()<=39) depth = 2;
    
    for (unsigned int  i = 0; i < lNextStates.size(); i++)
    {
        v = alphabeta(lNextStates[i],-INF, INF, this->nextplayer,depth);
        if (score < v)
        {
            score = v;
            index = i;
        }
        
    }
    // if (lNextStates.size() == 0) return GameState(pState, Move());

    /*
     * Here you should write your clever algorithms to get the best next move, ie the best
     * next state. This skeleton returns a random move instead.
     */
    
    return lNextStates[index];
}



int Player::alphabeta(const GameState pstate, int alpha, int beta, unsigned int player , unsigned int depth)
{

    std::vector<GameState> lNextStates;
    pstate.findPossibleMoves(lNextStates);

    int v = -INF; // because we are the max player

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
    }   
    return v;
}

int Player::eval(const GameState pstate, unsigned int player)
{
    int final_score = 0;
    int temp =0, temp2 =0;
  /////////////////////// ROWS /////////////////////////
    // 1) rows
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned j = 0; j < 4; j++)
        {
            temp = 0;
            temp2 = 0;
            for (unsigned  k = 0; k < 4; k++)
            {   
                if (pstate.at(i,j,k) == this->playa)
                {
                    temp++;
                }
                else if (pstate.at(i,j,k) == this->nextplayer)
                {
                    temp2++;
                }
            }
            final_score += weights[temp][temp2];      
        }
    }

    // 2) rows
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned j = 0; j < 4; j++)
        {
            temp = 0;
            temp2 = 0;
            for (unsigned  k = 0; k < 4; k++)
            {
                if (pstate.at(i,k,j) == this->playa)
                {
                    temp++;
                }
                else if (pstate.at(i,k,j) == this->nextplayer)
                {
                    temp2++;
                }
            }
            final_score += weights[temp][temp2];       
        }
    }

    //3) rows
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned j = 0; j < 4; j++)
        {
            temp = 0;
            temp2 = 0;
            for (unsigned  k = 0; k < 4; k++)
            {   
                if (pstate.at(k,i,j) == this->playa)
                {
                    temp++;
                }
                else if (pstate.at(k,i,j) == this->nextplayer)
                {
                    temp2++;
                }
            }
            final_score += weights[temp][temp2];       
        }
    }


  /////////////////////// 2D Diagonals /////////////////////////
    // 1) 2D diagonals ijj
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(i,j,j) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(i,j,j) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }

    // 2) 2D diagonals ij3-j
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(i,j,3-j) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(i,j,3-j) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }

    // 3) 2D diagonals jji
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(j,j,i) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(j,j,i) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }

    // 4) 2D diagonals j3-ji
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(j,3-j,i) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(j,3-j,i) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }

    // 5) 2D diagonals jij
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(j,i,j) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(j,i,j) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }

    // 6) 2D diagonals ji3-j
    for (unsigned int i = 0; i < 4; i++)
    {
        temp = 0;
        temp2 = 0;
        for (unsigned j = 0; j < 4; j++)
        {           
            // column score: i -> column, j -> row

            if (pstate.at(j,i,3-j) == this->playa)
            {
                temp++;
            }          
            else if (pstate.at(j,i,3-j) == this->nextplayer)
            {
                temp2++;
            }
        }
        final_score += weights[temp][temp2]; 
    }


  /////////////////////// 3D Diagonals /////////////////////////

    // 1) 3D diagonals iii
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {    
        if(pstate.at(i,i,i) == this->playa) 
        {
            temp++;
        }
        else if (pstate.at(i,i,i) == this->nextplayer)
        {
            temp2++;
        }        
    }
    final_score += weights[temp][temp2]; 

    // 2) 3D diagonals ii3-i
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {    
        if(pstate.at(i,i,3-i) == this->playa) 
        {
            temp++;
        }
        else if (pstate.at(i,i,3-i) == this->nextplayer)
        {
            temp2++;
        }        
    }
    final_score += weights[temp][temp2]; 

    // 3) 3D diagonals i3-ii
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {    
        if(pstate.at(i,3-i,i) == this->playa) 
        {
            temp++;
        }
        else if (pstate.at(i,3-i,i) == this->nextplayer)
        {
            temp2++;
        }        
    }
    final_score += weights[temp][temp2]; 

    // 4) 3D diagonals 3-iii
    temp = 0;
    temp2 = 0;
    for (unsigned int i = 0; i < 4; i++)
    {    
        if(pstate.at(3-i,i,i) == this->playa) 
        {
            temp++;
        }
        else if (pstate.at(3-i,i,i) == this->nextplayer)
        {
            temp2++;
        }        
    }
    final_score += weights[temp][temp2]; 

    return final_score;
}

/*namespace TICTACTOE3D*/ }
