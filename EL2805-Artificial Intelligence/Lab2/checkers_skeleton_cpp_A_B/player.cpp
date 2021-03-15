#include "player.hpp"
#include <cstdlib>

namespace checkers
{
#define INF 1999999999
GameState Player::play(const GameState &pState,const Deadline &pDue)
{
    // std::cerr << "Time befor: " << pDue.now().getSeconds()<< std::endl;

    this->time = pDue.now().getSeconds();
    std::vector<GameState> lNextStates;
    pState.findPossibleMoves(lNextStates);
    int index = 0, score =-INF, v =0;
    
    this->playa = pState.getNextPlayer();
    if (playa == CELL_RED) this->nextplayer= CELL_WHITE;
    else this->nextplayer= CELL_RED;
    
    unsigned int previous_index = 0;
    for (unsigned int  k = 4; k < 15; k++)
    {
        for (unsigned int  i = 0; i < lNextStates.size(); i++)
        {

            if(pDue.now().getSeconds() - this->time > 0.5) 
            {
                return lNextStates[previous_index];
                std::cerr << "Depth: " << k-1<< std::endl;
            }
            v = alphabeta(lNextStates[i],-INF, INF, this->nextplayer,k,pDue);
            if (score < v)
            {
                score = v;
                index = i;
            }
        }
        previous_index = index;
    }
    // if (lNextStates.size() == 0) return GameState(pState, Move());

    /*
     * Here you should write your clever algorithms to get the best next move, ie the best
     * next state. This skeleton returns a random move instead.
     */
    // std::cerr << "Depth: " << pDue.now().getSeconds()<< std::endl;
    return lNextStates[index];
}

int Player::alphabeta(const GameState &pstate, int alpha, int beta, unsigned int player , unsigned int depth, const Deadline &pDue)
{

    std::vector<GameState> lNextStates;
    pstate.findPossibleMoves(lNextStates);

    int v = -INF; // because we are the max player

    if (depth == 0 || lNextStates.empty())
    {
        v = eval(pstate, this->playa);
    }
    else if(pDue.now().getSeconds() - this->time > 0.5) return 0;  // return 0 cause we dont use this anyways
    else if (player == this->playa)
    {
        v = -INF;
        for (unsigned int i = 0; i < lNextStates.size(); i++)
        {
            // if(pDue.now().getSeconds() - this->time > 0.5) return v = eval(pstate, this->playa);
            v = std::max(v, alphabeta(lNextStates[i], alpha, beta, this->nextplayer, depth - 1, pDue ));
            alpha = std::max(alpha, v);
            if (beta <= alpha) break; // beta prune
        }
    }
    else  // player == opponent
    {
        v = INF;
         for (unsigned int i = 0; i < lNextStates.size(); i++)
        {
            // if(pDue.now().getSeconds() - this->time > 0.5) return v = eval(pstate, this->playa);
            v = std::min(v, alphabeta(lNextStates[i], alpha, beta, this->playa, depth - 1, pDue ));
            beta = std::min(beta, v);
            if (beta <= alpha) break; // alpha prune
        }
    }

     
    return v;
}


int Player::eval(const GameState &pstate, unsigned int player)
{

    // if end of game
    if ( pstate.isEOG())
    {
        if (player == CELL_WHITE)
        {
            if(pstate.isWhiteWin()) return INF;
            if (pstate.isRedWin()) return -INF;
            return 0;        
        }
        else
        {
            if(pstate.isWhiteWin()) return -INF;
            if (pstate.isRedWin()) return INF;
            return 0;       
        }
    }

    int first_digits = 0;
    int second_digits = 0;
    int third_digits = 0;
    int fourth_digits = 0;

    int pieces[8] = {0,0,0,0,0,0,0};
    int kings[8] = {0,0,0,0,0,0,0,0,};
    uint8_t temp= 0;



    for (unsigned int  i = 0; i < 8; i++)
    {
        for (unsigned int j = 0; j < 8; j++)
        {   
            temp = pstate.at(i,j);
            if(temp != CELL_EMPTY && temp != CELL_INVALID)
            {
                if (temp == this->playa)
                {
                    if (temp == CELL_KING)   // our king
                    {
                        kings[i]++;
                    }
                    else pieces[i]++; // our piece
                }
                else if (pstate.at(i,j)== this->nextplayer) 
                {
                    if (temp == CELL_KING)   // not our king
                    {
                        kings[i]--;
                    }
                    else pieces[i]--;
                }
                third_digits++;  // for now has the nr of total pieces
            }
        }
    }

    


    temp = third_digits; // temp has now total nr of pieces on the board
    for (unsigned int  i = 0; i < 8; i++)
    {
        first_digits += pieces[i] * 3 + kings[i]*5;
        second_digits += pieces[i] * weights[i];
        third_digits += pieces[i] + kings[i];               // used to find out who has more pieces
        fourth_digits += kings[i] * weights2[i];
    }

    // third digits
    if(third_digits == 0) third_digits = 0;
    if(third_digits < 0) third_digits = -(24 - temp)*4;
    else third_digits = (24 - temp)*4;
    
    // fifth digit
    temp = rand() % 10;


    return first_digits*10000000 + second_digits*100000 + third_digits*1000 + fourth_digits*10 + temp ;
}
/*namespace checkers*/ }
