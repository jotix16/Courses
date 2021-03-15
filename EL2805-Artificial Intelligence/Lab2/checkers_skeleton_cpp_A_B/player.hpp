#ifndef _CHECKERS_PLAYER_HPP_
#define _CHECKERS_PLAYER_HPP_

#include "constants.hpp"
#include "deadline.hpp"
#include "move.hpp"
#include "gamestate.hpp"
#include <vector>

namespace checkers
{

class Player
{
public:
    ///perform a move
    ///\param pState the current state of the board
    ///\param pDue time before which we must have returned
    ///\return the next state the board is in after our move
    GameState play(const GameState &pState, const Deadline &pDue);
    int alphabeta(const GameState &pstate, int alpha, int beta, unsigned int player, unsigned int depth, const Deadline &pDue);
    int eval(const GameState &pstate, unsigned int player);
    
private:
    unsigned int playa;
    unsigned int nextplayer;
    double time;
    int weights[8]= { 0,0,0,0,8,18,25,0};
    int weights2[8] = {0,15,30,40,40,30,15,0};

};

/*namespace checkers*/ }

#endif
