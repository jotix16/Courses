#ifndef _TICTACTOE3D_PLAYER_HPP_
#define _TICTACTOE3D_PLAYER_HPP_

#include "constants.hpp"
#include "deadline.hpp"
#include "move.hpp"
#include "gamestate.hpp"
#include <vector>

namespace TICTACTOE3D
{

class Player
{
public:
    ///perform a move
    ///\param pState the current state of the board
    ///\param pDue time before which we must have returned
    ///\return the next state the board is in after our move
    GameState play(const GameState &pState, const Deadline &pDue);
    int alphabeta(const GameState pstate, int alpha, int beta, unsigned int player, unsigned int depth);
    int eval(const GameState pstate, unsigned int player);
    
private:
    unsigned int playa;
    unsigned int nextplayer;
    int weights[4][4]= { {0,-10,-100,-1000}, 
                         {10, 1,10,0}, 
                         {100,-10,-1,0}, 
                         {1000,0,0,0} };

};

/*namespace TICTACTOE3D*/ }

#endif
