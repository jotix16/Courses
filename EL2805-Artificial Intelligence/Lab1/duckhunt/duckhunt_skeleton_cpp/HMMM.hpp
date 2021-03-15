#ifndef _HMMM_HPP_
#define _HMMM_HPP_
#include "matrix_class.h"
#include <time.h>

class HMM{

    private:

        int max_iters = 30; // Max iterations when estimating model
        int nr_states;
        int nr_emissions;
        Matrix A;
        Matrix B;
        Matrix PI;
    public:

        HMM()
        {}
        HMM(int anr_states, int anr_emissions ) : nr_states(anr_states), nr_emissions(anr_emissions), A{nr_states,nr_states,0.0}, B{nr_states,nr_emissions,0.0}, PI{1,nr_states, 0.0}
        {
        }
        
        Matrix getA()
        {
            return A;	
        }
        Matrix getB()
        {
            return B;	
        }
        Matrix getPI()
        {
            return PI;	
        }

        void initialize_hmm(HMM X)
        {
            A=X.getA();
            B=X.getB();
            PI=X.getPI();
        }

        void random_initialize_hmm()
        {
            cerr<<"MIKEL "<<nr_states;
              //random generator
            //random init of A
            double temp=0;
            for (int i = 0; i < nr_states; i++)
            {
                //cerr<< "i : "<<i<<" ";
                for (int  j = 0; j < nr_states; j++)
                {
                    A(i,j) = rand() % 100;
                    temp += A(i,j);
                }
                
                for (int  j = 0; j < nr_states; j++)
                {
                    A(i,j) = A(i,j)/temp;
                }
                temp = 0;
                //cerr<< "i : "<<i<<" ";
            }
            cerr<<"MIKEL22 "<<nr_states;
            // random init of B
            temp = 0;
            for (int i = 0; i < nr_states; i++)
            {
                for (int  j = 0; j < nr_emissions; j++)
                {
                    B(i,j) = rand() % 100;
                    temp += B(i,j);
                }
                
                for (int  j = 0; j < nr_emissions; j++)
                {
                    B(i,j) = B(i,j)/temp;
                }
                temp = 0;           
            }

            // random init of PI
            temp = 0;
            for (int i = 0; i < nr_states; i++)
            {
                PI(0,i) = rand() % 100;
                temp += B(0,i);
            }
            
            for (int i = 0; i < nr_states; i++)
            {
                 PI(0,i) =  PI(0,i)/temp;
            }            
            
        }

        HMM(Matrix aA, Matrix aB, Matrix aPI )
        {
            nr_states = aA.getRows();
            nr_emissions = aB.getCols();
            A = aA;
            B = aB;
            PI = aPI;
        }

        // ~HMM() = default;

        void print()
        {
            cerr<<"A: "<<endl;
            A.print();
            cerr<<"B: "<<endl;
            B.print();
        }
        
        void print2()
        {
            A.print2();
            B.print2();
        }

        // estimates the probab for next emission given a current distro
        Matrix estimate_probability_distro_next_emission(Matrix current_distribution)
        {
            return PI*A*B;
        }

        // gives the probability for a given emission sequence O
        double estimate_probability_of_emission_seq(Matrix O)
        {
            Matrix alfa(alfa_pass( A,  B,  PI,  O));
            return alfa.return_row(O.getCols()-1).sum_columns()(0,0);
        }

        // estimates the most probable hiden state sequence for a given emission sequence
        Matrix estimate_state_sequence(Matrix O)
        {
                Matrix delta(O.getCols(),A.getRows(),0.0);
                Matrix indexes(O.getCols(),A.getRows(),1);
                Matrix temp(1,A.getRows(),0.0);

                delta.insert_row(0, PI^B.return_column(O(0,0)).transpose());
                for (int i = 1; i < O.getCols(); i++)
                {
                    delta.insert_row(i,maxmul(delta.return_row(i-1),A,temp)^B.return_column(O(0,i)).transpose());
                    indexes.insert_row(i-1,temp);
                }

                Matrix states(1,O.getCols(),0.0);
                double x;
                delta.return_row(O.getCols()-1).maxo(x); //find index of max state
                states(0,O.getCols()-1)=x;
                for (int i = 1; i < O.getCols(); i++)
                {
                    states(0,O.getCols()-i-1)=indexes(O.getCols()-i-1,states(0,O.getCols()-i));
                }
                return states;
        }

        void train(Matrix O)
        {
            int N = nr_states;
            int T = O.getCols();
            int M = nr_emissions;;
            
            Matrix alfa(T,N,1);
            Matrix c(1,T,1);
            Matrix beta(T,N,1);
            vector <Matrix> gama_1(T-1, Matrix(N, N,0));
            Matrix gama_2( T, N, 1 );
            double denom = 0;
            double numer = 0;
            double log_prob;
            double old_log_prob =-9999999;
            int iters=0;
        
            while (1)
            {
                //alfa_pass
                    //compute alfa0
                    c(0,0) = 0;
                    for (int i = 0; i < N; i++)
                    {
                        alfa(0,i) = PI(0,i)*B(i,O(0,0));
                        c(0,0) = c(0,0) + alfa(0,i);

                    }
                        //scale alfa 0
                    if (c(0,0) != 0)
                    {
                        c(0,0) = 1/c(0,0);
                    }
                    
                    for (int i = 0; i < N; i++)
                    {
                        alfa(0,i) = alfa(0,i)*c(0,0);
                    }

                        // compute alfa t
                    for (int t = 1; t < T; t++)
                    {
                        c(0,t) = 0;
                        for (int i = 0; i < N; i++)
                        {
                            alfa(t,i) = 0;
                            for (int j = 0; j < N; j++)
                            {
                                alfa(t,i) = alfa(t,i)+alfa(t-1,j)*A(j,i);
                            }
                            alfa(t,i) = alfa(t,i)*B(i,O(0,t));
                            c(0,t) = c(0,t)+alfa(t,i);
                        }
                
                        // scale alfa t
                        if (c(0,t) != 0)
                        {
                            c(0,t) = 1/c(0,t);
                        }

                        for (int i = 0; i < N; i++)
                        {
                            alfa(t,i) = c(0,t)*alfa(t,i);
                        }
                    }
                    //alfa.sum_columns().print2();

                // Beta-pass
                    for (int i = 0; i < N; i++)
                    {
                        beta(T-1,i) = c(0,T-1);
                    }

                    for (int t =T-2; t >= 0; t--)
                    {
                        //cout<<"what"<<t<<endl;
                        for (int i = 0; i < N; i++)
                        {
                            beta(t,i)=0;
                            for (int j = 0; j < N; j++)
                            {
                                beta(t,i) = beta(t,i) + A(i,j)*B(j, O(0,t+1))*beta(t+1,j);
                            }
                            beta(t,i) = c(0,t)*beta(t,i);
                        }  
                    }
                
                // gammas
                    for (int t = 0; t < T-1; t++)
                    {
                        for (int i  = 0; i < N; i++)
                        {
                            gama_2(t,i) = 0;
                            for (int j = 0; j < N; j++)
                            {
                                gama_1[t](i,j) = alfa(t,i)*A(i,j)*B(j,O(0,t+1))*beta(t+1,j);
                                gama_2(t,i) = gama_2(t,i)+gama_1[t](i,j);
                            }
                        }
                    }
                    // last row of gamma is a_T-1
                    for (int i = 0; i < N; i++)
                    {
                        gama_2(T-1,i) = alfa(T-1,i);
                    }
                
                //re-estimate system
                    // re-estimate PI
                    for (int i = 0; i < N; i++)
                    {
                        PI(0,i)=gama_2(0,i);
                    }
                    
                    //re-estimate A
                    for (int i = 0; i < N; i++)
                    {
                        denom=0;
                        for (int t = 0; t < T-1; t++)
                        {
                            denom = denom+ gama_2(t,i);
                        }
                        
                        for (int j = 0; j < N; j++)
                        {
                            numer = 0;
                            for (int t = 0; t < T-1; t++)
                            {
                                numer = numer+ gama_1[t](i,j);
                            }
                            if (denom != 0)
                            {
                                A(i,j)= numer/denom;
                            }
                            else
                            {
                                A(i,j) = 0.0;
                            }
                            
                            
                        }
                    }

                    // re-estimate B
                    for (int i = 0; i < N; i++)
                    {
                        denom=0;
                        for (int t = 0; t < T; t++)
                        {
                            denom = denom + gama_2(t,i);
                        }
                        
                        for (int j = 0; j < M; j++)
                        {
                            numer = 0;
                            for (int t = 0; t < T; t++)
                            {
                                if (O(0,t)==j)
                                {
                                    numer = numer + gama_2(t,i);
                                }
                            }
                            
                            
                            if (denom != 0)
                            {
                                B(i,j)= numer/denom;
                            }
                            else
                            {
                                B(i,j)= 0.0;
                            }
                        }
                    }

                // compute log
                    log_prob=0;
                    for (int i = 0; i < T; i++)
                    {
                        if (c(0,i) != 0)
                        {
                            log_prob = log_prob + log10(c(0,i));
                        }
                        
                    }
                    log_prob = -log_prob;

                // iterate or not iterate
                    iters++;
                    if (iters < max_iters && log_prob > (old_log_prob))
                    {
                        old_log_prob = log_prob;
                        //return make_tuple(A,B,PI);
                    }else
                    {
                        // cout<<"ITer: "<<iters<<endl;
                        //cout<<"newprob: "<<log_prob<<" oldprob: "<<old_log_prob<<endl;
                        return ;
                    }
            }
        }

};


#endif