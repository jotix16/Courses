#ifndef _matri_x
#define _matri_x

#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numeric>


using std::vector;
using std::tuple;
using namespace std;


class Matrix {
    private:
        int m_rowSize;
        int m_colSize;
        vector<vector<double> > m_matrix;
    public:
    Matrix ()
    {

    }
    Matrix(int rowSize, int colSize, double initial)
    {
        m_rowSize = rowSize;
        m_colSize = colSize;
        m_matrix.resize(rowSize);
        for (int i = 0; i < m_matrix.size(); i++)
        {
            m_matrix[i].resize(colSize, initial);
        }
    }
    
    Matrix(const Matrix &B)
    {
            this->m_colSize = B.getCols();
            this->m_rowSize = B.getRows();
            this->m_matrix = B.m_matrix;
            
    }

    ~Matrix() = default;


    Matrix operator+(Matrix &B)
    {
        Matrix sum(m_colSize, m_rowSize, 0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                sum(i,j) = this->m_matrix[i][j] + B(i,j);
            }
        }
        return sum;
    }
    
    Matrix operator^(Matrix B)
    {   
        Matrix prod(m_rowSize, m_colSize, 0.0);
        if (B.getRows()!=m_rowSize || B.getCols() != m_colSize)
        {
            cout<< "Error, cannot point product the matrixes."<<endl;
            return prod;
        }
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                prod(i,j) = this->m_matrix[i][j] * B(i,j);
            }
        }
        return prod;
    }

    // Subtraction of Two Matrices
    Matrix operator-(Matrix & B)
    {
        Matrix diff(m_colSize, m_rowSize, 0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                diff(i,j) = this->m_matrix[i][j] - B(i,j);
            }
        }
        
        return diff;
    }

    // Multiplication of Two Matrices
    Matrix operator*(Matrix  B)
    {
        Matrix multip(m_rowSize,B.getCols(),0.0);
        if(m_colSize == B.getRows())
        {
            int i,j,k;
            double temp = 0.0;
            for (i = 0; i < m_rowSize; i++)
            {
                for (j = 0; j < B.getCols(); j++)
                {
                    temp = 0.0;
                    for (k = 0; k < m_colSize; k++)
                    {
                        temp += m_matrix[i][k] * B(k,j);
                    }
                    multip(i,j) = temp;
                    //cout << multip(i,j) << " ";
                }
                //cout << endl;
            }
            return multip;
        }

    }

    // Scalar Addition
    Matrix operator+(double scalar)
    {
        Matrix result(m_rowSize,m_colSize,0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                result(i,j) = this->m_matrix[i][j] + scalar;
            }
        }
        return result;
    }

    // Scalar Subraction
    Matrix operator-(double scalar)
    {
        Matrix result(m_rowSize,m_colSize,0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                result(i,j) = this->m_matrix[i][j] - scalar;
            }
        }
        return result;
    }

    // Scalar Multiplication
    Matrix operator*(double scalar)
    {
        Matrix result(m_rowSize,m_colSize,0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                result(i,j) = this->m_matrix[i][j] * scalar;
            }
        }
        return result;
    }

    // Scalar Division
    Matrix operator/(double scalar)
    {
        Matrix result(m_rowSize,m_colSize,0.0);
        int i,j;
        for (i = 0; i < m_rowSize; i++)
        {
            for (j = 0; j < m_colSize; j++)
            {
                result(i,j) = this->m_matrix[i][j] / scalar;
            }
        }
        return result;
    }


    // Returns value of given location when asked in the form A(x,y)
    double& operator()(const int &rowNo, const int & colNo)
    {
        return this->m_matrix[rowNo][colNo];
    }

    // No brainer - returns row #
    int getRows() const
    {
        return this->m_rowSize;
    }

    // returns col #
    int getCols() const
    {
        return this->m_colSize;
    }

    // Take any given matrices transpose and returns another matrix
    Matrix transpose()
    {
        Matrix Transpose(m_colSize,m_rowSize,0.0);
        for (int i = 0; i < m_colSize; i++)
        {
            for (int j = 0; j < m_rowSize; j++) {
                Transpose(i,j) = this->m_matrix[j][i];
            }
        }
        return Transpose;
    }

    // Prints the matrix beautifully
    void print2() const
    {   
        // cout << "Matrix: " << endl;
        cout<<m_rowSize<<" "<<m_colSize<<" ";
        
        for (int i = 0; i < m_rowSize; i++) {
            for (int j = 0; j < m_colSize; j++) {
                cout << m_matrix[i][j] << " ";
            }
        }
        cout<<endl;
    }

    void print() const
    {   
        // cout << "Matrix: " << endl;
        
        for (int i = 0; i < m_rowSize; i++) {
            for (int j = 0; j < m_colSize; j++) {
                cerr << m_matrix[i][j] << " ";
            }
            cerr<<endl;
        }
        // cout<<endl;
    }

    //Replaces row'th row of matrix with x
    void insert_row(int row, Matrix x)
    {
        if (x.getRows()!=1 || x.getCols() != m_colSize)
        {
            cout<< "Error, cannot insertaaa vector in matrix."<<endl;
            return;
        }
        
        for (int i = 0; i < m_colSize; i++)
        {
            m_matrix[row][i]=x(0,i);
        }
        
    }

    //returns row'th row of matrix
    Matrix return_row(int row)
    {
        Matrix x(1,m_colSize,0.0);
        for (int i = 0; i < m_colSize; i++)
        {
            x(0,i)=m_matrix[row][i];
        }
        return x;
        
    }

    //returns column'th column of matrix
    Matrix return_column(int column)
    {
        Matrix x(m_rowSize,1,0.0);
        for (int i = 0; i < m_rowSize; i++)
        {
            x(i,0)=m_matrix[i][column];
        }
        return x;
    }

    // returns a row vector equal to sum of rows of the matrix
    Matrix sum_rows()
    {
        Matrix sum_of_rows(1,m_colSize,0.0);
        double x;
        for (int i = 0; i < m_colSize; i++)
        {
            for (int j = 0; j < m_rowSize; j++) {
                x=x+m_matrix[j][i];
            }
            sum_of_rows(0,i)=x;
            x=0;
        }
        return sum_of_rows;
    }

    // returns a column vector equal to sum of columns of the matrix
     Matrix sum_columns()
    {
        Matrix sum_of_columns(m_rowSize,1,0.0);
        double x;
        for (int i = 0; i < m_rowSize; i++)
        {
            for (int j = 0; j < m_colSize; j++) {
                x=x+m_matrix[i][j];
            }
            sum_of_columns(i,0)=x;
            x=0;
        }
        return sum_of_columns;
    }

    // returns max of all columns 
    Matrix maxo( )
    {
        //returns max of all colums as a column vector
        Matrix max(m_rowSize,1,0.0);
        for (int i = 0; i < m_rowSize; i++)
        {
            max(i,0)=*max_element(m_matrix[i].begin(), m_matrix[i].end());
        }
        
        return max;
    }

    // returns max of a vector together with its index
    Matrix maxo( double &x )
    {
        //returns max of all colums as a column vector
        Matrix max(m_rowSize,1,0.0);
        //cout<<"hello"<<endl;
        vector<double>::iterator it;
         Matrix ix(m_rowSize,1,0.0);
        for (int i = 0; i < m_rowSize; i++)
        {
            
            it=max_element(m_matrix[i].begin(), m_matrix[i].end());
            max(i,0)=*it;
            //cout<<*it<<endl;
            ix(i,0)=distance(m_matrix[i].begin(),it);
        }
        //max.print();
        //ix.print2();
        x=ix(0,0);
        return max;
    }
};


inline Matrix create_matrix(string mat)
{
    stringstream ss(mat);
    int row;
    int column;
    double c;

    ss >>row;
    ss >>column;

    Matrix X(row,column,0.0);
    for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                ss >> c;
                X(i,j) = c;
            }        
        }
    return X;
}

inline Matrix readshots(string mat)
{
    int column;
    double c;
    stringstream ss(mat);
    ss >> column;
    //cout<<column<<endl;
    Matrix X(1,column,0.0);
    for (int j = 0; j < column; j++)
        {
            ss >> c;
            //cout<<c<<endl;
            X(0,j) = c;
        }        
    return X;
}

// does matrices multiplication where instead of sum, max is used. Index are the indexes needed for the viterbi algo.
inline Matrix maxmul(Matrix A, Matrix B,Matrix &Index)
{
    // A is a row vector and B is a matrix
    // it multiplies A with B where instead of addition
    // it takes the max summand. 

    // cout<<"state vor: ";
    // Index.print2();
    Matrix multip(A.getRows(),B.getCols(),0.0);
    Matrix index(A.getRows(),B.getCols(),0.0);
    if(A.getCols() == B.getRows())
    {
        int i,j,k;
        double temp = 0.0, x=0.0, temp2=0.0;
        for (i = 0; i < A.getRows(); i++)
        {
            for (j = 0; j < B.getCols(); j++)
            {
                temp = 0.0;
                temp2= 0;
                for (k = 0; k < A.getCols(); k++)
                {   
                    x=A(i,k) * B(k,j);
                    if (x>temp)
                    {
                        temp = x;
                        temp2 = k;
                    }
                     
                }
                multip(i,j) = temp;
                Index(i,j) = temp2;
            }
        }
            // cout<<"state nach: ";
            // Index.print2(); cout<<endl;
            return multip;
    }
}

// given transitionsmatrix A, emission matrix B, initial state PI and the ibservations O it returns the alfa matrix     
inline Matrix alfa_pass(Matrix A, Matrix B, Matrix PI, Matrix O)
{
    Matrix alfa(O.getCols(),A.getRows(),0.0);

    alfa.insert_row(0, PI^B.return_column(O(0,0)).transpose());


    for (int i = 1; i < O.getCols(); i++)
    {
        // cout<<i<<" ";
        // ((alfa.return_row(i-1)*A)^B.return_column(O(0,i)).transpose()).print();
        alfa.insert_row(i,(alfa.return_row(i-1)*A)^B.return_column(O(0,i)).transpose());
    }
    return alfa;
}

inline Matrix beta_pass(Matrix A, Matrix B, Matrix PI, Matrix O)
{
    Matrix beta(O.getCols(), A.getRows(), 1.0);
    for (int i = O.getCols()-2; i >=0; i--)
    {
        beta.insert_row(i,( A* (beta.return_row(i+1).transpose()^B.return_column(O(0,i+1))) ).transpose()  );
    }
    return beta;
}

inline tuple<Matrix, Matrix, Matrix> train(Matrix A, Matrix B, Matrix PI, Matrix O, int max_it)
{

    int N = A.getRows();
    int T = O.getCols();
    int M = B.getCols();
    
    Matrix alfa(T,N,1);
    Matrix c(1,T,1);
    Matrix beta(T,N,1);
    vector <Matrix> gama_1(T-1, Matrix(N, N,0));
    Matrix gama_2( T, N, 1 );
    double denom = 0;
    double numer = 0;
    double log_prob;
    double old_log_prob =-99999;
    int max_iters = max_it;
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
            c(0,0) = 1/c(0,0);
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
                c(0,t) = 1/c(0,t);
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
                    A(i,j)= numer/denom;
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
                    B(i,j)= numer/denom;
                }
            }

        // compute log
            log_prob=0;
            for (int i = 0; i < T; i++)
            {
                log_prob = log_prob + log10(c(0,i));
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
                return make_tuple(A,B,PI);
            }
    }
   
}




#endif
