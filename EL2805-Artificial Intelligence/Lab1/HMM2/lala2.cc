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


using std::vector;
using std::tuple;
using namespace std;


class Matrix {
    private:
        unsigned m_rowSize;
        unsigned m_colSize;
        vector<vector<double> > m_matrix;
    public:
    Matrix(unsigned rowSize, unsigned colSize, double initial)
    {
        m_rowSize = rowSize;
        m_colSize = colSize;
        m_matrix.resize(rowSize);
        for (unsigned i = 0; i < m_matrix.size(); i++)
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

    ~Matrix()
    {
    }



    Matrix operator+(Matrix &B)
    {
        Matrix sum(m_colSize, m_rowSize, 0.0);
        unsigned i,j;
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
        unsigned i,j;
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
        unsigned i,j;
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
    Matrix operator*(Matrix & B)
    {
        Matrix multip(m_rowSize,B.getCols(),0.0);
        if(m_colSize == B.getRows())
        {
            unsigned i,j,k;
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
        unsigned i,j;
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
        unsigned i,j;
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
        unsigned i,j;
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
        unsigned i,j;
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
    double& operator()(const unsigned &rowNo, const unsigned & colNo)
    {
        return this->m_matrix[rowNo][colNo];
    }

    // No brainer - returns row #
    unsigned getRows() const
    {
        return this->m_rowSize;
    }

    // returns col #
    unsigned getCols() const
    {
        return this->m_colSize;
    }

    // Take any given matrices transpose and returns another matrix
    Matrix transpose()
    {
        Matrix Transpose(m_colSize,m_rowSize,0.0);
        for (unsigned i = 0; i < m_colSize; i++)
        {
            for (unsigned j = 0; j < m_rowSize; j++) {
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
                cout << m_matrix[i][j] << " ";
            }
            cout<<endl;
        }
        // cout<<endl;
    }

    // Returns 3 values //First: Eigen Vector //Second: Eigen Value //Third: Flag
    tuple<Matrix, double, int> powerIter(unsigned rowNum, double tolerance){
        // Picks a classic X vector
        Matrix X(rowNum,1,1.0);
        // Initiates X vector with values 1,2,3,4
        for (unsigned i = 1; i <= rowNum; i++) {
            X(i-1,0) = i;
        }
        int errorCode = 0;
        double difference = 1.0; // Initiall value greater than tolerance
        unsigned j = 0;
        unsigned location;
        // Defined to find the value between last two eigen values
        vector<double> eigen;
        double eigenvalue = 0.0;
        eigen.push_back(0.0);
        
        while(abs(difference) > tolerance) // breaks out when reached tolerance
        {
            j++;
            // Normalize X vector with infinite norm
            for (int i = 0; i < rowNum; ++i)
            {
                eigenvalue = X(0,0);
                if (abs(X(i,0)) >= abs(eigenvalue))
                {
                    // Take the value of the infinite norm as your eigenvalue
                    eigenvalue = X(i,0);
                    location = i;
                }
            }
            if (j >= 5e5) {
                cout << "Oops, that was a nasty complex number wasn't it?" << endl;
                cout << "ERROR! Returning code black, code black!";
                errorCode = -1;
                return make_tuple(X,0.0,errorCode);
            }
            eigen.push_back(eigenvalue);
            difference = eigen[j] - eigen[j-1];
            // Normalize X vector with its infinite norm
            X = X / eigenvalue;
            
            // Multiply The matrix with X vector
            X = (*this) * X;
        }
        
        // Take the X vector and what you've found is an eigenvector!
        X = X / eigenvalue;
        return make_tuple(X,eigenvalue,errorCode);
    }

    Matrix deflation(Matrix &X, double &eigenvalue)
    {
        // Deflation formula exactly applied
        double denominator = eigenvalue / (X.transpose() * X)(0,0);
        Matrix Xtrans = X.transpose();
        Matrix RHS = (X * Xtrans);
        Matrix RHS2 = RHS * denominator;
        Matrix A2 = *this - RHS2;
        return A2;
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
        for (unsigned i = 0; i < m_colSize; i++)
        {
            for (unsigned j = 0; j < m_rowSize; j++) {
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
        for (unsigned i = 0; i < m_rowSize; i++)
        {
            for (unsigned j = 0; j < m_colSize; j++) {
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
        for (unsigned i = 0; i < m_rowSize; i++)
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
        for (unsigned i = 0; i < m_rowSize; i++)
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


Matrix create_matrix(string mat)
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

Matrix readshots(string mat)
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
Matrix maxmul(Matrix A, Matrix B,Matrix &Index)
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
        unsigned i,j,k;
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

// returns the most propable state sequence(maximized probability) on the first matrix, second matrix is the delta matrix and third matrix are the indexes
tuple<Matrix, Matrix, Matrix> viterbi(Matrix A, Matrix B, Matrix PI, Matrix O)
{
    Matrix delta(O.getCols(),A.getRows(),0.0);
    Matrix indexes(O.getCols(),A.getRows(),1);


    Matrix temp(1,A.getRows(),0.0);


    delta.insert_row(0, PI^B.return_column(O(0,0)).transpose());



    for (int i = 1; i < O.getCols(); i++)
    {
        // cout<<i<<" ";
        // ((delta.return_row(i-1)*A)^B.return_column(O(0,i)).transpose()).print();
        
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
    return make_tuple(states, delta,indexes);
}

int main(int argc, char * argv[]) {

 string matrixA;
 string matrixB;
 string matrixPI;
 string shootsS;

 getline(cin,matrixA);
 getline(cin,matrixB);
 getline(cin,matrixPI);
 getline(cin,shootsS);



Matrix A(create_matrix(matrixA));
Matrix B(create_matrix(matrixB));
Matrix PI(create_matrix(matrixPI));
Matrix O(readshots(shootsS));


// solution
auto x = viterbi( A,  B,  PI,  O);
get<0>(x).print();

}




