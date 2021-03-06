#include <stdio.h>
#include <fstream> // for file access
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

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

    // // Set operator
    // Matrix operator[](const unsigned &rowNo, const unsigned & colNo)
    // {
    //     this->m_matrix[rowNo][colNo]
    // }

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
    void print() const
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

    // Returns 3 values
    //First: Eigen Vector
    //Second: Eigen Value
    //Third: Flag
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
            cout<<endl;
        }
    return X;
}




using namespace std;

int main(int argc, char * argv[]) {
 string matrixA;
 string matrixB;
 string matrixPI;



 getline(cin,matrixA);
 getline(cin,matrixB);
 getline(cin,matrixPI);

Matrix A(create_matrix(matrixA));
Matrix B(create_matrix(matrixB));
Matrix PI(create_matrix(matrixPI));

(PI*A*B).print();


}




