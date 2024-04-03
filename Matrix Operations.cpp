#include <iostream>
#include <vector>
#include "matrixclass.h"

#include <algorithm> 


using namespace std;


int main() {
 /*   vector<vector<double>> matrix = { {1, 2}, {3, 4} };

    Matrix<double> mat(matrix);

    cout << "Matrix Norm: " << mat.matrixNorm() << endl;

    cout << "Matrix Rank: " << mat.matrixRank() << endl;

    Matrix<double> expMatrix = mat.matrixExponential(10);

    cout << "Matrix Exponential:" << endl;
    expMatrix.printMatrix();*/


    Matrix<int> m1(3, 3);

    vector<vector<int>> input = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    Matrix<int> m2(input);
    int rows = m2.getRows();
    int cols = m2.getCols();
     
    cout << "****************************************"<<endl;

    Matrix<int> identity = Matrix<int>::identityMatrix(3);
    identity.printMatrix();

    cout << "****************************************"<<endl;

    double norm = m2.matrixNorm();
    cout <<"Norm of M2 is : " << norm << endl;

    cout << "****************************************" << endl;

    Matrix<double> exp = m2.matrixExponential(3);
    cout << "matrixExponential" << endl;

    exp.printMatrix();

    cout << "****************************************" << endl;
    cout << "conv - 1" << endl;

    vector<vector<int>> kernel = { {1, 0, 1}, {0, 1, 0}, {1, 0, 1} };
    Matrix<int> convolutionResult = m2.matrixConvolution(kernel); 
    convolutionResult.printMatrix();

    cout << "****************************************" << endl;
    cout << "conv - 2" << endl;

    Matrix<int> convolutionResult2 = m2.Convolution(m2, kernel); 
    convolutionResult2.printMatrix();

    cout << "****************************************" << endl;
    int rank = m2.matrixRank(); 
    cout << "rank of M2 is : " << rank << endl;

    cout << "****************************************" << endl;

    Matrix<int> product = m2.matrixMultiplication(m2);
    cout << "matrixMultiplication" << endl;

    product.printMatrix();

    cout << "****************************************" << endl;

    m2.isOrthogonal(m2);

    cout << "****************************************" << endl;

    double condition = m2.matrixConditioning(); 
    cout << "condition of M2 is : " << condition << endl;

    cout << "****************************************" << endl;


    //Matrix<double> logMatrix = m2.matrixLogarithm(3); 
    //logMatrix.printMatrix();


    cout << "****************************************" << endl;







    return 0;
}


