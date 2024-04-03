#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
using namespace std;

template <class T>
class Matrix {
//private:

    /*vector<vector<T>> data;
    int rows, cols;*/
 
public:
    vector<vector<T>> data;
    int rows, cols;
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data.resize(rows, vector<T>(cols, 0));
    }

    Matrix(vector<vector<T>> const& input) {
        data = input;
        rows = input.size();
        cols = input[0].size();
    }

    int getRows() const {
        return rows;
    }

    int getCols() const {
        return cols;
    }
    const vector<vector<T>>& getData() const {
        return data;
    }

    void swap(vector<vector<T>>& data, int row1, int row2, int col) {
        for (int i = 0; i < col; i++) {
            T tmp = data[row1][i];
            data[row1][i] = data[row2][i];
            data[row2][i] = tmp;
        }
    }
    int factorial(int n) {
        if (n == 0 || n == 1)
            return 1;
        else
            return n * factorial(n - 1);

    }

    static Matrix identityMatrix(int size) {
        Matrix identity(size, size);
        for (int i = 0; i < size; ++i) {
            identity.data[i][i] = 1;
        }
        return identity;
    }

    void printMatrix()  {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << data[i][j] << "\t";
            }
            cout << endl;
        }
    }

    double matrixNorm()  {
        double sum = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sqrt(sum);
    }

    int matrixRank() {
        int rank = cols;

        for (int row = 0; row < rank; row++) {
            // Before we visit current row 'row', we make
            // sure that data[row][0],....data[row][row-1] are 0.

            // CASE -> 1 -- Diagonal element != zero
            if (data[row][row]) {
                for (int col = 0; col < rows; col++) {
                    if (col != row) {
                        //  makes all entries of current column as 0 except entry 'mat[row][row]'

                        double m = (double)data[col][row] / data[row][row];
                        for (int i = 0; i < rank; i++) {
                            data[col][i] -= m * data[row][i];
                        }
                    }
                }
            }

            // Diagonal element == zero. 2 cases-->
            // CASE ->1
            // If there is a row below it with non-zero
            //entry, then swap this row with that row
            //and process that row
            //CASE ->2
            //If all elements in current column below
            //mat[r][row] are 0, then remove this column
            //by swapping it with last column and
            //reducing number of columns by 1.
            else {
                bool reduce = true;

                /* Find the non-zero element in current
                    column  */
                for (int i = row + 1; i < rows; i++) {
                    // Swap the row with non-zero element
                    // with this row.
                    if (data[i][row]) {
                        swap(data, row, i, rank);
                        reduce = false;
                        break;
                    }
                }
                // If we did not find any row with non-zero
                // element in current column, then all
                // values in this column are 0.
                if (reduce) {
                    // Reduce number of columns
                    rank--;

                    // Copy the last column here
                    for (int i = 0; i < rows; i++)
                        data[i][row] = data[i][rank];
                }

                // Process this row again
                row--;
            }

            // Uncomment these lines to see intermediate results
            // display(data, R, C);
            // printf("\n");
        }
        return rank;
    }
    T& operator()(int row, int col) {
        return data[row][col];
    }

    const T& operator()(int row, int col) const {
        return data[row][col];
    }


    Matrix<double> matrixExponential(int n) {
        int size = rows;
        Matrix<double> result(size, size);

        for (int i = 0; i < size; ++i) {
            result(i, i) = 1;
        }

        Matrix<double> temp(size, size);

        for (int k = 1; k < n; ++k) {
            temp = result; 

            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    double sum = 0.0;
                    for (int m = 0; m < size; ++m)
                        sum += temp(i, m) * data[m][j] / factorial(k);
                    result(i, j) += sum; 
            }
        }

        return result;
    }


    Matrix<T> matrixMultiplication( Matrix<T>& other)  {
        if (cols != other.rows) {
            cerr << "Invalid dimensions " << endl;
            return Matrix<T>(0, 0); 
        }

        int rRows = rows;
        int rCols = other.cols;
        Matrix<T> result(rRows, rCols);

        for (int i = 0; i < rRows; ++i) {
            for (int j = 0; j < rCols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    
    Matrix<double> matrixLogarithm(int n) {
        Matrix<double> expMatrix = matrixExponential(n);


        Matrix<double> logMatrix(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                logMatrix.data[i][j] = log(expMatrix.data[i][j]);
            }
        }

        return logMatrix;
    }
    double myMax(double a, double b) {
        return (a > b) ? a : b;
    }
    double myMin(double a, double b) {
        return (a < b) ? a : b;
    }
    double matrixConditioning()  {
     
        Matrix<double> inv(rows, cols);
        for (int i = 0; i < min(rows, cols); ++i) {
            if (data[i][i] != 0)
                inv.data[i][i] = 1.0 / data[i][i];
            else
                inv.data[i][i] = 0;
        }

        double maxValue = 0.0;
        double minValue = numeric_limits<double>::max();

        for (int i = 0; i < min(rows, cols); ++i) {
            maxValue = myMax(maxValue, data[i][i]);
            minValue = myMin(minValue, data[i][i]);
        }

        return maxValue / minValue;
    }
    bool operator==( Matrix& other)  {
        // Check if the dimensions of the matrices are the same
        if (rows != other.rows || cols != other.cols) {
            return false;
        }

        // Compare each element of the matrices
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (data[i][j] != other.data[i][j]) {
                    return false;
                }
            }
        }

        // If all elements are equal, return true
        return true;
    }


    bool isOrthogonal( Matrix<T>& matrix)  {
         //Check  is square ???
        if (matrix.getRows() != matrix.getCols()) {
            cout << "Matrix is not square, hence not orthogonal." << endl;
            return false;
        }

        // Compute transpose 
        Matrix<T> transpose(matrix.getCols(), matrix.getRows());
        for (int i = 0; i < matrix.getRows(); ++i) {
            for (int j = 0; j < matrix.getCols(); ++j) {
                transpose.data[j][i] = matrix.data[i][j];
            }
        }

        // Compute the product 
        Matrix<T> p = matrix.matrixMultiplication(transpose);

        // Get an identity 
        Matrix<T> identity = identityMatrix(matrix.getRows());

        // Check if the product is equal to the identity matrix
        if (p == identity) {
            cout << "Matrix is orthogonal." << endl;
            return true;
        }
        else {
            cout << "Matrix is not orthogonal." << endl;
            return false;
        }
    }

    Matrix<T> matrixConvolution( Matrix<T>& kernel)  {
        int kernelRows = kernel.getRows();
        int kernelCols = kernel.getCols();
        int resultRows = rows - kernelRows + 1;
        int resultCols = cols - kernelCols + 1;

        if (resultRows <= 0 || resultCols <= 0) {
            cerr << "Invalid kernel size for convolution." << endl;
            return Matrix<T>(0, 0);
        }

        Matrix<T> result(resultRows, resultCols);

        for (int i = 0; i < resultRows; ++i) {
            for (int j = 0; j < resultCols; ++j) {
                T convolutionSum = 0;
                for (int ki = 0; ki < kernelRows; ++ki) {
                    for (int kj = 0; kj < kernelCols; ++kj) {
                        convolutionSum += data[i + ki][j + kj] * kernel.data[ki][kj];
                    }
                }
                result.data[i][j] = convolutionSum;
            }
        }

        return result;
    }
    // Implement the sum functionality to calculate the sum of all elements
    T sum()  {
        T total = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                total += data[i][j];
            }
        }
        return total;
    }

    // Implement the block functionality using Eigen
    Matrix<T> block(int startRow, int startCol, int blockRows, int blockCols) const {
        Matrix<T> result(blockRows, blockCols);
        for (int i = 0; i < blockRows; ++i) {
            for (int j = 0; j < blockCols; ++j) {
                result(i, j) = data[startRow + i][startCol + j];
            }
        }
        return result;
    }
    // Implement the cwiseProduct functionality using Eigen
    Matrix<T> cwiseProduct( Matrix<T>& other)  {
        Matrix<T> result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * other(i, j);
            }
        }
        return result;
    }
    Matrix<T> Convolution ( Matrix<T>& input,  Matrix<T>& kernel) {
         int kernel_rows = kernel.getRows();
         int kernel_cols = kernel.getCols();
         int rows = input.getRows() - kernel_rows + 1;
         int cols = input.getCols() - kernel_cols + 1;

        Matrix<T> result(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Matrix<T> block = input.block(i, j, kernel_rows, kernel_cols);
                result(i, j) = (block.cwiseProduct(kernel)).sum();
            }
        }

        return result;


        };




};
