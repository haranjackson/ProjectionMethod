#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <vector>

#include "lib/Eigen327/Dense"
#include "lib/Eigen327/Sparse"


typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;


void print(Mat A);
void print (SpMat A);
Mat vconcat(Mat A, Mat B, Mat C);
Mat hconcat(Mat A, Mat B, Mat C);
Mat avg(Mat A);
SpMat kron(SpMat A, SpMat B);
SpMat speye(int n);
Mat diff(Mat X);
Mat reshape(Mat A, int m, int n);
std::vector<std::vector<double> > conrec(Mat & Q, Vec x, Vec y,
                                         std::vector<double> z);


#endif // MATRIXOPS_H

