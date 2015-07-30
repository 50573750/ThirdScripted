#ifndef Experiments_Library_hpp
#define Experiments_Library_hpp

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <Eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

void random_probablistic_matrix(MatrixXd& mat)
{
    for(auto i=0; i<mat.rows(); ++i)
    {
        for(auto j=0; j<mat.cols(); ++j)
        {
            mat(i, j) = rand()%1000/1000.0;
        }
    }
}

void normalize_conditional_probablistic_matrix(MatrixXd& mat)
{
    for(auto i=0; i<mat.cols(); ++i)
    {
        mat.col(i) /= mat.col(i).sum();
    }
}

#endif
