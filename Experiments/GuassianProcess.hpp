#ifndef Experiments_GuassianProcess_hpp
#define Experiments_GuassianProcess_hpp

#include "Library.hpp"

class GuassianProcessExact
{
protected:
    MatrixXd                samplings;
    const MatrixXd          datasets;
    
public:
    GuassianProcessExact(const MatrixXd datasets, const MatrixXd labels)
    :datasets(datasets), samplings(datasets.rows(), datasets.rows())
    {
        for(auto i=0; i<datasets.rows(); ++i)
        {
            for(auto j=0; j<datasets.rows(); ++j)
            {
                samplings(i, j) = (datasets.row(i) * datasets.row(j).transpose())(0,0)
                / datasets.row(i).norm() /datasets.row(j).norm();
            }
        }
        samplings = samplings * labels.transpose();
    }
    
    MatrixXd result(const MatrixXd inferitem)
    {
        MatrixXd result_regression(inferitem.rows(), datasets.rows());
        for(auto i=0; i<datasets.rows(); ++i)
        {
            for(auto j=0; j<inferitem.rows(); ++j)
            {
                result_regression(i,j) = (datasets.row(i) * inferitem.row(j).transpose())(0,0)
                / datasets.row(i).norm() /inferitem.row(j).norm();
            }
        }
        
        return result_regression * samplings;
    }
};

#endif
