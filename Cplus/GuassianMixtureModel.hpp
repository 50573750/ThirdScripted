#ifndef Experiments_GuassianMixtureModel_hpp
#define Experiments_GuassianMixtureModel_hpp

#include "Library.hpp"

class GuassianMixtureModel
{
protected:
    const MatrixXd &    dataset;
    const unsigned &    cnt_mixed_factor;
    
public:
    vector<MatrixXd>    covariance;
    vector<VectorXd>    expectation;
    vector<double>      factor;
    
public:
    GuassianMixtureModel(const MatrixXd& dataset, const unsigned& cnt_mixed_factor)
    :dataset(dataset), cnt_mixed_factor(cnt_mixed_factor), factor(cnt_mixed_factor),
    expectation(cnt_mixed_factor), covariance(cnt_mixed_factor)
    {
        for_each(covariance.begin(), covariance.end(), [=](MatrixXd& elem){elem.resize(dataset.cols(), dataset.cols());});
        for_each(expectation.begin(), expectation.end(), [=](VectorXd& elem){elem.resize(dataset.cols());});
    }
    
    void run(unsigned epos)
    {
        for_each(covariance.begin(), covariance.end(),
                 [=](MatrixXd& elem){elem = MatrixXd::Identity(dataset.cols(), dataset.cols());});
        for_each(expectation.begin(), expectation.end(),
                 [=](VectorXd& elem){elem = VectorXd::Random(dataset.cols());});
        for_each(factor.begin(), factor.end(),
                 [=](double& elem){elem = rand()%100/100.0 + 0.1;});
        
        while(epos--)
        {
            MatrixXd prob_item(dataset.rows(), cnt_mixed_factor);
            for(auto cnt_i=0; cnt_i<dataset.rows(); ++cnt_i)
            {
                for(auto cnt_m=0; cnt_m<cnt_mixed_factor; ++cnt_m)
                {
                    prob_item(cnt_i, cnt_m) =
                            factor[cnt_m] / sqrt(covariance[cnt_m].determinant())
                            * exp(-((dataset.row(cnt_i).transpose() - expectation[cnt_m]).transpose()
                            * covariance[cnt_m]
                            * (dataset.row(cnt_i).transpose() - expectation[cnt_m]))(0,0)/2);
                }
            }
            
            for_each(covariance.begin(), covariance.end(),
                     [=](MatrixXd& elem){elem = MatrixXd::Zero(dataset.cols(), dataset.cols());});
            for_each(expectation.begin(), expectation.end(),
                     [=](VectorXd& elem){elem = VectorXd::Zero(dataset.cols());});
            
            for(auto cnt_m=0; cnt_m<cnt_mixed_factor; ++cnt_m)
            {
                for(auto cnt_i=0; cnt_i<dataset.rows(); ++cnt_i)
                {
                    expectation[cnt_m] += prob_item(cnt_i, cnt_m) / (prob_item.col(cnt_m).sum())
                    * dataset.row(cnt_i).transpose();
                }
            }
            
            for(auto cnt_m=0; cnt_m<cnt_mixed_factor; ++cnt_m)
            {
                for(auto cnt_i=0; cnt_i<dataset.rows(); ++cnt_i)
                {
                    covariance[cnt_m] += prob_item(cnt_i, cnt_m) / (prob_item.col(cnt_m).sum())
                    * (dataset.row(cnt_i).transpose() - expectation[cnt_m])
                    * (dataset.row(cnt_i).transpose() - expectation[cnt_m]).transpose();
                }
            }
            
            for(auto cnt_m=0; cnt_m<cnt_mixed_factor; ++cnt_m)
            {
                factor[cnt_m] = prob_item.col(cnt_m).sum() / prob_item.sum();
            }
        }
    }
};

#endif
