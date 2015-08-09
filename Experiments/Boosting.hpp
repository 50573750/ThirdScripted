#ifndef Experiments_Boosting_hpp
#define Experiments_Boosting_hpp

#include "Library.hpp"

class WLMS
{
protected:
    MatrixXd            linear;
    MatrixXd            sample_weight;
    
public:
    WLMS(const MatrixXd & dataset, const MatrixXd & label, const MatrixXd& weight, double sigma_noise = 0.1)
    :sample_weight(weight.cols(), weight.cols())
    {
        for(auto i=0; i<weight.cols(); ++i)
        {
            sample_weight(i, i) = weight(i);
        }
        
        linear = (dataset.transpose() * sample_weight * dataset
                  + sigma_noise * MatrixXd::Identity(dataset.cols(), dataset.cols())).inverse()
        * dataset.transpose() * sample_weight * label;
    }
    
    double infer(const MatrixXd & test_item)
    {
        return (test_item * linear)(0, 0) >= 0 ? +1 : -1;
    }
};

class AdaptiveBoosting
{
protected:
    vector<pair<WLMS*, double>>  boosting_item;
    const MatrixXd & dataset;
    const MatrixXd & label;
public:
    AdaptiveBoosting(const MatrixXd & dataset, const MatrixXd & label)
    :dataset(dataset), label(label)
    {
        ;
    }
    
    void train(unsigned n_boosting_item, double sigma_noise = 0.1)
    {
        MatrixXd    sample_weight(1, dataset.rows());
        sample_weight = MatrixXd::Ones(1, dataset.rows());
        sample_weight /= sample_weight.sum();
        
        while (n_boosting_item --> 0)
        {
            WLMS* new_boosting_item = new WLMS(dataset, label, sample_weight, sigma_noise);
            double error_rate = 1e-5;
            for(auto i=0; i<dataset.rows(); ++i)
            {
                if (label(i) * new_boosting_item->infer(dataset.row(i)) < 0)
                {
                    error_rate += sample_weight(i);
                }
            }
            
            double update_rate = log((1-error_rate)/error_rate);
            boosting_item.push_back(make_pair(new_boosting_item, update_rate));
            
            for(auto i=0; i<dataset.rows(); ++i)
            {
                sample_weight(i) *= exp(-update_rate * label(i) * new_boosting_item->infer(dataset.row(i)));
            }
            sample_weight /= sample_weight.sum();
        }
    }
    
    ~AdaptiveBoosting()
    {
        for(auto i=boosting_item.begin(); i!=boosting_item.end(); ++i)
        {
            delete i->first;
        }
    }
    
    double infer(const MatrixXd& test_item)
    {
        double decide = 0;
        for(auto i=boosting_item.begin(); i!=boosting_item.end(); ++i)
        {
            decide += i->second * i->first->infer(test_item);
        }
        
        return decide >= 0 ? +1 : -1;
    }
};

#endif
