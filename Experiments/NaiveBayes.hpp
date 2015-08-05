#ifndef Experiments_NaiveBayes_hpp
#define Experiments_NaiveBayes_hpp

#include "Library.hpp"

class NaiveBayes
{
protected:
    const unsigned      n_feature;
    const unsigned      n_cate;
    MatrixXd            model;
    
public:
    NaiveBayes(unsigned n_feature, unsigned n_cate)
    :n_feature(n_feature), n_cate(n_cate), model(n_cate, n_feature)
    {
        ;
    }
    
    void train(const MatrixXd& dataset, const vector<unsigned>& label, double prior)
    {
        for(auto item=0; item<dataset.rows(); ++item)
        {
            for(auto feature=0; feature<n_feature; ++feature)
            {
                model(label[item], feature) += dataset(item, feature);
            }
        }
        
        model = model + prior*MatrixXd::Ones(model.rows(), model.cols());
    }
    
    long int infer(const VectorXd& dataitem)
    {
        vector<double>  scores(n_cate);
        for(auto fea=0; fea<n_feature; ++fea)
        {
            if (dataitem(fea) == 0)
                continue;
            
            for(auto cate=0; cate<n_cate; ++cate)
            {
                scores[cate] += log(model(cate, fea)) * dataitem(fea);
            }
        }
        
        return max_element(scores.begin(), scores.end()) - scores.begin();
    }
};

#endif
