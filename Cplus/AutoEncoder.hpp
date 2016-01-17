#ifndef Experiments_AutoEncoder_hpp
#define Experiments_AutoEncoder_hpp

#include "Library.hpp"
#include "FeadForwardNeuralNetworkBackPropergation.hpp"

class AutoEncoder
{
protected:
    FeedForwardNeualNetwokBackPropergation  ffnnbp;
    unsigned                                dimension;
    unsigned                                layers;
    
public:
    AutoEncoder(unsigned dimension)
    :dimension(dimension), ffnnbp(dimension, dimension), layers(2)
    {
        ;
    }
    
    MatrixXd infer(const VectorXd& dataitem)
    {
        return ffnnbp.infer(dataitem);
    }
    
    MatrixXd infer_represent(const VectorXd& dataitem)
    {
        return ffnnbp.infer(dataitem, layers/2);
    }
    
    void add_layer(int number)
    {
        ++ layers;
        ffnnbp.add_hidden_layer(number);
    }
    
    void train(const MatrixXd& dataset, int epos, double alpha, double noise)
    {
        vector<pair<VectorXd, VectorXd>>    pairs;
        for(auto i=0; i<dataset.rows(); ++i)
        {
            pairs.push_back(make_pair(dataset.row(i).transpose(),
                                      dataset.row(i).transpose() + MatrixXd::Random(dataset.cols(), 1)*noise));
        }
        
        ffnnbp.train(pairs, epos, alpha);
    }
};

#endif
