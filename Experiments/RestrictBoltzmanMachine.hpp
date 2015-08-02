#ifndef Experiments_RestrictBoltzmanMachine_hpp
#define Experiments_RestrictBoltzmanMachine_hpp

#include "Library.hpp"

class RestrictBoltzmanMachineGibbisSampling
{
protected:
    const MatrixXd &        dataset;
    const unsigned          n_hidden_unit;
    MatrixXd                weights;
    MatrixXd                datagen_hidden;
    
public:
    RestrictBoltzmanMachineGibbisSampling(const MatrixXd & dataset, const unsigned & n_hidden_unit)
    :dataset(dataset), n_hidden_unit(n_hidden_unit)
    {
        weights = MatrixXd::Random(dataset.cols(), n_hidden_unit);
        datagen_hidden = MatrixXd::Random(dataset.rows(), n_hidden_unit);
    }
    
    const MatrixXd & result() const
    {
        return datagen_hidden;
    }
    
    const MatrixXd & get_weights() const
    {
        return weights;
    }
    
    void run(int epos, double alpha = 0.1, int cd_step=2)
    {
        auto sigm = [](double x) { return 1/(1 + exp(-x)); };
        
        while(epos --> 0)
        {
            for(auto cnt_sample=0; cnt_sample<dataset.rows(); ++cnt_sample)
            {
                MatrixXd sampled_hidden(1, n_hidden_unit);
                MatrixXd sampled_visual(1, dataset.cols());
                MatrixXd sampled_prob_hidden(1, n_hidden_unit);
                
                sampled_visual = dataset.row(cnt_sample);
                for(auto step=0; step<cd_step; ++step)
                {
                    sampled_hidden = sampled_visual * weights;
                    if (step == cd_step-1)
                    {
                        sampled_prob_hidden = sampled_hidden;
                    }
                    
                    for(auto elem=0; elem<sampled_hidden.cols(); ++elem)
                    {
                        sampled_prob_hidden(0, elem) = 1-sigm(sampled_prob_hidden(0,elem));
                        if (rand()%1000/1000.0 < sigm(sampled_hidden(0, elem)))
                        {
                            sampled_hidden(0, elem) = 0;
                        }
                        else
                        {
                            sampled_hidden(0, elem) = 1;
                        }
                    }
                    
                    sampled_visual = weights * sampled_hidden.transpose();
                    sampled_visual.transposeInPlace();
                    
                    for(auto elem=0; elem<sampled_visual.cols(); ++elem)
                    {
                        if (rand()%1000/1000.0 < sigm(sampled_visual(0, elem)))
                        {
                            sampled_visual(0, elem) = 0;
                        }
                        else
                        {
                            sampled_visual(0, elem) = 1;
                        }
                    }
                }
                
                datagen_hidden.row(cnt_sample) = sampled_hidden;
                weights += alpha * dataset.row(cnt_sample).transpose() * sampled_prob_hidden
                - alpha * sampled_visual.transpose() * sampled_hidden;
            }
        }
    }
};

class RestrictBoltzmanMachineNaiveMean
{
protected:
    const MatrixXd &        dataset;
    const unsigned          n_hidden_unit;
    MatrixXd                weights;
    MatrixXd                datagen_hidden;
    
public:
    RestrictBoltzmanMachineNaiveMean(const MatrixXd & dataset, const unsigned & n_hidden_unit)
    :dataset(dataset), n_hidden_unit(n_hidden_unit)
    {
        weights = MatrixXd::Random(dataset.cols(), n_hidden_unit);
        datagen_hidden = MatrixXd::Random(dataset.rows(), n_hidden_unit);
    }
    
    const MatrixXd & result() const
    {
        return datagen_hidden;
    }
    
    const MatrixXd & get_weights() const
    {
        return weights;
    }

    void run(int epos, double alpha=0.1)
    {
        auto sigm = [](double x) { return 1/(1 + exp(-x)); };
        MatrixXd    expections(dataset.rows(), n_hidden_unit);
        
        while(epos --> 0)
        {
            for(auto cnt_sample=0; cnt_sample<dataset.rows(); ++cnt_sample)
            {
                MatrixXd expection_item = dataset.row(cnt_sample) * weights;
                
                for(auto pos=0; pos<n_hidden_unit; ++pos)
                {
                    expections(cnt_sample, pos) = sigm(expection_item(0, pos));
                }
                
                weights += alpha * dataset.row(cnt_sample).transpose() * expections.row(cnt_sample)
                - alpha * weights;
            }
        }
        
        for(auto i=0; i<expections.rows(); ++i)
        {
            for(auto j=0; j<expections.cols(); ++j)
            {
                datagen_hidden(i, j) = expections(i, j) > 0.5 ? 1 : 0;
            }
        }
    }
};

#endif
