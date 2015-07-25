#ifndef Experiments_IsingModel_hpp
#define Experiments_IsingModel_hpp

#include "Library.hpp"

class IsingModelNaiveMean
{
protected:
    MatrixXd            expections;
    const MatrixXd&     image;
    
public:
    IsingModelNaiveMean(const MatrixXd& image)
    :image(image)
    {
        expections.resize(image.rows(), image.cols());
        for(auto i=0; i<image.rows(); ++i)
        {
            for(auto j=0; j<image.cols(); ++j)
            {
                expections(i,j) = rand()%200/100.0 - 1.0;
            }
        }
    }
    
    const MatrixXd& result() const
    {
        return expections;
    }
    
    void run(int epos, double factor_guassian_generator = 1.0, double factor_smoothing = 1.0)
    {
        auto f_label_positive = [=](double real)
        {
            return -factor_guassian_generator * (real - 1) * (real - 1);
        };
        
        auto f_label_nagtive = [=](double real)
        {
            return -factor_guassian_generator * (real + 1) * (real + 1);
        };
        
        
        while(epos --> 0)
        {
            MatrixXd expections_tmp = expections;
            for(auto i=1; i<image.rows()-1; ++i)
            {
                for(auto j=1; j<image.cols()-1; ++j)
                {
                    double m = 0.0;
                    m += expections_tmp(i-1 ,j);
                    m += expections_tmp(i+1 ,j);
                    m += expections_tmp(i ,j-1);
                    m += expections_tmp(i ,j+1);
                    
                    expections(i, j) = tanh(2 * factor_smoothing * m
                                            + f_label_positive(image(i, j)) - f_label_nagtive(image(i,j)));
                }
            }
        }
        
        for(auto i=1; i<image.rows()-1; ++i)
        {
            for(auto j=1; j<image.cols()-1; ++j)
            {
                expections(i,j) = expections(i, j) > 0? +1 : -1;
            }
        }
    }
};

class IsingModelGibbsSampling
{
protected:
    const MatrixXd&     image;
    MatrixXd            sampings;
    
public:
    IsingModelGibbsSampling(const MatrixXd& image)
    :image(image)
    {
        sampings = image;
    }
    
    const MatrixXd& result() const
    {
        return sampings;
    }
    
    void run(int epos, double factor_guassian_generator = 1.0)
    {
        auto f_label_positive = [=](double real)
        {
            return -factor_guassian_generator * (real - 1) * (real - 1);
        };
        
        auto f_label_nagtive = [=](double real)
        {
            return -factor_guassian_generator * (real + 1) * (real + 1);
        };
        
        while(epos --> 0)
        {
            for(auto i=1; i<image.rows()-1; ++i)
            {
                for(auto j=1; j<image.cols()-1; ++j)
                {
                    double prob = 0;
                    prob += sampings(i-1, j);
                    prob += sampings(i+1, j);
                    prob += sampings(i, j-1);
                    prob += sampings(i, j+1);
                    
                    double prob_uniformed = rand()%1000/1000.0;
                    double prob_positive = exp(prob + f_label_positive(image(i,j)));
                    double prob_nagtive = exp(-prob + f_label_nagtive(image(i,j)));
                    
                    if (prob_uniformed <= prob_positive/(prob_positive+prob_nagtive))
                    {
                        sampings(i, j) = 1;
                    }
                    else
                    {
                        sampings(i, j) = -1;
                    }
                }
            }
        }
    }
};

class IsingModelCollapsedSampling
{
protected:
    const MatrixXd&     image;
    MatrixXd            expections;
    MatrixXd            samplings;
    
public:
    IsingModelCollapsedSampling(const MatrixXd& image)
    :image(image), expections(image), samplings(image)
    {
        ;
    }
    
    const MatrixXd& result() const
    {
        return samplings;
    }
    
    void run(int epos, double factor_guassian_generator = 1.0, double factor_smoothing = 1.0)
    {
        auto f_label_positive = [=](double real)
        {
            return -factor_guassian_generator * (real - 1) * (real - 1);
        };
        
        auto f_label_nagtive = [=](double real)
        {
            return -factor_guassian_generator * (real + 1) * (real + 1);
        };
        
        while (epos --> 0)
        {
            MatrixXd expections_tmp = expections;
            for(auto i=1; i<image.rows()-1; ++i)
            {
                for(auto j=1; j<image.cols()-1; ++j)
                {
                    double m = 0.0;
                    m += expections_tmp(i-1 ,j);
                    m += expections_tmp(i+1 ,j);
                    m += expections_tmp(i ,j-1);
                    m += expections_tmp(i ,j+1);
                    
                    expections(i, j) = tanh(2 * factor_smoothing * m
                                            + f_label_positive(image(i, j)) - f_label_nagtive(image(i,j)));
                    
                    double prob_uniformed = rand()%1000/1000.0;
                    double prob_positive = exp(m + f_label_positive(image(i,j)));
                    double prob_negative = exp(-m + f_label_nagtive(image(i,j)));
                    
                    if (prob_uniformed <= prob_positive/(prob_positive + prob_negative))
                    {
                        samplings(i, j) = +1;
                    }
                    else
                    {
                        samplings(i, j) = -1;
                    }
                }
            }
        }
    }
};

#endif
