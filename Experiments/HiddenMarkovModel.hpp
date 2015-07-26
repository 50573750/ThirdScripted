#ifndef Experiments_HiddenMarkovModel_hpp
#define Experiments_HiddenMarkovModel_hpp

#include "Library.hpp"

class HiddenMarkovModel
{
protected:
    MatrixXd        translations;
    MatrixXd        obversations;
    const unsigned  n_state;
    const unsigned  n_obversation;
    
public:
    HiddenMarkovModel(unsigned n_state, unsigned n_obversation)
    :translations(n_state, n_state), obversations(n_obversation, n_obversation),
    n_state(n_state), n_obversation(n_obversation)
    {
        ;
    }
    
    double probability(const vector<unsigned>& sequence)
    {
        MatrixXd prob(1, n_state);
        prob = MatrixXd::Ones(1, n_state);
        
        for(auto pos=0; pos<sequence.size(); ++pos)
        {
            prob = prob * translations;
            prob = prob.cwiseProduct(obversations.row(sequence[pos]));
        }
        
        return prob.maxCoeff();
    }
};

#endif
