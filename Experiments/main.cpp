#include "Library.hpp"
#include "Tests.hpp"

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
    
    void viterbi_decoding(const vector<unsigned>& sequence, vector<unsigned>& states)
    {
        MatrixXi record_state(sequence.size(), n_state);
        
        VectorXd prob(n_state);
        prob = VectorXd::Ones(n_state);
        
        for(auto pos=0; pos<sequence.size(); ++pos)
        {
            VectorXd prob_tmp(n_state);
            for(auto i=0; i<n_state; ++i)
            {
                double      prob_max = -1;
                unsigned    state_max = 0;
                for(auto j=0; j<n_state; ++j)
                {
                    if (prob(j) * translations(j, i) * obversations(sequence[pos], j) >  prob_max)
                    {
                        prob_max = prob(j) * translations(j, i) * obversations(sequence[pos], j);
                        state_max = j;
                    }
                }
                record_state(pos, i) = state_max;
                prob_tmp(i) = prob_max;
            }
            
            prob = prob_tmp;
        }
        
        unsigned & last = states[sequence.size()-1];
        last = 0;
        for(auto i=0; i<n_state; ++i)
        {
            if (prob(last) < prob(i))
            {
                last = i;
            }
        }
        
        states.resize(sequence.size());
        for(auto pos=sequence.size()-2; pos>=0; ++pos)
        {
            states[pos] = record_state(pos, states[pos+1]);
        }
    }
    
    void baum_welch_training(const vector<vector<unsigned>>& dataset_sequence)
    {
        MatrixXd translations_trained = MatrixXd::Zero(n_state, n_state);
        MatrixXd obversations_trained = MatrixXd::Zero(n_obversation, n_state);
        for(auto seq=dataset_sequence.begin(); seq != dataset_sequence.end(); ++ seq)
        {
            MatrixXd prob_forward(seq->size() + 1, n_state);
            prob_forward.row(0) = MatrixXd::Ones(1, n_state);
            for(auto pos=0; pos < seq->size(); ++ pos)
            {
                prob_forward.row(pos+1) = prob_forward.row(pos-1) * translations;
                prob_forward.row(pos+1) = prob_forward.row(pos).cwiseProduct(obversations.row(seq->at(pos)));
            }
            
            MatrixXd prob_backward(seq->size() + 1, n_state);
            prob_backward.row(seq->size()) = MatrixXd::Ones(1, seq->size());
            for(auto pos=seq->size()-1; pos>=0; -- pos)
            {
                prob_backward.row(pos) = prob_backward.row(pos+1) * translations.transpose();
                prob_backward.row(pos) = prob_backward.row(pos).cwiseProduct(obversations.row(seq->at(pos)));
            }
            
            for(auto pos=0; pos<seq->size()-1; ++pos)
            {
                for(auto src=0; src<n_state; ++src)
                {
                    for(auto des=0; des<n_state; ++des)
                    {
                        translations_trained(src, des) += prob_forward(pos-1, src) * translations(src, des)
                        * obversations(seq->at(pos), des) * prob_backward(pos + 1, des);
                    }
                    
                    obversations(seq->at(pos), src) += prob_forward(pos-1, src) * prob_backward(pos, src);
                }
            }
        }
    }
};

int main()
{
    srand(time(nullptr));
    cout<<"Begin"<<endl;
    
    cout<<"End"<<endl;
    
    return 0;
}