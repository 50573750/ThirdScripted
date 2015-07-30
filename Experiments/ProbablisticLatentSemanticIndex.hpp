#ifndef Experiments_ProbablisticLatentSemanticIndex_h
#define Experiments_ProbablisticLatentSemanticIndex_h

#include "Library.hpp"

class ProbablisiticLatentSemanticIndex
{
protected:
    MatrixXd            prob_word_at_topic;
    MatrixXd            prob_document_at_topic;
    MatrixXd            prob_topic_at_word;
    MatrixXd            prob_topic_at_document;
    MatrixXd            prob_topic;
    const MatrixXd &    count_document_word;
    const unsigned      n_topic;
    
public:
    ProbablisiticLatentSemanticIndex(const MatrixXd& count_document_word, const unsigned n_topic)
    :prob_topic(n_topic, 1), prob_word_at_topic(count_document_word.cols(), n_topic),
    prob_document_at_topic(count_document_word.rows(), n_topic), n_topic(n_topic), count_document_word(count_document_word)
    {
        random_probablistic_matrix(prob_word_at_topic);
        random_probablistic_matrix(prob_document_at_topic);
        random_probablistic_matrix(prob_topic);
        
        normalize_conditional_probablistic_matrix(prob_word_at_topic);
        normalize_conditional_probablistic_matrix(prob_document_at_topic);
        normalize_conditional_probablistic_matrix(prob_topic);
    }
    
    const MatrixXd & result_topic_at_word()
    {
        prob_topic_at_word = prob_word_at_topic;
        for(auto i=0; i<prob_topic_at_word.cols(); ++i)
        {
            prob_topic_at_word.col(i) *= prob_topic(i);
        }
        for(auto i=0; i<prob_topic_at_word.rows(); ++i)
        {
            prob_topic_at_word.row(i) /= prob_topic_at_word.row(i).sum();
        }
        
        return prob_topic_at_word;
    }
    
    const MatrixXd & result_topic_at_document()
    {
        prob_topic_at_document = prob_document_at_topic;
        for(auto i=0; i<prob_topic_at_document.cols(); ++i)
        {
            prob_topic_at_document.col(i) *= prob_topic(i);
        }
        for(auto i=0; i<prob_topic_at_document.rows(); ++i)
        {
            prob_topic_at_document.row(i) /= prob_topic_at_document.row(i).sum();
        }
        
        return prob_topic_at_document;
    }
    
    const MatrixXd & result_topic() const
    {
        return prob_topic;
    }
    
    void run(int epos)
    {
        while(epos --> 0)
        {
            MatrixXd            training_prob_word_at_topic(count_document_word.cols(), n_topic);
            MatrixXd            training_prob_document_at_topic(count_document_word.rows(), n_topic);
            MatrixXd            training_prob_topic(n_topic, 1);
            
            for(auto w=0; w<count_document_word.cols(); ++w)
            {
                for(auto d=0; d<count_document_word.rows(); ++d)
                {
                    for(auto t=0; t<n_topic; ++t)
                    {
                        if (count_document_word(d, w) != 0)
                        {
                            training_prob_word_at_topic(w, t) += count_document_word(d, w) * prob_word_at_topic(w, t)
                            * prob_document_at_topic(d, t) * prob_topic(t);
                            training_prob_document_at_topic(d, t) += count_document_word(d, w) * prob_word_at_topic(w, t)
                            * prob_document_at_topic(d, t) * prob_topic(t);
                        }
                        
                        training_prob_topic(t) += prob_word_at_topic(w, t) * prob_document_at_topic(d, t) * prob_topic(t);
                    }
                }
            }
            
            prob_word_at_topic = training_prob_word_at_topic;
            prob_document_at_topic = training_prob_document_at_topic;
            prob_topic = training_prob_topic;
            
            normalize_conditional_probablistic_matrix(prob_topic);
            normalize_conditional_probablistic_matrix(prob_document_at_topic);
            normalize_conditional_probablistic_matrix(prob_word_at_topic);
        }
    }
};

#endif
