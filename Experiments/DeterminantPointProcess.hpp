#ifndef Experiments_DPP_hpp
#define Experiments_DPP_hpp

#include "Library.hpp"

class DeterminantPointProcess
{
protected:
    const MatrixXd&     similarity;
    unsigned                cnt_item;
    
public:
    DeterminantPointProcess(const MatrixXd& similarity, unsigned cnt_item)
    : similarity(similarity), cnt_item(cnt_item)
    {
        ;
    }
    
    void run(set<unsigned>& index, unsigned cnt_selected)
    {
        EigenSolver<MatrixXd> eigen_similairy(similarity);
        
        auto similarity_eigval = eigen_similairy.eigenvalues();
        auto similarity_eigvec = eigen_similairy.eigenvectors();
        
        set<unsigned>   index_selected;
        for(auto cnt_i=0; cnt_i<cnt_item; ++cnt_i)
        {
            if (rand() > similarity_eigval(cnt_i, 0).real() / (similarity_eigval(cnt_i, 0).real() + 1))
            {
                index_selected.insert(cnt_i);
            }
        }
        
        while(cnt_selected--)
        {
            double massprob_current = 0;
            vector<double>  samping_distribution(cnt_item);
            
            for(auto cnter_item=0; cnter_item<cnt_item; ++cnter_item)
            {
                double prob_current = 0;
                for(auto i=index_selected.begin(); i!=index_selected.end(); ++i)
                {
                    prob_current += similarity_eigvec(cnter_item, *i).real()
                    * similarity_eigvec(cnter_item, *i).real();
                }
                massprob_current += prob_current;
                samping_distribution[cnter_item] = massprob_current;
            }
            
            double prob_sampling = rand()%10000/10000.0 * massprob_current;
            for(auto cnter_item=0; cnter_item<cnt_item; ++cnter_item)
            {
                if (prob_sampling <= samping_distribution[cnter_item])
                {
                    index.insert(cnter_item);
                    for(auto i=index_selected.begin(); i!=index_selected.end(); ++i)
                    {
                        similarity_eigvec(cnter_item, *i) = 0;
                    }
                    break;
                }
            }
            
            for(auto i_outer=index_selected.begin(); i_outer!=index_selected.end(); ++i_outer)
            {
                auto vec_tmp = similarity_eigvec.col(*i_outer);
                for(auto i_inner=index_selected.begin(); i_inner!=i_outer; ++ i_inner)
                {
                    similarity_eigvec.col(*i_outer) -=
                    (vec_tmp.transpose() * similarity_eigvec.col(*i_inner))(0,0)
                    / (0.01 +
                       similarity_eigvec.col(*i_inner).norm()
                       * similarity_eigvec.col(*i_inner).norm())
                    * similarity_eigvec.col(*i_inner);
                }
            }
        }
    }
};

#endif
