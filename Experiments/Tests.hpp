#ifndef Experiments_Tests_hpp
#define Experiments_Tests_hpp

#include "PPM.hpp"
#include "IsingModel.hpp"
#include "HiddenMarkovModel.hpp"
#include "GuassianProcess.hpp"
#include "ProbablisticLatentSemanticIndex.hpp"

void draw_random_circle_ising_model()
{
    auto draw_random_circle = [](double x, double y)
    {
        if ( pow(x, 2) + pow(y, 2) <= rand()%10 * 10000 )
            return true;
        else
            return false;
    };
    
    MatrixXd    image(500, 500);
    for(auto i=0; i<500; ++i)
    {
        for(auto j=0; j<500; ++j)
        {
            if (draw_random_circle(i,j))
            {
                image(i, j) = 1;
            }
            else
            {
                image(i, j) = -1;
            }
        }
    }
    
    
    IsingModelNaiveMean imnm(image);
    imnm.run(30);
    PPM::write(imnm.result(), "/Users/Bookman/Documents/trys/draw_random_circle.ppm");
}

void test_for_HMM()
{
    HiddenMarkovModel           hmm(10, 5);
    
    vector<vector<unsigned>>    datasets = { {0,1,2,3,1,0}, {1,0,1,2,3,0}, {1,1,2,3,1,1,0} };
    hmm.baum_welch_training(datasets, 10);
    
    vector<vector<unsigned>>    testcase = { {1,2,3}, {0,1,2}, {0,0,0} };
    cout<<-log(hmm.probability(testcase[0]))<<endl;
    cout<<-log(hmm.probability(testcase[1]))<<endl;
    cout<<-log(hmm.probability(testcase[2]))<<endl;
}

void test_for_GP()
{
    MatrixXd    datasets(4,2);
    datasets<<1,1,-1,1,1,-1,-1,-1;
    MatrixXd    labels(1,4);
    labels<<1,1,-1,-1;
    
    GuassianProcessExact gpe(datasets, labels);
    cout<<gpe.result(datasets)<<endl;
}

void test_for_PLSI()
{
    MatrixXd    datasets(5,5);
    datasets<<1,2,1,1,5, 2,3,0,1,5, 1,0,4,5,0, 5,6,1,0,5, 0,1,6,6,0;
    
    ProbablisiticLatentSemanticIndex    plsi(datasets, 2);
    plsi.run(200);
    
    cout<<"Topic Distribution:"<<endl<<plsi.result_topic()<<endl<<endl;
    cout<<"Word Distribution:"<<endl<<plsi.result_topic_at_word()<<endl<<endl;
    cout<<"Document Distribution:"<<endl<<plsi.result_topic_at_document()<<endl<<endl;
}
#endif
