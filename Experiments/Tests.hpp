#ifndef Experiments_Tests_hpp
#define Experiments_Tests_hpp

#include "PPM.hpp"
#include "IsingModel.hpp"
#include "HiddenMarkovModel.hpp"

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

void test_for_hmm()
{
    HiddenMarkovModel           hmm(10, 4);
    
    vector<vector<unsigned>>    datasets = { {0,1,2,3,1,0}, {1,0,1,2,3,0}, {1,1,2,3,1,1,0} };
    hmm.baum_welch_training(datasets, 5);
    
    vector<vector<unsigned>>    testcase = { {1,2,3}, {0,1,2}, {0,0,0} };
    cout<<-log(hmm.probability(testcase[0]))<<endl;
    cout<<-log(hmm.probability(testcase[1]))<<endl;
    cout<<-log(hmm.probability(testcase[2]))<<endl;
}

#endif
