#ifndef Experiments_Tests_hpp
#define Experiments_Tests_hpp

#include "PPM.hpp"
#include "IsingModel.hpp"
#include "HiddenMarkovModel.hpp"
#include "GuassianProcess.hpp"
#include "ProbablisticLatentSemanticIndex.hpp"
#include "FeadForwardNeuralNetworkBackPropergation.hpp"
#include "AutoEncoder.hpp"
#include "Apriori.hpp"

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
    HiddenMarkovModel           hmm(10, 4);
    
    vector<vector<unsigned>>    datasets = { {0,1,2,3,1,0,2,0,1,2,3,0,1,1,2,3,1,1,0} };
    hmm.baum_welch_training(datasets, 100);
    
    vector<vector<unsigned>>    testcase = { {1,2,3}, {0,1,2}, {0,0,0} };
    cout<<(hmm.probability(testcase[0]))<<endl;
    cout<<(hmm.probability(testcase[1]))<<endl;
    cout<<(hmm.probability(testcase[2]))<<endl;
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

void test_for_FFNNBP()
{
    FeedForwardNeualNetwokBackPropergation ffnnbp(2, 1);
    ffnnbp.add_hidden_layer(5);
    ffnnbp.add_hidden_layer(5);
    ffnnbp.add_hidden_layer(5);
    ffnnbp.add_hidden_layer(5);
    ffnnbp.add_hidden_layer(10);
    ffnnbp.add_hidden_layer(5);
    
    vector<pair<VectorXd, VectorXd>>    datasets(4);
    
    for(auto i=0; i<4; ++i)
    {
        datasets[i].first.resize(2);
        datasets[i].second.resize(1);
    }
    
    datasets[0].first<<0,0;
    datasets[0].second<<1;
    datasets[1].first<<0,1;
    datasets[1].second<<0;
    datasets[2].first<<1,0;
    datasets[2].second<<0;
    datasets[3].first<<1,1;
    datasets[3].second<<1;
    
    ffnnbp.train(datasets, 10000, 0.1);
    cout<<ffnnbp.infer(datasets[0].first)<<endl;
    cout<<ffnnbp.infer(datasets[1].first)<<endl;
    cout<<ffnnbp.infer(datasets[2].first)<<endl;
    cout<<ffnnbp.infer(datasets[3].first)<<endl;
}

void test_for_AE()
{
    MatrixXd    datasets(4,2);
    datasets<<0,0,1,0,0,1,1,1;
    
    AutoEncoder ae(2);
    ae.add_layer(5);
    ae.add_layer(3);
    ae.add_layer(5);
    
    ae.train(datasets, 10000, 0.5, 0.1);
    cout<<ae.infer(datasets.row(0))<<endl;
    cout<<ae.infer(datasets.row(1))<<endl;
    cout<<ae.infer(datasets.row(2))<<endl;
    cout<<ae.infer(datasets.row(3))<<endl;
    
    cout<<ae.infer_represent(datasets.row(0))<<endl;
    cout<<ae.infer_represent(datasets.row(1))<<endl;
    cout<<ae.infer_represent(datasets.row(2))<<endl;
    cout<<ae.infer_represent(datasets.row(3))<<endl;
}

void test_for_apriori()
{
    vector<set<unsigned>>   dataset(6);
    dataset[0] = {0,1,2,3};
    dataset[1] = {0,1,3};
    dataset[2] = {0,1,2,3,4};
    dataset[3] = {4,5,6};
    dataset[4] = {2,3,7};
    dataset[5] = {0,1,2,3,4,5,6};
    
    Apriori ap(7);
    set<set<unsigned>>   result = ap.run(dataset, 4);
    for_each(result.begin(), result.end(), [](const set<unsigned>& elem)
             {
                 copy(elem.begin(), elem.end(), ostream_iterator<unsigned>(cout, ","));
                 cout<<endl;
             });
}

#endif
