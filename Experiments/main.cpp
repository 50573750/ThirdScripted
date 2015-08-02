#include "Library.hpp"
#include "Tests.hpp"
#include "RestrictBoltzmanMachine.hpp"

class FeedForwardNeualNetwokBackPropergation
{
protected:
    vector<MatrixXd>    weights;
    const unsigned      n_input;
    unsigned            n_output;
    
public:
    FeedForwardNeualNetwokBackPropergation(int n_input, int n_output)
    :n_input(n_input), n_output(n_output)
    {
        weights.push_back(MatrixXd::Random(n_input, n_output));
    }
    
    void add_hidden_layer(int n_nodes)
    {
        weights.back() = MatrixXd::Random(weights.back().rows(), n_nodes);
        weights.push_back(MatrixXd::Random(n_nodes, n_output));
    }
    
    MatrixXd infer(const VectorXd& input)
    {
        auto sigmod = [&](double& elem)
        {
            return 1/(1 + exp(-elem));
        };
        
        MatrixXd    output = input.transpose();
        for(auto layer=0; layer<weights.size(); ++layer)
        {
            output = output * weights[layer];
            for(auto i=0; i<output.size(); ++i)
            {
                output(i) = sigmod(output(i));
            }
        }
        
        return output;
    }
    
    void train(vector<pair<VectorXd, VectorXd>>& dataset, int epoc, double alpha = 0.2)
    {
        auto sigmod = [&](double& elem)
        {
            return 1/(1 + exp(-elem));
        };
        
        auto sigmod_dervation = [&](double& elem)
        {
            return elem * (1-elem);
        };

        MatrixXd    pretrained(dataset.size(), n_input);
        for(auto pos=0; pos<dataset.size(); ++pos)
        {
            pretrained.row(pos) = dataset[pos].first;
        }
                       
        for(auto layer=0; layer<weights.size(); ++layer)
        {
            RestrictBoltzmanMachineGibbisSampling      rbm(pretrained, weights[layer].cols());
            rbm.run(1000, 0.1, 1);
            weights[layer] = rbm.get_weights();
            
            pretrained = rbm.result();
        }
        
        while(epoc-->0)
        {
            for(auto pos=0; pos<dataset.size(); ++pos)
            {
                vector<MatrixXd>    hidden_output;
                hidden_output.push_back(dataset[pos].first.transpose());
                for(auto i=0; i<weights.size(); ++i)
                {
                    hidden_output.push_back(hidden_output.back() * weights[i]);
                    for(auto elem=0; elem<hidden_output.back().size(); ++elem)
                    {
                        hidden_output.back()(elem) = sigmod(hidden_output.back()(elem));
                    }
                }
                
                vector<MatrixXd>   dervative_output;
                dervative_output.push_back((hidden_output.back() - dataset[pos].second).transpose());
                for(int layer=weights.size()-1; layer>=0; --layer)
                {
                    for(auto elem=0; elem<dervative_output.back().size(); ++elem)
                    {
                        dervative_output.back()(elem) *= sigmod_dervation(hidden_output[layer+1](elem));
                    }
                    
                    weights[layer] -= alpha * hidden_output[layer].transpose() * dervative_output.back();
                    dervative_output.push_back(dervative_output.back() * weights[layer].transpose());
                }
            }
        }
    }
};

int main()
{
    srand(time(nullptr));
    
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
    
    return 0;
}