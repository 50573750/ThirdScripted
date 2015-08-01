#include "Library.hpp"
#include "Tests.hpp"

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
        auto sigmod = [&](MatrixXd::RowXpr& mat)
        {
            for(auto i=0; i<mat.rows(); ++i)
            {
                mat(i) = 1/(1 + exp(-mat(i)));
            }
        };
        
        auto sigmod_dervation = [&](MatrixXd::RowXpr& mat)
        {
            for(auto i=0; i<mat.rows(); ++i)
            {
                mat(i) = mat(i) * (1-mat(i));
            }
        };
        
        MatrixXd output(weights.size()+1, input.size());
        output.row(0) = input;
        for(auto i=0; i<weights.size(); ++i)
        {
            output.row(i+1) = output.row(i) * weights[i];
            sigmod(output.row(i+1));
        }
        
        return output.row(weights.size());
    }
};

int main()
{
    srand(time(nullptr));

    return 0;
}