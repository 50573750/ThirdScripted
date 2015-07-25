#include "Library.hpp"
#include "Tests.hpp"

int main()
{
    srand(time(nullptr));
    cout<<"Begin"<<endl;
    
    draw_random_circle_ising_model();
    
    cout<<"End"<<endl;
    
    return 0;
}