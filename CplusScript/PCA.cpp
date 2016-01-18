#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
	cout<<"ARMA Ready"<<endl;
	mat a = randu(100, 100);
	a = a + a.t();

	vec vals;
	mat vecs;

	eig_sym(vals, vecs, a);
	cout <<vals <<endl;

	return 0;
}