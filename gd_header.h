#include <iostream>
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

class gradient_descent
{
	public:
	double (*ptrFunc) (vec);
	int max_iter;
	int d;
	double step_size;
	double fval;
	bool quiet;
	double tol;
	double h;
	
	void initialize(double (*ptrFunc)(vec), int max_iter, double step_size,int d,bool quiet,double tol,double h);
	
	vec optimize();
	
	vec compute_gradient(vec x);
	
};

