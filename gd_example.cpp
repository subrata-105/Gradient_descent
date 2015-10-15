#include <iostream>
#include <armadillo>
#include <iomanip>
#include "gd_header.h"

using namespace std;
using namespace arma;

double cost_func(vec x)
{
	mat Q;
	Q << 1 << 0 << 0 << endr
	  << 0 << 1 << 0 << endr
	  << 0 << 0 << 1 << endr;
	  
	vec b;
	b << -4 << 5 << -6 << endr;
	
	double c = 0;
	
	double cost;
	
	cost = as_scalar(0.5*x.t()*Q*x + b.t()*x) + c;
	return cost;
}

int main()
{
	
	cout<<"\nWelcome to the sample run of gradient descent\n";
	
	gradient_descent gd;
	int max_iter = 20;
	double step_size = 0.2;
	int d = 3;
	bool quiet = false;
	double tol = 0.000001;
	
	
	gd.initialize(cost_func, max_iter, step_size, d, quiet, tol,0.01);
	vec x_star = gd.optimize();

	//vec x = ones<vec>(d);
	double fval = cost_func(x_star);
	cout<<"\nx* = \n";
	x_star.print();
	
	cout<<"\nf(x*) = "<<cost_func(x_star)<<"\n\n";
	
	return 0;
}
