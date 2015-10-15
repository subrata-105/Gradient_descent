#include "gd_header.h"

using namespace std;
using namespace arma;

void gradient_descent::initialize(double (*ptrFunc)(vec), int max_iter, double step_size,int d,bool quiet,double tol,double h)
	{
		this->ptrFunc = ptrFunc;
		
		if(max_iter<=100)
		{
				this->max_iter = max_iter;
		}
		else
		{
			cout<<"\nmax_iter cannot be greater than 100\n setting it to 100";
			this->max_iter = max_iter;
		}
		
		this->d = d;
		this->step_size = step_size;
		this->quiet = quiet;
		this->tol = tol;
		this->h = h;
	}  
	
	
vec gradient_descent::optimize()
	{
		vec grad;
		mat x = zeros<mat>(d,max_iter+1);
		
		if (!quiet)
		{	cout<<"\n################";
			cout<<"\n iter    f(x)\n----------------\n";   
		}	
		
		int k;
		for(k=1;k<=max_iter;k++)
		{
			grad = compute_gradient(x.col(k-1));
			if(as_scalar(grad.t()*grad)<=tol*tol)
			{
				cout<<"\nexiting due to small gradient size..\n";
				break;
			}
			
			x.col(k) = x.col(k-1) - step_size*grad;
			fval = ptrFunc(x.col(k));
			if (!quiet)
			{
				cout<<setw(4);
				cout<<k<<"    "<<fval<<endl;
				
			}
		}
		
		return x.col(k-1);
	}
	
	
vec gradient_descent::compute_gradient(vec x)
	{
		vec e = zeros<vec> (d);
		vec grad = zeros<vec>(d);
		vec x1,x2;
		double h_inv = 1/(2*h);
		
		for(int i=0;i<d;i++)
		{
			e(i) = h;
			x1 = x - e;
			x2 = x + e;
			grad(i) = h_inv*(ptrFunc(x2) - ptrFunc(x1));
			e(i) = 0.0;
		}
		
		return grad;
	}
	
