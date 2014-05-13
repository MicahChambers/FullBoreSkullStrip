
#include "itkFourier.h"

using std::complex;
using std::vector;

int roundup(int toround, const vector<int>& fact)
{
	//remove given factors
	int div = 1;
	int tmp = toround;
	int rounded = toround;
	int ii = 0;

	int step = 0;
	while(tmp > 1) {
		tmp = rounded;

		/* Factor for a while */
		for(ii = 0 ; ii < (int)fact.size(); ) {
			int m = tmp%fact[ii];
			if(m == 0) {
				tmp /= fact[ii];
				div *= fact[ii];
			} else if(step < fact[ii] - m) {
				ii++;
			}
		}

		rounded++;
	}
		
	return rounded-1;
}

// Window Functions
double hamming(double n, double N)
{
	const double alpha = 0.54;
	const double beta = 1 - alpha;
	const double PI = acos(-1);

//	return (alpha - beta*cos(2*PI*x/rad))/rad;
	return (alpha - beta*cos(2*PI*n/(N-1)));
}

double hann(double n, double N)
{
	const double PI = acos(-1);

	return .5*(1-cos(2*PI*n/(N-1)));
}

double gaussKern(double n, double N)
{
	double mu = (N-1)/2;
	
	// mu is the radius, so in the standard normal
	double scale = 3;
	double sigma = mu/(2*scale);
	double sigmasq = sigma*sigma;

	// n == 0 -> e^0 -> 1
	return exp(-0.5*(n-mu)*(n-mu)/sigmasq);
}

double tukey(double n, double N)
{
	const double r = .5;
	const double PI = acos(-1);
//	double x = (n-N/2)/N;
	double x = n/(N-1);

	if(x < r/2) {
		return .5*(1+cos(2*PI/r*(x-r/2)));
	} else if(x < 1-r/2) {
		return 1;
	} else if(x< 1) {
		return .5*(1+cos(2*PI/r*(x-1+r/2)));
	} else {
		return 0;
	}
}

double rect(double n, double N)
{
	return 1;
}

complex<double> shift(int k, int N, complex<double> m)
{
	const complex<double> i(0,1);
	double PI = acos(-1);
	
	// shift of x_{n-m} leads to the transform: 
	// e^{-2 PI i k m / N}
	return std::exp(-(double)2*PI*i*(double)k*m/(double)N);
}

template <>
itk::Image<float,3>::Pointer resize<itk::Image<float,3>>(
		itk::Image<float,3>::Pointer in, itk::Image<float,3>::SizeType& osz,
		double (*winfun)(double,double), size_t padsz);


