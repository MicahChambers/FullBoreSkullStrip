#include "itkConstantPadImageFilter.h"
#include "itkInverseFFTImageFilter.h"
#include "itkForwardFFTImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include <itkLinearInterpolateImageFunction.h>

#include <complex>
#include <vector>

double rect(double n, double N); 
double hamming(double n, double N);
double hann(double n, double N);
double tukey(double n, double N);
double gaussKern(double n, double N);
std::complex<double> shift(int k, int N, std::complex<double> m);

/**
 * @brief Rounds up to the nearest factor.
 *
 * @param toround	Number to round
 * @param fact		Set of factors that are considered valid
 *
 * @return 			Rounded number
 */
int roundup(int toround, const std::vector<int>& fact);

template <typename T>
typename T::Pointer padImage(typename T::Pointer in, int minpad[T::ImageDimension])
{
	typename T::SizeType upad;
	typename T::SizeType lpad;
	const unsigned int NDIM = T::ImageDimension;

	// pad image
	auto padFilter = itk::ConstantPadImageFilter<T, T>::New();
	padFilter->SetInput(in);
	padFilter->SetConstant(0);
	
	auto sz = in->GetRequestedRegion().GetSize();

//	auto psize = in->GetRequestedRegion().GetSize();

	std::vector<int> bases(3);
	bases[0] = 2;
	bases[1] = 3;
	bases[2] = 5;

	for(unsigned int ii = 0 ; ii < NDIM; ii++) {
		int tmp = roundup(sz[ii]+minpad[ii], bases);
		std::cerr << "Dim " << ii << " size " << sz[ii] << "rounded to " << tmp << std::endl;
		lpad[ii] = (tmp - sz[ii])/2;
		upad[ii] = tmp - lpad[ii]-sz[ii];
		std::cerr << lpad << ", " << upad << std::endl;
	}

	std::cerr << lpad << ", " << upad << std::endl;
	padFilter->SetPadLowerBound(lpad);
	padFilter->SetPadUpperBound(upad);
	
	try {
		std::cout << "Padding..." ;
		padFilter->Update();
		std::cout << "Done" << std::endl;
	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return NULL;
	}

	return padFilter->GetOutput();
}

/**
 * @brief Pads an image, by a minumum of minpad. The padding
 * makes the image dimensions a factor  of 2, 3 or 5.
 *
 * @tparam T			Image Type
 * @param in			Input image
 * @param minpad		Minimum padding size to use. More than this will be 
 * 							added to each image dimension
 * @return 
 */
template <typename T>
typename T::Pointer padImage(typename T::Pointer in, int minpad)
{
	int minpad2[T::ImageDimension];
	for(int dd = 0 ; dd < T::ImageDimension; dd++)
		minpad2[dd] = minpad;

	return padImage<T>(in, minpad2);
}

template <typename T, typename C>
typename C::Pointer fft(typename T::Pointer input)
{
	auto fft = itk::ForwardFFTImageFilter<T>::New();
	fft->SetInput(input);

	try {
		std::cout << "Computing FFT..." ;
		fft->Update();
		std::cout << "Done" << std::endl;
	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return NULL;
	}

	return fft->GetOutput();
}

template <typename C, typename T>
typename T::Pointer ifft(typename C::Pointer input)
{
	auto ifft = itk::InverseFFTImageFilter<C, T>::New();
	ifft->SetInput(input);
	try {
		std::cout << "Computing iFFT..." ;
		ifft->Update();
		std::cout << "Done" << std::endl;
	} catch( itk::ExceptionObject & error ) {
		std::cerr << "Error: " << error << std::endl;
		return NULL;
	}
	
	return ifft->GetOutput();
}


template<typename T>
typename T::Pointer resize(typename T::Pointer in, typename T::SizeType& osz,
			double (*winfun)(double,double), size_t padsz = 0)
{
	typedef typename itk::ForwardFFTImageFilter<T>::OutputImageType CImageT; //std::complex 
	const unsigned int NDIM = T::ImageDimension;
	assert(NDIM == 3);

	// round up outsize
	std::vector<int> bases(3);
	bases[0] = 2;
	bases[1] = 3;
	bases[2] = 5;
	for(unsigned int ii = 0 ; ii < NDIM; ii++) 
		osz[ii] = roundup(osz[ii], bases);
	
	// pad input for fourier transform
	auto padded = padImage<T>(in, padsz);
	
	auto isz = padded->GetRequestedRegion().GetSize();
	auto ispace = padded->GetSpacing();
	auto ospace = padded->GetSpacing();
	typename T::PointType oorigin;
	
	// fourier transform input
	auto freqimage = fft<T, CImageT>(padded);

	auto smallfreq = CImageT::New();
	smallfreq->SetRegions(osz);
	smallfreq->Allocate();
	smallfreq->FillBuffer(0);
	
	std::complex<float> norm = 1;
	for(unsigned int ii = 0 ; ii < NDIM; ii++) {
		ospace[ii] = ispace[ii]*isz[ii]/(double)osz[ii];
		norm *= (double)osz[ii]/(double)isz[ii];
	}
	
	itk::ImageRegionIteratorWithIndex<CImageT> oit(smallfreq, 
				smallfreq->GetRequestedRegion());
	auto interp = itk::LinearInterpolateImageFunction<CImageT>::New();
	for(oit.GoToBegin(); !oit.IsAtEnd(); ++oit) {
		typename T::IndexType index = oit.GetIndex();

		// if the output image is larger, there will be default 0 frequencies
		bool ignore = false;
		std::complex<float> windowf = 1;
		for(unsigned int dd = 0 ; dd < NDIM; dd++) {
			bool nfreq = false; //negative frequency?
			if(index[dd]*2 > (int)osz[dd]) {
				nfreq = true;
				windowf *= winfun((double)index[dd]-(osz[dd]-1)/2., osz[dd]);
				index[dd] = isz[dd]-(osz[dd]-index[dd]);
			} else {
				windowf *= winfun(index[dd]+(osz[dd]-1)/2., osz[dd]);
			}

			// in these cases we are in negative freqncyes for the input
			// but not the output, or vice-versa
			if((!nfreq && index[dd]*2 > (int)isz[dd]) || 
					(nfreq && index[dd]*2 < (int)isz[dd]))  {
				ignore = true;
				break;
			}
		}

		if(ignore) {
			oit.Set(0);
		} else {
			for(int ii = 0; ii < (int)NDIM; ii++)
				index[ii] += freqimage->GetRequestedRegion().GetIndex()[ii];
			oit.Set(freqimage->GetPixel(index)*norm*windowf);
		}
	}

	auto out = ifft<CImageT,T>(smallfreq);
	out->SetDirection(padded->GetDirection());
	out->SetSpacing(ospace);
	padded->TransformIndexToPhysicalPoint(
				padded->GetRequestedRegion().GetIndex(), oorigin);
	out->SetOrigin(oorigin);
	
	// inverse fourier transform 
	return out;
}


