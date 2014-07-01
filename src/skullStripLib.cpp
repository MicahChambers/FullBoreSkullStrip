#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
//#include "itkGradientDescentOptimizer.h"
#include <itkBSplineTransform.h>
#include "itkMutualInformationImageToImageMetric.h"
//#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkAffineTransform.h"
#include "itkRegularStepGradientDescentOptimizer.h"
//#include "itkGradientDescentOptimizer.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
//#include "itkMutualInformationImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"
#include "itkEuler3DTransform.h"
//#include "itkVersorRigid3DTransformOptimizer.h"
//#include "itkVersorRigid3DTransform.h"
//#include "itkCenteredVersorTransformInitializer.h"

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <itkDiscreteGaussianImageFilter.h>
//#include <itkLBFGSBOptimizer.h>

#include <cmath>

using namespace std;
typedef itk::Image<float,3> ImageT;
typedef itk::Image<short,3> LImageT;

const double PI = acos(-1);

double bSplineReg(itk::BSplineTransform<double, 3, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

double affineReg(itk::AffineTransform<double, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

double rigidReg(itk::Euler3DTransform<double>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

ImageT::Pointer apply(itk::Transform<double,3,3>::Pointer tfm, 
		ImageT::Pointer source, ImageT::Pointer target);

LImageT::Pointer applyNN(itk::Transform<double,3,3>::Pointer tfm, 
		LImageT::Pointer source, ImageT::Pointer target);

/* Helper function to write an image "out" to prefix + filename */
template <typename T, typename R = T>
void writeImage(std::string name, typename T::Pointer in);

template <typename T, typename R = T>
void writeImage( typename T::Pointer in, std::string name);

template <typename T>
typename T::Pointer readImage(std::string name);

struct ParamLessEqual {
	bool operator()(const itk::Array<double>& lhs,
			const itk::Array<double>& rhs)  const
	{
		const double TOL = 0.0000001;
		for(size_t ii=0 ;ii<lhs.GetSize(); ii++) {
			if(rhs[ii] - lhs[ii]>TOL) {
				return true;
			}
			if(lhs[ii] - rhs[ii]>TOL) {
				return false;
			}
		}
		return false;
	}
};


/**
 * @brief Returns a continuous index that is the centroid of the image
 *
 * @param in
 *
 * @return 
 */
ImageT::PointType getCenter(ImageT::Pointer in)
{
	itk::ContinuousIndex<double, 3> index = {{{0,0,0}}};
	ImageT::PointType opt;
	itk::ImageRegionIteratorWithIndex<ImageT> it(in,in->GetRequestedRegion());
	double mean = 0;
	size_t nn = 0;
	for(it.GoToBegin(); !it.IsAtEnd(); ++it) {
		mean += it.Get();
		nn++;
	}
	mean /= nn;

	nn = 0;
	for(it.GoToBegin(); !it.IsAtEnd(); ++it) {
		if(it.Get() > mean) {
			for(size_t ii=0 ; ii< 3; ii++) 
				index[ii] += it.GetIndex()[ii];
			nn++;
		}
	}

	for(size_t ii=0 ; ii< 3; ii++) 
		index[ii] /= nn;
	in->TransformContinuousIndexToPhysicalPoint(index, opt);
	return opt;
}



