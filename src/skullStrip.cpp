#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
//#include "itkGradientDescentOptimizer.h"
#include "itkMutualInformationImageToImageMetric.h"
//#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkRegularStepGradientDescentOptimizer.h"
//#include "itkGradientDescentOptimizer.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
//#include "itkMutualInformationImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"
//#include "itkVersorRigid3DTransformOptimizer.h"
//#include "itkVersorRigid3DTransform.h"
//#include "itkCenteredVersorTransformInitializer.h"
#include "itkImageFileWriter.h"

#include <itkDiscreteGaussianImageFilter.h>
//#include <itkLBFGSBOptimizer.h>

#include <cmath>

#include "skullStrip.h"
#include "itkFourier.h"

using namespace std;
typedef itk::Image<float,3> ImageT;
typedef itk::Image<short,3> LImageT;


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

/******************************************
 * Functions
 ******************************************/
template <typename T>
typename T::Pointer gaussianSmooth(typename T::Pointer in, double stddev)
{
	auto smoother = itk::DiscreteGaussianImageFilter<T, T>::New();
	smoother->SetVariance(stddev*stddev);
	smoother->SetInput(in);
	smoother->Update();

	return smoother->GetOutput();
}

double bSplineReg(itk::BSplineTransform<double, 3, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL)
{
	cerr << "BSpline Registration" << endl;
	if(sd > 0) {
		cerr << "Old Res: " << source->GetSpacing()<< ", " << target->GetSpacing() << endl;
		/******************************************************
		 * Low Resolution
		 *****************************************************/
		ImageT::SizeType osz;
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = source->GetRequestedRegion().GetSize()[ii]*
				source->GetSpacing()[ii]/(2*sd);
		source = resize<ImageT>(source, osz, gaussKern, 10);
		cerr << "New Res: " << source->GetSpacing()<< ", "; 

		// match the spacing
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = target->GetRequestedRegion().GetSize()[ii]*
				target->GetSpacing()[ii]/source->GetSpacing()[ii];
		target = resize<ImageT>(target, osz, gaussKern, 10);
		cerr << target->GetSpacing() << endl;
	}
	
	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "bspline_source_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(source);
		writer->Update();
	}
	
	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "bspline_target_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(target);
		writer->Update();
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

	auto opt = itk::RegularStepGradientDescentOptimizer::New();
	opt->SetMinimumStepLength(minstep);
	opt->SetMaximumStepLength(maxstep);
	opt->SetRelaxationFactor(relax);
	opt->SetNumberOfIterations(nstep);
	opt->SetGradientMagnitudeTolerance(TOL);
	opt->MinimizeOn();

	reg->SetOptimizer(opt);
	reg->SetTransform(tfm);
	reg->SetInitialTransformParameters(tfm->GetParameters());
	reg->SetInterpolator(interp);

	if(samecontrast) {
		std::cerr << "Normalized correlation metric...";
		auto metric = itk::NormalizedCorrelationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
	} else {
		std::cerr << "Mutual Information metric..."; 
		auto metric = itk::MattesMutualInformationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
		metric->SetNumberOfSpatialSamples(nsamp);
		metric->SetNumberOfHistogramBins(nbins);
	}
	
	reg->SetFixedImage(target);
	reg->SetMovingImage(source);
	reg->SetFixedImageRegion(target->GetLargestPossibleRegion());
  
	double value = INFINITY;
	try {
		std::cerr << "Performing BSpline Registration..."; 
		reg->Update();
		double before = reg->GetMetric()->GetValue(reg->GetInitialTransformParameters());
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << before << ") -> (" << value << ") " 
				<< opt->GetStopConditionDescription() << endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return -INFINITY;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
	return -value;
}

double affineReg(itk::AffineTransform<double, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL)
{
	cerr << "Affine Registration" << endl;
	if(sd > 0) {
		cerr << "Old Res: " << source->GetSpacing()<< ", " << target->GetSpacing() << endl;
		/******************************************************
		 * Low Resolution
		 *****************************************************/
		ImageT::SizeType osz;
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = source->GetRequestedRegion().GetSize()[ii]*
				source->GetSpacing()[ii]/(2*sd);
		source = resize<ImageT>(source, osz, gaussKern, 10);
		cerr << "New Res: " << source->GetSpacing()<< ", "; 

		// match the spacing
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = target->GetRequestedRegion().GetSize()[ii]*
				target->GetSpacing()[ii]/source->GetSpacing()[ii];
		target = resize<ImageT>(target, osz, gaussKern, 10);
		cerr << target->GetSpacing() << endl;
	}

	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "source_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(source);
		writer->Update();
	}
	
	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "target_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(target);
		writer->Update();
	}
	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

	auto opt = itk::RegularStepGradientDescentOptimizer::New();
	opt->SetMinimumStepLength(minstep);
	opt->SetMaximumStepLength(maxstep);
	opt->SetRelaxationFactor(relax);
	opt->SetNumberOfIterations(nstep);
	opt->SetGradientMagnitudeTolerance(TOL);
	opt->MinimizeOn();

	reg->SetOptimizer(opt);
	reg->SetTransform(tfm);
	reg->SetInitialTransformParameters(tfm->GetParameters());
	reg->SetInterpolator(interp);

	if(samecontrast) {
		std::cerr << "Normalized correlation metric...";
		auto metric = itk::NormalizedCorrelationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
	} else {
		std::cerr << "Mutual Information metric..."; 
		auto metric = itk::MattesMutualInformationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
		metric->SetNumberOfSpatialSamples(nsamp);
		metric->SetNumberOfHistogramBins(nbins);
	}
	
	reg->SetFixedImage(target);
	reg->SetMovingImage(source);
	reg->SetFixedImageRegion(target->GetLargestPossibleRegion());
  
	double value = INFINITY;
	try {
		std::cerr << "Performing Affine Registration..."; 
		reg->Update();
		double before = reg->GetMetric()->GetValue(reg->GetInitialTransformParameters());
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << before << ") -> (" << value << ") " 
				<< opt->GetStopConditionDescription() << endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return -INFINITY;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
	return -value;
}

double rigidReg(itk::Euler3DTransform<double>::Pointer tfm,
		itk::Image<float,3>::Pointer source, itk::Image<float,3>::Pointer target, 
		double sd, bool samecontrast,
		int nstep, double minstep, double maxstep,
		int nbins, double relax, int nsamp, double TOL)
{
	if(sd > 0) {
		cerr << "Old Res: " << source->GetSpacing()<< ", " << target->GetSpacing() << endl;
		/******************************************************
		 * Low Resolution
		 *****************************************************/
		ImageT::SizeType osz;
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = source->GetRequestedRegion().GetSize()[ii]*
				source->GetSpacing()[ii]/(2*sd);
		source = resize<ImageT>(source, osz, gaussKern, 10);
		cerr << "New Res: " << source->GetSpacing()<< ", "; 

		// match the spacing
		for(int ii = 0 ; ii < 3; ii++)
			osz[ii] = target->GetRequestedRegion().GetSize()[ii]*
				target->GetSpacing()[ii]/source->GetSpacing()[ii];
		target = resize<ImageT>(target, osz, gaussKern, 10);
		cerr << target->GetSpacing() << endl;
	}
	
	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "rsource_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(source);
		writer->Update();
	}
	
	{
		itk::ImageFileWriter<ImageT>::Pointer writer;
		writer = itk::ImageFileWriter<ImageT>::New();
		std::ostringstream oss;
		oss << "rtarget_smooth" << sd << ".nii.gz"; 
		writer->SetFileName(oss.str());
		writer->SetInput(target);
		writer->Update();
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

	itk::Array<double> scales(6);
	auto opt = itk::RegularStepGradientDescentOptimizer::New();
	opt->SetMinimumStepLength(minstep);
	opt->SetMaximumStepLength(maxstep);
	opt->SetRelaxationFactor(relax);
	opt->SetNumberOfIterations(nstep);
	opt->SetGradientMagnitudeTolerance(TOL);
	opt->MinimizeOn();

	// change 
	scales[0] = 1./500;
	scales[1] = 1./500;
	scales[2] = 1./500;
	scales[3] = 1;
	scales[4] = 1;
	scales[5] = 1;
	opt->SetScales(scales);

	reg->SetOptimizer(opt);
	reg->SetTransform(tfm);
	reg->SetInitialTransformParameters(tfm->GetParameters());
	reg->SetInterpolator(interp);

	if(samecontrast) {
		std::cerr << "Normalized correlation metric...";
		auto metric = itk::NormalizedCorrelationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
	} else {
		std::cerr << "Mutual Information metric..."; 
		auto metric = itk::MattesMutualInformationImageToImageMetric<ImageT, ImageT>::New();
		reg->SetMetric(metric);
		metric->SetNumberOfSpatialSamples(nsamp);
		metric->SetNumberOfHistogramBins(nbins);
	}

	reg->SetFixedImage(target);
	reg->SetMovingImage(source);
	reg->SetFixedImageRegion(target->GetLargestPossibleRegion());
  
	double value = INFINITY;
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		double before = reg->GetMetric()->GetValue(reg->GetInitialTransformParameters());
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << before << ") -> (" << value << ") " 
				<< opt->GetStopConditionDescription() << endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return -INFINITY;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
	return -value;
}


ImageT::Pointer apply(itk::Transform<double,3,3>::Pointer tfm, 
		ImageT::Pointer source, ImageT::Pointer target)
{
  	auto resample = itk::ResampleImageFilter<ImageT, ImageT>::New();
	resample->SetTransform(tfm);
	resample->SetInput(source);
	resample->SetSize(target->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(target->GetOrigin());
	resample->SetOutputSpacing(target->GetSpacing());
	resample->SetOutputDirection(target->GetDirection());
	resample->SetDefaultPixelValue(0);
	resample->Update();

	return resample->GetOutput();
}

LImageT::Pointer applyNN(itk::Transform<double,3,3>::Pointer tfm, 
		LImageT::Pointer source, ImageT::Pointer target)
{
  	auto resample = itk::ResampleImageFilter<LImageT, LImageT>::New();
	auto interp = itk::NearestNeighborInterpolateImageFunction<LImageT, double>::New();
	resample->SetTransform(tfm);
	resample->SetInput(source);
	resample->SetInterpolator(interp);
	resample->SetSize(target->GetLargestPossibleRegion().GetSize());
	resample->SetOutputOrigin(target->GetOrigin());
	resample->SetOutputSpacing(target->GetSpacing());
	resample->SetOutputDirection(target->GetDirection());
	resample->SetDefaultPixelValue(0);
	resample->Update();

	return resample->GetOutput();
}


