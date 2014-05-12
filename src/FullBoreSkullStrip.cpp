/** 
This file is part of Neural Programs Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Programs Library is free software: you can redistribute it and/or 
modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Neural Programs Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the Neural Programs Library.  If not, see 
<http://www.gnu.org/licenses/>.
*/

#include <tclap/CmdLine.h>
#include "version.h"
#include "registerIO.h"
RegisterIO REG;

#include "itkGradientDescentOptimizer.h"
#include <itkBSplineTransform.h>
#include "itkMutualInformationImageToImageMetric.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkAffineTransform.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkGradientDescentOptimizer.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkNormalizedCorrelationImageToImageMetric.h"
#include "itkMutualInformationImageToImageMetric.h"
#include "itkImageRegistrationMethod.h"
#include "itkVersorRigid3DTransformOptimizer.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredVersorTransformInitializer.h"

#include <itkDiscreteGaussianImageFilter.h>
#include <itkLBFGSBOptimizer.h>

using namespace std;
typedef itk::Image<float,3> ImageT;

void bSplineReg(itk::BSplineTransform<double, 3, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

void affineReg(itk::AffineTransform<double, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

void rigidReg(itk::VersorRigid3DTransform<double>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL);

ImageT::Pointer apply(itk::Transform<double,3,3>::Pointer tfm, 
		ImageT::Pointer source, ImageT::Pointer target);

int main(int argc, char** argv)
{
	try {
	/* 
	 * Command Line 
	 */
	TCLAP::CmdLine cmd("This program does k-space resampling of the input "
			"image. Proper pixel spacing is set. Note there may be ringing "
			"in the output.", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input Image",
			true, "", "image", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output Image",
			true, "", "image", cmd);
	
	std::vector<string> allowed({"rect","tukey", "hamming", "hann"});
	TCLAP::ValuesConstraint<string> allowedVals( allowed );
	TCLAP::ValueArg<string> a_window("w", "window", "Window function.",
			false, "rect", &allowedVals, cmd);

	TCLAP::MultiArg<int> a_size("z", "size", "Dimensions, first arg is "
			"x, second is y, third is z.", false, "int", cmd);
	TCLAP::MultiArg<double> a_space("s", "spacing", "Spacing, first arg is "
			"x spacing, second is y spacing, third is z.", false, "step", cmd);
	TCLAP::ValueArg<size_t> a_padsize("p", "padding", "Minimum amount of zero "
			"padding, helpful during registration where you don't want stuff "
			"falling out of the field of view", false, 0, "padsize", cmd);
	cmd.parse(argc, argv);

	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
	return 0;
}

template <typename T>
typename T::Pointer gaussianSmooth(typename T::Pointer in, double stddev)
{
	auto smoother = itk::DiscreteGaussianImageFilter<T, T>::New();
	smoother->SetVariance(stddev*stddev);
	smoother->SetInput(in);
	smoother->Update();

	return smoother->GetOutput();
}

void bSplineReg(itk::BSplineTransform<double, 3, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL)
{
	if(sd > 0) {
		source = gaussianSmooth<ImageT>(source, sd);
		target = gaussianSmooth<ImageT>(target, sd);
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

//	auto opt = itk::RegularStepGradientDescentOptimizer::New();
//	auto opt = itk::VersorRigid3DTransformOptimizer::New();
	auto opt = itk::LBFGSBOptimizer::New();
//	opt->SetMinimumStepLength(minstep);
//	opt->SetMaximumStepLength(maxstep);
//	opt->SetRelaxationFactor(relax);
	opt->SetMaximumNumberOfIterations(nstep);
//	opt->SetGradientMagnitudeTolerance(TOL);

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
  
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		std::cerr << " (" 
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< " -> "
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< ") " << opt->GetStopConditionDescription() << std::endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
}

void affineReg(itk::AffineTransform<double, 3>::Pointer tfm,
			ImageT::Pointer source, ImageT::Pointer target, 
			double sd, bool samecontrast,
			int nstep, double minstep, double maxstep,
			int nbins, double relax, int nsamp, double TOL)
{
	if(sd > 0) {
		source = gaussianSmooth<ImageT>(source, sd);
		target = gaussianSmooth<ImageT>(target, sd);
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

//	auto opt = itk::RegularStepGradientDescentOptimizer::New();
//	auto opt = itk::VersorRigid3DTransformOptimizer::New();
	auto opt = itk::LBFGSBOptimizer::New();
//	opt->SetMinimumStepLength(minstep);
//	opt->SetMaximumStepLength(maxstep);
//	opt->SetRelaxationFactor(relax);
	opt->SetMaximumNumberOfIterations(nstep);
//	opt->SetGradientMagnitudeTolerance(TOL);
	itk::Array<double> scales(6);

	//rotation
	for(int ii = 0 ; ii < 3; ii++)
		scales[ii] = 1;
	//translation
	for(int ii = 3 ; ii < 6; ii++)
		scales[ii] = .001;

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
  
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		std::cerr << " (" 
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< " -> "
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< ") " << opt->GetStopConditionDescription() << std::endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
}

void rigidReg(itk::VersorRigid3DTransform<double>::Pointer tfm,
		itk::Image<float,3>::Pointer source, itk::Image<float,3>::Pointer target, 
		double sd, bool samecontrast,
		int nstep, double minstep, double maxstep,
		int nbins, double relax, int nsamp, double TOL)
{
	auto init = itk::CenteredVersorTransformInitializer<ImageT, ImageT>::New();
	init->SetTransform(tfm);
	init->SetFixedImage(target);
	init->SetMovingImage(source);
  	init->MomentsOn();
  	init->InitializeTransform();
	
	if(sd > 0) {
		source = gaussianSmooth<ImageT>(source, sd);
		target = gaussianSmooth<ImageT>(target, sd);
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

	auto opt = itk::VersorRigid3DTransformOptimizer::New();
	opt->SetMinimumStepLength(minstep);
	opt->SetMaximumStepLength(maxstep);
	opt->SetRelaxationFactor(relax);
	opt->SetNumberOfIterations(nstep);
	opt->SetGradientMagnitudeTolerance(TOL);
	opt->MinimizeOn();
	itk::Array<double> scales(6);

	//rotation
	for(int ii = 0 ; ii < 3; ii++)
		scales[ii] = 1;
	//translation
	for(int ii = 3 ; ii < 6; ii++)
		scales[ii] = .001;

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
  
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		std::cerr << " (" 
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< " -> "
				<< reg ->GetMetric()->GetValue(reg->GetLastTransformParameters()) 
				<< ") " << opt->GetStopConditionDescription() << std::endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
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
