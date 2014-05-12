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
#include "itkFourier.h"
RegisterIO REG;


#include <itkNearestNeighborInterpolateImageFunction.h>
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

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#include <itkDiscreteGaussianImageFilter.h>
#include <itkLBFGSBOptimizer.h>

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

double rigidReg(itk::VersorRigid3DTransform<double>::Pointer tfm,
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
	bool operator()(const itk::VersorRigid3DTransform<double>::ParametersType& lhs,
			const itk::VersorRigid3DTransform<double>::ParametersType& rhs)  const
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

/******************************************
 * Main
 ******************************************/
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
	TCLAP::ValueArg<string> a_atlas("a", "atlas", "Atlas with same modality as input", 
			true, "", "image", cmd);
	TCLAP::ValueArg<string> a_atlas_label("m", "atlas-label", 
			"Brain-mask image in same space as atlas", true, "", "image", cmd);
	TCLAP::ValueArg<string> a_outl("o", "output", "Output brain mask in same "
			"space as input", false, "", "image", cmd);
	TCLAP::ValueArg<string> a_out("O", "output-fit", "Output brain image after "
			"transformation, in same space as input", false, "", "image", cmd);
	
	TCLAP::SwitchArg a_moments("M", "moments-align", "Moments align first.", cmd);
	
	TCLAP::MultiArg<double> a_rigid_smooth("R", "rigid-smooth",
			"Gaussian smoothing standard deviation during rigid "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	TCLAP::MultiArg<double> a_affine_smooth("A", "affine-smooth",
			"Gaussian smoothing standard deviation during affine "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	TCLAP::MultiArg<double> a_bspline_smooth("B", "bspline-smooth",
			"Gaussian smoothing standard deviation during bspline "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	
	TCLAP::SwitchArg  a_debug("D", "debug",
			"Write out intermediate images in the current directory "
			"(warning may overwrite stuff)", cmd);
	
	cmd.parse(argc, argv);

	auto input = readImage<ImageT>(a_in.getValue());
	auto atlas = readImage<ImageT>(a_atlas.getValue());
	auto labelmap = readImage<LImageT>(a_atlas_label.getValue());
	
	vector<double> rigid_smooth(a_rigid_smooth.getValue());
	vector<double> affine_smooth(a_affine_smooth.getValue());
	vector<double> bspline_smooth(a_bspline_smooth.getValue());

	if(!a_rigid_smooth.isSet()) {
		rigid_smooth.resize(3);
		rigid_smooth[0] = 3;
		rigid_smooth[1] = 1.5;
		rigid_smooth[2] = 0.5;
	}
	
	if(!a_affine_smooth.isSet()) {
		affine_smooth.resize(3);
		affine_smooth[0] = 3;
		affine_smooth[1] = 1.5;
		affine_smooth[2] = 0.5;
	}
	
	if(!a_bspline_smooth.isSet()) {
		bspline_smooth.resize(3);
		bspline_smooth[0] = 3;
		bspline_smooth[1] = 1.5;
		bspline_smooth[2] = 0.5;
	}

	auto rigid = itk::VersorRigid3DTransform<double>::New();
	if(a_moments.isSet()) {
		auto init = itk::CenteredVersorTransformInitializer<ImageT, ImageT>::New();
		init->SetTransform(rigid);
		init->SetFixedImage(input);
		init->SetMovingImage(atlas);
		init->MomentsOn();
		init->InitializeTransform();
	}
	
	auto result = apply(rigid.GetPointer(), atlas, input);
	writeImage<ImageT>("rigidinit.nii.gz", result);
	
	itk::Vector<double,3> xaxis((double)0);
	itk::Vector<double,3> yaxis((double)0);
	itk::Vector<double,3> zaxis((double)0);
	itk::Vector<double,3> zero((double)0);
	xaxis[0] = 1;
	yaxis[1] = 1;
	zaxis[2] = 1;

	auto xrigid = itk::VersorRigid3DTransform<double>::New();
	auto yrigid = itk::VersorRigid3DTransform<double>::New();
	auto zrigid = itk::VersorRigid3DTransform<double>::New();
	xrigid->SetFixedParameters(rigid->GetFixedParameters());
	yrigid->SetFixedParameters(rigid->GetFixedParameters());
	zrigid->SetFixedParameters(rigid->GetFixedParameters());
	
	xrigid->SetRotation(xaxis, PI/2.);
	yrigid->SetRotation(yaxis, PI/2.);
	zrigid->SetRotation(zaxis, PI/2.);
	xrigid->SetTranslation(zero);
	yrigid->SetTranslation(zero);
	zrigid->SetTranslation(zero);

	/* Rigid registration, try all different directions, but do it at low res */

	ImageT::SizeType osz;
	for(int ii = 0 ; ii < 3; ii++)
		osz[ii] = input->GetRequestedRegion().GetSize()[ii]*
			input->GetSpacing()[ii]/5;
	auto lr_input = resize<ImageT>(input, osz, tukey);
	
	for(int ii = 0 ; ii < 3; ii++)
		osz[ii] = atlas->GetRequestedRegion().GetSize()[ii]*
			input->GetSpacing()[ii]/5;
	auto lr_atlas = resize<ImageT>(atlas, osz, tukey);

	double bestval = 0;
	auto bestparams = rigid->GetParameters();
	std::set<itk::VersorRigid3DTransform<double>::ParametersType, ParamLessEqual> tested;
	for(int xx=0; xx < 4; xx++) {
		rigid->Compose(xrigid, true); // rotate 90degress
		for(int yy=0; yy < 4; yy++) {
			rigid->Compose(yrigid, true); //rotate 90 degress
			for(int zz=0; zz < 4; zz++) {
				rigid->Compose(zrigid, true); // rotate 90 degrees

				auto it = tested.insert(rigid->GetParameters());
				if(!it.second) {
					cerr << xx << "," << yy << "," << zz << endl;
					cerr << "Count > 0" << endl;
					continue;
				}

//				cerr << rigid->GetTranslation() << endl;
//				cerr << rigid->GetMatrix() << endl;
				cerr << rigid->GetParameters() << endl;
				// perform full registration
				double val = rigidReg(rigid, lr_atlas, lr_input, 4,
						true, 100, 0.01, .1, 0, 0.7, 0, 0.001);
//				cerr << rigid->GetTranslation() << endl;
//				cerr << rigid->GetMatrix() << endl;
				cerr << rigid->GetParameters() << endl << endl;
				if(val > bestval) {
					bestval = val;
					bestparams = rigid->GetParameters();
				}
			}
		}
	}
	
	result = apply(rigid.GetPointer(), atlas, input);
	writeImage<ImageT>("rigidResult_ai.nii.gz", result);
	
	result = apply(rigid.GetPointer(), input, atlas);
	writeImage<ImageT>("rigidResult_ia.nii.gz", result);
	
	result = apply(rigid.GetPointer(), lr_atlas, lr_input);
	writeImage<ImageT>("rigidResult_l_ai.nii.gz", result);
	
	result = apply(rigid.GetPointer(), lr_input, lr_atlas);
	writeImage<ImageT>("rigidResult_l_ia.nii.gz", result);

	// perform real rigid registration with the best parameters from above
	rigid->SetParameters(bestparams);
	cerr << rigid->GetTranslation() << endl;
	cerr << rigid->GetMatrix() << endl;
	cerr << rigid->GetParameters() << endl;
	// perform full registration
	for(size_t ii=0; ii<rigid_smooth.size(); ii++) {
		rigidReg(rigid, atlas, input, rigid_smooth[ii], 
				true, 1000, 0.01, 1, 0, 0.4, 0, 0.01);
	}
	cerr << rigid->GetTranslation() << endl;
	cerr << rigid->GetMatrix() << endl;
	cerr << rigid->GetParameters() << endl;

	/* Affine registration */
	auto affine = itk::AffineTransform<double, 3>::New();
	affine->SetTranslation(rigid->GetTranslation());
	affine->SetMatrix(rigid->GetMatrix());
	affine->SetCenter(rigid->GetCenter());

	for(int ii=0; affine_smooth.size(); ii++) {
		affineReg(affine, atlas, input, affine_smooth[ii], 
				true, 1000, 0.01, 1, 0, 0.4, 0, 0.01);
	}

	atlas = apply(affine.GetPointer(), atlas, input);
	labelmap = applyNN(affine.GetPointer(), labelmap, input);
	
	/* BSpline Registration */
	auto bspline = itk::BSplineTransform<double, 3, 3>::New();
	for(int ii=0; bspline_smooth.size(); ii++) {
		bSplineReg(bspline, atlas, input, bspline_smooth[ii], 
				true, 1000, 0.01, 1, 0, 0.4, 0, 0.01);
	}
	atlas = apply(bspline.GetPointer(), atlas, input);
	labelmap = applyNN(bspline.GetPointer(), labelmap, input);

	writeImage<ImageT>(a_out.getValue(), atlas);
	writeImage<LImageT>(a_outl.getValue(), labelmap);
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
	return 0;
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
  
	double value = INFINITY;
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << value << ") " << opt->GetStopConditionDescription() << endl;
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
	if(sd > 0) {
		source = gaussianSmooth<ImageT>(source, sd);
		target = gaussianSmooth<ImageT>(target, sd);
	}

	auto interp = itk::LinearInterpolateImageFunction<ImageT>::New();
	auto reg = itk::ImageRegistrationMethod<ImageT, ImageT>::New();

//	auto opt = itk::RegularStepGradientDescentOptimizer::New();
//	auto opt = itk::VersorRigid3DTransformOptimizer::New();
	auto opt = itk::LBFGSBOptimizer::New();
	itk::LBFGSBOptimizer::BoundSelectionType bsel(tfm->GetNumberOfParameters());
  	itk::LBFGSBOptimizer::BoundValueType bnull(tfm->GetNumberOfParameters());
	bsel.Fill( 0 );
	bnull.Fill( 0.0 );
	opt->SetBoundSelection(bsel);
	opt->SetUpperBound(bnull);
	opt->SetLowerBound(bnull);
	opt->SetCostFunctionConvergenceFactor( 1e+7);
	opt->SetProjectedGradientTolerance( 1e-6 );
	opt->SetMaximumNumberOfIterations(200);
	opt->SetMaximumNumberOfEvaluations(30);
	opt->SetMaximumNumberOfCorrections(5);
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
		scales[ii] = .0001;

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
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << value << ") " << opt->GetStopConditionDescription() << endl;
	} catch( itk::ExceptionObject & err ) {
		std::cerr<< "ExceptionObject" << std::endl << err << std::endl;
		return -INFINITY;
	}

	tfm->SetParameters(reg->GetLastTransformParameters());
	return -value;
}

double rigidReg(itk::VersorRigid3DTransform<double>::Pointer tfm,
		itk::Image<float,3>::Pointer source, itk::Image<float,3>::Pointer target, 
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
  
	double value = INFINITY;
	try {
		std::cerr << "Performing Rigid Registration..."; 
		reg->Update();
		value = reg ->GetMetric()->GetValue(reg->GetLastTransformParameters());
		cerr << " (" << value << ") " << opt->GetStopConditionDescription() << endl;
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

/* Helper function to write an image "out" to prefix + filename */
template <typename T, typename R = T>
void writeImage(std::string name, typename T::Pointer in)
{
	auto cast = itk::CastImageFilter<T, R>::New();
	cast->SetInput(in);
	cast->Update();

    typename itk::ImageFileWriter<T>::Pointer writer;
    writer = itk::ImageFileWriter<T>::New();
    writer->SetFileName(name);
    writer->SetInput(cast->GetOutput());
    writer->Update();
}

template <typename T, typename R = T>
void writeImage( typename T::Pointer in, std::string name)
{
	writeImage<T, R>(name, in);
}

template <typename T>
typename T::Pointer readImage(std::string name)
{
    typename itk::ImageFileReader<T>::Pointer reader;
    reader = itk::ImageFileReader<T>::New();
    reader->SetFileName( name );
    reader->Update();
    return reader->GetOutput();
}

