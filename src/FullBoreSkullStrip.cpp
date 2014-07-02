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

#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkCastImageFilter.h>
#include <itkBSplineTransformInitializer.h>

#include <tclap/CmdLine.h>
#include "version.h"
#include "biasCorrect.h"
#include "skullStrip.h"

#include "registerIO.h"
RegisterIO REG;

using itk::Image;
using std::map;
using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

typedef itk::Image<float,3> ImageT;
typedef itk::Image<short,3> LImageT;

template <typename T, typename R = T>
void writeImage(std::string name, typename T::Pointer in);

template <typename T, typename R = T>
void writeImage( typename T::Pointer in, std::string name);

template <typename T>
typename T::Pointer readImage(std::string name);


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
	
//	TCLAP::SwitchArg a_moments("M", "moments-align", "Moments align first.", cmd);
	TCLAP::SwitchArg a_reorient("X", "bad-orient", "If the orientation is "
			"incorrect, then this will restart in every 90 degree rotation", cmd);
	TCLAP::SwitchArg a_biascorr("b", "bias-correct", "Apply bias correction to "
			"input.", cmd);
	
	TCLAP::MultiArg<double> a_rigid_smooth("R", "rigid-smooth",
			"Gaussian smoothing standard deviation during rigid "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	TCLAP::ValueArg<double> a_rigidsteps("", "rigid-steps",
			"Maximum number of rigid steps.",  false, 1000, "iters", cmd);
	TCLAP::ValueArg<double> a_rigidminstep("", "rigid-min",
			"Minimum step size in rigid registration.", false,  0.0001, "step", cmd);
	TCLAP::ValueArg<double> a_rigidmaxstep("", "rigid-max",
			"Maximimum step size in rigid registration.", false, 0.01, "step",  cmd);

	TCLAP::MultiArg<double> a_affine_smooth("A", "affine-smooth",
			"Gaussian smoothing standard deviation during affine "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	TCLAP::ValueArg<double> a_affinesteps("", "affine-steps",
			"Maximum number of affine steps.", false, 100000, "iters", cmd);
	TCLAP::ValueArg<double> a_affineminstep("", "affine-min",
			"Minimum step size in affine registration.", false, 0.0001, "step", cmd);
	TCLAP::ValueArg<double> a_affinemaxstep("", "affine-max",
			"Maximimum step size in affine registration.", false, 0.01, "step", cmd);

	TCLAP::MultiArg<double> a_bspline_smooth("B", "bspline-smooth",
			"Gaussian smoothing standard deviation during bspline "
			"registration. Provide mutliple to schedule mutli-resolution "
			"registration strategy.", false, "double", cmd);
	TCLAP::ValueArg<double> a_bsplinesteps("", "bspline-steps",
			"Maximum number of bspline steps.", false, 1000, "iters", cmd);
	TCLAP::ValueArg<double> a_bsplineminstep("", "bspline-min",
			"Minimum step size in bspline registration.", false, 0.0001, "step", cmd);
	TCLAP::ValueArg<double> a_bsplinemaxstep("", "bspline-max",
			"Maximimum step size in bspline registration.", false, 0.01, "step", cmd);
	
	TCLAP::SwitchArg  a_debug("D", "debug",
			"Write out intermediate images in the current directory "
			"(warning may overwrite stuff)", cmd);
	
	cmd.parse(argc, argv);

	auto input = readImage<ImageT>(a_in.getValue());
	if(a_biascorr.isSet()) {
		input = biasCorrect(input);
		writeImage<ImageT>("bc.nii.gz", input);
	}

	auto atlas = readImage<ImageT>(a_atlas.getValue());
	auto labelmap = readImage<LImageT>(a_atlas_label.getValue());

	ImageT::PointType fixedCenter = getCenter(input);
	ImageT::PointType movingCenter = getCenter(atlas);;

	vector<double> rigid_smooth(a_rigid_smooth.getValue());
	vector<double> affine_smooth(a_affine_smooth.getValue());
	vector<double> bspline_smooth(a_bspline_smooth.getValue());
	std::set<itk::Array<double>, ParamLessEqual> tested;

	if(!a_rigid_smooth.isSet()) {
		rigid_smooth.resize(3);
		rigid_smooth[0] = 3;
		rigid_smooth[1] = 2; 
		rigid_smooth[2] = 1; 
	}
	
	if(!a_affine_smooth.isSet()) {
		affine_smooth.resize(3);
		affine_smooth[0] = 4;
		affine_smooth[1] = 3; 
		affine_smooth[2] = 2; 
	}
	
	if(!a_bspline_smooth.isSet()) {
		bspline_smooth.resize(3);
		bspline_smooth[0] = 3;
		bspline_smooth[1] = 2;
	}
	
	/* Affine registration, try all different directions, but do it at low res */
	auto rigid = itk::Euler3DTransform<double>::New();
	if(a_reorient.isSet()) {
		double bestval = -INFINITY;
		auto bestparams = rigid->GetParameters();
		for(int xx=0; xx < 4; xx++) {
			for(int yy=0; yy < 4; yy++) {
				for(int zz=0; zz < 4; zz++) {
					rigid->SetIdentity();
					rigid->SetTranslation(movingCenter-fixedCenter);
					rigid->SetCenter(fixedCenter);
					rigid->SetRotation(xx*PI/2., yy*PI/2., zz*PI/2.); 

					auto it = tested.insert(rigid->GetParameters());
					if(!it.second) {
						cerr << xx << "," << yy << "," << zz << endl;
						cerr << "Count > 0" << endl;
						continue;
					}

					cerr << rigid->GetParameters() << endl;
					// perform full registration
					double val = rigidReg(rigid, atlas, input, 5, 
							true, 5000, 0.001, 1, 0, 0.4, 0, 0.001);
					cerr << rigid->GetParameters() << endl << endl;
					if(val > bestval) {
						cerr << "New Best" << endl;
						bestval = val;
						bestparams = rigid->GetParameters();
					}
				}
			}
		}
		rigid->SetParameters(bestparams);
	} else {
		rigid->SetIdentity();
		rigid->SetTranslation(movingCenter-fixedCenter);
		rigid->SetCenter(fixedCenter);
	}
	
	/* Rigid registration */
	for(size_t ii=0; ii < rigid_smooth.size(); ii++) {
		rigidReg(rigid, atlas, input, rigid_smooth[ii], 
				true, a_rigidsteps.getValue(), a_rigidminstep.getValue(), 
				a_rigidmaxstep.getValue(), 0, 0.7, 0, 0.0001);

		std::ostringstream oss;
		oss << "rigid_" << rigid_smooth[ii] << ".nii.gz";
		auto tmp = apply(rigid.GetPointer(), atlas, input);
		writeImage<ImageT>(oss.str(), tmp);
	}
	
	/* Affine registration */
	auto affine = itk::AffineTransform<double, 3>::New();
	affine->SetIdentity();
	affine->SetMatrix(rigid->GetMatrix());
	affine->SetTranslation(rigid->GetTranslation());
	affine->SetCenter(rigid->GetCenter());

	auto tmp = apply(affine.GetPointer(), atlas, input);
	writeImage<ImageT>("affine_init.nii.gz", tmp);
	for(size_t ii=0; ii < affine_smooth.size(); ii++) {
		affineReg(affine, atlas, input, affine_smooth[ii], 
				true, a_affinesteps.getValue(), a_affineminstep.getValue(),
				a_affinemaxstep.getValue(), 0, 0.8, 0, 0.001);

		std::ostringstream oss;
		oss << "affine_" << affine_smooth[ii] << ".nii.gz";
		auto tmp = apply(affine.GetPointer(), atlas, input);
		writeImage<ImageT>(oss.str(), tmp);
	}

	atlas = apply(affine.GetPointer(), atlas, input);
	labelmap = applyNN(affine.GetPointer(), labelmap, input);

	writeImage<ImageT>("bspline_init.nii.gz", atlas);
	writeImage<ImageT>("bspline_target.nii.gz", input);
		
	
	// probably
	// do bias field correction based on the atlas

	/* BSpline Registration */
	auto bspline = itk::BSplineTransform<double, 3, 3>::New();
	{
		double space = 10;
		ImageT::SizeType size;
		for(size_t ii=0; ii<3; ii++) {
			size[ii] = input->GetRequestedRegion().GetSize()[ii]*
				input->GetSpacing()[ii]/space;
		}
		auto bspinit = itk::BSplineTransformInitializer<
			itk::BSplineTransform<double,3,3>,ImageT>::New();
		bspinit->SetImage(input);
		bspinit->SetTransform(bspline);
		bspinit->SetTransformDomainMeshSize(size);
		bspinit->InitializeTransform();
	}

	for(int ii=0; bspline_smooth.size(); ii++) {
		bSplineReg(bspline, atlas, input, bspline_smooth[ii], 
				true, a_bsplinesteps.getValue(), a_bsplineminstep.getValue(),
				a_bsplinemaxstep.getValue(), 0, 0.4, 0, 0.0001);

		std::ostringstream oss;
		oss << "bspline_" << affine_smooth[ii] << ".nii.gz";
		auto tmp = apply(bspline.GetPointer(), atlas, input);
		writeImage<ImageT>(oss.str(), tmp);
	}
	atlas = apply(bspline.GetPointer(), atlas, input);
	labelmap = applyNN(bspline.GetPointer(), labelmap, input);

	
	if(a_out.isSet()) {
		cerr << "Writing Deformed Atlas:" << endl << atlas << endl;
		writeImage<ImageT>(a_out.getValue(), atlas);
		cerr << "Done" << endl;
	}
	if(a_outl.isSet()) {
		cerr << "Writing Deformed Mask:" << endl << atlas << endl;
		writeImage<LImageT>(a_outl.getValue(), labelmap);
		cerr << "Done" << endl;
	}
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
	return 0;
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
