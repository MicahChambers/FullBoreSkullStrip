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

//todo deal with NAN in neighbors 
//todo figure out how to deal with unlabeled neighbors/ indicate unlabeled neighbors

#include "itkImageMomentsCalculator.h"
#include "itkN4BiasFieldCorrectionImageFilter.h"
#include "itkMRIBiasFieldCorrectionFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAbsImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkCropImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkPointSet.h"

#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>

#include "itkFourier.h"
#include "version.h"

#include "stdafx.h"
#include "dataanalysis.h"

#include "registerIO.h"
RegisterIO REG;

using itk::Image;
using std::map;
using std::vector;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

typedef itk::Image<float,3> FImageT;
typedef itk::Image<itk::Vector<float,1>,3> VImageT;
typedef itk::Image<int,3> LImageT;

/**************************************************
 *  Bias Field Correction 
 *************************************************/

// takes a high res and low res image, and labelmap and calculates the bias field 
FImageT::Pointer calcBiasField(FImageT::Pointer lowres, FImageT::Pointer highres, 
		LImageT::Pointer labelmap);
// apply a bias field 
FImageT::Pointer applyBiasField(FImageT::Pointer highres, FImageT::Pointer& fieldmap);

/*************************************************
 * Segmentation
 **********************************************/

// computes the segmentation based on k-means clustering of intensities
LImageT::Pointer kmeans(FImageT::Pointer in, int clusters);

// choose the most brain-like label
int chooseLabel(Image<int,3>::Pointer labelmap);


/*************************************************
 * Primary Function 
 **********************************************/
FImageT::Pointer biasCorrect(FImageT::Pointer in)
{
	Image<float,3>::Pointer fieldmap;
	cerr << "Calculating Bias Field" << endl;
	int uselabel = 0;

	FImageT::SizeType osz;
	for(int ii = 0 ; ii < 3; ii++)
		osz[ii] = in->GetRequestedRegion().GetSize()[ii]*
			in->GetSpacing()[ii]/5;

	cerr << "Resizing input " << osz << endl;
	auto lowres = resize<FImageT>(in, osz, tukey);
	auto labelmap = kmeans(lowres, 2); //essentially otsu
	uselabel = chooseLabel(labelmap);

	cerr << "Converting labelmap to mask" << endl;
	itk::ImageRegionIteratorWithIndex<Image<int,3>> lit(labelmap, 
			labelmap->GetRequestedRegion());
	for(lit.GoToBegin(); !lit.IsAtEnd(); ++lit) {
		if(lit.Get() == uselabel) 
			lit.Set(1);
		else 
			lit.Set(0);
	}

	cerr << "Bias Field Correcting" << endl;
	fieldmap = calcBiasField(lowres, in, labelmap);

	/* 
	 * Bias Field Correction 
	 */

	cerr << "Applying Bias Field" << endl;
	return applyBiasField(in, fieldmap);
}

FImageT::Pointer calcBiasField(FImageT::Pointer lowres, FImageT::Pointer highres, 
		LImageT::Pointer labelmap)
{
	typedef itk::N4BiasFieldCorrectionImageFilter<FImageT,LImageT,FImageT> N4BFCType;

	//iterations 
	auto biascorr = N4BFCType::New(); 

	/* 
	 * Settings 
	 */

	int splineorder = 3;
	int niter = 1000;
	auto iters = biascorr->GetMaximumNumberOfIterations();
	for(unsigned ii = 0 ; ii < iters.Size(); ii++)
		iters[ii] = niter;
	biascorr->SetMaximumNumberOfIterations(iters);

	biascorr->SetConvergenceThreshold(0.0001);
	biascorr->SetSplineOrder(splineorder);
//	biascorr->SetWienerFilterNoise(0.01);
//	biascorr->SetInputMask(labelmap);
	biascorr->SetMaskLabel(1);

	biascorr->SetInput(lowres);
	biascorr->SetMaskImage(labelmap);
	
	/* 
	 * Run it
	 */
	cerr << "Bias Correcting " << std::endl;
	biascorr->Update();
	cerr << "Done." << std::endl;
	
	/* 
	 * Run the correction on the high res image
	 */

	// first turn the vector lettace into scalars (because the two don't agree)
	VImageT::Pointer vlattice = biascorr->GetLogBiasFieldControlPointLattice();

	// F the guy who made it basically mandatory to use:
	// itk::N4BiasFieldCorrectionImageFilter<FImageT>::BiasFieldControlPointLatticeType
	auto bspliner = itk::BSplineControlPointImageFilter<
				N4BFCType::BiasFieldControlPointLatticeType,
				N4BFCType::ScalarImageType>::New();

	bspliner->SetInput(biascorr->GetLogBiasFieldControlPointLattice());
	bspliner->SetSplineOrder(biascorr->GetSplineOrder());
	bspliner->SetSize(highres->GetLargestPossibleRegion().GetSize());
	bspliner->SetOrigin(highres->GetOrigin());
	bspliner->SetDirection(highres->GetDirection());
	bspliner->SetSpacing(highres->GetSpacing());
	bspliner->Update();
	auto vfield = bspliner->GetOutput();

	auto sfield= FImageT::New();

	sfield->SetRegions(vfield->GetLargestPossibleRegion());
	sfield->SetDirection(vfield->GetDirection());
	sfield->SetOrigin(vfield->GetOrigin());
	sfield->SetSpacing(vfield->GetSpacing());
	sfield->Allocate();

	itk::ImageRegionIterator<FImageT> sit(sfield, 
				sfield->GetLargestPossibleRegion());
	itk::ImageRegionIterator<VImageT> vit(vfield, 
				vfield->GetLargestPossibleRegion());
	for(vit.GoToBegin(), sit.GoToBegin(); !vit.IsAtEnd(); ++vit, ++sit) {
		sit.Set(vit.Get()[0]);
	}

	return sfield;
}

FImageT::Pointer applyBiasField(FImageT::Pointer highres, 
		FImageT::Pointer& fieldmap)
{
	auto out = FImageT::New();
	out->SetRegions(highres->GetRequestedRegion());
	out->SetSpacing(highres->GetSpacing());
	out->SetDirection(highres->GetDirection());
	out->SetOrigin(highres->GetOrigin());
	out->Allocate();

	itk::ImageRegionIterator<FImageT> iit(highres, highres->GetRequestedRegion());
	itk::ImageRegionIterator<FImageT> oit(out, out->GetRequestedRegion());
	itk::ImageRegionIterator<FImageT> bit(fieldmap, fieldmap->GetRequestedRegion());

	for(bit.GoToBegin(), iit.GoToBegin(), oit.GoToBegin(); !iit.IsAtEnd() ; 
				++bit, ++iit, ++oit) {
		oit.Set(iit.Get()/exp(bit.Get()));
	}

	return out;
}

int chooseLabel(Image<int,3>::Pointer labelmap)
{
	//calculate statistics for each label
	//0 - volume (also the number of points for centroid)
	//1,2,3 - centroid [3]
	//4,5,6 - second moments
	itk::ImageRegionIteratorWithIndex<Image<int,3>> lit(labelmap, 
			labelmap->GetRequestedRegion());
	map<int, vector<double>> stats; 
	lit.GoToBegin();
	for(int pp = 0; !lit.IsAtEnd(); pp++, ++lit) {
		Image<float,3>::IndexType index = lit.GetIndex();
		Image<float,3>::PointType point;
		labelmap->TransformIndexToPhysicalPoint(index, point);

		auto mit = stats.insert(std::pair<int, vector<double>>(
					lit.Get(), vector<double>()));
		if(mit.second) { //if it was inserted
			mit.first->second.resize(7);
			mit.first->second[0] = 1;
			mit.first->second[1] = point[0];
			mit.first->second[2] = point[1];
			mit.first->second[3] = point[2];
			mit.first->second[4] = point[0]*point[0];
			mit.first->second[5] = point[1]*point[1];
			mit.first->second[6] = point[2]*point[2];
		} else { //otherwise
			mit.first->second[0]++;
			mit.first->second[1] += point[0];
			mit.first->second[2] += point[1];
			mit.first->second[3] += point[2];
			mit.first->second[4] += point[0]*point[0];
			mit.first->second[5] += point[1]*point[1];
			mit.first->second[6] += point[2]*point[2];
		}
	}

	for(auto it = stats.begin(); it != stats.end(); it++) {
		cerr << "stats" << std::endl;
		double n = it->second[0];

		//mean
		for(int ii = 0 ; ii < 3; ii++)
			it->second[ii+1] /= n;

		//variance
		for(int ii = 0 ; ii < 3; ii++)
			it->second[ii+4] = it->second[ii+4]/(n-1) - 
				pow(it->second[ii+1],2)*n/(n-1);

		cerr << "Label: " << it->first << ", Volume: " << n 
			<< ", Moments: " << std::endl;
		for(int ii = 0 ; ii < 3; ii++)
			cerr << "(" << it->second[ii+1] << "," << it->second[ii+4]
				<< ")" << std::endl;
	}

	double max = 0;
	int	labelmax = 0;
	double val = 0;
	for(auto it = stats.begin(); it != stats.end(); it++) {
		val = pow(it->second[0],.5)/(pow(it->second[4],2) + pow(it->second[5],2) +
				pow(it->second[6], 2));
		cerr << it->first << ": " << val << std::endl;
		if(val > max) {
			max = val;
			labelmax = it->first;
		}
	}
	cerr << "Bias Correcting based on label " << labelmax << std::endl;
	return labelmax;
}


/**
 * @brief Performs k-means on the non-zero voxels in the image.
 *
 * @param in
 * @param clusters
 *
 * @return 
 */
LImageT::Pointer kmeans(FImageT::Pointer in, int clusters)
{
	//each image, form dimensions
	FImageT::SizeType sz = in->GetRequestedRegion().GetSize();
	int nvoxels = sz[0]*sz[1]*sz[2];
	alglib::real_2d_array samples;
	alglib::real_1d_array means;
	
	cerr << "Allocating Grouping Vector " << nvoxels << std::endl; 
	vector<int> group(nvoxels);
	
	cerr << "Allocating Sample Matrix " << nvoxels << std::endl; 
	samples.setlength(nvoxels, 1);

	cerr << "Filling " << 1 << "-dimensional samples" << std::endl; 
	itk::ImageRegionIteratorWithIndex<FImageT> rit(in, in->GetRequestedRegion());
	rit.GoToBegin();
	for(int pp = 0; !rit.IsAtEnd(); pp++, ++rit) {
		samples(pp, 0) = rit.Get();
	}
	cerr << "Done" << std::endl;

	//normalize 
	cerr << "Normalizing" << std::endl;
	for(int dd = 0 ; dd < samples.cols(); dd++) {
		double mean = 0;
		double var = 0;
//		int count = 0;
		for(int pp = 0 ; pp < samples.rows(); pp++) {
			mean += samples(pp, dd);
			var += samples(pp, dd)*samples(pp, dd);
		}

		mean /= samples.rows();
		var = sqrt(var/samples.rows()- mean*mean);

		//normalize
		for(int pp = 0 ; pp < samples.rows(); pp++)
			samples(pp, dd) = (samples(pp, dd) - mean)/var;
	}
	cerr << "Done" << std::endl;

	alglib::clusterizerstate s;
	alglib::kmeansreport rep;
	cerr << "\tCreating Clusterizer" << endl;
	alglib::clusterizercreate(s);
	alglib::clusterizersetkmeanslimits(s, 3, 10);

	cerr << "\tAdding points to clusterizer" << endl;
	alglib::clusterizersetpoints(s, samples, 2);

	cerr << "Running Clusterizer" << endl;
	alglib::clusterizerrunkmeans(s, clusters, rep);
	cerr << "Done with Clusterizer" << endl;

	cerr << "\tTermination Type: " << rep.terminationtype << endl;
	
	cerr << "\tCluster centers" << endl;
	for(int ii = 0 ; ii < rep.k; ii++) {
		cerr << "\t" << rep.c(ii, 0) << ", ";
		cerr << endl;
	}

	/*
	 * Create Labelmap
	 */
	
	//match iteration from beginning, just overwrite input
	auto labelmap = itk::Image<int, 3>::New();
	labelmap->SetRegions(in->GetRequestedRegion());
	labelmap->CopyInformation(in);
	labelmap->Allocate();
	itk::ImageRegionIteratorWithIndex<itk::Image<int,3>> lit(labelmap,
			labelmap->GetRequestedRegion());
	lit.GoToBegin();
	for(int pp = 0; !lit.IsAtEnd(); pp++, ++lit) {
		lit.Set(rep.cidx[pp]);
	}
	
	return labelmap;
}
