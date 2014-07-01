#ifndef SKULLSTRIP_H
#define SKULLSTRIP_H

#include <itkBSplineTransform.h>
#include "itkAffineTransform.h"
#include "itkEuler3DTransform.h"

#include "itkImage.h"

const double PI = acos(-1);

typedef itk::Image<float,3> ImageT;
typedef itk::Image<short,3> LImageT;

ImageT::PointType getCenter(ImageT::Pointer in);

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

#endif
