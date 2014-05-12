#include "itkGDCMImageIOFactory.h"
#include "itkImageIOFactory.h"
#include "itkHDF5ImageIOFactory.h"
#include "itkGiplImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"
#include "itkTxtTransformIOFactory.h"
#include "itkVTKImageIOFactory.h"
#include "itkBioRadImageIOFactory.h"
#include "itkLSMImageIOFactory.h"
#include "itkSiemensVisionImageIOFactory.h"
#include "itkBMPImageIOFactory.h"
#include "itkStimulateImageIOFactory.h"
#include "itkPNGImageIOFactory.h"
#include "itkTransformIOFactory.h"
#include "itkJPEGImageIOFactory.h"
#include "itkBYUMeshIOFactory.h"
#include "itkOBJMeshIOFactory.h"
#include "itkVTKPolyDataMeshIOFactory.h"
#include "itkFreeSurferAsciiMeshIOFactory.h"
#include "itkGiftiMeshIOFactory.h"
#include "itkFreeSurferBinaryMeshIOFactory.h"
#include "itkMeshIOFactory.h"
#include "itkOFFMeshIOFactory.h"
#include "itkGE5ImageIOFactory.h"
#include "itkGEAdwImageIOFactory.h"
#include "itkGE4ImageIOFactory.h"
#include "itkNrrdImageIOFactory.h"
#include "itkHDF5TransformIOFactory.h"
#include "itkTIFFImageIOFactory.h"
#include "itkMatlabTransformIOFactory.h"
#include "version.h"
#include "registerIO.h"

RegisterIO::RegisterIO()
{
	itk::ObjectFactoryBase::RegisterFactory(itk::NiftiImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GDCMImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::HDF5ImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GiplImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::NiftiImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::TxtTransformIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::VTKImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::BioRadImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::LSMImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::SiemensVisionImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::BMPImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::StimulateImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::PNGImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::JPEGImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::BYUMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::OBJMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::VTKPolyDataMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::FreeSurferAsciiMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::OFFMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GiftiMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::FreeSurferBinaryMeshIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GE5ImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GEAdwImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::GE4ImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::NrrdImageIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::TIFFImageIOFactory::New());
	
	itk::ObjectFactoryBase::RegisterFactory(itk::HDF5TransformIOFactory::New());
	itk::ObjectFactoryBase::RegisterFactory(itk::MatlabTransformIOFactory::New());
	
}
