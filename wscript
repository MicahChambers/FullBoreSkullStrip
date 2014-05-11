#!/bin/env python
from waflib.Utils import subprocess
import os
from waflib import Options, Node, Build, Configure
import re

out = 'build'

itklibs = 'itkdouble-conversion ITKBiasCorrection ITKBioCell ITKCommon \
    ITKDICOMParser ITKEXPAT ITKFEM ITKIOBMP ITKIOBioRad ITKIOCSV \
    ITKIOGDCM ITKIOGE ITKIOGIPL ITKIOHDF5 ITKIOIPL ITKIOImageBase ITKIOJPEG \
    ITKIOLSM ITKIOMesh ITKIOMeta ITKIONIFTI ITKIONRRD ITKIOPNG ITKIOSiemens \
    ITKIOSpatialObjects ITKIOStimulate ITKIOTIFF ITKIOTransformBase \
    ITKIOTransformHDF5 ITKIOTransformInsightLegacy ITKIOTransformMatlab \
    ITKIOVTK ITKIOXML ITKKLMRegionGrowing ITKLabelMap ITKMesh ITKMetaIO \
    ITKNrrdIO ITKOptimizers ITKPath ITKPolynomials ITKQuadEdgeMesh \
    ITKSpatialObjects ITKStatistics ITKVNLInstantiation ITKVTK \
    ITKVideoCore ITKVideoIO ITKWatersheds ITKgiftiio ITKniftiio ITKznz \
    itkNetlibSlatec itkgdcmCommon itkgdcmDICT itkgdcmDSED itkgdcmIOD itkgdcmMSFF \
    itkgdcmjpeg12 itkgdcmjpeg16 itkgdcmjpeg8 itkgdcmuuid itkhdf5 itkhdf5_cpp \
    itkjpeg itkopenjpeg itkpng itksys itktiff itkv3p_lsqr itkv3p_netlib itkvcl \
    itkvnl itkvnl_algo itkzlib \
        ITKCommon itkv3p_netlib itkvnl'.split() # these are important last ones
#    itkvnl_algo itkzlib'.split()

vtklibs = 'vtkIOXMLParser vtkIOXML vtkIOGeometry vtkIOLegacy vtkCommonComputationalGeometry vtkCommonCore \
    vtkCommonDataModel vtkCommonExecutionModel vtkCommonMath vtkCommonMisc \
    vtkCommonSystem vtkCommonTransforms vtkDICOMParser vtkFiltersCore \
    vtkFiltersExtraction vtkFiltersGeneral vtkFiltersGeometry vtkFiltersHybrid \
    vtkFiltersImaging vtkFiltersModeling vtkFiltersSources vtkFiltersStatistics \
    vtkFiltersTexture vtkjsoncpp\
    vtkIOCore vtkIOImage vtkImagingColor \
    vtkImagingCore vtkImagingFourier vtkImagingGeneral vtkImagingHybrid \
    vtkImagingMath vtkImagingMorphological vtkImagingSources vtkImagingStatistics \
    vtkImagingStencil vtkInfovisCore vtkInfovisLayout vtkInteractionImage \
    vtkInteractionStyle vtkInteractionWidgets \
    vtkRenderingAnnotation vtkRenderingCore \
    vtkRenderingFreeType vtkRenderingFreeTypeOpenGL \
    vtkRenderingImage vtkRenderingLOD vtkRenderingLabel \
    vtkRenderingOpenGL vtkRenderingVolume vtkRenderingVolumeOpenGL \
    vtkalglib vtkexpat vtkfreetype vtkftgl vtkjpeg vtkmetaio vtkoggtheora vtkpng \
    vtksqlite vtksys vtktiff vtkzlib vtkCommonCore vtkCommonDataModel '.split()

def cmakeparse(cmakefile):
    ifile = open(cmakefile)
    ind = ifile.read()
    ifile.close()
    print(ind)
    libs = re.search('LIBRARIES "([a-zA-Z0-9_;-]*)"', ind).group(1)
    libs = libs.split(";")
    return libs

def configure(conf):
    join = os.path.join
    isabs = os.path.isabs
    abspath = os.path.abspath
    
    opts = vars(conf.options)
    conf.load('compiler_cxx python')
    conf.load('etest', tooldir='test')

    env = conf.env

    ############################### 
    # Basic Compiler Configuration
    ############################### 
#   conf.check_python_version((3,0,0))
#   conf.check_python_headers()

    conf.env.RPATH = []
    if opts['enable_rpath'] or opts['enable_build_rpath']:
        for pp in ['fmri', 'graph', 'libs', 'point3d', 'R']:
            conf.env.RPATH.append(join('$ORIGIN', '..', pp))
    
    if opts['enable_rpath'] or opts['enable_install_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')
    
    conf.env.DEFINES = ['AE_CPU=AE_INTEL', 'VCL_CAN_STATIC_CONST_INIT_FLOAT=0', 'VCL_CAN_STATIC_CONST_INIT_INT=0']
    # for static build
    if opts['static']: 
        conf.env.CXXFLAGS = ['-Wall', '-std=c++11', '-static-libgcc', '-static-libstdc++']
        conf.env.LINKFLAGS = ['-static-libgcc', '-static-libstdc++']
        conf.env.STATIC_LINK = True
    else:
        conf.env.LINKFLAGS = []
        conf.env.CXXFLAGS = ['-Wall', '-std=c++11']
        conf.env.STATIC_LINK = False


    if opts['profile']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-Wno-unused-parameter', '-Wno-sign-compare', '-Wno-unused-local-typedefs', '-Wall', '-Wextra','-g', '-pg'])
        conf.env.LINKFLAGS.append('-pg')
    elif opts['debug']:
        conf.env.DEFINES.append('DEBUG=1')
        conf.env.CXXFLAGS.extend(['-Wno-unused-parameter', '-Wno-sign-compare', '-Wno-unused-local-typedefs', '-Wall', '-Wextra','-g'])
    elif opts['release']:
        conf.env.DEFINES.append('NDEBUG=1')
        conf.env.CXXFLAGS.extend(['-O3', '-march=nocona'])
    elif opts['native']:
        conf.env.DEFINES.append('NDEBUG=1')
        conf.env.CXXFLAGS.extend(['-O3', '-march=native'])
    
    conf.check(header_name='stdio.h', features='cxx cxxprogram', mandatory=True)

        
    ############################### 
    # Library Configuration
    ############################### 
    for DD in ['LAPACK', 'ITK', 'VTK', 'GSL', 'OPENCL', 'CUDA', 'MAGMA', 'R']: 
        # ITK -> itk (etc)
        dd = DD.lower()

        # Set the library dir from the options/base
        if opts[ dd+'ldir' ]:
            if isabs(opts[ dd+'ldir' ]):
                env[ 'LIBPATH_'+DD ] = opts[ dd+'ldir' ]
            else:
                env[ 'LIBPATH_'+DD ] = join(opts[ dd+'base' ], opts[ dd+'ldir' ])
        else:
            env[ 'LIBPATH_'+DD ] = join(opts[ dd+'base' ], 'lib')

        # Set the include dir from the options/base
        if opts[ dd+'idir' ]:
            if isabs(opts[ dd+'idir' ]):
                env[ 'INCLUDES_'+DD ] = opts[ dd+'idir' ]
            else:
                env[ 'INCLUDES_'+DD ] = join(opts[ dd+'base' ], opts[ dd+'idir' ])
        elif DD == 'ITK':
            #ITK/VTK have include/vtk-6.0 or include/ITK-4.3 for some reason so add that
            env[ 'INCLUDES_'+DD ] = join(opts[ dd+'base' ], 'include', DD+'-'+opts[ dd+'vers' ])
        elif DD == 'VTK':
            env[ 'INCLUDES_'+DD ] = join(opts[ dd+'base' ], 'include', dd+'-'+opts[ dd+'vers' ])
        else:
            env[ 'INCLUDES_'+DD ] = join(opts[ dd+'base' ], 'include')

    ## ITK ##
    if opts['static']:
        env.LIB_ITK = [ll+'-'+opts['itkvers'] for ll in itklibs]*2 + ['pthread', 'dl', 'c', 'stdc++']
    else:
        env.LIB_ITK = [ll+'-'+opts['itkvers'] for ll in itklibs]

    conf.check(lib=' '.join(env.LIB_ITK), 
                header_name = 'itkGDCMImageIOFactory.h', 
                mandatory = True, 
                use = 'ITK')

    ## VTK ##
    if opts['static']:
        env.LIB_VTK = [ll+'-'+opts['vtkvers'] for ll in vtklibs]*2 + ['pthread', 'dl']
    else:
        env.LIB_VTK = [ll+'-'+opts['vtkvers'] for ll in vtklibs]

    conf.check(lib=' '.join(env.LIB_VTK), 
                header_name = 'vtkArrayData.h', 
                mandatory = True, 
                use = 'VTK')

    ## GSL ##
    conf.check_cfg(package="", path="gsl-config", args="--cflags --libs", uselib_store="GSL")
    conf.check(lib = ' '.join(env.LIB_GSL), header_name = 'gsl/gsl_matrix.h', use="GSL")
    
    ## Open CL ##
    env.LIB_OPENCL = ['OpenCL']
    env.HAVE_OPENCL = conf.check(
                lib=' '.join(env.LIB_OPENCL), 
                header_name = 'CL/cl.h', 
                mandatory = False, 
                use ='OPENCL', var = 'HAVE_OPENCL',
                define_name = 'HAVE_OPENCL')
    
    ## CUDA ##
    env.LIB_CUDA = ['cuda', 'cublas', 'cudart']
    env.HAVE_CUDA= conf.check(
                lib=' '.join(env.LIB_CUDA), 
                header_name = 'cuda.h', 
                mandatory = False, 
                use ='CUDA', var = 'HAVE_CUDA',
                define_name = 'HAVE_CUDA')
    
    ## LAPACK ##
    env.LIB_LAPACK = ['lapack', 'blas', 'gfortran']
    conf.check(lib=' '.join(env.LIB_LAPACK), 
                mandatory=True, 
                use ='LAPACK', var = 'HAVE_LAPACK',
                define_name = 'HAVE_LAPACK')
    
    ## MAGMA ##
    env.LIB_MAGMA = ['magma', 'magmablas']
    env.HAVE_MAGMA = conf.check(
                lib=' '.join(env.LIB_MAGMA), 
                header_name = 'magma.h', 
                mandatory=False, var = 'HAVE_MAGMA',
                use=['LAPACK', 'CUDA', 'MAGMA'], 
                define_name = 'HAVE_MAGMA')
    
    ## R ## 
    if opts['no_r']:
        env.HAVE_R = False
    else:
        env.HAVE_R = conf.find_program("R")
        conf.check_cfg(path=env["R"], args="CMD config --cppflags", package="", uselib_store="R")
        conf.check_cfg(path=env["R"], args="CMD config --ldflags", package="", uselib_store="R")
        conf.check(lib = ' '.join(env.LIB_R), header_name = 'R.h', use="R LAPACK")
    
    ## Python ## 
#    conf.find_program("python3-config")
#    conf.check_cfg(path="python3-config", args="--cflags", package="", uselib_store="PYTHON3")
#    conf.check_cfg(path="python3-config", args="--ldflags", package="", uselib_store="PYTHON3")
#    env.LIBPATH_PYTHON3 = join(conf.check_cfg(path=env["PYTHON3-CONFIG"], 
#                   args="--prefix", package="").rstrip(), "lib")
#    conf.check(lib = ' '.join(env.LIB_PYTHON3), header_name = 'Python.h', use="PYTHON3")
    
    #Set correct data and input data directories for extended tests (etest)
    for dd in conf.path.ant_glob("test/testdata/correct/*"):
        pth = dd.relpath()
        conf.path.make_node(pth)
        print(pth)
    
    for dd in conf.path.ant_glob("test/testdata/in/*"):
        pth = dd.relpath()
        conf.path.make_node(pth)
        print(pth)

def options(ctx):
    ctx.load('compiler_cxx')
    ctx.load('etest', tooldir='test')

    gr = ctx.get_option_group('configure options')
    
    for DD in ['LAPACK', 'ITK', 'VTK', 'GSL', 'OPENCL', 'CUDA', 'MAGMA', 'R']: 
        # ITK -> itk (etc)
        dd = DD.lower()
        gr.add_option('--'+dd+'base', action='store', default = '/usr', help = DD+' Prefix Dir')

        # Add the Options
        gr.add_option('--'+dd+'ldir', action='store', default = False, 
                    help = DD+' Library Dir, can be relative to {PREFIX}')

        gr.add_option('--'+dd+'idir', action='store', default = False, 
                        help = DD+' Include Dir, can be relative to {PREFIX}')

    gr.add_option('--vtkvers', action='store', default = '6.1', help = 'VTK Version')
    gr.add_option('--itkvers', action='store', default = '4.6', help = 'ITK Version')
    
    gr.add_option('--enable-rpath', action='store_true', default = False, help = 'Set RPATH to build/install dirs')
    gr.add_option('--enable-install-rpath', action='store_true', default = False, help = 'Set RPATH to install dir only')
    gr.add_option('--enable-build-rpath', action='store_true', default = False, help = 'Set RPATH to build dir only')
    
    gr.add_option('--debug', action='store_true', default = False, help = 'Build with debug flags')
    gr.add_option('--profile', action='store_true', default = False, help = 'Build with debug and profiler flags')
    gr.add_option('--release', action='store_true', default = False, help = 'Build with tuned compiler optimizations')
    gr.add_option('--native', action='store_true', default = False, help = 'Build with highly specific compiler optimizations')
    gr.add_option('--no-r', action='store_true', default = False, help = "Don't build R modules")
    gr.add_option('--static', action='store_true', default = False, help = "Build statically (turns off R and python)")
    
def gitversion():
    if not os.path.isdir(".git"):
        print("This does not appear to be a Git repository.")
        return
    try:
        HEAD = open(".git/HEAD");
        headtxt = HEAD.read();
        HEAD.close();

        headtxt = headtxt.split("ref: ")
        if len(headtxt) == 2:
            fname = ".git/"+headtxt[1].strip();
            master = open(fname);
            mastertxt = master.read().strip();
            master.close();
        else:
            mastertxt = headtxt[0].strip()
    
    except EnvironmentError:
        print("unable to get HEAD")
        return "unknown"
    return mastertxt

def build(bld):
    with open("src/libs/version.h", "w") as f:
        f.write('#define __version__ "%s"\n\n' % gitversion())
        f.close()

    bld.recurse('src scripts')
    bld.load('etest', tooldir='test')
