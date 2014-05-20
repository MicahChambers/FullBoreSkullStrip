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

    conf.env.RPATH = []
    if opts['enable_rpath'] or opts['enable_build_rpath']:
        conf.env.RPATH.append('$ORIGIN')
    
    if opts['enable_rpath'] or opts['enable_install_rpath']:
        conf.env.RPATH.append('$ORIGIN/../lib')
    
    conf.env.DEFINES = ['AE_CPU=AE_INTEL', 'VCL_CAN_STATIC_CONST_INIT_FLOAT=0', 'VCL_CAN_STATIC_CONST_INIT_INT=0']
    conf.env.LINKFLAGS = ['-lm']
    # for static build
    if opts['static']: 
        conf.env.CXXFLAGS = ['-Wall', '-std=c++11', '-static-libgcc', '-static-libstdc++']
        conf.env.LINKFLAGS.extend('-static-libgcc', '-static-libstdc++')
        conf.env.STATIC_LINK = True
    else:
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
    for DD in ['ITK']: 
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

def options(ctx):
    ctx.load('compiler_cxx')
    ctx.load('etest', tooldir='test')

    gr = ctx.get_option_group('configure options')
    
    for DD in ['ITK']: 
        # ITK -> itk (etc)
        dd = DD.lower()
        gr.add_option('--'+dd+'base', action='store', default = '/usr', help = DD+' Prefix Dir')

        # Add the Options
        gr.add_option('--'+dd+'ldir', action='store', default = False, 
                    help = DD+' Library Dir, can be relative to {PREFIX}')

        gr.add_option('--'+dd+'idir', action='store', default = False, 
                        help = DD+' Include Dir, can be relative to {PREFIX}')

    gr.add_option('--itkvers', action='store', default = '4.6', help = 'ITK Version')
    
    gr.add_option('--enable-rpath', action='store_true', default = False, help = 'Set RPATH to build/install dirs')
    gr.add_option('--enable-install-rpath', action='store_true', default = False, help = 'Set RPATH to install dir only')
    gr.add_option('--enable-build-rpath', action='store_true', default = False, help = 'Set RPATH to build dir only')
    
    gr.add_option('--debug', action='store_true', default = False, help = 'Build with debug flags')
    gr.add_option('--profile', action='store_true', default = False, help = 'Build with debug and profiler flags')
    gr.add_option('--release', action='store_true', default = False, help = 'Build with tuned compiler optimizations')
    gr.add_option('--native', action='store_true', default = False, help = 'Build with highly specific compiler optimizations')
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
    with open("src/version.h", "w") as f:
        f.write('#define __version__ "%s"\n\n' % gitversion())
        f.close()

    bld.recurse('src scripts')
    bld.load('etest', tooldir='test')
