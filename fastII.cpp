#include <iostream>
#include <algorithm>
#include <DGtal/base/Common.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/shapes/GaussDigitizer.h>
#include <DGtal/io/viewers/Viewer3D.h>
#include <DGtal/shapes/ShapeFactory.h>
#include <DGtal/io/writers/VolWriter.h>
#include <DGtal/io/readers/VolReader.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;

using namespace DGtal;
using namespace Z3i;


#include "FFT.h"
#include "IFFT.h"

/**
 * Missing parameter error message.
 *
 * @param param
 */
void missingParam ( std::string param )
{
  trace.error() <<" Parameter: "<<param<<" is required..";
  trace.info() <<std::endl;
  exit ( 1 );
}

/**
 * Linear remapping double (min,max)-> [0,255]
 */
struct ReMap
{
  ReMap(const double min, const double max): mymin(min),mymax(max)
  {}
  unsigned char operator()(const double val) const
  {
    return static_cast<unsigned char>((val-mymin)/(mymax-mymin)*255.0);
  }
  double mymin,mymax;
};

/**
 * Testing DGtal/FFTW API
 *
 */
int main(int argc, char **argv)
{
  // parse command line ----------------------------------------------
  po::options_description general_opt ( "Allowed options are: " );
  general_opt.add_options()
  ( "help,h", "display this message." )
  ( "radius,r", po::value<double>(),"Radius of the convolution kernel." )
  ( "input,i", po::value<std::string>(),"Input vol file." );
  
  bool parseOK=true;
  
  po::variables_map vm;
  try{
    po::store(po::parse_command_line(argc, argv, general_opt), vm);
  }catch(const std::exception& ex){
    parseOK=false;
    trace.info()<< "Error checking program options: "<< ex.what()<< endl;
  }
  
  po::notify ( vm );
  if ( !parseOK || vm.count ( "help" ) ||argc<=1 )
  {
    trace.info() << "Fast integral invariant convolution using FFT"<<std::endl
    << general_opt << "\n";
    return 0;
  }
  //Parse options
  if ( ! ( vm.count ( "input" ) ) ) missingParam ( "--input" );
  const std::string input = vm["input"].as<std::string>();
  if ( ! ( vm.count ( "radius" ) ) ) missingParam ( "--radius" );
  const double  radius = vm["radius"].as<double>();
  
  
  typedef ImageContainerBySTLVector<Domain, double> Image;

  //Source vol
  trace.beginBlock("Loading Vol file and generating kernels");
  Image vol = VolReader<Image>::importVol(input);
  Domain domain=vol.domain();
  Point mid = (domain.lowerBound()+domain.upperBound())/2;
  //Shifting the domain
  Image inputVol( Domain(domain.lowerBound() - mid,
                         domain.upperBound() - mid));
  for(auto it=inputVol.domain().begin(), itend= inputVol.domain().end();
      it != itend; ++it)
    if (vol(*it+mid) > 0)
      inputVol.setValue(*it, 1);
  domain = inputVol.domain();
  
  
  //Kernel
  Image kernel(inputVol.domain());
  for(auto it=kernel.begin(); it != kernel.end(); ++it)
    *it = 0.0;
  typedef ImplicitBall<Z3i::Space> Shape3D;
  Shape3D aShape( Point(0,0,0), radius);
  typedef GaussDigitizer<Z3i::Space,Shape3D> Gauss;
  Gauss dig;
  dig.attach( aShape );
  dig.init( aShape.getLowerBound()+Z3i::Vector(-1,-1,-1),
           aShape.getUpperBound()+Z3i::Vector(1,1,1), 1.0 );


  //double volume = 4.0/3.0*M_PI*radius*radius*radius;
  for(Z3i::Domain::ConstIterator it = dig.getDomain().begin() ; it != dig.getDomain().end();
      ++it)
  {
    Z3i::Point P = *it;
    if (dig(P))
      kernel.setValue( P , 1.0 );
  }
  trace.info()<<"Input: "<<inputVol<<std::endl;
  trace.info()<<"Kernel: "<<kernel<<std::endl;
  trace.endBlock();
  
  //FFT
  typedef FFT< Image > FFT3D;
  FFT3D fftV(inputVol);
  FFT3D fftK(kernel);
  trace.beginBlock("Computing the two FFTs");
  FFT3D::ComplexImage fftVol(domain);
  FFT3D::ComplexImage fftKernel(domain);
  fftV.compute(fftVol);
  fftK.compute(fftKernel);
  trace.endBlock();
  
  //Product
  trace.beginBlock("Product in Fourier Space");
  for(auto it=fftVol.begin(), itK=fftKernel.begin() , itend = fftVol.end(); it != itend ; ++it, ++itK)
    (*it) *= ((*itK));
  trace.endBlock();
  
  //iFFT
  typedef IFFT<FFT3D::ComplexImage> IFFT3D;
  Image imagereconstructed(domain);
  IFFT3D ifft(fftVol);
  trace.beginBlock("Computing IFFT");
  ifft.compute(imagereconstructed);
  trace.info()<<"Convolution: "<<imagereconstructed<<std::endl;
  trace.endBlock();
  
  //just an export of the  images
  double max= * std::max_element(imagereconstructed.begin(), imagereconstructed.end());
  double min= * std::min_element(imagereconstructed.begin(), imagereconstructed.end());
  trace.info()<< "max= "<< max<<" min= "<<min<<std::endl;
  trace.beginBlock("Exporting...");
  VolWriter<Image,ReMap>::exportVol("convolution.vol", imagereconstructed, ReMap(min,max));
  trace.endBlock();
  
  //iFFT of the kernel, just to make sure
  typedef IFFT<FFT3D::ComplexImage> IFFT3D;
  Image imagereconstructedK(domain);
  IFFT3D ifftK(fftKernel);
  trace.beginBlock("Computing IFFT");
  ifftK.compute(imagereconstructedK);
  trace.info()<<"Kernel: "<<imagereconstructedK<<std::endl;
  trace.endBlock();
  max= * std::max_element(imagereconstructedK.begin(), imagereconstructedK.end());
  min= * std::min_element(imagereconstructedK.begin(), imagereconstructedK.end());
  trace.info()<< "max= "<< max<<" min= "<<min<<std::endl;
  trace.beginBlock("Exporting...");
  VolWriter<Image,ReMap>::exportVol("kernel.vol", imagereconstructedK, ReMap(min,max));
  trace.endBlock();
  
}
