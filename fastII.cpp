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
    trace.info() << "Create space-time vol"<<std::endl
    << std::endl << "Basic usage: "<<std::endl
    << "\tcreateVol  --o <volOutputFileName>  ...;TODO....."<<std::endl
    << general_opt << "\n";
    return 0;
  }
  //Parse options
  if ( ! ( vm.count ( "input" ) ) ) missingParam ( "--input" );
  const std::string input = vm["input"].as<std::string>();
  if ( ! ( vm.count ( "radius" ) ) ) missingParam ( "--radius" );
  const double  radius = vm["radius"].as<double>();
  
  
  typedef ImageContainerBySTLVector<Domain, unsigned char> Image;

  //Source vol
  Image inputVol = VolReader<Image>::importVol(input);
  
  Domain domain=inputVol.domain();
  
  //Kernel
  Image kernel(inputVol);
  typedef ImplicitBall<Z3i::Space> Shape3D;
  Shape3D aShape( Point(64,64,64), radius);
  typedef GaussDigitizer<Z3i::Space,Shape3D> Gauss;
  Gauss dig;
  dig.attach( aShape );
  dig.init( aShape.getLowerBound()+Z3i::Vector(-1,-1,-1),
           aShape.getUpperBound()+Z3i::Vector(1,1,1), 1.0 );
  Z3i::Domain d3D = dig.getDomain();
  for(Z3i::Domain::ConstIterator it = d3D.begin() ; it != d3D.end();
      ++it)
  {
    Z3i::Point P = *it;
    if (dig(P))
      kernel.setValue( P , 128 );
  }
  
  
  //FFT
  typedef FFT< Image > FFT3D;
  FFT3D fftV(inputVol);
  FFT3D fftK(kernel);
  trace.beginBlock("Computing the two FFT");
  FFT3D::ComplexImage fftVol(domain);
  FFT3D::ComplexImage fftKernel(domain);
  fftV.compute(fftVol);
  fftK.compute(fftKernel);
  trace.endBlock();
  
  //Product
  for(auto it=fftVol.begin(), itK=fftKernel.begin() , itend = fftVol.end(); it != itend ; ++it, ++itK)
    (*it) *= (*itK);
  
  
  //iFFT
  typedef IFFT<FFT3D::ComplexImage> IFFT3D;
  Image imagereconstructed(domain);
  IFFT3D ifft(fftVol);
  trace.beginBlock("Computing IFFT");
  ifft.compute(imagereconstructed);
  trace.endBlock();
  
  //just an export of the reconstructed image
  VolWriter<Image  >::exportVol("original.vol", inputVol);
  VolWriter<Image  >::exportVol("inverse.vol", imagereconstructed);
  

  trace.info()<< "Export done."<<std::endl;

  
}
