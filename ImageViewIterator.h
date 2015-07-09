#pragma once

#include <type_traits>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/assert.hpp>
#include "Linearizer.h"


namespace DGtal
{

/** Random access iterator over an image given his definition domain and viewable domain.
 *
 * In order to work, the iterable class must expose a dereference method that,
 * given a point and an index (column-major ordered), return:
 * - a mutable reference to the corresponding value for a mutable iterator.
 * - a copy or a constant reference to the corresponding value for a constant iterator.
 *
 * @warning The iterable class must have Domain typedef.
 *
 * @tparam TIterableClass   Type of the iterable class.
 */
template <
  typename TIterableClass
>
class ImageViewIterator
  : public boost::iterator_facade <
      ImageViewIterator<TIterableClass>,
      typename TIterableClass::Value,
      std::random_access_iterator_tag,
      decltype( ((TIterableClass*)nullptr)->dereference( TIterableClass::Point::diagonal(0), typename TIterableClass::Point::Coordinate(0) ) )
    >
  {
  public:
    using Self = ImageViewIterator<TIterableClass>;
    using IterableClass = TIterableClass;
    using Domain = typename IterableClass::Domain; // or in template with default value ?
    using Point = typename Domain::Point;
    using Linearizer = DGtal::Linearizer<Domain, ColMajorStorage>; // hard-coded, but must be later set as template.
    using Reference = decltype( ((IterableClass*)nullptr)->dereference( Point::diagonal(0), typename Point::Coordinate(0) ) );

    /// Default constructor.
    ImageViewIterator()  
        : myIterableClassPtr(nullptr)
      {}

    /** Iterator from a point.
     *
     * @param anIterableClassPtr  Pointer to the iterable class instance.
     * @param aFullDomain         Full domain of the image.
     * @param aViewDomain         Viewable domain that the iterator will span.
     * @param aPoint              Point to which the iterator will point.
     */
    ImageViewIterator( IterableClass* anIterableClassPtr, Domain const& aFullDomain, Domain const& aViewDomain, Point const& aPoint ) 
      : myIterableClassPtr( anIterableClassPtr )
      , myFullDomain{ aFullDomain }
      , myViewDomain{ aViewDomain }
      , myFullExtent( myFullDomain.upperBound() - myFullDomain.lowerBound() + Point::diagonal(1) )
      , myViewExtent( myViewDomain.upperBound() - myViewDomain.lowerBound() + Point::diagonal(1) )
      , myPoint{ aPoint }
      , myFullIndex( Self::Linearizer::getIndex( myPoint - myFullDomain.lowerBound(), myFullExtent ) )
      {
        BOOST_ASSERT_MSG(
               myFullDomain.lowerBound().isLower( myViewDomain.lowerBound() )
            && myFullDomain.upperBound().isUpper( myViewDomain.upperBound() ),
            "The viewable domain must be included into the full domain."
        );
        BOOST_ASSERT_MSG(
            myViewDomain.isInside(aPoint),
            "The point is outside the viewable domain !"
        );
      }

    /// Copy constructor with type interoperability.
    template < typename TOtherIterableClass >
    explicit ImageViewIterator( 
        ImageViewIterator<TOtherIterableClass> const& other,
        typename std::enable_if< std::is_convertible<TOtherIterableClass*, IterableClass*>::value >::type* = 0 
    )
      : myIterableClassPtr( other.myIterableClassPtr )
      , myFullDomain{ other.myFullDomain }
      , myViewDomain{ other.myViewDomain }
      , myFullExtent{ other.myFullExtent }
      , myViewExtent{ other.myViewExtent }
      , myPoint{ other.myPoint }
      , myFullIndex{ other.myFullIndex }
      {}

    /// Move constructor with type interoperability.
    template < typename TOtherIterableClass >
    ImageViewIterator( 
        ImageViewIterator<TOtherIterableClass> && other,
        typename std::enable_if< std::is_convertible<TOtherIterableClass*, IterableClass*>::value >::type* = 0 
    ) noexcept
      : myIterableClassPtr( std::move(other.myIterableClassPtr) )
      , myFullDomain{ std::move(other.myFullDomain) }
      , myViewDomain{ std::move(other.myViewDomain) }
      , myFullExtent{ std::move(other.myFullExtent) }
      , myViewExtent{ std::move(other.myViewExtent) }
      , myPoint{ std::move(other.myPoint) }
      , myFullIndex{ std::move(other.myFullIndex) }
      {}

    /// Destructor.
    ~ImageViewIterator() 
      {}

    /// Copy assignment with type interoperability.
    template < typename TOtherIterableClass >
    typename std::enable_if< std::is_convertible<TOtherIterableClass*, IterableClass*>::value, Self& >::type
    operator= ( 
        ImageViewIterator<TOtherIterableClass> const& other
    ) 
      {
        myIterableClassPtr = other.myIterableClassPtr;
        myFullDomain = other.myFullDomain;
        myViewDomain = other.myViewDomain;
        myFullExtent = other.myFullExtent;
        myViewExtent = other.myViewExtent;
        myPoint = other.myPoint;
        myFullIndex = other.myFullIndex;
        return *this;
      }

    /// Move assignment constructor with type interoperability.
    template < typename TOtherIterableClass >
    typename std::enable_if< std::is_convertible<TOtherIterableClass*, IterableClass*>::value, Self& >::type
    operator= ( 
        ImageViewIterator<TOtherIterableClass> && other
    ) 
      {
        myIterableClassPtr = std::move(other.myIterableClassPtr);
        myFullDomain = std::move(other.myFullDomain);
        myViewDomain = std::move(other.myViewDomain);
        myFullExtent = std::move(other.myFullExtent);
        myViewExtent = std::move(other.myViewExtent);
        myPoint = std::move(other.myPoint);
        myFullIndex = std::move(other.myFullIndex);
        return *this;
      }

  public:

    /** Iterator pointing to the first value.
     *
     * @param anIterableClassPtr  Pointer to the iterable class instance.
     * @param aFullDomain         Full domain of the image.
     * @param aViewDomain         Viewable domain that the iterator will span.
     * @param aPoint              Point to which the iterator will point.
     */
    ImageViewIterator( IterableClass* anIterableClassPtr, Domain const& aFullDomain, Domain const& aViewDomain ) 
        : ImageViewIterator{ anIterableClassPtr, aFullDomain, aViewDomain, aViewDomain.lowerBound() }
      {}
    
    /** Iterator pointing to the first value and spanning the whole domain.
     *
     * @param anIterableClassPtr  Pointer to the iterable class instance.
     * @param aFullDomain         Full domain of the image.
     * @param aPoint              Point to which the iterator will point.
     */
    ImageViewIterator( IterableClass* anIterableClassPtr, Domain const& aFullDomain ) 
        : ImageViewIterator{ anIterableClassPtr, aFullDomain, aFullDomain }
      {}

    /** Iterator pointing after the last value of the viewable domain.
     *
     * @param anIterableClassPtr  Pointer to the iterable class instance.
     * @param aFullDomain         Full domain of the image.
     * @param aViewDomain         Viewable domain that the iterator will span.
     * @param aPoint              Point to which the iterator will point.
     */
    ImageViewIterator( IterableClass* anIterableClassPtr, Domain const& aFullDomain, Domain const& aViewDomain, bool /* last */ ) 
      : ImageViewIterator{ anIterableClassPtr, aFullDomain, aViewDomain, aViewDomain.upperBound() }
      {
        increment();
      }

    /** Iterator pointing after the last value of the whole domain.
     *
     * @param anIterableClassPtr  Pointer to the iterable class instance.
     * @param aFullDomain         Full domain of the image.
     * @param aPoint              Point to which the iterator will point.
     */
    ImageViewIterator( IterableClass* anIterableClassPtr, Domain const& aFullDomain, bool /* last */ ) 
      : ImageViewIterator{ anIterableClassPtr, aFullDomain, aFullDomain, true }
      {
      }

    /**
     * @return the point behind this iterator.
     */
    inline
    Point const& getPoint() const 
      {
        return myPoint;
      }

    /**
     * @return the distance from this iterator to a given point.
     */
    inline
    std::ptrdiff_t distance_to( Point const& aPoint ) const 
      {
        BOOST_ASSERT_MSG(
            myViewDomain.isInside(aPoint),
            "The point is outside the viewable domain !"
        );
        // return static_cast<std::ptrdiff_t>( Self::Linearizer::getIndex( aPoint - myPoint, myViewExtent ) ); // <- bad idea if aPoint is before myPoint
        return 
            static_cast<std::ptrdiff_t>( Linearizer::getIndex(aPoint, myViewDomain.lowerBound(), myViewExtent) )
          - static_cast<std::ptrdiff_t>( Linearizer::getIndex(myPoint, myViewDomain.lowerBound(), myViewExtent) );

      }

  private:

    /** Friendship of interoperability
     * \see http://www.boost.org/doc/libs/1_58_0/libs/iterator/doc/iterator_facade.html
     */
    template <class> friend class ImageViewIterator;
    friend class boost::iterator_core_access;
  

    /// Increment of one step.
    void increment() 
      {
        ++myFullIndex;
        ++myPoint[0];
        for ( std::size_t i = 1; i < Domain::dimension && myPoint[i-1] > myViewDomain.upperBound()[i-1]; ++i )
          {
            myPoint[i-1] = myViewDomain.lowerBound()[i-1];
            ++myPoint[i];
            std::size_t cum = myFullExtent[i-1] - myViewExtent[i-1];
            for ( std::size_t j = 0; j < i-1; ++j )
              cum *= myFullExtent[j];

            myFullIndex += cum;
          }
      }

    /// Decrement of one step.
    void decrement() 
      {
        --myFullIndex;
        --myPoint[0];
        for ( std::size_t i = 1; i < Domain::dimension && myPoint[i-1] < myViewDomain.lowerBound()[i-1]; ++i )
          {
            myPoint[i-1] = myViewDomain.upperBound()[i-1];
            --myPoint[i];
            std::size_t cum = myFullExtent[i-1] - myViewExtent[i-1];
            for ( std::size_t j = 0; j < i-1; ++j )
              cum *= myFullExtent[j];

            myFullIndex -= cum;
          }
      }

    /// Equality.
    inline
    bool equal( Self const& other ) const 
      {
        return myFullIndex == other.myFullIndex;
      }

    /// Constant dereference.
    inline
    Reference dereference() const
      {
        return myIterableClassPtr->dereference( myPoint, myFullIndex );
      }

    /// Distance to other iterator.
    inline
    std::ptrdiff_t distance_to( Self const& other ) const 
      {
        return distance_to( other.myPoint );
      }

    /// Advance by n steps. Not very efficient implementation ...
    void advance( std::ptrdiff_t n ) 
      {
        const auto pos = Self::Linearizer::getIndex( myPoint, myViewDomain.lowerBound(), myViewExtent );
        myPoint = Self::Linearizer::getPoint( pos + n, myViewDomain.lowerBound(), myViewExtent );
        myFullIndex = Self::Linearizer::getIndex( myPoint, myFullDomain.lowerBound(), myFullExtent ); 
      }

  private:
    IterableClass* myIterableClassPtr; ///< Pointer to the iterable class.
    Domain myFullDomain;  ///< Full domain of the image.
    Domain myViewDomain;  ///< Iterable (viewable) domain of the image.
    Point myFullExtent;   ///< Extent of the full domain.
    Point myViewExtent;   ///< Extent of the viewable domain.
    Point myPoint;        ///< Current point where the iterator point to.
    typename Point::Coordinate myFullIndex; ///< Linearized index of the current point.
    
  };




} // namespace DGtal

/* GNU coding style */
/* vim: set ts=2 sw=2 expandtab cindent cinoptions=>4,n-2,{2,^-2,:2,=2,g0,h2,p5,t0,+2,(0,u0,w1,m1 : */
