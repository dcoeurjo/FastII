#pragma once

#include <cstddef>

namespace DGtal
{

/// Tag specifying a row-major storage order.
struct RowMajorStorage {};

/// Tag specifying a col-major storage order.
struct ColMajorStorage {};

/// Tools
namespace
{

  /** Helper that calculates the dimension index given the current linearization step.
   * 
   * The step I is always decrementing from N-1 to 0.
   *
   * @tparam TStorageOrder  Storage order (RowMajorStorage of ColMajorStorage).
   * @tparam N              Dimension of the space.
   * @tparam I              Current step.
   */
  template < typename TStorageOrder, std::size_t N, std::size_t I >
  struct linearizer_helper;

  template < std::size_t N, std::size_t I >
  struct linearizer_helper< RowMajorStorage, N, I >
    {
      enum { dim = I }; ///< Dimension index related to the step I.
    };

  template < std::size_t N, std::size_t I >
  struct linearizer_helper< ColMajorStorage, N, I >
    {
      enum { dim = N-1 - I }; ///< Dimension index related to the step I.
    };

  /** Templated static structure for linearization of the coordinates of a point.
   *
   * @tparam TSize  Type of the linearizared index.
   * @tparam TStorageOrder  Storage order (RowMajorStorage of ColMajorStorage).
   * @tparam N      Dimension of the space.
   * @tparam I      Current step.
   */
  template < typename TSize, typename TStorageOrder, std::size_t N, std::size_t I = N-1  >
  struct linearizer_impl
    {

      /**
       * Return the linearized index from the coordinates of a point.
       *
       * @tparam    TPoint    Type of the point.
       * @tparam    TExtent   Type of the domain's extent.
       * @param[in] aPoint    The point to linearize.
       * @param[in] aExtent   The extent of the domain.
       * @return    the linearized index of the point.
       */
      template < typename TPoint, typename TExtent >
      static inline
      TSize apply( TPoint const& aPoint, TExtent const& aExtent ) noexcept
        {
          return 
              aPoint[ linearizer_helper<TStorageOrder, N, I>::dim ] 
            + aExtent[ linearizer_helper<TStorageOrder, N, I>::dim ] * linearizer_impl< TSize, TStorageOrder, N, I-1 >::apply( aPoint, aExtent );
        }
    };

  /**
   * Specialization of the structure linearizer_impl for the last step.
   *
   * It is actually used as a terminate condition of the recursive process.
   */
  template < typename TSize, typename TStorageOrder, std::size_t N >
  struct linearizer_impl< TSize, TStorageOrder, N, 0 >
    {
      template < typename TPoint, typename TExtent >
      static inline
      TSize apply( TPoint const& aPoint, TExtent const& /* aExtent */ ) noexcept
        {
          return aPoint[ linearizer_helper<TStorageOrder, N, 0>::dim ];
        }
    };

  /** Templated static structure for de-linearization of a point index.
   *
   * @tparam TSize  Typdde of the linearizared index.
   * @tparam TStorageOrder  Storage order (RowMajorStorage of ColMajorStorage).
   * @tparam N      Dimension of the space.
   * @tparam I      Current step.
   */
  template < typename TStorageOrder, std::size_t N, std::size_t I = N-1  >
  struct delinearizer_impl
    {

      /**
       * Return the de-linearized point from the linearized index of the point.
       *
       * @tparam    TPoint    Type of the point.
       * @tparam    TExtent   Type of the domain's extent.
       * @tparam    TSize     Type of the linearized index.
       * @param[out] aPoint   The point after de-linearization.
       * @param[in] aExtent   The extent of the domain.
       * @param[in] aIndex    The linearized index of the point.
       */
      template < typename TPoint, typename TExtent, typename TSize >
      static inline
      void apply( TPoint& aPoint, TExtent const& aExtent, TSize aIndex ) noexcept
        {
          const auto dim_extent = aExtent[ linearizer_helper<TStorageOrder, N, I>::dim ];
          aPoint[ linearizer_helper<TStorageOrder, N, I>::dim ] = aIndex % dim_extent;
          delinearizer_impl< TStorageOrder, N, I-1 >::apply( aPoint, aExtent, aIndex / dim_extent );
        }
    };

  /**
   * Specialization of the structure delinearizer_impl for the last step.
   *
   * It is actually used as a terminate condition of the recursive process.
   */
  template < typename TStorageOrder, std::size_t N >
  struct delinearizer_impl< TStorageOrder, N, 0 >
    {
      template < typename TPoint, typename TExtent, typename TSize >
      static inline
      void apply( TPoint& aPoint, TExtent const& /* aExtent */, TSize aIndex ) noexcept
        {
          aPoint[ linearizer_helper<TStorageOrder, N, 0>::dim ] = aIndex;
        }
    };

} // anonymous namespace


/** Linearization and de-linearization interface for HyperRectDomain.
 *
 * @tparam  TDomain       Type of the HyperRectDomain.
 * @tparam  TStorageOrder Storage Order (RowMajorStorage of ColMajorStorage).
 */
template < typename TDomain, typename TStorageOrder = ColMajorStorage >
struct Linearizer
{
  using Point = typename TDomain::Point;
  using Extent = Point;
  using Size = typename TDomain::Size;

  /** Linearized index of a point, given the domain lower-bound and extent.
   *
   * @param[in] aPoint      The point to be linearized.
   * @param[in] aLowerBound The lower-bound of the domain.
   * @param[in] aExtent     The extent of the domain.
   * @return the linearized index of the point.
   */
  static inline
  Size getIndex( Point aPoint, Point const& aLowerBound, Extent const& aExtent ) noexcept
    {
      aPoint -= aLowerBound;
      return linearizer_impl<Size, TStorageOrder, TDomain::dimension>::apply(aPoint, aExtent);
    }

  /** Linearized index of a point, given the domain extent.
   *
   * The lower-bound of the domain is defined to the origin.
   *
   * @param[in] aPoint    The Point to be linearized.
   * @param[in] aExtent   The extent of the domain.
   * @return the linearized index of the point.
   */
  static inline
  Size getIndex( Point aPoint, Extent const& aExtent ) noexcept
    {
      return linearizer_impl<Size, TStorageOrder, TDomain::dimension>::apply(aPoint, aExtent);
    }

  /** Linearized index of a point, given a domain.
   *
   * @param[in] aPoint    The Point to be linearized.
   * @param[in] aDomain   The domain.
   * @return the linearized index of the point.
   */
  static inline
  Size getIndex( Point aPoint, TDomain const& aDomain ) noexcept
    {
      return linearizer_impl<Size, TStorageOrder, TDomain::dimension>::apply(aPoint - aDomain.lowerBound(), aDomain.upperBound()-aDomain.lowerBound()+Point::diagonal(1));
    }

  /** De-linearization of an index, given the domain lower-bound and extent.
   *
   * @param[in] aIndex  The linearized index.
   * @param[in] aLowerBound The lower-bound of the domain.
   * @param[in] aExtent     The domain extent.
   * @return  the point whose linearized index is aIndex.
   */
  static inline
  Point getPoint( Size aIndex, Point const& aLowerBound, Extent const& aExtent ) noexcept
    {
      Point point{};
      delinearizer_impl<TStorageOrder, TDomain::dimension>::apply(point, aExtent, aIndex);
      return point + aLowerBound;
    }

  /** De-linearization of an index, given the domain extent.
   *
   * The lower-bound of the domain is set to the origin.
   *
   * @param[in] aIndex  The linearized index.
   * @param[in] aExtent     The domain extent.
   * @return  the point whose linearized index is aIndex.
   */
  static inline
  Point getPoint( Size aIndex, Extent const& aExtent ) noexcept
    {
      Point point{};
      delinearizer_impl<TStorageOrder, TDomain::dimension>::apply(point, aExtent, aIndex);
      return point;
    }

  /** De-linearization of an index, given a domain.
   *
   * @param[in] aIndex    The linearized index.
   * @param[in] aDomain   The domain.
   * @return  the point whose linearized index is aIndex.
   */
  static inline
  Point getPoint( Size aIndex, TDomain const& aDomain ) noexcept
    {
      Point point{};
      linearizer_impl<Size, TStorageOrder, TDomain::dimension>::apply(point, aDomain.upperBound()-aDomain.lowerBound()+Point::diagonal(1), aIndex);
      return point + aDomain.lowerBound();
    }
};

} // namespace DGtal

/* GNU coding style */
/* vim: set ts=2 sw=2 expandtab cindent cinoptions=>4,n-2,{2,^-2,:2,=2,g0,h2,p5,t0,+2,(0,u0,w1,m1 : */
