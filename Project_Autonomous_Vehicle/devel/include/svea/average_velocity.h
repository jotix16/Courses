// Generated by gencpp from file svea/average_velocity.msg
// DO NOT EDIT!


#ifndef SVEA_MESSAGE_AVERAGE_VELOCITY_H
#define SVEA_MESSAGE_AVERAGE_VELOCITY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace svea
{
template <class ContainerAllocator>
struct average_velocity_
{
  typedef average_velocity_<ContainerAllocator> Type;

  average_velocity_()
    : average_velocity(0.0)  {
    }
  average_velocity_(const ContainerAllocator& _alloc)
    : average_velocity(0.0)  {
  (void)_alloc;
    }



   typedef double _average_velocity_type;
  _average_velocity_type average_velocity;





  typedef boost::shared_ptr< ::svea::average_velocity_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::svea::average_velocity_<ContainerAllocator> const> ConstPtr;

}; // struct average_velocity_

typedef ::svea::average_velocity_<std::allocator<void> > average_velocity;

typedef boost::shared_ptr< ::svea::average_velocity > average_velocityPtr;
typedef boost::shared_ptr< ::svea::average_velocity const> average_velocityConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::svea::average_velocity_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::svea::average_velocity_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace svea

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'svea': ['/home/aljanabim/ros/svea_starter/src/svea/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::svea::average_velocity_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::svea::average_velocity_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::svea::average_velocity_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::svea::average_velocity_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::svea::average_velocity_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::svea::average_velocity_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::svea::average_velocity_<ContainerAllocator> >
{
  static const char* value()
  {
    return "320db02e2aa0448db8c012457af99cee";
  }

  static const char* value(const ::svea::average_velocity_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x320db02e2aa0448dULL;
  static const uint64_t static_value2 = 0xb8c012457af99ceeULL;
};

template<class ContainerAllocator>
struct DataType< ::svea::average_velocity_<ContainerAllocator> >
{
  static const char* value()
  {
    return "svea/average_velocity";
  }

  static const char* value(const ::svea::average_velocity_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::svea::average_velocity_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64 average_velocity\n\
";
  }

  static const char* value(const ::svea::average_velocity_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::svea::average_velocity_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.average_velocity);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct average_velocity_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::svea::average_velocity_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::svea::average_velocity_<ContainerAllocator>& v)
  {
    s << indent << "average_velocity: ";
    Printer<double>::stream(s, indent + "  ", v.average_velocity);
  }
};

} // namespace message_operations
} // namespace ros

#endif // SVEA_MESSAGE_AVERAGE_VELOCITY_H
