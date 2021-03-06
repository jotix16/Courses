// Generated by gencpp from file qualisys/Marker.msg
// DO NOT EDIT!


#ifndef QUALISYS_MESSAGE_MARKER_H
#define QUALISYS_MESSAGE_MARKER_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Point.h>

namespace qualisys
{
template <class ContainerAllocator>
struct Marker_
{
  typedef Marker_<ContainerAllocator> Type;

  Marker_()
    : name()
    , subject_name()
    , position()
    , occluded(false)  {
    }
  Marker_(const ContainerAllocator& _alloc)
    : name(_alloc)
    , subject_name(_alloc)
    , position(_alloc)
    , occluded(false)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _name_type;
  _name_type name;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _subject_name_type;
  _subject_name_type subject_name;

   typedef  ::geometry_msgs::Point_<ContainerAllocator>  _position_type;
  _position_type position;

   typedef uint8_t _occluded_type;
  _occluded_type occluded;





  typedef boost::shared_ptr< ::qualisys::Marker_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::qualisys::Marker_<ContainerAllocator> const> ConstPtr;

}; // struct Marker_

typedef ::qualisys::Marker_<std::allocator<void> > Marker;

typedef boost::shared_ptr< ::qualisys::Marker > MarkerPtr;
typedef boost::shared_ptr< ::qualisys::Marker const> MarkerConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::qualisys::Marker_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::qualisys::Marker_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace qualisys

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'qualisys': ['/home/aljanabim/ros/svea_starter/src/qualisys/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::qualisys::Marker_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::qualisys::Marker_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::qualisys::Marker_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::qualisys::Marker_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::qualisys::Marker_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::qualisys::Marker_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::qualisys::Marker_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a16c57ea269b234761b832931693cc90";
  }

  static const char* value(const ::qualisys::Marker_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa16c57ea269b2347ULL;
  static const uint64_t static_value2 = 0x61b832931693cc90ULL;
};

template<class ContainerAllocator>
struct DataType< ::qualisys::Marker_<ContainerAllocator> >
{
  static const char* value()
  {
    return "qualisys/Marker";
  }

  static const char* value(const ::qualisys::Marker_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::qualisys::Marker_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string name\n\
string subject_name\n\
geometry_msgs/Point position\n\
bool occluded\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
";
  }

  static const char* value(const ::qualisys::Marker_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::qualisys::Marker_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.name);
      stream.next(m.subject_name);
      stream.next(m.position);
      stream.next(m.occluded);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Marker_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::qualisys::Marker_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::qualisys::Marker_<ContainerAllocator>& v)
  {
    s << indent << "name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.name);
    s << indent << "subject_name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.subject_name);
    s << indent << "position: ";
    s << std::endl;
    Printer< ::geometry_msgs::Point_<ContainerAllocator> >::stream(s, indent + "  ", v.position);
    s << indent << "occluded: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.occluded);
  }
};

} // namespace message_operations
} // namespace ros

#endif // QUALISYS_MESSAGE_MARKER_H
