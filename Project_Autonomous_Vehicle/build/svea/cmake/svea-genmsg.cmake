# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "svea: 1 messages, 0 services")

set(MSG_I_FLAGS "-Isvea:/home/aljanabim/ros/svea_starter/src/svea/msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(svea_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_custom_target(_svea_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "svea" "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(svea
  "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/svea
)

### Generating Services

### Generating Module File
_generate_module_cpp(svea
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/svea
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(svea_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(svea_generate_messages svea_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_dependencies(svea_generate_messages_cpp _svea_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(svea_gencpp)
add_dependencies(svea_gencpp svea_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS svea_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(svea
  "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/svea
)

### Generating Services

### Generating Module File
_generate_module_eus(svea
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/svea
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(svea_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(svea_generate_messages svea_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_dependencies(svea_generate_messages_eus _svea_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(svea_geneus)
add_dependencies(svea_geneus svea_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS svea_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(svea
  "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/svea
)

### Generating Services

### Generating Module File
_generate_module_lisp(svea
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/svea
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(svea_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(svea_generate_messages svea_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_dependencies(svea_generate_messages_lisp _svea_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(svea_genlisp)
add_dependencies(svea_genlisp svea_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS svea_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(svea
  "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/svea
)

### Generating Services

### Generating Module File
_generate_module_nodejs(svea
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/svea
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(svea_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(svea_generate_messages svea_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_dependencies(svea_generate_messages_nodejs _svea_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(svea_gennodejs)
add_dependencies(svea_gennodejs svea_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS svea_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(svea
  "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/svea
)

### Generating Services

### Generating Module File
_generate_module_py(svea
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/svea
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(svea_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(svea_generate_messages svea_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/aljanabim/ros/svea_starter/src/svea/msg/average_velocity.msg" NAME_WE)
add_dependencies(svea_generate_messages_py _svea_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(svea_genpy)
add_dependencies(svea_genpy svea_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS svea_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/svea)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/svea
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(svea_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/svea)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/svea
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(svea_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/svea)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/svea
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(svea_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/svea)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/svea
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(svea_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/svea)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/svea\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/svea
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(svea_generate_messages_py std_msgs_generate_messages_py)
endif()
