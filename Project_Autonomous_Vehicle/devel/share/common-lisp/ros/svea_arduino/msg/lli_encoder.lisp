; Auto-generated. Do not edit!


(cl:in-package svea_arduino-msg)


;//! \htmlinclude lli_encoder.msg.html

(cl:defclass <lli_encoder> (roslisp-msg-protocol:ros-message)
  ((right_ticks
    :reader right_ticks
    :initarg :right_ticks
    :type cl:fixnum
    :initform 0)
   (left_ticks
    :reader left_ticks
    :initarg :left_ticks
    :type cl:fixnum
    :initform 0)
   (time_delta
    :reader time_delta
    :initarg :time_delta
    :type cl:fixnum
    :initform 0))
)

(cl:defclass lli_encoder (<lli_encoder>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <lli_encoder>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'lli_encoder)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name svea_arduino-msg:<lli_encoder> is deprecated: use svea_arduino-msg:lli_encoder instead.")))

(cl:ensure-generic-function 'right_ticks-val :lambda-list '(m))
(cl:defmethod right_ticks-val ((m <lli_encoder>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:right_ticks-val is deprecated.  Use svea_arduino-msg:right_ticks instead.")
  (right_ticks m))

(cl:ensure-generic-function 'left_ticks-val :lambda-list '(m))
(cl:defmethod left_ticks-val ((m <lli_encoder>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:left_ticks-val is deprecated.  Use svea_arduino-msg:left_ticks instead.")
  (left_ticks m))

(cl:ensure-generic-function 'time_delta-val :lambda-list '(m))
(cl:defmethod time_delta-val ((m <lli_encoder>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:time_delta-val is deprecated.  Use svea_arduino-msg:time_delta instead.")
  (time_delta m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <lli_encoder>) ostream)
  "Serializes a message object of type '<lli_encoder>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'right_ticks)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'left_ticks)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'time_delta)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'time_delta)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <lli_encoder>) istream)
  "Deserializes a message object of type '<lli_encoder>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'right_ticks)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'left_ticks)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'time_delta)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'time_delta)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<lli_encoder>)))
  "Returns string type for a message object of type '<lli_encoder>"
  "svea_arduino/lli_encoder")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'lli_encoder)))
  "Returns string type for a message object of type 'lli_encoder"
  "svea_arduino/lli_encoder")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<lli_encoder>)))
  "Returns md5sum for a message object of type '<lli_encoder>"
  "41aecfbb5a17be4edf75b1acdf69d991")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'lli_encoder)))
  "Returns md5sum for a message object of type 'lli_encoder"
  "41aecfbb5a17be4edf75b1acdf69d991")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<lli_encoder>)))
  "Returns full string definition for message of type '<lli_encoder>"
  (cl:format cl:nil "uint8 right_ticks~%uint8 left_ticks~%uint16 time_delta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'lli_encoder)))
  "Returns full string definition for message of type 'lli_encoder"
  (cl:format cl:nil "uint8 right_ticks~%uint8 left_ticks~%uint16 time_delta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <lli_encoder>))
  (cl:+ 0
     1
     1
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <lli_encoder>))
  "Converts a ROS message object to a list"
  (cl:list 'lli_encoder
    (cl:cons ':right_ticks (right_ticks msg))
    (cl:cons ':left_ticks (left_ticks msg))
    (cl:cons ':time_delta (time_delta msg))
))
