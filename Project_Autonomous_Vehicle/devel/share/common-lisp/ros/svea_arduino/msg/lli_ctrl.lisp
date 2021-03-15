; Auto-generated. Do not edit!


(cl:in-package svea_arduino-msg)


;//! \htmlinclude lli_ctrl.msg.html

(cl:defclass <lli_ctrl> (roslisp-msg-protocol:ros-message)
  ((steering
    :reader steering
    :initarg :steering
    :type cl:fixnum
    :initform 0)
   (velocity
    :reader velocity
    :initarg :velocity
    :type cl:fixnum
    :initform 0)
   (trans_diff
    :reader trans_diff
    :initarg :trans_diff
    :type cl:fixnum
    :initform 0)
   (ctrl
    :reader ctrl
    :initarg :ctrl
    :type cl:fixnum
    :initform 0))
)

(cl:defclass lli_ctrl (<lli_ctrl>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <lli_ctrl>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'lli_ctrl)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name svea_arduino-msg:<lli_ctrl> is deprecated: use svea_arduino-msg:lli_ctrl instead.")))

(cl:ensure-generic-function 'steering-val :lambda-list '(m))
(cl:defmethod steering-val ((m <lli_ctrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:steering-val is deprecated.  Use svea_arduino-msg:steering instead.")
  (steering m))

(cl:ensure-generic-function 'velocity-val :lambda-list '(m))
(cl:defmethod velocity-val ((m <lli_ctrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:velocity-val is deprecated.  Use svea_arduino-msg:velocity instead.")
  (velocity m))

(cl:ensure-generic-function 'trans_diff-val :lambda-list '(m))
(cl:defmethod trans_diff-val ((m <lli_ctrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:trans_diff-val is deprecated.  Use svea_arduino-msg:trans_diff instead.")
  (trans_diff m))

(cl:ensure-generic-function 'ctrl-val :lambda-list '(m))
(cl:defmethod ctrl-val ((m <lli_ctrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader svea_arduino-msg:ctrl-val is deprecated.  Use svea_arduino-msg:ctrl instead.")
  (ctrl m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <lli_ctrl>) ostream)
  "Serializes a message object of type '<lli_ctrl>"
  (cl:let* ((signed (cl:slot-value msg 'steering)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'velocity)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'trans_diff)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'ctrl)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <lli_ctrl>) istream)
  "Deserializes a message object of type '<lli_ctrl>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'steering) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'velocity) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'trans_diff) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'ctrl) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<lli_ctrl>)))
  "Returns string type for a message object of type '<lli_ctrl>"
  "svea_arduino/lli_ctrl")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'lli_ctrl)))
  "Returns string type for a message object of type 'lli_ctrl"
  "svea_arduino/lli_ctrl")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<lli_ctrl>)))
  "Returns md5sum for a message object of type '<lli_ctrl>"
  "f4c1d25e08fe7c24fca84a1ec3ad2a96")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'lli_ctrl)))
  "Returns md5sum for a message object of type 'lli_ctrl"
  "f4c1d25e08fe7c24fca84a1ec3ad2a96")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<lli_ctrl>)))
  "Returns full string definition for message of type '<lli_ctrl>"
  (cl:format cl:nil "int8 steering~%int8 velocity~%int8 trans_diff~%int8 ctrl~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'lli_ctrl)))
  "Returns full string definition for message of type 'lli_ctrl"
  (cl:format cl:nil "int8 steering~%int8 velocity~%int8 trans_diff~%int8 ctrl~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <lli_ctrl>))
  (cl:+ 0
     1
     1
     1
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <lli_ctrl>))
  "Converts a ROS message object to a list"
  (cl:list 'lli_ctrl
    (cl:cons ':steering (steering msg))
    (cl:cons ':velocity (velocity msg))
    (cl:cons ':trans_diff (trans_diff msg))
    (cl:cons ':ctrl (ctrl msg))
))
