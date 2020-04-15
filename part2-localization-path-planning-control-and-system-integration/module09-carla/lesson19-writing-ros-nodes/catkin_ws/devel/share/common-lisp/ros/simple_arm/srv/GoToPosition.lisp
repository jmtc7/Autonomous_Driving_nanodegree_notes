; Auto-generated. Do not edit!


(cl:in-package simple_arm-srv)


;//! \htmlinclude GoToPosition-request.msg.html

(cl:defclass <GoToPosition-request> (roslisp-msg-protocol:ros-message)
  ((joint_1
    :reader joint_1
    :initarg :joint_1
    :type cl:float
    :initform 0.0)
   (joint_2
    :reader joint_2
    :initarg :joint_2
    :type cl:float
    :initform 0.0))
)

(cl:defclass GoToPosition-request (<GoToPosition-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GoToPosition-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GoToPosition-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name simple_arm-srv:<GoToPosition-request> is deprecated: use simple_arm-srv:GoToPosition-request instead.")))

(cl:ensure-generic-function 'joint_1-val :lambda-list '(m))
(cl:defmethod joint_1-val ((m <GoToPosition-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader simple_arm-srv:joint_1-val is deprecated.  Use simple_arm-srv:joint_1 instead.")
  (joint_1 m))

(cl:ensure-generic-function 'joint_2-val :lambda-list '(m))
(cl:defmethod joint_2-val ((m <GoToPosition-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader simple_arm-srv:joint_2-val is deprecated.  Use simple_arm-srv:joint_2 instead.")
  (joint_2 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GoToPosition-request>) ostream)
  "Serializes a message object of type '<GoToPosition-request>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'joint_1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'joint_2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GoToPosition-request>) istream)
  "Deserializes a message object of type '<GoToPosition-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'joint_1) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'joint_2) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GoToPosition-request>)))
  "Returns string type for a service object of type '<GoToPosition-request>"
  "simple_arm/GoToPositionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GoToPosition-request)))
  "Returns string type for a service object of type 'GoToPosition-request"
  "simple_arm/GoToPositionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GoToPosition-request>)))
  "Returns md5sum for a message object of type '<GoToPosition-request>"
  "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GoToPosition-request)))
  "Returns md5sum for a message object of type 'GoToPosition-request"
  "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GoToPosition-request>)))
  "Returns full string definition for message of type '<GoToPosition-request>"
  (cl:format cl:nil "float64 joint_1~%float64 joint_2~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GoToPosition-request)))
  "Returns full string definition for message of type 'GoToPosition-request"
  (cl:format cl:nil "float64 joint_1~%float64 joint_2~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GoToPosition-request>))
  (cl:+ 0
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GoToPosition-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GoToPosition-request
    (cl:cons ':joint_1 (joint_1 msg))
    (cl:cons ':joint_2 (joint_2 msg))
))
;//! \htmlinclude GoToPosition-response.msg.html

(cl:defclass <GoToPosition-response> (roslisp-msg-protocol:ros-message)
  ((time_elapsed
    :reader time_elapsed
    :initarg :time_elapsed
    :type cl:real
    :initform 0))
)

(cl:defclass GoToPosition-response (<GoToPosition-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GoToPosition-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GoToPosition-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name simple_arm-srv:<GoToPosition-response> is deprecated: use simple_arm-srv:GoToPosition-response instead.")))

(cl:ensure-generic-function 'time_elapsed-val :lambda-list '(m))
(cl:defmethod time_elapsed-val ((m <GoToPosition-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader simple_arm-srv:time_elapsed-val is deprecated.  Use simple_arm-srv:time_elapsed instead.")
  (time_elapsed m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GoToPosition-response>) ostream)
  "Serializes a message object of type '<GoToPosition-response>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'time_elapsed)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'time_elapsed) (cl:floor (cl:slot-value msg 'time_elapsed)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GoToPosition-response>) istream)
  "Deserializes a message object of type '<GoToPosition-response>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'time_elapsed) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GoToPosition-response>)))
  "Returns string type for a service object of type '<GoToPosition-response>"
  "simple_arm/GoToPositionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GoToPosition-response)))
  "Returns string type for a service object of type 'GoToPosition-response"
  "simple_arm/GoToPositionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GoToPosition-response>)))
  "Returns md5sum for a message object of type '<GoToPosition-response>"
  "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GoToPosition-response)))
  "Returns md5sum for a message object of type 'GoToPosition-response"
  "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GoToPosition-response>)))
  "Returns full string definition for message of type '<GoToPosition-response>"
  (cl:format cl:nil "duration time_elapsed~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GoToPosition-response)))
  "Returns full string definition for message of type 'GoToPosition-response"
  (cl:format cl:nil "duration time_elapsed~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GoToPosition-response>))
  (cl:+ 0
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GoToPosition-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GoToPosition-response
    (cl:cons ':time_elapsed (time_elapsed msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GoToPosition)))
  'GoToPosition-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GoToPosition)))
  'GoToPosition-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GoToPosition)))
  "Returns string type for a service object of type '<GoToPosition>"
  "simple_arm/GoToPosition")