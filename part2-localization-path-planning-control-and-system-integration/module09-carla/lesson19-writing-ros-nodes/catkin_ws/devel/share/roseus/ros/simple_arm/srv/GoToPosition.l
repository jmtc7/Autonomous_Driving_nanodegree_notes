;; Auto-generated. Do not edit!


(when (boundp 'simple_arm::GoToPosition)
  (if (not (find-package "SIMPLE_ARM"))
    (make-package "SIMPLE_ARM"))
  (shadow 'GoToPosition (find-package "SIMPLE_ARM")))
(unless (find-package "SIMPLE_ARM::GOTOPOSITION")
  (make-package "SIMPLE_ARM::GOTOPOSITION"))
(unless (find-package "SIMPLE_ARM::GOTOPOSITIONREQUEST")
  (make-package "SIMPLE_ARM::GOTOPOSITIONREQUEST"))
(unless (find-package "SIMPLE_ARM::GOTOPOSITIONRESPONSE")
  (make-package "SIMPLE_ARM::GOTOPOSITIONRESPONSE"))

(in-package "ROS")





(defclass simple_arm::GoToPositionRequest
  :super ros::object
  :slots (_joint_1 _joint_2 ))

(defmethod simple_arm::GoToPositionRequest
  (:init
   (&key
    ((:joint_1 __joint_1) 0.0)
    ((:joint_2 __joint_2) 0.0)
    )
   (send-super :init)
   (setq _joint_1 (float __joint_1))
   (setq _joint_2 (float __joint_2))
   self)
  (:joint_1
   (&optional __joint_1)
   (if __joint_1 (setq _joint_1 __joint_1)) _joint_1)
  (:joint_2
   (&optional __joint_2)
   (if __joint_2 (setq _joint_2 __joint_2)) _joint_2)
  (:serialization-length
   ()
   (+
    ;; float64 _joint_1
    8
    ;; float64 _joint_2
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _joint_1
       (sys::poke _joint_1 (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _joint_2
       (sys::poke _joint_2 (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _joint_1
     (setq _joint_1 (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _joint_2
     (setq _joint_2 (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass simple_arm::GoToPositionResponse
  :super ros::object
  :slots (_time_elapsed ))

(defmethod simple_arm::GoToPositionResponse
  (:init
   (&key
    ((:time_elapsed __time_elapsed) (instance ros::time :init))
    )
   (send-super :init)
   (setq _time_elapsed __time_elapsed)
   self)
  (:time_elapsed
   (&optional __time_elapsed)
   (if __time_elapsed (setq _time_elapsed __time_elapsed)) _time_elapsed)
  (:serialization-length
   ()
   (+
    ;; duration _time_elapsed
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; duration _time_elapsed
       (write-long (send _time_elapsed :sec) s) (write-long (send _time_elapsed :nsec) s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; duration _time_elapsed
     (send _time_elapsed :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _time_elapsed :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(defclass simple_arm::GoToPosition
  :super ros::object
  :slots ())

(setf (get simple_arm::GoToPosition :md5sum-) "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(setf (get simple_arm::GoToPosition :datatype-) "simple_arm/GoToPosition")
(setf (get simple_arm::GoToPosition :request) simple_arm::GoToPositionRequest)
(setf (get simple_arm::GoToPosition :response) simple_arm::GoToPositionResponse)

(defmethod simple_arm::GoToPositionRequest
  (:response () (instance simple_arm::GoToPositionResponse :init)))

(setf (get simple_arm::GoToPositionRequest :md5sum-) "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(setf (get simple_arm::GoToPositionRequest :datatype-) "simple_arm/GoToPositionRequest")
(setf (get simple_arm::GoToPositionRequest :definition-)
      "float64 joint_1
float64 joint_2
---
duration time_elapsed

")

(setf (get simple_arm::GoToPositionResponse :md5sum-) "fc4e1ffd0bd5c9cc8c021e351562f1a8")
(setf (get simple_arm::GoToPositionResponse :datatype-) "simple_arm/GoToPositionResponse")
(setf (get simple_arm::GoToPositionResponse :definition-)
      "float64 joint_1
float64 joint_2
---
duration time_elapsed

")



(provide :simple_arm/GoToPosition "fc4e1ffd0bd5c9cc8c021e351562f1a8")


