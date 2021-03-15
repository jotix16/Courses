;; Auto-generated. Do not edit!


(when (boundp 'svea::average_velocity)
  (if (not (find-package "SVEA"))
    (make-package "SVEA"))
  (shadow 'average_velocity (find-package "SVEA")))
(unless (find-package "SVEA::AVERAGE_VELOCITY")
  (make-package "SVEA::AVERAGE_VELOCITY"))

(in-package "ROS")
;;//! \htmlinclude average_velocity.msg.html


(defclass svea::average_velocity
  :super ros::object
  :slots (_average_velocity ))

(defmethod svea::average_velocity
  (:init
   (&key
    ((:average_velocity __average_velocity) 0.0)
    )
   (send-super :init)
   (setq _average_velocity (float __average_velocity))
   self)
  (:average_velocity
   (&optional __average_velocity)
   (if __average_velocity (setq _average_velocity __average_velocity)) _average_velocity)
  (:serialization-length
   ()
   (+
    ;; float64 _average_velocity
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _average_velocity
       (sys::poke _average_velocity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _average_velocity
     (setq _average_velocity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get svea::average_velocity :md5sum-) "320db02e2aa0448db8c012457af99cee")
(setf (get svea::average_velocity :datatype-) "svea/average_velocity")
(setf (get svea::average_velocity :definition-)
      "float64 average_velocity

")



(provide :svea/average_velocity "320db02e2aa0448db8c012457af99cee")


