;; Auto-generated. Do not edit!


(when (boundp 'svea_arduino::lli_encoder)
  (if (not (find-package "SVEA_ARDUINO"))
    (make-package "SVEA_ARDUINO"))
  (shadow 'lli_encoder (find-package "SVEA_ARDUINO")))
(unless (find-package "SVEA_ARDUINO::LLI_ENCODER")
  (make-package "SVEA_ARDUINO::LLI_ENCODER"))

(in-package "ROS")
;;//! \htmlinclude lli_encoder.msg.html


(defclass svea_arduino::lli_encoder
  :super ros::object
  :slots (_right_ticks _left_ticks _time_delta ))

(defmethod svea_arduino::lli_encoder
  (:init
   (&key
    ((:right_ticks __right_ticks) 0)
    ((:left_ticks __left_ticks) 0)
    ((:time_delta __time_delta) 0)
    )
   (send-super :init)
   (setq _right_ticks (round __right_ticks))
   (setq _left_ticks (round __left_ticks))
   (setq _time_delta (round __time_delta))
   self)
  (:right_ticks
   (&optional __right_ticks)
   (if __right_ticks (setq _right_ticks __right_ticks)) _right_ticks)
  (:left_ticks
   (&optional __left_ticks)
   (if __left_ticks (setq _left_ticks __left_ticks)) _left_ticks)
  (:time_delta
   (&optional __time_delta)
   (if __time_delta (setq _time_delta __time_delta)) _time_delta)
  (:serialization-length
   ()
   (+
    ;; uint8 _right_ticks
    1
    ;; uint8 _left_ticks
    1
    ;; uint16 _time_delta
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _right_ticks
       (write-byte _right_ticks s)
     ;; uint8 _left_ticks
       (write-byte _left_ticks s)
     ;; uint16 _time_delta
       (write-word _time_delta s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _right_ticks
     (setq _right_ticks (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint8 _left_ticks
     (setq _left_ticks (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;; uint16 _time_delta
     (setq _time_delta (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get svea_arduino::lli_encoder :md5sum-) "41aecfbb5a17be4edf75b1acdf69d991")
(setf (get svea_arduino::lli_encoder :datatype-) "svea_arduino/lli_encoder")
(setf (get svea_arduino::lli_encoder :definition-)
      "uint8 right_ticks
uint8 left_ticks
uint16 time_delta

")



(provide :svea_arduino/lli_encoder "41aecfbb5a17be4edf75b1acdf69d991")


