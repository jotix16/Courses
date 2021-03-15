;; Auto-generated. Do not edit!


(when (boundp 'svea_arduino::lli_ctrl)
  (if (not (find-package "SVEA_ARDUINO"))
    (make-package "SVEA_ARDUINO"))
  (shadow 'lli_ctrl (find-package "SVEA_ARDUINO")))
(unless (find-package "SVEA_ARDUINO::LLI_CTRL")
  (make-package "SVEA_ARDUINO::LLI_CTRL"))

(in-package "ROS")
;;//! \htmlinclude lli_ctrl.msg.html


(defclass svea_arduino::lli_ctrl
  :super ros::object
  :slots (_steering _velocity _trans_diff _ctrl ))

(defmethod svea_arduino::lli_ctrl
  (:init
   (&key
    ((:steering __steering) 0)
    ((:velocity __velocity) 0)
    ((:trans_diff __trans_diff) 0)
    ((:ctrl __ctrl) 0)
    )
   (send-super :init)
   (setq _steering (round __steering))
   (setq _velocity (round __velocity))
   (setq _trans_diff (round __trans_diff))
   (setq _ctrl (round __ctrl))
   self)
  (:steering
   (&optional __steering)
   (if __steering (setq _steering __steering)) _steering)
  (:velocity
   (&optional __velocity)
   (if __velocity (setq _velocity __velocity)) _velocity)
  (:trans_diff
   (&optional __trans_diff)
   (if __trans_diff (setq _trans_diff __trans_diff)) _trans_diff)
  (:ctrl
   (&optional __ctrl)
   (if __ctrl (setq _ctrl __ctrl)) _ctrl)
  (:serialization-length
   ()
   (+
    ;; int8 _steering
    1
    ;; int8 _velocity
    1
    ;; int8 _trans_diff
    1
    ;; int8 _ctrl
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _steering
       (write-byte _steering s)
     ;; int8 _velocity
       (write-byte _velocity s)
     ;; int8 _trans_diff
       (write-byte _trans_diff s)
     ;; int8 _ctrl
       (write-byte _ctrl s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _steering
     (setq _steering (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _steering 127) (setq _steering (- _steering 256)))
   ;; int8 _velocity
     (setq _velocity (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _velocity 127) (setq _velocity (- _velocity 256)))
   ;; int8 _trans_diff
     (setq _trans_diff (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _trans_diff 127) (setq _trans_diff (- _trans_diff 256)))
   ;; int8 _ctrl
     (setq _ctrl (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _ctrl 127) (setq _ctrl (- _ctrl 256)))
   ;;
   self)
  )

(setf (get svea_arduino::lli_ctrl :md5sum-) "f4c1d25e08fe7c24fca84a1ec3ad2a96")
(setf (get svea_arduino::lli_ctrl :datatype-) "svea_arduino/lli_ctrl")
(setf (get svea_arduino::lli_ctrl :definition-)
      "int8 steering
int8 velocity
int8 trans_diff
int8 ctrl

")



(provide :svea_arduino/lli_ctrl "f4c1d25e08fe7c24fca84a1ec3ad2a96")


