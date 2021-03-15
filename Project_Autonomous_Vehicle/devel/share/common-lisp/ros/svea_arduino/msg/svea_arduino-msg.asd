
(cl:in-package :asdf)

(defsystem "svea_arduino-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "lli_ctrl" :depends-on ("_package_lli_ctrl"))
    (:file "_package_lli_ctrl" :depends-on ("_package"))
    (:file "lli_encoder" :depends-on ("_package_lli_encoder"))
    (:file "_package_lli_encoder" :depends-on ("_package"))
  ))