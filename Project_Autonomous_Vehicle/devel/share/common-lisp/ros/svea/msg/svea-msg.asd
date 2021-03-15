
(cl:in-package :asdf)

(defsystem "svea-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "average_velocity" :depends-on ("_package_average_velocity"))
    (:file "_package_average_velocity" :depends-on ("_package"))
  ))