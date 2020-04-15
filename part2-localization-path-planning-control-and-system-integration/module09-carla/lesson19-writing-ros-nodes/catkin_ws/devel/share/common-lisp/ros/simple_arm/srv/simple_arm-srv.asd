
(cl:in-package :asdf)

(defsystem "simple_arm-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "GoToPosition" :depends-on ("_package_GoToPosition"))
    (:file "_package_GoToPosition" :depends-on ("_package"))
  ))