#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pyttsx3 as tts

from rins_robot.srv import Speech

class speech_servicer(Node):

    def __init__(self):
        super().__init__("speech_servicer")

        self.greetServer = self.create_service(Speech,"/greet_service",self.greet)
        self.colorServer = self.create_service(Speech,"/sayColor_service",self.sayColor)

        self.engine = tts.init()

        self.get_logger().info(f"Speech service node initialized!")

    def greet(self,req,res):
        phrase = req.data
        self.engine.say(phrase)
        self.engine.runAndWait()
        res.success = True
        return res
    
    def sayColor(self,req,res):
        color = req.data
        self.engine.say(color)
        self.engine.runAndWait()
        res.success = True
        return res


def main(args=None):
    rclpy.init(args=args)
    node = speech_servicer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


