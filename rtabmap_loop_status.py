import rclpy
from rclpy.node import Node
from rtabmap_msgs.msg import Info

class LoopStatus(Node):
    def __init__(self):
        super().__init__('rtabmap_loop_status')
        self.sub = self.create_subscription(Info, '/rtabmap/info', self.cb, 10)

    def cb(self, msg: Info):
        kv = {}
        # stats_keys/values are parallel arrays
        for k, v in zip(msg.stats_keys, msg.stats_values):
            kv[k] = v

        accepted = kv.get('Loop/Accepted_hypothesis_id/', None)
        highest_val = kv.get('Loop/Highest_hypothesis_value/', None)
        ratio = kv.get('Loop/Hypothesis_ratio/', None)
        dist = kv.get('Loop/Distance_since_last_loc/m', None)
        lc_id = msg.loop_closure_id

        print("loop_closure_id=", lc_id, "accepted=", accepted, "highest=", highest_val,
              "ratio=", ratio, "dist_since_last_loc=", dist)

rclpy.init()
node = LoopStatus()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
node.destroy_node()
rclpy.shutdown()
