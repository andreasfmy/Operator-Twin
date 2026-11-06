# OperatorState.msg .
#Defines the structure of the Operator Awareness model that will be published by the /awareness_state node in /operator_states at 20Hz
int32 agent_id #hunav_msgs/Agent id
int32 robot_id
float32 awareness # A, is the probability that the operator is aware of the robot given its exposure history.
bool visual_evidence # a(t). Binary indicator of whether the robot is present in the operator's frontal half-plane and there's no occlusion. It's boolean as this model treats visual evidence as binary. 
float32 head_yaw # rad 
float32 relative_angle # rad. Angle between the head yaw and the direction of the robot's centroid. It's positive in the anti-clockwise sense. 
float32 distance # m
bool occluded # Raycasts to detect whether a physical object carrying a collider lies between the operator's head and the robot. It's true when no points of the robot are visible from the operator's head; false if at least one is.