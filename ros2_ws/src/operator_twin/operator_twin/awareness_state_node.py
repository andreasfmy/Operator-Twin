import math
import rclpy
from rclpy.node import Node
from hunav_msgs.msg import Agents, Agent
from operator_twin_msgs.msg import OperatorState, OperatorStates


class AwarenessStateNode(Node):
    """Publica el estado de atencion de cada operario respecto del robot.

    Suscribe:  /human_states  (hunav_msgs/Agents)
               /robot_states  (hunav_msgs/Agent)
    Publica:   /operator_states (operator_twin_msgs/OperatorStates)
    """

    def __init__(self):
        super().__init__("awareness_state")

        # --- parametros barribles desde el DOE -------------------------
        # 1/lambda_acq = tiempo medio de deteccion  [s]
        # 1/lambda_dec = tiempo medio de retencion  [s]
        self.declare_parameter("lambda_acq", 1.0)
        self.declare_parameter("lambda_dec", 0.1)
        self.declare_parameter("fov_half_angle", 90.0)   # grados
        self.declare_parameter("dt", 0.05)               # s, 20 Hz

        # --- estado persistente ----------------------------------------
        # Lo que distingue este nodo de los de HuNavSim: A recuerda.
        self._awareness = {}      # agent_id -> A
        self._robot = None        # ultimo Agent del robot

        # --- interfaces -------------------------------------------------
        self.create_subscription(Agent, "/robot_states", self.robot_callback, 10)
        self.create_subscription(Agents, "/human_states", self.humans_callback, 10)
        self._pub = self.create_publisher(OperatorStates, "/operator_states", 10)

    # ------------------------------------------------------------------
    def robot_callback(self, msg):
        self._robot = msg
    # ------------------------------------------------------------------
    def humans_callback(self, msg):
        if self._robot is None:
            return          # aun no ha llegado ninguna pose del robot

        dt = self.get_parameter("dt").value
        rx = self._robot.position.position.x
        ry = self._robot.position.position.y

        out = OperatorStates()
        out.header = msg.header        
        seen = set()

        for agent in msg.agents:
            ax = agent.position.position.x
            ay = agent.position.position.y

            dx, dy = rx - ax, ry - ay
            distance = math.hypot(dx, dy)

            # angulo del robot visto desde la cabeza, con signo,
            # normalizado a (-pi, pi]
            bearing = math.atan2(dy, dx)
            head_yaw = self.compute_head_yaw(agent)
            rel = (bearing - head_yaw + math.pi) % (2 * math.pi) - math.pi

            occluded = self.compute_occlusion(agent, self._robot)
            V = self.compute_visual_evidence(rel, occluded)
            A = self.integrate(agent.id, V, dt)

            st = OperatorState()
            st.agent_id = agent.id
            st.robot_id = self._robot.id
            st.awareness = A
            st.visual_evidence = V
            st.head_yaw = head_yaw
            st.relative_angle = rel
            st.distance = distance
            st.occluded = occluded
            out.operators.append(st)
            seen.add(agent.id)

        # poda: el estado refleja solo los agentes del ultimo ciclo
        for gone in set(self._awareness) - seen:
            del self._awareness[gone]

        self._pub.publish(out)
    # ------------------------------------------------------------------
    def compute_head_yaw(self, agent):
        return agent.yaw

    # ------------------------------------------------------------------
    def compute_occlusion(self, agent, robot):
        return False

    # ------------------------------------------------------------------
    def compute_visual_evidence(self, relative_angle, occluded):
        half = math.radians(self.get_parameter("fov_half_angle").value)
        return (abs(relative_angle) < half) and (not occluded)

    # ------------------------------------------------------------------
    def integrate(self, agent_id, visual_evidence, dt):
        lam_acq = self.get_parameter("lambda_acq").value
        lam_dec = self.get_parameter("lambda_dec").value

        A = self._awareness.get(agent_id, 0.0)   # agente nuevo entra en 0
        V = 1.0 if visual_evidence else 0.0

        dA = (lam_acq * V * (1.0 - A) - lam_dec * A) * dt
        A = min(1.0, max(0.0, A + dA))           # acota a [0, 1]

        self._awareness[agent_id] = A
        return A


def main():
    rclpy.init()
    node = AwarenessStateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()