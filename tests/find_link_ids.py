import pybullet as p
import pybullet_data

# Connect to PyBullet and load the URDF
p.connect(p.GUI)  # Use p.DIRECT if you don't need visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # If needed for loading models

# Load your quadruped URDF
robot_id = p.loadURDF("dog_walking/resources/dog.urdf", [0, 0, 0.5])

# Get the number of joints (and links)
num_joints = p.getNumJoints(robot_id)

# Print all joint and link information to find the foot link IDs
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    link_name = joint_info[12].decode('utf-8')  # Link name (encoded as bytes, so decode it)
    link_index = joint_info[0]  # Link index
    print(f"Link ID: {link_index}, Link Name: {link_name}")

# Disconnect from simulation
p.disconnect()