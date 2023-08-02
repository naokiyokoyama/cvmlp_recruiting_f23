import os

import habitat_sim
import habitat_sim.utils.viz_utils as vut
import magnum as mn
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(dir_path, "output/")
urdf_files = {
    "spot_arm": os.path.join(data_path, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf"),
}


def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [-0.15, -0.1, 1.0]
    # agent_state.position = [-0.15, -1.6, 1.0]
    agent_state.rotation = np.quaternion(-0.83147, 0, 0.55557, 0)
    agent = sim.initialize_agent(0, agent_state)
    return agent.scene_node.transformation_matrix()


def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [540, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())

    return observations


def place_robot_from_agent(
    sim,
    robot_id,
    angle_correction=-1.56,
    local_base_pos=None,
):
    if local_base_pos is None:
        local_base_pos = np.array([0.0, -0.1, -2.0])
    # place the robot root state relative to the agent
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    base_transform = mn.Matrix4.rotation(
        mn.Rad(angle_correction), mn.Vector3(1.0, 0, 0)
    )
    base_transform.translation = agent_transform.transform_point(local_base_pos)
    robot_id.transformation = base_transform


# This is wrapped such that it can be added to a unit test
def main(make_video=True, show_video=True):
    # [initialize]
    # create the simulator
    cfg = make_configuration()
    with habitat_sim.Simulator(cfg) as sim:
        place_agent(sim)
        observations = []

        # load a URDF file
        robot_file = urdf_files["spot_arm"]
        ao_mgr = sim.get_articulated_object_manager()
        robot_id = ao_mgr.add_articulated_object_from_urdf(robot_file, fixed_base=False)

        assert robot_id is not None, (
            "URDF failed to load from " + robot_file + "! Aborting."
        )

        # place the robot root state relative to the agent
        place_robot_from_agent(sim, robot_id)

        # set a better initial joint state for the spot_arm
        if robot_file == urdf_files["spot_arm"]:
            pose = robot_id.joint_positions
            calfDofs = [2, 5, 8, 11]
            for dof in calfDofs:
                pose[dof] = -1.0
                pose[dof - 1] = 0.45
                # also set a thigh
            robot_id.joint_positions = pose

        # simulate
        observations += simulate(sim, dt=1.5, get_frames=make_video)

        if make_video:
            vut.make_video(
                observations,
                "rgba_camera_1stperson",
                "color",
                output_path + "URDF_basics",
                open_vid=show_video,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    make_video = args.make_video

    if make_video and not os.path.exists(output_path):
        os.mkdir(output_path)

    main(make_video, show_video)