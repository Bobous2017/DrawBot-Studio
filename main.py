import capture_picture
import rigger_figure
import vision_agent_module
import dance_module


def run_robot_agent():
    print("--------- start Robot Agent ---------")

    raw_image = capture_picture.take_photo()
    if not raw_image:
        print("No photo captured")
        return

    print("---- cutting background ----")
    output_image = rigger_figure.remove_background(raw_image)

    print("---- extracting joints ----")
    joints = vision_agent_module.get_robot_joints(raw_image)
    print(joints)

    print("---- dancing ----")
    dance_module.start_dancing(joints, figure_image_path=output_image, show_skeleton=True)


if __name__ == "__main__":
    run_robot_agent()