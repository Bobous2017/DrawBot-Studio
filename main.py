import capture_picture
import rigger_figure
import vision_agent_module


def run_robot_agent():
    print("--------- start Robot Agent ---------")

    raw_image = capture_picture.take_photo()
    if not raw_image:
        print("No photo captured")
        return

    print("---- cutting background ----")
    output_image = rigger_figure.remove_background(raw_image)

    print("---- extracting joints ----")
    joints = vision_agent_module.get_robot_joints(output_image)
    print(joints)


if __name__ == "__main__":
    run_robot_agent()