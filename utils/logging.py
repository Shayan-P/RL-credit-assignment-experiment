# write logs then visualize?

from gym.wrappers.record_video import RecordVideo



def show_video():
    """
    Displays the recorded video of the gym environment.
    """
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        # mp4 = mp4list[0]
        mp4 = max(mp4list, key=os.path.getctime)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env(env):
    """
    Wraps the given gym environment to record videos.

    Parameters:
        env (gym.Env): The environment to wrap.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
    return env

