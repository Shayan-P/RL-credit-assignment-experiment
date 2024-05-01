# some util functions
def plot(logs, x_key, y_key, legend_key, **kwargs):
    nums = len(logs[legend_key].unique())
    palette = sns.color_palette("hls", nums)
    if 'palette' not in kwargs:
        kwargs['palette'] = palette
    ax = sns.lineplot(x=x_key, y=y_key, data=logs, hue=legend_key, **kwargs)
    return ax

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# set random seed
seed = 0
set_random_seed(seed=seed)





# plot(logs, x_key='step', y_key='reward', legend_key="Alg", estimator='mean', errorbar=None)




def play_video(video_dir: str, video_file: str = None) -> None:
    """
      Parameters:
      - video_dir (str): The directory path where video files are located. This is used if `video_file` is not provided.
      - video_file (str, optional): The path to a specific video file to play. If None, the function searches for
        'render_video.mp4' in `video_dir`.

      Returns:
        - None: This function does not return any value. It directly displays the video within the IPython notebook.
    """
    if video_file is None:
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob(f'**/render_video.mp4'))
        video_files.sort()
        video_file = video_files[-1]
    else:
        video_file = Path(video_file)
    compressed_file = video_file.parent.joinpath('comp.mp4')
    os.system(f"ffmpeg -i {video_file} -filter:v 'setpts=2.0*PTS' -vcodec libx264 {compressed_file.as_posix()}")
    mp4 = open(compressed_file.as_posix(),'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url))


# read tf log file
def read_tf_log(log_dir: str) -> Tuple[List[int], List[float], List[float]]:
    """
        Parameters:
      - log_dir (str): The directory path where TensorFlow log files are located. The function searches for files
        starting with 'events.' within this directory and its subdirectories.

      Returns:
      - Tuple[List[int], List[float], List[float]]: A tuple containing three lists:
          - steps (List[int]): A list of steps at which each episode's success rate was recorded.
          - returns (List[float]): A list of mean returns for each episode.
          - success_rate (List[float]): A list of success rates for each episode.
        Returns None if no log files are found or if there's an error in extracting scalar values.
    """
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f'**/events.*'))
    if len(log_files) < 1:
        return None
    log_file = log_files[0]
    event_acc = EventAccumulator(log_file.as_posix())
    event_acc.Reload()
    tags = event_acc.Tags()
    try:
        scalar_success = event_acc.Scalars('train/episode_success')
        success_rate = [x.value for x in scalar_success]
        steps = [x.step for x in scalar_success]
        scalar_return = event_acc.Scalars('train/episode_return/mean')
        returns = [x.value for x in scalar_return]
    except:
        return None
    return steps, returns, success_rate

