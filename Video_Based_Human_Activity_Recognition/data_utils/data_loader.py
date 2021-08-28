"""Module for loading the data.

Example:
    from data_utils.data_loader import load_data()
    data, label_dict = load_data("../data/something-something-mini")

"""
import sys
sys.path.append('../')
sys.path.append('.')

from config import Config


try:
    from data_utils.metadata_loader import MetadataLoader
    from data_utils.video_loader import VideoLoader
except ModuleNotFoundError:
    from metadata_loader import MetadataLoader
    from video_loader import VideoLoader


def load_data(config):
    """Return object containing all data contained in folders with the given root.

    Returns:
        data: Data object which has the below format:
              data = {
                      'train': {
                                id1: {'id': id1, 'action_label': int, 'video': Tensor(n, w, h, c)}
                                id2: {'id': id2, 'action_label': int, 'video': Tensor(n, w, h, c)}
                                ...
                      }
                      'valid': { -||- }
                      'test': { -||- }
              }
              where n is the number of frames in the video, w is the video width, h is the video
              height, and c is the number of channels.

        label_dict: dictionary mapping label indices to label descriptions.

    """

    video_loader = VideoLoader(config)
    videos = video_loader.load_all_videos()
    metadata_loader = MetadataLoader(config)
    metadata = metadata_loader.load_metadata()
    label_dict = metadata_loader.get_label_dict()

    data = {}
    for subset in metadata:
        data[subset] = {}
        subset_metadata = metadata[subset]
        for id in subset_metadata:
            sample_metadata = subset_metadata[id]
            data[subset][id] = {'id': id,
                                'action_label': sample_metadata['action_label'],
                                'data': videos[id]}
    return data, label_dict




if __name__ == "__main__":
    config = Config()
    data, label_dict = load_data(config)
    data['train']
