from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age, n_init=3, nn_budget=100)

    def update(self, detections, frame):
        """
        Takes detections and frame, outputs tracked tracks.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
