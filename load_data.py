'''
1. specify type of exercise
2. get video and score from each subject
3. 1 json per video? or 1 json per exercise?
'''

import os
import pandas as pd
import subprocess

class KiMoReDataLoader(object):
    def __init__(self, configs):
        super(DataLoader, self).__init__()
        self.configs = configs

    def get_clinical_scores(rootdir):
        ''' Get all clinical scores from KIMORE dataset

        :return: A DataFrame object that contains all the clinicial scores for all subjects

                    clinical TS Ex#1  ...  clinical CF Ex#5
        Subject ID                    ...
        NE_ID16                 47.0  ...              35.0
        NE_ID11                 46.0  ...              35.0
        '''

        scoreDataList = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                # TODO: CHANGE THE NAME BELOW  TO ClinicalAssessment
                if file.find("ClinicalAssessment") != -1 and file.endswith(".xlsx"):
                    filename = os.path.join(subdir, file)
                    data = pd.read_excel(filename)
                    df = pd.DataFrame(data)
                    scoreDataList.append(df)
                    # print(filename)

        result = pd.concat(scoreDataList)
        # remove duplicated rows
        result = result.drop_duplicates()
        # set Subject ID as index
        result = result.set_index('Subject ID')
        return result

    def extract_frames_from_video(self):
        ########################################################################
        # Extract Frames from videos
        extracted_frame_dir = config.get('dataset', 'extracted_frame_path')
        video_data_dir = config.get('dataset', 'dataset_path')
        suffix = config.get('dataset', 'video_suffix')
        fps = config.getint('dataset', 'fps')

        if os.path.exists(extracted_frame_dir):
            subprocess.call('rm -rf {}'.format(extracted_frame_dir), shell=True)

        subprocess.call('mkdir {}'.format(extracted_frame_dir), shell=True)
        for name in video_names:
            video_full_name = name + suffix
            video_path = os.path.join(video_data_dir, video_full_name)
            if os.path.exists(video_path):
                print(video_path)
                frame_dir = os.path.join(extracted_frame_dir, name)
                subprocess.call('mkdir {}'.format(frame_dir), shell=True)
                # extrat frames from video and store in tmp
                # subprocess.call('ffmpeg -i {} -vf "scale={}:{},fps={}" tmp/image_%04d.jpg'
                #     .format(video_path, fps), shell=True)

                subprocess.call('ffmpeg -i {} -vf "fps={}" {}/image_%04d.jpg'
                                .format(video_path, fps, frame_dir), shell=True)


    def load_data(self):
        KIMORE_path = self.configs.get('dataset', 'KIMORE_path')
        # get df that contains all clinical scores for all exercises
        scores_df = self.getClinicalScores(KIMORE_path)



