'''
1. specify type of exercise
2. get video and score from each subject
3. 1 json per video? or 1 json per exercise?
'''

import os
import pandas as pd
import subprocess

class KiMoReDataLoader(object):
    def __init__(self, config):
        super(KiMoReDataLoader, self).__init__()
        self.config = config
        self.df = None
        self.max_video_sec = -1

    def get_clinical_scores(self):
        ''' Get all clinical scores from KIMORE dataset

        :return: A DataFrame (68 x 15) object that contains all the clinicial scores for all subjects

                    clinical TS Ex#1  ...  clinical CF Ex#5
        Subject ID                    ...
        NE_ID16                 47.0  ...              35.0
        NE_ID11                 46.0  ...              35.0
        '''

        print('Extracting Clinical Scores...')
        rootdir = self.config.get('dataset', 'KIMORE_path')

        os.chdir(rootdir)

        list = []
        scoreDataList = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                # TODO: CHANGE THE NAME BELOW  TO ClinicalAssessment

                if 'ClinicalAssessment' in file != -1 and file.endswith('.xlsx'):
                    filename = os.path.join(subdir, file)
                    data = pd.read_excel(filename)
                    df = pd.DataFrame(data)
                    scoreDataList.append(df)
                    print(filename)
                    id = filename.split('/')[-4]
                    if id not in list:
                        list.append(id)
        result = pd.concat(scoreDataList)
        # remove duplicated rows
        result = result.drop_duplicates()
        # set Subject ID as index
        result = result.set_index('Subject ID')
        self.df = result
        return list

    def get_video_length(self, videoname):
        ''' Get the duration of videos in seconds   (ref: https://stackoverflow.com/a/3844467)

        :param videoname: path to video
        :return: A integer: duration of video in seconds (round down)
        '''
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", videoname],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        # round down
        return int(float(result.stdout))

    def get_video_names(self):
        ''' Add RGB video names to video_name column of df and get the max duration of videos for given exercise_type

        :return: Max duration of videos in second
        '''
        print('Extracting Video Names...')

        exercise_type = self.config.get('dataset', 'exercise_type')
        video_data_dir = self.config.get('dataset', 'KIMORE_RGB_path')
        dataset_filter = self.config.get('dataset', 'dataset_filter')

        # Add a new column "video_name" to df
        self.df = self.df.assign(video_name=0)
        max_video_sec = -1
        list = []
        os.chdir(video_data_dir)

        for subdir, dirs, files in os.walk(video_data_dir):
            for file in files:
                video_name = os.path.join(subdir, file)
                if dataset_filter in video_name and exercise_type in video_name and video_name.endswith(".mp4"):

                    # update max_video_length
                    duration = self.get_video_length(video_name)
                    max_video_sec = duration if duration > max_video_sec else max_video_sec

                    subject_ID = subdir.split('/')[-3]
                    if '_ID' not in subject_ID:
                        print('Wrong subject_ID type!')
                        assert False

                    # insert video name to df
                    self.df.loc[subject_ID, 'video_name'] = video_name
                    print(video_name)
                    list.append(subject_ID)

        self.max_video_sec = max_video_sec
        return list

    def find_missing_data(self, scores, videos):
        # Extract Frames from videos
        extracted_frame_root = self.config.get('dataset', 'extracted_frame_path')
        exercise_type = self.config.get('dataset', 'exercise_type')
        # output = os.path.join(extracted_frame_root, exercise_type)

        missing_score = []
        missing_video = []
        for i in scores + videos:
            if i not in scores:
                missing_score.append(i)
            if i not in videos:
                missing_video.append(i)

        output = os.path.join(extracted_frame_root,  exercise_type + "missing_data.txt")
        text_file = open(output, "w")
        text_data = "{} \n Missing RGB Videos: \n {} \n \n Missing Scores: \n {} ".format(exercise_type, missing_video, missing_score)
        text_file.write(text_data)
        text_file.close()

    def load_data(self):
        print('Loading KIMORE Dataset...')
        # get df that contains all clinical scores for all exercises
        scores_list = self.get_clinical_scores()

        # TODO: create method to clean NA data
        # NOTE: 'E_ID17' does not have scores
        self.df = self.df.dropna(subset=['clinical TS Ex#1'])

        # add video names to df and set max_video_sec
        videos_list = self.get_video_names()

        # find any missing RGB videos
        # self.find_missing_data(scores_list, videos_list)

        # TODO
        # remove missing data
        self.df = self.df[self.df['video_name'] != 0]
        df_path = self.config.get('dataset', 'df_path')
        os.chdir(df_path)
        dataset_filter = self.config.get('dataset', 'dataset_filter')
        df_name = dataset_filter + '_df'
        self.df.to_pickle(df_name)

        print('Finished loading Dataset')