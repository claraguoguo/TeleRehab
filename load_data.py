'''
1. specify type of exercise
2. get video and score from each subject
3. 1 json per video? or 1 json per exercise?
'''

import os
import pandas as pd
import subprocess
# import ffmpeg
import numpy as np
# import cv2
# import math
# import skvideo.io
from numpy import savetxt

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

        print('Change directory {}'.format(rootdir))
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

        # Add a new column "video_name" to df
        self.df = self.df.assign(video_name=0)
        max_video_sec = -1
        list = []
        print('Change directory {}'.format(video_data_dir))
        os.chdir(video_data_dir)

        for subdir, dirs, files in os.walk(video_data_dir):
            for file in files:
                video_name = os.path.join(subdir, file)
                if exercise_type in video_name and video_name.endswith(".mp4"):

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



    def extract_frames_from_video(self):
        # Extract Frames from videos
        extracted_frame_root = self.config.get('dataset', 'extracted_frame_path')
        exercise_type = self.config.get('dataset', 'exercise_type')
        extracted_frame_subfolder = os.path.join(extracted_frame_root, exercise_type)

        video_data_dir = self.config.get('dataset', 'KIMORE_RGB_path')

        # Note: fps MUST be s String, b/c kvideo.io.vread is expecting a string input.
        fps = self.config.get('dataset', 'fps')

        print('Extracting Frames for ' + exercise_type)
        if not os.path.exists(extracted_frame_root):
            subprocess.call('mkdir {}'.format(extracted_frame_root), shell=True)

        if os.path.exists(extracted_frame_subfolder):
            subprocess.call('rm -rf {}'.format(extracted_frame_subfolder), shell=True)
        counter = 0
        list = []
        subprocess.call('mkdir {}'.format(extracted_frame_subfolder), shell=True)
        max_num_frames = -1
        for subdir, dirs, files in os.walk(video_data_dir):
            for file in files:
                video_name = os.path.join(subdir, file)
                if exercise_type in video_name and video_name.endswith(".mp4"):
                    subject_ID = subdir.split('/')[-3]
                    if '_ID' not in subject_ID:
                        print('Wrong subject_ID type!')
                        assert False

                    print(video_name)
                    counter += 1
                    if subject_ID not in list:
                        list.append(subject_ID)
                    '''
                    # Note: we are not using the following code, because the npy file generated is TOO BIG (~1.6GB)!!
                    
                    videodata = skvideo.io.vread(video_name, outputdict={'-r': fps})
                    max_num_frames = videodata.shape[0] if videodata.shape[0] > max_num_frames else max_num_frames
                    output = os.path.join(extracted_frame_subfolder, subject_ID)
                    np.save(output, videodata)
                    '''
        print(counter)
        return list
        # count = 0
        # cap = cv.VideoCapture(filename)  # capturing the video from the given path
        # frameRate = cap.get(5)  # frame rate
        # outputPath = os.path.join(extracted_frame_subfolder, subject_ID)
        # while (cap.isOpened()):
        #     frameId = cap.get(1)  # current frame number
        #     ret, frame = cap.read()
        #     if (ret != True):
        #         break
        #     if (frameId % math.floor(frameRate) == 0):
        #         frameName = "frame%d.jpg" % count;
        #         count += 1
        #         cv.imwrite(os.path.join(outputPath, frameName), frame)
        # cap.release()
        # print("Done!")

        # out, _ = (
        #     ffmpeg
        #         .input(filename)
        #         .filter('fps', fps=fps, round='up')
        #         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        #         .run(capture_stdout=True)
        # )
        # video = (
        #     np
        #         .frombuffer(out, np.uint8)
        # )


        # for name in video_names:
        #     video_full_name = name + suffix
        #     video_path = os.path.join(video_data_dir, video_full_name)
        #     if os.path.exists(video_path):
        #         print(video_path)
        #         frame_dir = os.path.join(extracted_frame_dir, name)
        #         subprocess.call('mkdir {}'.format(frame_dir), shell=True)
        #         # extrat frames from video and store in tmp
        #         # subprocess.call('ffmpeg -i {} -vf "scale={}:{},fps={}" tmp/image_%04d.jpg'
        #         #     .format(video_path, fps), shell=True)
        #
        #         subprocess.call('ffmpeg -i {} -vf "fps={}" {}/image_%04d.jpg'
        #                         .format(video_path, fps, frame_dir), shell=True)

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
        self.df.to_pickle('dataframe')

        print('Finished loading Dataset')