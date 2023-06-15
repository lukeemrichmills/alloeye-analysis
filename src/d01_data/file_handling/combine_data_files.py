import numpy as np
import pandas as pd



#def combine_data_files(pid, block, practice=False, suffix="2"):
    # pass
pid = '51'
block = '2'
practice = False
suffix = '2'
overwrite_last = False

dir_in = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\test\\"
dir_out = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\test\\test_output\\"

r_or_p = 'r' if practice is False else 'p'
file_stem = f"{pid}{r_or_p}{block}"

gaze_headers = 'EyeTimestamp,UnityTime,EyeFrameSequence,FrameCount,FPS,Object,PosX,PosY,PosZ,ScaleX,' \
               'ScaleY,ScaleZ,ColX,ColY,ColZ,Trial,ViewNo,ViewPos,ViewingAngle,Rot?,ObjShifted,MoveCode,' \
               'cameraX,cameraY,cameraZ,camRotX,camRotY,camRotZ,leftPupilDiameter,rightPupilDiameter,' \
               'leftEyeOpenness,rightEyeOpenness,RcontrollerX,RcontrollerY,RcontrollerZ,LcontrollerX,' \
               'LcontrollerY,LcontrollerZ,RcontrRotX,RcontrRotY,RcontrRotZ,LcontrRotX,LcontrRotY,LcontrRotZ,' \
               'lGazeOrigin_x,lGazeOrigin_y,lGazeOrigin_z,rGazeOrigin_x,rGazeOrigin_y,rGazeOrigin_z,' \
               'lGazeDirection_x,lGazeDirection_y,lGazeDirection_z,rGazeDirection_x,rGazeDirection_y,' \
               'rGazeDirection_z,gazeConvergenceDist,gazeConvDistValidity,Object_notable,posX_notable,' \
               'posY_notable,posZ_notable,scaleX_notable,scaleY_notable,scaleZ_notable,colX_notable,' \
               'colY_notable,colZ_notable'
event_headers = "Unity_time,frame,event".split(',')
obj_pos_names = "Frame,Object,posX,posY,posZ,rotX,rotY,rotZ,R,G,B,A,NA".split(',')

trial_config_str = f"{file_stem}TrialConfigurations"
trial_info_str = f"{file_stem}TrialInfo"
trial_gaze_str = f"{file_stem}TrialGazeData"
all_gaze_str = f"{file_stem}AllGazeData"
obj_pos_str = f"{file_stem}ObjectPositions"
event_log_str = f"{file_stem}EventLog"
suffix1 = ".csv"
suffix2 = f"{suffix}.csv"
trial_config1 = pd.read_csv(f"{dir_in}{trial_config_str}{suffix1}").reset_index(drop=True)
trial_config2 = pd.read_csv(f"{dir_in}{trial_config_str}{suffix2}").reset_index(drop=True)
trial_info1 = pd.read_csv(f"{dir_in}{trial_info_str}{suffix1}").reset_index(drop=True)
trial_info2 = pd.read_csv(f"{dir_in}{trial_info_str}{suffix2}").reset_index(drop=True)
trial_gaze1 = pd.read_csv(f"{dir_in}{trial_gaze_str}{suffix1}", header=None, skiprows=1).reset_index(drop=True)
trial_gaze2 = pd.read_csv(f"{dir_in}{trial_gaze_str}{suffix2}", header=None, skiprows=1).reset_index(drop=True)
trial_gaze1.columns = gaze_headers.split(',')
trial_gaze2.columns = gaze_headers.split(',')
all_gaze1 = pd.read_csv(f"{dir_in}{all_gaze_str}{suffix1}", header=None, skiprows=1).reset_index(drop=True)
all_gaze2 = pd.read_csv(f"{dir_in}{all_gaze_str}{suffix2}", header=None, skiprows=1).reset_index(drop=True)
all_gaze1.columns = gaze_headers.split(',')
all_gaze2.columns = gaze_headers.split(',')
obj_positions1 = pd.read_csv(f"{dir_in}{obj_pos_str}{suffix1}", names=obj_pos_names).reset_index(drop=True).drop(['NA'], axis=1)
obj_positions2 = pd.read_csv(f"{dir_in}{obj_pos_str}{suffix2}", names=obj_pos_names).reset_index(drop=True).drop(['NA'], axis=1)
event_log1 = pd.read_csv(f"{dir_in}{event_log_str}{suffix1}", header=None, skiprows=1).reset_index(drop=True).iloc[:, :3]
event_log2 = pd.read_csv(f"{dir_in}{event_log_str}{suffix2}", header=None, skiprows=1).reset_index(drop=True).iloc[:, :3]
event_log1.columns = event_headers
event_log2.columns = event_headers



# for trial info, configurations
#
#   for n of copies - 1
#   append second copy trials to first
if overwrite_last is False:
    last_trial = trial_info1.TrialNumber.iloc[-1]
else:
    last_trial = trial_info1.TrialNumber.iloc[-2]

keep_trial_bool = trial_info1.TrialNumber <= last_trial
trial_info1 = trial_info1.loc[keep_trial_bool, :]
trial_info2 = trial_info2.loc[trial_info2.TrialNumber >= 0, :]
trial_info2.TrialNumber += last_trial + 1
trial_info_out = pd.concat([trial_info1, trial_info2]).reset_index(drop=True)
# trial_info_out.TrialNumber.iloc[1:] = [i for i in range(len(trial_info_out) - 1)]
keep_trial_bool_c = trial_config1.TrialNumber <= last_trial
trial_config1 = trial_config1.loc[keep_trial_bool_c, :]
trial_config2.TrialNumber += last_trial + 1
trial_config_out = pd.concat([trial_config1, trial_config2]).reset_index(drop=True)
# trial_config_out.TrialNumber.iloc[1:] = [i for i in range(len(trial_config_out) - 1)]
#   remove duplicated trial from end of first if it exists
#   adjust trial_no
#   replace relevant lines of trial config 1 with 2


trial_gaze1 = trial_gaze1.loc[trial_gaze1.Trial <= last_trial, :]
last_unity_frame = trial_gaze1.FrameCount.iloc[-1]
last_unity_time = trial_gaze1.UnityTime.iloc[-1]
last_eye_frame = trial_gaze1.EyeFrameSequence.iloc[-1]
last_eye_time = trial_gaze1.EyeTimestamp.iloc[-1]

trial_gaze2.FrameCount += last_unity_frame
trial_gaze2.UnityTime += last_unity_time
trial_gaze2.EyeFrameSequence += last_eye_frame
trial_gaze2.EyeTimestamp += last_eye_time
trial_gaze2.Trial += last_trial + 1

trial_gaze_out = pd.concat([trial_gaze1, trial_gaze2]).reset_index(drop=True)

all_gaze1 = all_gaze1.loc[all_gaze1.EyeTimestamp <= last_eye_time, :]
all_gaze2.FrameCount += last_unity_frame
all_gaze2.UnityTime += last_unity_time
all_gaze2.EyeFrameSequence += last_eye_frame
all_gaze2.EyeTimestamp += last_eye_time
all_gaze2.Trial += last_trial + 1

all_gaze_out = pd.concat([all_gaze1, all_gaze2]).reset_index(drop=True)

event_log1 = event_log1.loc[event_log1.frame <= last_unity_frame, :]
for i in range(len(event_log2)):
    try:
        event_log2.loc[i, 'Unity_time'] = str(float(event_log2.loc[i, 'Unity_time']) + last_unity_time)
    except ValueError:
        pass

event_log2.frame += last_unity_frame
event_log_out = pd.concat([event_log1, event_log2]).reset_index(drop=True)

bools = np.zeros(len(obj_positions1))
for i in range(len(obj_positions1)):
    try:
        bools[i] = 1 if int(obj_positions1.loc[i, 'Frame']) <= last_unity_frame else 0
    except ValueError:
        pass

for i in range(len(obj_positions2)):
    try:
        obj_positions2.loc[i, 'Frame'] = str(int(obj_positions2.loc[i, 'Frame']) + last_unity_frame)
    except ValueError:
        pass
obj_positions1 = obj_positions1.loc[obj_positions1.Frame <= last_unity_frame, :]

obj_positions_out = pd.concat([obj_positions1, obj_positions2])

# write
trial_info_out.to_csv(f"{dir_out}{trial_info_str}{suffix1}", index=False)
all_gaze_out.to_csv(f"{dir_out}{all_gaze_str}{suffix1}", index=False)
trial_gaze_out.to_csv(f"{dir_out}{trial_gaze_str}{suffix1}", index=False)
trial_config_out.to_csv(f"{dir_out}{trial_config_str}{suffix1}", index=False)
event_log_out.to_csv(f"{dir_out}{event_log_str}{suffix1}", index=False)
obj_positions_out.to_csv(f"{dir_out}{obj_pos_str}{suffix1}", index=False)

#
# if __name__ == __main__:
#     combine_data_files()