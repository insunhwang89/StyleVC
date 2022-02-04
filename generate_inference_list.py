
import numpy as np
import random
import os
import model.hparams as hp

print("이거 하면 기존 파일을 덮어씌우니 주의하자")

def generate_path(path, speaker, text_id):

    # 파일이 없는 경우 다시 추출해서 진행함
    try:
        
        line = "val/{}/{}_{:03d}.npz".format(speaker, speaker, text_id) # val/p226/p226_001.npz

        # 없는 파일 제외하기   '/dev/shm/INSUN/datasets/VCTK/VCTK_22K_hifi_org/val/p232/p232_001.npz'
        npz = np.load(os.path.join(path, line))   # /dev/shm/INSUN/datasets/' + dataset + '/' + dataset_path + '/train'

        return line
    except:
        # print("짧거나 긴 음성!, path: ", os.path.join(path, line))
        return False

def make_inference_sample_list(path, source_list, target_list, sample_speakers, sample_num):

    sample_list = list()
    # utterance마다, source speaker당 sample개수만큼 target speaker 선정해서 conversion pair 생성한다
    for source_speaker in source_list:
        data_number = np.random.randint(1, 20)
        source_line = generate_path(path, source_speaker, data_number) # val/p226/p226_001.npz'
        target_list_ = random.sample(target_list, sample_speakers) # p270
        for target_speaker in target_list_: 
            for _ in range(sample_num):
                data_number = np.random.randint(1, 20)
                target_line = generate_path(path, target_speaker, data_number)   #  val/p270/p270_005.npz
                if source_line is not False and target_line is not False: 
                    sample_list.append(source_line + "|" + target_line + "\n") # ['val/p226/p226_001.npz|val/p270/p270_005.npz\n', ...]
                        # val/p225/p225_003.npz|val/p225/p225_003.npz
    print("generate count: ", len(sample_list))

    return sample_list


print("\n\n[*] Generating inference sample list!")

path = 'data/VCTK/VCTK_16K'

unseen_speakers_male = hp.unseen_speakers_male
unseen_speakers_female = hp.unseen_speakers_female

seen_speakers_male = [ele for ele in hp.male if ele not in unseen_speakers_male]
seen_speakers_female = [ele for ele in hp.female if ele not in unseen_speakers_female]

seen_list = list()
unseen_list = list()

# 각 test set의 sample 1~20번까지 conversion pair 생성한다.
# Source speaker마다 random으로 target speaker 골라서 해당하는 utterance로 covnersion pair 생성한다.
sample_speakers = 9 # target speaker중에 고른다
sample_num=1 # 고른 target speaker에서 다른 sample고른다
seen_list += make_inference_sample_list(path, seen_speakers_male, seen_speakers_male, sample_speakers, sample_num)
seen_list += make_inference_sample_list(path, seen_speakers_female, seen_speakers_male, sample_speakers, sample_num)
seen_list += make_inference_sample_list(path, seen_speakers_male, seen_speakers_female, sample_speakers, sample_num)
seen_list += make_inference_sample_list(path, seen_speakers_female, seen_speakers_female, sample_speakers, sample_num)

sample_speakers = 10
sample_num = 4
unseen_list += make_inference_sample_list(path, unseen_speakers_male, unseen_speakers_male, sample_speakers, sample_num)
unseen_list += make_inference_sample_list(path, unseen_speakers_female, unseen_speakers_male, sample_speakers, sample_num)
unseen_list += make_inference_sample_list(path, unseen_speakers_male, unseen_speakers_female, sample_speakers, sample_num)
unseen_list += make_inference_sample_list(path, unseen_speakers_female, unseen_speakers_female, sample_speakers, sample_num)


# with open("/dev/shm/INSUN/datasets/evaluation_list_org_parallel_INTRA_" + dataset + ".txt", mode="w") as file:
with open(os.path.join('data', 'VCTK', 'seen') + "_list.txt", mode="w") as file:
    print("extract len: ", len(seen_list))
    seen_list = set(seen_list) # 순서 없음, 중복제거
    print("final len: ", len(seen_list))
    for line in seen_list:
        file.writelines(line)

with open(os.path.join('data', 'VCTK', 'unseen') + "_list.txt", mode="w") as file:
    print("extract len: ", len(unseen_list))
    unseen_list = set(unseen_list) # 순서 없음, 중복제거
    print("final len: ", len(unseen_list))
    for line in unseen_list:
        file.writelines(line)
