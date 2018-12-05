import pickle

a = pickle.load(open('GamePlay.illuminati', 'rb'))
for frame_num in range(len(a)):
    print(a[frame_num][1])
