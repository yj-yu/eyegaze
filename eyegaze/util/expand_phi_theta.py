import os
import sys
import subprocess

import json

video_id = sys.argv[1]

f = open('../phi_theta/' + video_id + '_inception.json', 'r')
phi_theta = json.load(f)['phi_theta_key']

phi_theta_expand = []
length = len(phi_theta)


length = length - length%5


for i in range(length/5):
    for j in range(80):
        phi_theta_expand.append(phi_theta[i*5])


f_new = open('../phi_theta/' + video_id + '_inception_expand.json', 'w')

phi_theta_expand_dict = {'phi_theta_key': phi_theta_expand}

json.dump(phi_theta_expand_dict, f_new)

