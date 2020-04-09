pseudo_pos = [1,2,3,4,5,6,7,8]

P_loc          =  [0.0167, 0.03860, 0.04900, 0.03860, 0.016900, 0.06510000, 0.0000000000, 0.0386] # 6?
P_obs_loc      =  [0.0000, 0.00699, 0.00852, 0.00000, 0.031300, 0.00094600, 0.0000038700, 0.0000] # 3?
P_loc_obs_r    =  [0.0000, 0.00000, 0.00418, 0.00542, 0.000531, 0.00000616, 0.0000000655, 0.0000] # 1?
P_loc_obs_norm =  [0.0000, 0.02590, 0.40100, 0.52100, 0.051000, 0.00000000, 0.0000062900, 0.0000] # 5?


P_obs_loc_3 = P_loc_obs_r[3]/P_loc[3]
print(P_obs_loc_3)

P_loc_obs_r_1 = P_obs_loc[1]*P_loc[1]
P_loc_obs_r[1] = P_loc_obs_r_1
print(P_loc_obs_r_1)

SUM_P_loc_obs_r = sum(P_loc_obs_r)
P_loc_obs_norm_5 = (P_loc_obs_r[5]/SUM_P_loc_obs_r)
print(P_loc_obs_norm_5)

P_loc_6 = P_obs_loc[6]*P_loc_obs_r[6]
print(P_loc_6)
