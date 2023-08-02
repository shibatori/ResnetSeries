from random import shuffle


# classes = ["Analysis","Backdoor","DoS","Exploits","Fuzzers",
#         "Generic","Normal","Reconnaissance","Shellcode","Worms"]

def preprocess_iris_data(unsw_data_file1, unsw_data_file2, train_file, test_file, header=True):
    cls_0 = "Analysis"
    cls_1 = "Backdoor"
    cls_2 = "DoS"
    cls_3 = "Exploits"
    cls_4 = "Fuzzers"
    cls_5 = "Generic"
    cls_6 = "Normal"
    cls_7 = "Reconnaissance"
    cls_8 = "Shellcode"
    cls_9 = "Worms"

    cls_0_samples = []
    cls_1_samples = []
    cls_2_samples = []
    cls_3_samples = []
    cls_4_samples = []
    cls_5_samples = []
    cls_6_samples = []
    cls_7_samples = []
    cls_8_samples = []
    cls_9_samples = []

    with open(unsw_data_file1, "r", encoding="UTF8") as fp:
        lines = fp.readlines()
        headers="120,4,setosa,versicolor,virginica"
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == headers:
                continue
            if cls_0 in line:
                cls_0_samples.append(line)
                continue
            if cls_1 in line:
                cls_1_samples.append(line)
                continue
            if cls_2 in line:
                cls_2_samples.append(line)
                continue
            if cls_3 in line:
                cls_3_samples.append(line)
                continue
            if cls_4 in line:
                cls_4_samples.append(line)
                continue
            if cls_5 in line:
                cls_5_samples.append(line)
                continue
            if cls_6 in line:
                cls_6_samples.append(line)
                continue
            if cls_7 in line:
                cls_7_samples.append(line)
                continue
            if cls_8 in line:
                cls_8_samples.append(line)
                continue
            if cls_9 in line:
                cls_9_samples.append(line)

    with open(unsw_data_file2, "r", encoding="UTF8") as fp:
        lines = fp.readlines()
        headers="120,4,setosa,versicolor,virginica"
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line == headers:
                continue
            if cls_0 in line:
                cls_0_samples.append(line)
                continue
            if cls_1 in line:
                cls_1_samples.append(line)
                continue
            if cls_2 in line:
                cls_2_samples.append(line)
                continue
            if cls_3 in line:
                cls_3_samples.append(line)
                continue
            if cls_4 in line:
                cls_4_samples.append(line)
                continue
            if cls_5 in line:
                cls_5_samples.append(line)
                continue
            if cls_6 in line:
                cls_6_samples.append(line)
                continue
            if cls_7 in line:
                cls_7_samples.append(line)
                continue
            if cls_8 in line:
                cls_8_samples.append(line)
                continue
            if cls_9 in line:
                cls_9_samples.append(line)

    shuffle(cls_0_samples)
    shuffle(cls_1_samples)
    shuffle(cls_2_samples)
    shuffle(cls_3_samples)
    shuffle(cls_4_samples)
    shuffle(cls_5_samples)
    shuffle(cls_6_samples)
    shuffle(cls_7_samples)
    shuffle(cls_8_samples)
    shuffle(cls_9_samples)

    print("number of class 0: {}".format(len(cls_0_samples)), flush=True)
    print("number of class 1: {}".format(len(cls_1_samples)), flush=True)
    print("number of class 2: {}".format(len(cls_2_samples)), flush=True)
    print("number of class 3: {}".format(len(cls_3_samples)), flush=True)
    print("number of class 4: {}".format(len(cls_4_samples)), flush=True)
    print("number of class 5: {}".format(len(cls_5_samples)), flush=True)
    print("number of class 6: {}".format(len(cls_6_samples)), flush=True)
    print("number of class 7: {}".format(len(cls_7_samples)), flush=True)
    print("number of class 8: {}".format(len(cls_8_samples)), flush=True)
    print("number of class 9: {}".format(len(cls_9_samples)), flush=True)

    train_samples = cls_0_samples[:2462] + cls_1_samples[:2142] + cls_2_samples[:15044] + \
                    cls_3_samples[:40963] + cls_4_samples[:22306] + cls_5_samples[:54161] + \
                    cls_6_samples[:85560] + cls_7_samples[:12868] + cls_8_samples[:1390] + cls_9_samples[:160]
    test_samples = cls_0_samples[2462:] + cls_1_samples[2142:] + cls_2_samples[15044:] + \
                    cls_3_samples[40963:] + cls_4_samples[22306:] + cls_5_samples[54161:] + \
                    cls_6_samples[85560:] + cls_7_samples[12868:] + cls_8_samples[1390:] + cls_9_samples[160:]

    header_content = "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,attack_cat,label"

    with open(train_file, "w", encoding="UTF8") as fp:
        fp.write("{}\n".format(header_content))
        for sample in train_samples:
            fp.write("{}\n".format(sample))

    with open(test_file, "w", encoding="UTF8") as fp:
        fp.write("{}\n".format(header_content))
        for sample in test_samples:
            fp.write("{}\n".format(sample))


def main():
    unsw_data_file1 = "UNSW_NB15_training-set.csv"
    unsw_data_file2 = "UNSW_NB15_testing-set.csv"
    unsw_train_file = "unsw_train.csv"
    unsw_test_file = "unsw_test.csv"

    preprocess_iris_data(unsw_data_file1, unsw_data_file2, unsw_train_file, unsw_test_file,header=False)

main()
