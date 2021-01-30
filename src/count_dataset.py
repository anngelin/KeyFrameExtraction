import pathlib


# path = pathlib.Path('/home/datasets/images/IJB/IJB-C/images/img/')


path = pathlib.Path('C:\\akharche\\MasterThesis\\ijbc_1N_probe_mixed.csv')
path_new ='C:\\akharche\\MasterThesis\\ijbc_1N_all_imgs.csv'

imgs = set()

with open(path_new, 'a', encoding='utf-8') as fw:
    with open(path, 'r', encoding='utf-8') as csvreader:
        all_data = csvreader.readlines()
        count = 0
        for line in all_data[1:]:
            old_line = line
            data = line.strip().split(',')
            template_id = data[0]
            frame_id = data[2]
            if 'img/' in frame_id:
                if frame_id in imgs:
                    print(frame_id)
                else:
                    imgs.add(frame_id)
                    line = old_line.strip()
                    fw.write(line + '\n')





