from tqdm import tqdm
import pickle


def save_preprocessed(state):
    r_path = './data/kowikitext/raw/kowikitext' + '.' + state
    w_path = './data/kowikitext/processed/kowikitext' + '.' + state

    with open(r_path, 'r') as f:
        lines = f.readlines()
    
    all_s = []
    for l in lines:
        l = l.strip()
        if len(l.split('.')) >= 3:
            s1, s2 = l.split('.')[:2]
            if len(' '.join(s1.split(' '))) > 20 and len(' '.join(s2.split(' '))) > 20:
                all_s.append('<:::::::>'.join([s1, s2 + '\n']))
    
    with open(w_path, 'w') as f:
        f.writelines(all_s)


def merge_data():
    all_s = []
    for state in ['train', 'val', 'test']:
        p = './data/kowikitext/processed/kowikitext' + '.' + state
        
        with open(p, 'r') as f:
            lines = f.readlines()

        for line in lines:
            s1, s2 = line.strip().split('<:::::::>')
            all_s.append(s1.strip() +'\n')
            all_s.append(s2.strip() +'\n')
    
    p = './data/kowikitext/processed/kowikitext.all'
    with open(p, 'w') as f:
        f.writelines(all_s)
    

def convert2pickle(state):
    for s in state:
        pairs = []
        r_path = './data/kowikitext/processed/kowikitext' + '.' + s
        w_path = './data/kowikitext/processed/kowikitext' + '.' + s + '.pkl'

        with open(r_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            s1, s2 = line.split('<:::::::>')
            s1, s2 = s1.strip(), s2.strip()
            pairs.append((s1, s2))
        
        with open(w_path, 'wb') as f:
            pickle.dump(pairs, f)
        


if __name__ == '__main__':
    state = ['train', 'val', 'test']
    # for s in tqdm(state):
    #     save_preprocessed(s)

    # merge_data()
    convert2pickle(state)

