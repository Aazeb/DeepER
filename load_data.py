import torch
import pickle
class Data:

    def __init__(self, data_dir="data/FB15k-237", reverse=False):
        print("Unimodals")
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

class MulData:

    def __init__(self, data_dir="data/FB15k-237", reverse=True):
        print("Multimodal")
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.entities_vemb = self.get_entities_vemb(data_dir)
        self.entities_semb = self.get_entities_semb(data_dir)
        self.mul_emb = self.get_multi_embeddings(self.entities_vemb, self.entities_semb, self.entities)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            data = [[i[0], i[2], i[1]] for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_entities_vemb(self, data_dir):
        with open("%s%s.pkl" % (data_dir, 'fb_vgg128_avg_normalized'), "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
        return p

    def get_entities_semb(self, data_dir):
        with open("%s%s.pkl" % (data_dir, 'FB_transE_100_norm'), "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
        return p

    def get_multi_embeddings(self, entities_vemb, entities_semb, entities):
        entities_vemb = entities_vemb
        entities_semb = entities_semb
        entities = entities
        entities_vemb_l = []
        entities_semb_l = []
        for e in entities:
            if e in entities_vemb:
                entities_vemb_l.append(entities_vemb[e])
        entities_vemb_t = torch.tensor(entities_vemb_l).cuda()

        for e in entities:
            if e in entities_semb:
                entities_semb_l.append(entities_semb[e])
        entities_semb_t = torch.tensor(entities_semb_l).cuda()
        mul_emb = torch.cat([entities_vemb_t, entities_semb_t], 1).cuda()
        mul_emb = mul_emb.float()
        return mul_emb
