import numpy as np

class FeatureExtractor:

    def freq_count(self,string, substr):
        return sum([1 for _ in range(len(string)-len(substr)+1) if string[_:(_+len(substr))] == substr])

    def extract_RDHP(self,dataframe):
        def RDHP(seq):
        
            sub = "qwer"
            subsub = [it1+it2 for it1 in sub for it2 in sub] 
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {} 
            aasub["q"] = "PALVIFWM"
            aasub["w"] = "QSTYCNG"
            aasub["e"] = "HKR"
            aasub["r"] = "DE"
            
            seq1 = seq
            lenn=len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa,key)
            
            freq2 ={}
            for item in sub:
                freq2[item] = self.freq_count(seq2, item)
            for item in subsub:
                freq2[item] = self.freq_count(seq2, item)
                
            freq1={}
            for item in aalist:
                freq1[item] = self.freq_count(seq1, item)
                
            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key]/lenn)
                
            for item in  aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item]/max(1,freq2[key]))
                        break
                        
            for item in subsub:
                feat.append(freq2[item]/(freq2[item[0]]+1))
            
            feat = np.array(feat)
            feat_len = len(feat)
            feat = feat.reshape(1,feat_len)
                
            return feat

        X_train=[]
        y_train=[]
        for index, row in dataframe.iterrows():
            X_train.append(RDHP(row.sequence)[0])
            y_train.append(row.label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        return X_train,y_train

    def extract_RSACID(self,dataframe):
        def reducedACID(seq):
            def fcount(string, substr):
                count = 0
                pos = 0
                while(True):
                    pos = string.find(substr , pos)
                    if pos > -1:
                        count = count + 1
                        pos += 1
                    else:
                        break
                return count

            
            sub = "akn"
            subsub = [it1+it2 for it1 in sub for it2 in sub] 
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["a"] = "DE"
            aasub["k"] = "KHR"
            aasub["n"] = "ACFGILMNPQSTVWY"
            
            seq1 = seq
            lenn=len(seq1)
        
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa,key)
            
            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)
                
            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)
                
            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key]/lenn)
                
            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item]/max(1,freq2[key]))
                        break
                        
            for item in subsub:
                feat.append(freq2[item]/(freq2[item[0]]+1))
            
            feat = np.array(feat)
            feat_len = len(feat)
            feat = feat.reshape(1,feat_len)
                
            return feat


        X=[]
        y=[]
        for index, row in dataframe.iterrows():
            X.append(reducedACID(row.sequence)[0])
            y.append(row.label)

        X = np.array(X)
        y = np.array(y)

        return X,y
    
    def extract_RSSECOND(self,dataframe):
        def reducedSECOND(seq):
            def fcount(string, substr):
                count = 0
                pos = 0
                while(True):
                    pos = string.find(substr , pos)
                    if pos > -1:
                        count = count + 1
                        pos += 1
                    else:
                        break
                return count

    
            sub = "qwe"
            subsub = [it1+it2 for it1 in sub for it2 in sub] 
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "EHALMQKR"
            aasub["w"] = "VTIYCWF"
            aasub["e"] = "GDNPS"
            
            seq1 = seq
            lenn=len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa,key)
            
            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)
                
            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)
                
            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key]/lenn)
                
            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item]/max(1,freq2[key]))
                        break
                        
            for item in subsub:
                feat.append(freq2[item]/(freq2[item[0]]+1))
            
            feat = np.array(feat)
            feat_len = len(feat)
            feat = feat.reshape(1,feat_len)
                
            return feat
        
        X=[]
        y=[]
        for index, row in dataframe.iterrows():
            X.append(reducedSECOND(row.sequence)[0])
            y.append(row.label)

        X = np.array(X)
        y = np.array(y)

        return X,y


    def extract_RSPOLAR(self,dataframe):
        def reducedPOLAR(seq):
            def fcount(string, substr):
                count = 0
                pos = 0
                while(True):
                    pos = string.find(substr , pos)
                    if pos > -1:
                        count = count + 1
                        pos += 1
                    else:
                        break
                return count

    
            sub = "qwert"
            subsub = [it1+it2 for it1 in sub for it2 in sub] 
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "DE"
            aasub["w"] = "RHK"
            aasub["e"] = "WYF"
            aasub["r"] = "SCMNQT"
            aasub["t"] = "GAVLIP"
            
            seq1 = seq
            lenn=len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa,key)
            
            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)
                
            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)
                
            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key]/lenn)
                
            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item]/max(1,freq2[key]))
                        break
                        
            for item in subsub:
                feat.append(freq2[item]/(freq2[item[0]]+1))
            
            feat = np.array(feat)
            feat_len = len(feat)
            feat = feat.reshape(1,feat_len)
                
            return feat
        
        X=[]
        y=[]
        for index, row in dataframe.iterrows():
            X.append(reducedPOLAR(row.sequence)[0])
            y.append(row.label)

        X = np.array(X)
        y = np.array(y)

        return X,y
    

    def extract_RSCHARGE(self,dataframe):

        def reducedCHARGE(seq):
            def fcount(string, substr):
                count = 0
                pos = 0
                while(True):
                    pos = string.find(substr , pos)
                    if pos > -1:
                        count = count + 1
                        pos += 1
                    else:
                        break
                return count

            sub = "qwe"
            subsub = [it1+it2 for it1 in sub for it2 in sub] 
            aalist = "ACDEFGHIKLMNPQRSTVWY"
            aasub = {}
            aasub["q"] = "KR"
            aasub["w"] = "AVNCQGHILMFPSTWY"
            aasub["e"] = "DE"
            
            seq1 = seq
            lenn=len(seq1)
            seq2 = seq1
            for key, value in aasub.items():
                for aa in value:
                    seq2 = seq2.replace(aa,key)
            
            freq2 = {}
            for item in sub:
                freq2[item] = fcount(seq2, item)
            for item in subsub:
                freq2[item] = fcount(seq2, item)
                
            freq1 = {}
            for item in aalist:
                freq1[item] = fcount(seq1, item)
                
            feat = []
            for key, value in aasub.items():
                feat.append(freq2[key]/lenn)
                
            for item in aalist:
                for key, value in aasub.items():
                    if item in value:
                        feat.append(freq1[item]/max(1,freq2[key]))
                        break
                        
            for item in subsub:
                feat.append(freq2[item]/(freq2[item[0]]+1))
            
            feat = np.array(feat)
            feat_len = len(feat)
            feat = feat.reshape(1,feat_len)
                
            return feat
        
        X=[]
        y=[]
        for index, row in dataframe.iterrows():
            X.append(reducedCHARGE(row.sequence)[0])
            y.append(row.label)

        X = np.array(X)
        y = np.array(y)

        return X,y
