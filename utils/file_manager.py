import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

class FileManager:

    def __init__(self) -> None:
        pass

    def read_fasta(self,file):#alt
        fasta = []
        for record in SeqIO.parse(file, "fasta"):
            fasta.append([record.id, str(record.seq)])
        return fasta

    def convert_fasta_to_df(self,file):
        data=self.read_fasta(file)
        rows = []
        for i, d in enumerate(data):
            rows.append({'id': d[0], 'sequence': d[1]})

        # create pandas dataframe from the list of dictionaries
        df = pd.DataFrame(rows)
        return df
