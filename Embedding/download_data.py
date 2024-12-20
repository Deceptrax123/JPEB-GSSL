from torch_geometric.datasets import Planetoid
import os 
from dotenv import load_dotenv

def main():
    load_dotenv('.env')

    inp_name=input("Enter dataset to be downloaded: ")
    cora_path=os.getenv('Cora')
    pubmed_path=os.getenv('Pubmed')
    citeseer_path=os.getenv('CiteSeer')

    if inp_name=='cora':
        cora=Planetoid(root=cora_path,name='Cora')
    elif inp_name=='pubmed':
        pubmed=Planetoid(root=pubmed_path,name='PubMed')
    elif inp_name=='citeseer':
        citeseer=Planetoid(root=citeseer_path,name='CiteSeer')
    

if __name__=='__main__':
    main()