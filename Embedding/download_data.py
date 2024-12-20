from torch_geometric.datasets import Planetoid,Amazon
import os 
from dotenv import load_dotenv

def main():
    load_dotenv('.env')

    inp_name=input("Enter dataset to be downloaded: ")
    cora_path=os.getenv('Cora')
    pubmed_path=os.getenv('Pubmed')
    citeseer_path=os.getenv('CiteSeer')
    computers_path=os.getenv('Computers')
    photos_path=os.getenv('Photo')

    if inp_name=='cora':
        cora=Planetoid(root=cora_path,name='Cora')
    elif inp_name=='pubmed':
        pubmed=Planetoid(root=pubmed_path,name='PubMed')
    elif inp_name=='citeseer':
        citeseer=Planetoid(root=citeseer_path,name='CiteSeer')
    elif inp_name=='computers':
        computers=Amazon(root=computers_path,name='Computers')
    elif inp_name=='photos':
        photos=Amazon(root=photos_path,name='Photo')

    

if __name__=='__main__':
    main()