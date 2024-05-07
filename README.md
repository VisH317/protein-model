## Ideas for protein modeling

**Current Tools**
- ProteinKG25 - a knowledge graph that provides information about a protein's functions, use cases, and involved processes, which can connect to other RELATED proteins
- MSAs - aligned sequences of similar length 
- RSA retriever - involves attention embedding of each sequence, which is then similarity searched to find similar sequences to cross attend to
- Structure - we can leverage 3d structure graphs of proteins, as well as atom-level graphs, to understand secondary and tertiary structure directly
- 

**Current Idea brainstorming**
- CLIP-based architecture?
- Retriever based on similarity from the knowledge graph + relational information


**KeAP-RSA Model Timeline (non-graph connected)**
- [X] Create the PKG25 dataloader
- [X] Clone KeAP model, 
- [X] train on PKG25 data
- [ ] Create structure-function vector database
- [ ] Set up RSA retriever using dense similarity search
- [ ] Set up transformer architecture with RSA-augmented sequences
- [ ] Test model dimensions
- [ ] Train

**ProtCLIP Timeline**
- [X] Create PKG25 dataloader
- [X] Find a Med LM and clone or find an API to call from
- [X] Find a protein model and clone to run in the model training (small version)
- [X] Set up CLIP objective module + lightning training loop
- [X] set up full training loop
- [ ] Test efficacy
- [ ] Train


**ProtKEAP Methods for Retrieval**
- Default retriever method: RSA retriever + cross attention on pooler embeds
- Proper structure + function combination - train on custom KeAP model
  - RSA method - retrieve 5 relevant sequences, parse each sequence and weight the output
  - Identical method for KeAP - retrieve KeAP, run on each triplet, weighted average triplet at output of the protein
- Problem - need to convert from a protein space to a PRA space to integrate structure and function
  - provided through a similar method through cross attention
- alternative method - providing the sequences to parse using ProtBERT, + KeAP model
- instead of plain retriever, use a graph-based retriever
  - graph processing to get information to provide more information (need to check this as an alternative method)
