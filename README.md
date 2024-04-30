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
- [ ] Clone KeAP model, 
- [ ] train on PKG25 data
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
- [ ] set up full training loop
- [ ] Train
