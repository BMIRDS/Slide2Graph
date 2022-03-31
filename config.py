class Config():
    #Train:
    batch_size = 8
    learning_rate = 0.00025
    epochs = 200
    weight_decay = 5e-4
    #Model:
    type = 'GraphCls'
    hidden_size = 256

    #Processed_graph:
    train_graphs = 'clean_graph_updated_4allconn/train_graphs.pkl'
    test_graphs = 'clean_graph_updated_4allconn/test_graphs.pkl'
    val_graphs = 'clean_graph_updated_4allconn/val_graphs.pkl'

    #Patch-level preprocessing for graph construction
    val_model = 'clean_085'
    validation_raw_src = 'allp085_updated'
    val_raw_pkl = 'allp085_updated.pkl'