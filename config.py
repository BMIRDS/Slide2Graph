

class Config():
    r"""The Config class defines
    - hyperparameters for model and training loop
    - paths of saved graphs and models

    """

    #Train:
    batch_size = 8
    learning_rate = 0.00025
    epochs = 200
    weight_decay = 5e-4

    #Model:
    type = 'GraphCls' #Customized graph model name
    hidden_size = 256

    #Processed_graph:
    train_graphs = 'clean_graph_updated_4allconn/train_graphs.pkl' #Where the graphs in the training set to be saved
    test_graphs = 'clean_graph_updated_4allconn/test_graphs.pkl' #Where the graphs in the test set to be saved
    val_graphs = 'clean_graph_updated_4allconn/val_graphs.pkl' #Where the graphs in the validation set to be saved

    #Patch-level preprocessing for graph construction
    val_model = 'clean_085'  #ResNet18 or other models that will be used to extract patches' features
    validation_raw_src = 'allp085_updated' #The directory to all the patch images
    val_raw_pkl = 'allp085_updated.pkl' #Dictionaries of patches' name, class, split
    #^ TODO: GENERALIZE NAMES ABOVE. ALSO SHOW DICT STRUCTURE.