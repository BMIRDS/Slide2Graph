

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
    output_dst = 'graphs_dir'
    train_graphs = output_dst + '/train_graphs.pkl' #Where the graphs in the training set to be saved
    test_graphs = output_dst + '/test_graphs.pkl' #Where the graphs in the test set to be saved
    val_graphs = output_dst + '/val_graphs.pkl' #Where the graphs in the validation set to be saved

    #Patch-level preprocessing for graph construction
    val_model = 'Patch_feature_extractor'  #ResNet18 or other models that will be used to extract patches' features
    validation_raw_src = 'patch_images_dir' #The directory to all the patch images
    val_raw_pkl = 'WSI_information.pkl' #DICT STRUCTURE: [{index:WSI_name}, {index:class}, {index:split}]
    name_map = {'NotAnnotated':1, 'Neoplastic':0, 'Positive':2}

    #Patch level parameters
    windows_size = 224
    nonoverlap_factor = 2/3
    num_classes = 3
    path_mean = [0.7725, 0.5715, 0.6842]
    path_std = [0.1523, 0.2136, 0.1721]

    #Train settings
    batch_size = 32
    num_workers = 8
    mode = 'train'
    model_path = 'trained_model.pth'
