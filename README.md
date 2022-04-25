# Slide2Graph: A Graph Convolutional Network-based Whole-Slide Analysis

Slide2Graph provides a way to analyze whole slide images using a graph convolutional neural network. In our framework, a whole slide image can be viewed as a graph where each tissue patch represents a node in the graph. Our Slide2Graph takes both local patches' features and global structural-and-positional information of the patches as inputs to construct a computational graph of a histopathology image. Our study shows that the structural and positional information is a critical component of Slide2Graph in achieving higher performance than a conventional deep convolutional neural network that takes only patch-based features into account.

 ![alt text](IMG/pipeline.jpg)


To get the high-dimensional representations & coordinates of patches in one whole slide images, [SlidePreprocessing](https://github.com/BMIRDS/SlidePreprocessing) is used to extract tissues and generate small fixed-size patches from whole slide images. After generating the patches, you can train a patch-level ResNet as a feature extractor using the patches. Notably, you can use a ResNet model pretrained on ImageNet as an alternative feature extractor; however, a ResNet trained on specific histopathological patches performs generally better. Finally, the ResNet without the last FC layer will be utilized to generate the high-dimensional representations for the small fixed-size patch images. 

<!--TODO:
    IF A USER WANT TO USE ANOTHER PREPROCESSING LIBRARY OTHER THAN SlidePreprocessing,
    WHAT IS THE REQUIREMENTS? HOW THE POSITIONAL INFORMATION WOULD BE ENCODED?
    I'M GUESSING A PATCH NAME SHOULD BE: 
        SLIDENAME_X_Y.PNG
    WHERE X AND Y ARE THE INDICES OF THE PATCH, LIKE I AND J IN INTEGER AND NOT PIXEL VALUE?
    IT MAY BE AMBIGUOUS SO IT'S BETTER TO CLARIFY THAT IN THE NEXT SECTION.
-->

## Usage
Run `python generate_graphs.py` to generate graphs from the patches' coordinates and the extracted high dimensional features of whole slide images. The default setting is creating edges between every node and its four neartest nodes. The edges will be weighted by the reciprocal euclidean distance.

Run `python main.py` to train the model. Some parameters can be modifed in `config.py`

<!--TODO:
    PLEASE ADD A SECTION TO EXPLAIN HOW A USER CAN USE THIS PIPELINE WITH
    THEIR OWN DATASET. 
    - HOW THE DATASET (I.E., FOLDERS OR PATCHES) SHOULD BE STRUCTURED? DIAGRAM LIKE
    THE FOLLOWING MIGHT BE HELPFUL TOO.
    ROOT
      |- DATASET
            |- SLIDENAME
                 |-patches.png
    - HOW TO FEED LABELS, AS A FOLDER NAME OR ANOTHER CSV FILE? HOW USERS HAVE
        TO MODIFY CONFIG FILE IN ORDER TO RUN ON THEIR DATASET?
-->

## Visualization
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;Important Nodes  &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; Annotations 
<div align=center><img width="280" src="IMG/figure1_r.jpg" alt="Important Nodes"> <img width="280" src="IMG/figure1_label.jpg" alt="Annotations"></div>
<div align=center><img width="280" src="IMG/figure2_r.jpg" alt="Important Nodes"> <img width="280" src="IMG/figure2_label.jpg" alt="Annotations"></div>
<div align=center><img width="280" src="IMG/figure3_r.jpg" alt="Important Nodes"> <img width="280" src="IMG/figure3_label.jpg" alt="Annotations"></div>
<div align=center><img width="280" src="IMG/figure4_r.jpg" alt="Important Nodes"> <img width="280" src="IMG/figure4_label.jpg" alt="Annotations"></div>

<!--TODO:
    PLEASE USE ILLUSTRATOR OR POWERPOINT TO PUT ALL THE LABELS AND FIGURES 
    INTO A BIG SINGLE FIGURE, SO YOU DONT NEED TO FORMAT IT HERE USING MULTIPLE ENSP-S.

-->
