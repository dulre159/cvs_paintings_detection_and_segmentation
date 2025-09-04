# Paintings detection and segmentation
This project aims at detecting and segmenting paintings located in an art gallery (in this case the Galleria Estense located in the city of Modena). More functions such as, paintings retraival, people detection (Yolov5 based) and localization are also included.
![paintings_detection_segmentation](assets/pds.gif)
![people_detection_gaze](assets/pdg.gif)

## Introduction
The code was written for my university exam of Computer Vision Systems.

## PrePrerequisites

Prepare a python or conda venv and execute to fetch all the required packages:
* pip install -r requirements.txt

Note about detectron2: To install for Windows follow the guidelines on this website https://medium.com/@dgmaxime/how-to-easily-install-detectron2-on-windows-10-39186139101c

## Performance
- Detectron2 tested on the **paintings_db** dataset:
<table><tbody>
<tr>
<td align="center">Metric</td>
<td align="center">IoU=0.50:0.05:0.95</td>
<td align="center">IoU=0.5</td>
<td align="center">IoU=0.75</td>
</tr>

<tr>
<td align='center'>mAP</td>
<td align='center'>0.769</td>
<td align='center'>0.946</td>
<td align='center'>0.885</td>
</tr>
<tr>
<td align='center'>mAR</td>
<td align='center'>0.219</td>
<td align='center'>0.785</td>
<td align='center'>0.882</td>
</tr>
</tbody></table>

- Yolov5 finetuned on people + busts dataset:
<table><tbody>
<tr>
<td align="center">P</td>
<td align="center">R</td>
<td align="center">mAP IoU=0.5</td>
<td align="center">mAP IoU=0.50:0.05:0.95</td>
</tr>

<tr>
<td align='left'>0.679</td>
<td align='center'>0.808</td>
<td align='center'>0.743</td>
<td align='center'>0.512</td>
</tr>

</tbody></table>

## Usage


* To see all the comand line options:</br>
     &nbsp; ```python main.py -h```</br>
* Example of comand to excute all the exercises:</br>
     &nbsp; ```python main.py 0 -vp 'path/to/video/video.mp4' -fts 10```</br>
* Example of comand to excute the exercise of paintings replacement in the 3D model:</br>
     &nbsp; ```python main.py 1```</br>
* Example of comand to save a video with the various detection and segmentation results:</br>
     &nbsp; ```python main.py 2 -fts 1 -vp 'path/to/video/video.mp4' -oe -ovdp 'path/to/output/directory/' -ovt 'ouput-video-title'```
      
</br>

Note: Every time an output window showing any result is opened it will wait for a key to continue.</br>


</br>
</br>

```
@misc{he2018maskrcnn,
      title={Mask R-CNN}, 
      author={Kaiming He and Georgia Gkioxari and Piotr Dollár and Ross Girshick},
      year={2018},
      eprint={1703.06870},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1703.06870}, 
}

@misc{khanam2024yolov5deeplookinternal,
      title={What is YOLOv5: A deep look into the internal features of the popular object detector}, 
      author={Rahima Khanam and Muhammad Hussain},
      year={2024},
      eprint={2407.20892},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20892}, 
}

@misc{kümmerer2015deepgazeiboosting,
      title={Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet}, 
      author={Matthias Kümmerer and Lucas Theis and Matthias Bethge},
      year={2015},
      eprint={1411.1045},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1411.1045}, 
}
```
